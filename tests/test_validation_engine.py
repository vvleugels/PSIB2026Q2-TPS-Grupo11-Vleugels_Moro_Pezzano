from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import mne
import numpy as np

from eeg_viewer.validation import (CleanResult, ICAResult, RuleProfile, assess, epochs4,
                                  evaluation_report, finalize, fit_ica, interpolate,
                                  prepare, proposals, robust_scores)
from eeg_viewer.validation_store import (Revision, atomic_write, digest_json, encode,
                                        freeze_profile, read_frozen, sha)
from eeg_viewer.validation_session import ValidationSession
from eeg_viewer.validation_ui import indices


def synthetic(seconds=40, positions=True):
    rng = np.random.default_rng(7)
    names = ['Fz', 'Cz', 'Pz', 'Oz', 'Fp1', 'Fp2']
    times = np.arange(round(seconds * 100)) / 100
    values = rng.normal(0, 1e-6, (len(names), len(times)))
    values += np.array([1, 2, 1.5, 1, .5, .8])[:, None] * np.sin(2 * np.pi * 10 * times) * 8e-6
    raw = mne.io.RawArray(values, mne.create_info(names, 100, 'eeg'), verbose=False)
    if positions:
        # Explicit synthetic geometry in metres; no assumptions about real dataset coordinates.
        pos = {'Fz': [0, .06, .06], 'Cz': [0, 0, .09], 'Pz': [0, -.06, .06],
               'Oz': [0, -.09, .01], 'Fp1': [-.03, .08, .02], 'Fp2': [.03, .08, .02]}
        raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head'))
    return raw


class ValidationEngineTests(unittest.TestCase):
    def test_four_second_boundaries_partial_and_no_drift(self):
        e = epochs4(150000, 500)
        self.assertEqual(len(e), 75)
        self.assertFalse(any(x.partial for x in e))
        for n, sf in [(1321, 99.5), (4001, 100)]:
            e = epochs4(n, sf)
            self.assertEqual(e[0].first, 0)
            self.assertEqual(e[-1].last, n)
            self.assertTrue(e[-1].partial)
            self.assertTrue(all(a.last == b.first for a, b in zip(e, e[1:])))

    def test_mad_zero_is_unknown_not_clean(self):
        scores = robust_scores(np.ones((5, 2)), np.ones(5, dtype=bool))
        self.assertTrue(np.isnan(scores).all())
        raw = synthetic()
        raw._data[:] = 0
        q = assess(raw)
        self.assertTrue(q.flags.all())
        self.assertFalse(q.evaluable.any())

    def test_channel_findings_do_not_reject_other_channels(self):
        raw = synthetic()
        raw._data[4, :400] *= 50
        q = assess(raw)
        self.assertTrue(q.flags[0, 4])
        self.assertFalse(q.flags[0, 1])
        self.assertIn('alta', q.reasons(0, 'Fp1'))

    def test_filter_branches_preserve_original_and_slow_activity(self):
        raw = synthetic(seconds=80)
        t = np.arange(raw.n_times) / 100
        raw._data += 30e-6 * np.sin(2 * np.pi * .5 * t)
        original = raw.get_data().copy()
        p = prepare(raw)
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(p.analysis.info['highpass'], .2)
        self.assertEqual(p.training.info['highpass'], 1.)
        self.assertEqual(p.analysis.ch_names, p.training.ch_names)
        self.assertEqual(p.analysis.info['custom_ref_applied'], p.training.info['custom_ref_applied'])
        sine = np.sin(2 * np.pi * .5 * t[2000:-2000])
        a = abs(np.dot(p.analysis.get_data()[0, 2000:-2000], sine))
        b = abs(np.dot(p.training.get_data()[0, 2000:-2000], sine))
        self.assertGreater(a, 2 * b)

    def test_nonfinite_rejected_before_filter(self):
        raw = synthetic()
        raw._data[0, 123] = np.nan
        with patch('eeg_viewer.validation.ContinuousFilter') as filtering:
            with self.assertRaisesRegex(ValueError, 'no finitas'):
                prepare(raw)
            filtering.assert_not_called()

    def test_missing_positions_exclude_persistent_without_fabricating_coordinates(self):
        raw = synthetic(positions=False)
        p = prepare(raw, ['Fp1'])
        self.assertEqual(p.unavailable, ['Fp1'])
        self.assertEqual(p.persistent_repaired, [])
        self.assertTrue(p.warnings)
        with self.assertRaisesRegex(ValueError, 'posiciones'):
            interpolate(raw, ['Fp1'])

    def test_interpolation_reduces_rank_and_keeps_both_branches_compatible(self):
        p = prepare(synthetic(), ['Fp1'])
        self.assertEqual(p.persistent_repaired, ['Fp1'])
        data = p.training.get_data()
        singular = np.linalg.svd(data - data.mean(axis=1, keepdims=True), compute_uv=False)
        self.assertEqual(np.sum(singular > singular[0] * 1e-6), 5)
        self.assertEqual(p.training.ch_names, p.analysis.ch_names)

    def test_local_repair_limit_includes_persistent_channels(self):
        raw = synthetic()
        raw._data[4, :400] *= 50
        q = assess(raw)
        p = proposals(q, ['Fz', 'Pz'], RuleProfile(max_repair=2))
        affected = next(p for p in p if p.epoch == 0)
        self.assertEqual(affected.total_affected, 3)
        self.assertFalse(affected.eligible)

    def test_ica_nonconvergence_blocks_result(self):
        prepared = prepare(synthetic())
        with patch('eeg_viewer.validation.mne.preprocessing.ICA') as factory:
            factory.return_value.n_iter_ = 1000
            with self.assertRaisesRegex(ValueError, 'no convergió'):
                fit_ica(prepared)

    def test_ica_excludes_partial_and_marked_epochs_and_passes_effective_rank(self):
        p = prepare(synthetic(40.5), ['Fp1'])
        with patch('eeg_viewer.validation.mne.preprocessing.ICA') as factory:
            ica = factory.return_value
            ica.n_iter_ = 10
            def trace(*args, **kwargs):
                with mne.utils.use_log_level('INFO'):
                    mne.utils.logger.info('step %d - lrate %5f, wchange %8.8f, angledelta %4.1f deg',
                                          10, .01, 1e-13, 20.)
            ica.fit.side_effect = trace
            ica.get_sources.return_value.get_data.return_value = np.random.default_rng(9).normal(size=(5, 3200))
            labels = {f'{e.index}:{c}': {'state': 'physiological'} for e in p.quality.epochs for c in p.analysis.ch_names}
            labels['1:Cz']['state'] = 'artifact'
            with np.errstate(invalid='ignore'):
                result = fit_ica(p, [0], labels)
            self.assertNotIn(0, result.fit_epochs)
            self.assertNotIn(1, result.fit_epochs)
            self.assertNotIn(10, result.fit_epochs)
            self.assertEqual(result.rank, 5)
            self.assertEqual(factory.call_args.kwargs['method'], 'infomax')
            self.assertTrue(factory.call_args.kwargs['fit_params']['extended'])

    def test_reference_is_fixed_and_bad_channel_does_not_reject_epoch(self):
        raw = synthetic()
        p = prepare(raw)
        adjusted = p.analysis.copy()
        adjusted._data[4, :400] *= 50
        before = adjusted.get_data().copy()
        with patch('eeg_viewer.validation.apply_components', return_value=adjusted):
            result = finalize(p, None, [], profile=RuleProfile(robust_z=100))
        self.assertNotIn('Fp1', result.reference_channels)
        self.assertIn('Cz', result.reference_channels)
        self.assertEqual(result.excluded_epochs, [])
        expected = before[[adjusted.ch_names.index(c) for c in result.reference_channels]].mean(axis=0)
        np.testing.assert_allclose(result.raw.get_data(), before - expected, atol=1e-18)

    def test_real_infomax_stopping_evidence_and_clean_copy(self):
        rng = np.random.default_rng(2)
        data = np.array([[1, .1, .2], [.2, 1, .1], [.1, .2, 1]]) @ rng.laplace(size=(3, 4000)) * 5e-6
        raw = mne.io.RawArray(data.copy(), mne.create_info(['Cz', 'Fz', 'Pz'], 100, 'eeg'), verbose=False)
        prepared = prepare(raw)
        result = fit_ica(prepared)
        self.assertLessEqual(result.convergence['actual_steps'], 1000)
        self.assertIn(result.convergence['stop_reason'], ('weight_change', 'small_angle'))
        cleaned = finalize(prepared, result, [])
        np.testing.assert_array_equal(raw.get_data(), data)
        np.testing.assert_allclose(cleaned.raw.get_data(picks=cleaned.reference_channels).mean(axis=0), 0, atol=1e-18)

    def test_unevaluable_never_counts_as_safe_acceptance(self):
        raw = synthetic()
        q = assess(raw)
        q.flags[:] = False
        q.evaluable[:] = False
        report = evaluation_report(q, {'0:Cz': {'state': 'artifact'}})
        self.assertEqual(report['unevaluable'], 1)
        self.assertEqual(report['missed_artifact'], 0)
        self.assertFalse(report['automatic_acceptance_enabled'])

    def test_human_indices_are_validated(self):
        self.assertEqual(indices('1,3,3', 4), [0, 2])
        for value in ('0', '5', '-2', 'a'):
            with self.assertRaises(ValueError):
                indices(value, 4)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_file = self.root / 'sub-18_ses-1_task-eyesopen_eeg.set'
        self.source_file.write_bytes(b'original EEG')
        stat = self.source_file.stat()
        self.source = dict(path=str(self.source_file), participant='sub-18', session='ses-1',
                           task='eyesopen', split='calibration', n_times=800, sfreq=100.,
                           channels=['Cz', 'TP9'], reference='FCz',
                           files=[dict(path=str(self.source_file), size=stat.st_size,
                                       mtime_ns=stat.st_mtime_ns, sha256=sha(self.source_file))])
        self.sample = self.root / 'sample.json'
        atomic_write(self.sample, encode(dict(records=[self.source])))
        self.annotations = self.root / 'annotations'
        self.annotations.mkdir()

    def revision(self):
        return Revision.create(self.annotations, self.source, self.sample)

    def complete(self, r):
        for index in range(2):
            r.label(index, ['Cz', 'TP9'], 'physiological')
        r.complete_blind()

    def test_legacy_backup_is_exact_and_no_labels_migrated(self):
        legacy = self.annotations / (self.source_file.stem + '.json')
        payload = b'{"old labels": "do not migrate"}\r\n'
        legacy.write_bytes(payload)
        r = self.revision()
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual((r.directory / r.data['legacy_backup']['path']).read_bytes(), payload)
        self.assertEqual(r.data['labels'], {})
        self.assertEqual(r.data['epoch_seconds'], 4)

    def test_blind_labels_cannot_be_rewritten_after_reveal(self):
        r = self.revision()
        with self.assertRaisesRegex(ValueError, 'Faltan etiquetas'):
            r.complete_blind()
        self.complete(r)
        with self.assertRaisesRegex(ValueError, 'ciega está cerrada'):
            r.label(0, ['Cz'], 'artifact')
        self.assertEqual(r.data['labels']['0:Cz']['origin'], 'manual')

    def test_external_revision_and_source_changes_are_blocked(self):
        r = self.revision()
        second = Revision(r.directory)
        r.label(0, ['Cz'], 'physiological')
        with self.assertRaisesRegex(ValueError, 'externamente'):
            second.save('conflict')
        self.source_file.write_bytes(b'modified source')
        with self.assertRaisesRegex(ValueError, 'origen cambió'):
            r.save('conflict')

    def test_artifact_tampering_blocks_reopening(self):
        r = self.revision()
        r.artifact('test', lambda p: p.write_bytes(b'result'), '.bin')
        r.save('result')
        r.read_artifact('test').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'modificado'):
            Revision(r.directory)

    def test_processing_invalidates_final_labels_and_preserves_history(self):
        r = self.revision()
        self.complete(r)
        r.data['stage'] = 'clean'
        r.label(0, ['Cz'], 'physiological', final=True)
        r.update_processing(excluded_components=[0])
        self.assertFalse(r.data['final_labels'])
        self.assertTrue(r.data['labels'])
        previous = [h['previous'] for h in r.data['history'] if 'previous' in h]
        self.assertEqual(previous[-1]['final_labels']['0:Cz']['state'], 'physiological')

    def test_failed_transaction_preserves_last_manifest(self):
        r = self.revision()
        self.complete(r)
        before = r.path.read_bytes()
        with self.assertRaises(OSError):
            with r.transaction():
                r.update_processing(persist=False, excluded_components=[1])
                r.artifact('failed', lambda p: (_ for _ in ()).throw(OSError('disk full')), '.bin')
        self.assertEqual(r.path.read_bytes(), before)
        self.assertEqual(r.data['excluded_components'], [])

    def test_evaluation_requires_frozen_profile_and_disallows_retuning(self):
        self.source['split'] = 'evaluation'
        atomic_write(self.sample, encode(dict(records=[self.source])))
        with self.assertRaisesRegex(ValueError, 'Congelar'):
            self.revision()
        from eeg_viewer.validation import ProcessingConfig
        frozen = self.root / 'frozen.json'
        profile = asdict(RuleProfile())
        atomic_write(frozen, encode(dict(version=1, status='frozen_for_evaluation',
                                         sample_sha256=sha(self.sample), profile=profile,
                                         profile_sha256=digest_json(profile), processing=asdict(ProcessingConfig()))))
        r = Revision.create(self.annotations, self.source, self.sample, frozen=frozen)
        self.complete(r)
        with self.assertRaisesRegex(ValueError, 'recalibrar'):
            r.update_processing(profile=asdict(RuleProfile(high_uv=200)))

    def test_freeze_requires_completed_calibration(self):
        r = self.revision()
        with self.assertRaisesRegex(ValueError, 'Completar revisión'):
            freeze_profile(self.sample, [r.directory], self.root / 'frozen.json')

    def test_cancellation_checkpoint(self):
        session = object.__new__(ValidationSession)
        session.cancel_event = threading.Event()
        session.cancel_event.set()
        with self.assertRaisesRegex(ValueError, 'cancelado'):
            session.checkpoint()

    def test_export_never_overwrites_original(self):
        r = self.revision()
        before = self.source_file.read_bytes()
        with self.assertRaisesRegex(ValueError, 'ya existe'):
            r.export(self.source_file)
        self.assertEqual(self.source_file.read_bytes(), before)

    def test_real_processing_roundtrip_report_and_freeze(self):
        rng = np.random.default_rng(2)
        data = np.array([[1, .1, .2], [.2, 1, .1], [.1, .2, 1]]) @ rng.laplace(size=(3, 4000)) * 5e-6
        raw = mne.io.RawArray(data.copy(), mne.create_info(['Cz', 'Fz', 'Pz'], 100, 'eeg'), verbose=False)
        self.source.update(channels=raw.ch_names, n_times=4000)
        atomic_write(self.sample, encode(dict(records=[self.source])))
        r = self.revision()
        for e in range(10):
            r.label(e, raw.ch_names, 'physiological')
        r.complete_blind()
        def recording(_):
            return SimpleNamespace(raw=raw.copy(), sfreq=100., close=lambda: None)
        with patch('eeg_viewer.validation_session.open_recording', side_effect=recording):
            s = ValidationSession(r)
            s.fit([], [])
            s.clean([], [])
            expected = s.cleaned.get_data().copy()
            s.close()
            reopened = ValidationSession(Revision(r.directory))
            np.testing.assert_array_equal(reopened.cleaned.get_data(), expected)
            self.assertEqual(reopened.fitted.channels, raw.ch_names)
            for e in range(10):
                reopened.revision.label(e, raw.ch_names, 'physiological', final=True)
            report = reopened.report()
            self.assertEqual(len(report['candidate_comparison']), 54)
            self.assertEqual(report['baseline']['unreviewed'], 0)
            self.assertEqual(len(report['spectral_preservation']), 3)
            self.assertFalse(report['approved_for_integration'])
            encode(report)  # Reject NaN/Infinity and non-JSON scalars.
            reopened.close()
            frozen_path = self.root / 'frozen.json'
            frozen = freeze_profile(self.sample, [r.directory], frozen_path)
            self.assertFalse(frozen['approved_for_integration'])
            self.assertEqual(frozen['calibration'][0]['snapshot']['stage'], 'clean')
            self.assertEqual(read_frozen(frozen_path, self.sample)['status'], 'frozen_for_evaluation')


if __name__ == '__main__':
    unittest.main()
