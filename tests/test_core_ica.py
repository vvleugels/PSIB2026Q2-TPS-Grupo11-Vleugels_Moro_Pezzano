"""ICA engine, persistence, shared signal paths, and batch input isolation."""
from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from scipy.io import savemat

from tps.core import ica
from tps.core.annotations import AnnotationStore, STATES, source_identity
from tps.core.data import Study, open_recording
from tps.core.processing import FilterConfig
from tps.core.screening import ScreeningConfig, scan_recording
from tps.core.spectral import SpectralConfig, run_analysis
from tps.pipeline import batch as run_batch


def fixture(root, session='ses-1', seconds=30):
    root = Path(root)
    directory = root / 'sub-01' / session / 'eeg'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'sub-01_{session}_task-eyesopen_eeg.set'
    sfreq = 200.
    t = np.arange(round(seconds * sfreq)) / sfreq
    rng = np.random.default_rng(42)
    blink = sum(np.exp(-((t-center)/.10)**2) for center in (2, 5.3, 8, 11.2, 15, 19.4, 23, 27))
    sources = np.array([blink * 100, np.sin(2*np.pi*10*t) * 12,
                        rng.laplace(size=len(t))*3])
    samples = np.array([[1, .3, .2], [.9, -.4, .5], [.1, 1, 1]]) @ sources
    values = samples.astype('<f4')
    values.ravel(order='F').tofile(path.with_suffix('.fdt'))
    savemat(path, {'EEG': dict(data=path.with_suffix('.fdt').name, nbchan=3,
        pnts=len(t), trials=1, srate=sfreq, xmin=0., xmax=t[-1],
        chanlocs=np.array([{'labels': c} for c in ('Fp1', 'Fp2', 'Cz')], dtype=object))})
    path.with_suffix('.json').write_text(json.dumps({'EEGReference': 'FCz'}))
    return Study(path, 'sub-01', session, 'eyesopen'), sources


class ICATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = tempfile.TemporaryDirectory()
        study, _ = fixture(cls.training.name)
        recording = open_recording(study)
        try:
            cls.model, cls.metadata = ica.fit_recording(recording, cls.training.name)
        finally:
            recording.close()

    @classmethod
    def tearDownClass(cls):
        cls.training.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study, self.sources = fixture(self.root / 'data')
        self.data = self.root / 'data'
        self.annotations = self.root / 'annotations'
        self.recording = open_recording(self.study)
        self.addCleanup(self.recording.close)
        self.metadata = deepcopy(self.metadata)
        self.metadata['source'] = source_identity(self.recording, self.data)

    def confirm(self, excluded=()):
        cleaned, value = ica.confirm(self.recording, self.data, self.annotations,
                                     self.model, self.metadata, excluded)
        ica.activate(self.recording, cleaned, value)
        return value

    def test_synthetic_artifact_reduction_and_retained_sources(self):
        before = self.recording.raw.get_data()
        raw = self.recording.raw.copy().load_data()
        sources = self.model.get_sources(raw).get_data()
        raw.close()
        correlations = [abs(np.corrcoef(x, self.sources[0])[0, 1]) for x in sources]
        k = int(np.argmax(correlations))
        self.assertGreater(correlations[k], .85)
        self.confirm([k])
        after = self.recording.signal_raw.get_data()
        def projection(signal, template):
            return np.dot(signal - signal.mean(), template-template.mean()) / np.dot(template-template.mean(), template-template.mean())
        self.assertLess(abs(projection(after[0], self.sources[0])), .2 * abs(projection(before[0], self.sources[0])))
        self.assertAlmostEqual(projection(after[2], self.sources[1]) / projection(before[2], self.sources[1]), 1., delta=.15)
        np.testing.assert_array_equal(self.recording.raw.get_data(), before)
        ica.activate(self.recording, None, None)
        np.testing.assert_array_equal(self.recording.signal_raw.get_data(), before)

    def test_revision_roundtrip_zero_exclusions_and_tampering(self):
        value = self.confirm()
        np.testing.assert_array_equal(self.recording.signal_raw.get_data(), self.recording.raw.get_data())
        ica.activate(self.recording, None, None)
        restored = ica.apply_revision(self.recording, self.data, self.annotations, value['revision'])
        self.assertEqual(restored, value)
        folder, _ = ica.read_revision(self.annotations, self.study)
        with (folder/'model-ica.fif').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'dañada'):
            ica.apply_revision(self.recording, self.data, self.annotations)

    def test_failed_publication_keeps_previous_revision(self):
        previous = self.confirm()
        original_write = ica.write_json
        def fail_pointer(path, value):
            if Path(path).name == 'latest.json':
                raise OSError('disk full')
            return original_write(path, value)
        with patch.object(ica, 'write_json', side_effect=fail_pointer), self.assertRaises(OSError):
            ica.confirm(self.recording, self.data, self.annotations, self.model, self.metadata, [0])
        self.assertEqual(ica.latest_revision(self.annotations, self.study), previous['revision'])
        self.assertEqual(self.recording.ica_context['revision'], previous['revision'])

    def test_source_and_version_mismatches(self):
        self.confirm()
        self.recording.metadata['EEGReference'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'registro actual'):
            ica.apply_revision(self.recording, self.data, self.annotations)
        self.recording.metadata['EEGReference'] = 'FCz'
        # Other library versions (e.g. the professors' computer): applied, with a note.
        with patch.object(ica, 'versions', return_value={'mne': '0.0'}):
            metadata = ica.apply_revision(self.recording, self.data, self.annotations)
            self.assertIn('MNE 0.0', ica.version_note(metadata))
        self.assertIsNotNone(self.recording.ica_context)

    def test_window_screening_welch_and_saved_review_use_same_revision(self):
        value = self.confirm([0])
        config = FilterConfig(True, 1, 45, 50)
        continuous = self.recording.filter.channel('Fp1', config)
        segment = self.recording.read_channels(['Fp1'], 5, config)
        np.testing.assert_allclose(segment.values_uv[0], continuous[1000:3000]*1e6)
        scan = scan_recording(self.recording, ['Fp1'], ScreeningConfig(high_uv=1, flat_uv=0), config)
        self.assertAlmostEqual(scan['findings'][1][0]['peak_to_peak_uv'], np.ptp(continuous[1000:2000])*1e6)
        store = AnnotationStore(self.annotations, self.recording, self.data)
        store.mark(1, STATES[1], '', '', ['Fp1'], config); store.save()
        self.assertEqual(store.records[1]['ica']['revision'], value['revision'])
        selected = ica.snapshot([self.study], self.data, self.annotations, 'confirmed')
        result = run_analysis([self.study], 'Fp1', config, SpectralConfig(), self.data,
                              self.annotations, ica_selections=selected)
        self.assertEqual(result['sessions'][0]['ica']['revision'], value['revision'])
        self.assertFalse(any('ICA' in w for w in result['sessions'][0]['warnings']))
        # A subsequent confirmation must not alter the already snapshotted input.
        self.confirm([1])
        again = run_analysis([self.study], 'Fp1', config, SpectralConfig(), self.data,
                             self.annotations, ica_selections=selected)
        np.testing.assert_allclose(result['sessions'][0]['psd_uv2_hz'], again['sessions'][0]['psd_uv2_hz'])
        uncorrected = run_analysis([self.study], 'Fp1', config, SpectralConfig(), self.data, self.annotations)
        self.assertTrue(any('otra ICA' in s for s in uncorrected['sessions'][0]['warnings']))

    def test_batch_missing_correction_and_cache_selection(self):
        cache = self.root/'cache'; cache.mkdir()
        selection = ica.snapshot([self.study], self.data, self.annotations, 'confirmed')[str(self.study.path)]
        result = run_batch.run_one(self.study, cache, 'test', selection, self.data, self.annotations)
        self.assertTrue(result['summary']['excluded'])
        self.assertIn('ICA confirmada', result['exclusions'][0]['reason'])
        self.assertTrue(run_batch.finished(run_batch.cache_path(cache, self.study), 'test', selection))
        self.confirm()
        selected = ica.snapshot([self.study], self.data, self.annotations, 'confirmed')[str(self.study.path)]
        self.assertFalse(run_batch.finished(run_batch.cache_path(cache, self.study), 'test', selected))
        output = self.root/'out'; output.mkdir()
        totals = run_batch.aggregate(cache, output, 'test', [], {self.study.path.stem: selected})
        self.assertEqual(totals['recordings'], 0)

    def test_available_mode_and_manual_epochs(self):
        store = AnnotationStore(self.annotations, self.recording, self.data)
        store.mark(2, STATES[2], 'Ocular', '', ['Fp1'], FilterConfig(True, 1, 45, 50)); store.save()
        # Without a confirmed ICA, 'available' keeps the recording with filters only.
        value = ica.snapshot([self.study], self.data, self.annotations, 'available')[str(self.study.path)]
        self.assertEqual(value['mode'], 'original')
        self.assertEqual(value['artifact_epochs'], {'2': 'Ocular'})
        confirmed = self.confirm()
        value = ica.snapshot([self.study], self.data, self.annotations, 'available')[str(self.study.path)]
        self.assertEqual((value['mode'], value['revision']), ('confirmed', confirmed['revision']))
        # The batch always rejects the épocas marked «Con artefactos».
        ica.activate(self.recording, None, None)
        result = run_batch.process_recording(self.recording, 'global', FilterConfig(True, 1, 45, 50),
                                             None, value['artifact_epochs'])
        epochs = {e['epoch']: e['reason'] for e in result['exclusions'] if e['level'] == 'epoch'}
        self.assertIn('marcada a mano', epochs['3'])
        result = run_batch.process_recording(self.recording, 'per_channel', FilterConfig(True, 1, 45, 50),
                                             None, value['artifact_epochs'])
        self.assertTrue(any('marcada a mano' in e['reason'] for e in result['exclusions']))

    def test_recordings_file(self):
        path = self.root / 'list.txt'
        path.write_text('\n'.join(['# comment', 'sub-01_ses-1_task-eyesopen_eeg',
                                   '# sub-01_ses-2_task-eyesopen_eeg',
                                   'sub-01/ses-1/eeg/sub-01_ses-1_task-eyesclosed_eeg.set  # path form']),
                        encoding='utf-8')
        self.assertEqual(run_batch.read_recordings_file(path),
                         {'sub-01_ses-1_task-eyesopen_eeg', 'sub-01_ses-1_task-eyesclosed_eeg'})
        path.write_text('# only comments', encoding='utf-8')
        with self.assertRaises(ValueError):
            run_batch.read_recordings_file(path)

    def test_invalid_training_and_components(self):
        with self.assertRaises(ValueError):
            ica.fit_recording(self.recording, self.data, excluded_channels=['Fp1', 'Fp2'])
        with self.assertRaises(ValueError):
            ica.fit_recording(self.recording, self.data, excluded_epochs=list(range(6)))
        with self.assertRaises(ValueError):
            ica.corrected_copy(self.recording, self.model, [999])

    def test_multiprocess_batch_with_missing_pair(self):
        data = self.root / 'batch-data'
        ns, _ = fixture(data, seconds=80)
        fixture(data, session='ses-2', seconds=80)
        (data / 'participants.tsv').write_text('participant_id\tSessionOrder\nsub-01\tNS->SD\n')
        recording = open_recording(ns)
        try:
            model, metadata = ica.fit_recording(recording, data)
            corrected, _ = ica.confirm(recording, data, self.annotations, model, metadata, [])
            corrected.close()
        finally:
            recording.close()
        out = self.root/'batch-out'
        command = [sys.executable, '-m', 'tps.pipeline.batch', '--data-dir', str(data),
                   '--out-dir', str(out), '--annotations-dir', str(self.annotations),
                   '--ica-mode', 'confirmed', '--jobs', '2', '--no-bandpass', '--notch', '50']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        coverage = json.loads((out/'ica_coverage.json').read_text())
        used = json.loads((out/'config_used.json').read_text())['filter']
        self.assertFalse(used['enabled'])
        self.assertEqual(used['notch'], 50)
        self.assertEqual(coverage['processing_modes'], ['confirmed'])
        self.assertEqual((coverage['confirmed'], coverage['excluded']), (1, 1))
        import csv
        with (out/'features_long.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertEqual({r['ses'] for r in rows}, {'ses-1'})
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            from tps.analysis.stats import paired_frame
            self.assertTrue(paired_frame(pd.DataFrame(rows), 'abs_power_uV2', ['channel', 'band']).empty)
        # Same output directory cannot aggregate corrected caches into an original run.
        command[command.index('confirmed')] = 'original'
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        coverage = json.loads((out/'ica_coverage.json').read_text())
        self.assertEqual(coverage['processing_modes'], ['original'])
        self.assertEqual((coverage['confirmed'], coverage['excluded']), (0, 0))

    def test_only_explicit_exclusions_are_applied(self):
        model = self.model.copy()
        expected = ica.corrected_copy(self.recording, model, [0])
        model.exclude = [1]
        actual = ica.corrected_copy(self.recording, model, [0])
        try:
            np.testing.assert_array_equal(actual.get_data(), expected.get_data())
            self.assertEqual(model.exclude, [1])
        finally:
            expected.close(); actual.close()
