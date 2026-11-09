"""Batch pipeline tests on synthetic signals (no dataset needed)."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tps import config
from tps.pipeline import batch as run_batch
from tps.core.data import Recording, Study

SFREQ = 250.0
SECONDS = 102.0   # 20 complete 5 s epochs + a 2 s partial epoch
CHANNELS = ['Fz', 'Cz', 'Pz']


def synthetic(signals_uv, name='sub-99_ses-1_task-eyesopen_eeg.set'):
    """Wrap per-channel µV arrays as a Recording, like open_recording would return."""
    info = mne.create_info(list(signals_uv), SFREQ, 'eeg')
    raw = mne.io.RawArray(np.array(list(signals_uv.values())) * 1e-6, info, verbose='ERROR')
    parts = name.split('_')
    study = Study(Path('sub-99') / parts[1] / 'eeg' / name, parts[0], parts[1], parts[2][5:])
    return Recording(study, raw, {})


def times():
    return np.arange(int(SECONDS * SFREQ)) / SFREQ


def sine(freq, amplitude_uv):
    return amplitude_uv * np.sin(2 * np.pi * freq * times())


def noise(scale_uv=0.5, seed=0):
    return np.random.default_rng(seed).normal(0, scale_uv, len(times()))


def band(rows, channel, name):
    return next(r for r in rows if r['channel'] == channel and r['band'] == name)


class SpectralFeatureTests(unittest.TestCase):
    def test_10hz_sine_puts_power_in_alpha(self):
        result = run_batch.process_recording(
            synthetic({c: sine(10, 20) + noise(seed=i) for i, c in enumerate(CHANNELS)}))
        for channel in CHANNELS:
            alpha = band(result['features'], channel, 'alpha')
            self.assertGreater(alpha['rel_power_pct'], 90)
            # A sine of amplitude A has power A²/2 = 200 µV².
            self.assertAlmostEqual(alpha['abs_power_uV2'], 200, delta=10)
            for other in ('delta', 'theta', 'beta'):
                self.assertLess(band(result['features'], channel, other)['rel_power_pct'], 5)
        self.assertEqual(result['summary']['n_epochs_used'], 20)
        self.assertEqual(result['summary']['n_epochs_rejected'], 0)
        self.assertEqual(len(result['features']), len(CHANNELS) * 4)
        # The exported mean PSD peaks at 10 Hz and stops at the filter's upper cutoff.
        psd = result['psd']
        self.assertEqual(psd['frequencies'][-1], config.FILTER.high)
        peak = int(np.argmax(psd['channels']['Cz']))
        self.assertEqual(psd['frequencies'][peak], 10.0)

    def test_theta_beta_ratio(self):
        # Powers 20²/2 (6 Hz, theta) and 10²/2 (20 Hz, beta) -> ratio 4.
        result = run_batch.process_recording(synthetic({'Cz': sine(6, 20) + sine(20, 10) + noise()}))
        self.assertAlmostEqual(result['theta_beta'][0]['ratio'], 4.0, delta=0.2)

    def test_output_columns_and_labels(self):
        result = run_batch.process_recording(synthetic({'Cz': sine(10, 20) + noise()}))
        row = result['features'][0]
        self.assertEqual(list(row), run_batch.FEATURE_COLUMNS)
        self.assertEqual((row['sub'], row['ses'], row['condition'], row['eyes']),
                         ('sub-99', 'ses-1', 'NS', 'open'))


class ScreeningTests(unittest.TestCase):
    def test_spike_rejects_only_its_epoch_and_partial_is_dropped(self):
        spiky = sine(10, 20) + noise()
        spiky[int(12 * SFREQ)] += 400    # inside epoch 3 (10-15 s)
        result = run_batch.process_recording(synthetic({'Fz': spiky, 'Cz': sine(10, 20) + noise(seed=1)}))
        self.assertEqual(result['summary']['n_epochs_rejected'], 1)
        self.assertEqual(result['summary']['n_epochs_used'], 19)
        epochs = [r for r in result['exclusions'] if r['level'] == 'epoch']
        self.assertIn('3', [r['epoch'] for r in epochs])
        self.assertTrue(any('Fz high' in r['reason'] for r in epochs))
        self.assertTrue(any('incomplete final epoch' in r['reason'] for r in epochs))
        # The rejection is global: the clean channel also uses 19 epochs.
        self.assertEqual(band(result['features'], 'Cz', 'alpha')['n_epochs_used'], 19)

    def test_persistently_noisy_channel_is_bad_not_epoch_killer(self):
        result = run_batch.process_recording(synthetic({
            'Fz': sine(10, 20) + noise(), 'Cz': sine(10, 20) + noise(seed=1),
            'TP9': sine(10, 120) + noise(seed=2)}))   # 240 µV p-p in every epoch
        self.assertEqual(result['summary']['n_bad_channels'], 1)
        self.assertEqual(result['summary']['n_epochs_used'], 20)
        self.assertNotIn('TP9', {r['channel'] for r in result['features']})
        self.assertIn('TP9', {r['channel'] for r in result['exclusions'] if r['level'] == 'channel'})

    def test_flat_channel_is_flagged(self):
        flat = noise(0.05)   # ~0.3 µV p-p, below the 1 µV flat limit
        result = run_batch.process_recording(synthetic({'Fz': flat, 'Cz': sine(10, 20) + noise()}))
        reasons = [r['reason'] for r in result['exclusions'] if r['level'] == 'channel']
        self.assertTrue(reasons and 'Fz' in [r['channel'] for r in result['exclusions'] if r['level'] == 'channel'])

    def test_too_few_clean_epochs_excludes_recording(self):
        # Each channel spikes in 3 different epochs (15 % < bad-channel limit), so no channel
        # is "bad" but 12 of the 20 epochs are rejected and only 8 (< 12) stay clean.
        signals = {}
        for i, channel in enumerate(['Fz', 'Cz', 'Pz', 'Oz']):
            signal = sine(10, 20) + noise(seed=i)
            for epoch in range(3 * i, 3 * i + 3):
                signal[int((epoch * 5 + 2) * SFREQ)] += 400
            signals[channel] = signal
        result = run_batch.process_recording(synthetic(signals))
        self.assertLess(result['summary']['n_epochs_used'], config.MIN_CLEAN_EPOCHS)
        self.assertTrue(result['summary']['excluded'])
        self.assertEqual(result['features'], [])
        self.assertTrue(any(r['level'] == 'recording' and 'clean epochs' in r['reason']
                            for r in result['exclusions']))


class PerChannelRejectionTests(unittest.TestCase):
    """--rejection-mode per_channel (ported from the former group pipeline)."""

    def test_spike_rejects_epochs_only_for_its_channel(self):
        spiky = sine(10, 20) + noise()
        for second in (12, 27, 42):   # epochs 3, 6 and 9
            spiky[int(second * SFREQ)] += 400
        recording = synthetic({'Fz': spiky, 'Cz': sine(10, 20) + noise(seed=1)})
        per_channel = run_batch.process_recording(recording, 'per_channel')
        self.assertEqual(band(per_channel['features'], 'Fz', 'alpha')['n_epochs_used'], 17)
        self.assertEqual(band(per_channel['features'], 'Cz', 'alpha')['n_epochs_used'], 20)
        global_mode = run_batch.process_recording(synthetic({'Fz': spiky, 'Cz': sine(10, 20) + noise(seed=1)}),
                                                  'global')
        self.assertEqual(band(global_mode['features'], 'Cz', 'alpha')['n_epochs_used'], 17)

    def test_channel_below_minimum_is_excluded(self):
        noisy = sine(10, 120) + noise(seed=2)   # 240 uV p-p in every epoch
        result = run_batch.process_recording(synthetic({'Fz': sine(10, 20) + noise(), 'TP9': noisy}),
                                             'per_channel')
        self.assertNotIn('TP9', {r['channel'] for r in result['features']})
        reasons = [r['reason'] for r in result['exclusions'] if r['channel'] == 'TP9']
        self.assertTrue(reasons and 'per-channel rejection' in reasons[0])
        self.assertFalse(result['summary']['excluded'])

    def test_recording_excluded_when_no_channel_survives(self):
        result = run_batch.process_recording(synthetic({'Fz': sine(10, 120) + noise()}), 'per_channel')
        self.assertTrue(result['summary']['excluded'])
        self.assertEqual(result['features'], [])


class InventoryAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        eeg = self.data / 'sub-01' / 'ses-1' / 'eeg'
        eeg.mkdir(parents=True)
        # Windows checks out git-annex links without content as small text files.
        (eeg / 'sub-01_ses-1_task-eyesopen_eeg.set').write_text(
            '../../../.git/annex/objects/61/85/SHA256E-s629416--ab.set/SHA256E-s629416--ab.set')
        (self.data / 'participants.tsv').write_text('participant_id\tSessionOrder\nsub-01\tSD->NS\n')

    def tearDown(self):
        self.temp.cleanup()

    def read(self, name):
        with (self.root / 'out' / name).open(encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def test_annex_placeholder_is_logged_and_run_resumes(self):
        args = ['--data-dir', str(self.data), '--out-dir', str(self.root / 'out')]
        self.assertEqual(run_batch.main(args), 0)
        exclusions = self.read('exclusions.csv')
        reasons = {r['level']: r['reason'] for r in exclusions}
        self.assertIn('git-annex', reasons['recording'])
        self.assertIn('pair', reasons)   # ses-2 counterpart is missing
        self.assertEqual(self.read('session_order.csv')[0]['first_condition'], 'SD')
        self.assertTrue((self.root / 'out' / 'config_used.json').is_file())
        cache = self.root / 'out' / 'per_recording' / 'sub-01_ses-1_task-eyesopen_eeg.json'
        stamp = cache.stat().st_mtime_ns
        with self.assertLogs('batch', 'INFO') as logs:
            run_batch.main(args)
        self.assertTrue(any('1 already finished, 0 to process' in line for line in logs.output))
        self.assertEqual(cache.stat().st_mtime_ns, stamp)

    def test_subjects_file(self):
        path = self.root / 'subset.txt'
        path.write_text('# seed: 2026\nsub-01\n\n  02  # comment\n', encoding='utf-8')
        self.assertEqual(run_batch.read_subjects_file(path), ['sub-01', '02'])
        studies, _ = run_batch.select_studies(self.data, None, run_batch.read_subjects_file(path))
        self.assertEqual({s.participant for s in studies}, {'sub-01'})
        empty = self.root / 'empty.txt'
        empty.write_text('# nothing\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            run_batch.read_subjects_file(empty)

    def test_eyes_and_subject_filters(self):
        studies, _ = run_batch.select_studies(self.data, 'closed', None)
        self.assertEqual(studies, [])
        studies, _ = run_batch.select_studies(self.data, 'open', ['1'])
        self.assertEqual([s.participant for s in studies], ['sub-01'])


if __name__ == '__main__':
    unittest.main()
