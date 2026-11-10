"""Bad-channel detection on synthetic recordings: each faulty channel is caught by its criterion."""
from pathlib import Path
import sys
import unittest

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tps.core.badchannels import BadChannelConfig, detect, find_bad_channels, robust_z
from tps.core.data import Recording, Study
from tps.core.processing import FilterConfig

SFREQ = 250.0
SECONDS = 60.0
N_GOOD = 12


def clean_channels(seed=0):
    """Twelve channels sharing an 'EEG' source (10 Hz + 5 Hz + slow) with varied gains and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SECONDS * SFREQ)) / SFREQ
    source = 15 * np.sin(2 * np.pi * 10 * t) + 8 * np.sin(2 * np.pi * 5 * t + 1) + rng.normal(0, 4, t.size)
    gains = np.linspace(0.7, 1.3, N_GOOD)
    channels = {f'E{i:02d}': g * source + rng.normal(0, 2, t.size) for i, g in enumerate(gains)}
    return t, source, channels, rng


def recording(signals_uv):
    info = mne.create_info(list(signals_uv), SFREQ, 'eeg')
    raw = mne.io.RawArray(np.array(list(signals_uv.values())) * 1e-6, info, verbose='ERROR')
    study = Study(Path('sub-99/ses-1/eeg/sub-99_ses-1_task-eyesopen_eeg.set'), 'sub-99', 'ses-1', 'eyesopen')
    return Recording(study, raw, {})


def run(signals_uv, filtering=FilterConfig()):
    return find_bad_channels(recording(signals_uv), filtering, BadChannelConfig())


class BadChannelTests(unittest.TestCase):
    def test_clean_channels_are_not_flagged(self):
        _, _, channels, _ = clean_channels()
        self.assertEqual(run(channels), {})
        self.assertEqual(run(channels, FilterConfig(True, 0.5, 45)), {})

    def test_each_fault_is_caught_by_its_criterion(self):
        t, source, channels, rng = clean_channels()
        channels['FLAT'] = np.full(t.size, 3.0) + rng.normal(0, 0.01, t.size)
        channels['LOUD'] = 10 * source + rng.normal(0, 2, t.size)
        channels['ALONE'] = rng.normal(0, 18, t.size)                       # uncorrelated noise
        channels['LINE'] = source + 6 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 2, t.size)
        bad = run(channels)
        self.assertEqual(set(bad), {'FLAT', 'LOUD', 'ALONE', 'LINE'}, bad)
        self.assertEqual(bad['FLAT']['criterion'], 'flat')
        self.assertEqual(bad['LOUD']['criterion'], 'deviation')
        self.assertIn('correlation', [bad['ALONE']['criterion']] + bad['ALONE']['reason'].split())
        self.assertIn('correlación', bad['ALONE']['reason'])
        self.assertEqual(bad['LINE']['criterion'], 'high_frequency')
        for item in bad.values():
            self.assertTrue(item['reason'])   # every flag says why

    def test_line_noise_is_seen_even_with_band_pass(self):
        t, source, channels, rng = clean_channels()
        channels['LINE'] = source + 6 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 2, t.size)
        bad = run(channels, FilterConfig(True, 0.5, 45))
        self.assertEqual(bad['LINE']['criterion'], 'high_frequency')

    def test_non_finite_and_few_channels(self):
        t, source, channels, _ = clean_channels()
        data = np.stack(list(channels.values()))
        data[3, 100] = np.nan
        bad = detect(data, data, list(channels), SFREQ, BadChannelConfig())
        self.assertEqual(bad['E03']['criterion'], 'non_finite')
        # Two channels: relative criteria are skipped, only absolute ones remain.
        few = np.stack([source, np.zeros_like(source)])
        bad = detect(few, few, ['A', 'B'], SFREQ, BadChannelConfig())
        self.assertEqual(set(bad), {'B'})
        self.assertEqual(bad['B']['criterion'], 'flat')

    def test_robust_z(self):
        z = robust_z([1, 1, 1, 1, 1])
        self.assertTrue(np.all(z == 0))   # no spread: nothing is an outlier
        z = robust_z([10, 11, 9, 10, 50])
        self.assertGreater(z[-1], 5)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            BadChannelConfig(min_correlation=1.5).validate()


class ManualMarkTests(unittest.TestCase):
    """Channels marked by hand are saved in the annotations and excluded by the batch."""

    def setUp(self):
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_core_ica import fixture
        from tps.core.data import open_recording
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.annotations = self.root / 'annotations'
        self.study, _ = fixture(self.data)
        self.recording = open_recording(self.study)
        self.addCleanup(self.recording.close)

    def store(self):
        from tps.core.annotations import AnnotationStore
        return AnnotationStore(self.annotations, self.recording, self.data)

    def test_marks_round_trip_and_reach_the_batch(self):
        import json
        from tps.core import ica
        from tps.pipeline import batch
        store = self.store()
        store.save()   # a file without marks, as before this feature
        self.assertNotIn('bad_channels', json.loads(store.path.read_text(encoding='utf-8')))
        self.assertEqual(self.store().bad_channels, {})
        store.mark_bad_channel('Cz', 'electrodo suelto')
        store.save()
        loaded = self.store()
        self.assertFalse(loaded.blocked)
        self.assertEqual(loaded.bad_channels['Cz']['reason'], 'electrodo suelto')
        with self.assertRaises(ValueError):
            loaded.mark_bad_channel('XX')
        selection = ica.snapshot([self.study], self.data, self.annotations, 'original')[str(self.study.path)]
        self.assertEqual(selection['bad_channels'], {'Cz': 'electrodo suelto'})
        cache = self.root / 'cache'
        cache.mkdir()
        result = batch.run_one(self.study, cache, 'test', selection, self.data, self.annotations)
        rows = [r for r in result['exclusions'] if r['level'] == 'channel']
        self.assertEqual([(r['channel'], r['criterion']) for r in rows], [('Cz', 'manual')])
        self.assertIn('electrodo suelto', rows[0]['reason'])
        self.assertNotIn('Cz', {r['channel'] for r in result['features']})
        # Unmarking changes the snapshot, so the cached result is no longer valid.
        loaded.unmark_bad_channel('Cz')
        loaded.save()
        again = ica.snapshot([self.study], self.data, self.annotations, 'original')[str(self.study.path)]
        self.assertFalse(batch.finished(batch.cache_path(cache, self.study), 'test', again))


class ChannelSpectraTests(ManualMarkTests):
    """«Todos los canales» spectra on the 3-channel fixture."""
    test_marks_round_trip_and_reach_the_batch = None   # reuse only the fixture, not the test

    def test_channel_spectra_match_single_channel_analysis(self):
        from matplotlib.figure import Figure
        from tps.core.spectral import (SpectralConfig, analyze_recording, channel_spectra,
                                       plot_channel_spectra)
        filtering = FilterConfig(True, 1, 45)
        result = channel_spectra(self.recording, filtering, SpectralConfig(), self.store(), True)
        self.assertEqual(result['channels'], ['Fp1', 'Fp2', 'Cz'])
        self.assertEqual(result['outliers'], [])   # fewer than 5 channels: no outlier test
        single = analyze_recording(self.recording, 'Cz', filtering, SpectralConfig(), self.store(), True)
        np.testing.assert_allclose(result['psd_uv2_hz'][2], single['psd_uv2_hz'])
        plot_channel_spectra(Figure(), result, db=True, bad={'Fp1'})   # draws without maps (< 4 placed)
        with self.assertRaises(ValueError):   # no reviewed épocas and no exploratory inclusion
            channel_spectra(self.recording, filtering, SpectralConfig(), self.store(), False)


if __name__ == '__main__':
    unittest.main()
