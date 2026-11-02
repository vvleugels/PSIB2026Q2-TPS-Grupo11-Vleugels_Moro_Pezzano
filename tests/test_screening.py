from pathlib import Path
import unittest

import mne
import numpy as np

from eeg_viewer.data import Recording, Study
from eeg_viewer.screening import ScreeningConfig, scan_recording
from eeg_viewer.processing import FilterConfig


class ScreeningTests(unittest.TestCase):
    def test_flags_channels_and_partial_epochs_without_changing_samples(self):
        times = np.arange(2250) / 100
        values = np.tile(20e-6 * np.sin(2 * np.pi * 5 * times), (2, 1))
        values[1, 500:1000] *= 5  # High amplitude on Fp1 only.
        values[0, 1000:1500] = 0  # Flat Cz.
        values[0, 1500] = np.nan  # Invalid data is never labeled clean.
        raw = mne.io.RawArray(values, mne.create_info(['Cz', 'Fp1'], 100, 'eeg'), verbose=False)
        recording = Recording(Study(Path('synthetic.set'), 'sub-01', 'ses-1', 'eyesopen'), raw, {})
        self.addCleanup(recording.close)
        before = raw.get_data().copy()
        result = scan_recording(recording, ['Cz', 'Fp1'], ScreeningConfig())
        self.assertEqual((result['total'], result['partial']), (5, 1))
        self.assertEqual(set(result['findings']), {1, 2, 3})
        self.assertEqual(result['findings'][1][0]['channel'], 'Fp1')
        self.assertEqual(result['findings'][1][0]['kind'], 'Alta')
        self.assertAlmostEqual(result['findings'][1][0]['peak_to_peak_uv'], 200)
        self.assertEqual(result['findings'][2][0]['kind'], 'Plana')
        self.assertEqual(result['findings'][3][0]['kind'], 'No finitos')
        selected = scan_recording(recording, ['Cz'], ScreeningConfig())
        self.assertEqual(set(selected['findings']), {2, 3})
        np.testing.assert_array_equal(raw.get_data(), before)
        self.assertEqual(recording.filter.channels, {})

    def test_strict_limits_invalid_thresholds_and_single_sample_tail(self):
        # Exactly at a threshold does not cross it. A one-sample tail is not flat.
        values = np.array([[0, 150e-6, 0, 1e-6, 0]])
        raw = mne.io.RawArray(values, mne.create_info(['Cz'], .4, 'eeg'), verbose=False)
        recording = Recording(Study(Path('synthetic.set'), 'sub-01', 'ses-1', 'eyesopen'), raw, {})
        self.addCleanup(recording.close)
        result = scan_recording(recording, ['Cz'], ScreeningConfig())
        self.assertEqual(set(result['findings']), {2})
        self.assertEqual(result['findings'][2][0]['kind'], 'Tramo insuficiente')
        for high, flat in ((0, 0), (1, 1), (1, -1), (float('nan'), 1), (150, float('inf'))):
            with self.subTest(high=high, flat=flat), self.assertRaises(ValueError):
                scan_recording(recording, ['Cz'], ScreeningConfig(high, flat))
        for channels in ([], ['missing'], ['Cz', 'Cz']):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                scan_recording(recording, channels, ScreeningConfig())

    def test_continuous_filter_removes_drift_before_scanning_and_preserves_source(self):
        times = np.arange(12000) / 100
        values = (500e-6 * np.sin(2 * np.pi * .02 * times)
                  + 20e-6 * np.sin(2 * np.pi * 10 * times))[None]
        raw = mne.io.RawArray(values, mne.create_info(['Cz'], 100, 'eeg'), verbose=False)
        recording = Recording(Study(Path('synthetic.set'), 'sub-01', 'ses-1', 'eyesopen'), raw, {})
        self.addCleanup(recording.close)
        before = raw.get_data().copy()
        original = scan_recording(recording, ['Cz'], ScreeningConfig())
        filtering = FilterConfig(True, 1, 30)
        filtered = scan_recording(recording, ['Cz'], ScreeningConfig(), filtering)
        self.assertEqual(original['findings'][10][0]['kind'], 'Alta')
        self.assertNotIn(10, filtered['findings'])
        self.assertEqual(filtered['filter'], filtering)
        self.assertFalse(original['filter'].enabled)
        # A low limit exposes the actual amplitude of the continuous filtered slice.
        lower_limit = scan_recording(recording, ['Cz'], ScreeningConfig(30, 1), filtering)
        expected = np.ptp(recording.filter.channels['Cz'][5000:5500]) * 1e6
        self.assertAlmostEqual(lower_limit['findings'][10][0]['peak_to_peak_uv'], expected)
        np.testing.assert_array_equal(raw.get_data(), before)
        raw._data[0, 6000] = np.nan
        with self.assertRaisesRegex(ValueError, 'barrido original'):
            scan_recording(recording, ['Cz'], ScreeningConfig(), filtering)
