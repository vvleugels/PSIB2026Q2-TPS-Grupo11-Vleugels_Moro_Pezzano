from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np

from tps.core.processing import ContinuousFilter, FilterConfig
from tps.core.annotations import AnnotationStore, STATES
from tps.core.data import Recording, Study
from tps.core.screening import scan_recording, ScreeningConfig
from tps.core.spectral import analyze_recording, SpectralConfig


class NotchTests(unittest.TestCase):
    def test_frequency_selection_cache_and_original(self):
        sfreq = 500
        t = np.arange(30 * sfreq) / sfreq
        original = sum(np.sin(2 * np.pi * f * t) for f in (10, 50, 60)) * 1e-6
        raw = mne.io.RawArray(original[None], mne.create_info(['Cz'], sfreq, 'eeg'), verbose=False)
        self.addCleanup(raw.close)
        cache = ContinuousFilter(raw)
        middle = slice(5 * sfreq, -5 * sfreq)
        def amplitude(signal, frequency):
            return abs(np.mean(signal[middle] * np.exp(-2j * np.pi * frequency * t[middle]))) * 2
        for frequency in (50, 60):
            for enabled in (False, True):
                with self.subTest(frequency=frequency, bandpass=enabled):
                    config = FilterConfig(enabled, 1, 90, frequency)
                    filtered = cache.channel('Cz', config)
                    self.assertLess(amplitude(filtered, frequency), 1e-8)
                    self.assertGreater(amplitude(filtered, 10), .95e-6)
                    self.assertGreater(amplitude(filtered, 110 - frequency), .9e-6)
        np.testing.assert_array_equal(cache.channel('Cz', FilterConfig()), original)
        np.testing.assert_array_equal(raw.get_data()[0], original)

    def test_nyquist_and_supported_frequencies(self):
        for frequency, sfreq in ((50, 100), (60, 120), (55, 500)):
            with self.subTest(frequency=frequency, sfreq=sfreq), self.assertRaises(ValueError):
                FilterConfig(notch=frequency).validate(sfreq)
        FilterConfig(notch=60).validate(500)

    def test_short_recording_rejected(self):
        raw = mne.io.RawArray(np.zeros((1, 9)), mne.create_info(['Cz'], 500, 'eeg'), verbose=False)
        self.addCleanup(raw.close)
        with self.assertRaisesRegex(ValueError, 'demasiado corto'):
            ContinuousFilter(raw).prepare(FilterConfig(notch=50))

    def test_notch_only_window_screening_spectrum_and_saved_context(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'signal.set'
            path.touch()
            sfreq = 500
            t = np.arange(30 * sfreq) / sfreq
            signal = (10 * np.sin(2*np.pi*10*t) + 100 * np.sin(2*np.pi*50*t)) * 1e-6
            raw = mne.io.RawArray(signal[None], mne.create_info(['Cz'], sfreq, 'eeg'), verbose=False)
            recording = Recording(Study(path, 'sub-01', 'ses-1', 'eyesopen'), raw, {})
            self.addCleanup(recording.close)
            config = FilterConfig(notch=50)
            window = recording.read_channels(['Cz'], 10, config)
            self.assertLess(np.ptp(window.values_uv), 21)
            self.assertTrue(scan_recording(recording, ['Cz'], ScreeningConfig())['findings'])
            self.assertNotIn(2, scan_recording(recording, ['Cz'], ScreeningConfig(), config)['findings'])
            store = AnnotationStore(Path(folder) / 'annotations', recording, folder)
            store.mark(2, STATES[1], '', '', ['Cz'], config)
            store.save()
            loaded = AnnotationStore(Path(folder) / 'annotations', recording, folder)
            self.assertFalse(loaded.blocked)
            self.assertEqual(loaded.records[2]['filter']['notch'], 50)
            result = analyze_recording(recording, 'Cz', config, SpectralConfig(), loaded)
            index = np.argmin(abs(result['frequencies'] - 50))
            self.assertLess(result['psd_uv2_hz'][index], .01)
            self.assertFalse(result['warnings'])
            changed = analyze_recording(recording, 'Cz', FilterConfig(notch=60), SpectralConfig(), loaded)
            self.assertTrue(any('otro filtrado' in warning for warning in changed['warnings']))
