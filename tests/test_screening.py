from pathlib import Path
import unittest

import mne
import numpy as np

from eeg_viewer.data import Recording, Study
from eeg_viewer.screening import ScreeningConfig, scan_recording
from eeg_viewer.processing import FILTER_OPTIONS, FilterConfig
from eeg_viewer.annotations import make_epochs
from mne.filter import filter_data


def reference_scan(recording, channels, config, filter_config=FilterConfig()):
    """The per-channel / per-epoch algorithm scan_recording used before blocking."""
    channels = tuple(channels)
    continuous = None
    if filter_config.enabled:
        continuous = {}
        for channel in channels:
            if not np.all(np.isfinite(recording.raw.get_data(picks=[channel]))):
                raise ValueError(f'{channel}: hay muestras no finitas. '
                                 'Usá el barrido original para localizarlas antes de filtrar.')
            continuous[channel] = recording.filter.channel(channel, filter_config)
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    findings = {}
    for epoch in epochs:
        values = ([continuous[channel][epoch.first:epoch.last] for channel in channels]
                  if continuous is not None else recording.raw.get_data(
                      picks=list(channels), start=epoch.first, stop=epoch.last))
        issues = []
        for channel, samples in zip(channels, values):
            if not np.all(np.isfinite(samples)):
                kind, amplitude = 'No finitos', None
            elif len(samples) < 2:
                kind, amplitude = 'Tramo insuficiente', None
            else:
                amplitude = float(np.ptp(samples) * 1e6)
                if amplitude > config.high_uv:
                    kind = 'Alta'
                elif amplitude < config.flat_uv:
                    kind = 'Plana'
                else:
                    continue
            issues.append(dict(channel=channel, kind=kind, peak_to_peak_uv=amplitude))
        if issues:
            findings[epoch.index] = issues
    return dict(findings=findings, channels=channels, config=config, filter=filter_config,
                total=len(epochs), partial=sum(epoch.partial for epoch in epochs))


def synthetic(n_channels=37, n_times=12_250, sfreq=100, seed=0):
    """Noisy channels with high, flat and drifting stretches and a partial last epoch."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_times) / sfreq
    values = 20e-6 * rng.standard_normal((n_channels, n_times))
    values += 300e-6 * np.sin(2 * np.pi * .03 * times) * rng.random((n_channels, 1))
    for row in range(0, n_channels, 3):
        start = rng.integers(0, n_times - 600)
        values[row, start:start + 500] *= 8
    for row in range(1, n_channels, 5):
        start = rng.integers(0, n_times - 600)
        values[row, start:start + 600] = 1e-7
    names = [f'E{index}' for index in range(n_channels)]
    return values, names, sfreq


def recording_of(values, names, sfreq):
    raw = mne.io.RawArray(values.copy(), mne.create_info(names, sfreq, 'eeg'), verbose=False)
    return Recording(Study(Path('synthetic.set'), 'sub-01', 'ses-1', 'eyesopen'), raw, {})


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
