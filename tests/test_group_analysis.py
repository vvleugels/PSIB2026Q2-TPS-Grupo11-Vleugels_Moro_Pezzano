import unittest
from pathlib import Path

import mne
import numpy as np

from eeg_viewer.data import Recording, Study
from eeg_viewer.group_analysis import extract_recording_features
from eeg_viewer.screening import ScreeningConfig


class GroupAnalysisTests(unittest.TestCase):
    def test_ten_hz_sine_peaks_in_alpha_and_carries_order_and_epoch_counts(self):
        sfreq = 500.0
        seconds = 80.0
        times = np.arange(int(sfreq * seconds)) / sfreq
        sine = 20e-6 * np.sin(2 * np.pi * 10 * times)
        info = mne.create_info(['Cz', 'Pz'], sfreq, ch_types='eeg')
        raw = mne.io.RawArray(np.vstack([sine, sine]), info, verbose='ERROR')
        study = Study(Path('sub-01_ses-1_task-eyesopen_eeg.set'), 'sub-01', 'ses-1', 'eyesopen')
        recording = Recording(study, raw, {'EEGReference': 'FCz'})
        try:
            features, summary = extract_recording_features(recording, 'NS->SD')
        finally:
            recording.close()
        channel = [row for row in features if row['channel'] == 'Cz']
        powers = {row['band']: row['absolute_uv2'] for row in channel}
        self.assertEqual(max(powers, key=powers.get), 'Alfa')
        self.assertTrue(all(row['SessionOrder'] == 'NS->SD' for row in features))
        self.assertEqual(summary['epochs_total'], 16)
        self.assertEqual(summary['epochs_used'], 16)
        self.assertEqual(summary['epochs_rejected_by_screening'], 0)
        self.assertEqual(summary['status'], 'ok')

    def test_channel_mode_keeps_clean_epochs_independently_and_records_channel_counts(self):
        sfreq = 500.0
        times = np.arange(int(sfreq * 80.0)) / sfreq
        clean = 20e-6 * np.sin(2 * np.pi * 10 * times)
        noisy = clean.copy()
        noisy[:int(5 * sfreq)] = 110e-6 * np.sin(2 * np.pi * 10 * times[:int(5 * sfreq)])
        raw = mne.io.RawArray(np.vstack([noisy, clean]),
                              mne.create_info(['Fp1', 'Cz'], sfreq, ch_types='eeg'),
                              verbose='ERROR')
        recording = Recording(
            Study(Path('sub-12_ses-1_task-eyesopen_eeg.set'), 'sub-12', 'ses-1', 'eyesopen'),
            raw, {'EEGReference': 'FCz'},
        )
        try:
            global_features, _ = extract_recording_features(
                recording, 'NS->SD', screening_config=ScreeningConfig(high_uv=150),
                rejection_mode='epoch', channels=['Fp1', 'Cz'])
            channel_features, summary = extract_recording_features(
                recording, 'NS->SD', screening_config=ScreeningConfig(high_uv=150),
                rejection_mode='channel', channels=['Fp1', 'Cz'])
        finally:
            recording.close()
        def count(rows, channel):
            return {row['epochs_used'] for row in rows if row['channel'] == channel}
        self.assertEqual(count(channel_features, 'Cz'), {16})
        self.assertEqual(len(count(global_features, 'Cz')), 1)
        self.assertLess(next(iter(count(channel_features, 'Fp1'))), 16)
        self.assertEqual({row['rejection_mode'] for row in channel_features}, {'channel'})
        self.assertEqual({row['high_threshold_uv'] for row in channel_features}, {150})
        self.assertEqual(summary['status'], 'ok')

    def test_channel_below_minimum_clean_epochs_is_excluded(self):
        sfreq = 500.0
        times = np.arange(int(sfreq * 40.0)) / sfreq
        sine = 20e-6 * np.sin(2 * np.pi * 10 * times)
        raw = mne.io.RawArray(np.vstack([sine, sine]),
                              mne.create_info(['Fp1', 'Cz'], sfreq, ch_types='eeg'),
                              verbose='ERROR')
        recording = Recording(
            Study(Path('sub-12_ses-1_task-eyesopen_eeg.set'), 'sub-12', 'ses-1', 'eyesopen'),
            raw, {'EEGReference': 'FCz'},
        )
        try:
            features, summary = extract_recording_features(recording, 'NS->SD', channels=['Fp1', 'Cz'])
        finally:
            recording.close()
        self.assertEqual(features, [])
        self.assertEqual(summary['status'], 'excluded_no_channel_with_min_epochs')
        self.assertEqual({row['channel'] for row in summary['channel_exclusions']}, {'Fp1', 'Cz'})


if __name__ == '__main__':
    unittest.main()
