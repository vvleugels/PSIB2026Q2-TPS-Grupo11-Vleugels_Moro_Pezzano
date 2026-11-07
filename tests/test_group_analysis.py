import unittest
from pathlib import Path

import mne
import numpy as np

from eeg_viewer.data import Recording, Study
from eeg_viewer.group_analysis import extract_recording_features


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


if __name__ == '__main__':
    unittest.main()
