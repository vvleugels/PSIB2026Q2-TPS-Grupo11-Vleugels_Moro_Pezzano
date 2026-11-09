import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from tps.core.data import annex_placeholder, discover, open_recording


def make_recording(folder, subject='sub-01', session='ses-1', task='eyesopen'):
    directory = Path(folder) / subject / session / 'eeg'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{subject}_{session}_task-{task}_eeg.set'
    # A known waveform in EEGLAB's native µV units, channel-major per sample.
    values = np.vstack((np.arange(1250) / 10, -np.arange(1250) / 5)).astype('<f4')
    values.ravel(order='F').tofile(path.with_suffix('.fdt'))
    savemat(path, {'EEG': {'data': path.with_suffix('.fdt').name, 'nbchan': 2,
                           'pnts': 1250, 'trials': 1, 'srate': 100., 'xmin': 0.,
                           'xmax': 12.49,
                           'chanlocs': np.array([{'labels': 'Cz'}, {'labels': 'Fp1'}], dtype=object)}})
    path.with_suffix('.json').write_text(json.dumps({'EEGReference': 'FCz'}))
    return path, values


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path, self.values = make_recording(self.temp.name)

    def recording(self):
        recording = open_recording(discover(self.temp.name)['sub-01'][0])
        self.addCleanup(recording.close)
        return recording

    def test_exact_values_units_time_and_partial_last_window(self):
        r = self.recording()
        self.assertFalse(r.raw.preload)
        self.assertEqual(r.channels, ['Cz', 'Fp1'])
        first = r.read_window('Fp1', 0)
        np.testing.assert_allclose(first.values_uv, self.values[1, :1000])
        np.testing.assert_allclose(first.times, np.arange(1000) / 100)
        last = r.read_window('Cz', 10)
        self.assertEqual((last.start, last.stop), (10, 12.5))
        np.testing.assert_allclose(last.values_uv, self.values[0, 1000:])
        self.assertEqual(len(last.times), 250)
        self.assertEqual(len(r.read_window('Cz', 12.4999).times), 1)

    def test_invalid_time_or_channel(self):
        r = self.recording()
        for start in (-1, 12.5, float('nan'), float('inf')):
            with self.subTest(start=start), self.assertRaises(ValueError):
                r.read_window('Cz', start)
        with self.assertRaises(ValueError):
            r.read_window('missing', 0)

    def test_read_channels_equals_per_channel_windows(self):
        r = self.recording()
        for channels in (['Cz'], ['Fp1', 'Cz'], ['Cz', 'Fp1', 'Cz']):
            for start, seconds in ((0, 10), (3.33, 2), (10, 10), (12.4999, 10)):
                with self.subTest(channels=channels, start=start, seconds=seconds):
                    segment = r.read_channels(channels, start, seconds=seconds)
                    windows = [r.read_window(c, start, seconds) for c in channels]
                    np.testing.assert_array_equal(segment.values_uv,
                                                  np.stack([w.values_uv for w in windows]))
                    np.testing.assert_array_equal(segment.times, windows[0].times)
                    self.assertEqual((segment.start, segment.stop),
                                     (windows[0].start, windows[0].stop))
                    self.assertEqual(segment.channels, tuple(channels))
        for start in (-1, 12.5, float('nan')):
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'El inicio debe'):
                r.read_channels(['Cz', 'Fp1'], start)
        with self.assertRaises(ValueError):
            r.read_channels(['Cz', 'missing'], 0)

    def test_inventory_sort_and_available_studies(self):
        make_recording(self.temp.name, subject='sub-10')
        make_recording(self.temp.name, subject='sub-2', session='ses-2', task='eyesclosed')
        (Path(self.temp.name) / 'sub-3').mkdir()
        inv = discover(self.temp.name)
        self.assertEqual(list(inv), ['sub-01', 'sub-2', 'sub-3', 'sub-10'])
        self.assertEqual(inv['sub-3'], [])
        self.assertEqual(inv['sub-2'][0].condition, 'Privación del sueño')
        self.assertEqual(inv['sub-2'][0].eyes, 'Ojos cerrados')

    def test_missing_and_malformed_files(self):
        self.path.with_suffix('.json').write_text('broken json')
        self.assertTrue(self.recording().warning)
        self.path.with_suffix('.json').unlink()
        self.assertEqual(self.recording().metadata, {})
        self.path.with_suffix('.fdt').unlink()
        with self.assertRaises((FileNotFoundError, OSError)):
            r = self.recording()
            r.read_window('Cz', 0)

    # A recording whose data was never downloaded (git-annex) must still be listed, so it can
    # be reported, and opening it must fail with a clear FileNotFoundError. On Linux/macOS the
    # file is a broken symlink; on Windows it is a small text pointer. Both forms are tested.

    def symlink_or_skip(self, link, target):
        """Create a symlink, or skip when the system does not allow it.

        Windows needs Developer Mode or admin rights (WinError 1314); the Windows form of
        the same case (text pointer) is covered by the placeholder tests below."""
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f'No se pueden crear symlinks en este sistema: {exc}')

    def test_broken_set_link_stays_discoverable(self):
        self.path.unlink()
        self.symlink_or_skip(self.path, 'missing.set')
        self.assertTrue(annex_placeholder(self.path))
        inv = discover(self.temp.name)
        self.assertEqual(len(inv['sub-01']), 1)
        with self.assertRaisesRegex(FileNotFoundError, 'git-annex'):
            open_recording(inv['sub-01'][0])

    def test_annex_text_placeholder_stays_discoverable(self):
        # How git checks out an annexed file without content on Windows.
        self.path.write_text('../../../.git/annex/objects/61/85/SHA256E-s629416--ab.set/'
                             'SHA256E-s629416--ab.set')
        self.assertTrue(annex_placeholder(self.path))
        inv = discover(self.temp.name)
        self.assertEqual(len(inv['sub-01']), 1)
        with self.assertRaisesRegex(FileNotFoundError, 'git-annex'):
            open_recording(inv['sub-01'][0])

    def test_set_removed_after_discovery_raises_file_not_found(self):
        study = discover(self.temp.name)['sub-01'][0]
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'no está disponible'):
            open_recording(study)

    def test_real_recording_is_not_a_placeholder(self):
        self.assertFalse(annex_placeholder(self.path))
        small = self.path.with_name('notes.txt')
        small.write_text('a short text file that is not a pointer')
        self.assertFalse(annex_placeholder(small))
        self.assertFalse(annex_placeholder(self.path.with_name('absent.set')))

    def test_truncated_binary_is_rejected_before_plotting(self):
        with self.path.with_suffix('.fdt').open('r+b') as stream:
            stream.truncate(100)
        with self.assertRaisesRegex(ValueError, 'Archivo EEG incompleto'):
            self.recording()


if __name__ == '__main__':
    unittest.main()
