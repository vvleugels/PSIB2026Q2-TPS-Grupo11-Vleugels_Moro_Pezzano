from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from mne.filter import filter_data

from eeg_viewer.annotations import AnnotationStore, make_epochs, STATES
from eeg_viewer.data import Recording, Study, discover, open_recording
from eeg_viewer.processing import FilterConfig, FILTER_OPTIONS
from test_data import make_recording


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.sfreq = 500
        self.times = np.arange(60000) / self.sfreq
        self.original = (np.sin(2 * np.pi * 10 * self.times)
                         + np.sin(2 * np.pi * 80 * self.times)
                         + np.sin(2 * np.pi * .02 * self.times)) * 1e-6
        raw = mne.io.RawArray(np.array([self.original, self.original * 2]),
                             mne.create_info(['Cz', 'Fp1'], self.sfreq, 'eeg'), verbose=False)
        self.recording = Recording(Study(Path('synthetic.set'), 'sub-01', 'ses-1', 'eyesopen'), raw, {})
        self.addCleanup(self.recording.close)

    def test_attenuation_passband_no_source_mutation(self):
        r = self.recording
        config = FilterConfig(True)
        r.read_channels(['Cz'], 0, config)
        filtered = r.filter.channels['Cz']
        interior = slice(10000, -10000)
        t = self.times[interior]
        # Least-squares projection separates the components, including the slow drift.
        design = np.column_stack([np.sin(2*np.pi*f*t) for f in (10, 80, .02)] + [np.ones(len(t))])
        amplitudes = np.linalg.lstsq(design, filtered[interior] * 1e6, rcond=None)[0]
        self.assertAlmostEqual(amplitudes[0], 1, delta=.03)
        self.assertLess(abs(amplitudes[1]), .01)
        self.assertLess(abs(amplitudes[2]), .2)
        np.testing.assert_array_equal(r.raw.get_data()[0], self.original)

    def test_continuous_slices_multichannel_and_cache(self):
        r = self.recording
        config = FilterConfig(True)
        segment = r.read_channels(['Cz', 'Fp1'], 5, config)
        expected = filter_data(self.original[None], self.sfreq, .2, 45,
                               **FILTER_OPTIONS, pad='reflect_limited', verbose=False)[0]
        np.testing.assert_allclose(segment.values_uv[0], expected[2500:7500]*1e6)
        np.testing.assert_allclose(segment.values_uv[1], segment.values_uv[0]*2)
        cached = r.filter.channels['Cz']
        r.read_channels(['Cz'], 10, config)
        self.assertIs(cached, r.filter.channels['Cz'])
        r.read_channels(['Cz'], 10, FilterConfig(True, 1, 30))
        self.assertIsNot(cached, r.filter.channels['Cz'])
        self.assertNotIn('Fp1', r.filter.channels)

    def test_invalid_cutoffs_and_short_recording(self):
        for low, high in ((0, 45), (45, 1), (.2, 250), (float('nan'), 45)):
            with self.assertRaises(ValueError):
                FilterConfig(True, low, high).validate(500)
        raw = mne.io.RawArray(np.zeros((1, 500)), mne.create_info(['Cz'], 500, 'eeg'), verbose=False)
        r = Recording(self.recording.study, raw, {})
        self.addCleanup(r.close)
        with self.assertRaisesRegex(ValueError, 'demasiado corto'):
            r.read_channels(['Cz'], 0, FilterConfig(True))
        with self.assertRaisesRegex(ValueError, 'demasiado corto'):
            r.read_channels(['Cz'], 0, FilterConfig(True, 1e-300, 45))


class EpochTests(unittest.TestCase):
    def test_sixty_epochs_and_partial_without_gaps(self):
        epochs = make_epochs(150000, 500)
        self.assertEqual(len(epochs), 60)
        self.assertFalse(any(e.partial for e in epochs))
        self.assertEqual(epochs[-1].last, 150000)
        for n, sf in ((6250, 500), (1321, 99.5)):
            epochs = make_epochs(n, sf)
            self.assertTrue(epochs[-1].partial)
            self.assertEqual(epochs[0].first, 0)
            self.assertEqual(epochs[-1].last, n)
            for left, right in zip(epochs, epochs[1:]):
                self.assertEqual(left.last, right.first)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data'
        self.path, _ = make_recording(self.data)
        self.recording = open_recording(discover(self.data)['sub-01'][0])
        self.addCleanup(self.recording.close)
        self.directory = Path(self.temp.name) / 'annotations'
        self.store = self.open_store()

    def open_store(self):
        return AnnotationStore(self.directory, self.recording, self.data)

    def mark(self):
        self.store.mark(0, STATES[2], 'Ocular', 'Parpadeo', ['Cz'], FilterConfig(True))

    def test_roundtrip_global_mark_context_and_partial(self):
        self.mark()
        self.store.save()
        loaded = self.open_store()
        self.assertFalse(loaded.blocked)
        self.assertFalse(loaded.dirty)
        self.assertEqual(loaded.records[0]['state'], STATES[2])
        self.assertEqual(loaded.records[0]['filter']['low'], .2)
        self.assertEqual(loaded.records[0]['channels'], ['Cz'])
        self.assertTrue(loaded.records[-1]['partial'])
        loaded.mark(0, STATES[1], '', '', ['Fp1'], FilterConfig())
        loaded.save()
        self.assertEqual(self.open_store().records[0]['state'], STATES[1])

    def test_corrupt_file_preserved_and_overwrite_blocked(self):
        self.directory.mkdir()
        self.store.path.write_text('{broken')
        loaded = self.open_store()
        self.assertTrue(loaded.blocked)
        with self.assertRaises(ValueError):
            loaded.save()
        self.assertEqual(loaded.path.read_text(), '{broken')

    def test_source_change_and_external_edit_are_protected(self):
        self.mark()
        self.store.save()
        self.path.with_suffix('.json').write_text('{}')
        loaded = self.open_store()
        self.assertTrue(loaded.blocked)
        with self.assertRaises(ValueError):
            self.store.save()

    def test_concurrent_annotation_edit_is_not_overwritten(self):
        self.mark()
        self.store.save()
        self.store.path.write_text('external edit')
        with self.assertRaisesRegex(ValueError, 'externamente'):
            self.store.save()
        self.assertEqual(self.store.path.read_text(), 'external edit')

    def test_atomic_failure_keeps_dirty_and_old_file_then_retry(self):
        self.store.save()
        before = self.store.path.read_bytes()
        self.mark()
        with patch('eeg_viewer.annotations.os.replace', side_effect=OSError('Disco lleno')):
            with self.assertRaises(OSError):
                self.store.save()
        self.assertTrue(self.store.dirty)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(list(self.directory.glob('*.tmp')), [])
        self.store.save()
        self.assertFalse(self.store.dirty)

    def test_malformed_review_context_is_protected(self):
        self.mark()
        self.store.save()
        payload = json.loads(self.store.path.read_text())
        payload['epochs'][0]['filter'] = None
        self.store.path.write_text(json.dumps(payload))
        self.assertTrue(self.open_store().blocked)
