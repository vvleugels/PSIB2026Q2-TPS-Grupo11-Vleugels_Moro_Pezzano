"""Annotations and ICA revisions shared through the repository work on another computer.

A copy of the same data on another computer has other file modification times, but the same
paths, sizes, sampling rate and samples; that must not block the shared files."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tps.core.annotations import REASONS, STATES, AnnotationStore, same_source, source_identity
from tps.core.data import open_recording
from tps.core.processing import FilterConfig


def touch_data(study):
    """Give every file of the recording another modification time, as a fresh download does."""
    for path in study.path.parent.glob(study.path.stem + '.*'):
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 3_600_000_000_000))


class SharedIdentityTests(unittest.TestCase):
    def setUp(self):
        from test_core_ica import fixture
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.data, self.annotations = root / 'data', root / 'annotations'
        self.study, _ = fixture(self.data)

    def open(self):
        recording = open_recording(self.study)
        self.addCleanup(recording.close)
        return recording

    def test_same_source_ignores_modification_time_only(self):
        identity = source_identity(self.open(), self.data)
        moved = json.loads(json.dumps(identity))
        for item in moved['files']:
            item['mtime_ns'] += 1
        self.assertTrue(same_source(identity, moved))
        resized = json.loads(json.dumps(identity))
        resized['files'][0]['size'] += 1
        self.assertFalse(same_source(identity, resized))
        shorter = dict(identity, n_times=identity['n_times'] - 1)
        self.assertFalse(same_source(identity, shorter))
        self.assertFalse(same_source(identity, {}))

    def test_annotations_from_another_computer_load_and_keep_their_identity(self):
        store = AnnotationStore(self.annotations, self.open(), self.data)
        store.mark(0, STATES[1], '', '', ['Cz'], FilterConfig())
        store.save()
        written = json.loads(store.path.read_text(encoding='utf-8'))['source']
        touch_data(self.study)
        other = AnnotationStore(self.annotations, self.open(), self.data)
        self.assertEqual(other.blocked, '')
        self.assertEqual(other.records[0]['state'], STATES[1])
        other.mark(1, STATES[2], REASONS[-1], '', ['Cz'], FilterConfig())
        other.save()
        # Saving keeps the identity already in the file: no churn between computers.
        self.assertEqual(json.loads(other.path.read_text(encoding='utf-8'))['source'], written)

    def test_ica_revision_from_another_computer_is_valid(self):
        from tps.core import ica
        recording = self.open()
        model, metadata = ica.fit_recording(recording, self.data)
        corrected, value = ica.confirm(recording, self.data, self.annotations, model, metadata, [])
        corrected.close()
        touch_data(self.study)
        _, saved = ica.read_revision(self.annotations, self.study)
        ica.validate_revision(self.open(), self.data, saved)   # does not raise
        self.assertEqual(ica.version_note(saved), '')   # fitted with this computer's versions

    def test_ica_revision_fitted_with_other_versions_is_applied_with_a_note(self):
        from tps.core import ica
        recording = self.open()
        model, metadata = ica.fit_recording(recording, self.data)
        corrected, value = ica.confirm(recording, self.data, self.annotations, model, metadata, [0])
        corrected.close()
        _, saved = ica.read_revision(self.annotations, self.study)
        saved['versions'] = dict(mne='1.6.0', numpy='1.26.0', scipy='1.11.0')   # a professor's computer
        ica.validate_revision(self.open(), self.data, saved)   # still valid
        note = ica.version_note(saved)
        self.assertIn('MNE 1.6.0', note)
        self.assertIn(ica.versions()['mne'], note)
        # A different recording is still rejected.
        saved['source']['n_times'] += 1
        with self.assertRaises(ValueError):
            ica.validate_revision(self.open(), self.data, saved)


if __name__ == '__main__':
    unittest.main()
