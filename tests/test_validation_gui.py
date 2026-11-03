from pathlib import Path
from types import SimpleNamespace
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from eeg_viewer.validation_store import Revision, atomic_write, encode
from eeg_viewer.validation_ui import ValidationApp
import test_validation as fixtures


class ValidationGuiTests(unittest.TestCase):
    def setUp(self):
        fixtures.RevisionTests.setUp(self)
        try:
            self.tkroot = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.tkroot.withdraw()
        raw = fixtures.synthetic()
        self.source.update(channels=raw.ch_names, n_times=int(raw.n_times))
        atomic_write(self.sample, encode(dict(records=[self.source])))
        self.revision = Revision.create(self.annotations, self.source, self.sample)
        recording = SimpleNamespace(raw=raw, sfreq=100., close=raw.close)
        loader = patch('eeg_viewer.validation_session.open_recording', return_value=recording)
        loader.start()
        self.addCleanup(loader.stop)
        self.app = ValidationApp(self.tkroot, self.revision)
        self.addCleanup(self.cleanup)
        self.wait_ready()

    def wait_ready(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.tkroot.update()
            time.sleep(.01)
        self.tkroot.update()
        self.assertFalse(self.app.busy, self.app.status.get())

    def cleanup(self):
        if not self.app.closed:
            self.app.close()
        self.app.executor.shutdown(wait=True)

    def test_blind_view_manual_save_navigation_and_no_suggestions(self):
        a = self.app
        self.assertIsNotNone(a.session)
        with self.assertRaisesRegex(ValueError, 'revisión ciega'):
            a.quality()
        self.assertEqual(len(a.figure.axes), 2)
        a.mark('physiological', False)
        a.move(1)
        self.assertEqual(a.current_epoch().start, 4.)
        a.mark('artifact', False)
        loaded = Revision(self.revision.directory)
        self.assertEqual(loaded.data['labels']['0:Fz']['state'], 'physiological')
        self.assertEqual(loaded.data['labels']['1:Fz']['state'], 'artifact')
        a.final.set(True)
        a.draw()
        self.assertFalse(a.final.get())

    def test_cancel_background_computation_does_not_publish(self):
        entered, release = threading.Event(), threading.Event()
        original = self.revision.path.read_bytes()
        results = []
        def work():
            entered.set()
            release.wait(5)
            self.app.session.checkpoint()
            return 'must not publish'
        self.app.submit(work, results.append)
        self.assertTrue(entered.wait(2))
        self.app.request_cancel()
        release.set()
        self.wait_ready()
        self.assertEqual(results, [])
        self.assertEqual(self.revision.path.read_bytes(), original)
        self.assertIn('cancelado', self.app.status.get())


if __name__ == '__main__':
    unittest.main()
