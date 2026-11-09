import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from tps.gui.app import App
from test_core_ica import fixture


class ICAGuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)/'data'
        fixture(self.data)
        self.app = App(self.root, self.data, Path(self.temp.name)/'annotations')
        self.addCleanup(self.close)
        self.wait()
        self.app.patient.set('sub-01'); self.app.select_patient()
        self.app.study_box.current(0); self.app.select_study(); self.wait()
        self.app.channel_box.current(0); self.app.select_channel(); self.wait()

    def close(self):
        if self.app.store:
            self.app.store.dirty = False
        self.app.close(); self.app.executor.shutdown(wait=True)

    def wait(self, fit=False):
        deadline = time.monotonic() + 30
        while (self.app.busy or (fit and self.app.ica_job)) and time.monotonic() < deadline:
            self.root.update(); time.sleep(.01)
        self.root.update()
        self.assertFalse(self.app.busy)
        if fit:
            self.assertIsNone(self.app.ica_job)

    def test_fit_inspect_preview_confirm_toggle_and_restore(self):
        a = self.app
        a.open_ica(); a.start_ica(); self.wait(fit=True)
        self.assertIsNotNone(a.ica_model, a.ica_message.get())
        self.assertEqual(a.exclusions_text.get(), '')
        a.inspect_ica(); self.wait()
        self.assertEqual(len(a.ica_figure.axes), 3)
        a.preview_ica(); self.wait()
        self.assertEqual(len(a.ica_figure.axes[0].lines), 2)
        with patch('tps.gui.ica.messagebox.askyesno', return_value=True):
            a.confirm_ica(); self.wait()
        revision = a.recording.ica_context['revision']
        self.assertIn(revision[:8], a.signal_label.get())
        a.start_ica(); a.cancel_ica()
        self.assertEqual(a.recording.ica_context['revision'], revision)
        a.cancel_ica(close=True)
        a.ica_enabled.set(False); a.toggle_ica(); self.wait()
        self.assertIsNone(a.recording.ica_context)
        a.ica_enabled.set(True); a.toggle_ica(); self.wait()
        self.assertEqual(a.recording.ica_context['revision'], revision)
        a.select_study(); self.wait()
        self.assertEqual(a.recording.ica_context['revision'], revision)

    def test_cancel_and_study_switch_discard_draft(self):
        a = self.app
        a.open_ica(); a.start_ica()
        process = a.ica_job
        a.cancel_ica()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(a.recording.ica_context)
        a.start_ica(); process = a.ica_job
        a.select_patient(); self.wait()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(a.ica_dialog)
        self.assertFalse(list((Path(self.temp.name)/'annotations').glob('ica/**/latest.json')))
