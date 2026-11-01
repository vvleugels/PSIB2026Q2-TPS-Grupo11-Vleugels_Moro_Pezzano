from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest

import numpy as np

from eeg_viewer.app import App
from test_data import make_recording


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Se necesita una pantalla o Xvfb: {exc}')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        make_recording(self.temp.name)
        make_recording(self.temp.name, subject='sub-02', session='ses-2')
        (Path(self.temp.name) / 'sub-03').mkdir()
        self.app = App(self.root, Path(self.temp.name))
        self.addCleanup(self.app.close)
        self.wait_ready()

    def wait_ready(self):
        deadline = time.monotonic() + 10
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.root.update()
        self.assertFalse(self.app.busy, self.app.status.get())

    def select(self):
        self.app.patient.set('sub-01')
        self.app.select_patient()
        self.app.study_box.current(0)
        self.app.select_study()
        self.wait_ready()
        self.app.channel_box.current(0)
        self.app.select_channel()
        self.wait_ready()

    def test_cascade_navigation_grid_zoom_and_resize(self):
        a = self.app
        self.assertEqual(a.patient.get(), '')
        self.assertEqual(str(a.study_box.cget('state')), 'disabled')
        self.select()
        self.assertEqual(len(a.ax.lines[0].get_xdata()), 1000)
        ticks = a.ax.xaxis.get_majorticklocs()
        np.testing.assert_allclose(ticks % 5, 0)
        self.assertEqual(str(a.previous.cget('state')), 'disabled')
        a.navigate(10)
        self.wait_ready()
        self.assertEqual((a.start, a.stop), (10, 12.5))
        self.assertEqual(str(a.next.cget('state')), 'disabled')
        a.ax.set_xlim(11, 12)
        a.toolbar.home()
        np.testing.assert_allclose(a.ax.get_xlim(), [10, 12.5])
        a.start_text.set('2,5')
        a.go()
        self.wait_ready()
        self.assertEqual(a.start, 2.5)
        a.start_text.set('nan')
        a.go()
        self.assertIn('Ingresá', a.status.get())
        self.root.geometry('960x640')
        self.root.update()
        self.assertTrue(a.go_button.winfo_viewable())
        a.patient.set('sub-03')
        a.select_patient()
        self.assertEqual(a.study.get(), '')
        self.assertEqual(a.channel.get(), '')
        self.assertEqual(len(a.ax.lines), 0)
        self.assertEqual(str(a.study_box.cget('state')), 'disabled')

    def test_stale_async_result_cannot_restore_old_selection(self):
        self.select()
        a = self.app
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def slow_result():
            ready.set()
            release.wait(5)
            return 'stale'
        received = []
        a.submit(slow_result, received.append)
        self.assertTrue(ready.wait(2))
        a.patient.set('sub-02')
        a.select_patient()
        a.study_box.current(0)
        a.select_study()
        release.set()
        self.wait_ready()
        self.assertEqual(received, [])
        self.assertEqual(a.recording.study.participant, 'sub-02')
        self.assertEqual(a.channel.get(), '')
        self.assertEqual(len(a.ax.lines), 0)

    def test_failed_read_clears_previous_signal(self):
        self.select()
        def broken(*args):
            raise OSError('Archivo ilegible')
        self.app.recording.read_window = broken
        self.app.navigate(10)
        self.wait_ready()
        self.assertEqual(len(self.app.ax.lines), 0)
        self.assertIn('Archivo ilegible', self.app.status.get())
