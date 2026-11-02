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
        import gc
        gc.collect()  # Dispose prior Tk objects on the UI thread, before starting workers.
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Se necesita una pantalla o Xvfb: {exc}')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        make_recording(self.temp.name)
        make_recording(self.temp.name, subject='sub-02', session='ses-2')
        (Path(self.temp.name) / 'sub-03').mkdir()
        self.app = App(self.root, Path(self.temp.name), Path(self.temp.name) / 'annotations')
        self.addCleanup(self.cleanup_app)
        self.wait_ready()

    def cleanup_app(self):
        if self.app.store:
            self.app.store.dirty = False  # Failed tests must not open a modal on cleanup.
        self.app.close()
        self.app.executor.shutdown(wait=True)

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

    def test_filter_transaction_rolls_back_then_multichannel_preserves_time(self):
        from eeg_viewer.processing import FilterConfig
        self.select()
        a = self.app
        before = a.ax.lines[0].get_ydata().copy()
        a.apply_config(FilterConfig(True))  # 12.5 s is too short for a 0.2-Hz FIR.
        self.wait_ready()
        self.assertFalse(a.config.enabled)
        np.testing.assert_array_equal(a.ax.lines[0].get_ydata(), before)
        self.assertIn('demasiado corto', a.status.get())
        a.start_text.set('5'); a.go(); self.wait_ready()
        a.select_epoch(1)
        a.apply_config(FilterConfig(True, 2, 30)); self.wait_ready()
        self.assertTrue(a.config.enabled)
        self.assertEqual((a.start, a.selected_epoch), (5, 1))
        a.multi.set(True); a.toggle_multi()
        a.channel_list.selection_set(0, 1); a.multi_changed(); self.wait_ready()
        self.assertEqual(len(a.figure.axes), 2)
        self.assertEqual(a.start, 5)
        a.multi.set(False); a.toggle_multi(); self.wait_ready()
        self.assertEqual(len(a.figure.axes), 1)
        self.assertEqual(a.start, 5)

    def test_global_annotations_recover_and_keyboard_ignores_text(self):
        from types import SimpleNamespace
        self.select()
        a = self.app
        a.select_epoch(1)
        a.mark_state.set('Con artefactos'); a.reason.set('Ocular'); a.note.set('parpadeo')
        a.apply_mark()
        self.assertEqual(a.save_label.get(), 'Cambios sin guardar')
        self.assertFalse(a.store.path.exists())
        self.assertTrue(a.save_annotations())
        self.assertEqual(a.save_label.get(), 'Guardado')
        a.channel_box.current(1); a.select_channel(); self.wait_ready()
        self.assertEqual(a.selected_epoch, 1)
        self.assertEqual(a.store.records[1]['state'], 'Con artefactos')
        a.shortcut(SimpleNamespace(widget=a.note_entry), 'Aceptada')
        self.assertEqual(a.store.records[1]['state'], 'Con artefactos')
        a.select_study(); self.wait_ready()
        self.assertEqual(a.store.records[1]['note'], 'parpadeo')
        self.assertEqual(a.store.records[1]['state'], 'Con artefactos')

    def test_click_and_navigation_select_visible_epoch(self):
        from types import SimpleNamespace
        self.select()
        a = self.app
        a.plot_clicked(SimpleNamespace(inaxes=a.ax, xdata=7.0))
        self.assertEqual(a.selected_epoch, 1)
        a.navigate(10); self.wait_ready()
        self.assertEqual(a.selected_epoch, 2)
        a.select_epoch(0, navigate=True); self.wait_ready()
        self.assertEqual(a.start, 0)

    def test_failed_save_preserves_changes_and_cancels_navigation(self):
        from unittest.mock import patch
        self.select()
        a = self.app
        with patch.object(a.store, 'save', side_effect=OSError('Disco lleno')):
            a.mark_state.set('Aceptada'); a.apply_mark()
            self.assertTrue(a.store.dirty)
            self.assertEqual(a.save_label.get(), 'Cambios sin guardar')
            with patch.object(a, 'unsaved_choice', return_value='cancel'):
                a.patient.set('sub-02'); a.select_patient()
                self.assertEqual(a.patient.get(), 'sub-01')
                self.assertIsNotNone(a.recording)
                a.close()
                self.assertFalse(a.closed)
        self.assertTrue(a.save_annotations())

    def test_obsolete_filter_completion_cannot_change_next_study(self):
        from eeg_viewer.processing import FilterConfig
        self.select()
        a = self.app
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = a.recording.filter.prepare
        def slow(config):
            ready.set(); release.wait(5)
            return original(config)
        a.recording.filter.prepare = slow
        a.apply_config(FilterConfig(True, 2, 30))
        self.assertTrue(ready.wait(2))
        a.patient.set('sub-02'); a.select_patient()
        a.study_box.current(0); a.select_study()
        release.set(); self.wait_ready()
        self.assertFalse(a.config.enabled)
        self.assertEqual(a.recording.study.participant, 'sub-02')
        self.assertIsNone(a.displayed)

    def test_manual_save_required_for_spectral_analysis(self):
        self.select()
        a = self.app
        a.mark_state.set('Aceptada'); a.apply_mark()
        self.assertTrue(a.store.dirty)
        self.assertFalse(a.store.path.exists())
        a.calculate_spectrum()
        self.assertIn('Guardá', a.spectral_message.get())
        self.assertIsNone(a.spectral_result)
        a.save_annotations(); a.calculate_spectrum(); self.wait_ready()
        self.assertIsNotNone(a.spectral_result, a.spectral_message.get())
        self.assertEqual(a.spectral_result['sessions'][0]['counts']['used'], 1)
        self.assertEqual(len(a.band_table.get_children()),4)
        self.assertEqual(str(a.export_button.cget('state')), 'normal')
        a.select_epoch(1); a.mark_state.set('Aceptada'); a.apply_mark()
        self.assertIsNone(a.spectral_result)
        self.assertEqual(str(a.export_button.cget('state')), 'disabled')

    def test_exploratory_spectrum_and_channel_invalidation(self):
        self.select()
        a = self.app
        a.calculate_spectrum(); self.wait_ready()
        self.assertIsNone(a.spectral_result)
        self.assertIn('no hay epochs', a.spectral_message.get())
        a.include_unreviewed.set(True); a.invalidate_spectrum()
        a.calculate_spectrum(); self.wait_ready()
        self.assertTrue(a.spectral_result['exploratory'])
        self.assertEqual(a.spectral_result['sessions'][0]['counts']['used'],2)
        a.spectral_db.set(False); a.draw_spectrum()
        self.assertEqual(a.spectral_figure.axes[0].get_ylabel(),'PSD (µV²/Hz)')
        a.channel_box.current(1); a.select_channel(); self.wait_ready()
        self.assertIsNone(a.spectral_result)
        a.spectral_mode.set('Comparar NS/SD'); a.calculate_spectrum()
        self.assertIn('estudio único',a.spectral_message.get())

    def test_save_discard_cancel_and_failed_save_on_leave(self):
        from unittest.mock import patch
        self.select()
        a = self.app
        a.mark_state.set('Aceptada'); a.apply_mark()
        with patch.object(a,'unsaved_choice',return_value='save'):
            with patch.object(a.store,'save',side_effect=OSError('Disco lleno')):
                a.patient.set('sub-02'); a.select_patient()
                self.assertEqual(a.patient.get(),'sub-01')
                self.assertTrue(a.store.dirty)
            a.patient.set('sub-02'); a.select_patient()
        self.assertEqual(a.patient.get(),'sub-02')
        a.patient.set('sub-01'); a.select_patient()
        a.study_box.current(0); a.select_study(); self.wait_ready()
        self.assertEqual(a.store.records[0]['state'],'Aceptada')
        a.channel_box.current(0); a.select_channel(); self.wait_ready()
        a.mark_state.set('Con artefactos'); a.apply_mark()
        with patch.object(a,'unsaved_choice',return_value='discard'):
            a.select_study()
        self.wait_ready()
        self.assertEqual(a.store.records[0]['state'],'Aceptada')

    def test_obsolete_spectral_result_after_parameter_change(self):
        from unittest.mock import patch
        from eeg_viewer.spectral import run_analysis
        self.select()
        a = self.app
        a.include_unreviewed.set(True)
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def delayed(*args):
            ready.set(); release.wait(5)
            return run_analysis(*args)
        with patch('eeg_viewer.spectral_ui.run_analysis',side_effect=delayed):
            a.calculate_spectrum()
            self.assertTrue(ready.wait(2))
            a.include_unreviewed.set(False); a.invalidate_spectrum()
            release.set(); self.wait_ready()
        self.assertIsNone(a.spectral_result)
        self.assertEqual(str(a.export_button.cget('state')), 'disabled')
