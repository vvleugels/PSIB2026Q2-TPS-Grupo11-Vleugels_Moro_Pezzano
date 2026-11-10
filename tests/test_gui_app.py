from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest

import numpy as np

from tps.gui.app import App
from test_core_data import make_recording


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
        from tps.core.processing import FilterConfig
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
        self.assertTrue(a.store.dirty)  # The note is debounced.
        self.assertTrue(a.flush_autosave())
        self.assertTrue(a.store.path.exists())
        self.assertEqual(a.save_label.get(), 'Guardado automáticamente')
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
            a.mark_state.set('Aceptada')
            self.assertTrue(a.store.dirty)
            self.assertEqual(a.save_label.get(), 'Error de guardado · Cambios en memoria')
            with patch.object(a, 'unsaved_choice', return_value='cancel'):
                a.patient.set('sub-02'); a.select_patient()
                self.assertEqual(a.patient.get(), 'sub-01')
                self.assertIsNotNone(a.recording)
                a.close()
                self.assertFalse(a.closed)
        self.assertTrue(a.save_annotations())

    def test_obsolete_filter_completion_cannot_change_next_study(self):
        from tps.core.processing import FilterConfig
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

    def test_autosave_allows_spectral_analysis_without_manual_save(self):
        self.select()
        a = self.app
        a.mark_state.set('Aceptada')
        self.assertFalse(a.store.dirty)
        self.assertTrue(a.store.path.exists())
        a.note.set('nota pendiente')
        a.calculate_spectrum(); self.wait_ready()
        self.assertFalse(a.store.dirty)
        self.assertIsNotNone(a.spectral_result, a.spectral_message.get())
        self.assertEqual(a.spectral_result['sessions'][0]['counts']['used'], 1)
        self.assertEqual(len(a.band_table.get_children()),4)
        self.assertEqual(str(a.export_button.cget('state')), 'normal')
        a.select_epoch(1); a.mark_state.set('Aceptada')
        self.assertIsNone(a.spectral_result)
        self.assertEqual(str(a.export_button.cget('state')), 'disabled')

    def test_exploratory_spectrum_and_channel_invalidation(self):
        self.select()
        a = self.app
        a.calculate_spectrum(); self.wait_ready()
        self.assertIsNone(a.spectral_result)
        self.assertIn('no hay épocas', a.spectral_message.get())
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
        with patch.object(a.store, 'save', side_effect=OSError('Disco lleno')):
            a.mark_state.set('Aceptada')
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
        with patch.object(a.store, 'save', side_effect=OSError('Disco lleno')):
            a.mark_state.set('Con artefactos')
            with patch.object(a,'unsaved_choice',return_value='discard'):
                a.select_study()
        self.wait_ready()
        self.assertEqual(a.store.records[0]['state'],'Aceptada')

    def test_obsolete_spectral_result_after_parameter_change(self):
        from unittest.mock import patch
        from tps.core.spectral import run_analysis
        self.select()
        a = self.app
        a.include_unreviewed.set(True)
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def delayed(*args):
            ready.set(); release.wait(5)
            return run_analysis(*args)
        with patch('tps.gui.spectral.run_analysis',side_effect=delayed):
            a.calculate_spectrum()
            self.assertTrue(ready.wait(2))
            a.include_unreviewed.set(False); a.invalidate_spectrum()
            release.set(); self.wait_ready()
        self.assertIsNone(a.spectral_result)
        self.assertEqual(str(a.export_button.cget('state')), 'disabled')

    def test_note_autosaves_on_pause_and_navigation_without_marking_other_epoch(self):
        import json
        self.select()
        a = self.app
        self.assertFalse(a.store.path.exists())
        a.mark_state.set('Con artefactos')
        a.reason.set('Ocular')
        a.note.set('parpadeo')
        self.assertTrue(a.store.dirty)
        deadline = time.monotonic() + 2
        while a.store.dirty and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(a.store.dirty)
        self.assertEqual(json.loads(a.store.path.read_text())['epochs'][0]['note'], 'parpadeo')
        a.note.set('nota antes de navegar')
        a.select_epoch(1)
        self.assertFalse(a.store.dirty)
        self.assertEqual(a.store.records[0]['note'], 'nota antes de navegar')
        self.assertEqual(a.store.records[1]['state'], 'Sin revisar')
        self.assertIsNone(a.store.records[1]['reviewed_at'])
        before = a.store.path.read_bytes()
        a.select_epoch(0)
        self.assertEqual(a.store.path.read_bytes(), before)
        self.assertEqual(a.note.get(), 'nota antes de navegar')

    def test_pending_note_saved_on_close_and_failure_blocks_spectrum(self):
        import json
        from unittest.mock import patch
        self.select()
        a = self.app
        a.mark_state.set('Aceptada')
        a.note.set('última nota')
        with patch.object(a.store, 'save', side_effect=OSError('Disco lleno')):
            a.calculate_spectrum()
            self.assertIsNone(a.spectral_result)
            self.assertIn('No se pudieron guardar', a.spectral_message.get())
            a.select_epoch(1)
            self.assertEqual(a.selected_epoch, 0)
        path = a.store.path
        a.close()
        self.assertTrue(a.closed)
        self.assertEqual(json.loads(path.read_text())['epochs'][0]['note'], 'última nota')

    def test_screening_navigation_scope_autoaccepts_and_preserves_manual_marks(self):
        self.select()
        a = self.app
        a.mark_state.set('Aceptada')
        manual = dict(a.store.records[0])
        a.screening_high.set('75')
        a.start_screening(); self.wait_ready()
        self.assertEqual(set(a.screening_result['findings']), {0, 1})
        self.assertEqual(a.epoch_tree.item('0')['values'][-1], 'Alta')
        self.assertIn('Fp1', a.screening_detail.get())
        a.next_suspect(1); self.wait_ready()
        self.assertEqual(a.selected_epoch, 1)
        a.next_suspect(1); self.wait_ready()
        self.assertEqual(a.selected_epoch, 0)  # Wrap around.
        self.assertEqual(a.store.records[0], manual)
        self.assertEqual(a.store.records[1]['state'], 'Sin revisar')
        self.assertEqual(a.store.records[2]['state'], 'Aceptada')
        self.assertIn('screening', a.store.records[2])
        self.assertFalse(a.store.dirty)
        a.screening_all.set(False)
        self.assertIsNone(a.screening_result)
        self.assertEqual(a.epoch_tree.item('0')['values'][-1], '')
        a.start_screening(); self.wait_ready()
        self.assertEqual(a.screening_result['channels'], ('Cz',))
        self.assertEqual(a.screening_result['findings'], {})
        self.assertEqual(a.store.records[0], manual)
        self.assertEqual(a.store.records[1]['state'], 'Aceptada')
        a.select_epoch(1)
        self.assertIn('Aceptación automática', a.context_label.get())
        a.calculate_spectrum(); self.wait_ready()
        self.assertEqual(a.spectral_result['sessions'][0]['counts']['used'], 2)
        from tps.core.annotations import AnnotationStore
        saved = AnnotationStore(a.annotations_dir, a.recording, a.data_dir)
        self.assertFalse(saved.blocked)
        self.assertEqual(saved.records, a.store.records)

    def test_obsolete_screening_result_is_ignored_on_parameter_or_patient_change(self):
        from unittest.mock import patch
        from tps.core.screening import scan_recording
        self.select()
        a = self.app
        for change in ('threshold', 'filter', 'patient'):
            ready, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(*args):
                ready.set(); release.wait(5)
                return scan_recording(*args)
            with patch('tps.gui.screening.scan_recording', side_effect=delayed):
                a.start_screening()
                self.assertTrue(ready.wait(2))
                if change == 'threshold':
                    a.screening_high.set('200')
                elif change == 'filter':
                    from tps.core.processing import FilterConfig
                    a.apply_config(FilterConfig(True, 2, 30))
                else:
                    a.patient.set('sub-02'); a.select_patient()
                release.set()
                # Wait for the obsolete worker as well as UI processing.
                a.executor.submit(lambda: None).result(timeout=5)
                self.wait_ready()
            self.assertIsNone(a.screening_result)
            self.assertFalse(a.screening_pending)
        self.assertEqual(a.patient.get(), 'sub-02')

    def test_space_bar_restores_the_zoom(self):
        from types import SimpleNamespace
        self.select()
        a = self.app
        home = a.ax.get_xlim(), a.ax.get_ylim()
        a.ax.set_xlim(2, 4); a.ax.set_ylim(-10, 10)
        self.assertEqual(a.space_restore(SimpleNamespace(widget=a.root)), 'break')
        self.assertEqual((a.ax.get_xlim(), a.ax.get_ylim()), home)
        a.ax.set_xlim(2, 4)
        self.assertIsNone(a.space_restore(SimpleNamespace(widget=a.note_entry)))   # typing a note
        self.assertEqual(a.ax.get_xlim(), (2, 4))

    def test_filter_applies_in_every_display_mode(self):
        """Regression: in «Todos apilados» the filter failed (channel limit of 8)."""
        from tps.core.processing import FilterConfig
        self.select()
        a = self.app
        for mode in ('Un electrodo', 'Varios (hasta 8)', 'Todos apilados'):
            a.display_mode.set(mode); a.display_mode_changed(); self.wait_ready()
            a.apply_config(FilterConfig()); self.wait_ready()
            self.assertFalse(a.config.active, mode)
            a.apply_config(FilterConfig(True, 2, 30)); self.wait_ready()
            self.assertTrue(a.config.active, mode)
            self.assertIn('Filtrado activo', a.filter_badge.cget('text'))
            self.assertNotIn('No se aplicó', a.status.get())
        a.display_mode.set('Un electrodo'); a.display_mode_changed(); self.wait_ready()

    def test_screening_uses_active_filter_and_invalidates_only_after_successful_change(self):
        from tps.core.processing import FilterConfig
        self.select()
        a = self.app
        self.assertIn('Filtrado desactivado', a.screening_signal.get())
        a.screening_all.set(False)
        filtering = FilterConfig(True, 2, 30)
        a.apply_config(filtering); self.wait_ready()
        a.start_screening(); self.wait_ready()
        result = a.screening_result
        self.assertEqual(result['filter'], filtering)
        self.assertIn(filtering.label, a.screening_summary.get())
        self.assertIn(filtering.label, a.screening_detail.get())
        a.apply_config(FilterConfig(True)); self.wait_ready()  # Too short: keep prior filter/results.
        self.assertEqual(a.config, filtering)
        self.assertIs(a.screening_result, result)
        a.apply_config(FilterConfig(True, 3, 25)); self.wait_ready()
        self.assertIsNone(a.screening_result)
        self.assertEqual(a.screening_signal.get(), a.config.label)
        self.assertEqual(a.epoch_tree.item('0')['values'][-1], '')
        self.assertFalse(hasattr(a, 'screening_use_filter'))   # always the active filter

    def test_screening_autosave_failure_keeps_batch_in_memory_and_allows_retry(self):
        import json
        from unittest.mock import patch
        self.select()
        a = self.app
        a.store.save()
        before = a.store.path.read_bytes()
        with patch.object(a.store, 'save', side_effect=OSError('Disco lleno')):
            a.start_screening(); self.wait_ready()
            self.assertEqual([r['state'] for r in a.store.records], ['Aceptada'] * 3)
            self.assertTrue(a.store.dirty)
            self.assertEqual(a.store.path.read_bytes(), before)
            self.assertIn('Guardado pendiente', a.screening_summary.get())
            a.calculate_spectrum()
            self.assertIsNone(a.spectral_result)
        self.assertTrue(a.save_annotations())
        self.assertFalse(a.store.dirty)
        saved = json.loads(a.store.path.read_text())
        self.assertTrue(all(r['screening']['high_uv'] == 150 for r in saved['epochs']))
        # A protected annotation file is not modified by a later scan.
        a.store.blocked = 'Anotaciones protegidas'
        before = a.store.path.read_bytes()
        a.start_screening(); self.wait_ready()
        self.assertIn('No se pudieron actualizar', a.screening_summary.get())
        self.assertEqual(a.store.path.read_bytes(), before)
