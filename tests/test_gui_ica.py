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

    def test_fit_preserves_annex_symlink_directory(self):
        # EEGLAB headers refer to a sibling .fdt; annex stores targets separately.
        # Windows needs Developer Mode or admin rights for symlinks (WinError 1314): check
        # first, before moving the fixture files. The Windows annex form (text pointers) is
        # covered by the placeholder tests in test_core_data.
        probe = Path(self.temp.name) / 'symlink-probe'
        try:
            probe.symlink_to(Path(self.temp.name), target_is_directory=True)
        except OSError as exc:
            self.skipTest(f'No se pueden crear symlinks en este sistema: {exc}')
        probe.unlink()
        a = self.app
        path = a.recording.study.path
        for extension in ('.set', '.fdt'):
            link = path.with_suffix(extension)
            target = self.data / '.git' / 'annex' / 'objects' / extension[1:] / ('object' + extension)
            target.parent.mkdir(parents=True, exist_ok=True)
            link.rename(target)
            link.symlink_to(target)
        a.select_study(); self.wait()
        a.open_ica(); a.start_ica()
        request = json.loads((Path(a.ica_temp.name) / 'request.json').read_text())
        self.wait(fit=True)
        self.assertIsNotNone(a.ica_model, a.ica_message.get())
        self.assertEqual(Path(request['path']), path.absolute())
        self.assertNotEqual(Path(request['path']), path.resolve())

    def test_component_map_grid_selection_and_inspection(self):
        from types import SimpleNamespace
        import numpy as np
        a = self.app
        a.open_ica(); a.start_ica(); self.wait(fit=True)
        original = np.array([ch['loc'].copy() for ch in a.ica_model.info['chs']])
        a.open_ica_maps(); self.root.update()
        grid = a.ica_maps
        self.assertEqual(len(grid.axes), a.ica_model.n_components_)
        ax = next(iter(grid.axes))
        grid.click(SimpleNamespace(inaxes=ax, button=1)); self.root.update()
        self.assertEqual(a.exclusions_text.get(), '1')
        self.assertIn('EXCLUIR', ax.get_title())
        a.exclusions_text.set('2'); self.root.update()
        self.assertIn('Conservar', ax.get_title())
        grid.click(SimpleNamespace(inaxes=ax, button=3)); self.wait()
        self.assertEqual(a.component_number.get(), '1')
        self.assertFalse(grid.winfo_exists())
        self.assertIsNone(a.recording.ica_context)
        np.testing.assert_array_equal(original, np.array([ch['loc'] for ch in a.ica_model.info['chs']]))
        a.open_ica_maps(); self.root.update()
        a.cancel_ica(close=True); self.root.update()
