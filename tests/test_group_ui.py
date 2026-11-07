import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk

from eeg_viewer.group_ui import GroupUI, group_view_files, load_group_view_data


def _screen_available():
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        return True
    except tk.TclError:
        return False


class GroupUIDataTests(unittest.TestCase):
    def test_paths_select_absolute_and_relative_assets(self):
        relative = group_view_files('analysis', 'Ojos abiertos', 'Alfa', 'Pares', 'Relativa')
        self.assertEqual(relative['figure'], Path('analysis/pairs_relative_open_alpha.png'))
        self.assertEqual(relative['stats'], Path('analysis/stats_relative_vs_absolute.csv'))
        absolute = group_view_files('analysis', 'Ojos cerrados', 'Theta', 'Topomapa', 'Absoluta')
        self.assertEqual(absolute['figure'], Path('analysis/topomap_closed_theta.png'))
        self.assertEqual(absolute['stats'], Path('analysis/stats_table.csv'))

    def test_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_group_view_data(temp_dir, 'Ojos abiertos', 'Delta', 'Pares', 'Absoluta')
        self.assertIn('stats_table.csv', ' '.join(result['missing']))
        self.assertIn('group_summary.csv', ' '.join(result['missing']))
        self.assertGreaterEqual(len(result['missing']), 4)


@unittest.skipUnless(_screen_available(), 'Requiere pantalla Tk; se omite en modo headless.')
class GroupUIScreenTests(unittest.TestCase):
    def test_tab_is_registered_and_missing_assets_show_a_message(self):
        root = tk.Tk()
        self.addCleanup(root.destroy)
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            ui = GroupUI(notebook, temp_dir)
            self.assertEqual(notebook.tab(ui.tab, 'text'), 'Resultados de grupo')
            self.assertIn('Faltan archivos', ui.message.get())


if __name__ == '__main__':
    unittest.main()
