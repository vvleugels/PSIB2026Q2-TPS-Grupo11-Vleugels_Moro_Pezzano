"""«Resultados del grupo»: data layer, pipeline runner and GUI tab, on synthetic results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tps.gui.runner import Options, Runner, command, missing_dependencies, parse_progress
from tps.gui.results_data import Results, filter_rows, fmt, sort_rows

try:
    import pandas as pd
except ModuleNotFoundError:   # pandas comes from requirements-analysis.txt
    pd = None

PROJECT = Path(__file__).resolve().parent.parent


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_results(folder: Path) -> None:
    """A small but complete results/ folder: batch tables, statistics, figures and captions."""
    from tps.analysis.figures import make_all
    from tps.analysis.stats import run_all
    from synthetic_features import make_features, make_psd, theta_beta_from
    features = make_features(6, {('theta', 'frontal'): 0.2}, ('open',), noise=0.1)
    features.to_csv(folder / 'features_long.csv', index=False)
    theta_beta_from(features).to_csv(folder / 'theta_beta.csv', index=False)
    make_psd(features).to_csv(folder / 'psd_channels.csv', index=False)
    recordings = features.drop_duplicates(['sub', 'ses', 'eyes'])
    write_csv(folder / 'recordings_summary.csv', [
        dict(sub=r['sub'], ses=r['ses'], condition=r['condition'], eyes=r['eyes'],
             file=f"{r['sub']}_{r['ses']}_task-eyesopen_eeg.set", excluded='False', n_channels=61,
             n_bad_channels=1 if r['sub'] == 'sub-01' else 0, n_epochs_complete=60,
             n_epochs_used=r['n_epochs_used'], n_epochs_rejected=r['n_epochs_rejected'])
        for _, r in recordings.iterrows()] + [
        dict(sub='sub-07', ses='ses-2', condition='SD', eyes='open', file='sub-07_ses-2_task-eyesopen_eeg.set',
             excluded='True', n_channels='', n_bad_channels='', n_epochs_complete='', n_epochs_used='',
             n_epochs_rejected='')])
    write_csv(folder / 'exclusions.csv', [
        dict(sub='sub-01', ses='ses-1', condition='NS', eyes='open', level='channel', channel='TP9', epoch='',
             reason='flagged in 31/60 complete epochs (> 30%); treated as a bad channel'),
        dict(sub='sub-01', ses='ses-1', condition='NS', eyes='open', level='epoch', channel='', epoch='3',
             reason='10-15 s: Fz high p-p > 150 µV (210.0 µV)'),
        dict(sub='sub-01', ses='ses-1', condition='NS', eyes='open', level='epoch', channel='', epoch='61',
             reason='incomplete final epoch (only complete 5 s epochs are used)'),
        dict(sub='sub-07', ses='ses-2', condition='SD', eyes='open', level='recording', channel='', epoch='',
             reason='cannot read recording: Archivo EEG incompleto'),
    ])
    run_all(folder, folder, clusters=False)
    make_all(folder)


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class ResultsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        build_results(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_status_tables_and_figures(self):
        results = Results(self.folder)
        status = results.status()
        self.assertTrue(all(s.done for s in status.values()))
        self.assertIn('1 excluidos', status['batch'].detail)
        self.assertIn('stats_region_log_power', results.available_tables())
        self.assertIn('exclusions', results.available_tables())
        names = [p.stem for p in results.figure_list()]
        # Eyes open only here (no EC data): absolute, then relative topomaps, then PSDs.
        self.assertEqual(names[:3], ['topomap_open', 'topomap_relative_open', 'psd_regions_open'])
        captions = results.captions()
        self.assertIn('topomap_open', captions)
        self.assertFalse(captions['topomap_open'].startswith('File:'))
        self.assertEqual(len(results.hypothesis_rows()) > 0, True)

    def test_missing_folder_is_empty_not_an_error(self):
        results = Results(self.folder / 'nothing')
        self.assertFalse(any(s.done for s in results.status().values()))
        self.assertEqual(results.figure_list(), [])
        self.assertEqual(results.captions(), {})
        self.assertFalse(results.study_summary('sub-01', 'ses-1', 'eyesopen').found)

    def test_study_summary(self):
        results = Results(self.folder)
        summary = results.study_summary('sub-01', 'ses-1', 'eyesopen')
        self.assertTrue(summary.found)
        self.assertFalse(summary.excluded)
        self.assertEqual(list(summary.bad_channels), ['TP9'])
        self.assertEqual(list(summary.rejected_epochs), [2])   # epoch 3 -> index 2; partial ignored
        self.assertEqual(set(summary.powers['Cz']), {'delta', 'theta', 'alpha', 'beta'})
        self.assertGreater(summary.theta_beta['Cz'], 0)
        excluded = results.study_summary('sub-07', 'ses-2', 'eyesopen')
        self.assertTrue(excluded.excluded)
        self.assertIn('incompleto', excluded.reason)
        self.assertFalse(results.study_summary('sub-01', 'ses-1', 'eyesclosed').found)

    def test_stale_outputs(self):
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            for name in ('features_long.csv', 'recordings_summary.csv'):
                (other / name).write_bytes((self.folder / name).read_bytes())
            (other / 'config_used.json').write_text(json.dumps({'config_hash': 'old'}))
            self.assertTrue(Results(other).status()['batch'].stale)
            # The batch's combined cache key differs from the settings fingerprint (it also
            # includes the ICA mode and folders); only settings_hash decides staleness.
            from tps.pipeline.batch import config_hash
            (other / 'config_used.json').write_text(json.dumps(
                {'config_hash': 'run-specific', 'settings_hash': config_hash()}))
            self.assertFalse(Results(other).status()['batch'].stale)

    def test_filter_sort_and_format(self):
        rows = [dict(eyes='open', band='theta', p='0.2'), dict(eyes='closed', band='theta', p='0.01'),
                dict(eyes='open', band='alpha', p='')]
        self.assertEqual(len(filter_rows(rows, eyes='open', band=None)), 2)
        self.assertEqual(len(filter_rows(rows, region='frontal')), 3)   # missing column ignored
        self.assertEqual([r['p'] for r in sort_rows(rows, 'p')], ['0.01', '0.2', ''])
        self.assertEqual(fmt('p_fdr_wilcoxon', '0.00001'), '< 0.001')
        self.assertEqual(fmt('n', '57.0'), '57')
        self.assertEqual(fmt('excluded', 'True'), 'Sí')
        self.assertEqual(fmt('sub', 'sub-01'), 'sub-01')


class PipelineTests(unittest.TestCase):
    def test_commands_and_progress(self):
        out = Path('results')
        batch = command('batch', Options(eyes='closed', jobs=3, force=True), out)
        self.assertEqual(batch[1:3], ['-m', 'tps.pipeline.batch'])
        self.assertIn('--force', batch)
        self.assertEqual(batch[batch.index('--eyes') + 1], 'closed')
        self.assertEqual(batch[batch.index('--jobs') + 1], '3')
        self.assertNotIn('--eyes', command('batch', Options(), out))
        stats = command('stats', Options(permutations=200), out)
        self.assertEqual(stats[stats.index('--permutations') + 1], '200')
        self.assertIn('tps.analysis.figures', command('figures', Options(), out))
        from tps.core.processing import FilterConfig
        # The explorer's filter goes to the batch explicitly.
        argv = command('batch', Options(filtering=FilterConfig(True, 1, 40, 50), ica_mode='confirmed'), out)
        self.assertEqual(argv[argv.index('--bandpass') + 1:argv.index('--bandpass') + 3], ['1', '40'])
        self.assertEqual(argv[argv.index('--notch') + 1], '50')
        self.assertEqual(argv[argv.index('--ica-mode') + 1], 'confirmed')
        self.assertIn('--no-bandpass', command('batch', Options(filtering=FilterConfig()), out))
        self.assertEqual(parse_progress('2026 INFO [25/100] sub-01 (12s) ...'), 25.0)
        self.assertIsNone(parse_progress('Session order: written'))
        self.assertEqual(missing_dependencies('batch'), [])

    @unittest.skipIf(pd is None, 'pandas not installed')
    def test_runner_runs_a_real_step(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            build_results(folder)
            (folder / 'figures' / 'captions.md').unlink()
            runner = Runner(PROJECT, folder)
            runner.start(['figures'], Options())
            events, deadline = [], time.monotonic() + 300
            while time.monotonic() < deadline:
                event = runner.events.get(timeout=300)
                events.append(event)
                if event[0] == 'finished':
                    break
            self.assertEqual(events[0], ('start', 'figures'))
            self.assertEqual(events[-1], ('finished', True, False))
            self.assertIn(('done', 'figures', 0), events)
            self.assertTrue((folder / 'figures' / 'captions.md').is_file())

    def test_cancel_stops_the_step_and_the_next_ones(self):
        from unittest import mock
        slow = [sys.executable, '-c', 'import time; print("[1/2] started", flush=True); time.sleep(60)']
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('tps.gui.runner.command', return_value=slow):
            runner = Runner(PROJECT, Path(folder))
            started = time.monotonic()
            runner.start(['batch', 'stats'], Options())
            events = []
            while True:
                event = runner.events.get(timeout=30)
                events.append(event)
                if event[0] == 'progress':
                    runner.cancel()   # cancel while the step is running
                if event[0] == 'finished':
                    break
            self.assertLess(time.monotonic() - started, 30)   # killed, not slept for 60 s
            self.assertEqual(events[-1], ('finished', False, True))
            self.assertNotIn(('start', 'stats'), events)   # the next step never starts
            self.assertIn(('progress', 'batch', 50.0), events)
            runner.thread.join(5)
            self.assertFalse(runner.running)

    def test_cancel_during_start_still_kills(self):
        from unittest import mock
        slow = [sys.executable, '-c', 'import time; time.sleep(60)']
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('tps.gui.runner.command', return_value=slow):
            runner = Runner(PROJECT, Path(folder))
            started = time.monotonic()
            runner.start(['batch'], Options())
            runner.cancel()   # before or while Popen runs
            while runner.events.get(timeout=30)[0] != 'finished':
                pass
            self.assertLess(time.monotonic() - started, 30)


@unittest.skipIf(pd is None, 'pandas not installed')
class ResultsTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.results_dir = Path(cls.temp.name) / 'results'
        cls.results_dir.mkdir()
        build_results(cls.results_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        import tkinter as tk
        from tps.gui.app import App
        from test_core_data import make_recording
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Se necesita una pantalla o Xvfb: {exc}')
        self.data = tempfile.TemporaryDirectory()
        self.addCleanup(self.data.cleanup)
        make_recording(self.data.name)
        self.app = App(self.root, Path(self.data.name), Path(self.data.name) / 'annotations', self.results_dir)
        self.addCleanup(self.cleanup_app)
        self.wait()

    def cleanup_app(self):
        if self.app.store:
            self.app.store.dirty = False
        self.app.close()
        self.app.executor.shutdown(wait=True)

    def wait(self):
        deadline = time.monotonic() + 20
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.root.update()

    def test_tab_shows_results(self):
        app = self.app
        self.assertEqual(app.view_tabs.tab(2, 'text'), 'Resultados del grupo')
        self.assertTrue(app.step_status['batch'].get().startswith('✔'))
        self.assertGreater(len(app.hypothesis_tree.get_children()), 0)
        self.assertGreater(app.figure_listbox.size(), 0)
        self.assertIsNotNone(app.figure_pil)
        self.assertGreater(len(app.table_tree.get_children()), 0)
        app.table_choice.set('Exclusiones del lote')
        app.show_table(reset_filters=True)
        self.assertEqual(len(app.table_tree.get_children()), 4)
        app.table_filters['sub'].set('sub-07')
        app.show_table()
        self.assertEqual(len(app.table_tree.get_children()), 1)
        app.show_table(sort='reason')
        self.assertIn('▲', app.table_tree.heading('reason', 'text'))
        app.view_tabs.select(2)
        self.root.update()
        self.assertFalse(app.info_card.winfo_manager())   # study info hidden on the group tab
        self.assertIn('Resultados completos', app.instruction.get())

    def test_batch_card_for_open_study(self):
        app = self.app
        app.patient.set('sub-01')
        app.select_patient()
        app.study_box.current(0)
        app.select_study()
        self.wait()
        self.assertIn('Incluido', app.batch_state.get())
        self.assertIn('TP9', app.batch_bad.get())
        app.channel_box.set('Cz')
        app.select_channel()
        self.wait()
        self.assertIn('delta', app.batch_channel.get())
        self.assertEqual(app.batch_rejection(2), '10-15 s: Fz high p-p > 150 µV (210.0 µV)')
        self.assertEqual(app.epoch_tree.item('2')['values'][2], 'Lote')
        app.show_study_exclusions()
        self.assertEqual(app.view_tabs.index('current'), 2)
        self.assertEqual(len(app.table_tree.get_children()), 3)   # sub-01 ses-1 open rows only


if __name__ == '__main__':
    unittest.main()
