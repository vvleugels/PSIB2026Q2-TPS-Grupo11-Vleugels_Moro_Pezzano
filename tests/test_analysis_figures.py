"""Figure and hypothesis-table tests on synthetic tables (no dataset needed)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pandas as pd
except ModuleNotFoundError:   # pandas comes from requirements-analysis.txt
    pd = None


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Synthetic batch + statistics outputs in a temporary results folder."""
        from tps.analysis.regions import REGIONS, frontal_theta_beta
        from tps.analysis.stats import cluster_tests, montage_channels, run_all
        from tps.analysis.figures import standard_info
        from synthetic_features import make_features, make_psd, theta_beta_from
        cls.temp = tempfile.TemporaryDirectory()
        cls.results = Path(cls.temp.name)
        shift = {('theta', r): 0.15 for r in REGIONS} | {('alpha', 'occipital'): -0.2}
        features = make_features(12, shift, ('open', 'closed'), noise=0.1)
        # Drop a few channels of one recording, as bad channels do in the real data.
        features = features[~((features['sub'] == 'sub-01') & (features['condition'] == 'SD')
                              & features['channel'].isin(['Fp1', 'Fp2']))]
        features.to_csv(cls.results / 'features_long.csv', index=False)
        theta_beta_from(features).to_csv(cls.results / 'theta_beta.csv', index=False)
        make_psd(features).to_csv(cls.results / 'psd_channels.csv', index=False)
        run_all(cls.results, cls.results, clusters=False)
        clusters, channels = cluster_tests(features, standard_info(montage_channels(features)), 100)
        clusters.to_csv(cls.results / 'cluster_results.csv', index=False)
        cls.clusters = clusters
        from tps.analysis.figures import make_all
        cls.out = cls.results / 'figures'
        cls.made = make_all(cls.results, cls.out)   # built once: many subplots make this slow

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_make_all_saves_every_figure_and_caption(self):
        out, made = self.out, self.made
        expected = [f'{kind}_{eyes}' for kind in ('topomap', 'psd_regions', 'paired', 'topomap_relative',
                                                  'paired_relative')
                    for eyes in ('open', 'closed')] + ['theta_beta']
        for name in expected:
            self.assertIn(name, made)
            path = out / f'{name}.png'
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)   # not an empty canvas
        for name in ('hypothesis_summary.csv', 'hypothesis_summary.md', 'captions.md'):
            self.assertTrue((out / name).is_file(), name)
        captions = (out / 'captions.md').read_text(encoding='utf-8')
        for name in expected:
            self.assertIn(f'## {name}', captions)
        self.assertNotIn('nan', captions.lower().replace('nan ', ''))

    def test_report_collects_tables_hypotheses_and_figures(self):
        """results/report.md replaces the former notebooks: one document with every result."""
        report = (self.results / 'report.md').read_text(encoding='utf-8')
        for section in ('## Region mapping', '## Participants', '## Epochs used per session',
                        '## Band power by region', '## Relative power', '## Absolute vs relative power',
                        '## Sensitivity', '## Frontal theta/beta ratio', '## Session order check',
                        '## Significant electrode clusters', '## Hypotheses', '## Figures'):
            self.assertIn(section, report)
        for name in self.made:
            if name != 'hypothesis_summary':
                self.assertIn(f'](figures/{name}.png)', report)   # every figure, linked relative to results/
        self.assertNotIn('nan |', report.lower())

    def test_topomap_marks_significant_cluster_channels(self):
        from tps.analysis.figures import significant_channels
        marked = significant_channels(self.clusters, 'open', 'theta')
        self.assertGreater(len(marked), 30)   # theta shift was simulated everywhere
        self.assertEqual(significant_channels(None, 'open', 'theta'), set())

    def test_hypothesis_summary_reflects_simulated_effects(self):
        from tps.analysis.hypotheses import HYPOTHESES, agreement
        table = pd.read_csv(self.out / 'hypothesis_summary.csv')
        self.assertEqual(set(table['hypothesis']), {h.id for h in HYPOTHESES})
        theta = table[(table['hypothesis'] == 'H1') & (table['band'] == 'theta') & (table['eyes'] == 'open')]
        self.assertTrue((theta['agreement'] == 'supported').all())
        # No delta shift simulated. EO only: EC has 12 subjects and 20 tests, where a chance
        # FDR hit can occur and would make the test depend on the random seed.
        delta = table[(table['hypothesis'] == 'H1') & (table['band'] == 'delta') & (table['eyes'] == 'open')]
        self.assertTrue((delta['agreement'] == 'not significant').all())
        alpha = table[(table['hypothesis'] == 'H2') & (table['region'] == 'occipital') & (table['eyes'] == 'open')]
        self.assertEqual(alpha['agreement'].iloc[0], 'supported')
        self.assertTrue(table[table['hypothesis'] == 'H4']['agreement'].str.startswith('exploratory').all())
        # Agreement logic.
        self.assertEqual(agreement('+', 0.2, 0.01), 'supported')
        self.assertEqual(agreement('+', -0.2, 0.01), 'opposite')
        self.assertEqual(agreement('-', -0.2, 0.2), 'not significant')
        self.assertEqual(agreement(None, 0.1, 0.01), 'exploratory: significant')

    def test_references_are_todo_not_invented(self):
        from tps.analysis.hypotheses import HYPOTHESES
        for h in HYPOTHESES:
            for ref in h.references:
                self.assertEqual((ref.authors, ref.year, ref.doi), ('TODO', 'TODO', 'TODO'))

    def test_missing_psd_skips_only_that_figure(self):
        from tps.analysis.figures import make_all
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ('features_long.csv', 'features_region.csv', 'theta_beta.csv',
                         'stats_region_log_power.csv', 'stats_frontal_theta_beta.csv'):
                (folder / name).write_bytes((self.results / name).read_bytes())
            made = make_all(folder)
            self.assertNotIn('psd_regions_open', made)
            self.assertIn('topomap_open', made)
            (folder / 'psd_channels.csv').write_bytes((self.results / 'psd_channels.csv').read_bytes())
            made = make_all(folder)
            self.assertIn('psd_regions_open', made)
            curves = [line for line in made['psd_regions_open'].axes[0].lines if line.get_label().startswith(('NS (', 'SD ('))]
            self.assertEqual(len(curves), 2)   # no cluster file: maps without markers


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(pd is None, 'pandas not installed')
class EmptyClusterTests(unittest.TestCase):
    def test_one_pair_exports_headers_and_legacy_empty_csv_still_renders(self):
        from synthetic_features import make_features, theta_beta_from, make_psd
        from tps.analysis.stats import run_all
        from tps.analysis.figures import make_all, load_inputs
        from tps.analysis.report import read
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            features = make_features(1, {}, ('open',))
            features.to_csv(root / 'features_long.csv', index=False)
            make_psd(features).to_csv(root / 'psd_channels.csv', index=False)
            theta_beta_from(features).to_csv(root / 'theta_beta.csv', index=False)
            from unittest.mock import patch
            from tps.analysis.figures import standard_info
            info = standard_info(sorted(features['channel'].unique()))
            with patch('tps.analysis.stats.montage_info', return_value=info):
                run_all(root, root, n_permutations=100)
            clusters = pd.read_csv(root / 'cluster_results.csv')
            channels = pd.read_csv(root / 'stats_cluster_channels.csv')
            descriptive = pd.read_csv(root / 'stats_region_log_power.csv')
            one = descriptive[descriptive['n'] == 1]
            self.assertFalse(one.empty)
            self.assertTrue(one['mean_diff'].notna().all())
            self.assertTrue(one['p_wilcoxon'].isna().all())
            self.assertTrue(clusters.empty)
            self.assertTrue(channels.empty)
            self.assertIn('significant', clusters.columns)
            self.assertIn('cluster_p', channels.columns)
            # Files already produced by older versions must also be usable.
            (root / 'cluster_results.csv').write_text('\n')
            self.assertIsNone(load_inputs(root)['clusters'])
            self.assertIsNone(read(root, 'cluster_results.csv'))
            made = make_all(root)
            self.assertIn('topomap_open', made)
            self.assertIn('psd_regions_open', made)
            curves = [line for line in made['psd_regions_open'].axes[0].lines if line.get_label().startswith(('NS (', 'SD ('))]
            self.assertEqual(len(curves), 2)
            self.assertTrue((root / 'report.md').exists())
