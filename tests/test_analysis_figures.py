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
        from analysis.regions import REGIONS, frontal_theta_beta
        from analysis.stats import cluster_tests, montage_channels, run_all
        from analysis.figures import standard_info
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
        from analysis.figures import make_all
        cls.out = cls.results / 'figures'
        cls.made = make_all(cls.results, cls.out)   # built once: many subplots make this slow

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_make_all_saves_every_figure_and_caption(self):
        out, made = self.out, self.made
        expected = [f'{kind}_{eyes}' for kind in ('topomap', 'psd_regions', 'paired')
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

    def test_topomap_marks_significant_cluster_channels(self):
        from analysis.figures import significant_channels
        marked = significant_channels(self.clusters, 'open', 'theta')
        self.assertGreater(len(marked), 30)   # theta shift was simulated everywhere
        self.assertEqual(significant_channels(None, 'open', 'theta'), set())

    def test_hypothesis_summary_reflects_simulated_effects(self):
        from analysis.hypotheses import HYPOTHESES, agreement
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
        from analysis.hypotheses import HYPOTHESES
        for h in HYPOTHESES:
            for ref in h.references:
                self.assertEqual((ref.authors, ref.year, ref.doi), ('TODO', 'TODO', 'TODO'))

    def test_missing_psd_skips_only_that_figure(self):
        from analysis.figures import make_all
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ('features_long.csv', 'features_region.csv', 'theta_beta.csv',
                         'stats_region_log_power.csv', 'stats_frontal_theta_beta.csv'):
                (folder / name).write_bytes((self.results / name).read_bytes())
            made = make_all(folder)
            self.assertNotIn('psd_regions_open', made)
            self.assertIn('topomap_open', made)   # no cluster file: maps without markers


if __name__ == '__main__':
    unittest.main()
