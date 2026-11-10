"""Statistics tests on synthetic paired data (no dataset needed)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pandas as pd
except ModuleNotFoundError:   # pandas comes from requirements-analysis.txt, not the GUI's
    pd = None

PROJECT = Path(__file__).resolve().parent.parent


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class PairedTestTests(unittest.TestCase):
    def test_change_in_db_and_percent(self):
        from tps.analysis.stats import paired_test
        ns = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
        row = paired_test(ns, ns + 0.1)   # +0.1 log10 = x 1.259 power
        self.assertAlmostEqual(row['change_db'], 1.0)
        self.assertAlmostEqual(row['change_pct'], (10 ** 0.1 - 1) * 100)
        self.assertAlmostEqual(row['mean_ns_db'], 10 * ns.mean())
        self.assertAlmostEqual(row['mean_sd_db'], 10 * (ns.mean() + 0.1))

    def test_p_is_the_paired_t_test_and_ci_is_converted(self):
        from scipy import stats
        from tps.analysis.stats import paired_test
        rng = np.random.default_rng(4)
        ns = rng.normal(1, 0.3, 20)
        sd = ns + rng.normal(0.08, 0.1, 20)
        row = paired_test(ns, sd)
        self.assertAlmostEqual(row['p'], stats.ttest_rel(sd, ns).pvalue)
        d = sd - ns
        half = stats.t.ppf(0.975, 19) * d.std(ddof=1) / np.sqrt(20)
        self.assertAlmostEqual(row['ci95_low_pct'], (10 ** (d.mean() - half) - 1) * 100)
        self.assertAlmostEqual(row['ci95_high_pct'], (10 ** (d.mean() + half) - 1) * 100)
        self.assertTrue(row['ci95_low_pct'] < row['change_pct'] < row['ci95_high_pct'])
        self.assertEqual(row['significant'], row['p'] < 0.05)

    def test_too_few_pairs_has_no_p(self):
        from tps.analysis.stats import paired_test
        row = paired_test(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        self.assertEqual(row['n'], 2)
        self.assertTrue(np.isnan(row['p']))
        self.assertFalse(row['significant'])
        self.assertAlmostEqual(row['change_db'], 5.0)   # descriptive values are still given


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class SummaryTableTests(unittest.TestCase):
    def tables(self, shift, n=30, eyes=('open',)):
        from tps.analysis.regions import frontal_theta_beta, region_features
        from synthetic_features import make_features, theta_beta_from
        features = make_features(n, shift, eyes)
        return region_features(features), frontal_theta_beta(theta_beta_from(features))

    def test_known_shift_is_detected_only_where_it_is(self):
        from tps.analysis.stats import summary_table
        regions, tbr = self.tables({('theta', 'frontal'): 0.15})
        eo = summary_table(regions, tbr).query("eyes == 'open'").set_index(['band', 'region'])
        self.assertEqual(len(eo), 21)   # 4 bands x 5 regions + frontal theta/beta
        hit = eo.loc[('theta', 'frontal')]
        self.assertLess(hit['p'], 0.05)
        self.assertTrue(hit['significant'])
        self.assertAlmostEqual(hit['change_db'], 1.5, delta=0.5)
        self.assertTrue(hit['ci95_low_pct'] < (10 ** 0.15 - 1) * 100 < hit['ci95_high_pct'])
        others = eo.drop(index=[('theta', 'frontal'), ('theta/beta', 'frontal')])
        self.assertLessEqual(int(others['significant'].sum()), 1)   # ~1 chance hit in 19 tests at most

    def test_no_shift_is_not_significant(self):
        from tps.analysis.stats import summary_table
        regions, tbr = self.tables({})
        eo = summary_table(regions, tbr).query("eyes == 'open'")
        self.assertLessEqual(int(eo['significant'].sum()), 1)
        self.assertGreater(eo.set_index('band').loc['theta/beta', 'p'], 0.05)

    def test_theta_beta_shift_is_detected(self):
        from tps.analysis.stats import summary_table
        regions, tbr = self.tables({('theta', 'frontal'): 0.2})
        row = summary_table(regions, tbr).query("eyes == 'open' and band == 'theta/beta'").iloc[0]
        self.assertLess(row['p'], 0.05)
        self.assertGreater(row['change_pct'], 0)

    def test_eyes_are_never_pooled_and_ec_uses_its_own_participants(self):
        from tps.analysis.regions import frontal_theta_beta, region_features
        from tps.analysis.stats import summary_table
        from synthetic_features import make_features, theta_beta_from
        eo = make_features(30, {}, ('open',), seed=1)
        ec = make_features(12, {('alpha', 'occipital'): 0.3}, ('closed',), seed=2)
        both = pd.concat([eo, ec])
        table = summary_table(region_features(both), frontal_theta_beta(theta_beta_from(both)))
        self.assertEqual(set(table[table['eyes'] == 'open']['n']), {30})
        self.assertEqual(set(table[table['eyes'] == 'closed']['n']), {12})
        ec_hit = table.set_index(['eyes', 'band', 'region']).loc[('closed', 'alpha', 'occipital')]
        self.assertLess(ec_hit['p'], 0.05)
        self.assertEqual(table[table['eyes'] == 'open']['analysis'].iloc[0], 'primary')

    def test_unpaired_subjects_are_dropped(self):
        from tps.analysis.regions import frontal_theta_beta, region_features
        from tps.analysis.stats import summary_table
        from synthetic_features import make_features, theta_beta_from
        features = make_features(10, {})
        features = features[~((features['sub'] == 'sub-01') & (features['condition'] == 'SD'))]
        table = summary_table(region_features(features), frontal_theta_beta(theta_beta_from(features)))
        self.assertEqual(set(table.query("eyes == 'open'")['n']), {9})

    def test_region_needs_half_of_its_channels(self):
        from tps.analysis.regions import region_features
        from synthetic_features import make_features
        features = make_features(2, {})
        frontal = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1']   # 9 of 16 dropped
        reduced = features[~((features['sub'] == 'sub-01') & features['channel'].isin(frontal))]
        regions = region_features(reduced)
        self.assertFalse(((regions['sub'] == 'sub-01') & (regions['region'] == 'frontal')).any())
        self.assertTrue(((regions['sub'] == 'sub-02') & (regions['region'] == 'frontal')).any())


class RegionMappingTests(unittest.TestCase):
    def test_every_channel_mapped_once(self):
        try:
            from tps.analysis.regions import REGIONS
        except ModuleNotFoundError:
            self.skipTest('pandas not installed')
        names = [c for chans in REGIONS.values() for c in chans]
        self.assertEqual(len(names), 61)
        self.assertEqual(len(set(names)), 61)

    def test_mapping_matches_bids_channels(self):
        try:
            from tps.analysis.regions import CHANNEL_REGION, channels_from_bids
        except ModuleNotFoundError:
            self.skipTest('pandas not installed')
        if not any((PROJECT / 'data').glob('sub-*/ses-*/eeg/*_channels.tsv')):
            self.skipTest('dataset not available')
        self.assertEqual(set(channels_from_bids(PROJECT / 'data')), set(CHANNEL_REGION))


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class EndToEndTests(unittest.TestCase):
    def test_run_all_writes_the_summary_and_removes_old_outputs(self):
        from tps.analysis.stats import SUMMARY_COLUMNS, run_all
        from synthetic_features import make_features, theta_beta_from
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            features = make_features(8, {('beta', 'central'): 0.2}, ('open', 'closed'))
            features.to_csv(folder / 'features_long.csv', index=False)
            theta_beta_from(features).to_csv(folder / 'theta_beta.csv', index=False)
            (folder / 'stats_session_order.csv').write_text('old\n')   # from an earlier version
            run_all(folder)
            for name in ('features_region', 'stats_summary', 'channel_regions'):
                self.assertTrue((folder / f'{name}.csv').is_file(), name)
            self.assertFalse((folder / 'stats_session_order.csv').exists())
            summary = pd.read_csv(folder / 'stats_summary.csv')
            self.assertEqual(list(summary.columns), SUMMARY_COLUMNS)
            self.assertEqual(len(summary), 42)   # 2 eyes x (20 band x region + theta/beta)
            regions = pd.read_csv(folder / 'channel_regions.csv')
            self.assertEqual(int(regions['n_channels'].sum()), 61)


if __name__ == '__main__':
    unittest.main()
