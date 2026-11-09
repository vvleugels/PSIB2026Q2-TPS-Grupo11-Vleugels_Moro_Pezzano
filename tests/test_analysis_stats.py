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
class FdrAndEffectSizeTests(unittest.TestCase):
    def test_bh_matches_known_values(self):
        from tps.analysis.stats import bh_fdr
        # Classic example: p * m / rank, made monotone from the largest p down.
        p = [0.01, 0.04, 0.03, 0.005, 0.5]
        expected = [0.025, 0.05, 0.05, 0.025, 0.5]
        np.testing.assert_allclose(bh_fdr(p), expected)

    def test_bh_matches_scipy_and_ignores_nan(self):
        from scipy.stats import false_discovery_control
        from tps.analysis.stats import bh_fdr
        p = np.random.default_rng(1).uniform(0, 1, 40)
        np.testing.assert_allclose(bh_fdr(p), false_discovery_control(p))
        with_nan = np.array([0.01, np.nan, 0.02])
        result = bh_fdr(with_nan)
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[[0, 2]], [0.02, 0.02])

    def test_fdr_controls_false_positives_under_the_null(self):
        from tps.analysis.stats import bh_fdr
        rng = np.random.default_rng(2)
        rejections = [np.sum(bh_fdr(rng.uniform(0, 1, 20)) < 0.05) for _ in range(500)]
        # Under the global null, BH controls the family-wise rate at about alpha.
        self.assertLess(np.mean(np.array(rejections) > 0), 0.08)

    def test_fdr_is_never_below_raw_p(self):
        from tps.analysis.stats import bh_fdr
        p = np.random.default_rng(3).uniform(0, 0.2, 25)
        self.assertTrue(np.all(bh_fdr(p) >= p - 1e-15))

    def test_effect_sizes(self):
        from tps.analysis.stats import paired_test, rank_biserial
        self.assertEqual(rank_biserial(np.array([1, 2, 3])), 1.0)
        self.assertEqual(rank_biserial(np.array([-1, -2, -3])), -1.0)
        ns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        row = paired_test(ns, ns + np.array([0.9, 1.1, 1.0, 0.8, 1.2]))
        self.assertAlmostEqual(row['mean_diff'], 1.0)
        self.assertAlmostEqual(row['cohens_dz'], 1.0 / np.std([0.9, 1.1, 1.0, 0.8, 1.2], ddof=1))
        self.assertLess(row['ci95_low'], 1.0)
        self.assertGreater(row['ci95_high'], 1.0)
        self.assertEqual(paired_test(ns[:2], ns[:2] + 1)['n'], 2)   # too few pairs: no p-values
        self.assertTrue(np.isnan(paired_test(ns[:2], ns[:2] + 1)['p_wilcoxon']))


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class PairedAnalysisTests(unittest.TestCase):
    def tables(self, shift, n=30, eyes=('open',)):
        from tps.analysis.regions import frontal_theta_beta, region_features
        from synthetic_features import make_features, theta_beta_from
        features = make_features(n, shift, eyes)
        return features, region_features(features), frontal_theta_beta(theta_beta_from(features))

    def test_known_shift_is_detected_only_where_it_is(self):
        from tps.analysis.stats import region_tests
        _, regions, _ = self.tables({('theta', 'frontal'): 0.15})
        table = region_tests(regions)
        eo = table[table['eyes'] == 'open'].set_index(['band', 'region'])
        hit = eo.loc[('theta', 'frontal')]
        self.assertLess(hit['p_fdr_wilcoxon'], 0.05)
        self.assertLess(hit['p_fdr_ttest'], 0.05)
        self.assertGreater(hit['mean_diff'], 0)
        self.assertGreater(hit['rank_biserial'], 0.5)
        self.assertAlmostEqual(hit['mean_diff'], 0.15, delta=0.05)
        self.assertTrue(hit['ci95_low'] < 0.15 < hit['ci95_high'])
        others = eo.drop(index=('theta', 'frontal'))
        self.assertTrue((others['p_fdr_wilcoxon'] >= 0.05).all())
        self.assertEqual(len(eo), 20)   # 4 bands x 5 regions in one FDR family

    def test_no_shift_is_not_significant(self):
        from tps.analysis.stats import region_tests, tbr_tests
        _, regions, tbr = self.tables({})
        table = region_tests(regions)
        self.assertTrue((table[table['eyes'] == 'open']['p_fdr_wilcoxon'] >= 0.05).all())
        self.assertTrue((table[table['eyes'] == 'open']['p_fdr_ttest'] >= 0.05).all())
        self.assertGreater(tbr_tests(tbr).set_index('eyes').loc['open', 'p_wilcoxon'], 0.05)

    def test_theta_beta_shift_is_detected(self):
        from tps.analysis.stats import tbr_tests
        _, _, tbr = self.tables({('theta', 'frontal'): 0.2})
        row = tbr_tests(tbr).set_index('eyes').loc['open']
        self.assertLess(row['p_wilcoxon'], 0.05)
        self.assertGreater(row['mean_diff'], 0)

    def test_eyes_are_never_pooled_and_ec_uses_its_own_participants(self):
        from tps.analysis.stats import region_tests
        from tps.analysis.regions import region_features
        from synthetic_features import make_features
        eo = make_features(30, {}, ('open',), seed=1)
        ec = make_features(12, {('alpha', 'occipital'): 0.3}, ('closed',), seed=2)
        table = region_tests(region_features(pd.concat([eo, ec])))
        self.assertEqual(set(table[table['eyes'] == 'open']['n']), {30})
        self.assertEqual(set(table[table['eyes'] == 'closed']['n']), {12})
        ec_hit = table.set_index(['eyes', 'band', 'region']).loc[('closed', 'alpha', 'occipital')]
        self.assertLess(ec_hit['p_fdr_wilcoxon'], 0.05)
        self.assertEqual(table[table['eyes'] == 'open']['analysis'].iloc[0], 'primary')

    def test_unpaired_subjects_are_dropped(self):
        from tps.analysis.stats import region_tests
        from tps.analysis.regions import region_features
        from synthetic_features import make_features
        features = make_features(10, {})
        features = features[~((features['sub'] == 'sub-01') & (features['condition'] == 'SD'))]
        self.assertEqual(set(region_tests(region_features(features))
                             .query("eyes == 'open'")['n']), {9})

    def test_region_needs_half_of_its_channels(self):
        from tps.analysis.regions import region_features
        from synthetic_features import make_features
        features = make_features(2, {})
        frontal = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1']   # 9 of 16 dropped
        reduced = features[~((features['sub'] == 'sub-01') & features['channel'].isin(frontal))]
        regions = region_features(reduced)
        self.assertFalse(((regions['sub'] == 'sub-01') & (regions['region'] == 'frontal')).any())
        self.assertTrue(((regions['sub'] == 'sub-02') & (regions['region'] == 'frontal')).any())

    def test_session_order_check(self):
        from tps.analysis.stats import session_order_tests
        _, regions, tbr = self.tables({('theta', 'frontal'): 0.15})
        order = pd.DataFrame({'sub': [f'sub-{i:02d}' for i in range(1, 31)],
                              'session_order': ['NS->SD', 'SD->NS'] * 15})
        table = session_order_tests(regions, tbr, order)
        eo = table[table['eyes'] == 'open']
        self.assertEqual(len(eo), 21)   # 20 band x region cells + frontal theta/beta
        self.assertEqual(set(eo['n_ns_first']), {15})
        self.assertTrue((eo['p_fdr_mannwhitney'] >= 0.05).all())   # no order effect simulated
        self.assertEqual(set(table[table['eyes'] == 'closed']['n_ns_first']), {0})   # no EC data
        self.assertIn('unavailable', session_order_tests(regions, tbr, None)['note'][0])


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class ComparisonTableTests(unittest.TestCase):
    """Tables ported from the former 20-participant group pipeline."""

    def test_relative_vs_absolute_side_by_side(self):
        from tps.analysis.regions import region_features
        from tps.analysis.stats import region_tests, relative_vs_absolute
        from synthetic_features import make_features
        regions = region_features(make_features(20, {('theta', 'frontal'): 0.2}))
        table = relative_vs_absolute(region_tests(regions, 'log10_power'), region_tests(regions, 'rel_power_pct'))
        self.assertEqual(len(table[table['eyes'] == 'open']), 20)
        for column in ('mean_diff_absolute', 'mean_diff_relative', 'p_fdr_wilcoxon_absolute',
                       'p_fdr_wilcoxon_relative', 'significant_absolute', 'significant_relative', 'same_direction'):
            self.assertIn(column, table)
        hit = table.set_index(['eyes', 'band', 'region']).loc[('open', 'theta', 'frontal')]
        self.assertTrue(hit['significant_absolute'])
        self.assertGreater(hit['mean_diff_relative'], 0)   # more theta also means a larger theta share

    def test_min_epochs_sensitivity_drops_short_recordings(self):
        from tps.analysis.stats import min_epochs_sensitivity
        from synthetic_features import make_features
        features = make_features(10, {})
        features.loc[features['sub'] == 'sub-01', 'n_epochs_used'] = 15   # below 20: left out
        table = min_epochs_sensitivity(features)
        self.assertEqual(set(table[table['eyes'] == 'open']['n']), {9})
        self.assertEqual(set(table['min_clean_epochs']), {20})


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class ClusterTests(unittest.TestCase):
    def info(self, channels):
        import mne
        info = mne.create_info(channels, 500.0, 'eeg')
        info.set_montage('colin27_1005')
        return info

    def test_cluster_finds_simulated_occipital_shift(self):
        from tps.analysis.regions import REGIONS
        from tps.analysis.stats import cluster_tests, montage_channels
        from synthetic_features import make_features
        features = make_features(20, {('alpha', 'occipital'): 0.2}, noise=0.1)
        info = self.info(montage_channels(features))
        clusters, channels = cluster_tests(features, info, n_permutations=200)
        alpha = clusters[(clusters['band'] == 'alpha') & clusters['significant']]
        self.assertEqual(len(alpha), 1)
        members = set(alpha['channels'].iloc[0].split(';'))
        self.assertTrue(set(REGIONS['occipital']) <= members | {'POz'})
        self.assertTrue(alpha['sign'].iloc[0].startswith('positive'))
        # No band without a simulated shift has a significant cluster.
        self.assertFalse(clusters[(clusters['band'] != 'alpha') & clusters['significant']].shape[0])
        self.assertEqual(set(channels.columns) >= {'channel', 't', 'cluster', 'n_imputed_zero'}, True)


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
    def test_run_all_writes_every_table(self):
        from tps.analysis.stats import run_all
        from synthetic_features import make_features, theta_beta_from
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            features = make_features(8, {('beta', 'central'): 0.2}, ('open', 'closed'))
            features.to_csv(folder / 'features_long.csv', index=False)
            theta_beta_from(features).to_csv(folder / 'theta_beta.csv', index=False)
            tables = run_all(folder, folder, clusters=False)
            for name in ('features_region', 'stats_region_log_power', 'stats_region_rel_power',
                         'stats_frontal_theta_beta', 'stats_session_order', 'stats_epochs',
                         'stats_participants', 'stats_relative_vs_absolute', 'stats_region_log_power_min20',
                         'channel_regions'):
                self.assertTrue((folder / f'{name}.csv').is_file(), name)
            regions = pd.read_csv(folder / 'channel_regions.csv')
            self.assertEqual(int(regions['n_channels'].sum()), 61)
            self.assertIn('unavailable', tables['stats_session_order']['note'][0])


if __name__ == '__main__':
    unittest.main()
