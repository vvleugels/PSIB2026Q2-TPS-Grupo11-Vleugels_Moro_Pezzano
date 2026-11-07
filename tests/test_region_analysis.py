import unittest

from eeg_viewer.region_analysis import (
    aggregate_regions,
    calculate_stats,
    region_for_channel,
)


class RegionAnalysisTests(unittest.TestCase):
    def test_region_mapping_excludes_polar_frontal_leads(self):
        self.assertIsNone(region_for_channel('Fp1'))
        self.assertIsNone(region_for_channel('AF8'))
        self.assertEqual(region_for_channel('F7'), 'Frontal')
        self.assertEqual(region_for_channel('FC2'), 'Central')
        self.assertEqual(region_for_channel('Cpz'), 'Parietal')
        self.assertEqual(region_for_channel('PO8'), 'Occipital')
        self.assertEqual(region_for_channel('TP9'), 'Temporal')

    def test_regional_summary_averages_log10_channel_powers(self):
        common = dict(participant='sub-01', session='ses-1',
                      session_condition='Sueño normal', eyes='Ojos abiertos',
                      SessionOrder='NS->SD', band='Alfa')
        rows = [
            {**common, 'channel': 'C1', 'absolute_uv2': '10'},
            {**common, 'channel': 'C2', 'absolute_uv2': '100'},
            {**common, 'channel': 'Fp1', 'absolute_uv2': '1000'},
        ]
        result = aggregate_regions(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['region'], 'Central')
        self.assertAlmostEqual(result[0]['log10_power_mean'], 1.5)
        self.assertEqual(result[0]['n_channels'], 2)

    def test_paired_stats_and_session_order_check(self):
        rows = []
        for participant, order, ns_value, sd_value in (
                ('sub-01', 'NS->SD', 1.0, 2.0),
                ('sub-02', 'NS->SD', 1.0, 3.0),
                ('sub-03', 'SD->NS', 1.0, 0.0),
                ('sub-04', 'SD->NS', 1.0, -1.0)):
            for condition, value, session in (
                    ('NS', ns_value, 'ses-1'), ('SD', sd_value, 'ses-2')):
                rows.append(dict(
                    participant=participant, session=session, condition=condition,
                    eyes='Ojos abiertos', SessionOrder=order, region='Central',
                    band='Alfa', log10_power_mean=value,
                ))
        stats = calculate_stats(rows)
        row = next(item for item in stats if item['eyes'] == 'Ojos abiertos'
                   and item['region'] == 'Central' and item['band'] == 'Alfa')
        self.assertEqual(row['n_pairs'], 4)
        self.assertAlmostEqual(row['median_difference_log10_sd_minus_ns'], 0.0)
        self.assertEqual(row['rank_biserial_effect'], 0.0)
        self.assertEqual(row['n_order_ns_to_sd'], 2)
        self.assertEqual(row['n_order_sd_to_ns'], 2)
        self.assertTrue(0 <= row['order_mannwhitney_p_value'] <= 1)
        self.assertTrue(0 <= row['fdr_q_value'] <= 1)


if __name__ == '__main__':
    unittest.main()
