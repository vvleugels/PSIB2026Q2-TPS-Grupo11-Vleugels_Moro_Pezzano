"""Figure, hypothesis-table and report tests on synthetic tables (no dataset needed)."""
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


def write_inputs(folder: Path, features) -> None:
    from synthetic_features import make_psd, theta_beta_from
    features.to_csv(folder / 'features_long.csv', index=False)
    theta_beta_from(features).to_csv(folder / 'theta_beta.csv', index=False)
    make_psd(features).to_csv(folder / 'psd_channels.csv', index=False)


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Synthetic batch + statistics outputs in a temporary results folder."""
        from tps.analysis.regions import REGIONS
        from tps.analysis.stats import run_all
        from tps.analysis.figures import make_all
        from synthetic_features import make_features
        cls.temp = tempfile.TemporaryDirectory()
        cls.results = Path(cls.temp.name)
        shift = {('theta', r): 0.15 for r in REGIONS} | {('alpha', 'occipital'): -0.2}
        features = make_features(12, shift, ('open', 'closed'), noise=0.1)
        # Drop a few channels of one recording, as bad channels do in the real data.
        features = features[~((features['sub'] == 'sub-01') & (features['condition'] == 'SD')
                              & features['channel'].isin(['Fp1', 'Fp2']))]
        write_inputs(cls.results, features)
        run_all(cls.results)
        cls.out = cls.results / 'figures'
        cls.out.mkdir()
        (cls.out / 'paired_open.png').write_bytes(b'old')   # a figure of an earlier version
        cls.made = make_all(cls.results)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_make_all_saves_the_two_figures_per_eye_condition(self):
        expected = [f'{kind}_{eyes}' for kind in ('psd_cz', 'topomaps') for eyes in ('open', 'closed')]
        for name in expected:
            self.assertIn(name, self.made)
            path = self.out / f'{name}.png'
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)   # not an empty canvas
        self.assertEqual(sorted(p.stem for p in self.out.glob('*.png')), sorted(expected))   # old one removed
        captions = (self.out / 'captions.md').read_text(encoding='utf-8')
        for name in expected + ['hypothesis_summary']:
            self.assertIn(f'## {name}', captions)
        self.assertIn('Leyendas de las figuras', captions)
        self.assertIn('Ojos abiertos (análisis principal)', captions)

    def test_spectrum_has_one_curve_per_condition(self):
        figure = self.made['psd_cz_open']
        curves = [line for line in figure.axes[0].lines if line.get_label().startswith(('NS (', 'SD ('))]
        self.assertEqual(len(curves), 2)
        self.assertIn('n = 12', figure.axes[0].get_title())

    def test_report_is_short_and_in_spanish(self):
        report = (self.results / 'report.md').read_text(encoding='utf-8')
        for section in ('## Datos', '## Resumen · ojos abiertos', '## Hipótesis',
                        '## Figuras · ojos abiertos', '## Avanzado · ojos cerrados'):
            self.assertIn(section, report)
        for name in self.made:
            if name != 'hypothesis_summary':
                self.assertIn(f'](figures/{name}.png)', report)
        self.assertIn('Cambio (%)', report)
        self.assertNotIn('nan |', report.lower())

    def test_hypothesis_summary_reflects_simulated_effects(self):
        from tps.analysis.hypotheses import HYPOTHESES, agreement
        table = pd.read_csv(self.out / 'hypothesis_summary.csv')
        self.assertEqual(set(table['hypothesis']), {h.id for h in HYPOTHESES})
        theta = table[(table['hypothesis'] == 'H1') & (table['band'] == 'theta') & (table['eyes'] == 'open')]
        self.assertTrue((theta['agreement'] == 'supported').all())
        self.assertTrue((theta['change_pct'] > 0).all())
        alpha = table[(table['hypothesis'] == 'H2') & (table['region'] == 'occipital') & (table['eyes'] == 'open')]
        self.assertEqual(alpha['agreement'].iloc[0], 'supported')
        self.assertTrue(table[table['hypothesis'] == 'H4']['agreement'].str.startswith('exploratory').all())
        self.assertEqual(agreement('+', 0.2, 0.01), 'supported')
        self.assertEqual(agreement('+', -0.2, 0.01), 'opposite')
        self.assertEqual(agreement('-', -0.2, 0.2), 'not significant')
        self.assertEqual(agreement(None, 0.1, 0.01), 'exploratory: significant')

    def test_references_are_todo_not_invented(self):
        from tps.analysis.hypotheses import HYPOTHESES
        for h in HYPOTHESES:
            for ref in h.references:
                self.assertEqual((ref.authors, ref.year, ref.doi), ('TODO', 'TODO', 'TODO'))


@unittest.skipIf(pd is None, 'pandas not installed (pip install -r requirements-analysis.txt)')
class MissingInputTests(unittest.TestCase):
    def test_missing_psd_skips_only_the_spectrum(self):
        from tps.analysis.figures import make_all
        from tps.analysis.stats import run_all
        from synthetic_features import make_features
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_inputs(folder, make_features(6, {}, ('open',)))
            (folder / 'psd_channels.csv').unlink()
            run_all(folder)
            made = make_all(folder)
            self.assertNotIn('psd_cz_open', made)
            self.assertIn('topomaps_open', made)
            self.assertTrue((folder / 'report.md').exists())

    def test_one_pair_draws_figures_without_p(self):
        from tps.analysis.figures import make_all
        from tps.analysis.stats import run_all
        from synthetic_features import make_features
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_inputs(folder, make_features(1, {}, ('open',)))
            run_all(folder)
            summary = pd.read_csv(folder / 'stats_summary.csv')
            self.assertTrue(summary.query("eyes == 'open'")['p'].isna().all())
            self.assertTrue(summary.query("eyes == 'open'")['change_pct'].notna().all())
            made = make_all(folder)
            self.assertIn('psd_cz_open', made)
            self.assertIn('topomaps_open', made)


if __name__ == '__main__':
    unittest.main()
