import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import welch, get_window

from tps.core.annotations import AnnotationStore, STATES
from tps.core.data import discover, open_recording
from tps.core.processing import FilterConfig
from tps.core.spectral import (SpectralConfig, analyze_recording, band_powers,
    integrate_band, psd_db, run_analysis, select_studies, export_result)
from test_core_data import make_recording


class SpectralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.annotations = self.root / 'annotations'
        self.path, _ = make_recording(self.data)
        t = np.arange(1250) / 100
        self.signal = 10 * np.sin(2*np.pi*10*t)
        samples = np.vstack([self.signal, self.signal * 2]).astype('<f4')
        samples.ravel(order='F').tofile(self.path.with_suffix('.fdt'))
        self.study = discover(self.data)['sub-01'][0]
        self.recording = open_recording(self.study)
        self.addCleanup(self.recording.close)
        self.store = AnnotationStore(self.annotations, self.recording, self.data)
        for i in range(3):
            self.store.mark(i, STATES[1], '', '', ['Cz'], FilterConfig())
        self.store.save()

    def analyze(self, channel='Cz', exploratory=False):
        return analyze_recording(self.recording, channel, FilterConfig(), SpectralConfig(),
                                 self.store, exploratory)

    def test_known_sine_units_bands_amplitude_and_partial(self):
        result = self.analyze()
        f, p = result['frequencies'], result['psd_uv2_hz']
        self.assertEqual(f[np.argmax(p)], 10)
        self.assertAlmostEqual(result['bands'][2]['absolute_uv2'], 50, places=4)
        self.assertAlmostEqual(result['bands'][2]['relative_percent'], 100, places=4)
        self.assertEqual(result['counts']['used'], 2)
        self.assertEqual(result['counts']['partial'], 1)
        self.assertEqual(result['counts']['seconds'], 10)
        doubled = self.analyze('Fp1')
        np.testing.assert_allclose(doubled['psd_uv2_hz'], p * 4)
        self.assertAlmostEqual(doubled['bands'][2]['relative_percent'], 100, places=4)

    def test_epoch_average_matches_independent_welch_not_concatenation(self):
        original = self.recording.raw.get_data(picks=['Cz'])[0]
        expected = []
        for start in (0, 500):
            _, psd = welch(original[start:start+500], fs=100,
                           window=get_window('hann',200,fftbins=True), nperseg=200,
                           noverlap=100, nfft=200, scaling='density', detrend='constant')
            expected.append(psd)
        np.testing.assert_allclose(self.analyze()['psd_uv2_hz'], np.mean(expected, axis=0)*1e12)

    def test_policies_dirty_and_no_eligible(self):
        self.store.mark(0, STATES[2], 'Ocular', '', ['Cz'], FilterConfig())
        with self.assertRaisesRegex(ValueError, 'Guardá'):
            self.analyze()
        self.store.mark(1, STATES[0], '', '', ['Cz'], FilterConfig())
        self.store.save()
        with self.assertRaisesRegex(ValueError, 'no hay épocas'):
            self.analyze()
        result = self.analyze(exploratory=True)
        self.assertEqual(result['counts']['used'], 1)
        self.assertEqual(result['counts']['artifacts'], 1)
        self.assertEqual(result['counts']['pending'], 1)
        self.assertEqual([e['number'] for e in result['epochs']], [2])

    def test_exact_interpolation_relative_zero_and_log_zero(self):
        f = np.array([0,1,2,3,4.])
        self.assertAlmostEqual(integrate_band(f, 2*f, .5, 3.5), 12)
        bands = band_powers(f, np.ones(5), (.2,1,2,3,3.8))
        self.assertAlmostEqual(sum(b['relative_percent'] for b in bands), 100)
        self.assertTrue(all(b['relative_percent'] is None for b in
                            band_powers(f, np.zeros(5), (0,1,2,3,4))))
        self.assertTrue(np.isnan(psd_db(np.array([0.]))[0]))
        self.assertEqual(psd_db(np.array([100.]))[0], 20)

    def test_invalid_parameters(self):
        for config in (SpectralConfig(0), SpectralConfig(6), SpectralConfig(.001),
                       SpectralConfig(2,100), SpectralConfig(2,-1),
                       SpectralConfig(float('nan')), SpectralConfig(edges=(0,4,4,13,30)),
                       SpectralConfig(edges=(.5,4,8,13,51))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.effective(100)

    def test_missing_corrupt_or_changed_annotations(self):
        self.store.path.unlink()
        result = run_analysis([self.study], 'Cz', FilterConfig(), SpectralConfig(),
                              self.data, self.annotations, True)
        self.assertTrue(result['exploratory'])
        with self.assertRaisesRegex(ValueError, 'no hay épocas'):
            run_analysis([self.study], 'Cz', FilterConfig(), SpectralConfig(),
                         self.data, self.annotations)
        self.store.path.write_text('bad json')
        with self.assertRaisesRegex(ValueError, 'protegidas'):
            run_analysis([self.study], 'Cz', FilterConfig(), SpectralConfig(),
                         self.data, self.annotations, True)

    def test_comparison_and_export_optional_png(self):
        make_recording(self.data, session='ses-2')
        inventory = discover(self.data)
        studies = select_studies(self.study, inventory, True)
        other = open_recording(studies[1])
        self.addCleanup(other.close)
        annotations = AnnotationStore(self.annotations, other, self.data)
        annotations.mark(0, STATES[1], '', '', ['Cz'], FilterConfig())
        annotations.save()
        result = run_analysis(studies, 'Cz', FilterConfig(), SpectralConfig(), self.data, self.annotations)
        self.assertEqual([s['counts']['used'] for s in result['sessions']], [2,1])
        output = export_result(result, self.root)
        self.assertFalse((output/'PSD.png').exists())
        with (output/'potencias.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)
        metadata = json.loads((output/'configuracion.json').read_text())
        self.assertEqual(metadata['sessions'][0]['epochs'][0]['number'], 1)
        output2 = export_result(result, self.root, True)
        self.assertNotEqual(output, output2)
        self.assertGreater((output2/'PSD.png').stat().st_size, 1000)

    def test_missing_session_channel_mismatched_source_and_sampling(self):
        with self.assertRaisesRegex(ValueError, 'estudio único'):
            select_studies(self.study, discover(self.data), True)
        with self.assertRaisesRegex(ValueError, 'electrodo'):
            self.analyze('missing')
        with self.assertRaisesRegex(ValueError, 'externamente'):
            run_analysis([self.study], 'Cz', FilterConfig(), SpectralConfig(), self.data,
                         self.annotations, expected={str(self.study.path): (self.store.identity, 'wrong')})
        from scipy.io import loadmat, savemat
        path, _ = make_recording(self.data, session='ses-2')
        content = loadmat(path, simplify_cells=True)
        content['EEG']['srate'] = 200.
        savemat(path, {'EEG':content['EEG']})
        with self.assertRaisesRegex(ValueError, 'distintas frecuencias'):
            run_analysis(select_studies(self.study,discover(self.data),True), 'Cz', FilterConfig(),
                         SpectralConfig(), self.data,self.annotations,True)

    def test_filter_review_difference_and_export_failure_cleanup(self):
        self.store.mark(0, STATES[1], '', '', ['Cz'], FilterConfig(True,2,30))
        self.store.save()
        self.assertTrue(self.analyze()['warnings'])
        result = run_analysis([self.study], 'Cz', FilterConfig(), SpectralConfig(),self.data,self.annotations)
        with patch('tps.core.spectral.json.dumps',side_effect=OSError('failure')):
            with self.assertRaises(OSError):
                export_result(result,self.root)
        self.assertEqual(list(self.root.glob('sub-01_*')), [])
