"""Headless validation workflow shared by CLI and the separate pilot viewer."""
from dataclasses import asdict
import json

import mne
import numpy as np
from scipy.signal import welch

from tps.core.data import Study, open_recording
from tps.validation.engine import (ICAResult, ProcessingConfig, Quality, apply_components, assess, candidate_profiles,
                         evaluation_report, finalize, fit_ica, interpolate, prepare, proposals)
from tps.validation.store import atomic_write, encode, utc, versions
from tps.core.jsonio import json_safe  # noqa: F401  (also used by callers of this module)


class ValidationSession:
    def __init__(self, revision):
        self.revision = revision
        self.cancel_event = None
        if revision.data['versions'] != versions():
            raise ValueError('El entorno cambió. Restaurar las versiones guardadas o crear otra revisión.')
        if revision.data['processing'] != json.loads(encode(asdict(ProcessingConfig()))):
            raise ValueError('La configuración del motor cambió; crear una nueva revisión.')
        source = revision.data['source']
        from pathlib import Path
        self.recording = open_recording(Study(Path(source['path']), source['participant'],
                                              source['session'], source['task']))
        try:
            self.prepared = prepare(self.recording.raw, revision.data['persistent'], revision.profile)
            self.fitted, self.cleaned = None, None
            if 'ica' in revision.data['artifacts']:
                ica = mne.preprocessing.read_ica(revision.read_artifact('ica'), verbose=False)
                metadata = json.loads(revision.read_artifact('fit').read_text())
                self.fitted = ICAResult(ica=ica, **metadata)
            if 'clean' in revision.data['artifacts']:
                self.cleaned = mne.io.read_raw_fif(revision.read_artifact('clean'), preload=True, verbose=False)
        except Exception:
            self.recording.close()
            raise

    def close(self):
        self.recording.close()

    def checkpoint(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ValueError('Cálculo cancelado antes de guardar; se conserva el resultado anterior.')

    def fit(self, persistent, excluded_epochs):
        r = self.revision
        if not r.data['blind_complete']:
            raise ValueError('Completar primero la revisión ciega.')
        if any(type(i) is not int or not 0 <= i < len(self.prepared.quality.epochs) for i in excluded_epochs):
            raise ValueError('Época excluida inválida.')
        prepared = prepare(self.recording.raw, persistent, r.profile)
        fitted = fit_ica(prepared, excluded_epochs, r.data['labels'], r.profile)
        # Compute successfully before replacing the last usable revision.
        self.checkpoint()
        previous = self.prepared, self.fitted, self.cleaned
        try:
            with r.transaction():
                r.update_processing(persist=False, persistent=list(persistent),
                                    excluded_epochs=sorted(set(excluded_epochs)),
                                    excluded_components=[], confirmed_repairs=[])
                self.prepared, self.fitted, self.cleaned = prepared, fitted, None
                self._save_fit()
                r.data['stage'] = 'components'
                r.save('ICA ajustada; ningún componente excluido automáticamente')
        except Exception:
            self.prepared, self.fitted, self.cleaned = previous
            raise

    def _save_fit(self):
        r, fitted = self.revision, self.fitted
        r.artifact('ica', lambda p: fitted.ica.save(p, overwrite=False, verbose=False), '-ica.fif')
        metadata = {k: getattr(fitted, k) for k in ('channels', 'rank', 'fit_epochs', 'samples', 'components', 'convergence')}
        r.artifact('fit', lambda p: atomic_write(p, encode(json_safe(metadata))), '-fit.json')
        r.artifact('filtered', lambda p: self.prepared.analysis.save(
            p, fmt='double', overwrite=False, verbose=False), '-filtered-raw.fif')

    def repair_proposals(self, components):
        if self.fitted is None:
            raise ValueError('Ajustar ICA primero.')
        residual = assess(apply_components(self.prepared, self.fitted, components), self.revision.profile)
        return proposals(residual, self.prepared.persistent_repaired + self.prepared.unavailable,
                         self.revision.profile, self.revision.data['excluded_epochs'], self.prepared.unavailable)

    def clean(self, components, repairs):
        if self.fitted is None:
            raise ValueError('Ajustar ICA antes de confirmar la limpieza.')
        r = self.revision
        # Baseline artifact labels describe the original; they must not automatically
        # mark the successfully cleaned output as still bad.
        result = finalize(self.prepared, self.fitted, components, repairs,
                          r.data['excluded_epochs'], profile=r.profile)
        self.checkpoint()
        with r.transaction():
            r.update_processing(persist=False, excluded_components=sorted(set(components)),
                                confirmed_repairs=sorted(set(repairs)))
            self._save_fit()
            r.artifact('clean', lambda p: result.raw.save(p, fmt='double', overwrite=False,
                                                        verbose=False), '-clean-raw.fif')
            r.artifact('cleaning', lambda p: atomic_write(p, encode(json_safe(dict(
                reference_channels=result.reference_channels, repairs=result.repairs,
                persistent_repaired=self.prepared.persistent_repaired,
                unavailable=self.prepared.unavailable, warnings=self.prepared.warnings,
                interpolation=dict(method='spherical_spline', origin=[0., 0., 0.]),
                excluded_components=result.excluded_components, excluded_epochs=result.excluded_epochs)))),
                       '-cleaning.json')
            r.data['reference_channels'] = result.reference_channels
            r.data['stage'] = 'clean'
            r.save('Limpieza confirmada; revisión final pendiente')
        self.cleaned = result.raw
        return result

    def report(self):
        r = self.revision
        r.check()
        if not r.data['blind_complete']:
            raise ValueError('El informe revela sugerencias: completar primero la revisión ciega.')
        source = r.data['source']
        # Baseline metrics must precede persistent interpolation.
        baseline = prepare(self.recording.raw, profile=r.profile).analysis
        q = assess(baseline, r.profile)
        result = dict(created_at=utc(), revision_id=r.data['id'], split=source['split'],
                      revision_sha256=r.digest, source_files=source['files'],
                      participant=source['participant'], session=source['session'], task=source['task'],
                      profile=r.data['profile'], processing=r.data['processing'], versions=r.data['versions'],
                      filters=r.data['filters'], excluded_epochs=r.data['excluded_epochs'],
                      excluded_components=r.data['excluded_components'],
                      baseline=evaluation_report(q, r.data['labels']),
                      reference_channels=r.data['reference_channels'],
                      complete_epochs=sum(not e.partial for e in q.epochs),
                      persistent_suggestions=q.persistent, automatic_acceptance_enabled=False,
                      approved_for_integration=False,
                      limitations=['Piloto de cuatro participantes; no estima desempeño poblacional.',
                                   'Indicadores no evaluables nunca se cuentan como aceptación segura.',
                                   'Revisión humana única; no mide acuerdo entre revisores.',
                                   'Porcentajes del paper son referencias, no objetivos de ajuste.'])
        if source['split'] == 'calibration':
            candidates = []
            for p in candidate_profiles():
                metrics = q.metrics
                rules = dict(no_finitos=metrics['rules']['no_finitos'],
                             alta=metrics['amplitude_uv'] > p.high_uv,
                             plana=metrics['amplitude_uv'] < p.flat_uv,
                             varianza_atipica=np.abs(metrics['variance_z']) > p.robust_z,
                             potencia_rapida_atipica=metrics['fast_ratio_z'] > p.robust_z)
                trial = Quality(q.channels, q.epochs, dict(metrics, rules=rules),
                                np.logical_or.reduce(list(rules.values())), q.evaluable, [])
                repairs = proposals(trial, r.data['persistent'], p, r.data['excluded_epochs'])
                candidates.append(dict(profile=asdict(p), counts=evaluation_report(trial, r.data['labels']),
                                       repairable_epochs=sum(x.eligible for x in repairs)))
            result['candidate_comparison'] = candidates
        if self.cleaned is not None:
            result['cleaning'] = json.loads(r.read_artifact('cleaning').read_text())
            result['ica_fit'] = json.loads(r.read_artifact('fit').read_text())
            final_quality = assess(self.cleaned, r.profile)
            result['residual'] = evaluation_report(final_quality, r.data['final_labels'])
            result['spectral_preservation'] = self._spectral_preservation(baseline, q)
        return json_safe(result)

    def _spectral_preservation(self, baseline, quality):
        r = self.revision
        baseline = baseline.copy().set_eeg_reference(r.data['reference_channels'], verbose=False)
        entries = []
        for j, c in enumerate(baseline.ch_names):
            eligible = [e for e in quality.epochs if not e.partial and e.index not in r.data['excluded_epochs']
                        and r.data['labels'].get(f'{e.index}:{c}', {}).get('state') == 'physiological']
            if not eligible:
                continue
            before, after = [], []
            for e in eligible:
                for raw, output in ((baseline, before), (self.cleaned, after)):
                    f, p = welch(raw.get_data(picks=[c], start=e.first, stop=e.last)[0],
                                 fs=raw.info['sfreq'], nperseg=round(2 * raw.info['sfreq']))
                    output.append(p)
            before, after = np.mean(before, axis=0), np.mean(after, axis=0)
            mask = (f >= .5) & (f <= 30)
            denominator = np.linalg.norm(before[mask])
            bands = {}
            for name, low, high in [('delta', .5, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta', 13, 30)]:
                from tps.core.spectral import integrate_band
                a, b = integrate_band(f, before, low, high), integrate_band(f, after, low, high)
                bands[name] = dict(before_uv2=a * 1e12, after_uv2=b * 1e12,
                                   relative_change=(b - a) / a if a else None)
            entries.append(dict(channel=c, epochs=[e.index for e in eligible],
                                relative_psd_error=float(np.linalg.norm(after[mask] - before[mask]) /
                                                         denominator) if denominator else None,
                                bands=bands))
        return entries

    def holdouts(self):
        """Reconstruct three known healthy channels, never use synthetic truth as EEG approval."""
        r = self.revision
        if not r.data['blind_complete']:
            raise ValueError('Completar la revisión ciega antes de evaluar interpolación.')
        raw = self.prepared.analysis
        results = []
        for c in ('Cz', 'Pz', 'Oz'):
            if c not in raw.ch_names or c in r.data['persistent']:
                continue
            epochs = [e for e in self.prepared.quality.epochs if not e.partial and
                      e.index not in r.data['excluded_epochs'] and all(
                          r.data['labels'].get(f'{e.index}:{other}', {}).get('state') == 'physiological'
                          for other in raw.ch_names)]
            if not epochs:
                results.append(dict(channel=c, status='sin_epochs_con_donantes_sanos'))
                continue
            try:
                fixed = interpolate(raw, [c])
            except ValueError as exc:
                results.append(dict(channel=c, status=str(exc)))
                continue
            original = np.concatenate([raw.get_data(picks=[c], start=e.first, stop=e.last)[0] for e in epochs])
            reconstructed = np.concatenate([fixed.get_data(picks=[c], start=e.first, stop=e.last)[0] for e in epochs])
            denom = np.std(original)
            results.append(dict(channel=c, epochs=[e.index for e in epochs],
                                normalized_rmse=float(np.sqrt(np.mean((original - reconstructed) ** 2)) / denom)
                                if denom else None))
        return results
