"""Experimental EEG cleaning for the validation pilot (Infomax fitting lives in tps.core.infomax).

All transformations work on copies. Findings are suggestions, not decisions.
Units are volts internally and microvolts only for amplitude thresholds.
"""
from dataclasses import asdict, dataclass
from itertools import product
import math
import warnings

import mne
import numpy as np

from tps.core.annotations import Epoch
from tps.core.processing import ContinuousFilter, FilterConfig, FILTER_OPTIONS
# Re-exported: the pilot's session/store and tests import these from here.
from tps.core.infomax import ICAResult, ProcessingConfig, RuleProfile, fit_infomax, power_ratio  # noqa: F401


@dataclass
class Quality:
    channels: list
    epochs: list
    metrics: dict
    flags: np.ndarray
    evaluable: np.ndarray
    persistent: list

    def reasons(self, epoch, channel):
        j = self.channels.index(channel)
        return [name for name, mask in self.metrics['rules'].items() if mask[epoch, j]]


@dataclass(frozen=True)
class RepairProposal:
    epoch: int
    channels: tuple
    total_affected: int
    eligible: bool


@dataclass
class Prepared:
    analysis: object
    training: object
    quality: Quality
    persistent_repaired: list
    unavailable: list
    warnings: list


@dataclass
class CleanResult:
    raw: object
    quality: Quality
    reference_channels: list
    repairs: list
    excluded_components: list
    excluded_epochs: list


def epochs4(n_times, sfreq):
    if n_times < 1 or not math.isfinite(sfreq) or sfreq < 0.25:
        raise ValueError('Se requieren muestras y al menos una muestra por época de 4 s.')
    epochs, first = [], 0
    while first < n_times:
        boundary = round((len(epochs) + 1) * 4 * sfreq)
        last = min(n_times, boundary)
        epochs.append(Epoch(len(epochs), first, last, sfreq, last < boundary))
        first = last
    return epochs


def robust_scores(values, complete):
    """Per-channel robust deviations; zero MAD deliberately remains unevaluable."""
    result = np.full_like(values, np.nan)
    for j in range(values.shape[1]):
        valid = complete & np.isfinite(values[:, j])
        if valid.sum() < 3:
            continue
        median = np.median(values[valid, j])
        mad = np.median(np.abs(values[valid, j] - median))
        if mad > np.finfo(float).eps:
            result[:, j] = 0.6744897501960817 * (values[:, j] - median) / mad
    return result


def assess(raw, profile=RuleProfile()):
    profile.validate()
    channels = raw.ch_names
    epochs = epochs4(raw.n_times, float(raw.info['sfreq']))
    shape = (len(epochs), len(channels))
    amp, variance, ratio = (np.full(shape, np.nan) for _ in range(3))
    nonfinite = np.zeros(shape, dtype=bool)
    for e in epochs:
        values = raw.get_data(start=e.first, stop=e.last)
        for j, samples in enumerate(values):
            nonfinite[e.index, j] = not np.all(np.isfinite(samples))
            if nonfinite[e.index, j] or len(samples) < 2:
                continue
            amp[e.index, j] = np.ptp(samples) * 1e6
            variance[e.index, j] = np.var(samples)
            ratio[e.index, j] = power_ratio(samples, raw.info['sfreq'])
    complete = np.array([not e.partial for e in epochs])
    logvar = np.full(shape, np.nan)
    np.log(variance, out=logvar, where=variance > 0)
    zvar, zratio = robust_scores(logvar, complete), robust_scores(ratio, complete)
    rules = {'no_finitos': nonfinite, 'alta': amp > profile.high_uv,
             'plana': amp < profile.flat_uv, 'varianza_atipica': np.abs(zvar) > profile.robust_z,
             'potencia_rapida_atipica': zratio > profile.robust_z}
    flags = np.logical_or.reduce(list(rules.values()))
    evaluable = np.isfinite(amp) & np.isfinite(zvar) & np.isfinite(zratio) & complete[:, None]
    persistent = [c for j, c in enumerate(channels) if complete.any()
                  and np.mean(flags[complete, j]) > profile.persistent_fraction]
    return Quality(channels, epochs, dict(amplitude_uv=amp, variance=variance,
                   fast_ratio=ratio, variance_z=zvar, fast_ratio_z=zratio, rules=rules),
                   flags, evaluable, persistent)


def valid_position(raw, name):
    position = raw.info['chs'][raw.ch_names.index(name)]['loc'][:3]
    return np.all(np.isfinite(position)) and np.linalg.norm(position) > 0


def interpolate(raw, bads):
    """Spherical interpolation with explicit origin; positions must be available."""
    if not bads:
        return raw.copy()
    if len(set(raw.ch_names) - set(bads)) < 4:
        raise ValueError('La interpolación requiere al menos cuatro canales donantes.')
    if not all(valid_position(raw, c) for c in raw.ch_names):
        raise ValueError('Faltan posiciones válidas para interpolar sin inventar geometría.')
    result = raw.copy()
    result.info['bads'] = list(bads)
    result.interpolate_bads(reset_bads=True, origin=(0., 0., 0.),
                            method=dict(eeg='spline'), verbose=False)
    return result


def prepare(raw, persistent=(), profile=RuleProfile()):
    original = raw.copy().pick('eeg').load_data()
    if not original.ch_names or raw.info['sfreq'] <= 90:
        raise ValueError('Se requieren canales EEG y frecuencia de muestreo mayor a 90 Hz.')
    if len(set(persistent)) != len(persistent) or not set(persistent) <= set(original.ch_names):
        raise ValueError('Lista de electrodos persistentes inválida.')
    # Reject before FIR: NaNs must never spread to other epochs.
    if not np.isfinite(original.get_data()).all():
        raise ValueError('Hay muestras no finitas. Revisar el original antes de filtrar.')
    outputs = []
    for low in (.2, 1.):
        cache = ContinuousFilter(original)
        cache.prepare(FilterConfig(True, low, 45.))
        outputs.append(original.copy().filter(low, 45., **FILTER_OPTIONS,
                                              pad='reflect_limited', verbose=False))
    analysis, training = outputs
    notes, unavailable, repaired = [], [], []
    if persistent:
        try:
            analysis = interpolate(analysis, persistent)
            training = interpolate(training, persistent)
            repaired = list(persistent)
        except ValueError as exc:
            unavailable = list(persistent)
            notes.append(str(exc) + ' Canales persistentes excluidos del ajuste y referencia.')
    return Prepared(analysis, training, assess(analysis, profile), repaired, unavailable, notes)


def fit_ica(prepared, excluded_epochs=(), manual=None, profile=RuleProfile()):
    manual = manual or {}
    quality = prepared.quality
    channels = [c for c in prepared.analysis.ch_names if c not in prepared.unavailable]
    chosen = []
    for e in quality.epochs:
        if e.partial or e.index in excluded_epochs:
            continue
        usable = True
        for c in channels:
            label = manual.get(f'{e.index}:{c}', {}).get('state')
            if c in prepared.persistent_repaired:
                label = None  # The original bad-channel label does not describe its interpolation.
            flagged = quality.flags[e.index, quality.channels.index(c)]
            if label == 'artifact' or (flagged and label != 'physiological'):
                usable = False
        if usable:
            chosen.append(e)
    if not chosen or len(channels) < 2:
        raise ValueError('No hay épocas/canales suficientes sin problemas pendientes para ICA.')
    data = np.concatenate([prepared.training.get_data(picks=channels, start=e.first, stop=e.last)
                           for e in chosen], axis=1)
    info = mne.pick_info(prepared.training.info, [prepared.training.ch_names.index(c) for c in channels])
    return fit_infomax(data, info, channels, [e.index for e in chosen], profile)


def apply_components(prepared, fitted, excluded):
    excluded = sorted(set(excluded))
    if any(type(i) is not int or not 0 <= i < fitted.ica.n_components_ for i in excluded):
        raise ValueError('Índices de componentes inválidos.')
    expected = [c for c in prepared.analysis.ch_names if c not in prepared.unavailable]
    if expected != fitted.channels or fitted.ica.ch_names != expected:
        raise ValueError('Los canales del ajuste y aplicación no son compatibles.')
    result = prepared.analysis.copy()
    fitted.ica.apply(result, exclude=excluded, verbose=False)
    return result


def proposals(quality, persistent=(), profile=RuleProfile(), excluded_epochs=(), unavailable=()):
    result = []
    for e in quality.epochs:
        if e.partial or e.index in excluded_epochs:
            continue
        bads = tuple(c for c, flag in zip(quality.channels, quality.flags[e.index]) if flag and c not in unavailable)
        if bads:
            total = len(set(bads) | set(persistent))
            result.append(RepairProposal(e.index, bads, total, total <= profile.max_repair))
    return result


def finalize(prepared, fitted, excluded_components, confirmed_repairs=(), excluded_epochs=(),
             manual=None, profile=RuleProfile()):
    raw = apply_components(prepared, fitted, excluded_components)
    quality = assess(raw, profile)
    available = {p.epoch: p for p in proposals(quality,
                                             prepared.persistent_repaired + prepared.unavailable,
                                             profile, excluded_epochs, prepared.unavailable)}
    repairs = []
    for index in sorted(set(confirmed_repairs)):
        p = available.get(index)
        if p is None or not p.eligible:
            raise ValueError(f'Época {index + 1}: reparación no elegible.')
        e = quality.epochs[index]
        picks = [j for j, c in enumerate(raw.ch_names) if c not in prepared.unavailable]
        segment = mne.io.RawArray(raw.get_data(picks=picks, start=e.first, stop=e.last),
                                 mne.pick_info(raw.info, picks), verbose=False)
        fixed = interpolate(segment, p.channels)
        raw._data[picks, e.first:e.last] = fixed.get_data()
        repairs.append(asdict(p))
    residual = assess(raw, profile)
    # Fixed across every usable epoch. A failed local channel is not a global epoch rejection.
    blocked = set(prepared.unavailable)
    manual = manual or {}
    for e in residual.epochs:
        if e.partial or e.index in excluded_epochs:
            continue
        for j, c in enumerate(residual.channels):
            label = manual.get(f'{e.index}:{c}', {}).get('state')
            if residual.flags[e.index, j] or label == 'artifact':
                blocked.add(c)
    reference = [c for c in raw.ch_names if c not in blocked]
    if len(reference) < 2:
        raise ValueError('Menos de dos canales válidos para una referencia fija; revisar el registro.')
    raw.set_eeg_reference(ref_channels=reference, projection=False, verbose=False)
    return CleanResult(raw, assess(raw, profile), reference, repairs,
                       sorted(set(excluded_components)), sorted(set(excluded_epochs)))


def candidate_profiles():
    return [RuleProfile(high, flat, z, limit)
            for high, flat, z, limit in product((100., 150., 200.), (.5, 1.), (4., 5., 6.), (1, 2, 3))]


def evaluation_report(quality, labels):
    counts = dict(true_artifact=0, false_alarm=0, missed_artifact=0, true_clean=0,
                  unevaluable=0, unreviewed=0)
    for e in quality.epochs:
        if e.partial:
            continue
        for j, c in enumerate(quality.channels):
            state = labels.get(f'{e.index}:{c}', {}).get('state')
            if state not in ('artifact', 'physiological'):
                counts['unreviewed'] += 1
                continue
            flag = quality.flags[e.index, j]
            if flag:
                counts['true_artifact' if state == 'artifact' else 'false_alarm'] += 1
            elif quality.evaluable[e.index, j]:
                counts['missed_artifact' if state == 'artifact' else 'true_clean'] += 1
            else:
                counts['unevaluable'] += 1
    counts['automatic_acceptance_enabled'] = False
    return counts
