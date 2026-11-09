"""Experimental EEG cleaning plus the shared Infomax fitting engine.

All transformations work on copies. Findings are suggestions, not decisions.
Units are volts internally and microvolts only for amplitude thresholds.
"""
from dataclasses import asdict, dataclass, field
from itertools import product
import logging
import math
import threading
import warnings

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis

from .annotations import Epoch
from .processing import ContinuousFilter, FilterConfig, FILTER_OPTIONS


@dataclass(frozen=True)
class ProcessingConfig:
    epoch_seconds: float = 4.0
    analysis_band: tuple = (0.2, 45.0)
    training_band: tuple = (1.0, 45.0)
    method: str = 'infomax'
    extended: bool = True
    seed: int = 42
    max_iter: int = 1000
    weight_change: float = 1e-12
    small_angle_steps: int = 20


@dataclass(frozen=True)
class RuleProfile:
    high_uv: float = 150.0
    flat_uv: float = 1.0
    robust_z: float = 5.0
    max_repair: int = 2
    persistent_fraction: float = 0.2
    component_kurtosis: float = 5.0
    component_ratio: float = 0.5
    frontal_correlation: float = 0.5

    def validate(self):
        values = asdict(self)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
            raise ValueError('El perfil debe contener números finitos.')
        if not (0 <= self.flat_uv < self.high_uv and self.robust_z > 0
                and type(self.max_repair) is int and self.max_repair in (1, 2, 3)
                and 0 < self.persistent_fraction < 1
                and self.component_kurtosis > 0 and 0 < self.component_ratio < 1
                and 0 < self.frontal_correlation < 1):
            raise ValueError('Perfil experimental inválido.')


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
class ICAResult:
    ica: object
    channels: list
    rank: int
    fit_epochs: list
    samples: int
    components: list
    convergence: dict = field(default_factory=dict)


class InfomaxTrace(logging.Filter):
    """Capture full-precision stopping evidence, without console iteration spam.

    MNE's Infomax sets its returned iteration counter to max_iter even when
    weight-change convergence occurs early. n_iter_ alone cannot identify failure.
    """
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()
        self.step = 0
        self.change = float('inf')

    def filter(self, record):
        if record.thread != self.thread:
            return True
        if str(record.msg).startswith('step %d - lrate') and len(record.args) >= 3:
            self.step, self.change = int(record.args[0]), float(record.args[2])
        return False


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
        raise ValueError('Se requieren muestras y al menos una muestra por epoch de 4 s.')
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


def power_ratio(samples, sfreq):
    if samples.shape[-1] < 2 or not np.all(np.isfinite(samples)):
        return float('nan')
    f, psd = welch(samples, fs=sfreq, nperseg=min(round(2 * sfreq), samples.shape[-1]))
    def power(low, high):
        inner = (f > low) & (f < high)
        x = np.r_[low, f[inner], high]
        return trapezoid(np.r_[np.interp(low, f, psd), psd[inner], np.interp(high, f, psd)], x)
    if f[-1] < 45:
        return float('nan')
    total = power(1, 45)
    return float(power(30, 45) / total) if total > 0 else float('nan')


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
        raise ValueError('No hay epochs/canales suficientes sin problemas pendientes para ICA.')
    data = np.concatenate([prepared.training.get_data(picks=channels, start=e.first, stop=e.last)
                           for e in chosen], axis=1)
    info = mne.pick_info(prepared.training.info, [prepared.training.ch_names.index(c) for c in channels])
    return fit_infomax(data, info, channels, [e.index for e in chosen], profile)


def fit_infomax(data, info, channels, fit_epochs, profile=RuleProfile()):
    """Shared fitting/convergence engine; callers select and continuously filter training data."""
    singular = np.linalg.svd(data - data.mean(axis=1, keepdims=True), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6)) if singular[0] > 0 else 0
    if rank < 2 or data.shape[1] <= rank:
        raise ValueError('Rango o cantidad de muestras insuficiente para ICA.')
    fit_raw = mne.io.RawArray(data, info, verbose=False)
    # No baseline, decimation, or refiltering across the concatenated fit intervals.
    config = ProcessingConfig()
    ica = mne.preprocessing.ICA(n_components=rank, method=config.method,
                                fit_params=dict(extended=True, w_change=config.weight_change,
                                                n_small_angle=config.small_angle_steps, verbose=True),
                                random_state=config.seed,
                                max_iter=config.max_iter)
    trace = InfomaxTrace()
    mne.utils.logger.addFilter(trace)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ica.fit(fit_raw, picks=channels, reject_by_annotation=False, verbose=True)
    finally:
        mne.utils.logger.removeFilter(trace)
    converged = trace.step > 0 and (trace.change < config.weight_change or ica.n_iter_ < config.max_iter)
    if not converged or any('converg' in str(w.message).lower() for w in caught):
        raise ValueError('ICA no convergió; no se puede confirmar este ajuste.')
    sources = ica.get_sources(fit_raw).get_data()
    components = []
    for k, source in enumerate(sources):
        kurt = float(kurtosis(source, fisher=True, bias=False)) if np.std(source) else float('nan')
        ratio = power_ratio(source, fit_raw.info['sfreq'])
        correlations = {}
        for c in ('Fp1', 'Fp2'):
            if c in channels:
                frontal = data[channels.index(c)]
                correlations[c] = float(np.corrcoef(source, frontal)[0, 1]) if np.std(frontal) and np.std(source) else None
        reasons = []
        if abs(kurt) > profile.component_kurtosis:
            reasons.append('curtosis')
        if ratio > profile.component_ratio:
            reasons.append('potencia_rapida')
        if any(v is not None and abs(v) > profile.frontal_correlation for v in correlations.values()):
            reasons.append('correlacion_frontal_no_EOG')
        components.append(dict(index=k, kurtosis=kurt, fast_ratio=ratio,
                               frontal_correlations=correlations, suggestions=reasons))
    return ICAResult(ica, channels, rank, list(fit_epochs), data.shape[1], components,
                     dict(actual_steps=trace.step, reported_iterations=int(ica.n_iter_),
                          final_weight_change=trace.change,
                          stop_reason='weight_change' if trace.change < config.weight_change else 'small_angle'))


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
            raise ValueError(f'Epoch {index + 1}: reparación no elegible.')
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
