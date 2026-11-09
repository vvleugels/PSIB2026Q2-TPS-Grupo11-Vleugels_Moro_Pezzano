"""Extended-Infomax ICA fitting shared by the explorer (tps.core.ica) and the validation pilot.

Kept in the core so ICA does not depend on the experimental validation tool.
"""
from dataclasses import asdict, dataclass, field
import logging
import math
import threading
import warnings

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis


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
