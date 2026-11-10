"""Automatic bad-channel detection over a whole recording, independent of Tk.

The criteria follow the ideas of EEGLAB's clean_rawdata / pop_rejchan and of the PREP
pipeline (Bigdely-Shamlo et al., 2015), reimplemented here with numpy; no code is copied.
Every flag carries a readable reason. They are suggestions with starting thresholds,
NOT validated against visual inspection for this dataset.

Criteria (each channel against the others of the same recording):
* non-finite samples;
* flat: the signal varies by less than ``flat_uv`` for at least ``flat_seconds``;
* amplitude deviation: robust z of the channel's robust SD (IQR / 1.349) > ``deviation_z``;
* low correlation: in 1 s windows, the 98th percentile of the channel's absolute
  correlation with the other channels is < ``min_correlation`` in more than
  ``max_bad_time`` of the windows (PREP's criterion without electrode positions);
* high-frequency noise: robust z of log10(power above ``hf_split_hz`` / power below it),
  on the signal without band-pass (so line noise and EMG remain visible) > ``hf_z``.
The relative criteria need at least ``MIN_CHANNELS`` channels; with fewer they are skipped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.signal import welch

from tps.core.processing import FilterConfig

MIN_CHANNELS = 5          # robust z-scores need a population of channels
MAD_TO_SD = 1.4826        # MAD of a normal distribution -> its SD
IQR_TO_SD = 1.349

CRITERIA = {
    'non_finite': 'muestras no finitas',
    'flat': 'señal plana',
    'deviation': 'amplitud anómala',
    'correlation': 'baja correlación con los demás canales',
    'high_frequency': 'ruido de alta frecuencia',
    'peak_to_peak': 'pico a pico fuera de límites en muchas épocas',
    'manual': 'marcado a mano',
}


@dataclass(frozen=True)
class BadChannelConfig:
    flat_uv: float = 1.0
    flat_seconds: float = 5.0
    deviation_z: float = 5.0
    min_correlation: float = 0.4
    correlation_window: float = 1.0
    max_bad_time: float = 0.20     # PREP uses 0.01 (it interpolates), EEGLAB clean_rawdata 0.5
    hf_split_hz: float = 45.0
    hf_z: float = 5.0

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Los parámetros de canales malos deben ser positivos y finitos.')
        if not 0 < self.min_correlation < 1 or not 0 < self.max_bad_time < 1:
            raise ValueError('La correlación mínima y el tiempo máximo deben estar entre 0 y 1.')


def robust_z(values: np.ndarray) -> np.ndarray:
    """(x − median) / (1.4826 · MAD); zeros when the MAD is 0 (no spread to compare with)."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * MAD_TO_SD
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(values)
    return (values - median) / mad


def longest_flat_run(values_uv: np.ndarray, sfreq: float, flat_uv: float, block_seconds: float = 0.5) -> float:
    """Longest stretch (s) of consecutive blocks whose peak-to-peak is below flat_uv."""
    block = max(2, int(round(block_seconds * sfreq)))
    n = len(values_uv) // block
    if n == 0:
        return 0.0
    ptp = np.ptp(values_uv[:n * block].reshape(n, block), axis=1)
    longest = current = 0
    for flat in ptp < flat_uv:
        current = current + 1 if flat else 0
        longest = max(longest, current)
    return longest * block / sfreq


def correlation_failures(data_uv: np.ndarray, sfreq: float, config: BadChannelConfig) -> np.ndarray:
    """Fraction of windows in which each channel's 98th-percentile |r| is below the minimum."""
    window = max(2, int(round(config.correlation_window * sfreq)))
    n_windows = data_uv.shape[1] // window
    if n_windows == 0:
        return np.zeros(len(data_uv))
    failures = np.zeros(len(data_uv))
    for k in range(n_windows):
        part = data_uv[:, k * window:(k + 1) * window]
        part = part - part.mean(axis=1, keepdims=True)
        norms = np.sqrt((part ** 2).sum(axis=1))
        alive = norms > 0
        if alive.sum() < MIN_CHANNELS:
            continue
        unit = part[alive] / norms[alive, None]
        corr = np.abs(unit @ unit.T)
        np.fill_diagonal(corr, np.nan)
        high = np.nanquantile(corr, 0.98, axis=1)
        failures[np.flatnonzero(alive)[high < config.min_correlation]] += 1
    return failures / n_windows


def high_frequency_ratio(raw_uv: np.ndarray, sfreq: float, split_hz: float) -> np.ndarray | None:
    """log10(power above split / power 1 Hz..split) per channel, or None if Nyquist ≤ split."""
    if sfreq / 2 <= split_hz + 1:
        return None
    frequencies, psd = welch(raw_uv, fs=sfreq, nperseg=min(raw_uv.shape[1], int(2 * sfreq)), axis=1)
    high = psd[:, frequencies > split_hz].sum(axis=1)
    low = psd[:, (frequencies >= 1) & (frequencies <= split_hz)].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(np.where(low > 0, high / low, np.nan))


def detect(data_uv: np.ndarray, raw_uv: np.ndarray, channels: list[str], sfreq: float,
           config: BadChannelConfig) -> dict[str, dict]:
    """Bad channels of arrays in µV: {channel: {'criterion', 'reason'}} (first criterion wins,
    all reasons are joined). data_uv is the analysed (filtered) signal; raw_uv the signal
    without band-pass for the high-frequency criterion."""
    config.validate()
    found: dict[str, list[tuple[str, str]]] = {c: [] for c in channels}
    finite = np.all(np.isfinite(data_uv), axis=1) & np.all(np.isfinite(raw_uv), axis=1)
    for i in np.flatnonzero(~finite):
        found[channels[i]].append(('non_finite', 'muestras no finitas'))
    for i in np.flatnonzero(finite):
        run = longest_flat_run(data_uv[i], sfreq, config.flat_uv)
        if run >= config.flat_seconds:
            found[channels[i]].append(
                ('flat', f'varía menos de {config.flat_uv:g} µV durante {run:.1f} s '
                         f'(≥ {config.flat_seconds:g} s)'))
    usable = [i for i in np.flatnonzero(finite) if not found[channels[i]]]
    if len(usable) >= MIN_CHANNELS:
        data = data_uv[usable]
        q75, q25 = np.percentile(data, [75, 25], axis=1)
        spread = (q75 - q25) / IQR_TO_SD
        z = robust_z(spread)
        for j, i in enumerate(usable):
            if abs(z[j]) > config.deviation_z:
                found[channels[i]].append(
                    ('deviation', f'amplitud {spread[j]:.1f} µV (DE robusta), z robusto {z[j]:+.1f} '
                                  f'(|z| > {config.deviation_z:g})'))
        failed = correlation_failures(data, sfreq, config)
        for j, i in enumerate(usable):
            if failed[j] > config.max_bad_time:
                found[channels[i]].append(
                    ('correlation', f'correlación con los demás canales < {config.min_correlation:g} '
                                    f'en el {failed[j]:.0%} de las ventanas de {config.correlation_window:g} s '
                                    f'(> {config.max_bad_time:.0%})'))
        ratio = high_frequency_ratio(raw_uv[usable], sfreq, config.hf_split_hz)
        if ratio is not None and np.all(np.isfinite(ratio)):
            z = robust_z(ratio)
            for j, i in enumerate(usable):
                if z[j] > config.hf_z:
                    found[channels[i]].append(
                        ('high_frequency', f'potencia > {config.hf_split_hz:g} Hz relativa elevada, '
                                           f'z robusto {z[j]:+.1f} (> {config.hf_z:g})'))
    return {channel: dict(criterion=items[0][0], reason='; '.join(text for _, text in items))
            for channel, items in found.items() if items}


def find_bad_channels(recording, filtering: FilterConfig, config: BadChannelConfig) -> dict[str, dict]:
    """Bad channels of an open recording over its whole length (see the module docstring).

    Uses the filtered continuous channels (cached by ContinuousFilter) for every criterion
    except high-frequency noise, which uses the signal without band-pass (after ICA, if
    one is active). Raises nothing for non-finite data: those channels are flagged."""
    channels = list(recording.channels)
    raw = recording.signal_raw.get_data(picks=channels) * 1e6
    finite = np.all(np.isfinite(raw), axis=1)
    if filtering.active and finite.all():
        cache = recording.filter.prefetch(channels, filtering)
        data = np.stack([cache[c] for c in channels]) * 1e6
    elif filtering.active:
        # Filtering would spread NaNs; filter only the finite channels.
        good = [c for c, ok in zip(channels, finite) if ok]
        cache = recording.filter.prefetch(good, filtering) if good else {}
        data = np.stack([cache[c] * 1e6 if c in cache else raw[i] for i, c in enumerate(channels)])
    else:
        data = raw
    return detect(data, raw, channels, recording.sfreq, config)
