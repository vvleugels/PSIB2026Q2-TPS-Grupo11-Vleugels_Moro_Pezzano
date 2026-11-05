"""Reproducible continuous filtering, without changing the source recording."""
from dataclasses import asdict, dataclass
import math

import numpy as np
from mne.filter import create_filter, filter_data


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    low: float = 0.2
    high: float = 45.0

    @property
    def label(self):
        return f'Filtrada · {self.low:g}–{self.high:g} Hz' if self.enabled else 'Original'

    def validate(self, sfreq):
        if not (math.isfinite(self.low) and math.isfinite(self.high)
                and 0 < self.low < self.high < sfreq / 2):
            raise ValueError(f'Los cortes deben cumplir 0 < inferior < superior < {sfreq / 2:g} Hz.')

    def context(self):
        import mne
        return {**asdict(self), **FILTER_OPTIONS, 'pad': 'reflect_limited',
                'mne_version': mne.__version__}


FILTER_OPTIONS = dict(method='fir', phase='zero', fir_window='hamming',
                      fir_design='firwin', filter_length='auto',
                      l_trans_bandwidth='auto', h_trans_bandwidth='auto')


class ContinuousFilter:
    """Worker-thread-only cache, bounded by the current study and cutoffs."""
    def __init__(self, raw):
        self.raw = raw
        self.key = None
        self.channels = {}

    def prepare(self, config):
        config.validate(float(self.raw.info['sfreq']))
        if not config.enabled:
            if self.key != (config.low, config.high):
                self.channels.clear()
                self.key = None
            return
        key = (config.low, config.high)
        if key != self.key:
            # Reject impractically narrow transitions before allocating FIR taps.
            # MNE's auto Hamming/firwin length is ~3.3 / min(transition widths).
            sfreq = float(self.raw.info['sfreq'])
            low_width = min(max(config.low * .25, 2), config.low)
            high_width = min(max(config.high * .25, 2), sfreq / 2 - config.high)
            if 3.3 / min(low_width, high_width) > self.raw.n_times / sfreq:
                raise ValueError('Registro demasiado corto para este filtro. '
                                 'Usá cortes con bandas de transición más amplias.')
            coefficients = create_filter(None, self.raw.info['sfreq'],
                                         config.low, config.high, **FILTER_OPTIONS, verbose=False)
            if len(coefficients) > self.raw.n_times:
                raise ValueError('Registro demasiado corto para este filtro '
                                 f'({len(coefficients)} coeficientes, {self.raw.n_times} muestras).')
            self.channels.clear()
            self.key = key

    def channel(self, name, config):
        self.prepare(config)
        if name not in self.channels:
            original = self.raw.get_data(picks=[name])
            self.channels[name] = filter_data(
                original, self.raw.info['sfreq'], config.low, config.high,
                **FILTER_OPTIONS, pad='reflect_limited', copy=True, verbose=False)[0]
        return self.channels[name]

    def prefetch(self, names, config, block=16):
        """Filter and cache several channels, reading and filtering them in blocks.

        Each row is filtered independently, exactly as `channel()` would, but one
        read and one FIR call per block avoid repeated whole-file reads. Raises on
        the first channel (in `names` order) with non-finite samples.
        """
        self.prepare(config)
        names = list(dict.fromkeys(names))
        block = max(1, int(block))
        for offset in range(0, len(names), block):
            group = names[offset:offset + block]
            # The source is read even for cached channels, as scanning always did:
            # a NaN would spread through FIR convolution beyond its source epoch.
            original = self.raw.get_data(picks=group)
            for name, row in zip(group, original):
                if not np.all(np.isfinite(row)):
                    raise ValueError(f'{name}: hay muestras no finitas. '
                                     'Usá el barrido original para localizarlas antes de filtrar.')
            missing = [index for index, name in enumerate(group) if name not in self.channels]
            if missing:
                filtered = filter_data(
                    original[missing], self.raw.info['sfreq'], config.low, config.high,
                    **FILTER_OPTIONS, pad='reflect_limited', copy=True, verbose=False)
                for index, row in zip(missing, filtered):
                    self.channels[group[index]] = row
        return {name: self.channels[name] for name in names}
