"""Reproducible continuous filtering, without changing the source recording."""
from dataclasses import asdict, dataclass
import math

from mne.filter import create_filter, filter_data
from scipy.signal import iirnotch, filtfilt


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    low: float = 0.2
    high: float = 45.0
    notch: float | None = None

    @property
    def active(self):
        return self.enabled or self.notch is not None

    @property
    def label(self):
        parts = [f'{self.low:g}–{self.high:g} Hz'] if self.enabled else []
        if self.notch is not None:
            parts.append(f'Notch {self.notch:g} Hz')
        return 'Filtrada · ' + ' · '.join(parts) if parts else 'Original'

    def validate(self, sfreq):
        if self.notch is not None and (self.notch not in (50, 60) or self.notch >= sfreq / 2):
            raise ValueError('El notch debe ser de 50 o 60 Hz y menor que la frecuencia de Nyquist.')
        if not (math.isfinite(self.low) and math.isfinite(self.high)
                and 0 < self.low < self.high < sfreq / 2):
            raise ValueError(f'Los cortes deben cumplir 0 < inferior < superior < {sfreq / 2:g} Hz.')

    def context(self):
        import mne
        return {**asdict(self), **FILTER_OPTIONS, 'pad': 'reflect_limited',
                'mne_version': mne.__version__,
                'notch_method': 'iirnotch', 'notch_q': 30., 'notch_phase': 'zero',
                'filter_order': 'bandpass_then_notch'}


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
        key = (config.enabled, config.low, config.high, config.notch)
        if config.notch is not None and self.raw.n_times <= 9:
            raise ValueError('Registro demasiado corto para el notch (requiere más de 9 muestras).')
        if key != self.key and config.enabled:
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
        if key != self.key:
            self.channels.clear()
            self.key = key

    def channel(self, name, config):
        self.prepare(config)
        if not config.active:
            return self.raw.get_data(picks=[name])[0]
        if name not in self.channels:
            values = self.raw.get_data(picks=[name])[0]
            if config.enabled:
                values = filter_data(
                    values, self.raw.info['sfreq'], config.low, config.high,
                    **FILTER_OPTIONS, pad='reflect_limited', copy=True, verbose=False)
            if config.notch is not None:
                b, a = iirnotch(config.notch, Q=30., fs=self.raw.info['sfreq'])
                values = filtfilt(b, a, values, padlen=9)
            self.channels[name] = values
        return self.channels[name]
