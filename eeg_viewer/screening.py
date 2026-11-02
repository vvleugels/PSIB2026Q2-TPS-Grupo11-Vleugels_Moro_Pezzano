"""Read-only amplitude screening, independent of review decisions."""
from dataclasses import dataclass
import math

import numpy as np

from .annotations import make_epochs
from .processing import FilterConfig


@dataclass(frozen=True)
class ScreeningConfig:
    high_uv: float = 150.0
    flat_uv: float = 1.0

    def validate(self):
        if not (math.isfinite(self.high_uv) and math.isfinite(self.flat_uv)
                and 0 <= self.flat_uv < self.high_uv):
            raise ValueError('Los umbrales deben cumplir 0 ≤ plana < alta, en µV pico a pico.')


def scan_recording(recording, channels, config, filter_config=FilterConfig()):
    """Flag a whole epoch if any selected channel crosses a strict p-p limit.

    Filtering uses the existing continuous-channel cache before slicing epochs.
    Without filtering, read one epoch at a time. Partial epochs are included, with
    fewer than two samples reported separately instead of being called flat.
    Nonfinite samples are reported as data problems, never as clean EEG.
    """
    config.validate()
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or any(
            name not in recording.channels for name in channels):
        raise ValueError('Seleccioná al menos un electrodo EEG válido, sin duplicados.')
    continuous = None
    if filter_config.enabled:
        filter_config.validate(recording.sfreq)
        continuous = {}
        for channel in channels:
            # A NaN would spread through FIR convolution beyond its source epoch.
            if not np.all(np.isfinite(recording.raw.get_data(picks=[channel]))):
                raise ValueError(f'{channel}: hay muestras no finitas. '
                                 'Usá el barrido original para localizarlas antes de filtrar.')
            continuous[channel] = recording.filter.channel(channel, filter_config)
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    findings = {}
    for epoch in epochs:
        values = ([continuous[channel][epoch.first:epoch.last] for channel in channels]
                  if continuous is not None else recording.raw.get_data(
                      picks=list(channels), start=epoch.first, stop=epoch.last))
        issues = []
        for channel, samples in zip(channels, values):
            if not np.all(np.isfinite(samples)):
                kind, amplitude = 'No finitos', None
            elif len(samples) < 2:
                kind, amplitude = 'Tramo insuficiente', None
            else:
                amplitude = float(np.ptp(samples) * 1e6)
                if amplitude > config.high_uv:
                    kind = 'Alta'
                elif amplitude < config.flat_uv:
                    kind = 'Plana'
                else:
                    continue
            issues.append(dict(channel=channel, kind=kind, peak_to_peak_uv=amplitude))
        if issues:
            findings[epoch.index] = issues
    return dict(findings=findings, channels=channels, config=config, filter=filter_config,
                total=len(epochs), partial=sum(epoch.partial for epoch in epochs))
