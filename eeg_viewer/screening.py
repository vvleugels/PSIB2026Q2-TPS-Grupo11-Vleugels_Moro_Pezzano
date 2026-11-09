"""Read-only amplitude screening, independent of review decisions."""
from dataclasses import dataclass
import math

import numpy as np

from .annotations import make_epochs
from .processing import FilterConfig

SCAN_BLOCK = 16  # channels read (or sliced) together while scanning


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

    Filtering fills the continuous-channel cache in blocks before slicing epochs.
    Without filtering, read whole channels in blocks of SCAN_BLOCK. Partial epochs
    are included, with fewer than two samples reported separately instead of being
    called flat.
    Nonfinite samples are reported as data problems, never as clean EEG.
    """
    config.validate()
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or any(
            name not in recording.channels for name in channels):
        raise ValueError('Seleccioná al menos un electrodo EEG válido, sin duplicados.')
    if filter_config.active:
        filter_config.validate(recording.sfreq)
        continuous = recording.filter.prefetch(channels, filter_config)

        def blocks():
            for offset in range(0, len(channels), SCAN_BLOCK):
                names = channels[offset:offset + SCAN_BLOCK]
                yield names, [continuous[name] for name in names]
    else:
        def blocks():
            # One whole-length read per block of channels bounds memory use.
            for offset in range(0, len(channels), SCAN_BLOCK):
                names = channels[offset:offset + SCAN_BLOCK]
                yield names, recording.raw.get_data(picks=list(names))
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    issues_by_epoch = {epoch.index: [] for epoch in epochs}
    for names, rows in blocks():
        rows = np.asarray(rows)
        for epoch in epochs:
            values = rows[:, epoch.first:epoch.last]
            finite = np.all(np.isfinite(values), axis=1)
            amplitudes = (np.ptp(values, axis=1) * 1e6 if values.shape[1] >= 2
                          else np.full(len(names), np.nan))
            issues = issues_by_epoch[epoch.index]
            for channel, is_finite, amplitude in zip(names, finite, amplitudes):
                if not is_finite:
                    kind, amplitude = 'No finitos', None
                elif values.shape[1] < 2:
                    kind, amplitude = 'Tramo insuficiente', None
                else:
                    amplitude = float(amplitude)
                    if amplitude > config.high_uv:
                        kind = 'Alta'
                    elif amplitude < config.flat_uv:
                        kind = 'Plana'
                    else:
                        continue
                issues.append(dict(channel=channel, kind=kind, peak_to_peak_uv=amplitude))
    findings = {index: issues for index, issues in issues_by_epoch.items() if issues}
    return dict(findings=findings, channels=channels, config=config, filter=filter_config,
                total=len(epochs), partial=sum(epoch.partial for epoch in epochs))
