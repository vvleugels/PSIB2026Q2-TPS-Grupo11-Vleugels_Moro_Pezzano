"""Shared analysis configuration: the single source of truth for the batch pipeline.

Teammates should import these values instead of copying numbers, e.g.
``from config import FILTER, SPECTRAL, BANDS``. The objects are the same classes
the GUI uses, so batch results and GUI results are computed identically.
"""
from __future__ import annotations

from pathlib import Path

from eeg_viewer.processing import FilterConfig
from eeg_viewer.screening import ScreeningConfig
from eeg_viewer.spectral import SpectralConfig

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'

# Band-pass filter: the GUI default (0.2-45 Hz, zero-phase FIR, Hamming, firwin),
# always applied to the whole continuous channel before epoching so that epoch
# edges do not get their own filter transients.
FILTER = FilterConfig(enabled=True, low=0.2, high=45.0)

# Epochs: 5 s, non-overlapping, starting at sample 0. Only complete epochs are used;
# the trailing partial epoch (if any) is discarded. The length is fixed by
# eeg_viewer.annotations.make_epochs, which the GUI also uses.
EPOCH_SECONDS = 5.0

# Welch: 2 s Hann (periodic) segments, 50 % overlap, density scaling, mean average.
# A 5 s epoch therefore yields 4 segments and a 0.5 Hz frequency resolution.
SPECTRAL = SpectralConfig(seconds=2.0, overlap=50.0, edges=(0.5, 4.0, 8.0, 13.0, 30.0))

# Band names in output files, in the same order as SPECTRAL.edges.
BANDS = ('delta', 'theta', 'alpha', 'beta')
# Relative power is each band divided by the total 0.5-30 Hz power (edges[0]..edges[-1]).

# Peak-to-peak amplitude screening per channel and epoch.
# These are STARTING values taken from the GUI defaults; they have NOT been validated
# against visual inspection for ds004902. "Accepted" only means "within these limits".
SCREENING = ScreeningConfig(high_uv=150.0, flat_uv=1.0)

# A channel flagged in more than this fraction of the complete epochs of a recording
# is treated as a bad channel for that recording: it is excluded (no feature rows) and
# not used to reject epochs. Without this step a single noisy electrode (e.g. TP9 in
# sub-01 ses-1 fails 31/60 epochs) would discard half the recording for all 61 channels.
# Set to None to reproduce the strict GUI rule (any flagged channel rejects the epoch).
BAD_CHANNEL_MAX_FRACTION: float | None = 0.30

# Minimum clean epochs per recording: 12 epochs = 60 s of clean EEG (20 % of a 300 s
# recording). With 2 s / 50 % Welch this averages 48 segments per channel. Resting-state
# band-power reliability studies report that ~40-60 s of artifact-free data is enough
# for stable estimates, so recordings below this are excluded rather than reported
# with very noisy spectra.
MIN_CLEAN_EPOCHS = 12

SESSIONS = {'ses-1': 'NS', 'ses-2': 'SD'}
EYES = {'eyesopen': 'open', 'eyesclosed': 'closed'}


def as_dict() -> dict:
    """Plain, JSON-serialisable view of the configuration (saved next to outputs)."""
    return dict(
        filter=FILTER.context(),
        epoch_seconds=EPOCH_SECONDS,
        complete_epochs_only=True,
        welch=dict(seconds=SPECTRAL.seconds, overlap_percent=SPECTRAL.overlap,
                   window='hann (periodic)', scaling='density', average='mean'),
        bands={name: [low, high] for name, low, high in
               zip(BANDS, SPECTRAL.edges, SPECTRAL.edges[1:])},
        relative_power_reference_hz=[SPECTRAL.edges[0], SPECTRAL.edges[-1]],
        screening=dict(high_uv=SCREENING.high_uv, flat_uv=SCREENING.flat_uv,
                       measure='peak-to-peak per channel and epoch, on the filtered signal',
                       validated=False),
        bad_channel_max_fraction=BAD_CHANNEL_MAX_FRACTION,
        min_clean_epochs=MIN_CLEAN_EPOCHS,
        sessions=SESSIONS,
        eyes=EYES,
    )
