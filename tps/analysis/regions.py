"""Channel-to-region mapping for the ds004902 montage and region-level features.

How the mapping was built
-------------------------
The 61 channel names were read from the BIDS ``*_channels.tsv`` sidecars; all 218
recordings list exactly the same 61 EEG channels (Brain Products 10-10 layout; FCz is the
online reference and AFz the ground, so neither is in the data). Each name was assigned
by its 10-10 prefix, which encodes the scalp area:

* frontal   Fp*, AF*, F*  (pre-frontal, anterior-frontal and frontal rows)
* central   FC*, C*, CP*  (fronto-central, central and centro-parietal rows, midline to C5/C6)
* temporal  FT*, T*, TP*  (the lateral temporal chain, incl. mastoid-adjacent TP9/TP10)
* parietal  P*            (parietal row)
* occipital PO*, O*       (parieto-occipital and occipital rows)

The order of the prefix tests matters (e.g. "FT7" must go to temporal before "F*").
``tests/test_stats.py`` checks that every channel of the real montage is mapped once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGIONS: dict[str, tuple[str, ...]] = {
    'frontal': ('Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'),
    'central': ('FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
                'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'),
    'temporal': ('FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'TP9', 'TP10'),
    'parietal': ('Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'),
    'occipital': ('POz', 'PO3', 'PO4', 'PO7', 'PO8', 'Oz', 'O1', 'O2'),
}
CHANNEL_REGION: dict[str, str] = {ch: region for region, chans in REGIONS.items() for ch in chans}

# A region value is only computed when at least this fraction of its channels survived
# screening. Bad channels (mostly frontal, from blinks) are missing rows in
# features_long.csv; averaging over 2 of 16 frontal sites would not represent the region.
MIN_CHANNEL_FRACTION = 0.5

KEYS = ['sub', 'ses', 'condition', 'eyes']


def channels_from_bids(data_dir: Path) -> list[str]:
    """Channel names of the first *_channels.tsv found (all recordings share the montage)."""
    path = next(Path(data_dir).glob('sub-*/ses-*/eeg/*_channels.tsv'))
    with path.open(encoding='utf-8') as stream:
        rows = [line.rstrip('\n').split('\t') for line in stream]
    header = rows[0]
    kind = header.index('type') if 'type' in header else None
    return [r[0] for r in rows[1:] if r and (kind is None or r[kind].upper() == 'EEG')]


def region_features(features: pd.DataFrame,
                    min_fraction: float = MIN_CHANNEL_FRACTION) -> pd.DataFrame:
    """Mean log10 absolute power and mean relative power per sub/ses/eyes/band/region.

    Averaging log10 power (= log of the geometric mean) keeps one high-power channel from
    dominating the region, and log-power differences are closer to normal for the t-test.
    """
    data = features[features['channel'].isin(CHANNEL_REGION)].copy()
    data['region'] = data['channel'].map(CHANNEL_REGION)
    data = data[data['abs_power_uV2'] > 0]
    data['log10_power'] = np.log10(data['abs_power_uV2'])
    grouped = data.groupby(KEYS + ['band', 'region'], observed=True)
    out = grouped.agg(log10_power=('log10_power', 'mean'), rel_power_pct=('rel_power_pct', 'mean'),
                      n_channels=('channel', 'nunique'), n_epochs_used=('n_epochs_used', 'first'),
                      n_epochs_rejected=('n_epochs_rejected', 'first')).reset_index()
    out['n_channels_region'] = out['region'].map({r: len(c) for r, c in REGIONS.items()})
    keep = out['n_channels'] >= np.ceil(min_fraction * out['n_channels_region'])
    return out[keep].reset_index(drop=True)


def frontal_theta_beta(theta_beta: pd.DataFrame,
                       min_fraction: float = MIN_CHANNEL_FRACTION) -> pd.DataFrame:
    """Frontal theta/beta per recording as the mean log10 ratio over frontal channels."""
    data = theta_beta[theta_beta['channel'].isin(REGIONS['frontal'])]
    data = data[pd.to_numeric(data['ratio'], errors='coerce') > 0].copy()
    data['log10_tbr'] = np.log10(data['ratio'].astype(float))
    out = data.groupby(KEYS, observed=True).agg(log10_tbr=('log10_tbr', 'mean'),
                                                n_channels=('channel', 'nunique')).reset_index()
    return out[out['n_channels'] >= np.ceil(min_fraction * len(REGIONS['frontal']))].reset_index(drop=True)
