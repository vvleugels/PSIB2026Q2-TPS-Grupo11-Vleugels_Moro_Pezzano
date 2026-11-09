"""Synthetic tables with the batch output schema, for testing the statistics offline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tps.analysis.regions import REGIONS

BANDS = ('delta', 'theta', 'alpha', 'beta')
BASE_LOG10 = {'delta': 1.2, 'theta': 0.8, 'alpha': 1.0, 'beta': 0.4}


def make_features(n_subjects: int = 30, shift: dict[tuple[str, str], float] | None = None,
                  eyes: tuple[str, ...] = ('open',), seed: int = 0,
                  noise: float = 0.15, regions: dict[str, tuple[str, ...]] = REGIONS) -> pd.DataFrame:
    """features_long-like table: log10 power = base + subject effect + channel noise (+ shift in SD).

    ``shift`` maps (band, region) -> log10 difference added to every SD channel of that region.
    Subject effects are shared by both sessions, so the design is truly paired.
    """
    rng = np.random.default_rng(seed)
    shift = shift or {}
    rows = []
    for eye in eyes:
        for s in range(1, n_subjects + 1):
            subject = f'sub-{s:02d}'
            person = {b: rng.normal(0, 0.3) for b in BANDS}   # between-subject differences
            for ses, condition in (('ses-1', 'NS'), ('ses-2', 'SD')):
                n_used = int(rng.integers(30, 55))
                for region, channels in regions.items():
                    for channel in channels:
                        logs = {b: BASE_LOG10[b] + person[b] + rng.normal(0, noise)
                                + (shift.get((b, region), 0.0) if condition == 'SD' else 0.0) for b in BANDS}
                        powers = {b: 10 ** v for b, v in logs.items()}
                        total = sum(powers.values())
                        for band in BANDS:
                            rows.append(dict(sub=subject, ses=ses, condition=condition, eyes=eye,
                                             channel=channel, band=band, abs_power_uV2=powers[band],
                                             rel_power_pct=100 * powers[band] / total,
                                             n_epochs_used=n_used, n_epochs_rejected=60 - n_used))
    return pd.DataFrame(rows)


def make_psd(features: pd.DataFrame, max_hz: float = 45.0) -> pd.DataFrame:
    """psd_channels.csv-like table: 1/f background scaled by each recording's band powers."""
    frequencies = np.arange(0, max_hz + 0.25, 0.5)
    edges = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    keys = ['sub', 'ses', 'condition', 'eyes', 'channel']
    wide = features.pivot_table(index=keys, columns='band', values='abs_power_uV2').reset_index()
    rows = []
    for _, r in wide.iterrows():
        psd = 10 / (1 + frequencies)
        for band, (low, high) in edges.items():
            inside = (frequencies >= low) & (frequencies < high)
            psd[inside] *= r[band] / BASE_POWER[band]
        rows.append([*(r[k] for k in keys), *psd])
    return pd.DataFrame(rows, columns=[*keys, *[f'{f:g}' for f in frequencies]])


BASE_POWER = {b: 10 ** v for b, v in BASE_LOG10.items()}


def theta_beta_from(features: pd.DataFrame) -> pd.DataFrame:
    """theta_beta.csv-like table derived from a features table."""
    wide = features.pivot_table(index=['sub', 'ses', 'condition', 'eyes', 'channel'], columns='band',
                                values='abs_power_uV2').reset_index()
    wide['ratio'] = wide['theta'] / wide['beta']
    return wide[['sub', 'ses', 'condition', 'eyes', 'channel', 'ratio']]
