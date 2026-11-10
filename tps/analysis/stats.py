"""SD vs NS comparison of band power: one summary table (group level, no clinical interpretation).

Run from the project folder:  python -m tps.analysis.stats [--results-dir results]

Eyes open is the main analysis (all valid participants); eyes closed is secondary (only the
participants who have it). They are never pooled. Every comparison is within subject:
difference = SD - NS for the same participant, tested with a paired t-test on log10 power.
The change is reported in dB (10 x log10, the unit of the reference paper) and in % of power.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from tps import config
from tps.analysis.regions import REGIONS, frontal_theta_beta, region_features, region_table

ANALYSES = {'open': 'primary', 'closed': 'secondary'}
ALPHA = 0.05
SUMMARY_COLUMNS = ['eyes', 'analysis', 'band', 'region', 'n', 'mean_ns_db', 'mean_sd_db', 'change_db',
                   'change_pct', 'ci95_low_pct', 'ci95_high_pct', 'p', 'significant']
# Outputs of earlier versions (removed analyses); deleted so they are never shown as current.
OBSOLETE_OUTPUTS = ('stats_region_log_power', 'stats_region_rel_power', 'stats_relative_vs_absolute',
                    'stats_region_log_power_min20', 'stats_frontal_theta_beta', 'stats_session_order',
                    'stats_epochs', 'stats_participants', 'cluster_results', 'stats_cluster_channels')


# ------------------------------------------------------------------------- core statistics
def db(log10_value: float) -> float:
    """log10 of a power -> dB (10 x log10)."""
    return 10 * log10_value


def pct(log10_ratio: float) -> float:
    """Mean log10 difference -> % change of power (geometric-mean ratio SD / NS)."""
    return (10 ** log10_ratio - 1) * 100


def paired_test(ns: np.ndarray, sd: np.ndarray) -> dict:
    """Paired t-test on SD - NS of log10 values, with the change in dB and in %.

    Returns n, the session means (dB), the mean change (dB and %), the 95 % CI of the change
    (in %, from the t-based CI of the mean log10 difference) and the two-sided p. With fewer
    than 3 complete pairs only the descriptive values are filled in."""
    ns, sd = np.asarray(ns, dtype=float), np.asarray(sd, dtype=float)
    keep = np.isfinite(ns) & np.isfinite(sd)
    d = sd[keep] - ns[keep]
    n = int(d.size)
    row = dict(n=n, mean_ns_db=np.nan, mean_sd_db=np.nan, change_db=np.nan, change_pct=np.nan,
               ci95_low_pct=np.nan, ci95_high_pct=np.nan, p=np.nan, significant=False)
    if n:
        row.update(mean_ns_db=db(ns[keep].mean()), mean_sd_db=db(sd[keep].mean()),
                   change_db=db(d.mean()), change_pct=pct(d.mean()))
    if n < 3:
        return row
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    row.update(ci95_low_pct=pct(d.mean() - half), ci95_high_pct=pct(d.mean() + half))
    if np.any(d != 0):
        p = float(stats.ttest_rel(sd[keep], ns[keep]).pvalue)
        row.update(p=p, significant=bool(p < ALPHA))
    return row


def paired_frame(data: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Wide NS/SD table (one row per subject and group) keeping complete pairs only."""
    if data.empty:   # e.g. no eyes-closed data: an empty table with the expected columns
        return pd.DataFrame(columns=['sub', *by, 'NS', 'SD'])
    wide = data.pivot_table(index=['sub', *by], columns='condition', values=value, aggfunc='first')
    wide = wide.reindex(columns=['NS', 'SD'])
    return wide.dropna(subset=['NS', 'SD']).reset_index()


# --------------------------------------------------------------------------- analysis
def summary_table(regions: pd.DataFrame, tbr: pd.DataFrame) -> pd.DataFrame:
    """One row per eyes x band x region (log10 absolute power), plus frontal theta/beta per eyes."""
    rows = []
    for eyes, analysis in ANALYSES.items():
        if not (regions['eyes'] == eyes).any():
            continue   # e.g. a run with --eyes open: no empty eyes-closed rows
        wide = paired_frame(regions[regions['eyes'] == eyes], 'log10_power', ['band', 'region'])
        for band in config.BANDS:
            for region in REGIONS:
                cell = wide[(wide['band'] == band) & (wide['region'] == region)]
                rows.append(dict(eyes=eyes, analysis=analysis, band=band, region=region,
                                 **paired_test(cell.get('NS', []), cell.get('SD', []))))
        ratio = paired_frame(tbr[tbr['eyes'] == eyes], 'log10_tbr', [])
        rows.append(dict(eyes=eyes, analysis=analysis, band='theta/beta', region='frontal',
                         **paired_test(ratio.get('NS', []), ratio.get('SD', []))))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------------- main
def load(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the batch outputs this step needs."""
    def read(name: str) -> pd.DataFrame:
        path = results_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; run python -m tps.pipeline.batch first')
        return pd.read_csv(path)
    return dict(features=read('features_long.csv'), theta_beta=read('theta_beta.csv'))


def run_all(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Write results/features_region.csv, channel_regions.csv and stats_summary.csv."""
    inputs = load(results_dir)
    regions = region_features(inputs['features'])
    tables = {
        'channel_regions': region_table(),
        'features_region': regions,
        'stats_summary': summary_table(regions, frontal_theta_beta(inputs['theta_beta'])),
    }
    for name, table in tables.items():
        table.to_csv(results_dir / f'{name}.csv', index=False)
    for name in OBSOLETE_OUTPUTS:
        (results_dir / f'{name}.csv').unlink(missing_ok=True)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='SD vs NS paired comparison of band power.')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)
    for name, table in run_all(args.results_dir).items():
        print(f'{name}.csv: {len(table)} rows')
    return 0


if __name__ == '__main__':
    sys.exit(main())
