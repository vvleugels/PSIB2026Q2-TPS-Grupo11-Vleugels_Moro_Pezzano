"""Paired SD vs NS statistics on band power (group level, no clinical interpretation).

Run from the project folder:  python -m analysis.stats [--results-dir results] [--permutations 5000]

Eyes open (EO) is the PRIMARY analysis (all valid participants); eyes closed (EC) is
SECONDARY (only participants who have it). EO and EC are never pooled. Every comparison
is within subject: difference = SD - NS for the same participant.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import warnings

if __package__ in (None, ''):   # allow "python analysis/stats.py" as well as "-m"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

import config
from analysis.regions import KEYS, REGIONS, frontal_theta_beta, region_features

ANALYSES = {'open': 'primary', 'closed': 'secondary'}
ALPHA = 0.05


# ------------------------------------------------------------------------- core statistics
def bh_fdr(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN entries are ignored and stay NaN)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted
    order = np.argsort(p[valid])
    ranked = p[valid][order] * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downwards, then cap at 1.
    ranked = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    values = np.empty(m)
    values[order] = ranked
    adjusted[valid] = values
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-), zeros dropped.

    +1 means every subject increased (SD > NS), -1 every subject decreased."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((plus - minus) / (plus + minus))


def paired_test(ns: np.ndarray, sd: np.ndarray) -> dict:
    """Wilcoxon signed-rank (main) and paired t-test (sensitivity) on SD - NS.

    Returns n, mean/median difference, its 95 % t-based CI, Cohen's dz, rank-biserial r."""
    ns, sd = np.asarray(ns, dtype=float), np.asarray(sd, dtype=float)
    keep = np.isfinite(ns) & np.isfinite(sd)
    d = sd[keep] - ns[keep]
    n = d.size
    row = dict(n=n, mean_ns=np.nan, mean_sd=np.nan, mean_diff=np.nan, median_diff=np.nan,
               ci95_low=np.nan, ci95_high=np.nan, cohens_dz=np.nan, rank_biserial=np.nan,
               wilcoxon_W=np.nan, p_wilcoxon=np.nan, t=np.nan, p_ttest=np.nan)
    if n < 3:   # too few pairs for any test; keep the row so the gap is visible
        return row
    sd_d = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd_d / math.sqrt(n)
    row.update(mean_ns=ns[keep].mean(), mean_sd=sd[keep].mean(), mean_diff=d.mean(),
               median_diff=float(np.median(d)), ci95_low=d.mean() - half, ci95_high=d.mean() + half,
               cohens_dz=d.mean() / sd_d if sd_d > 0 else np.nan, rank_biserial=rank_biserial(d))
    if np.any(d != 0):
        with warnings.catch_warnings():   # scipy warns about ties / normal approximation
            warnings.simplefilter('ignore')
            w = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
            t = stats.ttest_rel(sd[keep], ns[keep])
        row.update(wilcoxon_W=float(w.statistic), p_wilcoxon=float(w.pvalue),
                   t=float(t.statistic), p_ttest=float(t.pvalue))
    return row


def add_fdr(table: pd.DataFrame, columns: tuple[str, ...] = ('p_wilcoxon', 'p_ttest')) -> pd.DataFrame:
    """Add BH-FDR columns within the table (one family = one table)."""
    table = table.copy()
    for column in columns:
        table[column.replace('p_', 'p_fdr_')] = bh_fdr(table[column].to_numpy())
    return table


def paired_frame(data: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Wide NS/SD table (one row per subject and group) keeping complete pairs only."""
    if data.empty:   # e.g. no eyes-closed data: an empty table with the expected columns
        return pd.DataFrame(columns=['sub', *by, 'NS', 'SD'])
    wide = data.pivot_table(index=['sub', *by], columns='condition', values=value, aggfunc='first')
    wide = wide.reindex(columns=['NS', 'SD'])
    return wide.dropna(subset=['NS', 'SD']).reset_index()


# --------------------------------------------------------------------------- analyses
def region_tests(regions: pd.DataFrame, value: str = 'log10_power') -> pd.DataFrame:
    """SD vs NS per eyes x band x region, FDR across bands x regions within each eyes."""
    rows = []
    for eyes, analysis in ANALYSES.items():
        subset = regions[regions['eyes'] == eyes]
        wide = paired_frame(subset, value, ['band', 'region'])
        for band in config.BANDS:
            for region in REGIONS:
                cell = wide[(wide['band'] == band) & (wide['region'] == region)]
                rows.append(dict(eyes=eyes, analysis=analysis, measure=value, band=band, region=region,
                                 **paired_test(cell.get('NS', []), cell.get('SD', []))))
    table = pd.DataFrame(rows)
    return pd.concat([add_fdr(group) for _, group in table.groupby('eyes', sort=False)], ignore_index=True)


def tbr_tests(tbr: pd.DataFrame) -> pd.DataFrame:
    """Frontal log10 theta/beta ratio, SD vs NS, per eyes condition."""
    rows = []
    for eyes, analysis in ANALYSES.items():
        wide = paired_frame(tbr[tbr['eyes'] == eyes], 'log10_tbr', [])
        rows.append(dict(eyes=eyes, analysis=analysis, measure='log10_frontal_theta_beta',
                         **paired_test(wide.get('NS', []), wide.get('SD', []))))
    # Two tests (EO, EC) are separate analyses, so no FDR across them.
    return pd.DataFrame(rows)


def session_order_tests(regions: pd.DataFrame, tbr: pd.DataFrame, order: pd.DataFrame | None) -> pd.DataFrame:
    """Does the SD - NS difference depend on which session came first? (Mann-Whitney U)."""
    if order is None or order.empty:
        return pd.DataFrame([dict(note='session order unavailable: results/session_order.csv not found')])
    order = order[['sub', 'session_order']]
    rows = []

    def compare(wide: pd.DataFrame, **labels) -> None:
        wide = wide.merge(order, on='sub', how='left')
        wide['diff'] = wide['SD'] - wide['NS']
        groups = {name: g['diff'].dropna().to_numpy() for name, g in wide.groupby('session_order')}
        a, b = groups.get('NS->SD', np.array([])), groups.get('SD->NS', np.array([]))
        row = dict(**labels, n_ns_first=a.size, n_sd_first=b.size,
                   median_diff_ns_first=np.median(a) if a.size else np.nan,
                   median_diff_sd_first=np.median(b) if b.size else np.nan, U=np.nan, p_mannwhitney=np.nan)
        if a.size >= 3 and b.size >= 3:
            test = stats.mannwhitneyu(a, b, alternative='two-sided')
            row.update(U=float(test.statistic), p_mannwhitney=float(test.pvalue))
        rows.append(row)

    for eyes, analysis in ANALYSES.items():
        wide = paired_frame(regions[regions['eyes'] == eyes], 'log10_power', ['band', 'region'])
        for band in config.BANDS:
            for region in REGIONS:
                compare(wide[(wide['band'] == band) & (wide['region'] == region)],
                        eyes=eyes, analysis=analysis, measure='log10_power', band=band, region=region)
        compare(paired_frame(tbr[tbr['eyes'] == eyes], 'log10_tbr', []),
                eyes=eyes, analysis=analysis, measure='log10_frontal_theta_beta', band='theta/beta', region='frontal')
    table = pd.DataFrame(rows)
    return pd.concat([add_fdr(g, ('p_mannwhitney',)) for _, g in table.groupby('eyes', sort=False)],
                     ignore_index=True)


def epoch_counts(features: pd.DataFrame) -> pd.DataFrame:
    """Clean/rejected epochs per session (not equalised), with a paired test of clean epochs."""
    recordings = features.drop_duplicates(KEYS)[KEYS + ['n_epochs_used', 'n_epochs_rejected']]
    rows = []
    for eyes, analysis in ANALYSES.items():
        subset = recordings[recordings['eyes'] == eyes]
        wide = paired_frame(subset, 'n_epochs_used', [])
        test = paired_test(wide.get('NS', []), wide.get('SD', []))
        for condition in ('NS', 'SD'):
            part = subset[subset['condition'] == condition]
            rows.append(dict(eyes=eyes, analysis=analysis, condition=condition, n_recordings=len(part),
                             epochs_used_mean=part['n_epochs_used'].mean(),
                             epochs_used_median=part['n_epochs_used'].median(),
                             epochs_used_min=part['n_epochs_used'].min(),
                             epochs_rejected_mean=part['n_epochs_rejected'].mean(),
                             paired_n=test['n'], paired_mean_diff_sd_minus_ns=test['mean_diff'],
                             paired_p_wilcoxon=test['p_wilcoxon']))
    return pd.DataFrame(rows)


def participants(features: pd.DataFrame, exclusions: pd.DataFrame | None) -> pd.DataFrame:
    """Which participants enter each analysis, and why the others do not."""
    rows = []
    recordings = features.drop_duplicates(KEYS)[KEYS]
    reasons = {}
    if exclusions is not None and not exclusions.empty:
        for _, r in exclusions[exclusions['level'] == 'recording'].iterrows():
            reasons[(r['sub'], r['eyes'], r['condition'])] = r['reason']
    subjects = sorted(set(recordings['sub']) | {k[0] for k in reasons})
    for eyes, analysis in ANALYSES.items():
        have = recordings[recordings['eyes'] == eyes]
        for sub in subjects:
            sessions = set(have[have['sub'] == sub]['condition'])
            missing = [c for c in ('NS', 'SD') if c not in sessions]
            if not sessions and not any((sub, eyes, c) in reasons for c in ('NS', 'SD')):
                continue   # this subject simply has no recording for this eye condition
            reason = '; '.join(f"{c}: {reasons.get((sub, eyes, c), 'no recording')}" for c in missing)
            rows.append(dict(eyes=eyes, analysis=analysis, sub=sub, included=not missing,
                             reason=reason or ''))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- cluster permutation
def montage_info(data_dir: Path, channels: list[str]):
    """mne.Info with the dataset's own electrode positions (electrodes.tsv, mm -> m)."""
    import mne
    path = next(Path(data_dir).glob('sub-*/ses-*/eeg/*_electrodes.tsv'))
    table = pd.read_csv(path, sep='\t')
    positions = {r['name']: np.array([r['x'], r['y'], r['z']], dtype=float) / 1000 for _, r in table.iterrows()}
    info = mne.create_info(channels, 500.0, 'eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos={c: positions[c] for c in channels},
                                                   coord_frame='head'))
    return info


def cluster_tests(features: pd.DataFrame, info, n_permutations: int = 5000, seed: int = 0
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-electrode SD - NS log10 power, cluster-based permutation (sign-flip) per eyes x band.

    Channels missing for a subject (bad channel in either session) are set to a 0
    difference: this is conservative (pulls toward no effect) and keeps every subject;
    the number of such imputed values is reported per channel.
    """
    import mne
    channels = info.ch_names
    adjacency, _ = mne.channels.find_ch_adjacency(info, 'eeg')
    data = features[features['abs_power_uV2'] > 0].copy()
    data['log10_power'] = np.log10(data['abs_power_uV2'])
    clusters, per_channel = [], []
    for eyes, analysis in ANALYSES.items():
        for band in config.BANDS:
            part = data[(data['eyes'] == eyes) & (data['band'] == band)]
            wide = part.pivot_table(index=['sub', 'channel'], columns='condition',
                                    values='log10_power', aggfunc='first')
            if not {'NS', 'SD'} <= set(wide.columns):
                continue
            diff = (wide['SD'] - wide['NS']).unstack('channel').reindex(columns=channels)
            # Keep subjects that have both sessions (at least one channel in common).
            diff = diff[diff.notna().any(axis=1)]
            n = len(diff)
            if n < 3:
                continue
            imputed = diff.isna().sum()
            x = diff.fillna(0.0).to_numpy()
            threshold = stats.t.ppf(1 - ALPHA / 2, n - 1)   # two-tailed cluster-forming p < .05
            t_obs, found, p_values, _ = mne.stats.permutation_cluster_1samp_test(
                x, threshold=threshold, n_permutations=n_permutations, tail=0, adjacency=adjacency,
                rng=seed, out_type='mask', verbose=False)
            membership = np.zeros(len(channels), dtype=int)
            for index, (mask, p) in enumerate(zip(found, p_values), 1):
                members = [c for c, inside in zip(channels, mask) if inside]
                membership[np.asarray(mask)] = index
                clusters.append(dict(eyes=eyes, analysis=analysis, band=band, cluster=index,
                                     sign='positive (SD > NS)' if t_obs[mask].sum() > 0 else 'negative (SD < NS)',
                                     n_channels=len(members), channels=';'.join(members),
                                     t_sum=float(t_obs[mask].sum()), p_cluster=float(p_values[index - 1]),
                                     significant=bool(p_values[index - 1] < ALPHA),
                                     n_subjects=n, n_permutations=n_permutations,
                                     cluster_threshold_t=float(threshold)))
            for i, channel in enumerate(channels):
                per_channel.append(dict(eyes=eyes, analysis=analysis, band=band, channel=channel,
                                        t=float(t_obs[i]), mean_diff=float(np.nanmean(diff[channel])),
                                        n_subjects=n, n_imputed_zero=int(imputed[channel]),
                                        cluster=int(membership[i]),
                                        cluster_p=float(p_values[membership[i] - 1]) if membership[i] else np.nan))
            if not found:
                clusters.append(dict(eyes=eyes, analysis=analysis, band=band, cluster=0, sign='',
                                     n_channels=0, channels='', t_sum=np.nan, p_cluster=np.nan,
                                     significant=False, n_subjects=n, n_permutations=n_permutations,
                                     cluster_threshold_t=float(threshold)))
    return pd.DataFrame(clusters), pd.DataFrame(per_channel)


# ---------------------------------------------------------------------------------- main
def load(results_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Read the batch outputs; optional files become None."""
    def read(name: str, required: bool = False) -> pd.DataFrame | None:
        path = results_dir / name
        if not path.is_file():
            if required:
                raise FileNotFoundError(f'{path} not found; run python -m batch.run_batch first')
            return None
        return pd.read_csv(path)
    return dict(features=read('features_long.csv', True), theta_beta=read('theta_beta.csv', True),
                exclusions=read('exclusions.csv'), session_order=read('session_order.csv'))


def run_all(results_dir: Path, data_dir: Path, n_permutations: int = 5000, clusters: bool = True
            ) -> dict[str, pd.DataFrame]:
    """Compute every table and write results/features_region.csv and results/stats_*.csv."""
    inputs = load(results_dir)
    features = inputs['features']
    regions = region_features(features)
    tbr = frontal_theta_beta(inputs['theta_beta'])
    tables = {
        'features_region': regions,
        'stats_region_log_power': region_tests(regions, 'log10_power'),
        'stats_region_rel_power': region_tests(regions, 'rel_power_pct'),
        'stats_frontal_theta_beta': tbr_tests(tbr),
        'stats_session_order': session_order_tests(regions, tbr, inputs['session_order']),
        'stats_epochs': epoch_counts(features),
        'stats_participants': participants(features, inputs['exclusions']),
    }
    if clusters:
        channels = montage_channels(features)
        info = montage_info(data_dir, channels)
        tables['cluster_results'], tables['stats_cluster_channels'] = cluster_tests(features, info, n_permutations)
    for name, table in tables.items():
        table.to_csv(results_dir / f'{name}.csv', index=False)
    return tables


def montage_channels(features: pd.DataFrame) -> list[str]:
    """Channels present in the features, grouped by region (a stable order for outputs)."""
    present = set(features['channel'])
    return [c for chans in REGIONS.values() for c in chans if c in present]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='SD vs NS paired statistics on band power.')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR)
    parser.add_argument('--data-dir', type=Path, default=config.DATA_DIR,
                        help='BIDS folder, used only for electrode positions (cluster test)')
    parser.add_argument('--permutations', type=int, default=5000)
    parser.add_argument('--no-clusters', action='store_true', help='skip the per-electrode cluster test')
    args = parser.parse_args(argv)
    tables = run_all(args.results_dir, args.data_dir, args.permutations, not args.no_clusters)
    for name, table in tables.items():
        print(f'{name}.csv: {len(table)} rows')
    return 0


if __name__ == '__main__':
    sys.exit(main())
