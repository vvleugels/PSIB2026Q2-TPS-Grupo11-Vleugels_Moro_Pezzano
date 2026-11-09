"""Result figures for the SD vs NS comparison, saved to results/figures/ with captions.

Run from the project folder:  python -m tps.analysis.figures [--results-dir results]

Every function returns the matplotlib Figure and saves a PNG; make_all also writes captions.md
and results/report.md (tps.analysis.report).
Figures show group-level spectral differences only; no clinical interpretation is implied.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from tps import config
from tps.analysis.hypotheses import evaluate
from tps.analysis.regions import CHANNEL_REGION, MIN_CHANNEL_FRACTION, REGIONS, frontal_theta_beta

# One color per condition everywhere (same teal/rose as the GUI's spectral plot).
COLORS = {'NS': '#0e7c86', 'SD': '#b94b6b'}
LABELS = {'NS': 'NS (normal sleep)', 'SD': 'SD (sleep deprivation)'}
EYES_TITLE = {'open': 'Eyes open (primary)', 'closed': 'Eyes closed (secondary)'}
BAND_LIMITS = dict(zip(config.BANDS, zip(config.SPECTRAL.edges, config.SPECTRAL.edges[1:])))
FONT = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10}
DPI = 150


def _figure(width: float, height: float) -> Figure:
    """Figure without pyplot, so figures can be built in tests/threads without a display."""
    import matplotlib
    matplotlib.rcParams.update(FONT)
    return Figure(figsize=(width, height), layout='constrained')


def _save(fig: Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=DPI)
    return path


def _ci95(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and t-based 95 % CI along an axis (NaNs ignored)."""
    mean = np.nanmean(values, axis=axis)
    n = np.sum(np.isfinite(values), axis=axis)
    sem = np.nanstd(values, axis=axis, ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sem
    return mean, mean - half, mean + half


def _p_label(p: float) -> str:
    return 'p = n/a' if p is None or not np.isfinite(p) else ('p < .001' if p < 0.001 else f'p = {p:.3f}')


# --------------------------------------------------------------------------- topomaps
def standard_info(channels: list[str]):
    """mne.Info with an MNE standard 10-05 montage that contains the dataset's channel names."""
    import mne
    info = mne.create_info(channels, 500.0, 'eeg')   # sfreq is irrelevant for a topomap
    available = mne.channels.get_builtin_montages()
    # 'standard_1005' was renamed 'colin27_1005' in recent MNE; use whichever exists.
    name = 'colin27_1005' if 'colin27_1005' in available else 'standard_1005'
    info.set_montage(mne.channels.make_standard_montage(name))
    return info


# 'absolute': log10 absolute power (µV²); 'relative': relative power (% of 0.5-30 Hz).
MEASURES = {
    'absolute': dict(value='log10_power', stats='region_stats', suffix='',
                     unit='SD − NS (Δ log10 µV²)', axis='log10 power (µV²)', what='band power'),
    'relative': dict(value='rel_power_pct', stats='rel_stats', suffix='_relative',
                     unit='SD − NS (Δ percentage points)', axis='relative power (%)',
                     what='relative band power'),
}


def channel_differences(features: pd.DataFrame, eyes: str, measure: str = 'absolute') -> pd.DataFrame:
    """Per band x channel: mean SD - NS (log10 power or relative %) over participants with both sessions."""
    data = features[(features['eyes'] == eyes) & (features['abs_power_uV2'] > 0)].copy()
    data['log10_power'] = np.log10(data['abs_power_uV2'])
    data['rel_power_pct'] = pd.to_numeric(data['rel_power_pct'], errors='coerce')
    wide = data.pivot_table(index=['sub', 'band', 'channel'], columns='condition',
                            values=MEASURES[measure]['value'], aggfunc='first').reindex(columns=['NS', 'SD'])
    wide['diff'] = wide['SD'] - wide['NS']
    return wide['diff'].dropna().groupby(['band', 'channel']).agg(['mean', 'count']).reset_index()


def significant_channels(clusters: pd.DataFrame | None, eyes: str, band: str) -> set[str]:
    """Channels belonging to a significant cluster (cluster_results.csv) for eyes x band."""
    if clusters is None or clusters.empty:
        return set()
    rows = clusters[(clusters['eyes'] == eyes) & (clusters['band'] == band)
                    & clusters['significant'].astype(bool)]
    return {c for text in rows['channels'].dropna() for c in str(text).split(';') if c}


def plot_topomaps(features: pd.DataFrame, clusters: pd.DataFrame | None, eyes: str,
                  out_dir: Path, measure: str = 'absolute') -> Figure:
    """SD - NS per electrode, one map per band; white dots = significant cluster.

    Clusters come from the test on log10 absolute power, so relative maps have no markers."""
    import mne
    spec = MEASURES[measure]
    if measure != 'absolute':
        clusters = None
    diffs = channel_differences(features, eyes, measure)
    channels = [c for c in CHANNEL_REGION if c in set(diffs['channel'])]
    info = standard_info(channels)
    # One symmetric colour scale for all bands, so maps are comparable.
    limit = float(np.nanmax(np.abs(diffs['mean']))) if len(diffs) else 0.1
    fig = _figure(4 * len(config.BANDS) + 1, 4.4)
    axes = fig.subplots(1, len(config.BANDS))
    image = None
    for ax, band in zip(np.atleast_1d(axes), config.BANDS):
        values = diffs[diffs['band'] == band].set_index('channel')['mean'].reindex(channels).to_numpy()
        marked = significant_channels(clusters, eyes, band)
        mask = np.array([c in marked for c in channels])
        image, _ = mne.viz.plot_topomap(
            np.nan_to_num(values), info, axes=ax, show=False, cmap='RdBu_r', vlim=(-limit, limit),
            mask=mask, mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                                        markersize=6), contours=4, sensors=True)
        low, high = BAND_LIMITS[band]
        title = f'{band.capitalize()} ({low:g}–{high:g} Hz)'
        # The cluster test runs on absolute power only; relative maps carry no cluster count.
        ax.set_title(title + (f'\n{int(mask.sum())} electrodes in significant clusters'
                              if clusters is not None else ''))
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.75, location='right')
        bar.set_label(spec['unit'])
    n = int(diffs['count'].max()) if len(diffs) else 0
    fig.suptitle(f"{EYES_TITLE[eyes]}: change in {spec['what']} after SD (n = {n})", fontsize=13)
    _save(fig, out_dir, f"topomap{spec['suffix']}_{eyes}")
    return fig


# ------------------------------------------------------------------------------- PSDs
def region_psd(psd: pd.DataFrame, eyes: str) -> tuple[np.ndarray, dict]:
    """Per region and condition: subjects x frequencies of mean log10 PSD (paired subjects only)."""
    meta = ['sub', 'ses', 'condition', 'eyes', 'channel']
    freq_columns = [c for c in psd.columns if c not in meta]
    frequencies = np.array([float(c) for c in freq_columns])
    data = psd[psd['eyes'] == eyes].copy()
    data['region'] = data['channel'].map(CHANNEL_REGION)
    values = np.log10(np.clip(data[freq_columns].to_numpy(dtype=float), 1e-12, None))
    logs = pd.DataFrame(values, columns=freq_columns, index=data.index)
    logs[['sub', 'condition', 'region', 'channel']] = data[['sub', 'condition', 'region', 'channel']]
    out = {}
    for region, channels in REGIONS.items():
        part = logs[logs['region'] == region]
        counts = part.groupby(['sub', 'condition'])['channel'].nunique()
        # Same rule as features_region.csv: at least half of the region's channels.
        valid = counts[counts >= math.ceil(MIN_CHANNEL_FRACTION * len(channels))].index
        means = part.groupby(['sub', 'condition'])[freq_columns].mean().loc[valid]
        subjects = sorted(set.intersection(*[{s for s, c in valid if c == cond} for cond in ('NS', 'SD')]))
        out[region] = {cond: means.loc[[(s, cond) for s in subjects]].to_numpy() for cond in ('NS', 'SD')}
    return frequencies, out


def plot_psd_by_region(psd: pd.DataFrame, eyes: str, out_dir: Path) -> Figure:
    """Group-average PSD (geometric mean) with 95 % CI per condition, one panel per region."""
    frequencies, regions = region_psd(psd, eyes)
    fig = _figure(20, 4.6)
    axes = fig.subplots(1, len(REGIONS), sharey=True)
    for ax, (region, data) in zip(axes, regions.items()):
        for cond in ('NS', 'SD'):
            values = data[cond]
            if len(values) < 2:
                continue
            mean, low, high = _ci95(values)
            # Averaging log10 PSD = geometric mean; plotted back on a log axis in µV²/Hz.
            ax.plot(frequencies, 10 ** mean, color=COLORS[cond], lw=1.8, label=LABELS[cond])
            ax.fill_between(frequencies, 10 ** low, 10 ** high, color=COLORS[cond], alpha=0.25, lw=0)
        for edge in config.SPECTRAL.edges:
            ax.axvline(edge, color='0.75', lw=0.8, ls='--', zorder=0)
        ax.set_yscale('log')
        # Start at the first band edge: below it the 0.2 Hz high-pass dominates the spectrum.
        ax.set_xlim(config.SPECTRAL.edges[0], frequencies[-1])
        ax.set_title(f'{region.capitalize()} (n = {len(data["NS"])})')
        ax.set_xlabel('Frequency (Hz)')
        ax.grid(alpha=0.2, which='both')
    axes[0].set_ylabel('PSD (µV²/Hz)')
    axes[0].legend(loc='upper right')
    fig.suptitle(f'{EYES_TITLE[eyes]}: group-average PSD, shaded = 95 % CI; dashed = band edges', fontsize=13)
    _save(fig, out_dir, f'psd_regions_{eyes}')
    return fig


# ------------------------------------------------------------------------- paired plots
def plot_paired(regions: pd.DataFrame, region_stats: pd.DataFrame | None, eyes: str,
                out_dir: Path, measure: str = 'absolute') -> Figure:
    """Per band x region: one grey line per participant (NS -> SD) plus group mean ± 95 % CI."""
    spec = MEASURES[measure]
    data = regions[regions['eyes'] == eyes]
    fig = _figure(17, 13)
    axes = fig.subplots(len(config.BANDS), len(REGIONS), squeeze=False)
    for i, band in enumerate(config.BANDS):
        for j, region in enumerate(REGIONS):
            ax = axes[i, j]
            cell = data[(data['band'] == band) & (data['region'] == region)]
            wide = cell.pivot_table(index='sub', columns='condition', values=spec['value']).reindex(
                columns=['NS', 'SD']).dropna()
            for _, row in wide.iterrows():
                ax.plot([0, 1], [row['NS'], row['SD']], color='0.6', lw=0.7, alpha=0.6)
            for x, cond in enumerate(('NS', 'SD')):
                ax.scatter(np.full(len(wide), x), wide[cond], s=10, color=COLORS[cond], alpha=0.7, zorder=3)
                if len(wide) > 1:
                    mean, low, high = _ci95(wide[cond].to_numpy())
                    ax.errorbar(x + (0.12 if x else -0.12), mean, yerr=[[mean - low], [high - mean]],
                                fmt='o', color='black', mfc=COLORS[cond], ms=7, capsize=4, zorder=4)
            if len(wide) > 1:
                ax.plot([-0.12, 1.12], [wide['NS'].mean(), wide['SD'].mean()], color='black', lw=2, zorder=4)
            p = np.nan
            if region_stats is not None:
                match = region_stats[(region_stats['eyes'] == eyes) & (region_stats['band'] == band)
                                     & (region_stats['region'] == region)]
                if len(match):
                    p = match.iloc[0]['p_fdr_wilcoxon']
            ax.set_title(f'{band.capitalize()} · {region}\nn = {len(wide)}, FDR {_p_label(p)}', fontsize=10)
            ax.set_xticks([0, 1], ['NS', 'SD'])
            ax.set_xlim(-0.4, 1.4)
            if j == 0:
                ax.set_ylabel(spec['axis'])
            ax.grid(axis='y', alpha=0.2)
    fig.suptitle(f"{EYES_TITLE[eyes]}: regional {spec['what']} per participant (grey lines) and group "
                 'mean ± 95 % CI (black)', fontsize=13)
    _save(fig, out_dir, f"paired{spec['suffix']}_{eyes}")
    return fig


# ---------------------------------------------------------------------------- theta/beta
def plot_theta_beta(tbr: pd.DataFrame, tbr_stats: pd.DataFrame | None, out_dir: Path) -> Figure:
    """Frontal theta/beta (log10) per condition: violins, paired lines, one panel per eye condition."""
    fig = _figure(10, 5)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, eyes in zip(axes, ('open', 'closed')):
        wide = tbr[tbr['eyes'] == eyes].pivot_table(index='sub', columns='condition', values='log10_tbr') \
            .reindex(columns=['NS', 'SD']).dropna()
        if len(wide) > 1:
            parts = ax.violinplot([wide['NS'], wide['SD']], positions=[0, 1], showmedians=True, widths=0.7)
            for body, cond in zip(parts['bodies'], ('NS', 'SD')):
                body.set_facecolor(COLORS[cond])
                body.set_edgecolor(COLORS[cond])
                body.set_alpha(0.35)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                parts[key].set_color('0.3')
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row['NS'], row['SD']], color='0.6', lw=0.7, alpha=0.6)
        for x, cond in enumerate(('NS', 'SD')):
            ax.scatter(np.full(len(wide), x), wide[cond], s=12, color=COLORS[cond], zorder=3)
        p = np.nan
        if tbr_stats is not None and (tbr_stats['eyes'] == eyes).any():
            p = tbr_stats[tbr_stats['eyes'] == eyes].iloc[0]['p_wilcoxon']
        ax.set_title(f'{EYES_TITLE[eyes]}\nn = {len(wide)}, Wilcoxon {_p_label(p)}')
        ax.set_xticks([0, 1], [LABELS['NS'], LABELS['SD']])
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Frontal log10(theta/beta) (unitless)')
    fig.suptitle('Frontal theta/beta ratio by condition', fontsize=13)
    _save(fig, out_dir, 'theta_beta')
    return fig


# --------------------------------------------------------------------------- summary
def hypothesis_summary(region_stats: pd.DataFrame, tbr_stats: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Hypothesis table (CSV + Markdown): expected vs observed direction, corrected p, effect size."""
    table = evaluate(region_stats, tbr_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'hypothesis_summary.csv', index=False)
    columns = ['hypothesis', 'eyes', 'band', 'region', 'expected', 'observed', 'n', 'mean_diff',
               'p_corrected', 'rank_biserial', 'cohens_dz', 'agreement', 'largest_region']
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for _, row in table[columns].iterrows():
        cells = [f'{v:.3g}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    notes = ('\n\n`mean_diff` = SD − NS in log10 units (theta/beta: log10 ratio). `p_corrected` = '
             'BH-FDR Wilcoxon p across 20 band × region tests (H3: single uncorrected test). '
             '`largest_region` = region with the largest |dz| for that band.\n')
    (out_dir / 'hypothesis_summary.md').write_text('\n'.join(lines) + notes, encoding='utf-8')
    return table


# ----------------------------------------------------------------------------- captions
def captions(inputs: dict, summary: pd.DataFrame | None) -> dict[str, str]:
    """Figure captions with the actual sample sizes, in the same order as the figures."""
    perms = ''
    clusters = inputs.get('clusters')
    if clusters is not None and len(clusters) and 'n_permutations' in clusters:
        perms = f" ({int(clusters['n_permutations'].iloc[0])} sign-flip permutations)"
    low, high = config.SPECTRAL.edges[0], config.SPECTRAL.edges[-1]
    text = {}
    for eyes in ('open', 'closed'):
        text[f'topomap_{eyes}'] = (
            f'{EYES_TITLE[eyes]}. Mean within-participant difference SD − NS of log10 absolute band '
            f'power (µV²) at each electrode, averaged over participants with both sessions. Red: higher '
            f'power after SD; blue: lower. One symmetric colour scale for all bands. White dots mark '
            f'electrodes in clusters with p < .05 in the cluster-based permutation test{perms}. '
            'Electrode positions: MNE standard 10-05 montage.')
        text[f'psd_regions_{eyes}'] = (
            f'{EYES_TITLE[eyes]}. Group-average power spectral density per region for NS (teal) and SD '
            f'(rose): geometric mean over participants of the region-averaged Welch PSD (2 s Hann, 50 % '
            f'overlap, 0.5 Hz resolution). Shaded areas: 95 % confidence interval of the mean (t-based, on '
            f'log10 PSD). Dashed lines: band edges ({", ".join(f"{e:g}" for e in config.SPECTRAL.edges)} Hz). '
            'Only participants with both sessions and at least half of the region\'s channels are included.')
        text[f'paired_{eyes}'] = (
            f'{EYES_TITLE[eyes]}. Mean log10 absolute power (µV²) per band and region. Grey lines: one '
            'participant (NS → SD); coloured points: individual values; black: group mean ± 95 % CI. '
            'Panel titles give n and the BH-FDR-corrected Wilcoxon signed-rank p (20 tests per eye condition).')
        text[f'topomap_relative_{eyes}'] = (
            f'{EYES_TITLE[eyes]}. Mean within-participant difference SD − NS of relative band power '
            f'(percentage points of total {low:g}–{high:g} Hz power) at each electrode. Red: larger share '
            'after SD; blue: smaller. The cluster test was run on absolute log10 power, so no electrodes are marked.')
        text[f'paired_relative_{eyes}'] = (
            f'{EYES_TITLE[eyes]}. Mean relative power (% of total {low:g}–{high:g} Hz power) per band and '
            'region. Grey lines: one participant (NS → SD); black: group mean ± 95 % CI. Panel titles give n '
            'and the BH-FDR-corrected Wilcoxon signed-rank p for relative power.')
    text['theta_beta'] = (
        'Frontal theta/beta ratio (mean log10 ratio over frontal channels) for NS (teal) and SD (rose), '
        'eyes open (primary) and eyes closed (secondary). Violins: distribution; grey lines: one '
        'participant; titles: n and Wilcoxon signed-rank p (single pre-specified test, uncorrected).')
    text['hypothesis_summary'] = (
        f'Hypotheses H1–H4 (docs/hypotheses.md) against the observed paired effects. Expected and '
        f'observed direction of SD − NS, corrected p (BH-FDR across 20 band × region tests; H3 single '
        f'test), rank-biserial r and Cohen\'s dz. "supported" = expected sign and p < .05. Relative power '
        f'is relative to {low:g}–{high:g} Hz. Group-level spectral differences only.')
    return text


def write_captions(text: dict[str, str], out_dir: Path) -> Path:
    lines = ['# Figure captions', '',
             'Generated by `python -m tps.analysis.figures`. Group-level spectral differences only; '
             'no clinical interpretation.', '']
    for name, caption in text.items():
        file = f'{name}.md / .csv' if name == 'hypothesis_summary' else f'{name}.png'
        lines += [f'## {name}', '', f'File: `{file}`', '', caption, '']
    path = out_dir / 'captions.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


# --------------------------------------------------------------------------------- main
def load_inputs(results_dir: Path) -> dict:
    """Read the batch and statistics outputs; missing optional files become None."""
    def read(name: str) -> pd.DataFrame | None:
        path = results_dir / name
        return pd.read_csv(path) if path.is_file() else None
    inputs = dict(features=read('features_long.csv'), regions=read('features_region.csv'),
                  region_stats=read('stats_region_log_power.csv'), rel_stats=read('stats_region_rel_power.csv'),
                  tbr_stats=read('stats_frontal_theta_beta.csv'),
                  clusters=read('cluster_results.csv'), psd=read('psd_channels.csv'),
                  theta_beta=read('theta_beta.csv'))
    if inputs['features'] is None or inputs['regions'] is None:
        raise FileNotFoundError('Run python -m tps.pipeline.batch and python -m tps.analysis.stats first.')
    inputs['tbr'] = frontal_theta_beta(inputs['theta_beta']) if inputs['theta_beta'] is not None else None
    return inputs


def make_all(results_dir: Path, out_dir: Path | None = None) -> dict[str, Figure | pd.DataFrame]:
    """Build every figure, the hypothesis table and captions.md; returns them by name."""
    out_dir = out_dir or results_dir / 'figures'
    inputs = load_inputs(results_dir)
    made: dict[str, Figure | pd.DataFrame] = {}
    for eyes in ('open', 'closed'):
        if not (inputs['features']['eyes'] == eyes).any():
            continue
        made[f'topomap_{eyes}'] = plot_topomaps(inputs['features'], inputs['clusters'], eyes, out_dir)
        if inputs['psd'] is not None and len(inputs['psd']):
            made[f'psd_regions_{eyes}'] = plot_psd_by_region(inputs['psd'], eyes, out_dir)
        made[f'paired_{eyes}'] = plot_paired(inputs['regions'], inputs['region_stats'], eyes, out_dir)
        made[f'topomap_relative_{eyes}'] = plot_topomaps(inputs['features'], None, eyes, out_dir, 'relative')
        made[f'paired_relative_{eyes}'] = plot_paired(inputs['regions'], inputs['rel_stats'], eyes, out_dir,
                                                      'relative')
    if inputs['tbr'] is not None:
        made['theta_beta'] = plot_theta_beta(inputs['tbr'], inputs['tbr_stats'], out_dir)
    if inputs['region_stats'] is not None and inputs['tbr_stats'] is not None:
        made['hypothesis_summary'] = hypothesis_summary(inputs['region_stats'], inputs['tbr_stats'], out_dir)
    text = captions(inputs, made.get('hypothesis_summary'))
    write_captions({k: v for k, v in text.items() if k in made}, out_dir)
    if out_dir == results_dir / 'figures':   # the report links figures/<name>.png
        from tps.analysis.report import write_report
        write_report(results_dir)
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Result figures for SD vs NS.')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR)
    parser.add_argument('--out-dir', type=Path, default=None, help='default: <results-dir>/figures')
    args = parser.parse_args(argv)
    made = make_all(args.results_dir, args.out_dir)
    print('Saved:', ', '.join(made), '->', args.out_dir or args.results_dir / 'figures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
