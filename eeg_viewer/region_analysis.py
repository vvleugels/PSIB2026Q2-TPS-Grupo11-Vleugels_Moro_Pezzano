"""Regional paired statistics and figures from an exported group features CSV."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon


REGIONS = ('Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal')
BANDS = ('Delta', 'Theta', 'Alfa', 'Beta')
EYES = ('Ojos abiertos', 'Ojos cerrados')
REQUIRED_COLUMNS = {
    'participant', 'session', 'session_condition', 'eyes', 'SessionOrder',
    'channel', 'band', 'absolute_uv2',
}


def region_for_channel(channel: str) -> str | None:
    """Map 10-20/10-10 labels to five lobar groups; omit polar Fp/AF leads."""
    name = channel.strip().upper()
    if name.startswith(('FP', 'AF')):
        return None
    if name.startswith(('FT', 'TP', 'T')):
        return 'Temporal'
    if name.startswith('FC'):
        return 'Central'
    if name.startswith('CP'):
        return 'Parietal'
    if name.startswith('PO'):
        return 'Occipital'
    if name.startswith('C'):
        return 'Central'
    if name.startswith('P'):
        return 'Parietal'
    if name.startswith('O'):
        return 'Occipital'
    if name.startswith('F'):
        return 'Frontal'
    return None


def _read_features(path: Path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f'Faltan columnas requeridas en features.csv: {missing}')
        rows = list(reader)
    if any(row.get('rejection_mode') not in (None, '', 'channel') for row in rows):
        raise ValueError('El análisis regional requiere features del modo channel.')
    return rows


def aggregate_regions(rows):
    """Average channel-wise log10 absolute power within participant and region."""
    values = defaultdict(list)
    for row in rows:
        region = region_for_channel(row['channel'])
        if region is None:
            continue
        if row['session'] not in {'ses-1', 'ses-2'}:
            continue
        if row['eyes'] not in EYES or row['band'] not in BANDS:
            continue
        power = float(row['absolute_uv2'])
        if not np.isfinite(power) or power <= 0:
            continue
        condition = 'NS' if row['session'] == 'ses-1' else 'SD'
        key = (row['participant'], row['session'], condition, row['eyes'],
               row['SessionOrder'], region, row['band'])
        values[key].append((row['channel'], float(np.log10(power))))

    output = []
    for key, channel_values in sorted(values.items()):
        participant, session, condition, eyes, order, region, band = key
        output.append(dict(
            participant=participant, session=session, condition=condition,
            eyes=eyes, SessionOrder=order, region=region, band=band,
            log10_power_mean=float(np.mean([value for _, value in channel_values])),
            n_channels=len(channel_values),
        ))
    return output


def _fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values; NaNs remain NaN."""
    result = [float('nan')] * len(p_values)
    valid = [(index, float(value)) for index, value in enumerate(p_values)
             if np.isfinite(value)]
    if not valid:
        return result
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    adjusted = [0.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        _, p_value = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, p_value * count / rank)
        adjusted[rank_index] = min(1.0, running)
    for (index, _), q_value in zip(ordered, adjusted):
        result[index] = q_value
    return result


def _wilcoxon_effect(differences):
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    median = float(np.median(differences))
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0, 1.0, median
    statistic, p_value = wilcoxon(nonzero, alternative='two-sided', method='auto')
    ranks = rankdata(np.abs(nonzero), method='average')
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    effect = (positive - negative) / (positive + negative)
    return float(statistic), float(p_value), median, effect


def calculate_stats(regional_rows):
    """Wilcoxon SD-NS by eye, band, region plus exploratory order interaction."""
    lookup = {}
    for row in regional_rows:
        key = (row['participant'], row['eyes'], row['region'], row['band'], row['condition'])
        lookup[key] = row

    output = []
    for eyes in EYES:
        for band in BANDS:
            for region in REGIONS:
                paired = []
                for participant in sorted({r['participant'] for r in regional_rows}):
                    ns = lookup.get((participant, eyes, region, band, 'NS'))
                    sd = lookup.get((participant, eyes, region, band, 'SD'))
                    if ns is not None and sd is not None:
                        paired.append((participant, float(sd['log10_power_mean']) -
                                       float(ns['log10_power_mean']), ns['SessionOrder']))
                differences = [difference for _, difference, _ in paired]
                if differences:
                    statistic, p_value, median_difference, effect = _wilcoxon_effect(differences)
                else:
                    statistic = p_value = median_difference = effect = float('nan')
                ns_to_sd = [difference for _, difference, order in paired if order == 'NS->SD']
                sd_to_ns = [difference for _, difference, order in paired if order == 'SD->NS']
                if len(ns_to_sd) >= 2 and len(sd_to_ns) >= 2:
                    order_p = float(mannwhitneyu(ns_to_sd, sd_to_ns,
                                                 alternative='two-sided', method='auto').pvalue)
                else:
                    order_p = float('nan')
                output.append(dict(
                    eyes=eyes, band=band, region=region,
                    n_pairs=len(paired),
                    median_difference_log10_sd_minus_ns=median_difference,
                    wilcoxon_statistic=statistic, p_value=p_value,
                    fdr_q_value=float('nan'),
                    rank_biserial_effect=effect,
                    n_order_ns_to_sd=len(ns_to_sd),
                    median_difference_order_ns_to_sd=float(np.median(ns_to_sd)) if ns_to_sd else float('nan'),
                    n_order_sd_to_ns=len(sd_to_ns),
                    median_difference_order_sd_to_ns=float(np.median(sd_to_ns)) if sd_to_ns else float('nan'),
                    order_mannwhitney_p_value=order_p,
                    order_fdr_q_value=float('nan'),
                ))
        eye_rows = [i for i, row in enumerate(output) if row['eyes'] == eyes]
        q = _fdr_bh([output[i]['p_value'] for i in eye_rows])
        order_q = _fdr_bh([output[i]['order_mannwhitney_p_value'] for i in eye_rows])
        for index, q_value, order_q_value in zip(eye_rows, q, order_q):
            output[index]['fdr_q_value'] = q_value
            output[index]['order_fdr_q_value'] = order_q_value
    return output


def _save_csv(path: Path, rows):
    columns = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_pair_plot(path, regional_rows, eyes, band):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(17, 4.5), sharey=True)
    for axis, region in zip(axes, REGIONS):
        values = defaultdict(dict)
        for row in regional_rows:
            if row['eyes'] == eyes and row['band'] == band and row['region'] == region:
                values[row['participant']][row['condition']] = float(row['log10_power_mean'])
        for participant, pair in sorted(values.items()):
            if {'NS', 'SD'}.issubset(pair):
                axis.plot([0, 1], [pair['NS'], pair['SD']], marker='o',
                          color='#5078a0', alpha=0.55, linewidth=0.9, markersize=3)
        axis.set_title(region)
        axis.set_xticks([0, 1], ['NS', 'SD'])
        axis.grid(axis='y', alpha=0.25)
    axes[0].set_ylabel('Potencia absoluta media (log10 µV²)')
    eye_label = 'ojos abiertos' if eyes == 'Ojos abiertos' else 'ojos cerrados'
    fig.suptitle(f'Pares NS vs SD — {band} — {eye_label}')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _write_topomap(path, rows, eyes, band):
    by_channel = defaultdict(dict)
    for row in rows:
        if row['eyes'] != eyes or row['band'] != band:
            continue
        condition = 'NS' if row['session'] == 'ses-1' else 'SD'
        by_channel[row['participant'], row['channel']][condition] = np.log10(
            float(row['absolute_uv2']))
    participant_diffs = defaultdict(list)
    for (participant, channel), pair in by_channel.items():
        if {'NS', 'SD'}.issubset(pair):
            participant_diffs[channel].append(pair['SD'] - pair['NS'])

    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    canonical = {name.casefold(): name for name in positions}
    by_position = defaultdict(list)
    for channel, differences in participant_diffs.items():
        montage_name = canonical.get(channel.casefold())
        if montage_name is not None:
            by_position[montage_name].append(float(np.mean(differences)))
    names = sorted(by_position)
    values = np.asarray([np.mean(by_position[name]) for name in names], dtype=float)
    info = mne.create_info(names, sfreq=500.0, ch_types='eeg')
    info.set_montage(montage, match_case=False, on_missing='raise')
    limit = float(np.max(np.abs(values))) if values.size else 1.0
    if limit == 0:
        limit = 1.0
    fig, axis = plt.subplots(figsize=(6.2, 5.4))
    image, _ = mne.viz.plot_topomap(
        values, info, ch_type='eeg', sensors=True,
        contours=6, cmap='RdBu_r', vlim=(-limit, limit), axes=axis, show=False,
    )
    fig.colorbar(image, ax=axis, shrink=0.78, label='Diferencia media SD − NS (log10 µV²)')
    eye_label = 'ojos abiertos' if eyes == 'Ojos abiertos' else 'ojos cerrados'
    axis.set_title(f'{band} — {eye_label} — SD − NS')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def run_region_analysis(features_path, output_dir):
    features_path, output_dir = Path(features_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_features(features_path)
    regional = aggregate_regions(rows)
    stats_rows = calculate_stats(regional)
    _save_csv(output_dir / 'regional_features.csv', regional)
    _save_csv(output_dir / 'stats_table.csv', stats_rows)
    channel_map = [{'channel': name, 'region': region_for_channel(name) or 'Excluded (Fp/AF)'}
                   for name in sorted({row['channel'] for row in rows})]
    _save_csv(output_dir / 'channel_region_map.csv', channel_map)
    eye_slug = {'Ojos abiertos': 'open', 'Ojos cerrados': 'closed'}
    band_slug = {'Delta': 'delta', 'Theta': 'theta', 'Alfa': 'alpha', 'Beta': 'beta'}
    for eyes in EYES:
        for band in BANDS:
            _write_pair_plot(output_dir / f"pairs_{eye_slug[eyes]}_{band_slug[band]}.png",
                             regional, eyes, band)
            _write_topomap(output_dir / f"topomap_{eye_slug[eyes]}_{band_slug[band]}.png",
                           rows, eyes, band)
    return dict(regional=regional, stats=stats_rows, channels=len(channel_map),
                feature_rows=len(rows), regional_rows=len(regional))
