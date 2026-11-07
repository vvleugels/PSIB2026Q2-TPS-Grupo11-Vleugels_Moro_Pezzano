"""Regenerate group-analysis figures and display metadata from the exported CSVs."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

from .region_analysis import BANDS, EYES, REGIONS, aggregate_regions


EYE_SLUG = {'Ojos abiertos': 'open', 'Ojos cerrados': 'closed'}
BAND_SLUG = {'Delta': 'delta', 'Theta': 'theta', 'Alfa': 'alpha', 'Beta': 'beta'}


def read_feature_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No hay features en {path}')
    return rows


def _relative_feature_rows(rows):
    converted = []
    for row in rows:
        rel = row.get('relative_percent', '').strip()
        if not rel:
            raise ValueError('features.csv no contiene relative_percent; regenere la exportación relativa.')
        clone = dict(row)
        clone['absolute_uv2'] = float(rel) / 100.0
        converted.append(clone)
    return converted


def _pairs_plot(path, regional, eyes, band, power_kind):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(17, 4.5), sharey=True)
    for axis, region in zip(axes, REGIONS):
        by_participant = defaultdict(dict)
        for row in regional:
            if row['eyes'] == eyes and row['band'] == band and row['region'] == region:
                by_participant[row['participant']][row['condition']] = float(row['log10_power_mean'])
        for pair in by_participant.values():
            if {'NS', 'SD'}.issubset(pair):
                axis.plot((0, 1), (pair['NS'], pair['SD']), marker='o',
                          color='#5078a0', alpha=0.55, linewidth=0.9, markersize=3)
        axis.set_title(region)
        axis.set_xticks((0, 1), ('NS', 'SD'))
        axis.grid(axis='y', alpha=0.25)
    ylabel = ('Potencia absoluta media (log10 µV²)' if power_kind == 'absolute'
              else 'Potencia relativa media (log10 proporción)')
    axes[0].set_ylabel(ylabel)
    eye_label = 'ojos abiertos' if eyes == 'Ojos abiertos' else 'ojos cerrados'
    fig.suptitle(f'Pares NS vs SD — {band} — {eye_label} — {power_kind}')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _topomap(path, rows, eyes, band, power_kind):
    pairs = defaultdict(dict)
    for row in rows:
        if row['eyes'] != eyes or row['band'] != band:
            continue
        condition = 'NS' if row['session'] == 'ses-1' else 'SD'
        channel_key = row['channel'].casefold()
        power = float(row['absolute_uv2'])
        if power_kind == 'relative':
            rel = row.get('relative_percent', '').strip()
            if not rel:
                raise ValueError('features.csv no contiene relative_percent.')
            power = float(rel) / 100.0
        pairs[(row['participant'], channel_key)][condition] = np.log10(power)

    participant_differences = defaultdict(list)
    for (_, channel), values in pairs.items():
        if {'NS', 'SD'}.issubset(values):
            participant_differences[channel].append(values['SD'] - values['NS'])

    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    canonical = {name.casefold(): name for name in positions}
    position_values = defaultdict(list)
    for channel, differences in participant_differences.items():
        montage_name = canonical.get(channel)
        if montage_name:
            position_values[montage_name].append(float(np.mean(differences)))
    names = sorted(position_values)
    values = np.asarray([np.mean(position_values[name]) for name in names], dtype=float)
    info = mne.create_info(names, sfreq=500.0, ch_types='eeg')
    info.set_montage(montage, match_case=False, on_missing='raise')

    # A 11 cm head sphere contains the standard-10/20 electrode arc (about 9–10 cm).
    sphere = (0.0, 0.0, 0.0, 0.11)
    limit = float(np.max(np.abs(values))) if values.size else 1.0
    if limit == 0:
        limit = 1.0
    fig, axis = plt.subplots(figsize=(6.2, 5.4))
    image, _ = mne.viz.plot_topomap(
        values, info, ch_type='eeg', sensors=True, contours=6,
        cmap='RdBu_r', vlim=(-limit, limit), sphere=sphere,
        extrapolate='head', axes=axis, show=False,
    )
    unit = 'log10 µV²' if power_kind == 'absolute' else 'log10 proporción'
    fig.colorbar(image, ax=axis, shrink=0.78,
                 label=f'Diferencia media SD − NS ({unit})')
    eye_label = 'ojos abiertos' if eyes == 'Ojos abiertos' else 'ojos cerrados'
    axis.set_title(f'{band} — {eye_label} — SD − NS — {power_kind}')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _write_group_summary(rows, analysis_dir):
    participants = sorted({row['participant'] for row in rows},
                          key=lambda value: int(value.removeprefix('sub-'))
                          if value.removeprefix('sub-').isdigit() else value)
    modes = sorted({row.get('rejection_mode', '').strip() for row in rows if row.get('rejection_mode')})
    metrics = [
        ('participant_count', str(len(participants))),
        ('participants', ', '.join(participants)),
        ('rejection_mode', ', '.join(modes) if modes else 'No disponible'),
        ('channel_exclusion_counts', 'Ver epochs_and_exclusions_by_session_condition.csv; conteo inferido de canales ausentes'),
    ]
    with (analysis_dir / 'group_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(('metric', 'value'))
        writer.writerows(metrics)


def regenerate_group_figures(features_path, analysis_dir):
    features_path, analysis_dir = Path(features_path), Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows = read_feature_rows(features_path)
    absolute_regional = aggregate_regions(rows)
    relative_rows = _relative_feature_rows(rows)
    relative_regional = aggregate_regions(relative_rows)
    _write_group_summary(rows, analysis_dir)

    generated = []
    for eyes in EYES:
        eye = EYE_SLUG[eyes]
        for band in BANDS:
            band_name = BAND_SLUG[band]
            # Repair the existing absolute maps with a sphere large enough for every sensor.
            absolute_map = analysis_dir / f'topomap_{eye}_{band_name}.png'
            _topomap(absolute_map, rows, eyes, band, 'absolute')
            generated.append(absolute_map)
            relative_pairs = analysis_dir / f'pairs_relative_{eye}_{band_name}.png'
            _pairs_plot(relative_pairs, relative_regional, eyes, band, 'relative')
            generated.append(relative_pairs)
            relative_map = analysis_dir / f'topomap_relative_{eye}_{band_name}.png'
            _topomap(relative_map, rows, eyes, band, 'relative')
            generated.append(relative_map)
    _write_regional_relative(relative_regional, analysis_dir / 'regional_relative_features.csv')
    return generated


def _write_regional_relative(rows, path):
    columns = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
