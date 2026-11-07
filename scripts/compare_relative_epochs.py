"""Relative-power, epoch-retention, and minimum-epoch paired analyses."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from eeg_viewer.region_analysis import aggregate_regions, calculate_stats


EXPECTED_BANDS = ('Delta', 'Theta', 'Alfa', 'Beta')
EYES = ('Ojos abiertos', 'Ojos cerrados')


def read_features(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'participant', 'session', 'eyes', 'SessionOrder', 'channel',
                    'band', 'absolute_uv2', 'epochs_used', 'epochs_used_global'}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f'Faltan columnas requeridas: {missing}')
        rows = list(reader)
    return rows


def _relative_rows(rows):
    """Use exported relative power, or derive band/0.5-30 total if absent."""
    by_channel_record = defaultdict(dict)
    for row in rows:
        key = (row['participant'], row['session'], row['eyes'], row['channel'])
        by_channel_record[key][row['band']] = row
    converted = []
    source_counts = defaultdict(int)
    for band_rows in by_channel_record.values():
        if not all(band in band_rows for band in EXPECTED_BANDS):
            continue
        relative_available = all(
            band_rows[band].get('relative_percent', '').strip() for band in EXPECTED_BANDS
        )
        if relative_available:
            ratios = {band: float(band_rows[band]['relative_percent']) / 100
                      for band in EXPECTED_BANDS}
            if all(np.isfinite(value) and value > 0 for value in ratios.values()):
                source = 'relative_percent'
            else:
                relative_available = False
        if not relative_available:
            powers = {band: float(band_rows[band]['absolute_uv2']) for band in EXPECTED_BANDS}
            total = sum(powers.values())
            if total <= 0 or not np.isfinite(total):
                continue
            ratios = {band: value / total for band, value in powers.items()}
            source = 'calculated_from_four_bands'
        for band in EXPECTED_BANDS:
            clone = dict(band_rows[band])
            clone['absolute_uv2'] = ratios[band]
            converted.append(clone)
            source_counts[source] += 1
    return converted, source_counts


def _save_csv(path, rows):
    columns = list(rows[0]) if rows else []
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _stats_by_key(rows):
    return {(row['eyes'], row['band'], row['region']): row for row in rows}


def compare_stats(absolute_stats, relative_stats):
    absolute = _stats_by_key(absolute_stats)
    relative = _stats_by_key(relative_stats)
    output = []
    for key in sorted(absolute):
        a, r = absolute[key], relative[key]
        sig_a = float(a['fdr_q_value']) < 0.05
        sig_r = float(r['fdr_q_value']) < 0.05
        output.append(dict(
            eyes=key[0], band=key[1], region=key[2],
            n_pairs_absolute=a['n_pairs'],
            median_difference_log10_absolute=a['median_difference_log10_sd_minus_ns'],
            rank_biserial_absolute=a['rank_biserial_effect'],
            p_absolute=a['p_value'], q_absolute=a['fdr_q_value'],
            n_pairs_relative=r['n_pairs'],
            median_difference_log10_relative=r['median_difference_log10_sd_minus_ns'],
            rank_biserial_relative=r['rank_biserial_effect'],
            p_relative=r['p_value'], q_relative=r['fdr_q_value'],
            significant_absolute_fdr=sig_a, significant_relative_fdr=sig_r,
            significance_changed=sig_a != sig_r,
        ))
    return output


def epoch_retention_table(rows):
    channel_sets = defaultdict(set)
    channel_epochs = defaultdict(dict)
    global_epochs = defaultdict(set)
    all_channels = set()
    for row in rows:
        record = (row['participant'], row['session'], row['eyes'])
        canonical = row['channel'].casefold()
        all_channels.add(canonical)
        channel_sets[record].add(canonical)
        channel_epochs[record][canonical] = int(row['epochs_used'])
        global_epochs[record].add(int(row['epochs_used_global']))

    by_condition = defaultdict(list)
    for record, channel_values in channel_epochs.items():
        participant, session, eyes = record
        included = set(channel_values)
        global_values = global_epochs[record]
        if len(global_values) != 1:
            raise ValueError(f'epochs_used_global no es constante en {record}.')
        by_condition[(session, eyes)].append(dict(
            channel_median=float(np.median(list(channel_values.values()))),
            channel_min=min(channel_values.values()),
            channel_max=max(channel_values.values()),
            global_epochs=int(next(iter(global_values))),
            n_channels=len(included),
            n_excluded=len(all_channels - included),
        ))

    output = []
    for (session, eyes), records in sorted(by_condition.items()):
        session_condition = 'NS' if session == 'ses-1' else 'SD'
        channel_medians = [record['channel_median'] for record in records]
        channel_counts = [record['channel_min'] for record in records]
        global_counts = [record['global_epochs'] for record in records]
        excluded = [record['n_excluded'] for record in records]
        channels_included = [record['n_channels'] for record in records]
        output.append(dict(
            session=session, condition=session_condition, eyes=eyes,
            n_records=len(records), expected_channels_casefold=len(all_channels),
            median_epochs_used_per_record=float(np.median(channel_medians)),
            min_record_median_channel_epochs=float(min(channel_medians)),
            max_record_median_channel_epochs=float(max(channel_medians)),
            min_channel_epochs_any_record=min(record['channel_min'] for record in records),
            max_channel_epochs_any_record=max(record['channel_max'] for record in records),
            median_global_epochs_per_record=float(np.median(global_counts)),
            min_global_epochs_per_record=min(global_counts),
            max_global_epochs_per_record=max(global_counts),
            channels_included_median_per_record=float(np.median(channels_included)),
            channels_excluded_total_inferred=sum(excluded),
            channels_excluded_median_per_record=float(np.median(excluded)),
            channels_excluded_min_per_record=min(excluded),
            channels_excluded_max_per_record=max(excluded),
        ))
    return output


def _min20_rows(rows):
    counts = {}
    for row in rows:
        key = (row['participant'], row['eyes'], row['channel'].casefold(), row['session'])
        counts[key] = int(row['epochs_used'])
    eligible = set()
    participants = {(row['participant'], row['eyes'], row['channel'].casefold()) for row in rows}
    for participant, eyes, channel in participants:
        ns = counts.get((participant, eyes, channel, 'ses-1'), 0)
        sd = counts.get((participant, eyes, channel, 'ses-2'), 0)
        if ns >= 20 and sd >= 20:
            eligible.add((participant, eyes, channel))
    filtered = [row for row in rows if
                (row['participant'], row['eyes'], row['channel'].casefold()) in eligible]
    return filtered


def run(features_path, output_dir):
    rows = read_features(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_regional = aggregate_regions(rows)
    absolute_stats = calculate_stats(absolute_regional)
    relative_rows, sources = _relative_rows(rows)
    relative_regional = aggregate_regions(relative_rows)
    relative_stats = calculate_stats(relative_regional)
    _save_csv(output_dir / 'stats_relative_vs_absolute.csv',
              compare_stats(absolute_stats, relative_stats))

    retention = epoch_retention_table(rows)
    _save_csv(output_dir / 'epochs_and_exclusions_by_session_condition.csv', retention)

    min20_regional = aggregate_regions(_min20_rows(rows))
    min20_stats = calculate_stats(min20_regional)
    _save_csv(output_dir / 'stats_absolute_min20.csv', min20_stats)
    return dict(rows=len(rows), relative_sources=dict(sources), retention=retention,
                min20_stats=min20_stats, absolute_stats=absolute_stats,
                relative_stats=relative_stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Comparaciones relativas y de retención desde features.csv.')
    parser.add_argument('--features', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'features.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'analysis')
    args = parser.parse_args(argv)
    try:
        result = run(args.features, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Filas de features: {result['rows']}")
    print(f"Origen de potencia relativa: {result['relative_sources']}")
    print(f"Salidas: {args.output}")


if __name__ == '__main__':
    main()
