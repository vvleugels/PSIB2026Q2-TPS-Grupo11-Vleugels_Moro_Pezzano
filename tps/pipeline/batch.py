"""Batch band-power extraction for every subject, session and eye condition (no GUI).

Run from the project folder:  python -m tps.pipeline.batch [--eyes open|closed] [--subjects 01 02]

Reuses the GUI modules for every computation step (BIDS inventory, loading, continuous
FIR filtering, 5 s epochs, peak-to-peak screening, Welch and band integration), so a
recording analysed here and in the GUI gives the same numbers for the same settings.
Results are cached per recording, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

from tps import config
from tps.core.annotations import make_epochs
from tps.core.badchannels import find_bad_channels
from tps.core.data import Recording, Study, annex_placeholder, discover, open_recording
from tps.core.processing import FilterConfig
from tps.core.screening import scan_recording
from tps.core.spectral import band_powers, welch_epoch

log = logging.getLogger('batch')

FEATURE_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'channel', 'band', 'abs_power_uV2',
                   'rel_power_pct', 'n_epochs_used', 'n_epochs_rejected']
THETA_BETA_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'channel', 'ratio']
EXCLUSION_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'level', 'channel', 'epoch', 'criterion', 'reason']
# Bumped whenever the per-recording output format changes, so old caches are recomputed.
CACHE_FORMAT = 4   # 4: bad-channel criteria and manual marks; 3: ICA provenance
# PSD bins exported to psd_channels.csv: up to the filter's upper cutoff, where the spectrum
# is still meaningful (0.5 Hz resolution with 2 s Welch segments).
PSD_MAX_HZ = config.FILTER.high


def config_hash() -> str:
    """Short fingerprint of the analysis settings; cached results are reused only if it matches."""
    text = json.dumps({'format': CACHE_FORMAT, **config.as_dict()}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def labels(study: Study) -> dict:
    """Identifying columns shared by every output row of a recording."""
    return dict(sub=study.participant, ses=study.session,
                condition=config.SESSIONS.get(study.session, study.session),
                eyes=config.EYES.get(study.task, study.task))


def exclusion(study: Study, level: str, reason: str, channel: str = '', epoch: str = '',
              criterion: str = '') -> dict:
    return {**labels(study), 'level': level, 'channel': channel, 'epoch': epoch,
            'criterion': criterion, 'reason': reason}


# ----------------------------------------------------------------------------- inventory
def file_problem(study: Study) -> str | None:
    """Reason why a recording cannot be read from disk, or None if its files look complete."""
    set_path = study.path
    if annex_placeholder(set_path):
        return 'git-annex link without local content (.set); run datalad get or download the file'
    if not set_path.is_file():
        return 'missing .set file'
    fdt = set_path.with_suffix('.fdt')
    if annex_placeholder(fdt):
        return 'git-annex link without local content (.fdt); run datalad get or download the file'
    # A missing .fdt is fine only if the data are embedded in the .set; open_recording
    # verifies the .fdt size against the header, so incomplete downloads are caught there.
    return None


def read_subjects_file(path: Path) -> list[str]:
    """Participant IDs from a text file: one per line, '#' starts a comment."""
    subjects = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            subjects.append(line)
    if not subjects:
        raise ValueError(f'{path}: no participant IDs found')
    return subjects


def select_studies(data_dir: Path, eyes: str | None, subjects: Iterable[str] | None
                   ) -> tuple[list[Study], list[dict]]:
    """BIDS inventory filtered by eye condition and subjects, plus pair-level exclusions."""
    inventory = discover(data_dir)
    wanted = None
    if subjects:
        wanted = {s if s.startswith('sub-') else f'sub-{int(s):02d}' if s.isdigit() else f'sub-{s}'
                  for s in subjects}
        unknown = sorted(wanted - set(inventory))
        if unknown:
            log.warning('Subjects not found in %s: %s', data_dir, ', '.join(unknown))
    tasks = [task for task, name in config.EYES.items() if eyes is None or name == eyes]
    studies, notes = [], []
    for subject, items in inventory.items():
        if wanted is not None and subject not in wanted:
            continue
        for task in tasks:
            found = [s for s in items if s.task == task]
            studies.extend(found)
            sessions = {s.session for s in found}
            # A within-subject comparison needs both sessions; log a missing counterpart.
            if found and sessions != set(config.SESSIONS):
                for missing in sorted(set(config.SESSIONS) - sessions):
                    notes.append({'sub': subject, 'ses': missing,
                                  'condition': config.SESSIONS[missing],
                                  'eyes': config.EYES[task], 'level': 'pair', 'channel': '',
                                  'epoch': '', 'reason': 'recording not present in the dataset; '
                                                         'the other session has no NS/SD counterpart'})
    return studies, notes


# ---------------------------------------------------------------------------- per recording
@dataclass
class EpochDecision:
    """Screening outcome of the complete epochs of one recording."""
    used: list           # Epoch objects kept for the spectrum
    rejected: dict       # epoch index -> reason
    bad_channels: dict   # channel -> reason
    partial: list        # trailing incomplete epochs (never used)


def decide_epochs(recording: Recording, scan: dict, known_bad: dict | None = None) -> EpochDecision:
    """Turn per-channel screening findings into bad channels and rejected epochs.

    known_bad: {channel: reason} already bad for the whole recording (manual marks and the
    automatic criteria of tps/core/badchannels.py); they never reject epochs."""
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    complete = [e for e in epochs if not e.partial]
    partial = [e for e in epochs if e.partial]
    findings = scan['findings']
    flagged: dict[str, int] = {}
    for epoch in complete:
        for issue in findings.get(epoch.index, []):
            flagged[issue['channel']] = flagged.get(issue['channel'], 0) + 1
    bad = dict(known_bad or {})
    limit = config.BAD_CHANNEL_MAX_FRACTION
    if limit is not None and complete:
        for channel, count in flagged.items():
            if channel not in bad and count / len(complete) > limit:
                bad[channel] = (f'flagged in {count}/{len(complete)} complete epochs '
                                f'(> {limit:.0%}); treated as a bad channel')
    used, rejected = [], {}
    for epoch in complete:
        issues = [i for i in findings.get(epoch.index, []) if i['channel'] not in bad]
        if not issues:
            used.append(epoch)
            continue
        parts = []
        for issue in issues:
            amplitude = issue['peak_to_peak_uv']
            what = {'Alta': f'high p-p > {config.SCREENING.high_uv:g} µV',
                    'Plana': f'flat p-p < {config.SCREENING.flat_uv:g} µV'}.get(issue['kind'], issue['kind'])
            parts.append(f"{issue['channel']} {what}" + (f' ({amplitude:.1f} µV)' if amplitude is not None else ''))
        rejected[epoch.index] = '; '.join(parts)
    return EpochDecision(used, rejected, bad, partial)


def decide_epochs_per_channel(recording: Recording, scan: dict, known_bad: dict | None = None
                              ) -> tuple[dict, dict, list]:
    """Per-channel rejection: each channel keeps the complete epochs in which it passed.

    Returns ({channel: [epochs]}, {channel: reason} for channels below MIN_CHANNEL_EPOCHS,
    trailing partial epochs)."""
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    complete = [e for e in epochs if not e.partial]
    failed = {channel: set() for channel in recording.channels}
    for epoch in complete:
        for issue in scan['findings'].get(epoch.index, []):
            failed[issue['channel']].add(epoch.index)
    kept, excluded = {}, dict(known_bad or {})
    for channel in recording.channels:
        if channel in excluded:
            continue
        own = [e for e in complete if e.index not in failed[channel]]
        if len(own) < config.MIN_CHANNEL_EPOCHS:
            excluded[channel] = (f'only {len(own)}/{len(complete)} clean epochs for this channel '
                                 f'(< {config.MIN_CHANNEL_EPOCHS}, per-channel rejection)')
        else:
            kept[channel] = own
    return kept, excluded, [e for e in epochs if e.partial]


def channel_psd(recording: Recording, channel: str, epochs: list, effective: dict,
                filtering: FilterConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD (µV²/Hz) over the given epochs of the filtered continuous channel."""
    # The cache was filled by scan_recording, so the channel is filtered only once.
    continuous = recording.filter.channel(channel, filtering or config.FILTER)
    lengths = {e.last - e.first for e in epochs}
    if len(lengths) == 1:   # usual case: equal epochs -> one vectorised Welch call
        stacked = np.stack([continuous[e.first:e.last] for e in epochs])
        frequencies, psd = welch_epoch(stacked, recording.sfreq, effective)
        mean = psd.mean(axis=0)
    else:   # sfreq not a multiple of 1/5 s: epoch lengths differ by a sample
        spectra = [welch_epoch(continuous[e.first:e.last], recording.sfreq, effective) for e in epochs]
        frequencies, mean = spectra[0][0], np.mean([p for _, p in spectra], axis=0)
    # Volts² -> µV² (×1e12), as in the GUI.
    return frequencies, mean * 1e12


def known_bad_channels(recording: Recording, filtering: FilterConfig, manual: dict | None
                       ) -> tuple[dict, dict]:
    """({channel: reason}, {channel: criterion}) of manual marks plus the automatic criteria."""
    reasons, criteria = {}, {}
    for channel, reason in (manual or {}).items():
        if channel in recording.channels:
            reasons[channel] = 'marked manually' + (f': {reason}' if reason else '')
            criteria[channel] = 'manual'
    if config.BAD_CHANNELS is not None:
        for channel, found in find_bad_channels(recording, filtering, config.BAD_CHANNELS).items():
            if channel not in reasons:
                reasons[channel] = found['reason']
                criteria[channel] = found['criterion']
    return reasons, criteria


def process_recording(recording: Recording, rejection_mode: str | None = None,
                      filtering: FilterConfig | None = None, manual_bad: dict | None = None) -> dict:
    """Screen, epoch and compute band powers for every good channel of one open recording.

    Returns a JSON-serialisable dict with feature rows, theta/beta rows, exclusions and a summary.
    rejection_mode: 'global' or 'per_channel' (default: config.REJECTION_MODE).
    filtering: the signal's filter (default: config.FILTER); the GUI passes the explorer's filter,
    so the group results use the same signal the user sees.
    """
    study = recording.study
    filtering = filtering or config.FILTER
    effective = config.SPECTRAL.effective(recording.sfreq)
    # Screening runs on the filtered signal over ALL EEG channels.
    scan = scan_recording(recording, recording.channels, config.SCREENING, filtering)
    known, criteria = known_bad_channels(recording, filtering, manual_bad)
    if (rejection_mode or config.REJECTION_MODE) == 'per_channel':
        return process_per_channel(recording, scan, effective, filtering, known, criteria)
    decision = decide_epochs(recording, scan, known)
    exclusions = [exclusion(study, 'epoch', 'incomplete final epoch (only complete 5 s epochs are used)',
                            epoch=str(e.index + 1)) for e in decision.partial]
    exclusions += [exclusion(study, 'channel', reason, channel=channel,
                             criterion=criteria.get(channel, 'peak_to_peak'))
                   for channel, reason in sorted(decision.bad_channels.items())]
    epochs_by_index = {e.index: e for e in make_epochs(recording.raw.n_times, recording.sfreq)}
    for index, reason in sorted(decision.rejected.items()):
        epoch = epochs_by_index[index]
        exclusions.append(exclusion(study, 'epoch', f'{epoch.start:g}-{epoch.stop:g} s: {reason}',
                                    epoch=str(index + 1)))
    n_used, n_rejected = len(decision.used), len(decision.rejected)
    summary = dict(**labels(study), file=study.path.name, n_channels=len(recording.channels),
                   n_bad_channels=len(decision.bad_channels), n_epochs_complete=n_used + n_rejected,
                   n_epochs_used=n_used, n_epochs_rejected=n_rejected, excluded=False)
    if n_used < config.MIN_CLEAN_EPOCHS:
        exclusions.append(exclusion(study, 'recording',
                                    f'only {n_used} clean epochs (< {config.MIN_CLEAN_EPOCHS} required)'))
        summary['excluded'] = True
        return dict(features=[], theta_beta=[], psd=None, exclusions=exclusions, summary=summary)
    if any(e.last - e.first < effective['nperseg'] for e in decision.used):
        raise ValueError('Welch segment longer than an epoch')
    features, ratios = [], []
    spectra: dict[str, list[float]] = {}
    kept_frequencies = None
    good = [c for c in recording.channels if c not in decision.bad_channels]
    for channel in good:
        frequencies, psd = channel_psd(recording, channel, decision.used, effective, filtering)
        keep = frequencies <= PSD_MAX_HZ
        kept_frequencies = frequencies[keep].tolist()
        spectra[channel] = psd[keep].tolist()
        bands = band_powers(frequencies, psd, config.SPECTRAL.edges)
        powers = {}
        for name, band in zip(config.BANDS, bands):
            powers[name] = band['absolute_uv2']
            relative = band['relative_percent']
            features.append({**labels(study), 'channel': channel, 'band': name,
                             'abs_power_uV2': band['absolute_uv2'],
                             'rel_power_pct': '' if relative is None else relative,
                             'n_epochs_used': n_used, 'n_epochs_rejected': n_rejected})
        ratio = powers['theta'] / powers['beta'] if powers['beta'] > 0 else ''
        ratios.append({**labels(study), 'channel': channel, 'ratio': ratio})
    psd_table = dict(frequencies=kept_frequencies, channels=spectra)
    return dict(features=features, theta_beta=ratios, psd=psd_table, exclusions=exclusions, summary=summary)


def channel_features(recording: Recording, channel: str, epochs: list, effective: dict,
                     n_used: int, n_rejected: int, filtering: FilterConfig | None = None
                     ) -> tuple[list[dict], dict, list, list]:
    """Feature rows, theta/beta row and PSD of one channel over the given clean epochs."""
    study = recording.study
    frequencies, psd = channel_psd(recording, channel, epochs, effective, filtering)
    keep = frequencies <= PSD_MAX_HZ
    rows, powers = [], {}
    for name, band in zip(config.BANDS, band_powers(frequencies, psd, config.SPECTRAL.edges)):
        powers[name] = band['absolute_uv2']
        relative = band['relative_percent']
        rows.append({**labels(study), 'channel': channel, 'band': name,
                     'abs_power_uV2': band['absolute_uv2'],
                     'rel_power_pct': '' if relative is None else relative,
                     'n_epochs_used': n_used, 'n_epochs_rejected': n_rejected})
    ratio = powers['theta'] / powers['beta'] if powers['beta'] > 0 else ''
    return rows, {**labels(study), 'channel': channel, 'ratio': ratio}, frequencies[keep].tolist(), psd[keep].tolist()


def process_per_channel(recording: Recording, scan: dict, effective: dict,
                        filtering: FilterConfig | None = None, known: dict | None = None,
                        criteria: dict | None = None) -> dict:
    """Per-channel rejection: n_epochs_used / n_epochs_rejected are per channel in the outputs."""
    study = recording.study
    kept, excluded, partial = decide_epochs_per_channel(recording, scan, known)
    criteria = criteria or {}
    complete = len([e for e in make_epochs(recording.raw.n_times, recording.sfreq) if not e.partial])
    exclusions = [exclusion(study, 'epoch', 'incomplete final epoch (only complete 5 s epochs are used)',
                            epoch=str(e.index + 1)) for e in partial]
    exclusions += [exclusion(study, 'channel', reason, channel=channel,
                             criterion=criteria.get(channel, 'peak_to_peak'))
                   for channel, reason in sorted(excluded.items())]
    used = sorted(len(epochs) for epochs in kept.values())
    median_used = used[len(used) // 2] if used else 0
    summary = dict(**labels(study), file=study.path.name, n_channels=len(recording.channels),
                   n_bad_channels=len(excluded), n_epochs_complete=complete,
                   n_epochs_used=median_used, n_epochs_rejected=complete - median_used, excluded=False)
    if not kept:
        exclusions.append(exclusion(study, 'recording', 'no channel kept enough clean epochs '
                                                        f'(per-channel rejection, minimum {config.MIN_CHANNEL_EPOCHS})'))
        summary['excluded'] = True
        return dict(features=[], theta_beta=[], psd=None, exclusions=exclusions, summary=summary)
    features, ratios, spectra, frequencies = [], [], {}, None
    for channel in recording.channels:
        if channel not in kept:
            continue
        rows, ratio, frequencies, psd = channel_features(
            recording, channel, kept[channel], effective, len(kept[channel]), complete - len(kept[channel]),
            filtering)
        features += rows
        ratios.append(ratio)
        spectra[channel] = psd
    return dict(features=features, theta_beta=ratios, psd=dict(frequencies=frequencies, channels=spectra),
                exclusions=exclusions, summary=summary)


def cache_path(cache_dir: Path, study: Study) -> Path:
    return cache_dir / f'{study.path.stem}.json'


def run_one(study: Study, cache_dir: Path, digest: str, selection=None, data_dir=None, annotations_dir=None,
            rejection_mode: str | None = None, filtering: FilterConfig | None = None) -> dict:
    """Process one recording in a worker and write its cache file; never raises."""
    started = time.monotonic()
    problem = file_problem(study)
    if problem:
        result = dict(features=[], theta_beta=[], exclusions=[exclusion(study, 'recording', problem)],
                      summary={**labels(study), 'file': study.path.name, 'excluded': True})
    else:
        try:
            recording = open_recording(study)
        except Exception as exc:   # incomplete .fdt, unreadable header, ...
            result = dict(features=[], theta_beta=[],
                          exclusions=[exclusion(study, 'recording', f'cannot read recording: {exc}')],
                          summary={**labels(study), 'file': study.path.name, 'excluded': True})
        else:
            try:
                if selection is not None:
                    from tps.core.ica import apply_snapshot
                    apply_snapshot(recording, data_dir, annotations_dir, selection)
                result = process_recording(recording, rejection_mode, filtering,
                                           (selection or {}).get('bad_channels'))
                from tps.core.ica import context
                result['ica'] = context(recording)
            except Exception as exc:   # non-finite samples, filter too long, ...
                result = dict(features=[], theta_beta=[],
                              exclusions=[exclusion(study, 'recording', f'processing failed: {exc}')],
                              summary={**labels(study), 'file': study.path.name, 'excluded': True})
            finally:
                recording.close()
    result['config_hash'] = digest
    result['input_selection'] = selection
    result['summary']['processing_mode'] = selection['mode'] if selection else 'original'
    result['summary']['filter'] = (filtering or config.FILTER).label
    result['summary']['ica_revision'] = (result.get('ica') or {}).get('revision', '')
    result['seconds'] = round(time.monotonic() - started, 1)
    path = cache_path(cache_dir, study)
    temporary = path.with_suffix('.tmp')
    # Write then rename, so an interrupted run never leaves a half-written cache file.
    temporary.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return result


def finished(path: Path, digest: str, selection=None) -> bool:
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        return saved.get('config_hash') == digest and (selection is None or saved.get('input_selection') == selection)
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------------------- outputs
def write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> int:
    count = 0
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_session_order(data_dir: Path, out_dir: Path) -> str:
    """Copy session order / EEG sampling times from participants.tsv, if present."""
    source = data_dir / 'participants.tsv'
    path = out_dir / 'session_order.csv'
    if not source.is_file():
        return 'unavailable: participants.tsv not found'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if not rows or 'SessionOrder' not in rows[0]:
        return 'unavailable: participants.tsv has no SessionOrder column'
    times = ['EEG_SamplingTime_Open_NS', 'EEG_SamplingTime_Closed_NS',
             'EEG_SamplingTime_Open_SD', 'EEG_SamplingTime_Closed_SD']
    out = []
    for row in rows:
        order = row.get('SessionOrder', 'n/a')
        first = {'NS->SD': 'NS', 'SD->NS': 'SD'}.get(order, '')
        out.append({'sub': row['participant_id'], 'session_order': order, 'first_condition': first,
                    **{name: row.get(name, '') for name in times}})
    write_csv(path, ['sub', 'session_order', 'first_condition', *times], out)
    return f'written from participants.tsv ({len(out)} participants)'


def write_psd(path: Path, results: list[dict]) -> None:
    """Wide PSD table: one row per recording and channel, one column per frequency (µV²/Hz)."""
    with_psd = [r for r in results if r.get('psd') and r['psd'].get('frequencies')]
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if not with_psd:
            writer.writerow(['sub', 'ses', 'condition', 'eyes', 'channel'])
            return
        frequencies = with_psd[0]['psd']['frequencies']
        writer.writerow(['sub', 'ses', 'condition', 'eyes', 'channel', *[f'{f:g}' for f in frequencies]])
        for result in sorted(with_psd, key=lambda r: (r['summary']['sub'], r['summary']['ses'], r['summary']['eyes'])):
            s = result['summary']
            for channel, values in result['psd']['channels'].items():
                writer.writerow([s['sub'], s['ses'], s['condition'], s['eyes'], channel,
                                 *[f'{v:.6g}' for v in values]])


def aggregate(cache_dir: Path, out_dir: Path, digest: str, pair_notes: list[dict], selections=None,
              filtering: FilterConfig | None = None) -> dict:
    """Merge every up-to-date cached recording into the final CSV files."""
    results = []
    for path in sorted(cache_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if data.get('config_hash') == digest and (selections is None or
                (path.stem in selections and data.get('input_selection') == selections[path.stem])):
            results.append(data)
    order = lambda r: (r['sub'], r['ses'], r['eyes'])
    features = [row for r in sorted(results, key=lambda r: order(r['summary'])) for row in r['features']]
    ratios = [row for r in sorted(results, key=lambda r: order(r['summary'])) for row in r['theta_beta']]
    exclusions = [row for r in results for row in r['exclusions']] + pair_notes
    level_rank = {'pair': 0, 'recording': 1, 'channel': 2, 'epoch': 3}
    exclusions.sort(key=lambda r: (r['sub'], r['ses'], r['eyes'], level_rank.get(r['level'], 9),
                                   r['channel'], int(r['epoch'] or 0)))
    write_csv(out_dir / 'features_long.csv', FEATURE_COLUMNS, features)
    write_csv(out_dir / 'theta_beta.csv', THETA_BETA_COLUMNS, ratios)
    write_psd(out_dir / 'psd_channels.csv', results)
    write_csv(out_dir / 'exclusions.csv', EXCLUSION_COLUMNS, exclusions)
    summaries = [r['summary'] for r in results]
    modes = sorted({r.get('processing_mode', 'original') for r in summaries})
    filtering = filtering or config.FILTER
    write = dict(processing_modes=modes, filter=filtering.label, filter_settings=asdict(filtering),
                 recordings=len(results),
                 confirmed=sum(bool(r.get('ica')) for r in results),
                 excluded=sum(bool(r['excluded']) for r in summaries),
                 corrections=[r['ica'] for r in results if r.get('ica')])
    (out_dir / 'ica_coverage.json').write_text(json.dumps(write, indent=2), encoding='utf-8')
    write_csv(out_dir / 'recordings_summary.csv',
              ['sub', 'ses', 'condition', 'eyes', 'file', 'excluded', 'n_channels', 'n_bad_channels',
               'n_epochs_complete', 'n_epochs_used', 'n_epochs_rejected', 'processing_mode', 'ica_revision',
               'filter'],
              sorted(summaries, key=order))
    return dict(recordings=len(results), excluded=sum(s['excluded'] for s in summaries),
                feature_rows=len(features), exclusion_rows=len(exclusions))


# ----------------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ica-mode', choices=('original', 'confirmed'), default='original')
    parser.add_argument('--annotations-dir', type=Path, default=config.PROJECT_DIR / 'annotations')
    parser.add_argument('--bandpass', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help=f'FIR band-pass cutoffs in Hz (default from config.py: '
                             f'{config.FILTER.low:g} {config.FILTER.high:g})')
    parser.add_argument('--no-bandpass', action='store_true', help='do not band-pass filter')
    parser.add_argument('--notch', type=float, choices=(50.0, 60.0), help='also apply a 50 or 60 Hz notch')
    parser.add_argument('--eyes', choices=('open', 'closed'), help='only this eye condition (default: both)')
    parser.add_argument('--subjects', nargs='+', metavar='SUB', help='e.g. 01 02 or sub-01 (default: all)')
    parser.add_argument('--subjects-file', type=Path, metavar='FILE',
                        help='participant IDs, one per line (e.g. subsets/group20_seed2026.txt)')
    parser.add_argument('--rejection-mode', choices=('global', 'per_channel'), default=None,
                        help=f'epoch rejection (default from config.py: {config.REJECTION_MODE})')
    parser.add_argument('--data-dir', type=Path, default=config.DATA_DIR)
    parser.add_argument('--out-dir', type=Path, default=config.RESULTS_DIR)
    parser.add_argument('--jobs', type=int, default=1, help='recordings processed in parallel')
    parser.add_argument('--force', action='store_true', help='recompute recordings already finished')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    cache_dir = out_dir / 'per_recording'
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Progress goes to the console and to results/batch.log (appended across runs).
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handlers = [logging.StreamHandler(), logging.FileHandler(out_dir / 'batch.log', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        return run(args, out_dir, cache_dir)
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


def filter_from_args(args: argparse.Namespace) -> FilterConfig:
    """The signal's filter: config.FILTER unless the command line (or the GUI) overrides it."""
    low, high = getattr(args, 'bandpass', None) or (config.FILTER.low, config.FILTER.high)
    enabled = not getattr(args, 'no_bandpass', False) and config.FILTER.enabled
    if getattr(args, 'bandpass', None):
        enabled = True
    notch = getattr(args, 'notch', None)
    return FilterConfig(enabled=enabled, low=low, high=high,
                        notch=config.FILTER.notch if notch is None else notch)


def run(args: argparse.Namespace, out_dir: Path, cache_dir: Path) -> int:
    mode = getattr(args, 'ica_mode', 'original')
    filtering = filter_from_args(args)
    annotations_dir = getattr(args, 'annotations_dir', Path(args.data_dir).parent / 'annotations')
    from tps.core.ica import snapshot
    all_studies = [s for group in discover(args.data_dir).values() for s in group]
    selections = snapshot(all_studies, args.data_dir, annotations_dir, mode)
    rejection = getattr(args, 'rejection_mode', None) or config.REJECTION_MODE
    # The default (global) rejection leaves the fingerprint unchanged, so existing caches stay valid.
    extra = '' if rejection == 'global' else f'per_channel:{config.MIN_CHANNEL_EPOCHS}'
    # A filter other than config.FILTER selects other caches (the default keeps existing ones valid).
    if filtering != config.FILTER:
        extra += f'filter:{filtering.enabled}:{filtering.low:g}:{filtering.high:g}:{filtering.notch}'
    digest = hashlib.sha256((config_hash() + mode + str(Path(annotations_dir).resolve()) + extra).encode()).hexdigest()[:12]
    (out_dir / 'config_used.json').write_text(json.dumps(
        # settings_hash: config.py alone (the GUI uses it to tell whether outputs are outdated);
        # config_hash: also mode and folders, which select the per-recording caches.
        {'config_hash': digest, 'settings_hash': config_hash(), 'created_at': datetime.now(timezone.utc).isoformat(),
         'command': ' '.join(sys.argv), 'ica_mode': mode, 'ica_inputs': selections,
         **config.as_dict(), 'filter': filtering.context(), 'filter_label': filtering.label,
         'rejection_mode': rejection, 'min_channel_epochs': config.MIN_CHANNEL_EPOCHS},
        indent=2, ensure_ascii=False), encoding='utf-8')

    subjects = list(args.subjects or [])
    if getattr(args, 'subjects_file', None):
        subjects += read_subjects_file(args.subjects_file)
    studies, _ = select_studies(args.data_dir, args.eyes, subjects or None)
    # Missing NS/SD counterparts are dataset facts, so list them for the whole dataset.
    _, pair_notes = select_studies(args.data_dir, None, None)
    todo = [s for s in studies if args.force or not finished(cache_path(cache_dir, s), digest, selections[str(s.path)])]
    log.info('%d recordings selected, %d already finished, %d to process (config %s)',
             len(studies), len(studies) - len(todo), len(todo), digest)
    log.info('Signal: %s%s', filtering.label, ' + confirmed ICA' if mode == 'confirmed' else '')
    started = time.monotonic()

    def report(done: int, study: Study, result: dict) -> None:
        s = result['summary']
        state = ('EXCLUDED: ' + result['exclusions'][-1]['reason']) if s['excluded'] else \
            f"{s['n_epochs_used']} epochs used, {s['n_epochs_rejected']} rejected, {s['n_bad_channels']} bad channels"
        eta = (time.monotonic() - started) / done * (len(todo) - done)
        log.info('[%d/%d] %s (%.0fs) %s | ETA %.0f min', done, len(todo), study.path.name,
                 result['seconds'], state, eta / 60)

    if args.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, s, cache_dir, digest, selections[str(s.path)], args.data_dir,
                                   annotations_dir, rejection, filtering): s for s in todo}
            for done, future in enumerate(as_completed(futures), 1):
                report(done, futures[future], future.result())
    else:
        for done, study in enumerate(todo, 1):
            report(done, study, run_one(study, cache_dir, digest, selections[str(study.path)], args.data_dir,
                                        annotations_dir, rejection, filtering))

    # Aggregation covers every finished recording, not just this run's selection, so
    # running --eyes open and later --eyes closed builds one combined table.
    totals = aggregate(cache_dir, out_dir, digest, pair_notes, {Path(k).stem: v for k, v in selections.items()},
                       filtering)
    order = write_session_order(args.data_dir, out_dir)
    log.info('Session order: %s', order)
    log.info('Done: %(recordings)d recordings in the tables (%(excluded)d excluded), '
             '%(feature_rows)d feature rows, %(exclusion_rows)d exclusion rows', totals)
    log.info('Outputs in %s', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
