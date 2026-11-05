"""Batch band-power extraction for every subject, session and eye condition (no GUI).

Run from the project folder:  python -m batch.run_batch [--eyes open|closed] [--subjects 01 02]

Reuses the GUI modules for every computation step (BIDS inventory, loading, continuous
FIR filtering, 5 s epochs, peak-to-peak screening, Welch and band integration), so a
recording analysed here and in the GUI gives the same numbers for the same settings.
Results are cached per recording, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Iterable

if __package__ in (None, ''):   # allow "python batch/run_batch.py" as well as "-m"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from eeg_viewer.annotations import make_epochs
from eeg_viewer.data import Recording, Study, discover, open_recording
from eeg_viewer.screening import scan_recording
from eeg_viewer.spectral import band_powers, welch_epoch

log = logging.getLogger('batch')

FEATURE_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'channel', 'band', 'abs_power_uV2',
                   'rel_power_pct', 'n_epochs_used', 'n_epochs_rejected']
THETA_BETA_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'channel', 'ratio']
EXCLUSION_COLUMNS = ['sub', 'ses', 'condition', 'eyes', 'level', 'channel', 'epoch', 'reason']
# Bumped whenever the per-recording output format changes, so old caches are recomputed.
CACHE_FORMAT = 2   # 2: per-channel mean PSD added for the figures
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


def exclusion(study: Study, level: str, reason: str, channel: str = '', epoch: str = '') -> dict:
    return {**labels(study), 'level': level, 'channel': channel, 'epoch': epoch, 'reason': reason}


# ----------------------------------------------------------------------------- inventory
def annex_placeholder(path: Path) -> bool:
    """True for a git-annex pointer/symlink left without local content.

    On Windows (no symlinks) git checks such files out as tiny text files whose content
    is the link target, e.g. '../../.git/annex/objects/...'."""
    if path.is_symlink():
        return not path.exists()
    try:
        if path.stat().st_size > 1024:
            return False
        return b'annex/objects' in path.read_bytes()
    except OSError:
        return False


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


def decide_epochs(recording: Recording, scan: dict) -> EpochDecision:
    """Turn per-channel screening findings into bad channels and rejected epochs."""
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    complete = [e for e in epochs if not e.partial]
    partial = [e for e in epochs if e.partial]
    findings = scan['findings']
    flagged: dict[str, int] = {}
    for epoch in complete:
        for issue in findings.get(epoch.index, []):
            flagged[issue['channel']] = flagged.get(issue['channel'], 0) + 1
    bad = {}
    limit = config.BAD_CHANNEL_MAX_FRACTION
    if limit is not None and complete:
        for channel, count in flagged.items():
            if count / len(complete) > limit:
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


def channel_psd(recording: Recording, channel: str, epochs: list, effective: dict
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD (µV²/Hz) over the given epochs of the filtered continuous channel."""
    # The cache was filled by scan_recording, so the channel is filtered only once.
    continuous = recording.filter.channel(channel, config.FILTER)
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


def process_recording(recording: Recording) -> dict:
    """Screen, epoch and compute band powers for every good channel of one open recording.

    Returns a JSON-serialisable dict with feature rows, theta/beta rows, exclusions and a summary.
    """
    study = recording.study
    effective = config.SPECTRAL.effective(recording.sfreq)
    # Screening runs on the filtered signal over ALL EEG channels.
    scan = scan_recording(recording, recording.channels, config.SCREENING, config.FILTER)
    decision = decide_epochs(recording, scan)
    exclusions = [exclusion(study, 'epoch', 'incomplete final epoch (only complete 5 s epochs are used)',
                            epoch=str(e.index + 1)) for e in decision.partial]
    exclusions += [exclusion(study, 'channel', reason, channel=channel)
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
        frequencies, psd = channel_psd(recording, channel, decision.used, effective)
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


def cache_path(cache_dir: Path, study: Study) -> Path:
    return cache_dir / f'{study.path.stem}.json'


def run_one(study: Study, cache_dir: Path, digest: str) -> dict:
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
                result = process_recording(recording)
            except Exception as exc:   # non-finite samples, filter too long, ...
                result = dict(features=[], theta_beta=[],
                              exclusions=[exclusion(study, 'recording', f'processing failed: {exc}')],
                              summary={**labels(study), 'file': study.path.name, 'excluded': True})
            finally:
                recording.close()
    result['config_hash'] = digest
    result['seconds'] = round(time.monotonic() - started, 1)
    path = cache_path(cache_dir, study)
    temporary = path.with_suffix('.tmp')
    # Write then rename, so an interrupted run never leaves a half-written cache file.
    temporary.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return result


def finished(path: Path, digest: str) -> bool:
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('config_hash') == digest
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


def aggregate(cache_dir: Path, out_dir: Path, digest: str, pair_notes: list[dict]) -> dict:
    """Merge every up-to-date cached recording into the final CSV files."""
    results = []
    for path in sorted(cache_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if data.get('config_hash') == digest:
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
    write_csv(out_dir / 'recordings_summary.csv',
              ['sub', 'ses', 'condition', 'eyes', 'file', 'excluded', 'n_channels', 'n_bad_channels',
               'n_epochs_complete', 'n_epochs_used', 'n_epochs_rejected'],
              sorted(summaries, key=order))
    return dict(recordings=len(results), excluded=sum(s['excluded'] for s in summaries),
                feature_rows=len(features), exclusion_rows=len(exclusions))


# ----------------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--eyes', choices=('open', 'closed'), help='only this eye condition (default: both)')
    parser.add_argument('--subjects', nargs='+', metavar='SUB', help='e.g. 01 02 or sub-01 (default: all)')
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


def run(args: argparse.Namespace, out_dir: Path, cache_dir: Path) -> int:
    digest = config_hash()
    (out_dir / 'config_used.json').write_text(json.dumps(
        {'config_hash': digest, 'created_at': datetime.now(timezone.utc).isoformat(),
         'command': ' '.join(sys.argv), **config.as_dict()}, indent=2, ensure_ascii=False), encoding='utf-8')

    studies, _ = select_studies(args.data_dir, args.eyes, args.subjects)
    # Missing NS/SD counterparts are dataset facts, so list them for the whole dataset.
    _, pair_notes = select_studies(args.data_dir, None, None)
    todo = [s for s in studies if args.force or not finished(cache_path(cache_dir, s), digest)]
    log.info('%d recordings selected, %d already finished, %d to process (config %s)',
             len(studies), len(studies) - len(todo), len(todo), digest)
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
            futures = {pool.submit(run_one, s, cache_dir, digest): s for s in todo}
            for done, future in enumerate(as_completed(futures), 1):
                report(done, futures[future], future.result())
    else:
        for done, study in enumerate(todo, 1):
            report(done, study, run_one(study, cache_dir, digest))

    # Aggregation covers every finished recording, not just this run's selection, so
    # running --eyes open and later --eyes closed builds one combined table.
    totals = aggregate(cache_dir, out_dir, digest, pair_notes)
    order = write_session_order(args.data_dir, out_dir)
    log.info('Session order: %s', order)
    log.info('Done: %(recordings)d recordings in the tables (%(excluded)d excluded), '
             '%(feature_rows)d feature rows, %(exclusion_rows)d exclusion rows', totals)
    log.info('Outputs in %s', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
