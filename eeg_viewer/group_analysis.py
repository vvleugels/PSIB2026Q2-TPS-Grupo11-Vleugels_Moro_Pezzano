"""Participant-level group features for the ds004902 resting EEG dataset."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import mne
import numpy as np
from scipy.signal import get_window, welch

from .annotations import make_epochs
from .data import Study, discover, open_recording
from .processing import FilterConfig
from .screening import ScreeningConfig, scan_recording
from .spectral import BANDS, SpectralConfig, band_powers


DEFAULT_FILTER = FilterConfig(enabled=True, low=0.2, high=45.0)
FEATURE_COLUMNS = (
    'participant', 'session', 'session_condition', 'eyes', 'SessionOrder',
    'channel', 'band', 'absolute_uv2', 'relative_percent', 'epochs_total',
    'epochs_partial', 'epochs_rejected_by_screening', 'epochs_rejected_high',
    'epochs_rejected_flat',
    'epochs_rejected_nonfinite', 'epochs_used',
)
SUMMARY_COLUMNS = (
    'participant', 'session', 'session_condition', 'eyes', 'SessionOrder',
    'recording', 'epochs_total', 'epochs_partial', 'epochs_rejected_by_screening',
    'epochs_rejected_high', 'epochs_rejected_flat', 'epochs_rejected_nonfinite',
    'epochs_used', 'status',
)
EXCLUSION_COLUMNS = (
    'participant', 'session', 'session_condition', 'eyes', 'SessionOrder',
    'recording', 'reason',
)


def read_session_order(data_dir: Path) -> dict[str, str]:
    """Read the within-subject session sequence from BIDS participants.tsv."""
    path = Path(data_dir) / 'participants.tsv'
    if not path.is_file():
        raise FileNotFoundError(f'No se encontró {path}; se necesita SessionOrder.')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not reader.fieldnames or 'participant_id' not in reader.fieldnames or 'SessionOrder' not in reader.fieldnames:
            raise ValueError('participants.tsv debe incluir participant_id y SessionOrder.')
        rows = list(reader)
    result = {}
    for row in rows:
        participant = row['participant_id'].strip()
        order = row['SessionOrder'].strip()
        if not participant or participant in result:
            raise ValueError(f'participant_id vacío o duplicado en {path}.')
        result[participant] = order
    return result


def _participant_key(value):
    number = value.removeprefix('sub-')
    return (0, int(number)) if number.isdigit() else (1, value)


def expected_tasks(participant: str) -> tuple[str, ...]:
    match = re.fullmatch(r'sub-(\d+)', participant)
    return ('eyesopen', 'eyesclosed') if match and int(match.group(1)) <= 38 else ('eyesopen',)


def extract_recording_features(recording, session_order: str,
                               screening_config: ScreeningConfig = ScreeningConfig(),
                               filter_config: FilterConfig = DEFAULT_FILTER,
                               spectral_config: SpectralConfig = SpectralConfig()):
    """Screen globally by epoch, then average Welch PSDs over retained epochs."""
    spectral = spectral_config.effective(recording.sfreq)
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    screening = scan_recording(recording, recording.channels, screening_config, filter_config)
    findings = screening['findings']
    partial = {epoch.index for epoch in epochs if epoch.partial}
    rejected = set(findings)
    kept = [epoch for epoch in epochs if not epoch.partial and epoch.index not in rejected]

    high = {index for index, issues in findings.items() if any(issue['kind'] == 'Alta' for issue in issues)}
    flat = {index for index, issues in findings.items() if any(issue['kind'] == 'Plana' for issue in issues)}
    nonfinite = {index for index, issues in findings.items() if any(issue['kind'] == 'No finitos' for issue in issues)}
    summary = dict(
        participant=recording.study.participant,
        session=recording.study.session,
        session_condition=recording.study.condition,
        eyes=recording.study.eyes,
        SessionOrder=session_order,
        recording=recording.study.path.name,
        epochs_total=len(epochs),
        epochs_partial=len(partial),
        epochs_rejected_by_screening=len(rejected),
        epochs_rejected_high=len(high),
        epochs_rejected_flat=len(flat),
        epochs_rejected_nonfinite=len(nonfinite),
        epochs_used=len(kept),
    )
    if not kept:
        summary['status'] = 'excluded_no_usable_epochs'
        return [], summary

    window = get_window('hann', spectral['nperseg'], fftbins=True)
    features = []
    for channel in recording.channels:
        samples_continuous = recording.filter.channel(channel, filter_config)
        psd_sum = None
        for epoch in kept:
            samples = samples_continuous[epoch.first:epoch.last]
            if len(samples) < spectral['nperseg']:
                raise ValueError(f'Epoch {epoch.index + 1}: menos muestras que el segmento Welch.')
            if not np.all(np.isfinite(samples)):
                raise ValueError(f'Epoch {epoch.index + 1}, canal {channel}: muestras no finitas tras filtrar.')
            frequencies, psd = welch(
                samples, fs=recording.sfreq, window=window,
                nperseg=spectral['nperseg'], noverlap=spectral['noverlap'],
                nfft=spectral['nfft'], detrend='constant', return_onesided=True,
                scaling='density', average='mean',
            )
            psd_sum = psd.copy() if psd_sum is None else psd_sum + psd
        mean_psd_uv2_hz = psd_sum / len(kept) * 1e12
        for band in band_powers(frequencies, mean_psd_uv2_hz, spectral_config.edges):
            features.append(dict(
                participant=recording.study.participant,
                session=recording.study.session,
                session_condition=recording.study.condition,
                eyes=recording.study.eyes,
                SessionOrder=session_order,
                channel=channel,
                band=band['band'],
                absolute_uv2=band['absolute_uv2'],
                relative_percent=band['relative_percent'],
                epochs_total=len(epochs),
                epochs_partial=len(partial),
                epochs_rejected_amplitude=len(rejected),
                epochs_rejected_flat=len(flat),
                epochs_rejected_nonfinite=len(nonfinite),
                epochs_used=len(kept),
            ))
    summary['status'] = 'ok'
    return features, summary


def _write_csv(path: Path, columns, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run_group_analysis(data_dir='data', output_dir=None, max_participants=None,
                       participants=None, screening_config=ScreeningConfig(),
                       filter_config=DEFAULT_FILTER, spectral_config=SpectralConfig()):
    """Process requested BIDS recordings and write fresh CSV outputs."""
    data_dir = Path(data_dir).resolve()
    orders = read_session_order(data_dir)
    inventory = discover(data_dir)
    roster = sorted(orders, key=_participant_key)
    selected = list(participants) if participants else roster
    unknown = sorted(set(selected) - set(roster))
    if unknown:
        raise ValueError(f'Participantes no presentes en participants.tsv: {unknown}')
    if len(set(selected)) != len(selected):
        raise ValueError('La lista de participantes contiene duplicados.')
    selected = sorted(selected, key=_participant_key)
    if max_participants is not None:
        if max_participants < 1:
            raise ValueError('--max-participants debe ser mayor que cero.')
        selected = selected[:max_participants]
    screening_config.validate()

    features, summaries, exclusions = [], [], []
    for participant in selected:
        for session in ('ses-1', 'ses-2'):
            for task in expected_tasks(participant):
                matches = [study for study in inventory.get(participant, [])
                           if study.session == session and study.task == task]
                context = Study(Path(), participant, session, task)
                common = dict(participant=participant, session=session,
                              session_condition=context.condition, eyes=context.eyes,
                              SessionOrder=orders[participant],
                              recording=f'{participant}_{session}_task-{task}_eeg.set')
                if not matches:
                    exclusions.append({**common, 'reason': 'No se encontró el archivo .set esperado.'})
                    continue
                if len(matches) > 1:
                    exclusions.append({**common, 'reason': f'Se encontraron {len(matches)} registros para la misma sesión y tarea.'})
                    continue
                study = matches[0]
                try:
                    recording = open_recording(study)
                    try:
                        rows, record_summary = extract_recording_features(
                            recording, orders[participant], screening_config,
                            filter_config, spectral_config)
                    finally:
                        recording.close()
                    summaries.append(record_summary)
                    if rows:
                        features.extend(rows)
                    else:
                        exclusions.append({**common, 'recording': study.path.name,
                                           'reason': 'No quedaron epochs completas utilizables.'})
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    summaries.append({**common, 'epochs_total': '', 'epochs_partial': '',
                                      'epochs_rejected_by_screening': '', 'epochs_rejected_high': '',
                                      'epochs_rejected_flat': '', 'epochs_rejected_nonfinite': '',
                                      'epochs_used': 0, 'status': 'excluded'})
                    exclusions.append({**common, 'recording': study.path.name,
                                       'reason': f'{type(exc).__name__}: {exc}'})

    features.sort(key=lambda row: (_participant_key(row['participant']), row['session'], row['eyes'], row['channel'], BANDS.index(row['band'])))
    summaries.sort(key=lambda row: (_participant_key(row['participant']), row['session'], row['eyes']))
    exclusions.sort(key=lambda row: (_participant_key(row['participant']), row['session'], row['eyes']))
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_dir = data_dir.parent / 'results' / f'group-analysis-{stamp}'
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / 'features.csv', FEATURE_COLUMNS, features)
    _write_csv(output_dir / 'record_summary.csv', SUMMARY_COLUMNS, summaries)
    _write_csv(output_dir / 'exclusions.csv', EXCLUSION_COLUMNS, exclusions)
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(), data_dir=str(data_dir),
        participants=selected, filter=filter_config.context(),
        screening=asdict(screening_config), spectral=asdict(spectral_config),
        versions=dict(mne=mne.__version__, numpy=np.__version__),
        output=dict(feature_rows=len(features), record_rows=len(summaries), exclusions=len(exclusions)),
    )
    (output_dir / 'analysis.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return dict(output_dir=output_dir, features=features, summaries=summaries,
                exclusions=exclusions, metadata=metadata)
