"""Epoch-wise Welch analysis and reproducible exports, independent of Tk."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import platform
import re
import shutil
import tempfile

import mne
import numpy as np
import scipy
from scipy.signal import get_window, welch
from scipy.integrate import trapezoid

from .annotations import AnnotationStore, STATES, source_identity
from .data import open_recording

BANDS = ('Delta', 'Theta', 'Alfa', 'Beta')


@dataclass(frozen=True)
class SpectralConfig:
    seconds: float = 2.0
    overlap: float = 50.0
    edges: tuple = (.5, 4., 8., 13., 30.)

    def effective(self, sfreq):
        if not math.isfinite(self.seconds) or not 0 < self.seconds <= 5:
            raise ValueError('El segmento Welch debe durar más de 0 y hasta 5 s.')
        if not math.isfinite(self.overlap) or not 0 <= self.overlap < 100:
            raise ValueError('El solapamiento debe estar entre 0 y menos de 100 %.')
        n = round(self.seconds * sfreq)
        overlap = math.floor(n * self.overlap / 100)
        if n < 2 or overlap >= n:
            raise ValueError('Se requieren al menos dos muestras y solapamiento menor al segmento.')
        maximum = (n // 2) * sfreq / n
        if (len(self.edges) != 5 or not all(math.isfinite(v) for v in self.edges)
                or not 0 <= self.edges[0] or self.edges[-1] > maximum
                or any(a >= b for a, b in zip(self.edges, self.edges[1:]))):
            raise ValueError(f'Ingresá cinco fronteras crecientes entre 0 y {maximum:g} Hz.')
        return dict(nperseg=n, noverlap=overlap, nfft=n, seconds=n / sfreq,
                    overlap_percent=100 * overlap / n, frequency_step=sfreq / n,
                    window='hann_periodic', detrend='constant', scaling='density',
                    return_onesided=True, average='mean')


def integrate_band(frequencies, psd, low, high):
    """Integrate a continuous piecewise-linear PSD at the exact requested edges."""
    if not frequencies[0] <= low < high <= frequencies[-1]:
        raise ValueError('La banda está fuera del rango espectral disponible.')
    inside = (frequencies > low) & (frequencies < high)
    x = np.concatenate(([low], frequencies[inside], [high]))
    y = np.concatenate(([np.interp(low, frequencies, psd)], psd[inside],
                        [np.interp(high, frequencies, psd)]))
    return float(trapezoid(y, x))


def band_powers(frequencies, psd, edges):
    total = integrate_band(frequencies, psd, edges[0], edges[-1])
    rows = []
    for name, low, high in zip(BANDS, edges, edges[1:]):
        power = integrate_band(frequencies, psd, low, high)
        rows.append(dict(band=name, low_hz=low, high_hz=high, absolute_uv2=power,
                         relative_percent=100 * power / total if total > 0 else None))
    return rows


def psd_db(psd):
    # Zero remains undefined in the displayed logarithmic curve, never a fake floor.
    result = np.full_like(psd, np.nan, dtype=float)
    positive = psd > 0
    result[positive] = 10 * np.log10(psd[positive])
    return result


def select_studies(current, inventory, compare):
    if not compare:
        return [current]
    selected = []
    for session in ('ses-1', 'ses-2'):
        matches = [s for s in inventory.get(current.participant, [])
                   if s.task == current.task and s.session == session]
        if len(matches) != 1:
            raise ValueError(f'No hay un estudio único de {session} / {current.eyes}. '
                             'Podés analizar el estudio actual.')
        selected.append(matches[0])
    return selected


def verify_store(store):
    if store.blocked:
        raise ValueError(store.blocked)
    if source_identity(store.recording, store.data_dir) != store.identity:
        raise ValueError('El registro cambió durante el cálculo. Volvé a cargarlo.')
    content = store.path.read_bytes() if store.path.exists() else None
    digest = store.checksum(content) if content is not None else None
    if digest != store.digest:
        raise ValueError('Las anotaciones cambiaron durante el cálculo. Volvé a calcular.')


def analyze_recording(recording, channel, filter_config, config, store, include_unreviewed=False):
    """Read only selected epochs; filtering always operates on the continuous channel."""
    verify_store(store)
    if store.dirty:
        raise ValueError('Guardá las anotaciones antes de calcular.')
    if channel not in recording.channels:
        raise ValueError(f'{recording.study.label}: el electrodo {channel} no está disponible.')
    effective = config.effective(recording.sfreq)
    eligible = []
    counts = dict(used=0, seconds=0., artifacts=0, pending=0, partial=0, accepted=0)
    mismatches = []
    for epoch, record in zip(store.epochs, store.records):
        if epoch.partial:
            counts['partial'] += 1
            continue
        if record['state'] == STATES[2]:
            counts['artifacts'] += 1
            continue
        if record['state'] == STATES[0]:
            counts['pending'] += 1
            if not include_unreviewed:
                continue
        else:
            counts['accepted'] += 1
        eligible.append(epoch)
        prior = record['filter']
        if prior and (prior['enabled'] != filter_config.enabled or
                      (filter_config.enabled and (prior['low'], prior['high']) !=
                       (filter_config.low, filter_config.high))):
            mismatches.append(epoch.index + 1)
    if not eligible:
        raise ValueError(f'{recording.study.label}: no hay epochs elegibles. '
                         'Revisá y guardá epochs o activá la inclusión exploratoria.')
    if any(e.last - e.first < effective['nperseg'] for e in eligible):
        raise ValueError('El segmento Welch supera las muestras de una epoch completa.')
    if filter_config.enabled:
        continuous = recording.filter.channel(channel, filter_config)
    else:
        continuous = None
    total = None
    window = get_window('hann', effective['nperseg'], fftbins=True)
    for epoch in eligible:
        samples = (continuous[epoch.first:epoch.last] if continuous is not None else
                   recording.raw.get_data(picks=[channel], start=epoch.first, stop=epoch.last)[0])
        if not np.all(np.isfinite(samples)):
            raise ValueError(f'Epoch {epoch.index + 1}: hay muestras no finitas; no se omiten automáticamente.')
        frequencies, psd = welch(samples, fs=recording.sfreq, window=window,
                                 nperseg=effective['nperseg'], noverlap=effective['noverlap'],
                                 nfft=effective['nfft'], detrend='constant',
                                 return_onesided=True, scaling='density', average='mean')
        total = psd.copy() if total is None else total + psd
    averaged = total / len(eligible) * 1e12
    if not np.all(np.isfinite(averaged)):
        raise ValueError('El cálculo produjo potencias no finitas; revisá los datos de entrada.')
    counts['used'] = len(eligible)
    counts['seconds'] = sum((e.last - e.first) / recording.sfreq for e in eligible)
    warnings = []
    if mismatches:
        warnings.append('Marcas realizadas con otro filtrado: epochs ' + ', '.join(map(str, mismatches)))
    if filter_config.enabled:
        attenuated = [name for name, low, high in zip(BANDS, config.edges, config.edges[1:])
                      if low < filter_config.low or high > filter_config.high]
        if attenuated:
            warnings.append('Bandas parcialmente fuera del pasabanda: ' + ', '.join(attenuated))
    verify_store(store)
    return dict(participant=recording.study.participant, session=recording.study.session,
                condition=recording.study.condition, task=recording.study.task,
                eyes=recording.study.eyes, channel=channel, sfreq=recording.sfreq,
                reference=recording.metadata.get('EEGReference') or 'No disponible',
                source=store.identity, annotations_sha256=store.digest,
                annotation_file=store.path.name, frequencies=frequencies, psd_uv2_hz=averaged,
                bands=band_powers(frequencies, averaged, config.edges), counts=counts,
                epochs=[dict(number=e.index + 1, first=e.first, last=e.last,
                             review=store.records[e.index]) for e in eligible],
                effective=effective, warnings=warnings)


def run_analysis(studies, channel, filter_config, config, data_dir, annotations_dir,
                 include_unreviewed=False, expected=None):
    """Own all auxiliary readers within the worker and return a complete comparison only."""
    if not studies or len({(s.participant, s.task) for s in studies}) != 1:
        raise ValueError('La comparación requiere el mismo participante y condición de ojos.')
    with ExitStack() as stack:
        recordings, stores = [], []
        for study in studies:
            recording = open_recording(study)
            stack.callback(recording.close)
            store = AnnotationStore(annotations_dir, recording, data_dir)
            verify_store(store)
            if expected and str(study.path) in expected:
                identity, digest = expected[str(study.path)]
                if identity != store.identity or digest != store.digest:
                    raise ValueError('El estudio o sus anotaciones cambiaron externamente. Volvé a cargarlo.')
            recordings.append(recording)
            stores.append(store)
        if len({r.sfreq for r in recordings}) != 1:
            raise ValueError('Las sesiones tienen distintas frecuencias de muestreo. No se remuestrean.')
        sessions = [analyze_recording(r, channel, filter_config, config, store, include_unreviewed)
                    for r, store in zip(recordings, stores)]
        for store in stores:
            verify_store(store)
    return dict(created_at=datetime.now(timezone.utc).isoformat(),
                exploratory=include_unreviewed, filter=filter_config.context(),
                spectral=asdict(config), sessions=sessions,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, mne=mne.__version__))


def plot_result(figure, result, db=True):
    figure.clear()
    ax = figure.add_subplot(111)
    colors = {'ses-1': '#0e7c86', 'ses-2': '#b94b6b'}
    for session in result['sessions']:
        values = psd_db(session['psd_uv2_hz']) if db else session['psd_uv2_hz']
        ax.plot(session['frequencies'], values, color=colors.get(session['session']),
                label=f"{session['condition']} (n={session['counts']['used']})", linewidth=1.2)
    filtering = result['filter']
    upper = filtering['high'] + 5 if filtering['enabled'] else max(45, result['spectral']['edges'][-1])
    upper = min(upper, min(s['frequencies'][-1] for s in result['sessions']))
    ax.set_xlim(0, upper)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('PSD (dB)\nre 1 µV²/Hz' if db else 'PSD (µV²/Hz)')
    first = result['sessions'][0]
    ax.set_title(f"{first['participant']} · {first['eyes']} · {first['channel']}"
                 + (' · Exploratorio' if result['exploratory'] else ''), fontsize=10)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    return ax


def export_result(result, parent, include_png=False, db=True):
    """A new directory per export; never overwrite existing results."""
    from matplotlib.figure import Figure
    first = result['sessions'][0]
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{first['participant']}_{first['task']}_{first['channel']}")
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = Path(tempfile.mkdtemp(prefix=f'{safe}_{stamp}_', dir=parent))
    try:
        with (directory / 'PSD.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['participant', 'session', 'task', 'electrode', 'frequency_hz', 'psd_uv2_hz'])
            for s in result['sessions']:
                writer.writerows((s['participant'], s['session'], s['task'], s['channel'], f, p)
                                 for f, p in zip(s['frequencies'], s['psd_uv2_hz']))
        with (directory / 'potencias.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['participant', 'session', 'task', 'electrode', 'band', 'low_hz',
                             'high_hz', 'absolute_uv2', 'relative_percent'])
            for s in result['sessions']:
                for band in s['bands']:
                    writer.writerow([s['participant'], s['session'], s['task'], s['channel'],
                                     *[band[k] for k in ('band', 'low_hz', 'high_hz',
                                                         'absolute_uv2', 'relative_percent')]])
        metadata = {**result, 'sessions': [{k: v for k, v in s.items()
                                           if k not in ('frequencies', 'psd_uv2_hz')}
                                          for s in result['sessions']]}
        metadata['plot_scale'] = 'dB' if db else 'linear'
        (directory / 'configuracion.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if include_png:
            figure = Figure(figsize=(10, 5), layout='constrained')
            plot_result(figure, result, db)
            figure.savefig(directory / 'PSD.png', dpi=180)
    except Exception:
        shutil.rmtree(directory)
        raise
    return directory
