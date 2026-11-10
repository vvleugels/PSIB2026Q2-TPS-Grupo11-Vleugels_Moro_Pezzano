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

from tps.core.annotations import AnnotationStore, STATES, source_identity
from tps.core.data import open_recording

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


def welch_epoch(samples, sfreq, effective):
    """Welch PSD of one epoch (V²/Hz), with the parameters from SpectralConfig.effective.

    Shared by the GUI and the batch pipeline so both compute the spectrum identically."""
    window = get_window('hann', effective['nperseg'], fftbins=True)
    return welch(samples, fs=sfreq, window=window, nperseg=effective['nperseg'],
                 noverlap=effective['noverlap'], nfft=effective['nfft'], detrend='constant',
                 return_onesided=True, scaling='density', average='mean', axis=-1)


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
    ica_mismatches = []
    from tps.core.ica import context as ica_context
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
        if record.get('ica') != ica_context(recording):
            ica_mismatches.append(epoch.index + 1)
        prior = record['filter']
        if prior and (prior.get('notch') != filter_config.notch or prior['enabled'] != filter_config.enabled or
                      (filter_config.enabled and (prior['low'], prior['high']) !=
                       (filter_config.low, filter_config.high))):
            mismatches.append(epoch.index + 1)
    if not eligible:
        raise ValueError(f'{recording.study.label}: no hay épocas elegibles. '
                         'Revisá y guardá épocas o activá la inclusión exploratoria.')
    if any(e.last - e.first < effective['nperseg'] for e in eligible):
        raise ValueError('El segmento Welch supera las muestras de una época completa.')
    if filter_config.active:
        continuous = recording.filter.channel(channel, filter_config)
    else:
        continuous = None
    total = None
    for epoch in eligible:
        samples = (continuous[epoch.first:epoch.last] if continuous is not None else
                   recording.signal_raw.get_data(picks=[channel], start=epoch.first, stop=epoch.last)[0])
        if not np.all(np.isfinite(samples)):
            raise ValueError(f'Época {epoch.index + 1}: hay muestras no finitas; no se omiten automáticamente.')
        frequencies, psd = welch_epoch(samples, recording.sfreq, effective)
        total = psd.copy() if total is None else total + psd
    averaged = total / len(eligible) * 1e12
    if not np.all(np.isfinite(averaged)):
        raise ValueError('El cálculo produjo potencias no finitas; revisá los datos de entrada.')
    counts['used'] = len(eligible)
    counts['seconds'] = sum((e.last - e.first) / recording.sfreq for e in eligible)
    warnings = []
    if ica_mismatches:
        warnings.append('Marcas realizadas con otra ICA: épocas ' + ', '.join(map(str, ica_mismatches)))
    if mismatches:
        warnings.append('Marcas realizadas con otro filtrado: épocas ' + ', '.join(map(str, mismatches)))
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
                source=store.identity, ica=ica_context(recording), annotations_sha256=store.digest,
                annotation_file=store.path.name, frequencies=frequencies, psd_uv2_hz=averaged,
                bands=band_powers(frequencies, averaged, config.edges), counts=counts,
                epochs=[dict(number=e.index + 1, first=e.first, last=e.last,
                             review=store.records[e.index]) for e in eligible],
                effective=effective, warnings=warnings)


def run_analysis(studies, channel, filter_config, config, data_dir, annotations_dir,
                 include_unreviewed=False, expected=None, ica_selections=None):
    """Own all auxiliary readers within the worker and return a complete comparison only."""
    if not studies or len({(s.participant, s.task) for s in studies}) != 1:
        raise ValueError('La comparación requiere el mismo participante y condición de ojos.')
    with ExitStack() as stack:
        recordings, stores = [], []
        for study in studies:
            recording = open_recording(study)
            stack.callback(recording.close)
            if ica_selections is not None:
                from tps.core.ica import apply_snapshot
                apply_snapshot(recording, data_dir, annotations_dir, ica_selections[str(study.path)])
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


def standard_info(channels):
    """mne.Info with an MNE standard 10-05 montage that contains the dataset's channel names."""
    info = mne.create_info(list(channels), 500.0, 'eeg')   # sfreq is irrelevant for a topomap
    available = mne.channels.get_builtin_montages()
    # 'standard_1005' was renamed 'colin27_1005' in recent MNE; use whichever exists.
    name = 'colin27_1005' if 'colin27_1005' in available else 'standard_1005'
    info.set_montage(mne.channels.make_standard_montage(name))
    return info


def montage_channels(channels):
    """The channels that have a position in the standard 10-05 montage, in the given order."""
    available = mne.channels.get_builtin_montages()
    name = 'colin27_1005' if 'colin27_1005' in available else 'standard_1005'
    known = set(mne.channels.make_standard_montage(name).ch_names)
    return [c for c in channels if c in known]


OUTLIER_Z = 3.0   # robust z of high-frequency (30-45 Hz) log power above which a channel is highlighted


def channel_spectra(recording, filter_config, config, store, include_unreviewed=False):
    """Welch PSD of every EEG channel over the same eligible épocas (EEGLAB «channel spectra»).

    Reuses analyze_recording per channel, so the numbers equal the single-channel analysis.
    Channels whose mean log power in 30-45 Hz is a robust outlier (z > OUTLIER_Z) are listed
    in 'outliers'; channels that cannot be analysed (non-finite samples) in 'skipped'."""
    from tps.core.badchannels import robust_z
    if filter_config.active:
        recording.filter.prefetch(recording.channels, filter_config)   # one block filter pass
    sessions, skipped = [], []
    for channel in recording.channels:
        try:
            sessions.append(analyze_recording(recording, channel, filter_config, config, store,
                                              include_unreviewed))
        except ValueError as exc:
            if 'no finitas' not in str(exc):
                raise   # no eligible épocas, dirty annotations…: the same for every channel
            skipped.append(channel)
    if not sessions:
        raise ValueError('Ningún canal se pudo analizar.')
    frequencies = sessions[0]['frequencies']
    psd = np.array([s['psd_uv2_hz'] for s in sessions])
    upper = min(45.0, filter_config.high if filter_config.enabled else 45.0)
    band = (frequencies >= 30) & (frequencies <= upper)
    outliers = []
    if band.any() and len(sessions) >= 5:
        z = robust_z(np.log10(np.clip(psd[:, band], 1e-12, None)).mean(axis=1))
        outliers = [s['channel'] for s, value in zip(sessions, z) if value > OUTLIER_Z]
    first = sessions[0]
    return dict(kind='channels', created_at=datetime.now(timezone.utc).isoformat(),
                exploratory=include_unreviewed, filter=filter_config.context(),
                spectral=asdict(config), participant=first['participant'], session=first['session'],
                condition=first['condition'], task=first['task'], eyes=first['eyes'],
                channels=[s['channel'] for s in sessions], frequencies=frequencies, psd_uv2_hz=psd,
                bands={s['channel']: s['bands'] for s in sessions}, counts=first['counts'],
                warnings=first['warnings'], effective=first['effective'], reference=first['reference'],
                ica=first['ica'], outliers=outliers, skipped=skipped,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, mne=mne.__version__))


def plot_channel_spectra(figure, result, db=True, bad=()):
    """All channels' PSD (median in bold, bad / outlier channels in red) and band-power maps."""
    figure.clear()
    grid = figure.add_gridspec(2, len(BANDS), height_ratios=(1.5, 1))
    ax = figure.add_subplot(grid[0, :])
    frequencies = result['frequencies']
    psd = result['psd_uv2_hz']
    values = psd_db(psd) if db else psd
    flagged = set(result['outliers']) | set(bad)
    for channel, row in zip(result['channels'], values):
        red = channel in flagged
        line, = ax.plot(frequencies, row, color='#c0392b' if red else '#7fb3b8',
                        linewidth=1.1 if red else .6, alpha=1 if red else .7, zorder=3 if red else 2,
                        picker=4)
        line.set_gid(channel)
    median = psd_db(np.median(psd, axis=0)) if db else np.median(psd, axis=0)
    ax.plot(frequencies, median, color='#0a3d42', linewidth=2.2, label='Mediana de los canales', zorder=4)
    filtering = result['filter']
    upper = filtering['high'] + 5 if filtering['enabled'] else max(45, result['spectral']['edges'][-1])
    ax.set_xlim(0, min(upper, frequencies[-1]))
    # y range from the analysed band only: the filter's stopband (tens of dB lower) would
    # otherwise flatten every curve.
    passband = (frequencies >= result['spectral']['edges'][0]) & (
        frequencies <= (filtering['high'] if filtering['enabled'] else upper))
    if passband.any():
        shown = values[:, passband]
        low, high = np.nanpercentile(shown, 0.5), np.nanmax(shown)
        pad = .08 * (high - low) if high > low else 1
        if db:
            ax.set_ylim(low - pad, high + pad)
        else:
            ax.set_yscale('log')
            ax.set_ylim(max(low, 1e-6) / 1.5, high * 1.5)
        # Names of the red curves, at each curve's strongest point inside the passband.
        for channel, row in zip(result['channels'], values):
            if channel in flagged:
                k = np.flatnonzero(passband)[int(np.nanargmax(row[passband]))]
                ax.annotate(channel, (frequencies[k], row[k]), fontsize=8, color='#c0392b',
                            xytext=(0, 4), textcoords='offset points', ha='center')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('PSD (dB)\nre 1 µV²/Hz' if db else 'PSD (µV²/Hz)')
    ax.set_title(f"{result['participant']} · {result['eyes']} · {result['condition']} · "
                 f"{len(result['channels'])} canales" + (' · Exploratorio' if result['exploratory'] else ''),
                 fontsize=10)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper right')
    # Band-power maps: log10 absolute power per channel (standard 10-05 positions).
    placed = montage_channels(result['channels'])
    if len(placed) >= 4:
        info = standard_info(placed)
        for column, (index, name) in enumerate(zip(range(len(BANDS)), BANDS)):
            ax_map = figure.add_subplot(grid[1, column])
            data = np.array([np.log10(max(result['bands'][c][index]['absolute_uv2'], 1e-12)) for c in placed])
            mne.viz.plot_topomap(data, info, axes=ax_map, show=False, cmap='viridis', contours=0,
                                 sensors=True)
            low, high = result['spectral']['edges'][index], result['spectral']['edges'][index + 1]
            ax_map.set_title(f'{name.capitalize()} {low:g}–{high:g} Hz\nlog10 µV²', fontsize=8)
    return ax


def export_channel_spectra(result, parent):
    """A new directory with the per-channel PSD and band powers (never overwrites)."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{result['participant']}_{result['task']}_canales")
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = Path(parent) / f'{safe}_{stamp}'
    directory.mkdir(parents=False, exist_ok=False)
    with open(directory / 'psd_canales.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['canal'] + [f'{f:g}' for f in result['frequencies']])
        for channel, row in zip(result['channels'], result['psd_uv2_hz']):
            writer.writerow([channel] + [f'{v:.6g}' for v in row])
    with open(directory / 'bandas_canales.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['canal', 'banda', 'potencia_uV2', 'relativa_pct', 'atipico_30_45Hz'])
        for channel in result['channels']:
            for name, band in zip(BANDS, result['bands'][channel]):
                relative = band['relative_percent']
                writer.writerow([channel, name, f"{band['absolute_uv2']:.6g}",
                                 '' if relative is None else f'{relative:.4g}',
                                 channel in result['outliers']])
    return directory


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
