"""Inventario BIDS y acceso de solo lectura a registros EEGLAB."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import mne
import numpy as np
from .processing import ContinuousFilter, FilterConfig


def natural_key(value):
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r'(\d+)', str(value)))


@dataclass(frozen=True)
class Study:
    path: Path
    participant: str
    session: str
    task: str

    @property
    def condition(self):
        return {'ses-1': 'Sueño normal', 'ses-2': 'Privación del sueño'}.get(
            self.session, self.session)

    @property
    def eyes(self):
        return {'eyesopen': 'Ojos abiertos', 'eyesclosed': 'Ojos cerrados'}.get(
            self.task, self.task)

    @property
    def label(self):
        return f'{self.condition} · {self.eyes} ({self.session})'


def discover(data_dir):
    """Incluye enlaces rotos para poder informar archivos no descargados."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'No se encontró la carpeta de datos: {data_dir}')
    inventory = {}
    for subject in sorted(data_dir.glob('sub-*'), key=lambda p: natural_key(p.name)):
        if not subject.is_dir():
            continue
        studies = []
        for path in sorted(subject.glob('ses-*/eeg/*_eeg.set'),
                           key=lambda p: natural_key(str(p))):
            match = re.search(r'_task-([^_]+)', path.name)
            if match:
                studies.append(Study(path, subject.name, path.parent.parent.name, match[1]))
        inventory[subject.name] = studies
    return inventory


@dataclass
class Segment:
    times: np.ndarray
    values_uv: np.ndarray
    start: float
    stop: float
    channels: tuple = ()


class Recording:
    def __init__(self, study, raw, metadata, warning=''):
        self.study = study
        self.raw = raw
        self.metadata = metadata
        self.warning = warning
        self.filter = ContinuousFilter(raw)
        self.sfreq = float(raw.info['sfreq'])
        self.duration = raw.n_times / self.sfreq
        self.channels = [name for name, kind in zip(raw.ch_names, raw.get_channel_types())
                         if kind == 'eeg']
        if not self.channels or raw.n_times < 1:
            raise ValueError('El registro no contiene muestras o canales EEG.')

    def read_window(self, channel, start, seconds=10):
        if channel not in self.channels:
            raise ValueError('El electrodo no pertenece al registro seleccionado.')
        if not math.isfinite(start) or not 0 <= start < self.duration:
            raise ValueError(f'El inicio debe estar entre 0 y {self.duration:g} s (sin incluir el final).')
        first = min(int(round(start * self.sfreq)), self.raw.n_times - 1)
        last = min(first + int(round(seconds * self.sfreq)), self.raw.n_times)
        values = self.raw.get_data(picks=[channel], start=first, stop=last)[0]
        return Segment(np.arange(first, last) / self.sfreq, values * 1e6,
                       first / self.sfreq, last / self.sfreq)

    def close(self):
        self.filter.channels.clear()
        self.raw.close()

    def read_channels(self, channels, start, config=FilterConfig(), seconds=10):
        if not channels or len(channels) > 8 or any(c not in self.channels for c in channels):
            raise ValueError('Seleccioná entre 1 y 8 electrodos del registro.')
        if not config.enabled:
            segments = [self.read_window(c, start, seconds) for c in channels]
            first = segments[0]
            return Segment(first.times, np.stack([s.values_uv for s in segments]),
                           first.start, first.stop, tuple(channels))
        if not math.isfinite(start) or not 0 <= start < self.duration:
            raise ValueError('Inicio fuera del registro.')
        self.filter.prepare(config)
        first = min(round(start * self.sfreq), self.raw.n_times - 1)
        last = min(first + round(seconds * self.sfreq), self.raw.n_times)
        values = np.stack([self.filter.channel(c, config)[first:last] for c in channels])
        return Segment(np.arange(first, last) / self.sfreq, values * 1e6,
                       first / self.sfreq, last / self.sfreq, tuple(channels))


def open_recording(study):
    if not study.path.is_file():
        raise FileNotFoundError(f'El archivo no está disponible: {study.path.name}')
    metadata = {}
    warning = ''
    sidecar = study.path.with_suffix('.json')
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8-sig'))
            if not isinstance(metadata, dict):
                raise ValueError('Se esperaba un objeto JSON')
        except (OSError, ValueError) as exc:
            warning = f'No se pudieron leer los metadatos: {exc}'
            metadata = {}
    else:
        warning = 'Metadatos JSON no disponibles; se muestra la información del registro.'
    raw = mne.io.read_raw_eeglab(study.path, preload=False, verbose='ERROR')
    try:
        # EEGLAB external data is float32, one value per channel and sample.
        # Fail early instead of displaying a file whose final windows cannot be read.
        for filename in raw.filenames:
            if filename is not None and Path(filename).suffix.lower() == '.fdt':
                expected = raw.n_times * len(raw.ch_names) * 4
                actual = Path(filename).stat().st_size
                if actual < expected:
                    raise ValueError(
                        f'Archivo EEG incompleto: {Path(filename).name}. '
                        f'Tiene {actual:,} bytes; la cabecera requiere {expected:,}. '
                        'Verificá la descarga del registro.'
                    )
        return Recording(study, raw, metadata, warning)
    except Exception:
        raw.close()
        raise
