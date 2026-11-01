"""Five-second epochs and atomic annotation storage outside the dataset."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

STATES = ('Sin revisar', 'Aceptada', 'Con artefactos')
REASONS = ('', 'Ocular', 'Muscular', 'Movimiento/electrodo', 'Otro')


@dataclass(frozen=True)
class Epoch:
    index: int
    first: int
    last: int
    sfreq: float
    partial: bool

    @property
    def start(self):
        return self.first / self.sfreq

    @property
    def stop(self):
        return self.last / self.sfreq


def make_epochs(n_times, sfreq):
    n_times, sfreq = int(n_times), float(sfreq)
    # Boundaries are rounded from absolute times, avoiding accumulated drift.
    epochs = []
    first = 0
    while first < n_times:
        boundary = round((len(epochs) + 1) * 5 * sfreq)
        last = min(n_times, max(first + 1, boundary))
        epochs.append(Epoch(len(epochs), first, last, sfreq, last < boundary))
        first = last
    return epochs


def source_identity(recording, data_dir):
    paths = {recording.study.path}
    paths.update(Path(p) for p in recording.raw.filenames if p)
    sidecar = recording.study.path.with_suffix('.json')
    if sidecar.exists():
        paths.add(sidecar)
    files = []
    for path in sorted(paths):
        stat = path.stat()
        files.append(dict(path=path.absolute().relative_to(Path(data_dir).absolute()).as_posix(),
                          size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(files=files, sfreq=float(recording.sfreq), n_times=int(recording.raw.n_times))


class AnnotationStore:
    def __init__(self, directory, recording, data_dir):
        self.recording = recording
        self.data_dir = Path(data_dir)
        self.path = Path(directory) / f'{recording.study.path.stem}.json'
        self.identity = source_identity(recording, data_dir)
        self.epochs = make_epochs(recording.raw.n_times, recording.sfreq)
        self.records = [dict(index=e.index, first=e.first, last=e.last,
                             partial=e.partial, state=STATES[0], reason='', note='',
                             reviewed_at=None, channels=[], filter=None) for e in self.epochs]
        self.dirty = False
        self.blocked = ''
        self.digest = None
        self.load()

    @staticmethod
    def checksum(content):
        return hashlib.sha256(content).hexdigest()

    def load(self):
        if not self.path.exists():
            return
        try:
            content = self.path.read_bytes()
            self.digest = self.checksum(content)
            saved = json.loads(content)
            if saved['version'] != 1 or saved['source'] != self.identity:
                raise ValueError('La versión o identidad del registro cambió.')
            records = saved['epochs']
            if len(records) != len(self.records):
                raise ValueError('La cantidad de epochs no coincide.')
            for expected, actual in zip(self.records, records):
                if any(actual[k] != expected[k] for k in ('index', 'first', 'last', 'partial')):
                    raise ValueError('Los límites de epochs no coinciden.')
                if actual['state'] not in STATES or actual['reason'] not in REASONS:
                    raise ValueError('Estado o motivo desconocido.')
                if not isinstance(actual['note'], str) or len(actual['note']) > 500:
                    raise ValueError('Nota inválida.')
                if not isinstance(actual['channels'], list) or any(
                        c not in self.recording.channels for c in actual['channels']):
                    raise ValueError('Electrodos de revisión inválidos.')
                if actual['reviewed_at'] is not None:
                    datetime.fromisoformat(actual['reviewed_at'])
                    context = actual['filter']
                    if not isinstance(context, dict) or type(context['enabled']) is not bool:
                        raise ValueError('Contexto de filtrado inválido.')
                    from .processing import FilterConfig
                    FilterConfig(context['enabled'], context['low'], context['high']).validate(
                        self.recording.sfreq)
                elif actual['state'] != STATES[0] or actual['filter'] is not None:
                    raise ValueError('Falta el contexto de revisión.')
            self.records = records
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self.blocked = f'Anotaciones protegidas: {exc} Archivo: {self.path.name}'

    def mark(self, index, state, reason, note, channels, config):
        if self.blocked:
            raise ValueError(self.blocked)
        if state not in STATES or reason not in REASONS or len(note) > 500:
            raise ValueError('Marca inválida (la nota admite hasta 500 caracteres).')
        self.records[index].update(
            state=state, reason=reason if state == STATES[2] else '', note=note,
            reviewed_at=datetime.now(timezone.utc).isoformat(), channels=list(channels),
            filter=config.context())
        self.dirty = True

    def save(self):
        if self.blocked:
            raise ValueError(self.blocked)
        if source_identity(self.recording, self.data_dir) != self.identity:
            self.blocked = 'El registro original cambió. No se sobrescribirán las anotaciones.'
            raise ValueError(self.blocked)
        content = self.path.read_bytes() if self.path.exists() else None
        if (self.checksum(content) if content is not None else None) != self.digest:
            self.blocked = 'El archivo de anotaciones cambió externamente. No se sobrescribirá.'
            raise ValueError(self.blocked)
        payload = json.dumps(dict(version=1, source=self.identity, epoch_seconds=5,
                                  epochs=self.records), ensure_ascii=False, indent=2).encode('utf-8')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.annotations-',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self.digest = self.checksum(payload)
            self.dirty = False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
