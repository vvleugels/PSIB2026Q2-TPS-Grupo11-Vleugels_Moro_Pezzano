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
                    from tps.core.processing import FilterConfig
                    FilterConfig(context['enabled'], context['low'], context['high'], context.get('notch')).validate(
                        self.recording.sfreq)
                elif actual['state'] != STATES[0] or actual['filter'] is not None:
                    raise ValueError('Falta el contexto de revisión.')
                correction = actual.get('ica')
                if correction is not None:
                    if (not isinstance(correction, dict)
                            or not isinstance(correction.get('revision'), str)
                            or len(correction['revision']) != 32
                            or any(c not in '0123456789abcdef' for c in correction['revision'])
                            or not isinstance(correction.get('model_sha256'), str)
                            or len(correction['model_sha256']) != 64
                            or not isinstance(correction.get('excluded_components'), list)
                            or any(type(i) is not int or i < 0 for i in correction['excluded_components'])
                            or not isinstance(correction.get('excluded_channels'), list)
                            or any(c not in self.recording.channels for c in correction['excluded_channels'])):
                        raise ValueError('Contexto ICA inválido.')
                if actual.get('screening') is not None:
                    from tps.core.screening import ScreeningConfig
                    screening = actual['screening']
                    if (not isinstance(screening, dict) or screening['method'] != 'peak_to_peak'
                            or actual['reviewed_at'] is None or actual['state'] == STATES[2]):
                        raise ValueError('Contexto del barrido automático inválido.')
                    ScreeningConfig(screening['high_uv'], screening['flat_uv']).validate()
            self.records = records
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self.blocked = f'Anotaciones protegidas: {exc} Archivo: {self.path.name}'

    def mark(self, index, state, reason, note, channels, config, *, screening=None):
        if self.blocked:
            raise ValueError(self.blocked)
        if state not in STATES or reason not in REASONS or len(note) > 500:
            raise ValueError('Marca inválida (la nota admite hasta 500 caracteres).')
        self.records[index].update(
            state=state, reason=reason if state == STATES[2] else '', note=note,
            reviewed_at=datetime.now(timezone.utc).isoformat(), channels=list(channels),
            filter=config.context())
        from tps.core.ica import context
        self.records[index]['ica'] = context(self.recording)
        if screening is None:
            self.records[index].pop('screening', None)
        else:
            self.records[index]['screening'] = dict(screening)
        self.dirty = True

    def apply_screening(self, result):
        """Accept eligible epochs and recheck prior automatic decisions as a batch.

        Explicit manual acceptance/rejection takes precedence. A prior automatic
        acceptance returns to unreviewed if a new scan flags it. Saving is left
        to the caller so the whole batch is written once.
        """
        if self.blocked:
            raise ValueError(self.blocked)
        if result['total'] != len(self.epochs):
            raise ValueError('El barrido no coincide con las epochs del estudio.')
        config = result['config']
        context = dict(method='peak_to_peak', high_uv=config.high_uv, flat_uv=config.flat_uv)
        filtering = result['filter']
        filter_context = filtering.context()
        from tps.core.ica import context as ica_context
        channels = list(result['channels'])
        counts = dict(accepted=0, pending=0)
        for index, record in enumerate(self.records):
            automatic = record.get('screening') is not None
            if not automatic and record['state'] != STATES[0]:
                continue
            state = STATES[0] if index in result['findings'] else STATES[1]
            if state == STATES[0] and not automatic:
                continue
            if (record['state'] == state and record.get('screening') == context
                    and record['channels'] == channels and record['filter'] == filter_context
                    and record.get('ica') == ica_context(self.recording)):
                continue
            self.mark(index, state, '', record['note'], channels, filtering, screening=context)
            counts['accepted' if state == STATES[1] else 'pending'] += 1
        return counts

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
