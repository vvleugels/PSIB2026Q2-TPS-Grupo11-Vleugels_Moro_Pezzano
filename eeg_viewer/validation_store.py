"""Versioned experimental revisions, immutable artifacts and pilot membership."""
from dataclasses import asdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import uuid

import mne
import numpy as np
import scipy

from .data import discover, open_recording
from .processing import FilterConfig
from .validation import ProcessingConfig, RuleProfile, epochs4, valid_position


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(encode(value)).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False).encode('utf-8')


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def export_new(path, value):
    """Exports never overwrite originals, revisions, or previous reports."""
    if Path(path).exists():
        raise ValueError('El destino ya existe; elegir un nombre nuevo para exportar.')
    atomic_write(path, encode(value))


def versions():
    package = Path(__file__).parent
    implementation = {name: sha(package / name) for name in
                      ('validation.py', 'validation_session.py', 'validation_store.py', 'processing.py')}
    return dict(python=platform.python_version(), mne=mne.__version__,
                numpy=np.__version__, scipy=scipy.__version__, implementation=implementation)


def preflight(study):
    recording = open_recording(study)
    try:
        raw = recording.raw
        if recording.sfreq <= 90 or len(recording.channels) < 4:
            raise ValueError('Se requieren al menos cuatro canales EEG y muestreo >90 Hz.')
        if not recording.metadata.get('EEGReference'):
            raise ValueError('La referencia de adquisición no está documentada.')
        for start in range(0, raw.n_times, 100000):
            if not np.isfinite(raw.get_data(start=start, stop=min(start + 100000, raw.n_times))).all():
                raise ValueError('El registro contiene muestras no finitas.')
        paths = {study.path.absolute(), *(Path(p).absolute() for p in raw.filenames if p)}
        sidecar = study.path.with_suffix('.json')
        if sidecar.exists():
            paths.add(sidecar.absolute())
        # External positions/channel metadata are also part of the provenance.
        paths.update(p.absolute() for p in study.path.parent.glob('*electrodes.tsv'))
        paths.update(p.absolute() for p in study.path.parent.glob('*coordsystem.json'))
        channel_file = study.path.with_name(study.path.name.replace('_eeg.set', '_channels.tsv'))
        if channel_file.exists():
            paths.add(channel_file.absolute())
        return dict(path=str(study.path.absolute()), participant=study.participant,
                    session=study.session, task=study.task, sfreq=recording.sfreq,
                    n_times=int(raw.n_times), channels=recording.channels,
                    reference=recording.metadata['EEGReference'],
                    missing_positions=[c for c in recording.channels if not valid_position(raw, c)],
                    positions={c: raw.info['chs'][raw.ch_names.index(c)]['loc'][:3].tolist()
                               if valid_position(raw, c) else None for c in recording.channels},
                    files=[dict(path=str(p), resolved_path=str(p.resolve()), size=p.stat().st_size,
                                mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)) for p in sorted(paths)])
    finally:
        recording.close()


def create_sample(data_dir, output):
    output = Path(output)
    if output.exists():
        raise ValueError('La muestra ya existe; no se sobrescribe.')
    inventory = discover(data_dir)
    groups = {True: [], False: []}
    for participant, studies in inventory.items():
        keys = {(s.session, s.task) for s in studies}
        if all((ses, 'eyesopen') in keys for ses in ('ses-1', 'ses-2')):
            closed = all((ses, 'eyesclosed') in keys for ses in ('ses-1', 'ses-2'))
            groups[closed].append(participant)
    entries, replacements, used = [], [], set()
    slots = [('calibration', True, 'sub-18'), ('calibration', False, 'sub-54'),
             ('evaluation', True, 'sub-02'), ('evaluation', False, 'sub-70')]
    for split, closed, preferred in slots:
        order = sorted(groups[closed], key=lambda p: hashlib.sha256(
            ('eeg-validation-v1:' + p).encode()).hexdigest())
        candidates = [preferred] + [p for p in order if p != preferred]
        failures = []
        selected = None
        for participant in candidates:
            if participant in used or participant not in groups[closed]:
                continue
            studies = [s for s in inventory[participant]
                       if s.session in ('ses-1', 'ses-2') and
                       s.task in (('eyesopen', 'eyesclosed') if closed else ('eyesopen',))]
            try:
                records = [preflight(s) for s in studies]
                if len(records) != (4 if closed else 2):
                    raise ValueError('Registros ambiguos o incompletos para el estrato.')
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(dict(participant=participant, error=str(exc)))
                continue
            selected = participant
            used.add(participant)
            entries.extend(dict(split=split, **r) for r in records)
            if selected != preferred or failures:
                replacements.append(dict(requested=preferred, selected=selected, failures=failures))
            break
        if selected is None:
            raise ValueError(f'No hay participante accesible para {split}/{closed}: {failures}')
    payload = dict(version=1, created_at=utc(), seed='eeg-validation-v1',
                   data_dir=str(Path(data_dir).absolute()), records=entries, substitutions=replacements)
    atomic_write(output, encode(payload))
    return payload


def verify_source(source, full=False):
    for saved in source['files']:
        p = Path(saved['path'])
        if saved.get('resolved_path', str(p.resolve())) != str(p.resolve()):
            raise ValueError('El enlace al origen cambió; revisión protegida.')
        stat = p.stat()
        if stat.st_size != saved['size'] or stat.st_mtime_ns != saved['mtime_ns']:
            raise ValueError('El origen cambió; crear una nueva revisión.')
        if full and sha(p) != saved['sha256']:
            raise ValueError('El hash del origen cambió; revisión protegida.')


def read_frozen(path, sample):
    value = json.loads(Path(path).read_text())
    if value.get('version') != 1 or value.get('status') != 'frozen_for_evaluation':
        raise ValueError('Se requiere un perfil congelado por el flujo de calibración.')
    if value['sample_sha256'] != sha(sample) or value['profile_sha256'] != digest_json(value['profile']):
        raise ValueError('Perfil congelado incompatible o modificado.')
    RuleProfile(**value['profile']).validate()
    if value['processing'] != json.loads(encode(asdict(ProcessingConfig()))):
        raise ValueError('La configuración de procesamiento difiere del perfil congelado.')
    return value


class Revision:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'revision.json'
        self.data = json.loads(self.path.read_text())
        if self.data['version'] != 1:
            raise ValueError('Versión de revisión desconocida.')
        self.digest = sha(self.path)
        self.check()
        verify_source(self.data['source'], full=True)
        for artifact in self.data['artifacts'].values():
            if sha(self.directory / artifact['path']) != artifact['sha256']:
                raise ValueError('Artefacto de procesamiento modificado.')

    @classmethod
    def create(cls, root, source, sample, profile=RuleProfile(), frozen=None):
        profile.validate()
        verify_source(source, full=True)
        sample = Path(sample).absolute()
        if source not in json.loads(sample.read_text())['records']:
            raise ValueError('El registro no pertenece a esta muestra.')
        frozen_value = None
        if source['split'] == 'evaluation':
            if frozen is None:
                raise ValueError('Congelar el perfil de calibración antes de abrir la evaluación.')
            frozen_value = read_frozen(frozen, sample)
            profile = RuleProfile(**frozen_value['profile'])
        identifier = uuid.uuid4().hex
        stem = Path(source['path']).stem
        directory = Path(root) / 'validation' / stem / identifier
        directory.mkdir(parents=True)
        legacy = Path(root) / (stem + '.json')
        backup = None
        if legacy.exists():
            content = legacy.read_bytes()
            backup_path = directory / 'legacy-5s.backup.json'
            atomic_write(backup_path, content)
            if backup_path.read_bytes() != content or legacy.read_bytes() != content:
                raise ValueError('No se pudo verificar el respaldo anterior.')
            backup = dict(path=backup_path.name, sha256=sha(backup_path))
        data = dict(version=1, id=identifier, created_at=utc(), source=source,
                    sample=str(sample), sample_sha256=sha(sample), profile=asdict(profile),
                    processing=asdict(ProcessingConfig()), frozen=frozen_value, legacy_backup=backup,
                    filters=dict(analysis=FilterConfig(True, .2, 45.).context(),
                                 training=FilterConfig(True, 1., 45.).context()),
                    versions=versions(), epoch_seconds=4, labels={}, final_labels={}, history=[],
                    blind_complete=False, persistent=[], excluded_epochs=[], excluded_components=[],
                    confirmed_repairs=[], reference_channels=[], artifacts={}, processing_version=0,
                    stage='blind', automatic_acceptance_enabled=False)
        atomic_write(directory / 'revision.json', encode(data))
        return cls(directory)

    @property
    def profile(self):
        return RuleProfile(**self.data['profile'])

    def check(self):
        if sha(self.path) != self.digest:
            raise ValueError('La revisión cambió externamente; volver a abrirla.')
        verify_source(self.data['source'])
        if sha(self.data['sample']) != self.data['sample_sha256']:
            raise ValueError('La muestra cambió; revisión protegida.')

    def save(self, action):
        self.check()
        self.data['history'].append(dict(at=utc(), action=action,
                                        processing_version=self.data['processing_version']))
        atomic_write(self.path, encode(self.data))
        self.digest = sha(self.path)

    @contextmanager
    def transaction(self):
        """Publish the manifest last; failed artifact writes leave the old revision usable."""
        previous, digest = deepcopy(self.data), self.digest
        try:
            yield
        except Exception:
            self.data, self.digest = previous, digest
            raise

    def label(self, epoch, channels, state, note='', final=False):
        source = self.data['source']
        epochs = epochs4(source['n_times'], source['sfreq'])
        if not 0 <= epoch < len(epochs) or not set(channels) <= set(source['channels']):
            raise ValueError('Epoch/canal inválido.')
        if state not in ('physiological', 'artifact', 'unreviewed') or len(note) > 500:
            raise ValueError('Evaluación inválida.')
        if not final and self.data['blind_complete']:
            raise ValueError('La evaluación ciega está cerrada; crear otra revisión para cambiarla.')
        if final and self.data['stage'] != 'clean':
            raise ValueError('Primero confirmar el procesamiento final.')
        target = self.data['final_labels' if final else 'labels']
        for c in channels:
            target[f'{epoch}:{c}'] = dict(state=state, note=note, at=utc(), origin='manual',
                                        processing_version=self.data['processing_version'])
        self.save(dict(label=dict(epoch=epoch, channels=list(channels), state=state, final=final)))

    def complete_blind(self):
        if self.data['blind_complete']:
            raise ValueError('La revisión ciega ya está cerrada.')
        source = self.data['source']
        for e in epochs4(source['n_times'], source['sfreq']):
            if not e.partial:
                for c in source['channels']:
                    if self.data['labels'].get(f'{e.index}:{c}', {}).get('state') not in ('artifact', 'physiological'):
                        raise ValueError('Faltan etiquetas de epochs completas; la revisión ciega sigue abierta.')
        self.data['blind_complete'] = True
        self.data['stage'] = 'quality'
        self.save('Cerrar revisión ciega y revelar sugerencias')

    def update_processing(self, *, persist=True, **changes):
        self.check()
        if not self.data['blind_complete']:
            raise ValueError('Completar la revisión ciega antes de limpiar.')
        if 'profile' in changes:
            if self.data['frozen'] is not None:
                raise ValueError('No se puede recalibrar el perfil de evaluación.')
            RuleProfile(**changes['profile']).validate()
        allowed = {'persistent', 'excluded_epochs', 'excluded_components', 'confirmed_repairs', 'profile'}
        if not set(changes) <= allowed:
            raise ValueError('Cambio de procesamiento desconocido.')
        self.data['history'].append(dict(at=utc(), previous=dict(
            processing_version=self.data['processing_version'], final_labels=self.data['final_labels'],
            artifacts=self.data['artifacts'], reference_channels=self.data['reference_channels'],
            settings={k: self.data[k] for k in allowed})))
        self.data.update(changes)
        self.data['processing_version'] += 1
        self.data['final_labels'] = {}
        self.data['reference_channels'] = []
        self.data['stage'] = 'quality'
        self.data['artifacts'] = {}
        if persist:
            self.save(dict(processing_changes=changes))

    def artifact(self, kind, writer, suffix):
        self.check()
        path = self.directory / (uuid.uuid4().hex + suffix)
        writer(path)
        entry = dict(path=path.name, sha256=sha(path))
        self.data['artifacts'][kind] = entry
        return entry

    def read_artifact(self, kind):
        item = self.data['artifacts'][kind]
        path = self.directory / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('El artefacto cambió externamente.')
        return path

    def export(self, destination):
        self.check()
        export_new(destination, self.data)


def freeze_profile(sample, revisions, output):
    if Path(output).exists():
        raise ValueError('No se sobrescribe un perfil congelado.')
    sample_hash = sha(sample)
    expected = {r['path'] for r in json.loads(Path(sample).read_text())['records']
                if r['split'] == 'calibration'}
    loaded = [Revision(p) for p in revisions]
    if len(loaded) != len(expected) or {r.data['source']['path'] for r in loaded} != expected:
        raise ValueError('Se requieren exactamente las seis revisiones de calibración.')
    for r in loaded:
        r.check()
        if r.data['sample_sha256'] != sample_hash or not r.data['blind_complete'] or r.data['stage'] != 'clean':
            raise ValueError('Completar revisión ciega y limpieza de todas las calibraciones.')
        for e in epochs4(r.data['source']['n_times'], r.data['source']['sfreq']):
            if e.partial or e.index in r.data['excluded_epochs']:
                continue
            if any(r.data['final_labels'].get(f'{e.index}:{c}', {}).get('state') not in
                   ('physiological', 'artifact') for c in r.data['source']['channels']):
                raise ValueError('Falta la revisión final de calibración.')
    profiles = {digest_json(r.data['profile']) for r in loaded}
    if len(profiles) != 1:
        raise ValueError('Todas las calibraciones deben usar el mismo perfil definitivo.')
    from .validation_session import ValidationSession
    evidence = []
    for r in loaded:
        session = ValidationSession(r)
        try:
            report = session.report()
            report['interpolation_holdouts'] = session.holdouts()
            evidence.append(dict(path=str(r.directory.absolute()), sha256=r.digest,
                                 snapshot=r.data, report=report))
        finally:
            session.close()
    payload = dict(version=1, created_at=utc(), sample_sha256=sample_hash,
                   profile=loaded[0].data['profile'], profile_sha256=profiles.pop(),
                   processing=asdict(ProcessingConfig()), versions=versions(),
                   calibration=evidence,
                   status='frozen_for_evaluation', approved_for_integration=False)
    atomic_write(output, encode(payload))
    return payload
