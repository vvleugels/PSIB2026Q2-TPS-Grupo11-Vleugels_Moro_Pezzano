"""Confirmed ICA revisions and shared signal preparation. Source files stay read-only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import mne
import numpy as np
import scipy

from tps.core.annotations import make_epochs, same_source, source_identity
from tps.core.processing import ContinuousFilter, FilterConfig
from tps.core.infomax import fit_infomax
from tps.core.jsonio import json_safe, write_json


def versions():
    return dict(mne=mne.__version__, numpy=np.__version__, scipy=scipy.__version__)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fit_recording(recording, data_dir, excluded_channels=(), excluded_epochs=()):
    channels = [c for c in recording.channels if c not in excluded_channels]
    if len(channels) < 2 or not set(excluded_channels) <= set(recording.channels):
        raise ValueError('ICA requiere al menos dos canales EEG válidos.')
    epochs = make_epochs(recording.raw.n_times, recording.sfreq)
    if any(type(i) is not int or i < 0 or i >= len(epochs) or epochs[i].partial for i in excluded_epochs):
        raise ValueError('Solo se pueden excluir épocas completas válidas del entrenamiento.')
    kept = [e for e in epochs if not e.partial and e.index not in excluded_epochs]
    if not kept:
        raise ValueError('No quedan épocas completas para ajustar ICA.')
    identity = source_identity(recording, data_dir)
    # This copy always comes from the original, never from a prior ICA result.
    training = recording.raw.copy().pick(channels).load_data()
    try:
        if not np.isfinite(training.get_data()).all():
            raise ValueError('Hay muestras no finitas en los canales de entrenamiento.')
        cache = ContinuousFilter(training)
        filtered = cache.prefetch(channels, FilterConfig(True, 1, 45))
        data = np.stack([np.concatenate([filtered[c][e.first:e.last] for e in kept]) for c in channels])
        fitted = fit_infomax(data, training.info, channels, [e.index for e in kept])
    finally:
        training.close()
    if source_identity(recording, data_dir) != identity:
        raise ValueError('El registro cambió durante el ajuste.')
    metadata = json_safe(dict(
        schema=1, source=identity, channels=channels, original_channels=recording.raw.ch_names,
        sfreq=recording.sfreq, reference=recording.metadata.get('EEGReference'),
        excluded_channels=list(excluded_channels), excluded_epochs=sorted(set(excluded_epochs)),
        training=dict(low=1., high=45., seed=42, max_iter=1000, method='extended_infomax', epoch_seconds=5),
        rank=fitted.rank, fit_epochs=fitted.fit_epochs, samples=fitted.samples,
        components=fitted.components, convergence=fitted.convergence, versions=versions()))
    return fitted.ica, metadata


def revision_root(annotations_dir, study):
    return Path(annotations_dir) / 'ica' / study.path.stem


def latest_revision(annotations_dir, study):
    path = revision_root(annotations_dir, study) / 'latest.json'
    if not path.exists():
        raise ValueError(f'{study.path.name}: falta una ICA confirmada.')
    return json.loads(path.read_text(encoding='utf-8'))['revision']


def read_revision(annotations_dir, study, revision=None):
    revision = revision or latest_revision(annotations_dir, study)
    if not isinstance(revision, str) or len(revision) != 32 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Identificador de revisión ICA inválido.')
    folder = revision_root(annotations_dir, study) / revision
    metadata = json.loads((folder / 'revision.json').read_text(encoding='utf-8'))
    if metadata['revision'] != revision or metadata['schema'] != 1 or digest(folder / 'model-ica.fif') != metadata['model_sha256']:
        raise ValueError('La revisión ICA está dañada o fue modificada.')
    return folder, metadata


def validate_revision(recording, data_dir, metadata):
    if (not same_source(metadata['source'], source_identity(recording, data_dir))
            or metadata['original_channels'] != recording.raw.ch_names
            or metadata['sfreq'] != recording.sfreq
            or metadata['reference'] != recording.metadata.get('EEGReference')):
        raise ValueError('La ICA guardada no corresponde al registro actual.')
    # Library versions are not compared: the saved model is a fixed unmixing matrix, so it is
    # applied the same with other MNE/NumPy/SciPy versions. version_note() reports a difference.


def version_note(metadata):
    """'' when the revision was fitted with this computer's library versions, else a short note."""
    saved, current = metadata.get('versions') or {}, versions()
    if saved == current:
        return ''
    names = {'mne': 'MNE', 'numpy': 'NumPy', 'scipy': 'SciPy'}
    def text(values):
        return ' · '.join(f'{names[k]} {values.get(k, "?")}' for k in names)
    return f'Ajustada con {text(saved)}; este equipo usa {text(current)}. Se aplica igual.'


def corrected_copy(recording, model, excluded):
    excluded = sorted(set(excluded))
    if any(type(i) is not int or not 0 <= i < model.n_components_ for i in excluded):
        raise ValueError('Componente ICA inválido.')
    result = recording.raw.copy().load_data()
    try:
        if excluded:
            # MNE unions explicit exclusions with model.exclude (which plots may change).
            # Only the user's confirmed list is authoritative in this workflow.
            selected_model = model.copy()
            selected_model.exclude = []
            selected_model.apply(result, exclude=excluded, verbose=False)
    except Exception:
        result.close()
        raise
    return result


def confirm(recording, data_dir, annotations_dir, model, metadata, excluded):
    validate_revision(recording, data_dir, metadata)
    corrected = corrected_copy(recording, model, excluded)
    revision = uuid.uuid4().hex
    root = revision_root(annotations_dir, recording.study)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.pending-', dir=root))
    try:
        saved_model = model.copy()
        saved_model.exclude = []
        saved_model.save(temporary / 'model-ica.fif', overwrite=False, verbose=False)
        value = {**metadata, 'revision': revision, 'excluded_components': sorted(set(excluded)),
                 'model_sha256': digest(temporary / 'model-ica.fif'),
                 'created_at': datetime.now(timezone.utc).isoformat()}
        write_json(temporary / 'revision.json', value)
        validate_revision(recording, data_dir, metadata)
        temporary.rename(root / revision)
        # Publishing the pointer is the transaction's final step.
        write_json(root / 'latest.json', dict(revision=revision, enabled=True))
    except Exception:
        corrected.close()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return corrected, value


def activate(recording, corrected, metadata):
    previous = recording.signal_raw
    recording.signal_raw = corrected if corrected is not None else recording.raw
    recording.ica_context = metadata
    recording.filter = ContinuousFilter(recording.signal_raw)
    if previous is not recording.raw and previous is not corrected:
        previous.close()


def apply_revision(recording, data_dir, annotations_dir, revision=None):
    folder, metadata = read_revision(annotations_dir, recording.study, revision)
    validate_revision(recording, data_dir, metadata)
    model = mne.preprocessing.read_ica(folder / 'model-ica.fif', verbose=False)
    if model.ch_names != metadata['channels']:
        raise ValueError('Canales del modelo ICA incompatibles.')
    activate(recording, corrected_copy(recording, model, metadata['excluded_components']), metadata)
    return metadata


def context(recording):
    value = recording.ica_context
    return None if value is None else {key: value[key] for key in (
        'revision', 'model_sha256', 'excluded_components', 'excluded_channels')}


def manual_bad_channels(recording, data_dir, annotations_dir):
    """{channel: reason} marked by hand in the explorer; {} without a valid annotation file."""
    from tps.core.annotations import AnnotationStore
    store = AnnotationStore(annotations_dir, recording, data_dir)
    if store.blocked:
        return {}
    return {channel: value['reason'] for channel, value in sorted(store.bad_channels.items())}


def snapshot(studies, data_dir, annotations_dir, mode):
    """Immutable input selection, including explicit failures; never silently fall back."""
    if mode not in ('original', 'confirmed'):
        raise ValueError('Modo ICA inválido.')
    from tps.core.data import open_recording
    result = {}
    for study in studies:
        try:
            recording = open_recording(study)
            try:
                value = dict(source=source_identity(recording, data_dir), mode=mode)
                value['bad_channels'] = manual_bad_channels(recording, data_dir, annotations_dir)
                if mode == 'confirmed':
                    _, metadata = read_revision(annotations_dir, study)
                    validate_revision(recording, data_dir, metadata)
                    value.update(revision=metadata['revision'], model_sha256=metadata['model_sha256'])
                result[str(study.path)] = value
            finally:
                recording.close()
        except Exception as exc:
            result[str(study.path)] = dict(error=str(exc), mode=mode)
    return result


def apply_snapshot(recording, data_dir, annotations_dir, selection):
    if selection.get('error'):
        raise ValueError(selection['error'])
    if not same_source(selection['source'], source_identity(recording, data_dir)):
        raise ValueError('El registro cambió después de iniciar el análisis.')
    if selection['mode'] == 'confirmed':
        metadata = apply_revision(recording, data_dir, annotations_dir, selection['revision'])
        if metadata['model_sha256'] != selection['model_sha256']:
            raise ValueError('El modelo ICA cambió durante el análisis.')
