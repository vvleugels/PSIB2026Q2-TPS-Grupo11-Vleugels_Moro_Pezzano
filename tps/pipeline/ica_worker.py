"""Isolated, cancellable ICA fitting process. Writes a draft, never confirms it."""
import json
from pathlib import Path
import sys

from tps.core.data import Study, open_recording
from tps.core.ica import fit_recording, write_json


def main():
    folder = Path(sys.argv[1])
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    study = Study(Path(request['path']), request['participant'], request['session'], request['task'])
    recording = None
    try:
        recording = open_recording(study)
        model, metadata = fit_recording(recording, request['data_dir'], request['channels'], request['epochs'])
        model.save(folder / 'draft-ica.fif', overwrite=False, verbose=False)
        write_json(folder / 'fit.json', metadata)
    except Exception as exc:
        write_json(folder / 'error.json', dict(error=str(exc)))
        return 1
    finally:
        if recording is not None:
            recording.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
