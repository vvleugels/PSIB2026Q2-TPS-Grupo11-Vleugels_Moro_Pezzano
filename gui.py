"""Ejecutar con python gui.py desde cualquier directorio."""
from pathlib import Path


if __name__ == '__main__':
    try:
        from eeg_viewer.app import main
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f'Falta la dependencia {exc.name}. Instalá requirements.txt. '
            'En Linux, Tkinter requiere python3-tk.'
        ) from exc
    main(Path(__file__).resolve().parent / 'data')
