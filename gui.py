"""Ejecutar con python gui.py desde cualquier directorio."""
from pathlib import Path
import sys
import tkinter as tk

ROOT = Path(__file__).resolve().parent


def loading_window() -> tk.Tk:
    """Show the window right away; MNE, SciPy and Matplotlib take ~3 s to import."""
    if sys.platform == 'win32':
        import ctypes   # identidad propia en la barra de tareas, separada de python.exe
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('psib.eeg.explorer')
    root = tk.Tk()
    root.title('PSIB — Explorador EEG')
    root.configure(background='#0f2e36')
    root.geometry('520x180')
    tk.Label(root, text='PSIB · Explorador EEG', font=('Segoe UI', 16, 'bold'),
             foreground='white', background='#0f2e36').pack(pady=(42, 6))
    tk.Label(root, text='Cargando el explorador EEG…', font=('Segoe UI', 10),
             foreground='#9db8be', background='#0f2e36').pack()
    tk.Frame(root, background='#f5a524', height=3).pack(side='bottom', fill='x')
    root.update()
    return root


if __name__ == '__main__':
    root = loading_window()
    try:
        from eeg_viewer.app import main
    except ModuleNotFoundError as exc:
        root.destroy()
        raise SystemExit(
            f'Falta la dependencia {exc.name}. Instalá requirements.txt. '
            'En Linux, Tkinter requiere python3-tk.'
        ) from exc
    main(ROOT / 'data', root)
