#!/usr/bin/env bash
# Lanzador Linux: prepara el entorno local y abre el explorador EEG.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo 'Instalá Python 3.10 o superior para ejecutar el programa.' >&2
        exit 1
    fi
    if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        echo 'Se necesita Python 3.10 o superior.' >&2
        exit 1
    fi
    echo 'Primera ejecución: creando el entorno de Python...'
    if ! python3 -m venv .venv; then
        echo 'No se pudo crear el entorno. En Ubuntu/Debian instalá python3-venv.' >&2
        exit 1
    fi
fi

if ! .venv/bin/python -c 'import tkinter' >/dev/null 2>&1; then
    echo 'Falta Tkinter. En Ubuntu/Debian: sudo apt install python3-tk' >&2
    exit 1
fi

# The marker name changed when the analysis packages (pandas, Jupyter) were added, so
# existing environments install them once. requirements-analysis.txt includes requirements.txt.
if [[ ! -f .venv/TPS-analysis-dependencies-installed ]]; then
    echo 'Instalando dependencias. Esto puede tardar unos minutos...'
    if ! .venv/bin/python -m pip install -r requirements-analysis.txt; then
        echo 'No se pudieron instalar las dependencias. Revisá tu conexión e intentá de nuevo.' >&2
        exit 1
    fi
    touch .venv/TPS-analysis-dependencies-installed
fi

exec .venv/bin/python gui.py
