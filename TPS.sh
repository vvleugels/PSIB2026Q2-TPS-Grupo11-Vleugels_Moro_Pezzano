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

# Exact tested versions (requirements-lock.txt, Python 3.12+), so every computer gets the same
# numbers; on an older Python, the compatible ranges of requirements-analysis.txt instead.
# The marker name changes with the dependencies, so existing environments install them once.
if [[ ! -f .venv/TPS-dependencies-v4-installed ]]; then
    echo 'Instalando dependencias. Esto puede tardar unos minutos...'
    if ! .venv/bin/python -m pip install -r requirements-lock.txt; then
        echo 'No se pudieron instalar las versiones exactas; se instalan versiones compatibles...'
        if ! .venv/bin/python -m pip install -r requirements-analysis.txt; then
            echo 'No se pudieron instalar las dependencias. Revisá tu conexión e intentá de nuevo.' >&2
            exit 1
        fi
    fi
    touch .venv/TPS-dependencies-v4-installed
fi

exec .venv/bin/python gui.py
