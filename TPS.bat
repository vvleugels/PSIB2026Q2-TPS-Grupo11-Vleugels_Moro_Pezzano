@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Primera ejecucion: creando el entorno de Python...
    py -3 -m venv ".venv" >nul 2>&1
    if errorlevel 1 python -m venv ".venv"
    if errorlevel 1 (
        echo.
        echo No se pudo crear el entorno. Instala Python 3.12 o superior
        echo y asegurate de incluir el lanzador py o agregar Python al PATH.
        pause
        exit /b 1
    )
)

rem Exact tested versions (requirements-lock.txt, Python 3.12+), so every computer gets the same
rem numbers; on an older Python, the compatible ranges of requirements-analysis.txt instead.
rem The marker name changes with the dependencies, so existing environments install them once.
if not exist ".venv\TPS-dependencies-v4-installed" (
    echo Instalando dependencias. Esto puede tardar unos minutos...
    ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
    if errorlevel 1 (
        echo No se pudieron instalar las versiones exactas; se instalan versiones compatibles...
        ".venv\Scripts\python.exe" -m pip install -r requirements-analysis.txt
        if errorlevel 1 (
            echo.
            echo No se pudieron instalar las dependencias. Revisa tu conexion e intenta de nuevo.
            pause
            exit /b 1
        )
    )
    type nul > ".venv\TPS-dependencies-v4-installed"
)

".venv\Scripts\python.exe" gui.py
if errorlevel 1 pause
