@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Primera ejecucion: creando el entorno de Python...
    py -3 -m venv ".venv" >nul 2>&1
    if errorlevel 1 python -m venv ".venv"
    if errorlevel 1 (
        echo.
        echo No se pudo crear el entorno. Instala Python 3.10 o superior
        echo y asegurate de incluir el lanzador py o agregar Python al PATH.
        pause
        exit /b 1
    )
)

rem The marker name changed when the analysis packages (pandas) were added, so
rem existing environments install them once. requirements-analysis.txt includes requirements.txt.
if not exist ".venv\TPS-analysis-dependencies-installed" (
    echo Instalando dependencias. Esto puede tardar unos minutos...
    ".venv\Scripts\python.exe" -m pip install -r requirements-analysis.txt
    if errorlevel 1 (
        echo.
        echo No se pudieron instalar las dependencias. Revisa tu conexion e intenta de nuevo.
        pause
        exit /b 1
    )
    type nul > ".venv\TPS-analysis-dependencies-installed"
)

".venv\Scripts\python.exe" gui.py
if errorlevel 1 pause
