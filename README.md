# PSIB — Explorador EEG

GUI de escritorio para explorar los registros originales de sueño normal y privación
del sueño. Python + Tkinter + Matplotlib, con lectura de EEGLAB mediante MNE.

## Instalación y ejecución

Requiere Python 3.10 o superior y una sesión gráfica. Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gui.py
```

En Ubuntu/Debian, si faltan Tkinter o venv: `sudo apt install python3-tk python3-venv`.
En Windows, activar con `.venv\Scripts\activate` y usar `python` al crear el entorno.
También se puede ejecutar `gui.py` por su ruta absoluta desde otro directorio:
la aplicación siempre busca `data` junto al lanzador.

## Uso

1. Elegir **Paciente**, **Estudio** y **Electrodo**, en ese orden.
2. Al elegir estudio aparecen sus datos de adquisición. Solo se ofrecen los registros
   existentes; `ses-1` es sueño normal y `ses-2` es privación del sueño.
3. El gráfico muestra los primeros 10 segundos del electrodo, en µV, con líneas
   verticales cada 5 segundos. **Anterior**, **Siguiente** e **Inicio (s) → Ir**
   permiten recorrer el registro. Se admiten coma o punto decimal.
4. La barra del gráfico permite zoom, desplazamiento y restauración de la ventana.
   Cambiar electrodo vuelve a los primeros 10 segundos. Cambiar paciente o estudio
   limpia las selecciones dependientes.

Se conservan todas las muestras del tramo mostrado. No se aplican filtros, remuestreo,
normalización, cambio de referencia ni eliminación de artefactos. La conversión de
voltios a µV es únicamente de unidades. La última ventana puede durar menos de 10 s.
Las lecturas ocurren en segundo plano; las respuestas de selecciones anteriores
se descartan. La escala vertical se ajusta a cada ventana.

## Datos y estructura

Se espera la estructura BIDS `data/sub-*/ses-*/eeg/*_eeg.set`, sus archivos `.fdt`
y los metadatos JSON. Los enlaces git-annex deben tener disponible su contenido local.
Los archivos se abren en modo de lectura; no se modifica el dataset.
La frecuencia, duración y canales mostrados provienen del registro cargado; la
referencia proviene del JSON. Los metadatos ausentes se indican como no disponibles.

- `eeg_viewer/data.py`: inventario, metadatos y lectura por electrodo y tramo.
- `eeg_viewer/app.py`: interfaz, navegación y coordinación de cargas.
- `gui.py`: lanzador independiente del proyecto Competencia.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Las pruebas crean datos sintéticos temporales de valores conocidos; verifican
amplitudes, unidades, tiempos, ventanas finales, archivos ausentes y navegación.
Las pruebas de interfaz necesitan pantalla y se omiten si no está disponible.
En un servidor con Xvfb: `xvfb-run -a python -m unittest discover -s tests -v`.
