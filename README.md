# PSIB — Explorador EEG

GUI de escritorio para explorar registros de sueño normal y privación del sueño,
filtrar opcionalmente y revisar artefactos en epochs de 5 segundos. Python,
Tkinter, Matplotlib y MNE; los archivos EEG originales nunca se modifican.

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
Se puede ejecutar el lanzador por su ruta absoluta desde otro directorio: `data`
y `annotations` siempre se resuelven junto al proyecto.

## Explorar y filtrar

1. Elegir **Paciente**, **Estudio** y **Electrodo**. `ses-1` es sueño normal y
   `ses-2` es privación del sueño. La información de adquisición aparece al cargar.
2. Se muestran ventanas de **10 segundos**, amplitud en **µV** y líneas verticales
   cada **5 segundos**. **Anterior**, **Siguiente** e **Inicio (s) → Ir** recorren
   el registro. El inicio acepta punto o coma decimal; la última ventana puede ser parcial.
3. La barra del gráfico permite zoom, desplazamiento y restauración. Cambiar de
   electrodo conserva el tiempo y la epoch seleccionada.
4. **Configuración** permite activar el filtrado y ajustar sus cortes. Al iniciar,
   está apagado y los cortes son **0,2–45 Hz**. **Aplicar** valida y prepara el filtro
   en segundo plano; **Cancelar** conserva la configuración anterior. Restablecer
   recupera esos cortes sin modificar el toggle.
5. El encabezado identifica **Original** o **Filtrada · X–Y Hz**. La configuración
   se conserva durante la sesión de la aplicación. Un filtro inválido o demasiado
   largo para el registro no reemplaza la vista anterior.
6. **Varios electrodos** está apagado por defecto. Al activarlo permite elegir entre
   1 y 8 canales, en paneles con tiempo compartido y desplazamiento vertical.
   Al desactivarlo se vuelve al electrodo individual.

### Implementación del filtro

Se utiliza FIR de fase cero en MNE, diseño `firwin`, ventana Hamming, longitud y
bandas de transición automáticas, padding `reflect_limited`. La GUI configura
los cortes; el resto del diseño queda fijo. Se filtra **todo el canal continuo**
antes de extraer ventanas, nunca cada epoch por separado. Los canales filtrados
se conservan en una caché en memoria para el estudio y los cortes activos.

El pasabanda se basa en el paper; su diseño concreto no está especificado allí,
por lo que esta implementación no pretende reproducir exactamente su filtro.
Los bordes del registro pueden tener transitorios de filtrado: el padding no
los elimina por completo. No se aplica notch, remuestreo, ICA, interpolación,
cambio de referencia ni normalización. Se dibujan todas las muestras del tramo.

## Revisar epochs

- Las epochs son intervalos consecutivos de **5 s**, sin solapamiento, desde el
  inicio: `[0,5)`, `[5,10)`, etc. Un registro de 300 s tiene 60 epochs. Un resto
  final se conserva como **parcial**. El paper utiliza 4 s; aquí se eligieron 5 s.
- Elegir una epoch en la lista lateral lleva a una ventana que la contiene completa.
  También se puede seleccionarla haciendo clic en su intervalo del gráfico, con
  las herramientas de zoom/desplazamiento desactivadas. El borde ámbar indica
  la selección; colores y texto indican su estado.
- Los estados son **Sin revisar**, **Aceptada** y **Con artefactos**. Todas comienzan
  sin revisar. Las marcas son **globales para la epoch del estudio**, no por canal.
- Elegir estado, motivo opcional (ocular, muscular, movimiento/electrodo u otro)
  y nota de hasta 500 caracteres; pulsar **Aplicar marca**. Se puede cambiar la
  evaluación tantas veces como sea necesario.
- Atajos fuera de campos de texto: **A** acepta y guarda, **U** vuelve a sin revisar
  y guarda; **R** selecciona artefactos y enfoca el motivo, para completar y pulsar
  **Aplicar marca**. Los cambios del formulario no se guardan hasta aplicarlos.
- Los contadores resumen el progreso. Las epochs con artefactos siguen visibles:
  esta versión **no elimina ni excluye muestras**.
- Cambiar de filtro o canal conserva las marcas. El editor muestra el filtro y los
  canales observados durante la última evaluación; activar un filtro nuevo no
  revalida automáticamente una decisión anterior.

## Anotaciones y protección de datos

Cada marca aplicada se guarda automáticamente en `annotations/<registro>_eeg.json`,
fuera de `data`. Estos archivos están excluidos de Git: respaldarlos por separado.
Al abrir un estudio se recuperan las marcas guardadas.

El JSON versionado contiene identidad de los archivos fuente (ruta relativa,
tamaño y modificación), frecuencia, número de muestras, límites de cada epoch,
estado, motivo, nota, fecha UTC, canales observados y configuración de filtrado,
incluyendo la versión de MNE. La escritura utiliza un temporal y reemplazo atómico.

Si falla el guardado, las marcas permanecen en memoria y aparece **Cambios sin
guardar**. **Reintentar guardado** vuelve a intentar. Antes de cambiar de estudio,
paciente o cerrar, la aplicación permite reintentar, descartar o cancelar la salida.
Si cambia el origen, el archivo de anotaciones fue modificado externamente o su
contenido es inválido, se protege contra sobrescritura y se informa el problema.
Conservar una copia del archivo y resolver la inconsistencia antes de continuar
la revisión; no se reemplaza silenciosamente un archivo anterior.

## Datos y organización

Se espera BIDS: `data/sub-*/ses-*/eeg/*_eeg.set`, sus `.fdt` y metadatos JSON.
Los enlaces git-annex deben tener contenido local. Los archivos incompletos se
rechazan antes de graficar. Frecuencia, duración y canales provienen del registro;
la referencia proviene del JSON. Los campos ausentes se indican como no disponibles.

- `eeg_viewer/data.py`: inventario y acceso de solo lectura por canal/tramo.
- `eeg_viewer/processing.py`: configuración y caché del filtro continuo.
- `eeg_viewer/annotations.py`: epochs, identidad del origen y persistencia atómica.
- `eeg_viewer/app.py` y `eeg_viewer/review_ui.py`: interfaz y coordinación asíncrona.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan archivos temporales y señales conocidas para comprobar unidades,
filtrado, conservación del origen, límites de epochs, persistencia y fallos de
escritura; también cubren navegación, multicanal y respuestas asíncronas obsoletas.
Las pruebas de GUI requieren pantalla y se omiten si no hay una disponible.
Con Xvfb: `xvfb-run -a python -m unittest discover -s tests -v`.
