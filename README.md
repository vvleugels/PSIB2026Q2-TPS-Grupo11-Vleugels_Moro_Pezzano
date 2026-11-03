# PSIB — Explorador EEG

GUI de escritorio para explorar registros de sueño normal y privación del sueño,
filtrar opcionalmente, revisar artefactos en epochs de 5 segundos y analizar
espectros con Welch y potencia por bandas. Python,
Tkinter, Matplotlib y MNE; los archivos EEG originales nunca se modifican.

El [piloto de limpieza EEG y validación](VALIDATION.md) se ejecuta por separado con
`validate_eeg.py`. Implementa el motor experimental de 4 segundos y la revisión
humana previa a la integración; todavía no modifica el flujo de esta GUI ni habilita
aceptación automática definitiva.

## Instalación y ejecución

Requiere Python 3.10 o superior y una sesión gráfica. Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gui.py
```

En Ubuntu/Debian, si faltan Tkinter o venv: `sudo apt install python3-tk python3-venv`.
En Windows, hace doble clic en `TPS.bat`. En el primer inicio crea `.venv` e instala
las dependencias; hace falta una conexión a internet. Si falla la instalación,
corregí el problema y volvé a ejecutar el lanzador: reintentará la instalación.
También se puede ejecutar `TPS.bat` por su ruta absoluta desde otro directorio:
`data` y `annotations` siempre se resuelven junto al proyecto.

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
  y nota de hasta 500 caracteres. El estado y el motivo se guardan automáticamente
  al cambiarlos; las notas, tras 400 ms sin escribir. No hay que aplicar ni guardar
  cada marca. Se puede cambiar la evaluación tantas veces como sea necesario.
- Atajos fuera de campos de texto: **A** acepta, **U** vuelve a sin revisar;
  **R** marca artefactos y enfoca el motivo. Todos guardan automáticamente.
- Los contadores resumen el progreso. Las epochs con artefactos siguen visibles:
  esta versión **no elimina ni excluye muestras**.
- Cambiar de filtro o canal conserva las marcas. El editor muestra el filtro y los
  canales observados durante la última evaluación; activar un filtro nuevo no
  revalida automáticamente una decisión anterior.

## Barrido de posibles artefactos

Después de elegir un electrodo, pulsar **Buscar posibles artefactos** en el panel
lateral. Por defecto recorre todos los canales EEG del estudio; desactivar
**Todos los electrodos EEG** limita el barrido a los visibles (uno o hasta ocho).
Se calcula `máximo − mínimo` por canal en cada epoch de 5 s. Por defecto está
marcado **Usar el filtro activo**: utiliza la misma configuración de filtrado que
la vista temporal y el análisis espectral. El FIR se aplica al **canal continuo
completo antes de dividirlo en epochs**, aprovechando la caché existente. El panel,
el resumen y el detalle identifican **Filtrada · X–Y Hz** o **Original**.

Si el filtrado está desactivado en **Configuración**, el barrido utiliza el original
y lo indica explícitamente; esta opción no enciende el filtro automáticamente.
Desmarcar **Usar el filtro activo** permite revisar siempre la señal original,
aunque se esté mostrando una señal filtrada. Así se pueden distinguir avisos por
deriva que disminuyen con el pasabanda de problemas visibles en la adquisición.

Los límites iniciales, ajustables en µV pico a pico, son **Alta > 150** y
**Plana < 1**. Son valores de partida, no umbrales validados para este conjunto
de datos. Una epoch recibe un aviso si al menos uno de los canales analizados
cruza un límite. La igualdad no dispara el aviso; se exige `0 ≤ plana < alta`.
El barrido original también avisa de muestras no finitas. Si se solicita filtrado
y existen muestras no finitas, se detiene e indica usar el original para localizarlas,
porque filtrar esos valores puede extender el problema a otras epochs.
Se incluye el tramo final parcial;
un tramo de una sola muestra se identifica como insuficiente, no como plano.

La columna **Aviso** muestra Alta, Plana u otro problema. El gráfico añade
**REVISAR** y el editor detalla los canales y amplitudes de la epoch seleccionada.
**← Sospechosa / Sospechosa →** recorre los avisos, volviendo al principio/final
cuando corresponde. El resumen identifica qué electrodos, límites y filtrado se utilizaron.
Si el canal que disparó el aviso no está visible, seleccionarlo para inspeccionarlo.

Al terminar, las epochs **Sin revisar** que no tienen avisos pasan a **Aceptada**
y se guardan automáticamente en una sola operación. Las que tienen avisos quedan
pendientes de revisión. Se conservan las aceptaciones y rechazos manuales, junto
con sus notas. Una aceptación automática significa que se cumplen los límites del
barrido, no que se haya realizado una inspección visual; los avisos pueden incluir
actividad fisiológica de gran amplitud y no detectan todos los artefactos.

Las marcas automáticas guardan el método, los umbrales, los canales y el filtro
utilizados. El editor las identifica como **Aceptación automática**. Al repetir el
barrido, se reevalúan las decisiones automáticas: si una epoch antes aceptada ahora
tiene avisos, vuelve a **Sin revisar**. Cambiar manualmente su estado o nota la
convierte en una decisión manual que los barridos posteriores conservan.
Las aceptadas completas participan en el análisis espectral; las parciales siguen
excluidas. Si falla el guardado, los cambios quedan en memoria y se puede reintentar.

Los avisos del barrido permanecen en memoria para el estudio; cambiar umbrales,
alcance o estudio los descarta y exige repetir el barrido. Esto no borra las marcas
guardadas: se reevalúan al ejecutar el próximo barrido. Las muestras no se modifican.
Al usar el filtro activo, cambiar sus cortes o encenderlo/apagarlo descarta los
avisos y exige repetir el barrido. Si una nueva configuración falla, se conservan
el filtro y los avisos anteriores. En el modo original, cambiar el filtro de
visualización conserva los avisos: siguen refiriéndose al original y a los canales
indicados en el resumen.

## Anotaciones y protección de datos

El guardado automático escribe las marcas en `annotations/<registro>_eeg.json`,
fuera de `data`. Estos archivos están excluidos de Git: respaldarlos por separado.
Al abrir un estudio se recuperan las marcas guardadas.

El JSON versionado contiene identidad de los archivos fuente (ruta relativa,
tamaño y modificación), frecuencia, número de muestras, límites de cada epoch,
estado, motivo, nota, fecha UTC, canales observados y configuración de filtrado,
incluyendo la versión de MNE. Las marcas del barrido incluyen además su método y
umbrales de aceptación automática. La escritura utiliza un temporal y reemplazo atómico.

El indicador muestra **Guardando…** y luego **Guardado automáticamente**.
Las notas pendientes se guardan antes de cambiar de epoch, ventana, canal, filtro,
estudio o paciente, calcular el espectro o cerrar. Si falla el guardado, se conservan
los cambios en memoria y se habilita **Reintentar guardado**; se bloquean el cambio
de epoch/ventana y el análisis hasta resolverlo. Al cambiar de estudio o paciente,
o cerrar con cambios que no se pudieron guardar, se ofrece **Guardar**, **Descartar**
o **Cancelar**.
Si cambia el origen, el archivo de anotaciones fue modificado externamente o su
contenido es inválido, se protege contra sobrescritura y se informa el problema.
Conservar una copia del archivo y resolver la inconsistencia antes de continuar
la revisión; no se reemplaza silenciosamente un archivo anterior.

## Análisis espectral

En la pestaña **Análisis espectral**, elegir **Estudio actual** o **Comparar NS/SD**
y pulsar **Calcular**. Se usa el electrodo individual seleccionado; la lista
multicanal pertenece únicamente a la vista temporal. NS/SD compara el mismo
participante, electrodo y condición de ojos, sin mezclar ojos abiertos y cerrados.
Se requiere igual frecuencia de muestreo y al menos una epoch elegible por sesión.

Por defecto se usan solo las epochs **Aceptadas** y guardadas, incluidas las
aceptadas automáticamente por el barrido. **Incluir también
las no revisadas** habilita un análisis **Exploratorio**, claramente identificado.
Siempre se excluyen las epochs con artefactos y las parciales. Las parciales se
cuentan por separado; los otros conteos se refieren a epochs completas. Si no hay
archivo de anotaciones, todas las epochs se consideran no revisadas. Las notas
pendientes se guardan antes de calcular. Si falla ese guardado o hay un archivo
inválido/incompatible, el cálculo se bloquea.

Welch se calcula de manera independiente por epoch completa de 5 s, **sin concatenar
intervalos**. Se promedian las PSD con igual peso en escala lineal. Se aplica el
filtro activo sobre cada canal continuo antes de segmentar, con la referencia
original, sin agregar otros pasos de preprocesamiento. Los resultados muestran
cantidad de epochs, segundos utilizados, exclusiones y referencia por sesión.
Una diferencia entre el filtro de revisión y el de análisis se informa sin cambiar
las marcas. Diferentes cantidades de epochs NS/SD se conservan, sin equilibrado artificial.

**Parámetros espectrales** permite ajustar el segmento Welch (2 s por defecto),
solapamiento (50 %) y cinco fronteras compartidas entre las bandas (0,5 / 4 / 8 /
13 / 30 Hz). Se usa Hann periódica, detrend constante, espectro unilateral,
`scaling='density'`, promedio aritmético y FFT igual a la longitud del segmento,
sin relleno de ceros. El diálogo y el resumen muestran los parámetros efectivos
en muestras y la separación entre frecuencias. El detrend se aplica dentro de
Welch, sin cambiar las muestras del registro original ni la vista temporal.

La PSD se presenta inicialmente en dB respecto de **1 µV²/Hz** y puede mostrarse
en escala lineal **µV²/Hz**. La conversión a dB se realiza después del promedio;
los ceros se omiten de la curva logarítmica, sin inventar un piso de potencia.

Las bandas iniciales son delta 0,5–4, theta 4–8, alfa 8–13 y beta 13–30 Hz. Se
integra la PSD promedio mediante trapecios, interpolando los límites de cada banda.
La tabla muestra potencia absoluta en **µV²** y relativa en **%** respecto del
intervalo entre la primera y última frontera (inicialmente 0,5–30 Hz). Si la
potencia total es cero, la relativa se indica como no disponible. Las bandas
parcialmente fuera del pasabanda se señalan, sin recortarlas silenciosamente.

Cambiar de participante, estudio, electrodo individual, filtrado, parámetros,
política de inclusión o marcas invalida los resultados y exige recalcular.
No se realizan pruebas estadísticas ni agregación grupal en esta versión.

### Exportar

**Exportar resultados** permite elegir una carpeta. Se crea una subcarpeta nueva
con participante, condición de ojos, electrodo y fecha UTC, sin sobrescribir resultados:

- `PSD.csv`: PSD lineal por frecuencia y sesión, en µV²/Hz.
- `potencias.csv`: potencias absolutas y relativas por banda y sesión.
- `configuracion.json`: procedencia, identidad del origen y anotaciones, índices
  y límites de epochs utilizadas, referencias, parámetros efectivos y versiones.
- `PSD.png`: solo si se activa **Incluir gráfico PNG**, apagado por defecto.

Con el filtro activo, el eje de frecuencias llega al corte superior más **5 Hz**,
sin superar la frecuencia máxima disponible en la PSD. Sin filtrado, conserva el
rango habitual (al menos 45 Hz o hasta el límite superior de las bandas).
La figura exportada usa la escala elegida y este mismo rango inicial,
con leyenda y etiqueta exploratoria cuando corresponde. Los CSV siempre conservan
valores lineales. La exportación es manual y no modifica anotaciones ni datos originales.

## Datos y organización

Se espera BIDS: `data/sub-*/ses-*/eeg/*_eeg.set`, sus `.fdt` y metadatos JSON.
Los enlaces git-annex deben tener contenido local. Los archivos incompletos se
rechazan antes de graficar. Frecuencia, duración y canales provienen del registro;
la referencia proviene del JSON. Los campos ausentes se indican como no disponibles.

- `eeg_viewer/data.py`: inventario y acceso de solo lectura por canal/tramo.
- `eeg_viewer/processing.py`: configuración y caché del filtro continuo.
- `eeg_viewer/annotations.py`: epochs, identidad del origen y persistencia atómica.
- `eeg_viewer/spectral.py`: Welch por epoch, bandas y exportación.
- `eeg_viewer/screening.py`, `eeg_viewer/screening_ui.py`: barrido de amplitud y avisos de revisión.
- `eeg_viewer/app.py`, `eeg_viewer/review_ui.py` y `eeg_viewer/spectral_ui.py`:
  interfaz y coordinación asíncrona.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan archivos temporales y señales conocidas para comprobar unidades,
filtrado, conservación del origen, límites de epochs, persistencia y fallos de
escritura; también cubren navegación, multicanal y respuestas asíncronas obsoletas.
Las pruebas de GUI requieren pantalla y se omiten si no hay una disponible.
Con Xvfb: `xvfb-run -a python -m unittest discover -s tests -v`.
