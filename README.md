# PSIB — Explorador EEG

GUI de escritorio para explorar registros de sueño normal (NS) y privación del sueño (SD)
del dataset OpenNeuro ds004902: filtrado opcional (pasabanda y notch), ICA con
confirmación manual, revisión de artefactos en épocas de 5 segundos, espectros de Welch
y potencia por bandas, y el análisis de grupo SD vs NS (lote, estadística y figuras),
que también se puede ejecutar desde la GUI. Python, Tkinter, Matplotlib y MNE; los
archivos EEG originales nunca se modifican.

| Documento | Contenido |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | método del análisis de grupo: lote, regiones, estadística y clusters |
| [docs/hypotheses.md](docs/hypotheses.md) | hipótesis H1–H4 (referencias pendientes, TODO) |
| [docs/methods_draft.md](docs/methods_draft.md) | borrador de Materiales y Métodos |
| [docs/validation.md](docs/validation.md) | piloto de limpieza y validación (`validate_eeg.py`), independiente de la GUI |

## Instalación y ejecución

Requiere Python 3.10 o superior y una sesión gráfica. Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt   # incluye requirements.txt
python gui.py
```

En Ubuntu/Debian, si faltan Tkinter o venv: `sudo apt install python3-tk python3-venv`.
En Linux también podés ejecutar `./TPS.sh`: crea `.venv` e instala las dependencias
en el primer inicio (requiere internet). Si la instalación falla, la reintenta al
volver a ejecutarlo. Acepta su ruta absoluta desde cualquier carpeta.
En Windows, hace doble clic en `TPS.bat`. En el primer inicio crea `.venv` e instala
las dependencias (también pandas, para «Resultados del grupo»); hace falta una conexión a internet. Si falla la instalación,
corregí el problema y volvé a ejecutar el lanzador: reintentará la instalación.
También se puede ejecutar `TPS.bat` por su ruta absoluta desde otro directorio:
`data` y `annotations` siempre se resuelven junto al proyecto.

## Guía en pantalla

La ventana sigue el estilo de la GUI de la guía PSIB. La tarjeta **Pasos** del panel
izquierdo lista el flujo (paciente, estudio, electrodo,
filtrado opcional, búsqueda opcional de artefactos, revisión de épocas y espectro): el paso pendiente se resalta en ámbar
y **Ir al siguiente paso ▶** abre su herramienta. La franja ámbar sobre el gráfico
indica qué hacer ahora, y al dejar el mouse sobre un control aparece una ayuda.
**◀** en la esquina superior derecha de **Registro** pliega el panel lateral;
**▶** vuelve a mostrarlo. **▲** en la esquina inferior derecha de **Información
del estudio** la reduce a un resumen; **▼** la expande. La elección se conserva
al redimensionar la ventana.

## Resultados del grupo

La tercera pestaña reúne el análisis por lotes, la estadística SD vs NS y las figuras,
sin usar la terminal. Lee y escribe en `results/` (junto a `data/`).

- **Proceso**: tres pasos (1 Potencias por lote, 2 Estadística, 3 Figuras) con su estado
  (listo, sin resultados o desactualizado si cambió `tps/config.py` o un paso anterior), sus
  opciones (ojos, procesos en paralelo, recalcular todo, permutaciones) y **Ejecutar**.
  **Ejecutar todo** los encadena; **Cancelar** detiene el paso en curso (el lote es
  reanudable). Los pasos corren en un proceso aparte: el explorador sigue disponible.
  El registro muestra la salida de cada paso y se abre solo si algo falla.
- **Hipótesis**: H1–H4 y la tabla de dirección esperada vs observada, p corregido y
  tamaño del efecto, coloreada por acuerdo (verde apoyada, rojo opuesta, gris no
  significativa). Las referencias siguen pendientes (TODO).
- **Tablas**: todas las tablas `stats_*.csv`, clusters, exclusiones y resumen por registro,
  con filtros por ojos, sesión, banda, región y participante, y orden por columna
  (clic en el encabezado). Se resaltan las filas con p < 0,05.
- **Figuras**: topomapas, PSD por región, gráficos pareados y theta/beta con su leyenda
  (`captions.md`), en versión absoluta y relativa; zoom con los botones o Ctrl + rueda
  y arrastre para mover.

En el panel lateral, **Resultado del lote** muestra para el estudio abierto si el lote lo
incluyó o lo excluyó (y por qué), las épocas usadas y rechazadas, los canales malos y,
para el electrodo elegido, sus potencias por banda y el cociente theta/beta. Si el
electrodo fue un canal malo en el lote se avisa en ámbar. Las épocas que el lote rechazó
aparecen con «Lote» en la columna Aviso, y su motivo en el detalle de la época.
**Ver exclusiones** abre la tabla de exclusiones filtrada a ese estudio. Aceptada por el
lote significa «dentro de los límites de amplitud», no inspección visual.

## Explorar y filtrar

1. Elegir **Paciente**, **Estudio** y **Electrodo** (el ícono ⓘ muestra la disposición de los electrodos). `ses-1` es sueño normal y
   `ses-2` es privación del sueño. La información de adquisición aparece al cargar.
2. Se muestran ventanas de **10 segundos**, amplitud en **µV** y líneas verticales
   cada **5 segundos**. **Anterior**, **Siguiente** e **Inicio (s) → Ir** recorren
   el registro. El inicio acepta punto o coma decimal; la última ventana puede ser parcial.
3. La barra del gráfico permite zoom, desplazamiento y restauración. Cambiar de
   electrodo conserva el tiempo y la época seleccionada.
4. **⚙ Configurar filtrado**, en la tarjeta **Filtrado** del panel lateral, permite activar el filtrado y ajustar sus cortes. Al iniciar,
   está apagado y los cortes son **0,2–45 Hz**. **Aplicar** valida y prepara el filtro
   en segundo plano; **Cancelar** conserva la configuración anterior. Restablecer
   recupera esos cortes sin modificar el toggle.
5. El encabezado identifica **Original** o **Filtrada · X–Y Hz**. La configuración
   se conserva durante la sesión de la aplicación. Un filtro inválido o demasiado
   largo para el registro no reemplaza la vista anterior.
6. **Vista** (tarjeta «Registro») elige cómo se muestra la señal:
   - **Un electrodo** (por defecto).
   - **Varios (hasta 8)**: de 1 a 8 canales en paneles con tiempo compartido.
   - **Todos apilados**: los 61 canales en un solo gráfico con una escala común en µV
     (como el *eegplot* de EEGLAB). Cada traza se centra en su mediana, así un canal plano
     o ruidoso se distingue enseguida. **Escala** − / + cambia los µV entre trazas y
     **Canales** elige cuántos entran en pantalla (el resto se ve desplazando el gráfico).
     Al pasar el cursor se ve el canal y su valor; **un clic en el nombre de un canal lo
     marca o desmarca como malo**. Los canales malos se dibujan en gris con el nombre en rojo.

### Canales malos

La tarjeta **Canales malos** reúne dos fuentes:

- **Detectar canales malos** busca en todo el registro, con el filtrado del explorador,
  canales con señal plana (< 1 µV durante ≥ 5 s), amplitud anómala (z robusto > 5 de su
  desvío robusto), baja correlación con los demás canales (< 0,4 en más del 20 % de las
  ventanas de 1 s) o ruido de alta frecuencia (z robusto > 5 de la potencia sobre 45 Hz, sin
  pasabanda). Son los criterios de EEGLAB (clean_rawdata, pop_rejchan) y PREP, reimplementados
  en `tps/core/badchannels.py`; los umbrales están en `tps/config.py` y **no están validados**
  para este dataset. Cada canal indica su motivo.
- **Marcar / desmarcar electrodo actual** (o un clic en su nombre en la vista apilada) lo
  marca a mano, con un **Motivo** opcional. Se guarda en las anotaciones del estudio.

El análisis de grupo excluye ambos (además de la regla de pico a pico en más del 30 % de las
épocas) y anota el criterio en `exclusions.csv`. El barrido de posibles artefactos ignora los
canales marcados a mano, para no rechazar épocas por un canal que ya se sabe malo.

### Implementación del filtro

Se utiliza FIR de fase cero en MNE, diseño `firwin`, ventana Hamming, longitud y
bandas de transición automáticas, padding `reflect_limited`. La GUI configura
los cortes; el resto del diseño queda fijo. Se filtra **todo el canal continuo**
antes de extraer ventanas, nunca cada época por separado. Los canales filtrados
se conservan en una caché en memoria para el estudio y los cortes activos.

El pasabanda se basa en el paper; su diseño concreto no está especificado allí,
por lo que esta implementación no pretende reproducir exactamente su filtro.
Los bordes del registro pueden tener transitorios de filtrado: el padding no
los elimina por completo. El notch opcional se activa en **Configurar filtrado**, con un selector de **50 o 60 Hz**,
independiente del pasabanda y apagado por defecto. Se aplica después del pasabanda
sobre el canal continuo: IIR de segundo orden, Q = 30, ida y vuelta (fase cero,
padding impar de 9 muestras). La frecuencia debe ser menor que Nyquist.
La vista, el barrido con filtro activo y el espectro utilizan la misma configuración;
las anotaciones y exportaciones registran el notch. Cambiarlo invalida los avisos
del barrido filtrado. Las anotaciones antiguas se interpretan sin notch.
No se aplica remuestreo ni interpolación automática,
cambio de referencia ni normalización. Se dibujan todas las muestras del tramo.

## ICA: corrección manual de artefactos

En el panel **ICA**, abrí **Ajustar / revisar ICA**. El ajuste utiliza todos los
canales EEG salvo los que excluyas por nombre, una copia continua de entrenamiento
**1–45 Hz**, épocas completas de **5 s**, Infomax extendida y semilla 42.
Las exclusiones de entrenamiento son independientes de las marcas de revisión:
los avisos de amplitud y las épocas rechazadas **no se excluyen automáticamente**.
Conservá ejemplos representativos de parpadeos para que ICA pueda separarlos.

1. Opcionalmente indicá canales y números de épocas que no deben entrar al ajuste.
   Los números de épocas y componentes mostrados empiezan en **1**.
2. Pulsá **Ajustar ICA**. Corre en otro proceso y admite **Cancelar ajuste**.
   Cancelar, cambiar de estudio o cerrar conserva la corrección confirmada anterior.
3. Elegí un componente y pulsá **Inspeccionar** para ver señal, espectro e indicadores.
   **Mapas de componentes** abre una grilla desplazable: clic para marcar/conservar,
   clic derecho para inspeccionar. Los rótulos verdes/rojos indican la selección
   pendiente, sincronizada con la lista de exclusiones; aún requiere confirmar.
   Si el modelo no tiene geometría, se usan posiciones estándar 10–20 por nombre,
   rotuladas como **aproximadas**, sin modificar el modelo ni la señal. Canales
   desconocidos impiden dibujar el mapa. Cada mapa normaliza sus pesos a ±1;
   sus colores no representan µV ni una clasificación de artefactos. Las sugerencias y la
   correlación frontal no identifican automáticamente artefactos ni sustituyen EOG.
4. Escribí los componentes que querés excluir. **Antes / después** compara la misma
   ventana y filtros, con ambas señales superpuestas sobre la misma escala.
5. **Confirmar y aplicar** guarda una revisión nueva. Podés confirmar cero exclusiones.
   **Usar ICA confirmada** permite desactivar/reactivar la corrección.

Los canales excluidos del ajuste quedan **sin corregir** y se identifican en el
panel y los metadatos. No se aplica interpolación automática. Cambiar las exclusiones
de entrenamiento requiere otro ajuste. El modelo se aplica a una copia de la señal
original y después se aplican el pasabanda y notch elegidos; cambiar estos filtros
no requiere volver a ajustar ICA.

La vista, el barrido y Welch usan la misma señal seleccionada. Cambiar ICA invalida
avisos y espectros anteriores; conserva las marcas de épocas y advierte cuando
fueron hechas con otra corrección. Una época rechazada no pasa a aceptada solo por
activar ICA. Para comparar NS/SD con ICA, ambas sesiones necesitan una revisión
confirmada compatible; cada sesión usa su propio modelo.

Las revisiones se guardan fuera de `data`, en `annotations/ica/<registro>/`, junto
con el modelo, componentes excluidos, canales sin corregir, fuente y versiones.
Se restaura la última corrección si estaba activada. Fuente o entorno incompatibles
impiden reutilizarla. Respaldá esta carpeta junto con las anotaciones.
Las revisiones del piloto de 4 s siguen siendo independientes.

En **Resultados del grupo**, **Señal de entrada** tiene dos opciones, y las dos usan
el **mismo filtrado que el explorador** (panel «Filtrado»; debajo del selector se
indica cuál es):

- **Filtrado activo**: la señal con ese filtrado.
- **Filtrado activo + ICA confirmada**: primero la ICA confirmada de cada registro y
  después el mismo filtrado.

El pasabanda y notch se copian al iniciar el lote; cambiar los filtros durante una
ejecución requiere volver a ejecutar Potencias por lote para aplicarlos. El panel
muestra los parámetros actuales y los de los resultados guardados.

El lote nunca ajusta modelos ni elige componentes: excluye e informa registros sin
corrección válida (antes de ejecutarlo se avisa cuántos registros tienen ICA
confirmada). El aviso ámbar indica con qué señal se calcularon los resultados y
advierte si el filtrado del explorador cambió desde entonces. Las estadísticas pareadas usan
solo pares NS/SD completos. Se registra cobertura en `ica_coverage.json`, contexto
por registro y revisiones en `config_used.json`; las cachés distinguen las fuentes
y revisiones. Una revisión nueva no cambia los insumos de un lote ya iniciado.

```bash
python -m tps.pipeline.batch --ica-mode confirmed --annotations-dir annotations
```

## Revisar épocas

- Las épocas son intervalos consecutivos de **5 s**, sin solapamiento, desde el
  inicio: `[0,5)`, `[5,10)`, etc. Un registro de 300 s tiene 60 épocas. Un resto
  final se conserva como **parcial**. El paper utiliza 4 s; aquí se eligieron 5 s.
- Elegir una época en la lista lateral lleva a una ventana que la contiene completa.
  También se puede seleccionarla haciendo clic en su intervalo del gráfico, con
  las herramientas de zoom/desplazamiento desactivadas. El borde ámbar indica
  la selección; colores y texto indican su estado.
- Los estados son **Sin revisar**, **Aceptada** y **Con artefactos**. Todas comienzan
  sin revisar. Las marcas son **globales para la época del estudio**, no por canal.
- Elegir estado, motivo opcional (ocular, muscular, movimiento/electrodo u otro)
  y nota de hasta 500 caracteres. El estado y el motivo se guardan automáticamente
  al cambiarlos; las notas, tras 400 ms sin escribir. No hay que aplicar ni guardar
  cada marca. Se puede cambiar la evaluación tantas veces como sea necesario.
- Atajos fuera de campos de texto: **A** acepta, **U** vuelve a sin revisar;
  **R** marca artefactos y enfoca el motivo. Todos guardan automáticamente.
- Los contadores resumen el progreso. Las épocas con artefactos siguen visibles:
  esta versión **no elimina ni excluye muestras**.
- Cambiar de filtro o canal conserva las marcas. El editor muestra el filtro y los
  canales observados durante la última evaluación; activar un filtro nuevo no
  revalida automáticamente una decisión anterior.

## Barrido de posibles artefactos

Después de elegir un electrodo, pulsar **Buscar posibles artefactos** en el panel
lateral. Por defecto recorre todos los canales EEG del estudio; desactivar
**Todos los electrodos EEG** limita el barrido a los visibles (uno o hasta ocho).
Se calcula `máximo − mínimo` por canal en cada época de 5 s. Por defecto está
marcado **Usar el filtro activo**: utiliza la misma configuración de filtrado que
la vista temporal y el análisis espectral. El FIR se aplica al **canal continuo
completo antes de dividirlo en épocas**, aprovechando la caché existente. El panel,
el resumen y el detalle identifican **Filtrada · X–Y Hz** o **Original**.

Si el filtrado está desactivado en **⚙ Configurar filtrado**, el barrido utiliza el original
y lo indica explícitamente; esta opción no enciende el filtro automáticamente.
Desmarcar **Usar el filtro activo** permite revisar siempre la señal original,
aunque se esté mostrando una señal filtrada. Así se pueden distinguir avisos por
deriva que disminuyen con el pasabanda de problemas visibles en la adquisición.

Los límites iniciales, ajustables en µV pico a pico, son **Alta > 150** y
**Plana < 1**. Son valores de partida, no umbrales validados para este conjunto
de datos. Una época recibe un aviso si al menos uno de los canales analizados
cruza un límite. La igualdad no dispara el aviso; se exige `0 ≤ plana < alta`.
El barrido original también avisa de muestras no finitas. Si se solicita filtrado
y existen muestras no finitas, se detiene e indica usar el original para localizarlas,
porque filtrar esos valores puede extender el problema a otras épocas.
Se incluye el tramo final parcial;
un tramo de una sola muestra se identifica como insuficiente, no como plano.

La columna **Aviso** muestra Alta, Plana u otro problema. El gráfico añade
**REVISAR** y el editor detalla los canales y amplitudes de la época seleccionada.
**← Sospechosa / Sospechosa →** recorre los avisos, volviendo al principio/final
cuando corresponde. El resumen identifica qué electrodos, límites y filtrado se utilizaron.
Si el canal que disparó el aviso no está visible, seleccionarlo para inspeccionarlo.

Al terminar, las épocas **Sin revisar** que no tienen avisos pasan a **Aceptada**
y se guardan automáticamente en una sola operación. Las que tienen avisos quedan
pendientes de revisión. Se conservan las aceptaciones y rechazos manuales, junto
con sus notas. Una aceptación automática significa que se cumplen los límites del
barrido, no que se haya realizado una inspección visual; los avisos pueden incluir
actividad fisiológica de gran amplitud y no detectan todos los artefactos.

Las marcas automáticas guardan el método, los umbrales, los canales y el filtro
utilizados. El editor las identifica como **Aceptación automática**. Al repetir el
barrido, se reevalúan las decisiones automáticas: si una época antes aceptada ahora
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
tamaño y modificación), frecuencia, número de muestras, límites de cada época,
estado, motivo, nota, fecha UTC, canales observados y configuración de filtrado,
incluyendo la versión de MNE. Las marcas del barrido incluyen además su método y
umbrales de aceptación automática. La escritura utiliza un temporal y reemplazo atómico.

El indicador muestra **Guardando…** y luego **Guardado automáticamente**.
Las notas pendientes se guardan antes de cambiar de época, ventana, canal, filtro,
estudio o paciente, calcular el espectro o cerrar. Si falla el guardado, se conservan
los cambios en memoria y se habilita **Reintentar guardado**; se bloquean el cambio
de época/ventana y el análisis hasta resolverlo. Al cambiar de estudio o paciente,
o cerrar con cambios que no se pudieron guardar, se ofrece **Guardar**, **Descartar**
o **Cancelar**.
Si cambia el origen, el archivo de anotaciones fue modificado externamente o su
contenido es inválido, se protege contra sobrescritura y se informa el problema.
Conservar una copia del archivo y resolver la inconsistencia antes de continuar
la revisión; no se reemplaza silenciosamente un archivo anterior.

## Análisis espectral

En la pestaña **Análisis espectral**, elegir **Estudio actual**, **Comparar NS/SD** o
**Todos los canales** y pulsar **Calcular**.

**Todos los canales** (como «Channel spectra and maps» de EEGLAB) calcula la PSD de cada
canal con las mismas épocas: las curvas se superponen con la mediana en negrita, y los
canales malos o atípicos en 30–45 Hz (z robusto > 3) aparecen en rojo con su nombre; un clic
en una curva muestra su canal. Debajo, mapas topográficos del log10 de la potencia por banda
(posiciones estándar 10-05). **Exportar resultados** guarda la PSD y las bandas de cada canal.

En los otros dos modos Se usa el electrodo individual seleccionado; la lista
multicanal pertenece únicamente a la vista temporal. NS/SD compara el mismo
participante, electrodo y condición de ojos, sin mezclar ojos abiertos y cerrados.
Se requiere igual frecuencia de muestreo y al menos una época elegible por sesión.

Por defecto se usan solo las épocas **Aceptadas** y guardadas, incluidas las
aceptadas automáticamente por el barrido. **Incluir también
las no revisadas** habilita un análisis **Exploratorio**, claramente identificado.
Siempre se excluyen las épocas con artefactos y las parciales. Las parciales se
cuentan por separado; los otros conteos se refieren a épocas completas. Si no hay
archivo de anotaciones, todas las épocas se consideran no revisadas. Las notas
pendientes se guardan antes de calcular. Si falla ese guardado o hay un archivo
inválido/incompatible, el cálculo se bloquea.

Welch se calcula de manera independiente por época completa de 5 s, **sin concatenar
intervalos**. Se promedian las PSD con igual peso en escala lineal. Se aplica el
filtro activo sobre cada canal continuo antes de segmentar, con la referencia
original, sin agregar otros pasos de preprocesamiento. Los resultados muestran
cantidad de épocas, segundos utilizados, exclusiones y referencia por sesión.
Una diferencia entre el filtro de revisión y el de análisis se informa sin cambiar
las marcas. Diferentes cantidades de épocas NS/SD se conservan, sin equilibrado artificial.

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
  y límites de épocas utilizadas, referencias, parámetros efectivos y versiones.
- `PSD.png`: solo si se activa **Incluir gráfico PNG**, apagado por defecto.

Con el filtro activo, el eje de frecuencias llega al corte superior más **5 Hz**,
sin superar la frecuencia máxima disponible en la PSD. Sin filtrado, conserva el
rango habitual (al menos 45 Hz o hasta el límite superior de las bandas).
La figura exportada usa la escala elegida y este mismo rango inicial,
con leyenda y etiqueta exploratoria cuando corresponde. Los CSV siempre conservan
valores lineales. La exportación es manual y no modifica anotaciones ni datos originales.

## Análisis por lotes, estadística y figuras

Se ejecuta desde **Resultados del grupo › Proceso** o en la terminal, desde la carpeta
del proyecto (en Windows, `.venv\Scripts\python.exe` en lugar de `python`):

```bash
python -m tps.pipeline.batch --jobs 6   # 1. potencia por bandas de cada registro (~4 min)
python -m tps.analysis.stats            # 2. regiones y comparaciones SD vs NS (~20 s)
python -m tps.analysis.figures          # 3. figuras, tabla de hipótesis y leyendas
```

Cada paso lee lo que produce el anterior en `results/`. El lote es reanudable: vuelve a
calcular solo los registros nuevos o los afectados por un cambio de `tps/config.py`.
Opciones del lote:

- `--eyes open|closed`, `--subjects 01 02`, `--force`, `--jobs N`
- `--subjects-file subsets/group20_seed2026.txt`: un subconjunto fijo de participantes
  (uno por línea, `#` para comentarios). La lista de 20 participantes (semilla 2026) está
  en `subsets/`.
- `--rejection-mode global|per_channel`: con `global` (por defecto) una época marcada en
  cualquier canal se descarta para todos; con `per_channel` cada canal usa sus propias
  épocas limpias y se excluye si le quedan menos de `MIN_CHANNEL_EPOCHS` (10).
- `--ica-mode original|confirmed`, `--annotations-dir`: usar la ICA confirmada de cada registro.
- `--bandpass LOW HIGH`, `--no-bandpass`, `--notch 50|60`: filtrado de la señal (por
  defecto el de `tps/config.py`, pasabanda 0,2–45 Hz). La interfaz pasa el filtrado del explorador.

Ojos abiertos es el análisis principal y ojos cerrados el secundario; nunca se combinan.
Los resultados describen diferencias espectrales a nivel de grupo, sin interpretación
clínica. «Aceptada» en el barrido automático significa «dentro de los límites de
amplitud», no inspección visual. El paso 3 también escribe `results/report.md`: un solo
documento con todas las tablas, las hipótesis y las figuras con sus leyendas (botón
**Abrir informe** en **Proceso**).

El detalle del método (orden de cada paso, umbrales, pruebas y limitaciones) está en
[docs/pipeline.md](docs/pipeline.md).

Salidas en `results/` (no se versionan; se regeneran con los comandos anteriores):

| Archivo | Contenido |
|---|---|
| `features_long.csv`, `theta_beta.csv` | potencia absoluta y relativa por canal y banda; cociente theta/beta |
| `psd_channels.csv` | PSD media por registro y canal, 0–45 Hz (µV²/Hz) |
| `exclusions.csv`, `recordings_summary.csv` | registros, canales y épocas excluidos, con el motivo |
| `session_order.csv`, `config_used.json`, `ica_coverage.json` | orden NS/SD; configuración e ICA usadas |
| `features_region.csv` | log10 de la potencia y potencia relativa por región |
| `stats_region_log_power.csv`, `stats_region_rel_power.csv` | pruebas pareadas SD vs NS por banda y región |
| `stats_relative_vs_absolute.csv` | resultados absolutos y relativos lado a lado |
| `stats_region_log_power_min20.csv` | sensibilidad: solo registros con ≥ 20 épocas limpias |
| `stats_frontal_theta_beta.csv`, `stats_session_order.csv`, `stats_epochs.csv`, `stats_participants.csv` | theta/beta, orden de sesión, épocas y participantes |
| `cluster_results.csv`, `stats_cluster_channels.csv` | prueba de clusters por electrodo |
| `figures/*.png`, `figures/captions.md` | figuras (absolutas y relativas) y sus leyendas |
| `channel_regions.csv` | canales de cada región |
| `report.md` | informe con todas las tablas, hipótesis y figuras |
| `figures/hypothesis_summary.csv/.md` | hipótesis, dirección esperada y observada, p corregido, tamaño del efecto |

## Estructura del código

```
gui.py              abre el explorador (también TPS.bat / TPS.sh)
validate_eeg.py     piloto de validación (docs/validation.md)
tps/
  config.py         configuración compartida: filtro, épocas, Welch, bandas, umbrales, mínimos
  core/             procesamiento sin interfaz: datos BIDS, filtros, épocas y anotaciones,
                    barrido, canales malos, Welch y bandas, ICA (revisiones e Infomax),
                    utilidades JSON
  gui/              ventanas Tk: app, tema, revisión, barrido, espectro, ICA,
                    «Resultados del grupo» y ejecución de los pasos en segundo plano
  pipeline/         lote (batch.py) y ajuste de ICA en un proceso aparte (ica_worker.py)
  analysis/         regiones, estadística, hipótesis, figuras e informe (report.py)
  validation/       piloto de limpieza y validación (motor, sesión, almacén, interfaz, CLI)
tests/              pruebas (sintéticas y de GUI)
docs/               documentación detallada
assets/             imágenes (disposición de electrodos)
subsets/            listas de participantes para el lote
```

`core` no depende de Tk ni del piloto de validación; `gui`, `pipeline` y `analysis` usan
`core`. Los datos se esperan en `data/` (BIDS: `data/sub-*/ses-*/eeg/*_eeg.set`, sus
`.fdt` y metadatos JSON); las anotaciones y revisiones ICA en `annotations/`, y los
resultados en `results/`, todos junto al proyecto. Los enlaces git-annex deben tener
contenido local; los archivos incompletos se rechazan antes de graficar.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan archivos temporales y señales conocidas para comprobar unidades,
filtrado, conservación del origen, límites de épocas, persistencia y fallos de
escritura; también cubren navegación, multicanal y respuestas asíncronas obsoletas.
Las pruebas de GUI requieren pantalla y se omiten si no hay una disponible.
Con Xvfb: `xvfb-run -a python -m unittest discover -s tests -v`.

### Configuración explícita del lote y muestras pequeñas

Desde la GUI, los filtros activos se transmiten como parámetros a los procesos de
trabajo, sin modificar la configuración global. Desde CLI, los valores por defecto
siguen siendo pasabanda 0,2–45 Hz y notch apagado; se imprimen al iniciar y se guardan
en `config_used.json`. Se pueden elegir explícitamente, por ejemplo:

```bash
python -m tps.pipeline.batch --bandpass enabled --low-hz 1 --high-hz 40 --notch-hz 50 --ica-mode confirmed
python -m tps.pipeline.batch --bandpass disabled --notch-hz 0 --ica-mode original
```

Las figuras descriptivas y diferencias NS/SD se muestran desde **un par completo**.
La PSD de un solo participante se dibuja sin intervalo de confianza. El mínimo de
**3 pares** es una política de las pruebas inferenciales del proyecto, no de las
figuras ni de ICA; no garantiza por sí solo una muestra suficiente. Con menos pares
las medias/diferencias siguen disponibles y los p-valores quedan sin calcular.
