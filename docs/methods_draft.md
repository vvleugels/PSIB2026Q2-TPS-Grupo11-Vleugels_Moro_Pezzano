# Materiales y métodos (borrador)

> Borrador generado a partir del código y de `tps/config.py`. Cada número de este documento sale
> de la configuración o de los archivos del dataset; los puntos marcados **TODO** deben ser
> revisados o completados por los autores antes de la entrega. No se hacen afirmaciones clínicas:
> el análisis describe solo diferencias espectrales a nivel de grupo.
>
> Describe la ejecución por defecto (pasabanda 0,2–45 Hz, sin ICA). TODO: si los resultados
> finales se calculan con otro filtrado o con la opción «Filtrado activo + ICA confirmada»,
> actualizar las secciones de preprocesamiento (los valores usados quedan en
> `results/config_used.json`).

## Dataset

Se usó el dataset público «A Resting-state EEG Dataset for Sleep Deprivation» (OpenNeuro
ds004902, versión 1.0.8, doi:10.18112/openneuro.ds004902.v1.0.8; descriptor de datos: Xiang, Fan,
Bai, Lv y Lei, *Scientific Data* 11, 427, 2024, https://doi.org/10.1038/s41597-024-03268-2),
organizado en BIDS 1.8.0 y publicado bajo licencia CC0.

* **Participantes:** 71 (sexo y edad en `participants.tsv`). TODO: informar la distribución por
  sexo y la edad media ± DE de la muestra analizada, y los criterios de inclusión del descriptor.
* **Diseño:** intrasujeto. Cada participante se registró tras una noche de sueño normal (NS,
  `ses-1`) y tras privación del sueño (SD, `ses-2`). El número de sesión no refleja el orden
  temporal: el orden se contrabalanceó y figura en la columna `SessionOrder` de `participants.tsv`
  (NS→SD o SD→NS). TODO: describir el protocolo de privación del sueño (duración de la vigilia,
  supervisión, tiempo entre sesiones) según el descriptor.
* **Registros:** EEG en reposo de 5 min (300 s) con ojos abiertos para todos los participantes;
  un subgrupo (38 participantes) tiene además registros con ojos cerrados. TODO: confirmar si el
  descriptor presenta la condición de ojos cerrados como disponible «parcialmente», y las
  instrucciones dadas.
* **Adquisición (según los archivos BIDS):** sistema Brain Products GmbH, 61 electrodos EEG en
  disposición 10-10, frecuencia de muestreo 500 Hz, referencia en línea FCz, frecuencia de red
  50 Hz, impedancias menores a 5 kΩ (README del dataset). TODO: modelo de amplificador, electrodo
  de tierra y filtros en línea (el archivo indica `SoftwareFilters: n/a`).

## Preprocesamiento

El procesamiento se hizo en Python con MNE-Python, NumPy y SciPy (`tps/pipeline/batch.py`; las
versiones se guardan en `results/config_used.json`. TODO: copiar aquí las versiones exactas). Los
archivos originales nunca se modificaron; los datos se analizaron con la referencia original
FCz, sin re-referenciar, interpolar, aplicar ICA ni remuestrear.

1. **Control de archivos:** se excluyeron los registros cuyo archivo binario `.fdt` es más corto
   de lo que exige su encabezado. Cinco registros se excluyeron por este motivo (OA: sub-01 SD,
   sub-28 SD, sub-44 SD; OC: sub-03 SD, sub-28 NS).
2. **Filtrado:** cada canal continuo se filtró con un pasabanda de 0,2 a 45 Hz antes de dividirlo
   en épocas, con un filtro FIR de fase cero (no causal) y una sola pasada (diseño por ventana,
   `firwin`, ventana Hamming, rizado en banda de paso de 0,0194, atenuación en banda de rechazo de
   53 dB). MNE fijó automáticamente los anchos de transición: 0,2 Hz en el borde inferior (−6 dB
   en 0,1 Hz) y 11,25 Hz en el borde superior (−6 dB en 50,6 Hz), con una longitud de filtro de
   8251 muestras (16,5 s). Los bordes de la señal se extendieron con `reflect_limited`. No se
   aplicó filtro notch; como el punto de −6 dB superior queda por encima de 50 Hz, el ruido de
   línea solo se atenúa en parte, lo que no afecta a las bandas analizadas (≤ 30 Hz).
3. **Épocas:** la señal filtrada se dividió en épocas consecutivas de 5 s sin solapamiento, desde
   la primera muestra; la última época incompleta se descartó (registros de 300 s → 60 épocas).
4. **Barrido de artefactos:** para cada época y cada uno de los 61 canales, la amplitud pico a
   pico de la señal filtrada se comparó con límites fijos: > 150 µV (amplitud alta) o < 1 µV
   (señal plana). Estos límites son valores iniciales y **no se validaron** contra inspección
   visual en este dataset. Un canal se consideró malo para un registro (excluido de todas las
   variables, sin interpolar y no usado para rechazar épocas) si fallaba en más del 30 % de las
   épocas completas, si se había marcado a mano tras inspección visual, o si cumplía alguno de
   estos criterios sobre el registro completo, adaptados de PREP (Bigdely-Shamlo et al., 2015) y
   EEGLAB clean_rawdata: variación menor a 1 µV durante al menos 5 s; z robusto > 5 de su desvío
   robusto respecto de los demás canales; percentil 98 de su correlación absoluta con los demás
   canales < 0,4 en más del 20 % de las ventanas de 1 s; o z robusto > 5 de la potencia relativa
   sobre 45 Hz. TODO: informar cuántos canales excluyó cada criterio (columna `criterion` de
   `exclusions.csv`) y verificar la referencia de PREP. Una época se rechazó para todos los canales si fallaba en cualquiera de
   los canales restantes. Los registros con menos de 12 épocas limpias (60 s) se excluirían;
   ninguno llegó a ese límite. La cantidad de épocas no se igualó entre sesiones.

## Variables espectrales

* **Densidad espectral de potencia:** método de Welch por época limpia (segmentos Hann periódicos
  de 2 s, 50 % de solapamiento, media de segmentos, escala de densidad, media removida de cada
  segmento), con una resolución de 0,5 Hz y cuatro segmentos por época. Los espectros de las
  épocas se promediaron por canal y registro, y se convirtieron a µV²/Hz.
* **Potencia por bandas:** la potencia absoluta se obtuvo por integración trapezoidal de la PSD
  media sobre delta (0,5–4 Hz), theta (4–8 Hz), alfa (8–13 Hz) y beta (13–30 Hz), con
  interpolación lineal en los bordes exactos de cada banda. La potencia relativa fue la potencia
  de la banda dividida por la potencia total de 0,5–30 Hz (%). El cociente theta/beta fue la
  potencia absoluta theta dividida por la potencia absoluta beta.
* **Regiones:** los canales se agruparon según su nombre 10-10 en frontal (Fp/AF/F; 16 canales),
  central (FC/C/CP; 20), temporal (FT/T/TP; 8), parietal (P; 9) y occipital (PO/O; 8). Los valores
  regionales son la media del log10 de la potencia absoluta (y de la potencia relativa) sobre los
  canales disponibles de la región, calculada solo cuando al menos el 50 % de sus canales era
  válido. Theta/beta frontal fue la media del log10 del cociente sobre los canales frontales.

## Análisis estadístico

* **Análisis:** ojos abiertos fue el análisis principal (todos los participantes con ambas
  sesiones); ojos cerrados fue el secundario (solo participantes con ambas sesiones de ojos
  cerrados). Las dos condiciones nunca se combinaron. Los participantes a los que les faltaba una
  sesión en una condición se excluyeron solo de ese análisis.
* **Comparaciones pareadas:** para cada banda × región, la diferencia intrasujeto SD − NS del
  log10 de la potencia absoluta se evaluó con una prueba t pareada a dos colas. El cambio se
  informa en dB (10 · diferencia media de log10) y como porcentaje de cambio de la potencia
  ((10^diferencia media − 1) · 100, es decir, el cociente de medias geométricas), con su intervalo
  de confianza del 95 % (distribución t). El cociente theta/beta frontal se evaluó igual, en una
  prueba por condición de ojos.
* **Comparaciones múltiples:** no se aplicó corrección; con 20 pruebas por condición de ojos se
  espera alrededor de un resultado significativo por azar. TODO: decidir si el informe final
  aplica una corrección (por ejemplo, Benjamini–Hochberg).
* **Figuras:** espectro de potencia en Cz en dB (media ± DE) y topografías de potencia por banda
  para NS y SD, en el formato de las Fig. 4 y 5 de Xiang et al. (2024).
* **Hipótesis:** H1–H4 se especificaron antes de inspeccionar los resultados
  (`docs/hypotheses.md`). TODO: confirmar la fecha en que se fijaron las hipótesis y agregar las
  referencias verificadas.
* **Nivel de significación:** α = 0,05 (a dos colas).

## Resumen de TODO

* Descripción de la muestra (sexo, edad, criterios de inclusión) y protocolo de privación del sueño.
* Modelo de amplificador, electrodo de tierra y filtros en línea.
* Versiones exactas del software (de `results/config_used.json` y `requirements*.txt`).
* Referencias verificadas para H1–H4 (`docs/hypotheses.md`).
* Referencias que justifiquen los límites de 150 µV / 1 µV, la regla del 30 % y los criterios automáticos para canales malos
  y el mínimo de 12 épocas, o una aclaración de que son decisiones pragmáticas.
* Si los resultados finales usan otro filtrado o ICA confirmada, actualizar el preprocesamiento.
