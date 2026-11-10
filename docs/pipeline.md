# Análisis de grupo: detalle del método

Cómo calculan sus resultados los tres pasos del análisis de grupo. Los comandos, las opciones
y la lista de archivos de salida están en el README («Análisis por lotes, estadística y
figuras»); este documento solo cubre qué ocurre dentro de cada paso y los supuestos en los
que se apoya.

1. `python -m tps.pipeline.batch`: potencia por bandas de cada registro (`tps/pipeline/batch.py`)
2. `python -m tps.analysis.stats`: regiones y estadística SD vs NS (`tps/analysis/stats.py`)
3. `python -m tps.analysis.figures`: figuras, tabla de hipótesis y `results/report.md`

Toda la configuración sale de [`tps/config.py`](../tps/config.py). Los resultados son
**solo diferencias espectrales a nivel de grupo**; nada de esto sustenta afirmaciones clínicas.

## 1. Lote: potencia por bandas de cada registro

Cada cálculo reutiliza los módulos de `tps/core/`, de modo que un registro da los mismos
números aquí y en la GUI con la misma configuración:

| Paso | Función reutilizada |
|---|---|
| Inventario BIDS | `data.discover` |
| Carga y control de `.fdt` incompleto | `data.open_recording` |
| ICA confirmada (opcional) | `ica.apply_snapshot` |
| Filtro FIR sobre el canal continuo | `processing.ContinuousFilter` (vía `recording.filter.channel`) |
| Épocas de 5 s | `annotations.make_epochs` |
| Barrido pico a pico | `screening.scan_recording` |
| Welch por época | `spectral.welch_epoch` |
| Integración por bandas y potencia relativa | `spectral.band_powers` |

Para cada registro:

1. Se omiten y registran los registros cuyo `.set`/`.fdt` es un enlace git-annex sin
   contenido, o cuyo `.fdt` es más corto de lo que exige su encabezado.
2. Con `--ica-mode confirmed` se aplica la ICA confirmada del registro. Un registro sin una
   ICA confirmada válida se excluye (nunca se procesa sin ella en silencio).
3. Se filtra cada canal continuo completo con el filtrado elegido: por defecto el FIR de fase
   cero 0,2–45 Hz de `config.FILTER`; la GUI pasa el filtrado del explorador
   (`--bandpass` / `--no-bandpass` / `--notch`). El filtrado usado queda registrado en
   `config_used.json`, `ica_coverage.json` y `recordings_summary.csv`.
4. Se cortan épocas consecutivas de 5 s sin solapamiento desde la muestra 0; la última época
   incompleta se descarta.
5. Se barren **los 61 canales EEG** sobre la señal filtrada: un canal falla en una época si su
   amplitud pico a pico es > 150 µV (alta) o < 1 µV (plana).
6. **Canales malos:** un canal que falla en más del 30 % de las épocas completas se considera
   malo para ese registro: no genera filas de potencia y no rechaza épocas. Si no, un solo
   electrodo ruidoso (por ejemplo TP9 en sub-01 ses-1, que falla en 31/60 épocas) descartaría
   la mitad del registro para todos los canales.
7. **El rechazo de épocas es global** por defecto: una época se rechaza si falla en cualquiera
   de los canales restantes, así todos los canales de un registro usan las mismas épocas. Con
   `--rejection-mode per_channel` cada canal conserva sus propias épocas limpias y se excluye
   si le quedan menos de `MIN_CHANNEL_EPOCHS` (10).
8. Si quedan menos de 12 épocas limpias (60 s), se excluye el registro completo.
9. Welch por época (segmentos Hann de 2 s, 50 % de solapamiento, densidad), promediado sobre
   las épocas limpias; la potencia de cada banda es la integral de la PSD media sobre la banda.
   La potencia relativa se calcula respecto de 0,5–30 Hz.

La ejecución es **reanudable**: cada registro terminado se guarda en
`results/per_recording/<archivo>.json`, marcado con un hash de la configuración, el modo ICA,
el filtrado y el modo de rechazo. Al volver a ejecutar se saltean los registros guardados
salvo que cambie alguno de esos datos (o con `--force`). Los CSV finales se reconstruyen a
partir de todos los registros guardados, así que `--eyes open` seguido de `--eyes closed`
produce una sola tabla combinada.

Columnas de `features_long.csv`: `sub, ses, condition (NS/SD), eyes (open/closed), channel, band,
abs_power_uV2, rel_power_pct, n_epochs_used, n_epochs_rejected`. `exclusions.csv` lista cada
registro, canal malo, época y contraparte NS/SD faltante que se excluyó, con su motivo
(`level` = recording / channel / epoch / pair).

### Qué significa «aceptada»

La aceptación automática solo significa que la época quedó **dentro de los límites de
amplitud** en todos los canales que no son malos. **No** es inspección visual: los artefactos
oculares, musculares o de movimiento por debajo de 150 µV pico a pico pasan, y los umbrales de
150 µV / 1 µV son valores iniciales tomados de la GUI, **no validados** para este dataset. Los
parpadeos en los registros con ojos abiertos suelen superar 150 µV en los canales frontales
(Fp1, Fpz, AF3/AF4…), que terminan como «canales malos». Conviene revisar `exclusions.csv` antes
de interpretar resultados frontales. Sin pasabanda, las derivas lentas también aumentan la
amplitud pico a pico, así que se marcan más épocas.

## 2. Estadística: regiones y SD vs NS

### Diseño

* **Principal:** ojos abiertos (OA), todos los participantes con ambas sesiones.
  **Secundario:** ojos cerrados (OC), solo los participantes con ambas sesiones OC.
  OA y OC nunca se combinan.
* **Pareado, intrasujeto:** diferencia = SD − NS del mismo participante. Los participantes a
  los que les falta una sesión (por ejemplo, un registro truncado) quedan fuera de ese análisis
  y se listan en `stats_participants.csv`.
* **Se necesitan al menos 3 pares completos** para una prueba o un cluster; con menos, la fila
  se conserva con la estadística vacía para que el hueco sea visible, y las figuras no tienen
  nada que mostrar.
* **La cantidad de épocas no se iguala** entre sesiones; se informa en `stats_epochs.csv`
  (en OA, SD tiene menos épocas limpias que NS).

### Variables por región (`tps/analysis/regions.py`)

Los 61 nombres de canal salen del `channels.tsv` de BIDS (idéntico en los 218 registros) y se
agrupan según su prefijo 10-10: frontal (Fp/AF/F, 16), central (FC/C/CP, 20),
temporal (FT/T/TP, 8), parietal (P, 9) y occipital (PO/O, 8).

`features_region.csv` contiene, por sujeto, sesión, ojos, banda y región, la **media del log10
de la potencia absoluta** (log de la media geométrica) y la **media de la potencia relativa**
sobre los canales de la región, con la cantidad de canales usados. El valor de una región solo
se calcula si sobrevivió al barrido al menos el **50 % de sus canales**, porque los canales malos
(sobre todo frontales, por parpadeos) dejarían regiones representadas por pocos electrodos.

### Pruebas y correcciones

* **Prueba principal:** Wilcoxon de rangos con signo sobre SD−NS del log10 de la potencia.
  **Sensibilidad:** prueba t pareada, y las mismas pruebas restringidas a registros con
  ≥ 20 épocas limpias.
* **Tamaños del efecto:** correlación biserial de rangos r = (W+ − W−)/(W+ + W−);
  d_z de Cohen = media(d)/de(d). El IC 95 % es el intervalo basado en t de la diferencia
  pareada media (en unidades log10: 0,1 ≈ +26 % de potencia).
* **Comparaciones múltiples:** BH-FDR sobre las 20 pruebas banda × región, por separado para
  OA y OC; se informan los p corregidos y sin corregir. La prueba de theta/beta frontal es una
  única prueba preespecificada por condición de ojos (sin corrección FDR).
* **Orden de sesión:** diferencia SD−NS en participantes NS→SD vs SD→NS (U de Mann-Whitney, FDR).
* **Prueba de clusters:** `mne.stats.permutation_cluster_1samp_test` sobre SD−NS del log10 de la
  potencia por electrodo, a dos colas, umbral de formación de clusters t(n−1) para p < 0,05,
  5000 permutaciones por cambio de signo (semilla 0), adyacencia a partir de las posiciones del
  `electrodes.tsv` del propio dataset.

## 3. Figuras e informe

Los topomapas (absolutos y relativos), las PSD por región, los gráficos pareados y theta/beta
frontal se dibujan a partir de las tablas del paso 2; la tabla de hipótesis compara H1–H4
([hypotheses.md](hypotheses.md)) con los efectos observados. `results/report.md` reúne todas
las tablas, hipótesis y figuras con sus leyendas.

## Supuestos y limitaciones

* El barrido es solo por amplitud, no inspección visual (ver arriba).
* Los canales malos difieren entre registros. En la prueba de clusters, un electrodo faltante se
  toma como diferencia 0 para ese sujeto, lo cual es conservador (tira hacia la ausencia de
  efecto); la cantidad está en `n_imputed_zero`. Las regiones frontal y occipital tienen menos
  participantes que la central.
* Los registros SD tienen menos épocas limpias que los NS. La PSD promedio sigue sin sesgo, pero
  es más ruidosa en SD; por diseño, esto no se igualó.
* El orden de sesión (contrabalanceado) se controla con una prueba entre grupos, que tiene poca
  potencia con ~30 participantes por grupo de orden; un resultado no significativo no prueba que
  no haya efecto de orden.
* OC es un subgrupo (~36 pares completos) y es secundario; sus resultados no son independientes
  de los de OA (son los mismos participantes).

## Pruebas

```bash
python -m unittest discover -s tests -p "test_pipeline_batch.py" -v
python -m unittest discover -s tests -p "test_analysis_*.py" -v
```

Las pruebas del lote usan solo señales sintéticas: un seno de 10 Hz debe poner > 90 % de su
potencia en alfa, una mezcla de 6 Hz + 20 Hz da el cociente theta/beta esperado, los picos
rechazan exactamente su época, un canal ruidoso persistente pasa a canal malo, pocas épocas
limpias excluyen el registro, el filtrado elegido cambia la señal, los marcadores de git-annex
se registran y una nueva ejecución saltea los registros terminados.
