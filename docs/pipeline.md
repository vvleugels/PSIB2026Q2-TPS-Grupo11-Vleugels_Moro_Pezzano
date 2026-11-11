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
   ICA confirmada válida se excluye (nunca se procesa sin ella en silencio). Con
   `--ica-mode available` se aplica donde existe y el resto se procesa solo con filtros; el modo
   de cada registro queda en `recordings_summary.csv` (`processing_mode`).
   Las épocas marcadas «Con artefactos» en el explorador se descartan siempre.
3. Se filtra cada canal continuo completo con el filtrado elegido: por defecto el FIR de fase
   cero 0,2–45 Hz de `config.FILTER`; la GUI pasa el filtrado del explorador
   (`--bandpass` / `--no-bandpass` / `--notch`). El filtrado usado queda registrado en
   `config_used.json`, `ica_coverage.json` y `recordings_summary.csv`.
4. Se cortan épocas consecutivas de 5 s sin solapamiento desde la muestra 0; la última época
   incompleta se descarta.
5. Se barren **los 61 canales EEG** sobre la señal filtrada: un canal falla en una época si su
   amplitud pico a pico es > 150 µV (alta) o < 1 µV (plana).
6. **Canales malos:** un canal es malo para ese registro si cumple cualquiera de estos criterios
   (columna `criterion` de `exclusions.csv`):
   - `manual`: marcado a mano en el explorador (anotaciones del estudio);
   - `flat`: varía menos de 1 µV durante al menos 5 s;
   - `deviation`: z robusto > 5 de su desvío robusto (IQR / 1,349) frente a los demás canales;
   - `correlation`: el percentil 98 de su correlación absoluta con los demás canales es < 0,4
     en más del 20 % de las ventanas de 1 s (PREP usa 1 % porque luego interpola; EEGLAB
     clean_rawdata, 50 %);
   - `high_frequency`: z robusto > 5 del log10 de la potencia sobre 45 Hz respecto de 1–45 Hz,
     medido sin pasabanda para que el ruido de línea y el EMG sigan visibles;
   - `peak_to_peak`: falla el barrido en más del 30 % de las épocas completas.

   Los criterios automáticos (`tps/core/badchannels.py`, umbrales en `config.BAD_CHANNELS`)
   reimplementan las ideas de EEGLAB (clean_rawdata, pop_rejchan) y PREP; **no están validados**
   para este dataset. Un canal malo no genera filas de potencia, no rechaza épocas y **no se
   interpola**. Si no se excluyera, un solo electrodo ruidoso (por ejemplo TP9 en sub-01 ses-1,
   que falla en 31/60 épocas) descartaría la mitad del registro para todos los canales. Las marcas
   manuales se leen al iniciar el lote; cambiarlas invalida solo la caché de ese registro.
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

## 2. Estadística: resumen SD vs NS

### Diseño

* **Principal:** ojos abiertos (OA), todos los participantes con ambas sesiones, como en el paper
  de referencia (Xiang et al., 2024). **Secundario** («Avanzado»): ojos cerrados (OC), solo los
  participantes con ambas sesiones OC. OA y OC nunca se combinan.
* **Pareado, intrasujeto:** diferencia = SD − NS del mismo participante. Los participantes a los
  que les falta una sesión (por ejemplo, un registro truncado) quedan fuera de ese análisis.
* **Se necesitan al menos 3 pares completos** para calcular el p; con menos, la fila conserva los
  valores descriptivos y el p queda vacío.
* **La cantidad de épocas no se iguala** entre sesiones (`recordings_summary.csv` la informa).

### Variables por región (`tps/analysis/regions.py`)

Los 61 nombres de canal salen del `channels.tsv` de BIDS (idéntico en los 218 registros) y se
agrupan según su prefijo 10-10: frontal (Fp/AF/F, 16), central (FC/C/CP, 20),
temporal (FT/T/TP, 8), parietal (P, 9) y occipital (PO/O, 8).

`features_region.csv` contiene, por sujeto, sesión, ojos, banda y región, la **media del log10
de la potencia absoluta** (log de la media geométrica) sobre los canales de la región, con la
cantidad de canales usados. El valor de una región solo se calcula si sobrevivió al barrido al
menos el **50 % de sus canales**, porque los canales malos (sobre todo frontales, por parpadeos)
dejarían regiones representadas por pocos electrodos.

### Tabla resumen (`stats_summary.csv`)

Una fila por ojos × banda × región (20 por condición de ojos) más una fila de theta/beta frontal
(H3):

| Columna | Qué es |
|---|---|
| `n` | participantes con ambas sesiones |
| `mean_ns_db`, `mean_sd_db` | 10 · media del log10 de la potencia (dB re 1 µV²), la unidad del paper |
| `change_db` | 10 · media de (log10 SD − log10 NS) |
| `change_pct` | (10^media de la diferencia − 1) · 100: cambio de la potencia (media geométrica) |
| `ci95_low_pct`, `ci95_high_pct` | IC 95 % del cambio en %, desde el IC basado en t de la diferencia log10 |
| `p` | prueba t pareada (a dos colas) sobre los valores log10 |
| `significant` | p < 0,05 |

No se corrige por comparaciones múltiples: con 20 pruebas por condición de ojos se espera, por
azar, alrededor de un resultado «significativo», y así se aclara en el informe.

## 3. Figuras e informe

Por condición de ojos, con los participantes que tienen ambas sesiones:

* `psd_cz_<ojos>.png`: espectro de potencia en Cz en dB para NS y SD, media ± 1 desvío estándar
  entre participantes (como la Fig. 4 del paper), a partir de `psd_channels.csv`.
* `topomaps_<ojos>.png`: potencia por banda en cada electrodo, en dB, NS arriba y SD abajo, con una
  escala de color por banda compartida por NS y SD (como la Fig. 5 del paper; se agrega delta por
  H1). La zona central de menor potencia refleja la referencia FCz: el paper re-referencia al
  promedio y acá se conserva la referencia original.

La tabla de hipótesis compara H1–H4 ([hypotheses.md](hypotheses.md)) con los cambios observados.
`results/report.md` reúne los datos usados, el resumen de ojos abiertos, las hipótesis, las
figuras y, como «Avanzado», ojos cerrados.

## Supuestos y limitaciones

* El barrido es solo por amplitud, no inspección visual (ver arriba).
* Los canales malos difieren entre registros: las regiones frontal y occipital tienen menos
  participantes que la central.
* Los registros SD tienen menos épocas limpias que los NS. La PSD promedio sigue sin sesgo, pero
  es más ruidosa en SD; por diseño, esto no se igualó.
* Las sesiones están contrabalanceadas, pero el efecto del orden no se analiza.
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
