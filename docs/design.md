# Diseño acordado para la próxima implementación

> **Nota histórica.** Este es el documento de diseño original. Las rutas y comandos que
> menciona (`eeg_viewer/`, `scripts/analyze_group.py`, `group_ui.py`, etc.) corresponden a
> la estructura anterior. El análisis de grupo de 20 participantes se unificó con el lote
> (`python -m tps.pipeline.batch --subjects-file subsets/group20_seed2026.txt
> --rejection-mode per_channel`) y `tps.analysis`; ver el README y [batch.md](batch.md).

Este documento describe el diseño acordado para la próxima implementación del
procesamiento y la revisión de EEG; **no describe el comportamiento actual**.
Las decisiones siguientes fijan la dirección del desarrollo. Los detalles de
implementación y validación que aún no se acordaron se enumeran por separado.
La creación inicial de este documento fue exclusivamente documental. El estado
del piloto posterior se detalla a continuación y en [validation.md](validation.md).

## Estado de implementación

Se implementó la primera etapa en un motor y visor de validación independientes
(`validate_eeg.py`), con epochs de 4 segundos, ICA clásica, revisión por canal,
reparaciones confirmadas y trazabilidad. Sus parámetros son experimentales.
La aplicación principal conserva el flujo anterior hasta completar la validación
y obtener aprobación explícita del perfil. No hay aceptación automática definitiva.

La muestra inicial está verificada, pero faltan etiquetas humanas y geometría
documentada: los `.set` contienen posiciones NaN y los TSV no incluyen un archivo
de sistema de coordenadas. La validación de mapas/interpolación permanece pendiente.

## Análisis grupal implementado

Como flujo aparte de ese piloto, se agregó `eeg_viewer/group_analysis.py` y el
comando `scripts/analyze_group.py`. Procesa las sesiones `ses-1` y `ses-2` para
ojos abiertos y, para `sub-01` a `sub-38`, también ojos cerrados. La selección de
participantes se lee de `participants.tsv` y cada fila incorpora `SessionOrder`.

Cada registro usa el FIR continuo de 0,2–45 Hz de `processing.py`. Las epochs de
5 segundos se filtran automáticamente con los límites de `screening.py` (pico a
pico alto >150 µV y plano <1 µV); las parciales se excluyen. El modo predeterminado
`--rejection-mode epoch` rechaza una epoch para todos los canales si cualquiera la
marca. La alternativa `--rejection-mode channel` construye la máscara de limpieza
por separado para cada canal: una marca en otro electrodo no elimina esa epoch para
el canal actual. `--high-uv` permite cambiar el límite alto y `--flat-uv` el límite
plano. En ambos modos se excluye del registro cada canal con menos de 10 epochs
limpias (`--min-channel-epochs` permite ajustar el mínimo), con participante,
sesión, canal y motivo en `exclusions.csv`.

Para las epochs retenidas se calcula Welch y potencia absoluta/relativa con
`SpectralConfig` y `band_powers` de `spectral.py`. `features.csv` conserva el número
de epochs usadas en `epochs_used` en cada fila identificada por participante,
sesión, condición, canal y banda, así como `rejection_mode` y `high_threshold_uv`.
En modo canal también incluye `epochs_used_global`, el número que habría quedado
con el rechazo global, calculado en el mismo screening. `--participants-file` lee
un ID por línea e ignora comentarios que comienzan con `#`;
`--checkpoint-each-record` muestra progreso y escribe las tablas al terminar cada
registro.
`--eyes open` o `--eyes closed` limita la condición ocular, y `--channels Fp1 Cz Oz`
restringe el análisis a canales seleccionados. No se usan anotaciones manuales. Las
salidas son `features.csv`, `record_summary.csv`, `exclusions.csv` y `analysis.json`.

Ejemplo acotado para comparar modos con tres participantes y ojos abiertos:

```powershell
python scripts/analyze_group.py --participants sub-12 sub-13 sub-14 --eyes open --channels Fp1 Cz Oz --rejection-mode epoch --output results/three-epoch
python scripts/analyze_group.py --participants sub-12 sub-13 sub-14 --eyes open --channels Fp1 Cz Oz --rejection-mode channel --output results/three-channel
```

## Análisis regional desde `features.csv`

`scripts/analyze_regions.py` consume solo el CSV grupal ya generado. Aplica `log10`
a la potencia absoluta de cada canal y promedia esos logaritmos por participante,
sesión, condición ocular, banda y región. La asignación de canales es frontal `F`
(sin `Fp` ni `AF`), central `FC/C`, parietal `CP/P`, occipital `PO/O` y temporal
`FT/T/TP`. Los canales `Fp` y `AF` se excluyen de las medias regionales, pero se
conservan en los mapas. El mapa usa el montaje estándar 10-20 de MNE; dos etiquetas
que solo difieren en mayúsculas (por ejemplo `Cpz`/`CPz`) se promedian en una misma
posición para el mapa.

La tabla `analysis/stats_table.csv` compara NS (`ses-1`) y SD (`ses-2`) con Wilcoxon
pareado sobre las features regionales de log-potencia. Informa los pares completos,
la mediana de SD−NS y el tamaño de efecto rank-biserial pareado. Benjamini–Hochberg
se aplica por separado a las 20 pruebas región-banda de cada condición ocular.
Como chequeo exploratorio del orden, compara las diferencias pareadas de los grupos
`NS->SD` y `SD->NS` con Mann–Whitney y corrige esos 20 valores p por condición ocular.
La salida incluye `regional_features.csv`, `channel_region_map.csv`, `stats_table.csv`,
y los ocho `pairs_{open|closed}_{delta|theta|alpha|beta}.png` y ocho topomaps SD−NS.

```bash
python scripts/analyze_regions.py
```

Las figuras absolutas se regeneran con `python scripts/regenerate_group_figures.py`.
El mapa usa esfera MNE de 11 cm y extrapolación hasta el contorno para contener los
electrodos. El mismo comando crea gráficos de pares y topomaps de potencia relativa
con el prefijo `relative_`, y `group_summary.csv` para la interfaz.
`eeg_viewer/group_ui.py` es la pestaña de solo lectura “Resultados de grupo”: ofrece
selectores de ojos, banda, tipo de figura y potencia; una barra Matplotlib permite
zoom; y una tabla muestra pares, mediana, efecto y q. Lee solo archivos de `analysis/`
y presenta un mensaje con la lista si falta alguno.

`python scripts/compare_relative_epochs.py` amplía la comparación usando solo
`features.csv`, sin volver a procesar señales. Repite Wilcoxon/FDR con la potencia
relativa de `relative_percent` (o la calcula como potencia de banda dividida por la
suma de las cuatro bandas si esa columna falta), genera una comparación lado a lado
con la estadística absoluta, resume epochs globales y por canal por sesión/condición,
y repite la estadística absoluta restringiendo cada canal pareado a ≥20 epochs en
ambas sesiones. Guarda `stats_relative_vs_absolute.csv`,
`epochs_and_exclusions_by_session_condition.csv` y `stats_absolute_min20.csv` en
`analysis/`, con nombres nuevos. Como el archivo de features no lista exclusiones,
el número de canales excluidos se infiere de los canales ausentes respecto del
conjunto observado en el propio CSV.

Se agregó una prueba con una senoide de 10 Hz que comprueba que la banda alfa tiene
la mayor potencia. También se corrió el pipeline con tres participantes. En la
copia local auditada hay 71 participantes en `participants.tsv`, 61 carpetas y
175 de 218 archivos `.set` esperados; además se detectaron tres `.fdt` truncados.
El pipeline conserva estos casos como exclusiones y deja intacto `data/`.

En la suite final corrieron 75 pruebas: 74 pasaron y una no pudo crear un symlink
por falta de privilegio en Windows (`WinError 1314`). El test de caracteres
acentuados pasa al ejecutar con UTF-8 explícito. El análisis regional descrito
arriba ya incorpora la estadística pareada, los mapas y el resumen por regiones;
la comparación entre filtros sigue pendiente.

## Design decisions

1. **Métodos permitidos.** Se utilizarán ICA clásica y reglas estadísticas
   explicables. Se excluyen ICLabel, AutoReject, redes neuronales y clasificadores
   entrenados. **Motivo:** permitir que las propuestas de limpieza se inspeccionen
   y justifiquen mediante criterios explícitos, sin depender de clasificadores
   entrenados.

2. **Reemplazar el flujo actual.** Se pasará de epochs de 5 segundos a epochs de
   **4 segundos**. Se respaldarán las anotaciones anteriores y se comenzará una
   revisión nueva, sin trasladar automáticamente etiquetas entre intervalos
   diferentes. **Motivo:** adoptar la duración acordada y evitar atribuir a un
   intervalo nuevo una evaluación realizada sobre otras muestras, conservando
   el historial anterior.

3. **Filtrado final.** Se conservará el pasabanda de **0,2–45 Hz** para la señal
   de análisis y los espectros. **Motivo:** mantener el rango de interés acordado,
   incluida la actividad lenta, y la coherencia entre señal y análisis espectral.

4. **Entrenamiento de ICA — adaptación respecto del paper.** Se entrenará ICA
   sobre una copia auxiliar filtrada a **1–45 Hz** y se aplicará la descomposición
   a la señal de **0,2–45 Hz**, manteniendo canales y referencia compatibles
   entre ambas señales. Esta separación entre la señal de entrenamiento y la
   señal de análisis es una adaptación acordada respecto del procedimiento
   descrito en el paper, no una reproducción literal de este.
   **Motivo:** reducir la influencia de las derivas lentas en el ajuste de ICA
   sin imponer el corte de 1 Hz a la señal destinada al análisis. Su efecto
   sobre la conservación de actividad fisiológica deberá validarse.

5. **Selección de componentes.** Se mostrarán mapas, señales, espectros e
   indicadores estadísticos de los componentes. Las reglas sugerirán candidatos;
   el usuario confirmará cuáles excluir. **Motivo:** apoyar una decisión
   informada con evidencia visible y evitar que una sugerencia estadística
   implique por sí sola la eliminación de actividad.

6. **Calidad por canal y epoch.** Se distinguirá un electrodo problemático de
   una epoch globalmente inutilizable. Un aviso en TP9 no invalidará
   automáticamente el análisis de Cz. **Motivo:** conservar los datos útiles
   de canales no afectados y distinguir problemas locales de contaminación
   que comprometa toda la epoch.

7. **Reparación.** Se permitirá interpolar electrodos malos del registro y
   reparar epochs con pocos canales afectados. Se mostrarán las reparaciones
   propuestas para confirmarlas conjuntamente. **Motivo:** recuperar datos
   aprovechables cuando la afectación sea limitada y mantener la supervisión
   del usuario sobre las modificaciones. La cantidad de canales admisible
   para reparar una epoch queda pendiente de definición y validación.

8. **Referencia promedio.** Se aplicará al final de la limpieza, después de
   resolver los canales afectados. **Motivo:** evitar que los canales
   problemáticos contaminen la referencia compartida por los demás canales.

9. **Aceptación automática.** Se conservará por canal cuando se cumplan los
   criterios definitivos, distinguiendo las decisiones automáticas de las
   manuales. **Motivo:** agilizar la revisión de datos que cumplan criterios
   validados sin perder la procedencia de cada decisión. Los criterios y sus
   umbrales todavía no están definidos de forma definitiva.

10. **Trazabilidad.** Los datos originales se mantendrán intactos. Se registrarán
    los filtros, los canales reparados, los componentes excluidos, los criterios
    utilizados y las versiones. **Motivo:** permitir auditar y reproducir la
    limpieza, relacionar sus resultados con el origen y revisar las decisiones
    sin modificar los datos fuente.

11. **Validación.** Los porcentajes del paper se utilizarán como referencia,
    sin ajustar parámetros para forzar esos porcentajes. **Motivo:** evaluar
    la conservación de actividad fisiológica y la detección de artefactos en
    los registros disponibles, sin convertir un resultado de otro estudio en
    una cuota de aceptación, reparación o exclusión.

## Detalles pendientes

Los siguientes puntos quedan explícitamente **pendientes de definición y
validación**; las decisiones anteriores no fijan sus valores ni sus soluciones:

- Algoritmo y parámetros de ICA.
- Fórmulas y umbrales de las reglas estadísticas.
- Cuántos canales afectados permiten reparar una epoch.
- Formato de almacenamiento y respaldo de las nuevas revisiones.
- Muestra de registros para validar la conservación de actividad fisiológica
  y la detección de artefactos.

## Verificación del documento

- Las once decisiones acordadas incluyen su motivo y se expresan como diseño
  para la próxima implementación, no como funcionalidades actuales.
- La copia auxiliar de 1–45 Hz para entrenar ICA y su aplicación a la señal de
  0,2–45 Hz se identifican explícitamente como una adaptación respecto del paper.
- Los parámetros, umbrales, límites de reparación, almacenamiento y muestra de
  validación permanecen separados como pendientes, sin asumir valores definitivos.
- La creación inicial fue exclusivamente documental. El piloto posterior tiene
  pruebas propias y se mantiene separado de la aplicación principal hasta validar
  y aprobar el perfil.
