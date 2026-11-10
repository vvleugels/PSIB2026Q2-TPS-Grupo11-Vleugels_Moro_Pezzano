# Hipótesis: EEG en reposo tras privación del sueño (SD) vs sueño normal (NS)

Hipótesis preespecificadas y verificables. La versión que lee el programa (usada por las
figuras y la tabla resumen) es `tps/analysis/hypotheses.py`; hay que mantener ambas sincronizadas.

* **Contraste:** SD − NS pareado dentro de cada participante.
* **Medida:** media del log10 de la potencia absoluta por banda y región (`results/features_region.csv`).
* **Análisis:** ojos abiertos (OA) es el principal y ojos cerrados (OC) el secundario. OA y OC
  nunca se combinan.
* **Regla de decisión:** una celda de hipótesis queda *apoyada* (`supported`) cuando el cambio
  tiene el signo esperado y el p de la prueba t pareada es < 0,05. Un efecto significativo en el
  sentido contrario se informa como *opuesto* (`opposite`).
* **Topografía:** «mayor sobre la región X» se verifica mirando qué región tiene el mayor
  |cambio %| en esa banda (`largest_region`).
* **Alcance:** son solo diferencias espectrales a nivel de grupo; no se pretende ninguna
  interpretación clínica.

| ID | Banda | Regiones (mayor efecto esperado) | Dirección esperada | Ojos | Carácter |
|---|---|---|---|---|---|
| H1 | delta, theta | frontal, central (incl. fronto-central FC*) | aumento (SD > NS) | OA y OC | confirmatoria |
| H2 | alfa | parietal, occipital | disminución (SD < NS) | más clara en OC que en OA | confirmatoria, literatura dividida |
| H3 | cociente theta/beta frontal | frontal | aumento | OA y OC | confirmatoria (una prueba por condición de ojos) |
| H4 | beta | todas las regiones | sin predicción | OA y OC | exploratoria |

## H1: actividad de ondas lentas

La potencia delta (0,5–4 Hz) y theta (4–8 Hz) aumenta tras SD, con el mayor aumento sobre las
regiones frontal / fronto-central.

* Referencias: TODO (autores, año, DOI), a verificar antes de completarlas.

## H2: alfa

La potencia alfa (8–13 Hz) disminuye tras SD, con la mayor disminución sobre las regiones
posteriores (parietal, occipital). Se espera un efecto más claro con ojos cerrados, donde
predomina el alfa en reposo. **La literatura está dividida:** se informaron tanto disminuciones
como aumentos del alfa en reposo tras la pérdida de sueño, así que un resultado nulo u opuesto es
plausible y se informa tal como se observa.

* Referencias: TODO (autores, año, DOI), a verificar antes de completarlas.

## H3: cociente theta/beta frontal

El cociente entre la potencia theta y beta sobre los canales frontales aumenta tras SD. Se calcula
como la media del log10(theta/beta) sobre los canales frontales y se prueba una vez por condición
de ojos.

* Referencias: TODO (autores, año, DOI), a verificar antes de completarlas.

## H4: beta (exploratoria)

La potencia beta (13–30 Hz) se examina en todas las regiones sin predicción de dirección. Los
resultados se interpretan como exploratorios.

* Referencias: TODO (autores, año, DOI), a verificar antes de completarlas.
