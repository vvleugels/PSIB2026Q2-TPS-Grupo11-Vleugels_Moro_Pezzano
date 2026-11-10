> La aplicación principal ahora ofrece ICA con confirmación manual y épocas de 5 s
> (ver README). Este documento describe el piloto independiente de 4 s y su protocolo
> de validación; sus revisiones y aprobación automática no se transfieren a la GUI.
> Ambos flujos comparten el motor de ajuste Infomax y sus controles de convergencia.

# Validación de la limpieza EEG

## Estado y alcance

La primera etapa está implementada como motor y visor **experimentales separados**
de `gui.py`. La aplicación habitual conserva su procesamiento actual y sus épocas
de 5 segundos. Integrarlo a la aplicación (según la nota de diseño original, hoy en el historial de git) requiere primero
revisión humana de calibración, congelamiento del perfil y evaluación independiente.
No hay aceptación automática habilitada ni un perfil aprobado para uso definitivo.

El piloto implementa épocas de 4 segundos; copias continuas de 0,2–45 Hz y 1–45 Hz;
ICA Infomax extendida; reglas explicables; confirmación de componentes y reparaciones;
referencia promedio fija por registro y revisiones versionadas. No utiliza
ICLabel, AutoReject, redes neuronales ni clasificadores entrenados.

## Comenzar

Desde la raíz del repositorio, con las dependencias de `requirements.txt`:

```bash
.venv/bin/python validate_eeg.py sample
.venv/bin/python validate_eeg.py new --record sub-18_ses-1_task-eyesopen_eeg
.venv/bin/python validate_eeg.py review RUTA_DE_REVISION
```

`sample` crea `annotations/validation/sample.json` y no lo sobrescribe si ya existe.
En ese caso, continuar con `new`. `new` imprime la ruta que debe pasarse a `review`.
Cada ejecución de `new` crea otra revisión; para continuar una existente, usar su
ruta con `review`. Todas las rutas por defecto se resuelven respecto del repositorio.
El visor necesita una sesión gráfica Tk. Los comandos `sample`, `new`, `freeze`
y `report` pueden ejecutarse sin pantalla.

También se puede agregar `--open` a `new` para crear y abrir la revisión en un paso.

La muestra prevista es:

| Grupo | Participantes | Registros |
| --- | --- | --- |
| Calibración | sub-18 (abiertos/cerrados), sub-54 (abiertos) | 6, ambas sesiones |
| Evaluación independiente | sub-02 (abiertos/cerrados), sub-70 (abiertos) | 6, ambas sesiones |

Se comprueban integridad, muestras finitas, frecuencia, canales, referencia y
geometría. Si un participante no es accesible se reemplaza dentro de su estrato
mediante el orden SHA-256 de `eeg-validation-v1:<participante>`. El manifiesto
registra la sustitución y su causa. Las posiciones ausentes son una limitación
explícita; no impiden la revisión de señales.

## Recorrido de revisión

1. **Revisión ciega:** observar original y filtrada, marcar cada canal/época como
   fisiológico o con artefacto y agregar notas. El botón de época fisiológica marca
   todos sus canales mediante una acción manual. La ventana muestra 12 segundos,
   la cuadrícula 4 segundos y el espectro la época seleccionada. Las parciales
   son visibles, pero no entran en ICA, estadísticas de validación ni espectros.
2. **Cerrar revisión ciega:** requiere etiquetas para todos los canales de todas
   las épocas completas. Bloquea estas etiquetas y revela las sugerencias. No
   confundir cerrar esta revisión con aprobar el perfil.
3. **Calidad inicial:** confirmar nombres de electrodos persistentemente malos y
   números de épocas globalmente inutilizables. Las listas no se rellenan con
   exclusiones automáticas. Ajustar el perfil solo en calibración.
4. **ICA:** confirmar esas decisiones y ajustar. Se excluyen del ajuste las
   épocas globales, parciales y con problemas no resueltos en los canales usados.
   Una etiqueta manual fisiológica permite revisar una sugerencia estadística;
   una etiqueta de artefacto impide usar esa época en el ajuste. Se muestran mapa,
   señal, espectro, curtosis, potencia rápida y correlaciones frontales por componente.
   La ausencia de mapa se explica si falta geometría. Correlación frontal no es EOG.
5. **Reparaciones:** escribir los componentes a excluir, solicitar propuestas y
   elegir los números de las épocas a reparar. La confirmación conjunta aplica
   únicamente esas selecciones. Se cuentan los canales persistentemente interpolados
   dentro del límite de reparación local. Una reparación no equivale a aceptación.
6. **Revisión final:** comparar filtrada y limpia con la misma referencia promedio,
   revisar nuevamente y guardar etiquetas finales. La referencia usa un conjunto
   fijo de canales sin problemas residuales en todas las épocas utilizables. Un
   canal localmente inutilizable queda fuera de esa referencia para todo el registro;
   no rechaza automáticamente los otros canales ni toda la época.
7. **Informe:** exportar evaluaciones y el informe desde el visor o la CLI. Los
   destinos de exportación deben ser nuevos para no sobrescribir originales ni
   revisiones. Los resultados siguen siendo experimentales.

Las listas de épocas/componentes de la interfaz empiezan en **1**; las estructuras
internas y los JSON usan índices desde **0**. Los canales se identifican por nombre.
Los cálculos corren fuera del hilo de Tk. Cancelar un ajuste o limpieza solicita
descartar el cálculo antes de publicar el resultado; no interrumpe internamente
una iteración de MNE. Cerrar espera ese punto seguro.

## Perfil y criterios experimentales

- ICA: Infomax extendida, semilla 42, máximo 1000 iteraciones, sin submuestreo ni
  corrección de línea base. Se usa el rango de los datos centrados, contando valores
  singulares mayores que `máximo × 10⁻⁶`. Rango menor a 2 o no convergencia bloquean
  el ajuste. Se conserva la referencia de adquisición entre entrenamiento y aplicación.
  Se registran los pasos reales y el criterio de parada de Infomax: cambio cuadrático
  de pesos <10⁻¹² o parada por más de 20 pasos de ángulo pequeño, según MNE. No se usa solo
  `n_iter_` para decidir convergencia: MNE puede devolver 1000 incluso cuando alcanzó
  el criterio de cambio de pesos antes de ese límite. Si falta evidencia de parada,
  se bloquea la confirmación.
- FIR continuo: los mismos parámetros Hamming, fase cero, `firwin`, transiciones
  y longitud automáticas y padding `reflect_limited` del filtro actual. Se valida
  su longitud antes de ejecutarlo; no se concatenan intervalos antes de filtrar.
- Indicadores por canal/época: pico a pico en µV, varianza, muestras no finitas,
  potencia de 30–45 Hz dividida por potencia de 1–45 Hz. Welch utiliza segmentos
  de 2 segundos y solapamiento de 50 %.
- Desviación robusta: `0,67448975 × (valor − mediana) / MAD`, por canal sobre épocas
  completas. Se aplica al logaritmo natural de la varianza y a la proporción de
  potencia rápida. MAD cero o menos de tres observaciones finitas dejan el indicador
  no evaluable; eso nunca cuenta como aceptación automática segura.
- Perfil inicial: alta `>150 µV`, plana `<1 µV`, `|z_varianza| >5`, `z_potencia >5`;
  persistencia `>20 %`; reparación de hasta 2 canales afectados. El informe compara
  alta 100/150/200, plana 0,5/1, z 4/5/6 y límites de reparación 1/2/3, sin elegir
  automáticamente el ganador.
- Componentes sugeridos: exceso de curtosis absoluto >5, proporción rápida >0,5,
  o correlación absoluta con Fp1/Fp2 >0,5. Solo la selección manual excluye componentes.
- Interpolación: splines esféricos con origen `(0,0,0)` en coordenadas de cabeza,
  posiciones válidas y al menos cuatro donantes. Ambas copias usan la misma
  geometría y selección de canales. No se inventan posiciones ni se infieren unidades.

Los parámetros no están validados. El informe compara avisos y posibles aceptaciones
con tus etiquetas; separa indicadores no evaluables y pendientes. Los cambios de PSD
y potencia por bandas se calculan en épocas originalmente etiquetadas como fisiológicas,
aplicando a ambas señales exactamente la misma referencia final. Incluye pruebas de
interpolación ocultando Cz, Pz y Oz cuando hay geometría y épocas con todos los donantes
etiquetados como sanos. La muestra pequeña y un único revisor limitan las conclusiones.

## Congelar y evaluar

Después de revisar las seis calibraciones, aplicar el mismo perfil a todas, volver
a procesar y completar sus etiquetas finales. Congelar usando sus seis directorios:

```bash
.venv/bin/python validate_eeg.py freeze --revisions R1 R2 R3 R4 R5 R6 --output annotations/validation/perfil-congelado.json
.venv/bin/python validate_eeg.py new --record sub-02_ses-1_task-eyesopen_eeg --frozen annotations/validation/perfil-congelado.json
.venv/bin/python validate_eeg.py report RUTA_DE_REVISION --output annotations/validation/informe-nuevo.json
```

El congelamiento exige revisiones ciegas y finales completas, limpieza confirmada,
origen y artefactos intactos, muestra coincidente y perfil común. El perfil congelado
se copia dentro de cada revisión de evaluación y no se puede editar allí. El informe
de evaluación no incluye barrido de perfiles alternativos. Si la evidencia es
insuficiente, se necesita otra validación; no se reajusta el perfil sobre esta muestra
y luego se la presenta como independiente.

**Congelado no significa aprobado.** La integración en la aplicación principal y
la aceptación automática definitiva esperan tu aprobación explícita de los resultados.

## Almacenamiento y recuperación

Cada revisión se guarda en `annotations/validation/<registro>/<id>/revision.json`.
Antes de crearla, se respalda byte por byte el JSON anterior de 5 segundos, si existe,
y se verifica su contenido. Las etiquetas anteriores no se importan.

El manifiesto guarda hashes SHA-256 del origen y del código del motor, geometría
disponible, parámetros, versiones, decisiones manuales, historial y referencias a artefactos inmutables:
ICA FIF, señal filtrada, señal limpia y metadatos de ajuste/reparación. El manifiesto
se publica atómicamente después de guardar los artefactos. Un fallo conserva la
última revisión válida; puede dejar un artefacto huérfano que no se usa. Cambios de
procesamiento archivan decisiones/resultados previos e invalidan las etiquetas finales.
Si cambia el código del motor o el entorno, se requiere restaurar las versiones
guardadas o crear otra revisión; no se recomputan silenciosamente resultados bajo
las etiquetas anteriores. El congelamiento conserva copias de las revisiones y
sus informes, además de sus hashes.

Los datos y las revisiones están excluidos de Git. Respaldar la carpeta `annotations`
completa, incluidos manifiesto de muestra, perfiles congelados y artefactos FIF.

## Hallazgos de la verificación inicial del dataset

Los 12 registros previstos fueron accesibles: 61 canales EEG, 500 Hz, 300 segundos,
referencia FCz y 75 épocas completas por registro. No se necesitaron sustituciones.
Sin embargo, las posiciones en los `.set` son NaN para todos los canales. Existen
archivos `electrodes.tsv`, pero no `coordsystem.json` que documente sus unidades y
sistema de coordenadas. Un objeto de montaje de MNE con 61 nombres **no garantiza**
que sus posiciones sean válidas.

En estos registros se pueden revisar señales y ajustar ICA, pero los mapas y la
interpolación quedan limitados hasta disponer de geometría documentada. Los canales
persistentes confirmados sin geometría se excluyen del ajuste y de la referencia.
Este hallazgo impide afirmar que la validación completa de reparación esté terminada.
No se han generado etiquetas humanas ni una aprobación ficticia.

## Pruebas

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_validation*.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Las pruebas usan señales sintéticas y archivos temporales. Cubren segmentación,
conservación del original, filtros auxiliares, rango, no convergencia, avisos por
canal, reparación, referencia fija, respaldo, bloqueo de evaluación, invalidación,
conflictos y fallos de guardado. La suite anterior tenía un fallo de visibilidad del
botón de navegación en `test_cascade_navigation_grid_zoom_and_resize`; Xvfb no está
instalado en este entorno. Ese fallo se registra como línea base, no como prueba
de validación fisiológica.
