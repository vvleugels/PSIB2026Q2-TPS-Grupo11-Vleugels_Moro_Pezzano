# Diseño acordado para la próxima implementación

Este documento describe el diseño acordado para la próxima implementación del
procesamiento y la revisión de EEG; **no describe el comportamiento actual**.
Las decisiones siguientes fijan la dirección del desarrollo. Los detalles de
implementación y validación que aún no se acordaron se enumeran por separado.
La creación inicial de este documento fue exclusivamente documental. El estado
del piloto posterior se detalla a continuación y en [VALIDATION.md](VALIDATION.md).

## Estado de implementación

Se implementó la primera etapa en un motor y visor de validación independientes
(`validate_eeg.py`), con epochs de 4 segundos, ICA clásica, revisión por canal,
reparaciones confirmadas y trazabilidad. Sus parámetros son experimentales.
La aplicación principal conserva el flujo anterior hasta completar la validación
y obtener aprobación explícita del perfil. No hay aceptación automática definitiva.

La muestra inicial está verificada, pero faltan etiquetas humanas y geometría
documentada: los `.set` contienen posiciones NaN y los TSV no incluyen un archivo
de sistema de coordenadas. La validación de mapas/interpolación permanece pendiente.

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
