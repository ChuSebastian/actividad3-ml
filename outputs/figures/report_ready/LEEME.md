# Figuras para el informe

Generar desde la raíz con `python -m src.report_ready_figures`. Lee PLY, tablas y caché existentes; no ejecuta segmentación ni cambia resultados. Para repetir las figuras de ensayos se necesitan `outputs/cache/experiments/*_labels.npy`, generados por los experimentos del proyecto. La ausencia de esa evidencia se informa como error, no se inventa.

## Selección recomendada

| Parte del paper | Figura nueva | Sustituye / función |
|---|---|---|
| Entrada | metodo/entrada.png | 01_original.png |
| Resumen | metodo/resumen_pipeline.png | comparativa_etapas.png |
| Densidad | graficas/eda_densidad.png | eda_densidad.png |
| Secciones | graficas/secciones_originales.png | secciones_originales.png |
| Consistencia | metodo/consistencia_soporte.png | semantic_1102_comparison.png; antes/después real de clases |
| EDA | graficas/eda_normales_vecinos.png | eda_normales_vecinos.png; ejes, unidades y leyendas completos |
| EDA | graficas/eda_distribuciones.png y eda_proyecciones.png | Gráficas exploratorias anteriores |
| Voxelización | metodo/voxelizacion_general.png y voxelizacion_detalle.png | Nueva evidencia visual; igual cámara por par |
| Voxelización | graficas/voxelizacion_resumen.png | Nueva gráfica de los ensayos guardados |
| RANSAC | metodo/planes.png | 03_planos.png |
| DBSCAN | metodo/dbscan_residual.png; dbscan_detalle.png | Nueva comparación de la entrada residual y las etiquetas espaciales reales |
| Atributos | metodo/attributes.png | 05_atributos.png |
| Refinamiento | metodo/refined_regions.png | 06_regiones_refinadas.png |
| Cilindros | metodo/refinamiento_cilindrico.png | refinamiento_tubo.png |
| Cilindros | graficas/validacion_perfil_cilindrico.png | Nueva sección real y modelo aceptado |
| Parámetros | metodo/comparacion_*.png | experimentos_dbscan/ransac/atributos.png, sin frases internas |
| Parámetros | graficas/experimentos_*.png | Comparación numérica complementaria, no accuracy |
| Resultado | vistas/isometrica.png y frontal.png | isometrica_comparada.jpg y frontal_comparada.jpg |
| Otras perspectivas | vistas/lateral_izquierda.png, lateral_derecha.png, superior.png, inferior_oblicua.png | RGB, instancias y clases con encuadre idéntico |
| Leyenda | leyenda_clases.png | Leyenda separada; ID 0 mostrado como Desconocido |
| Evaluación dirigida | revision/id_31_contexto.png, id_97_contexto.png | caso_31.jpg y caso_97.jpg |
| Posible mezcla | revision/id_154_contexto.png | caso_154.jpg |
| Persona o fragmentos (hipótesis) | revision/id_182_contexto.png | caso_182.jpg |
| Maquinaria/vehículo (hipótesis) | revision/id_456_contexto.png | Recorte adicional |
| Soportes y desconocidos | revision/id_1102_contexto.png, id_1097_contexto.png | semantic_1102_comparison.png como contexto actual; no sustituye su prueba antes/después |
| Extremos lineales | revision/id_812_contexto.png, id_1019_contexto.png, id_1940_contexto.png | Casos adicionales pendientes de revisión |
| Ejemplos candidatos | revision/id_10_contexto.png, id_0_contexto.png, id_1016_contexto.png, id_1154_contexto.png | Cubierta, piso, soporte y elemento lineal; sin declarar corrección |

Los archivos `_detalle` muestran otra perspectiva ajustada al segmento completo. Sus puntos no se recortan; parte del contexto periférico puede quedar fuera del encuadre de detalle. El archivo `_contexto` conserva el recorte completo para interpretar posibles fusiones. Las vistas de resultado y las comparaciones de parámetros son anchas: usar el ancho completo disponible o `figure*` si el documento tiene dos columnas. Las revisiones son 2×2; las estructuras verticales estrechas usan cuatro paneles en fila para reducir espacios vacíos. Mantener `keepaspectratio`; no fijar ancho y alto deformando la imagen. La leyenda se incluye solo cuando sea necesaria, fuera de los paneles.

No recomiendo conservar para el paper las versiones 1×4 verticales ni los paneles sueltos que repiten un montaje. Tampoco hace falta usar las seis perspectivas generales a la vez: priorizar isométrica y frontal; seleccionar las restantes según qué estructura se discuta. No se modifica el LaTeX automáticamente. Las etiquetas del archivo y las clases automáticas permanecen iguales. Las unidades son las del escáner (u), no metros certificados.
