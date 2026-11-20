# Actividad 3 — Segmentación multietapa de una nube de puntos industrial 3D

## Descripción

Este proyecto desarrolla una segmentación no supervisada de una nube de puntos de una instalación industrial. A partir de las coordenadas tridimensionales (XYZ), los colores (RGB) y las normales de cada punto, buscamos separar las principales superficies y estructuras de la escena y asignarles colores que permitan visualizar los resultados.

La solución combina varios métodos en etapas. Primero reduce la cantidad de puntos utilizada para el procesamiento, después identifica superficies extensas y agrupa las estructuras restantes. Finalmente, refina regiones complejas, comprueba algunos tramos cilíndricos y asigna categorías geométricas cuando existe información suficiente. El resultado se exporta en formato PLY, tanto por **instancias** (regiones identificadas) como por **clases** (tipos de estructura).

## Datos

El archivo de entrada es `data/FARO_Scan_Full_Sub1_down3_with_normals.ply`, que contiene 1.647.021 puntos. Cada punto incluye coordenadas y normales en `float32`, además de color RGB en `uint8`.

Los archivos originales de `data/` se mantienen sin modificaciones. Las unidades del escáner no están confirmadas como metros, por lo que las distancias de este proyecto se expresan en **unidades de la nube (u)**. El eje Z se considera aproximadamente vertical a partir de la orientación observada.

## Metodología

El procesamiento se organiza en las siguientes etapas:

1. **Voxelización.** Se reduce la nube de trabajo mediante celdas de 0,08 u. Se obtienen 459.991 representantes y se conserva la correspondencia con los puntos originales.
2. **Detección de superficies con RANSAC.** Se buscan planos grandes, como el piso y los paneles de cubierta. La configuración utiliza una tolerancia de 0,055 u, hasta 18 modelos y 1.200 iteraciones; después se consideran orientación y continuidad para separar las regiones.
3. **Agrupamiento del resto con DBSCAN y un grafo de vecinos.** Se combinan cercanía espacial y características locales como normales, RGB, curvatura y tangencia. La configuración seleccionada incluye radio de 0,18 u, ángulo de 28°, diferencia RGB de 0,40 y tangencia de 0,065 u.
4. **Refinamiento geométrico.** Se dividen regiones demasiado grandes, se revisan uniones entre superficies y se recuperan puntos próximos a los bordes cuando corresponde.
5. **Comprobación de cilindros.** Se analizan perfiles circulares y su continuidad a lo largo de los tramos. Se identificaron siete tramos candidatos; las separaciones longitudinales superiores a 0,45 u permiten distinguir instancias.
6. **Clasificación geométrica y exportación.** Se aplican reglas conservadoras a las regiones y se transfieren las etiquetas a los 1.647.021 puntos originales, respetando su orden.

Los parámetros seleccionados, las comparaciones y las explicaciones más detalladas están en `notebooks/segmentacion_fabrica.ipynb` y `outputs/metrics/parameters.json`.

## Resultados

La configuración final produce **1.967 regiones** y asigna una instancia al **97,080 %** de los puntos originales. En la clasificación semántica, **412.170 puntos** permanecen como desconocidos.

Se exportan dos visualizaciones principales:

- `outputs/segmentacion_instancias.ply`: cada región identificada recibe un color distinto.
- `outputs/segmentacion_clases.ply`: las regiones se colorean según su categoría geométrica.

Las categorías empleadas son:

| ID | Categoría |
|---:|---|
| 0 | Desconocido |
| 1 | Piso |
| 2 | Cubierta superior |
| 3 | Superficie plana |
| 4 | Soporte vertical candidato |
| 5 | Elemento lineal candidato |
| 6 | Conducción cilíndrica candidata |

Las categorías marcadas como candidatas se asignan a partir de su geometría y no confirman la función industrial del objeto. La cantidad de regiones y el porcentaje de puntos con instancia describen el resultado, pero **no representan accuracy**.

También se exploraron reglas para reconocer personas y vehículos. Estas pruebas no dieron resultados suficientemente consistentes para incorporar dichas categorías a la clasificación final. Se complementó el análisis con una revisión visual de regiones seleccionadas, útil para identificar errores de fragmentación, agrupamiento y clasificación.

## Requisitos e instalación

Se recomienda utilizar **Python 3.10**, aproximadamente **8 GB de RAM** y espacio disponible para las nubes y archivos intermedios. Las dependencias están fijadas en `requirements.txt`, incluyendo Open3D, NumPy, SciPy y scikit-learn.

En WSL o Linux, desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si falta el módulo `venv`, debe instalarse `python3-venv` desde el gestor de paquetes de la distribución. Para trabajos con muchos archivos pequeños, puede resultar más rápido usar el sistema de archivos Linux en lugar de `/mnt/c`.

## Ejecución

Para recalcular la segmentación seleccionada y ejecutar el notebook:

```bash
python -m src.pipeline --recompute --notebook
```

Otros modos disponibles:

```bash
python -m src.pipeline
python -m src.pipeline --recompute
python -m src.pipeline --recompute --experiments --notebook
```

- Sin opciones, se reutiliza una caché compatible para actualizar las salidas y validaciones.
- `--recompute` vuelve a calcular la segmentación desde la nube original.
- `--experiments` ejecuta las comparaciones de parámetros implementadas y requiere más tiempo.
- `--notebook` ejecuta las celdas de análisis sobre los resultados disponibles.

El procesamiento utiliza semilla 42 y una configuración controlada de hilos. El notebook también puede abrirse directamente en Jupyter o VS Code utilizando el intérprete de `.venv`; por defecto, sus celdas analizan resultados existentes y no vuelven a ejecutar toda la segmentación.

### Inspección de regiones

Para estudiar una región específica y sus alrededores:

```bash
python -m src.inspect_segment --id 1102
```

El comando genera recortes, distintas vistas y descriptores en `temp/review/instance_<ID>/`. También admite varios IDs y opciones como `--context-margin` y `--max-neighbors`. Estas visualizaciones son auxiliares y no modifican las etiquetas ni los PLY finales.

Las figuras preparadas para el artículo se encuentran en `outputs/figures/report_ready/`. Su galería e índice están disponibles en `index.html` y `LEEME.md`. Se pueden regenerar con:

```bash
python -m src.report_ready_figures
```

## Organización del repositorio

```text
data/                 Nubes de puntos proporcionadas
informe/              Informe
notebooks/            Notebook con análisis y explicaciones
src/                  Implementación del pipeline y herramientas auxiliares
outputs/              Resultados, tablas, métricas y figuras
  figures/            Visualizaciones de etapas, experimentos y resultado
  metrics/            Parámetros y estadísticas de la ejecución
  experiments/        Comparaciones experimentales
  cache/              Archivos intermedios regenerables (excluidos de Git)
  human_review/       Material auxiliar para inspección y anotación visual
temp/                 Archivos temporales y recortes (excluidos de Git)
requirements.txt      Dependencias de Python
README.md             Instrucciones del proyecto
```

`src.pipeline` es el punto de entrada principalel procesamiento. Otros módulos de `src/` sirven para inspeccionar regiones, generar figuras y realizar análisis complementarios.

Entre los archivos auxiliares de salida se encuentran `outputs/segments.csv` (descriptores y categorías por región), `outputs/labels.npy`, `outputs/class_labels.npy` y `outputs/original_to_voxel.npy` (etiquetas y correspondencia entre puntos originales y representantes). Los PLY finales conservan coordenadas, normales, orden original y atributos para identificar instancias y clases.

## Experimentos y revisión visual

Las carpetas `outputs/experiments/` y `outputs/metrics/` reúnen comparaciones de parámetros y resultados de las etapas principales. Las figuras muestran cómo cambian las agrupaciones al variar la voxelización, los parámetros de RANSAC, DBSCAN o las restricciones del grafo.

El análisis cualitativo utiliza vistas RGB, instancias y clases para examinar regiones de interés. Los candidatos y plantillas de revisión se mantienen separados de las predicciones automáticas; las observaciones humanas no cambian por sí solas el resultado de segmentación.

## Reproducibilidad y limitaciones

La ejecución principal puede recalcular la nube segmentada, mientras que algunas comparaciones previas dependen de tablas de evidencia conservadas en el repositorio. Por ello, **no se deben borrar indiscriminadamente `outputs/metrics/` ni `outputs/experiments/`** antes de intentar reproducir todas las figuras del análisis. En particular, algunas comparaciones de preprocesamiento, fragmentos y restos pequeños no cuentan actualmente con un generador completo en los módulos disponibles.

La segmentación todavía presenta fragmentaciones en tuberías, uniones entre estructuras próximas y regiones difíciles de clasificar. Una región puede representar solo una parte de un objeto físico. Las pruebas de personas y vehículos mostraron también las dificultades de utilizar únicamente reglas geométricas en una escena industrial compleja. Al no contar con etiquetas de referencia para toda la nube, no se reportan métricas globales de clasificación como accuracy, precision, recall o IoU.

Los archivos PLY finales ocupan aproximadamente 55 MiB cada uno. GitHub puede mostrar advertencias por archivos mayores de 50 MiB, aunque el límite ordinario por archivo es superior; si se utiliza Git LFS o una descarga externa, deben actualizarse las instrucciones del repositorio.
