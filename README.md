# Actividad 3: Segmentación multietapa de una nube industrial 3D

## Descripción y objetivo

Segmentación no supervisada de una instalación industrial a partir de posición XYZ, RGB y normales. El método separa superficies extensas, agrupa el residuo y refina regiones y conducciones cilíndricas.

## Dataset

Se utiliza `data/FARO_Scan_Full_Sub1_down3_with_normals.ply`: **1.647.021 puntos** con coordenadas y normales `float32` y RGB `uint8`. Los PLY suministrados en `data/` se conservan sin modificar; el archivo auxiliar no interviene en el procesamiento. Las unidades del escáner no están certificadas como metros. Z se interpreta como vertical aproximada a partir de los datos.

## Requisitos

Python 3.10, unos 8 GB de RAM y espacio para datos, resultados y caché. Las versiones están fijadas en `requirements.txt`: Open3D 0.19.0, NumPy 1.26.4, SciPy 1.15.3 y scikit-learn 1.7.2, entre otras dependencias. No se requiere servidor gráfico: las figuras se generan sin ventanas.

## Instalación en WSL

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si falta el módulo `venv`, instalar `python3-venv` mediante el gestor de paquetes de la distribución. Para mejorar el acceso a muchos archivos pequeños, es preferible trabajar en el sistema de archivos Linux en lugar de `/mnt/c`.

## Ejecución

```bash
python -m src.pipeline --recompute --notebook
```

El comando recalcula el método seleccionado, exporta los resultados, genera las figuras, valida la integridad y ejecuta el notebook. Usa semilla 42, un hilo OpenMP y dos hilos OpenBLAS.

- `python -m src.pipeline`: reutiliza una caché compatible y actualiza exportaciones, figuras y validaciones.
- `--recompute`: recalcula desde la nube original.
- `--experiments`: reproduce las comparaciones de parámetros ya incluidas en el análisis; requiere más tiempo.
- `--notebook`: ejecuta las celdas de análisis sobre los resultados disponibles.
- `python -m unittest discover -s tests -v`: ejecuta los controles de integridad y equivalencia numérica.

El notebook también puede abrirse en Jupyter o VS Code, seleccionando el intérprete de `.venv`. Sus celdas cargan tablas y figuras; no ejecutan automáticamente la segmentación completa.

## Estructura

```text
data/          Datos suministrados
notebooks/     segmentacion_fabrica.ipynb
src/           Lectura, preprocesamiento, segmentación, refinamiento,
               cilindros, clasificación, evaluación y visualización
tests/         Controles de integridad y de resultados
outputs/       Entregables, métricas, figuras y evidencia experimental
outputs/cache/ Intermediarios regenerables, ignorados por Git
temp/          Archivos de trabajo y logs, ignorados por Git
```

`src/pipeline.py` es el único punto de entrada. Las funciones están agrupadas por responsabilidad; los experimentos están separados del cálculo de la configuración seleccionada.

## Algoritmos y configuración

1. Voxelización trazable de 0,08 unidades, con 459.991 representantes y correspondencia exacta al índice original.
2. RANSAC de planos: tolerancia 0,055 u, hasta 18 modelos y 1.200 iteraciones; orientación y conectividad separan los soportes de cada plano.
3. DBSCAN espacial y grafo de vecinos con normales, RGB, curvatura y tangencia. Configuración seleccionada: radio 0,18 u, ángulo 28°, diferencia RGB 0,40 y tangencia 0,065 u.
4. Refinamiento local de regiones, uniones coplanares comprobadas y recuperación de bordes.
5. Perfiles circulares en puntos originales, ajuste robusto de eje/radio y comprobación longitudinal de cilindros. Se conservan siete tramos; los huecos mayores de 0,45 u separan instancias.
6. Clasificación geométrica parcial y transferencia a todos los puntos originales.

Los parámetros y los ensayos están documentados en el notebook y `outputs/metrics/parameters.json`. Cambiar parámetros exige volver a evaluar los resultados; las verificaciones numéricas detectan diferencias respecto a la configuración seleccionada.

## Archivos de salida

- `segmentacion_instancias.ply` y `segmentacion_clases.ply`: **todos los puntos originales y su orden**, normales, RGB de visualización, RGB original en propiedades adicionales, `instance_id` y `semantic_id`.
- `labels.npy`, `class_labels.npy` y `original_to_voxel.npy`: etiquetas e índice de transferencia.
- `segments.csv`: tamaño, centroide, caja, geometría, categoría y justificación por región; la fila `-1` agrega puntos sin instancia y no es un objeto.
- `metrics/`: resultados finales, categorías, parámetros, modelos cilíndricos, residuos y validaciones.
- `figures/`: etapas con cámara fija, resultados, detalles y comparaciones de parámetros.
- `experiments/`: evidencia reproducible de DBSCAN, RANSAC, restricciones geométricas y controles cilíndricos.

Clases: 0 unknown, 1 piso, 2 cubierta superior, 3 superficie plana, 4 soporte vertical candidato, 5 elemento lineal candidato y 6 conducción cilíndrica candidata. El identificador de instancia `-1` significa sin región asignada.

El resultado tiene **1.967 regiones**, 97,080% de puntos con instancia y 427.058 puntos `unknown` semánticos. Cobertura y número de regiones son métricas descriptivas, no accuracy.

## Archivos grandes y Git

Los PLY definitivos ocupan aproximadamente 55 MiB cada uno y se conservan como entregables. Todos los PLY del proyecto son menores de 100 MiB, por lo que Git LFS no es obligatorio para alojarlos en GitHub. Sí pueden producir la advertencia de 50 MiB. Véanse los [límites oficiales de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Se mantienen archivos PLY completos, sin punteros LFS; no hace falta `git lfs pull` para obtenerlos.

La caché de procesamiento puede superar 100 MiB y está excluida de Git porque se regenera. Tampoco se incluyen el entorno virtual, logs ni `temp/`.

## Limitaciones

Una región puede representar parte de un objeto. Persisten fragmentos, contactos ambiguos y superficies coplanares unidas. Los cilindros se verifican en un volumen frontal de dirección principal observada; no se resuelven todos los codos o ramificaciones. La forma circular respalda una conducción candidata, sin confirmar su función. El voxel limita el detalle y las categorías protegidas pueden conservar errores locales. Sin etiquetas de referencia no se reportan accuracy, precision, recall ni IoU semántica.
