"""Rasterización ortográfica con z-buffer: sin OpenGL ni servidor gráfico."""

import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fabrica-mpl")
import matplotlib

matplotlib.use("Agg")
from PIL import Image, ImageDraw, ImageFont


def palette(labels):
    """Asigna colores por ID y reserva gris para puntos sin instancia."""
    labels = np.asarray(labels)
    segment_ids = np.maximum(labels, 0)
    hue = (segment_ids * 0.618033988749895 + 0.11) % 1
    import matplotlib.colors as matplotlib_colors

    hsv = np.column_stack(
        [hue, 0.65 + 0.25 * (segment_ids % 3) / 2, 0.73 + 0.24 * (segment_ids % 2)]
    )
    colors = matplotlib_colors.hsv_to_rgb(hsv)
    colors[labels < 0] = [0.48, 0.51, 0.55]
    return colors


def camera(points, azimuth=-60, elevation=24, width=1800, height=1250):
    """Fija una cámara ortográfica a partir de los límites de la nube."""
    (azimuth_radians, elevation_radians) = np.deg2rad([azimuth, elevation])
    look = np.array(
        [
            np.cos(elevation_radians) * np.cos(azimuth_radians),
            np.cos(elevation_radians) * np.sin(azimuth_radians),
            np.sin(elevation_radians),
        ]
    )
    right = np.array([-np.sin(azimuth_radians), np.cos(azimuth_radians), 0])
    up = np.cross(look, right)
    basis = np.array([right, up, look])
    center = (points.min(0) + points.max(0)) / 2
    projected_points = (points - center) @ basis.T
    scale = min(
        (width - 150) / np.ptp(projected_points[:, 0]),
        (height - 160) / np.ptp(projected_points[:, 1]),
    )
    return dict(
        center=center.tolist(),
        basis=basis.tolist(),
        scale=float(scale),
        width=width,
        height=height,
        azimuth=azimuth,
        elevation=elevation,
    )


def render(points, colors, path, title="", cam=None, radius=1, subtitle=""):
    """Rasteriza puntos reales con z-buffer y guarda una imagen reproducible."""
    cam = cam or camera(points)
    (width, height) = (cam["width"], cam["height"])
    projected_points = (points - np.array(cam["center"])) @ np.array(cam["basis"]).T
    x = np.rint(width / 2 + projected_points[:, 0] * cam["scale"]).astype(int)
    y = np.rint(height / 2 - projected_points[:, 1] * cam["scale"] + 20).astype(int)
    canvas = np.full((height * width, 3), 247, np.uint8)
    depth = np.full(height * width, -np.inf)
    point_colors = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    offsets = (
        [
            (dx, dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        if radius > 1
        else ([(0, 0)] if radius == 0 else [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)])
    )
    for dx, dy in offsets:
        (shifted_x, shifted_y) = (x + dx, y + dy)
        valid = (
            (shifted_x >= 20)
            & (shifted_x < width - 20)
            & (shifted_y >= 85)
            & (shifted_y < height - 40)
            & np.isfinite(projected_points).all(1)
        )
        point_indices = np.flatnonzero(valid)
        pixel_indices = shifted_y[point_indices] * width + shifted_x[point_indices]
        order = np.lexsort((-projected_points[point_indices, 2], pixel_indices))
        pixel_indices = pixel_indices[order]
        point_indices = point_indices[order]
        take = np.r_[True, pixel_indices[1:] != pixel_indices[:-1]]
        pixel_indices = pixel_indices[take]
        point_indices = point_indices[take]
        closer = projected_points[point_indices, 2] > depth[pixel_indices]
        point_indices = point_indices[closer]
        pixel_indices = pixel_indices[closer]
        canvas[pixel_indices] = point_colors[point_indices]
        depth[pixel_indices] = projected_points[point_indices, 2]
    rendered_image = Image.fromarray(canvas.reshape(height, width, 3))
    draw = ImageDraw.Draw(rendered_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 27)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((35, 20), title, fill=(28, 39, 51), font=font)
    draw.text((35, 57), subtitle, fill=(78, 86, 96), font=small)
    rendered_image.save(path, dpi=(200, 200))
    return cam


def montage(paths, path, columns=2):
    """Compone imágenes con sus tamaños y proporciones originales."""
    images = [Image.open(p).convert("RGB") for p in paths]
    (width, height) = images[0].size
    result = Image.new(
        "RGB",
        (width * columns, height * ((len(images) + columns - 1) // columns)),
        (247, 247, 247),
    )
    for i, frame in enumerate(images):
        result.paste(frame, (i % columns * width, i // columns * height))
    result.save(path, dpi=(200, 200))


def plot_pipeline_results(original, original_colors, labels, classes, working):
    """Genera figuras de etapas con cámaras fijas y comparaciones experimentales."""
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.classification import COLORS, NAMES
    from src.paths import CACHE, EXPERIMENTS, FIGURES, METRICS
    from src.evaluation import metrics

    view = camera(original)
    (METRICS / "camera.json").write_text(json.dumps(view, indent=2))
    points = working["xyz"]
    weights = working["counts"]
    inverse = working["inverse"]
    render(
        original,
        original_colors,
        FIGURES / "01_original.png",
        "1 · Nube original RGB",
        view,
        subtitle=f"{len(original):,} puntos originales",
    )
    render(
        points,
        working["rgb"],
        FIGURES / "02_preprocesada.png",
        "2 · Voxelización trazable, 0,08 u",
        view,
        subtitle=f"{len(points):,} representantes; todos los puntos originales se conservan",
    )
    stages = [
        (
            "planes",
            "03_planos.png",
            "3 · Superficies principales",
            "RANSAC, normales y conectividad",
        ),
        (
            "coarse",
            "04_gruesa.png",
            "4 · Clustering espacial del residuo",
            "DBSCAN después de separar los planos",
        ),
        (
            "attributes",
            "05_atributos.png",
            "5 · Regiones con atributos geométricos",
            "Orientación, RGB, curvatura y tangencia",
        ),
        (
            "refined_regions",
            "06_regiones_refinadas.png",
            "6 · Refinamiento de regiones",
            "División local, soportes y recuperación de bordes",
        ),
    ]
    stage_metrics = []
    for name, filename, title, caption in stages:
        reduced = np.load(CACHE / f"{name}.npy")
        render(
            points, palette(reduced), FIGURES / filename, title, view, subtitle=caption
        )
        stage_metrics.append({"stage": name, **metrics(reduced, weights)})
    before = np.load(CACHE / "before_cylinders.npy")[inverse]
    before_classes = np.load(CACHE / "classes_before_cylinders.npy")[inverse]
    highlight = np.full((len(original), 3), [0.66, 0.68, 0.71])
    highlight[classes == 6] = COLORS[6]
    render(
        original,
        highlight,
        FIGURES / "07_cilindros.png",
        "7 · Tramos cilíndricos aceptados",
        view,
        subtitle="Perfiles circulares originales, eje y radio comunes; resto visible",
    )
    render(
        original,
        palette(labels),
        FIGURES / "08_instancias.png",
        "8 · Segmentación final por instancias",
        view,
        subtitle=f"{metrics(labels)['n_segments']} regiones; gris = sin instancia",
    )
    render(
        original,
        COLORS[classes],
        FIGURES / "09_clases.png",
        "9 · Clasificación semántica parcial",
        view,
        subtitle="Las categorías candidatas no confirman la función industrial",
    )
    stage_metrics.append({"stage": "cylinder_refinement", **metrics(labels)})
    pd.DataFrame(stage_metrics).to_csv(METRICS / "stage_metrics.csv", index=False)
    montage(
        [
            FIGURES / name
            for name in [
                "01_original.png",
                "03_planos.png",
                "06_regiones_refinadas.png",
                "08_instancias.png",
            ]
        ],
        FIGURES / "comparativa_etapas.png",
    )
    montage(
        [FIGURES / "01_original.png", FIGURES / "08_instancias.png"],
        FIGURES / "original_resultado.png",
    )
    plot_refinement_details(original, original_colors, before, labels, classes)
    plot_experiment_panels(points, weights, view)
    plot_summary_charts(stage_metrics, classes, labels)
    plot_density(original, working)
    plot_original_sections(original, original_colors)


def plot_refinement_details(original, colors, before, labels, classes):
    """Recortes idénticos que permiten observar continuidad y detalles estructurales."""
    import json
    from src.classification import COLORS
    from src.paths import FIGURES, METRICS

    views = {}
    masks = {
        "conductos": (original[:, 1] < 3)
        & (original[:, 2] > 0.2)
        & (original[:, 2] < 5.5),
        "soportes": (original[:, 1] > 3.5)
        & (original[:, 1] < 7)
        & (original[:, 0] > 2)
        & (original[:, 0] < 14)
        & (original[:, 2] > -1.8)
        & (original[:, 2] < 8.8),
    }
    for name, mask in masks.items():
        view = camera(original[mask], azimuth=-75, elevation=12)
        views[name] = {"camera": view, "n_points": int(mask.sum())}
        render(
            original[mask],
            colors[mask],
            FIGURES / f"detalle_{name}_rgb.png",
            f"Detalle de {name} · RGB original",
            view,
            subtitle="Recorte de la nube, sin alterar la geometría",
        )
        render(
            original[mask],
            palette(labels[mask]),
            FIGURES / f"detalle_{name}_instancias.png",
            f"Detalle de {name} · instancias",
            view,
            subtitle="Mismos puntos y encuadre que el recorte RGB",
        )
        if name == "conductos":
            render(
                original[mask],
                palette(before[mask]),
                FIGURES / "conductos_antes_cilindros.png",
                "Conductos · antes del refinamiento cilíndrico",
                view,
                subtitle="Resultado del refinamiento de regiones",
            )
            montage(
                [
                    FIGURES / "conductos_antes_cilindros.png",
                    FIGURES / "detalle_conductos_instancias.png",
                ],
                FIGURES / "refinamiento_conductos.png",
            )
    models = json.loads((METRICS / "cylinder_models.json").read_text())
    model = max(models, key=lambda item: item["radius"])
    axis = np.asarray(model["axis"])
    offsets = original - np.asarray(model["center"])
    axial = offsets @ axis
    radial = np.linalg.norm(offsets - axial[:, None] * axis, axis=1)
    mask = (abs(axial - np.median(axial[classes == 6])) < 3) & (
        radial < model["radius"] + 0.12
    )
    view = camera(original[mask], azimuth=-80, elevation=23)
    for stage, stage_labels, title in [
        ("antes", before, "Tramo recto · antes del refinamiento cilíndrico"),
        ("despues", labels, "Tramo recto · después del refinamiento cilíndrico"),
    ]:
        render(
            original[mask],
            palette(stage_labels[mask]),
            FIGURES / f"tubo_{stage}.png",
            title,
            view,
            subtitle="Recorte geométrico idéntico; contexto a radio r + 0,12 u",
        )
    montage(
        [FIGURES / "tubo_antes.png", FIGURES / "tubo_despues.png"],
        FIGURES / "refinamiento_tubo.png",
    )
    views["tubo"] = {"camera": view, "n_points": int(mask.sum())}
    (METRICS / "detail_cameras.json").write_text(json.dumps(views, indent=2))


def plot_experiment_panels(points, weights, view):
    """Recrea comparaciones a partir de las etiquetas de los ensayos registrados."""
    import json
    import pandas as pd
    from src.paths import CACHE, EXPERIMENTS, FIGURES, TEMP

    preview = {**view, "width": 900, "height": 625, "scale": view["scale"] / 2}
    groups = {
        "experimentos_dbscan": [
            f"baseline_e{eps:.2f}_m{minimum}"
            for (eps, minimum) in [(0.12, 6), (0.16, 6), (0.2, 6), (0.16, 12)]
        ],
        "experimentos_ransac": [
            f"planes_t{threshold:.3f}" for threshold in [0.035, 0.055, 0.08]
        ],
        "experimentos_atributos": [
            f"regions_a{angle}_c{color:.2f}"
            for (angle, color) in [
                (18, 0.4),
                (28, 0.4),
                (40, 0.4),
                (28, 0.22),
                (28, 1.8),
            ]
        ],
    }
    for filename, names in groups.items():
        images = []
        for name in names:
            path = CACHE / "experiments" / f"{name}_labels.npy"
            if not path.exists():
                continue
            experiment_labels = np.load(path)
            image_path = TEMP / f"{name}.png"
            render(
                points,
                palette(experiment_labels),
                image_path,
                name,
                preview,
                subtitle="Mismos representantes y cámara; gris = sin instancia",
            )
            images.append(image_path)
        if images:
            montage(
                images,
                FIGURES / f"{filename}.png",
                columns=2 if len(images) == 4 else 3,
            )
    controls = pd.read_csv(EXPERIMENTS / "cylinder_controls.csv")
    import matplotlib.pyplot as plt

    (figure, axes) = plt.subplots(1, 2, figsize=(13, 4))
    for axis, crop in zip(axes, ["conductos_izquierda", "control_vigas"]):
        for method, label in [
            ("radial_normales", "Distancia y normales"),
            ("ocupacion_interior", "Añadiendo ocupación interior"),
        ]:
            data = controls[
                (controls.crop == crop) & (controls.experiment == method)
            ].sort_values("threshold")
            axis.plot(data.threshold, data.accepted_models, marker="o", label=label)
        axis.set_title(crop.replace("_", " "))
        axis.set_xlabel("Tolerancia radial (u)")
        axis.set_ylabel("Ajustes aceptados")
        axis.legend(fontsize=8)
    figure.suptitle(
        "Los ajustes parciales también pueden aceptar perfiles estructurales"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "experimentos_cilindros.png", dpi=220)
    plt.close(figure)


def plot_summary_charts(stage_metrics, classes, labels):
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.paths import FIGURES
    from src.classification import NAMES, COLORS

    data = pd.DataFrame(stage_metrics)
    (figure, axes) = plt.subplots(1, 2, figsize=(13, 4))
    names = ["Planos", "Espacial", "Atributos", "Regiones", "Cilindros"]
    axes[0].bar(names, data.n_segments, color="#427a99")
    axes[0].set_ylabel("Número de regiones")
    axes[0].set_title("Una región no equivale necesariamente a un objeto")
    axes[1].bar(names, data.coverage_pct, label="Con instancia", color="#4c9d88")
    axes[1].bar(
        names,
        data.unknown_pct,
        bottom=data.coverage_pct,
        label="Sin instancia",
        color="#a5adb5",
    )
    axes[1].set_ylabel("% de puntos originales")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "metricas_etapas.png", dpi=220)
    plt.close(figure)
    sizes = np.unique(labels[labels >= 0], return_counts=True)[1]
    (figure, axis) = plt.subplots(figsize=(10, 4))
    axis.hist(
        sizes, bins=np.geomspace(max(1, sizes.min()), sizes.max(), 35), color="#5d719c"
    )
    axis.set_xscale("log")
    axis.set_xlabel("Puntos originales por región (escala logarítmica)")
    axis.set_ylabel("Regiones")
    figure.tight_layout()
    figure.savefig(FIGURES / "tamanos_segmentos.png", dpi=220)
    plt.close(figure)
    (figure, axis) = plt.subplots(figsize=(11, 3))
    axis.axis("off")
    for index, name in enumerate(NAMES):
        height = 1 - index * 0.14
        axis.add_patch(
            plt.Rectangle((0, height - 0.055), 0.03, 0.075, color=COLORS[index])
        )
        axis.text(0.045, height - 0.02, f"{index}: {name}", va="center")
    figure.savefig(FIGURES / "leyenda_clases.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_density(original, working):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from src.paths import FIGURES

    (figure, axes) = plt.subplots(1, 3, figsize=(15, 4.6))
    histogram = axes[0].hist2d(
        original[:, 0], original[:, 1], bins=110, norm=LogNorm(), cmap="viridis"
    )
    figure.colorbar(histogram[3], ax=axes[0], label="Puntos por celda XY")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X (u)")
    axes[0].set_ylabel("Y (u)")
    axes[0].set_title("Densidad proyectada")
    axes[1].hist(working["counts"], bins=np.arange(1, 30), color="#5a8c9a")
    axes[1].set_xlabel("Puntos originales por voxel")
    axes[1].set_ylabel("Voxeles")
    indices = np.random.default_rng(42).choice(len(working["xyz"]), 35000, False)
    points = working["xyz"]
    scatter = axes[2].scatter(
        points[indices, 0],
        points[indices, 1],
        c=working["neighbor_distances"][indices, 8],
        s=0.7,
        cmap="magma",
        vmin=0.08,
        vmax=0.3,
    )
    figure.colorbar(scatter, ax=axes[2], label="Distancia al vecino 8 (u)")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("X (u)")
    axes[2].set_ylabel("Y (u)")
    axes[2].set_title("Escala local de vecinos")
    figure.tight_layout()
    figure.savefig(FIGURES / "eda_densidad.png", dpi=220)
    plt.close(figure)


def plot_original_sections(points, colors):
    """Muestra perfiles originales en U, I y arcos para contrastar los ajustes."""
    import matplotlib.pyplot as plt
    from src.paths import FIGURES

    sections = [
        (
            "Estructura elevada roja",
            (points[:, 0] > 15.5)
            & (points[:, 0] < 17)
            & (points[:, 1] > 4.8)
            & (points[:, 1] < 6)
            & (points[:, 2] > 5)
            & (points[:, 2] < 6.3),
            1,
            2,
        ),
        (
            "Tramo superior, eje Y",
            (points[:, 1] > -4.5)
            & (points[:, 1] < -3.5)
            & (points[:, 0] > 9.1)
            & (points[:, 0] < 10.4)
            & (points[:, 2] > 4.3)
            & (points[:, 2] < 5.3),
            0,
            2,
        ),
        (
            "Red frontal, eje X",
            (points[:, 0] > 5)
            & (points[:, 0] < 6)
            & (points[:, 1] > -7.8)
            & (points[:, 1] < -1)
            & (points[:, 2] > 1.5)
            & (points[:, 2] < 3.4),
            1,
            2,
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, (title, mask, horizontal, vertical) in zip(axes, sections):
        axis.scatter(
            points[mask, horizontal], points[mask, vertical], c=colors[mask], s=3
        )
        axis.set_title(title)
        axis.set_xlabel("XYZ"[horizontal] + " (u)")
        axis.set_ylabel("XYZ"[vertical] + " (u)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "secciones_originales.png", dpi=220)
    plt.close(figure)


def plot_semantic_comparisons(
    original, colors, working, labels, baseline, corrected, candidates
):
    """Compara RGB, instancias y clases sobre recortes con puntos y cámara idénticos."""
    import hashlib
    import json
    from src.classification import COLORS
    from src.paths import FIGURES, METRICS, TEMP

    inverse = working["inverse"]
    full_labels, before, after = labels[inverse], baseline[inverse], corrected[inverse]
    cameras = {}
    selected = [1102, 937, 812, 871, 1019, 1940, 1097, 2001]
    # Los recortes mantienen contexto sin asignar identidad a las hipótesis.
    compact = candidates[candidates.possible_problem.str.contains("humana", na=False)]
    if len(compact):
        selected.append(int(compact.nlargest(1, "n_points").iloc[0].segment_id))
    complex_case = candidates[
        candidates.possible_problem.str.contains("Conjunto complejo", na=False)
    ]
    if len(complex_case):
        selected.append(int(complex_case.nlargest(1, "n_points").iloc[0].segment_id))
    frame_directory = TEMP / "semantic_frames"
    frame_directory.mkdir(exist_ok=True)
    for sid in dict.fromkeys(selected):
        mask_target = full_labels == sid
        if not mask_target.any():
            continue
        lo = original[mask_target].min(0) - 0.4
        hi = original[mask_target].max(0) + 0.4
        # En el conducto interesa la sección local; todo el tramo es demasiado largo.
        if sid == 2001:
            lo[0], hi[0] = 8.0, 11.0
        mask = np.all((original >= lo) & (original <= hi), axis=1)
        view = camera(
            original[mask], azimuth=-70, elevation=18, width=1400, height=1050
        )
        paths = []
        panels = [
            ("rgb", "RGB original", colors),
            ("instances", "Instancias", palette(full_labels)),
            ("before", "Clases antes de consistencia", COLORS[before]),
            ("after", "Clases después de consistencia", COLORS[after]),
        ]
        for name, title, rgb in panels:
            path = frame_directory / f"semantic_{sid}_{name}.png"
            render(
                original[mask],
                rgb[mask],
                path,
                f"Región {sid} · {title}",
                view,
                radius=2 if mask.sum() < 15000 else 1,
            )
            paths.append(path)
        montage(paths, FIGURES / f"semantic_{sid}_comparison.png")
        cameras[str(sid)] = dict(
            camera=view,
            bbox_min=lo.tolist(),
            bbox_max=hi.tolist(),
            n_points=int(mask.sum()),
            point_index_sha256=hashlib.sha256(
                np.flatnonzero(mask).tobytes()
            ).hexdigest(),
        )
    (METRICS / "semantic_cameras.json").write_text(json.dumps(cameras, indent=2))
