"""Inspección auxiliar: python -m src.inspect_segment --id 1097."""

import argparse
import colorsys
import itertools
import json
import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

from src.paths import OUTPUTS, TEMP
from src.ply_io import read_records, write_labeled_ply
from src.visualization import camera, montage, render

LOGGER = logging.getLogger(__name__)
HIGHLIGHT = np.array([1.0, 0.05, 0.55])
BACKGROUND = np.array([0.76, 0.78, 0.80])
VIEWS = [
    ("Oblicua", -60, 24),
    ("Frente", -90, 0),
    ("Lateral", 0, 0),
    ("Superior", -90, 90),
]


def nonnegative_id(value: str) -> int:
    instance_id = int(value)
    if instance_id < 0:
        raise argparse.ArgumentTypeError(
            "El ID debe ser >=0; -1 agrupa puntos sin instancia, no un objeto."
        )
    return instance_id


def positive_distance(value: str) -> float:
    distance = float(value)
    if not np.isfinite(distance) or distance <= 0:
        raise argparse.ArgumentTypeError(
            "La distancia debe ser finita y mayor que cero."
        )
    return distance


def load_metadata() -> dict[int, dict]:
    """Prefiere la tabla de candidatos y completa IDs desde segments.csv."""
    metadata = {}
    for name in ["segments.csv", "review_candidates.csv"]:
        path = OUTPUTS / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.segment_id.duplicated().any():
            raise ValueError(f"Hay IDs duplicados en {path}.")
        for record in table.to_dict(orient="records"):
            row = {
                key: None if pd.isna(value) else value for key, value in record.items()
            }
            # Las predicciones automáticas permanecen separadas de futuras anotaciones.
            row.pop("human_label", None)
            row.pop("reviewed", None)
            row["source_table"] = f"outputs/{name}"
            metadata[int(row["segment_id"])] = row
    return metadata


def review_camera(points: np.ndarray, azimuth: float, elevation: float) -> dict:
    """Añade margen a la caja para encuadrar incluso regiones planas o lineales."""
    lower, upper = points.min(0), points.max(0)
    padding = max(float(np.ptp(points, axis=0).max()) * 0.02, 0.05)
    corners = np.array(list(itertools.product(*zip(lower - padding, upper + padding))))
    return camera(corners, azimuth, elevation, width=1200, height=900)


def local_context(points, labels, selected, margin, max_neighbors):
    """Recorta la caja ampliada y ordena vecinos por distancia mínima al segmento."""
    target = points[selected]
    lower, upper = target.min(0) - margin, target.max(0) + margin
    crop_indices = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
    distances, _ = cKDTree(target).query(points[crop_indices], k=1, workers=1)
    local_labels = labels[crop_indices]
    close = (distances <= margin) & (local_labels >= 0) & ~selected[crop_indices]
    neighbor_ids, inverse = np.unique(local_labels[close], return_inverse=True)
    minimum_distances = np.full(len(neighbor_ids), np.inf)
    np.minimum.at(minimum_distances, inverse, distances[close])
    ordered = sorted(zip(minimum_distances, neighbor_ids))[:max_neighbors]
    return crop_indices, ordered, lower, upper


def make_figures(points, selected, crop_indices, context_colors, folder, instance_id):
    """Guarda cuatro vistas aisladas y cuatro vistas del contexto sin muestrear puntos."""
    frame_directory = folder / "frames"
    frame_directory.mkdir(exist_ok=True)
    views = {}
    for filename, view_points, view_colors in [
        (
            "segmento",
            points[selected],
            np.broadcast_to(HIGHLIGHT, (int(selected.sum()), 3)),
        ),
        ("contexto", points[crop_indices], context_colors),
    ]:
        paths, views[filename] = [], {}
        for index, (name, azimuth, elevation) in enumerate(VIEWS):
            view = review_camera(view_points, azimuth, elevation)
            path = frame_directory / f"{filename}_{index}.png"
            render(
                view_points,
                view_colors,
                path,
                f"ID {instance_id} · {name} · {filename}",
                view,
                radius=2,
            )
            paths.append(path)
            views[filename][name] = view
        montage(paths, folder / f"{filename}_perspectivas.png")
    return views


def add_context_legend(path, instance_id, neighbors):
    """Relaciona los colores del contexto con sus IDs, sin interpretar los objetos."""
    with Image.open(path) as image:
        panel = image.convert("RGB")
    height = 55 + 42 * max(1, (len(neighbors) + 3) // 4)
    result = Image.new("RGB", (panel.width, panel.height + height), (247, 247, 247))
    result.paste(panel, (0, 0))
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (35, panel.height + 10),
        f"Magenta: ID {instance_id} seleccionado · Gris: resto del contexto",
        fill=(28, 39, 51),
        font=font,
    )
    for index, neighbor in enumerate(neighbors):
        x = 35 + (index % 4) * (panel.width // 4)
        y = panel.height + 50 + (index // 4) * 42
        color = tuple(neighbor[key] for key in ["red", "green", "blue"])
        draw.rectangle((x, y, x + 25, y + 23), fill=color)
        draw.text(
            (x + 35, y), f"ID {neighbor['instance_id']}", fill=(28, 39, 51), font=font
        )
    result.save(path, dpi=(200, 200))


def inspect(
    instance_id, records, points, normals, original_rgb, metadata, margin, max_neighbors, save_ply=True
):
    """Exporta copias de revisión; conserva coordenadas, orden y etiquetas del PLY."""
    labels, classes = records["instance_id"], records["semantic_id"]
    selected = labels == instance_id
    folder = TEMP / "review" / f"instance_{instance_id}"
    folder.mkdir(parents=True, exist_ok=True)
    if save_ply:
        display_colors = np.tile(BACKGROUND, (len(points), 1))
        display_colors[selected] = HIGHLIGHT
        write_labeled_ply(
            folder / "resaltado.ply",
            points,
            normals,
            original_rgb,
            display_colors,
            labels,
            classes,
        )

    crop_indices, neighbors, lower, upper = local_context(
        points, labels, selected, margin, max_neighbors
    )
    context_labels = labels[crop_indices]
    context_colors = np.tile(BACKGROUND, (len(crop_indices), 1))
    neighbor_rows = []
    for index, (distance, neighbor_id) in enumerate(neighbors):
        # La gama de vecinos evita el magenta reservado a la región seleccionada.
        hue = 0.05 + 0.65 * index / max(len(neighbors) - 1, 1)
        color = np.array(colorsys.hsv_to_rgb(hue, 0.8, 0.85))
        context_colors[context_labels == neighbor_id] = color
        row = dict(
            instance_id=int(neighbor_id),
            min_distance=float(distance),
            points_in_crop=int((context_labels == neighbor_id).sum()),
            semantic=metadata.get(int(neighbor_id), {}).get("semantic"),
        )
        row.update(
            zip(["red", "green", "blue"], np.rint(color * 255).astype(int).tolist())
        )
        neighbor_rows.append(row)
    context_colors[context_labels == instance_id] = HIGHLIGHT
    pd.DataFrame(
        neighbor_rows,
        columns=[
            "instance_id",
            "min_distance",
            "points_in_crop",
            "semantic",
            "red",
            "green",
            "blue",
        ],
    ).to_csv(folder / "vecinos.csv", index=False)
    views = make_figures(
        points, selected, crop_indices, context_colors, folder, instance_id
    )
    add_context_legend(folder / "contexto_perspectivas.png", instance_id, neighbor_rows)

    description = dict(
        metadata[instance_id],
        instance_id=instance_id,
        input_ply="outputs/segmentacion_instancias.ply",
        context_margin=margin,
        max_neighbors=max_neighbors,
        crop_min=lower.tolist(),
        crop_max=upper.tolist(),
        n_context_points=len(crop_indices),
        highlight_rgb=np.rint(HIGHLIGHT * 255).astype(int).tolist(),
        background_rgb=np.rint(BACKGROUND * 255).astype(int).tolist(),
        neighbor_definition="Distancia mínima entre puntos <= margen; contexto visual, sin inferir clase ni fusionar regiones",
    )
    (folder / "descriptores.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metadata[instance_id]]).to_csv(
        folder / "descriptores.csv", index=False
    )
    (folder / "camaras.json").write_text(json.dumps(views, indent=2), encoding="utf-8")
    row = metadata[instance_id]
    LOGGER.info(
        "ID %s · %s · %s puntos · dimensiones (u): %.3f × %.3f × %.3f",
        instance_id,
        row["semantic"],
        int(selected.sum()),
        row["dimension_x"],
        row["dimension_y"],
        row["dimension_z"],
    )
    LOGGER.info(
        "PCA: linealidad %.4f; planaridad %.4f; alineación vertical %.4f",
        row["linearity"],
        row["planarity"],
        row["pca_vertical_alignment"],
    )
    LOGGER.info("Archivos de revisión: %s", folder)


def main():
    parser = argparse.ArgumentParser(
        description="Inspección visual de instancias, sin alterar predicciones."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidates", action="store_true", help="Solo IDs de review_candidates.csv; sin duplicar PLY completos")
    selection.add_argument("--id", type=nonnegative_id, help="ID de una instancia")
    selection.add_argument(
        "--ids",
        type=nonnegative_id,
        nargs="+",
        help="Lista de IDs separados por espacios",
    )
    parser.add_argument(
        "--context-margin",
        type=positive_distance,
        default=1.0,
        help="Margen y distancia máxima de vecindad en unidades del escáner (1,0 por defecto)",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=8,
        help="Máximo de vecinos coloreados (8 por defecto; el resto queda gris)",
    )
    options = parser.parse_args()
    if not 0 <= options.max_neighbors <= 12:
        parser.error("--max-neighbors debe estar entre 0 y 12.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    instance_ids = list(
        dict.fromkeys([options.id] if options.id is not None else (options.ids or []))
    )
    try:
        records = read_records(OUTPUTS / "segmentacion_instancias.ply")
        metadata = load_metadata()
        if options.candidates:
            instance_ids = sorted(pd.read_csv(OUTPUTS / "review_candidates.csv").segment_id.astype(int).unique().tolist())
        # Validar el lote entero antes de crear archivos evita revisiones parciales por IDs erróneos.
        for instance_id in instance_ids:
            selected = records["instance_id"] == instance_id
            if not selected.any():
                raise ValueError(f"El instance_id {instance_id} no existe en el PLY.")
            if instance_id not in metadata:
                raise ValueError(
                    f"No hay descriptores registrados para el ID {instance_id}."
                )
            row = metadata[instance_id]
            if int(row["n_points"]) != int(selected.sum()) or not np.all(
                records["semantic_id"][selected] == int(row["semantic_id"])
            ):
                raise ValueError(
                    f"La tabla y el PLY no coinciden para el ID {instance_id}."
                )
        points = np.column_stack([records[key] for key in ["x", "y", "z"]])
        normals = np.column_stack([records[key] for key in ["nx", "ny", "nz"]])
        original_rgb = (
            np.column_stack(
                [records[f"original_{key}"] for key in ["red", "green", "blue"]]
            )
            / 255.0
        )
        for instance_id in instance_ids:
            inspect(
                instance_id,
                records,
                points,
                normals,
                original_rgb,
                metadata,
                options.context_margin,
                options.max_neighbors,
                save_ply=not options.candidates,
            )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error de inspección: {error}\n")


if __name__ == "__main__":
    main()
