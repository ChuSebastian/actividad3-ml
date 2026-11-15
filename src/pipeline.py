"""Punto de entrada del método: python -m src.pipeline --recompute."""

import os

# La semilla y el paralelismo forman parte de la configuración reproducible.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fabrica-mpl")

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import COLORS, NAMES, planar_unknown, semantic_labels
from src.clustering import dbscan, graph_regions, combine_regions
from src.cylinders import apply_cylinders, deduplicate, proposals, validate_and_extend
from src.eda import analyze_cloud
from src.evaluation import metrics, segment_table
from src.merging import merge_fragments
from src.paths import (
    CACHE,
    DATA,
    EXPERIMENTS,
    METRICS,
    OUTPUTS,
    ROOT,
    ensure_directories,
)
from src.ply_io import write_labeled_ply
from src.preprocessing import local_geometry, read_ply, voxelize
from src.refinement import refine
from src.region_geometry import analyze_neighbors
from src.segmentation import extract_planes, infer_vertical
from src.validation import execute_notebook, validate_results
from src.visualization import palette, plot_pipeline_results

LOGGER = logging.getLogger(__name__)


def save_json(path: Path, content: dict | list) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_cylinders(original, original_normals, working):
    """Aplica los perfiles y las posiciones seleccionados en los experimentos."""
    accepted_and_rejected = []
    profile_records = []
    for name, bounds, positions in [
        ("conductos", [-1, 24, -8, 3, 0.5, 5.4], [5.0, 11.0, 17.0]),
        ("vigas", [2, 19, 4.2, 6.8, 5, 6.6], [6.0, 11.0, 16.0]),
    ]:
        lower = np.array(bounds)[[0, 2, 4]]
        upper = np.array(bounds)[[1, 3, 5]]
        original_mask = np.all((original >= lower) & (original <= upper), axis=1)
        working_mask = np.all(
            (working["xyz"] >= lower) & (working["xyz"] <= upper), axis=1
        )
        candidates, records = proposals(
            original, original_normals, 0, positions, original_mask
        )
        for record in records:
            record["crop"] = name
        profile_records.extend(records)
        for candidate in candidates:
            fitted = validate_and_extend(
                working["xyz"],
                working["normals"],
                original,
                original_normals,
                working_mask,
                candidate,
            )
            if fitted is not None:
                fitted["crop"] = name
                accepted_and_rejected.append(fitted)
    # El orden de las propuestas determina sus IDs y se conserva al serializar.
    proposal_path = CACHE / "cylinder_proposals.json"
    save_json(proposal_path, accepted_and_rejected)
    fitted_models = json.loads(proposal_path.read_text(encoding="utf-8"))
    pd.DataFrame(profile_records).to_csv(
        EXPERIMENTS / "section_profiles.csv", index=False
    )
    models, duplicate_decisions = deduplicate(fitted_models)
    save_json(METRICS / "cylinder_deduplication.json", duplicate_decisions)
    return models


def compute_stages(original, colors, original_normals):
    """Calcula las etapas seleccionadas sin repetir la búsqueda de parámetros."""
    LOGGER.info("Preprocesamiento: voxel de 0,08 unidades")
    working = voxelize(original, colors, original_normals, 0.08)
    features = local_geometry(working["xyz"], k=20)
    mean_distance = features["neighbor_distances"][:, 1:].mean(1)
    statistical_candidates = (
        mean_distance > mean_distance.mean() + 2 * mean_distance.std()
    )
    working.update(features)
    working["sor_candidates"] = statistical_candidates
    np.savez_compressed(CACHE / "working.npz", **working)
    points, normals = working["xyz"], working["normals"]
    working_colors, curvature = working["rgb"], working["curvature"]
    LOGGER.info("Representantes: %s", len(points))

    LOGGER.info("Segmentación de superficies principales")
    planes, models = extract_planes(points, normals, threshold=0.055)
    vertical = infer_vertical(models, points)
    save_json(METRICS / "planes.json", models)
    save_json(METRICS / "vertical.json", vertical)
    remaining = np.flatnonzero(planes < 0)
    coarse = combine_regions(planes, dbscan(points[remaining], 0.18, 6), remaining)
    LOGGER.info("Regiones basadas en normales, color y geometría local")
    attribute_labels = graph_regions(
        points[remaining],
        normals[remaining],
        working_colors[remaining],
        curvature[remaining],
        radius=0.18,
        angle=28,
        color=0.40,
        tangent=0.065,
        min_size=25,
    )
    attribute_labels = combine_regions(planes, attribute_labels, remaining)
    refined, refinement = refine(
        points, normals, working_colors, curvature, attribute_labels
    )
    # Estas categorías preliminares protegen piso, cubierta y soportes.
    preliminary_classes, _, _, references = semantic_labels(
        points, normals, refined, vertical["vector"]
    )
    refinement["semantic_references"] = references
    save_json(METRICS / "region_refinement.json", refinement)
    for name, labels in [
        ("planes", planes),
        ("coarse", coarse),
        ("attributes", attribute_labels),
        ("refined_regions", refined),
        ("preliminary_classes", preliminary_classes),
    ]:
        np.save(CACHE / f"{name}.npy", labels)

    LOGGER.info("Continuidad y uniones de fragmentos coplanares")
    metadata = pd.DataFrame(
        {
            "segment_id": np.unique(refined),
            "semantic_id": [
                int(preliminary_classes[refined == sid][0])
                for sid in np.unique(refined)
            ],
        }
    ).set_index("segment_id")
    metadata["semantic"] = metadata.semantic_id.map(dict(enumerate(NAMES)))
    rows, groups, neighbors = analyze_neighbors(
        points, normals, working_colors, refined, working["counts"], metadata
    )
    # Se conserva la lectura CSV de las distancias usada en la selección validada.
    neighbor_path = CACHE / "neighbor_pairs.csv"
    neighbors.to_csv(neighbor_path, index=False)
    neighbors = pd.read_csv(neighbor_path)
    merged, decisions, merge_groups = merge_fragments(
        points,
        normals,
        working_colors,
        refined,
        preliminary_classes,
        working["counts"],
        rows,
        groups,
        neighbors,
        threshold=500,
    )
    decisions.to_csv(EXPERIMENTS / "selected_merge_decisions.csv", index=False)
    save_json(EXPERIMENTS / "selected_merge_groups.json", merge_groups)
    merged_classes = preliminary_classes.copy()
    for group in merge_groups:
        merged_classes[np.isin(refined, group["sources"])] = 3
    np.save(CACHE / "before_cylinders.npy", merged)
    np.save(CACHE / "classes_before_cylinders.npy", merged_classes)

    LOGGER.info("Identificación y refinamiento de superficies cilíndricas")
    cylinder_models = detect_cylinders(original, original_normals, working)
    labels, classes, instances, cylinder_records = apply_cylinders(
        points, normals, merged, merged_classes, working["counts"], cylinder_models
    )
    LOGGER.info("Clasificación semántica parcial y comprobación de planos")
    classes, planar_checks = planar_unknown(
        points,
        normals,
        labels,
        classes,
        working["counts"],
        original,
        working["inverse"],
    )
    np.save(CACHE / "final_labels.npy", labels)
    np.save(CACHE / "final_classes.npy", classes)
    save_json(METRICS / "cylinder_models.json", cylinder_records)
    pd.DataFrame(instances).to_csv(METRICS / "cylinder_instances.csv", index=False)
    pd.DataFrame(planar_checks).to_csv(
        METRICS / "unknown_planar_checks.csv", index=False
    )
    LOGGER.info(
        "Cilindros aceptados: %s; tramos: %s", len(cylinder_records), len(instances)
    )
    return working


def export_results(original, colors, normals, working):
    """Recupera las filas originales y genera los entregables definitivos."""
    reduced_labels = np.load(CACHE / "final_labels.npy")
    reduced_classes = np.load(CACHE / "final_classes.npy")
    inverse = working["inverse"]
    labels = reduced_labels[inverse]
    classes = reduced_classes[inverse]
    np.save(OUTPUTS / "labels.npy", labels)
    np.save(OUTPUTS / "class_labels.npy", classes)
    np.save(OUTPUTS / "original_to_voxel.npy", inverse)
    vertical = json.loads((METRICS / "vertical.json").read_text())["vector"]
    preliminary = np.load(CACHE / "refined_regions.npy")
    _, _, preliminary_reasons, _ = semantic_labels(
        working["xyz"], working["normals"], preliminary, vertical
    )
    preliminary_classes = np.load(CACHE / "preliminary_classes.npy")
    names, reasons = {}, {}
    for segment_id in np.unique(reduced_labels[reduced_labels >= 0]):
        class_ids = np.unique(reduced_classes[reduced_labels == segment_id])
        assert len(class_ids) == 1
        class_id = int(class_ids[0])
        names[int(segment_id)] = NAMES[class_id]
        if class_id == 6:
            reasons[int(segment_id)] = (
                "Modelo de eje/radio común; al menos 3 de 5 perfiles originales completos; "
                "compatible con conducción, función no confirmada"
            )
        elif (
            segment_id in preliminary_reasons
            and int(preliminary_classes[preliminary == segment_id][0]) == class_id
        ):
            reasons[int(segment_id)] = preliminary_reasons[segment_id]
        else:
            reasons[int(segment_id)] = (
                "Superficie plana con continuidad o residual original p95 <0,03 u y "
                "normales compatibles; no se infiere identidad del objeto"
            )
    table = segment_table(original, normals, labels, names, reasons, vertical)
    table.to_csv(OUTPUTS / "segments.csv", index=False)
    for name, display_colors in [
        ("segmentacion_instancias.ply", palette(labels)),
        ("segmentacion_clases.ply", COLORS[classes]),
    ]:
        write_labeled_ply(
            OUTPUTS / name, original, normals, colors, display_colors, labels, classes
        )
        LOGGER.info("Exportado: %s", OUTPUTS / name)
    final_metrics = metrics(labels)
    final_metrics.update(
        semantic_unknown_points=int((classes == 0).sum()),
        semantic_unknown_pct=100 * float((classes == 0).mean()),
    )
    save_json(METRICS / "final.json", final_metrics)
    from src.cylinders import geometry

    checks = []
    models = json.loads((METRICS / "cylinder_models.json").read_text())
    for model in models:
        for instance_id in model["instance_ids"]:
            indices = np.flatnonzero(labels == instance_id)
            residual, alignment, axial_alignment, _, _ = geometry(
                original[indices], normals[indices], model
            )
            checks.append(
                {
                    "instance_id": instance_id,
                    "proposal_id": model["proposal_id"],
                    "n_original_points": len(indices),
                    "radius": model["radius"],
                    "p50_radial_original": float(np.median(residual)),
                    "p95_radial_original": float(np.quantile(residual, 0.95)),
                    "p99_radial_original": float(np.quantile(residual, 0.99)),
                    "normal_alignment_p10": float(np.quantile(alignment, 0.1)),
                    "p95_normal_axis": float(np.quantile(axial_alignment, 0.95)),
                    "profile_passed": model["profile_passed"],
                    "profile_tested": model["profile_tested"],
                }
            )
    pd.DataFrame(checks).to_csv(METRICS / "cylinder_original_geometry.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_id": index,
                "semantic": name,
                "n_points": int((classes == index).sum()),
                "pct": 100 * float((classes == index).mean()),
            }
            for index, name in enumerate(NAMES)
        ]
    ).to_csv(METRICS / "semantic_summary.csv", index=False)
    return labels, classes


def calculation_signature() -> str:
    """Invalida intermediarios al cambiar datos u operaciones, no las figuras."""
    import ast

    class WithoutDocstrings(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
            return node

    sources = []
    for name in [
        "preprocessing.py",
        "segmentation.py",
        "clustering.py",
        "refinement.py",
        "region_geometry.py",
        "merging.py",
        "cylinders.py",
        "classification.py",
        "pipeline.py",
    ]:
        tree = WithoutDocstrings().visit(ast.parse((ROOT / "src" / name).read_text()))
        sources.append(ast.dump(tree, include_attributes=False).encode())
    return hashlib.sha256(DATA.read_bytes() + b"".join(sources)).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Segmentación multietapa de la nube industrial"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recalcular todas las etapas seleccionadas",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Reproducir las comparaciones de parámetros registradas",
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Ejecutar el notebook sobre las salidas obtenidas",
    )
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_directories()
    start = time.perf_counter()
    original, colors, normals, header = read_ply(DATA)
    LOGGER.info("Carga de datos: %s puntos", len(original))
    signature = calculation_signature()
    manifest_path = CACHE / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    required = [
        "working.npz",
        "final_labels.npy",
        "final_classes.npy",
        "refined_regions.npy",
        "preliminary_classes.npy",
    ]
    cached = previous.get("signature") == signature and all(
        (CACHE / name).exists() for name in required
    )
    if options.recompute or not cached:
        analyze_cloud(original, colors, normals, header, DATA)
        working = compute_stages(original, colors, normals)
        save_json(manifest_path, {"signature": signature, "seed": 42})
    else:
        working = dict(np.load(CACHE / "working.npz"))
        LOGGER.info("Se reutilizan las etapas calculadas con esta configuración")
    if options.experiments:
        from src.experiments import run_experiments

        run_experiments(original, colors, normals, working)
    labels, classes = export_results(original, colors, normals, working)
    plot_pipeline_results(original, colors, labels, classes, working)
    validate_results(original, colors, normals, working)
    if options.notebook:
        execute_notebook(ROOT / "notebooks" / "segmentacion_fabrica.ipynb")
    LOGGER.info(
        "Resultado: %s regiones; %.2f s",
        metrics(labels)["n_segments"],
        time.perf_counter() - start,
    )


if __name__ == "__main__":
    main()
