"""Comprobaciones de integridad y de equivalencia con los resultados seleccionados."""

import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from src.paths import CACHE, DATA, FIGURES, METRICS, OUTPUTS, ROOT, TEMP
from src.ply_io import read_records

LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_results(original, colors, normals, working) -> dict:
    """Comprueba cada índice original, los atributos PLY y los modelos aceptados."""
    labels = np.load(OUTPUTS / "labels.npy")
    classes = np.load(OUTPUTS / "class_labels.npy")
    inverse = np.load(OUTPUTS / "original_to_voxel.npy")
    reference = json.loads((METRICS / "automatic_reference.json").read_text())
    assert len(labels) == len(classes) == len(original) == reference["n_points"]
    assert file_hash(DATA) == reference["source_sha256"]
    assert file_hash(OUTPUTS / "labels.npy") == reference["labels_sha256"]
    from src.semantic_consistency import consistency

    baseline = np.load(CACHE / "classes_before_consistency.npy")
    buffer = io.BytesIO()
    np.save(buffer, baseline[inverse])
    assert hashlib.sha256(buffer.getvalue()).hexdigest() == reference["classes_sha256"]
    models_semantic = json.loads((METRICS / "cylinder_models.json").read_text())
    expected, changes, _, _ = consistency(
        working["xyz"],
        working["normals"],
        working["rgb"],
        np.load(CACHE / "final_labels.npy"),
        baseline,
        working["counts"],
        models_semantic,
    )
    assert np.array_equal(classes, expected[inverse])
    known = baseline[inverse] > 0
    assert np.array_equal(classes[known], baseline[inverse][known])
    changes_recorded = pd.read_csv(METRICS / "semantic_reclassifications.csv")
    pd.testing.assert_frame_equal(
        changes_recorded, changes, check_dtype=False, check_exact=False
    )
    changed_points = classes != baseline[inverse]
    assert np.isin(labels[changed_points], changes.segment_id).all()
    assert int(changes.n_points.sum()) == int(changed_points.sum())
    assert (
        file_hash(METRICS / "cylinder_models.json")
        == reference["cylinder_models_sha256"]
    )
    assert np.array_equal(labels, np.load(CACHE / "final_labels.npy")[inverse])
    assert np.array_equal(classes, np.load(CACHE / "final_classes.npy")[inverse])
    assert np.array_equal(np.bincount(inverse), working["counts"])
    original_keys = np.floor(
        (original - working["origin"]) / float(working["size"])
    ).astype(np.int32)
    representative_keys = np.floor(
        (working["xyz"] - working["origin"]) / float(working["size"])
    ).astype(np.int32)
    assert np.array_equal(original_keys, representative_keys[inverse])
    exports = {}
    for name in ["segmentacion_instancias.ply", "segmentacion_clases.ply"]:
        path = OUTPUTS / name
        records = read_records(path)
        assert len(records) == len(original)
        for index, key in enumerate("xyz"):
            assert np.array_equal(records[key], original[:, index])
        for index, key in enumerate(["nx", "ny", "nz"]):
            assert np.array_equal(records[key], normals[:, index])
        for index, key in enumerate(["red", "green", "blue"]):
            assert np.array_equal(
                records["original_" + key],
                np.rint(colors[:, index] * 255).astype("u1"),
            )
        assert np.array_equal(records["instance_id"], labels)
        assert np.array_equal(records["semantic_id"], classes)
        assert len(o3d.io.read_point_cloud(str(path)).points) == len(original)
        # El campo semantic_id cambia solo en regiones justificadas por el registro.
        exports[name] = {
            "readable": True,
            "points": len(records),
            "sha256": file_hash(path),
        }
    table = pd.read_csv(OUTPUTS / "segments.csv")
    segment_ids, counts = np.unique(labels, return_counts=True)
    assert np.array_equal(
        table.set_index("segment_id").loc[segment_ids, "n_points"], counts
    )
    numeric_columns = reference["geometry_columns"]
    numeric_hash = hashlib.sha256(
        table[numeric_columns].to_numpy().tobytes()
    ).hexdigest()
    assert numeric_hash == reference["geometry_sha256"]
    protected = np.isin(np.load(CACHE / "preliminary_classes.npy"), [1, 2, 4])
    assert np.array_equal(
        np.load(CACHE / "final_labels.npy")[protected],
        np.load(CACHE / "refined_regions.npy")[protected],
    )
    models = json.loads((METRICS / "cylinder_models.json").read_text())
    assert len(models) == reference["n_cylinder_models"]
    assert (
        sum(len(model["instance_ids"]) for model in models)
        == reference["n_cylinder_instances"]
    )
    result = {
        "n_points": len(labels),
        "original_order_and_xyz_preserved": True,
        "normals_and_original_rgb_preserved": True,
        "index_mapping_exact": True,
        "instance_labels_exact": True,
        "semantic_labels_reproducible": True,
        "cylinder_models_exact": True,
        "segment_geometry_unchanged": True,
        "protected_labels_unchanged": True,
        "previously_known_semantics_unchanged": True,
        "changed_class_points": int(changed_points.sum()),
        "reclassified_regions": len(changes),
        "integrity_is_not_quality_evaluation": True,
        "exports": exports,
        "python": sys.version.split()[0],
        "open3d": o3d.__version__,
    }
    (METRICS / "validation.json").write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "Validación correcta: integridad e índices conservados; clasificación reproducible (no mide accuracy)"
    )
    return result


def execute_notebook(path: Path) -> None:
    """Ejecuta las celdas con el mismo intérprete que ejecuta el pipeline."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    os.environ.setdefault("IPYTHONDIR", str(TEMP / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(TEMP / "jupyter"))
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_manager=manager,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(
        output.output_type == "error" for cell in cells for output in cell.outputs
    )
    summary = {"executed_code_cells": len(cells), "error_outputs": 0}
    (METRICS / "notebook_validation.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Notebook ejecutado: %s celdas de código", len(cells))
