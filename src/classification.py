"""Clases geométricas parciales y comprobación de regiones desconocidas."""

import numpy as np
from src.region_geometry import region_descriptors

NAMES = [
    "unknown",
    "piso",
    "cubierta_superior",
    "superficie_plana",
    "soporte_vertical_candidato",
    "elemento_lineal_candidato",
    "conduccion_cilindrica_candidata",
]
COLORS = np.array(
    [
        [0.53, 0.56, 0.6],
        [0.2, 0.63, 0.83],
        [0.89, 0.63, 0.23],
        [0.68, 0.4, 0.74],
        [0.27, 0.72, 0.46],
        [0.91, 0.37, 0.32],
        [0.12, 0.28, 0.86],
    ]
)


def planar_unknown(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    original: np.ndarray,
    inverse: np.ndarray,
):
    """Reetiqueta solo forma plana, nunca identidad de equipo; verificación en originales."""
    result = classes.copy()
    records = []
    for segment_id in np.unique(labels[(labels >= 0) & (classes == 0)]):
        indices = np.flatnonzero(labels == segment_id)
        if len(indices) < 50 or not (classes[indices] == 0).all():
            continue
        region_points = points[indices]
        center = region_points.mean(0)
        (eigenvalues, eigenvectors) = np.linalg.eigh(np.cov(region_points.T))
        normal = eigenvectors[:, 0]
        plane_distances = abs((region_points - center) @ normal)
        span = np.ptp((region_points - center) @ eigenvectors, axis=0)
        normal_fraction = float(np.mean(abs(normals[indices] @ normal) > 0.94))
        planarity = float(
            (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-12)
        )
        candidate = bool(
            np.quantile(plane_distances, 0.95) < 0.022
            and normal_fraction > 0.9
            and (span[1] > 0.28)
            and (planarity > 0.35)
        )
        record = dict(
            segment_id=int(segment_id),
            n_points=int(weights[indices].sum()),
            p95_reduced=float(np.quantile(plane_distances, 0.95)),
            normal_fraction=normal_fraction,
            planarity=planarity,
            accepted=False,
            p95_original=None,
        )
        if candidate:
            full_ids = np.flatnonzero(np.isin(inverse, indices))
            original_res = abs((original[full_ids] - center) @ normal)
            original_p95 = float(np.quantile(original_res, 0.95))
            record["p95_original"] = original_p95
            if original_p95 < 0.03:
                result[indices] = 3
                record["accepted"] = True
                record["reason"] = (
                    "Superficie plana comprobada en originales; no se infiere tipo de objeto"
                )
        records.append(record)
    return (result, records)


def semantic_labels(
    points: np.ndarray, normals: np.ndarray, labels: np.ndarray, vertical
):
    """Clases solo cuando existe evidencia; soporte/elemento lineal se declaran candidatos."""
    semantic_names = {}
    justifications = {}
    rows = region_descriptors(points, labels)
    vertical_direction = np.asarray(vertical)
    height = points @ vertical_direction
    low = float(np.quantile(height, 0.05))
    high = float(np.quantile(height, 0.95))
    for region in rows:
        segment_id = region["id"]
        indices = np.flatnonzero(labels == segment_id)
        region_points = points[indices]
        normal = normals[indices]
        heights = region_points @ vertical_direction
        horizontal = float(np.mean(np.abs(normal @ vertical_direction) > 0.94))
        dimensions = region["dim"]
        width = max(dimensions[:2])
        class_id = 0
        reason = "Geometría ambigua; no se infiere identidad del objeto"
        if (
            horizontal > 0.8
            and np.median(heights) < low + 0.45
            and (min(dimensions[:2]) > 1.8)
        ):
            class_id = 1
            reason = "Superficie horizontal extensa próxima a cota baja; compatible con piso en vista RGB"
        elif (
            horizontal > 0.75
            and np.median(heights) > high - 0.65
            and (min(dimensions[:2]) > 1.3)
        ):
            class_id = 2
            reason = "Superficie horizontal extensa en cota superior, consistente con paneles de cubierta visibles"
        elif (
            dimensions[2] > 3.2
            and width < 1.3
            and (abs(region["axis"] @ vertical_direction) > 0.97)
            and (region["linearity"] > 0.86)
        ):
            class_id = 4
            reason = "Región vertical esbelta y alta; posible soporte/columna, sin validación de ingeniería"
        elif (
            region["linearity"] > 0.91
            and np.linalg.norm(dimensions) > 1.5
            and (np.partition(dimensions, 1)[1] < 0.85)
        ):
            class_id = 5
            reason = "Región esbelta alargada: puede ser tubería, viga o baranda; no se distingue sin evidencia adicional"
        else:
            coherence = np.linalg.eigvalsh(normal.T @ normal / len(normal))[-1]
            if coherence > 0.97 and len(indices) >= 100:
                class_id = 3
                reason = "Orientación de normales coherente; superficie plana sin identidad semántica suficiente"
        semantic_names[segment_id] = NAMES[class_id]
        justifications[segment_id] = reason
    reduced = np.zeros(len(points), np.uint8)
    mapping = {name: i for (i, name) in enumerate(NAMES)}
    for segment_id, name in semantic_names.items():
        reduced[labels == segment_id] = mapping[name]
    return (
        reduced,
        semantic_names,
        justifications,
        dict(low_reference=low, high_reference=high),
    )
