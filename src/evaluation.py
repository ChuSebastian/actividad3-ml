"""Métricas descriptivas sin ground truth; no equivalen a accuracy."""

import numpy as np
import pandas as pd


def metrics(labels: np.ndarray, weights: np.ndarray = None):
    """Resume grupos, tamaños y cobertura; la etiqueta -1 queda sin instancia."""
    weights = np.ones(len(labels), np.int64) if weights is None else weights
    assigned = labels >= 0
    (segment_ids, inverse_labels) = np.unique(labels[assigned], return_inverse=True)
    sizes = (
        np.bincount(inverse_labels, weights=weights[assigned])
        if len(segment_ids)
        else np.array([])
    )
    total = int(weights.sum())
    unassigned_count = int(weights[~assigned].sum())
    return dict(
        n_segments=len(segment_ids),
        n_points=total,
        unknown_points=unassigned_count,
        unknown_pct=100 * unassigned_count / total,
        coverage_pct=100 * (total - unassigned_count) / total,
        largest_pct=100 * float(sizes.max(initial=0)) / total,
        size_min=int(sizes.min()) if len(sizes) else 0,
        size_median=float(np.median(sizes)) if len(sizes) else 0,
        size_p90=float(np.quantile(sizes, 0.9)) if len(sizes) else 0,
        size_max=int(sizes.max(initial=0)),
        small_segments_lt100=int((sizes < 100).sum()),
    )


def segment_table(
    xyz: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    semantic,
    justifications,
    vertical,
):
    """Describe cada región usando las coordenadas y normales originales."""
    rows = []
    vertical = np.asarray(vertical)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        segment_id = int(labels[indices[0]])
        segment_points = xyz[indices]
        lower_bound = segment_points.min(0)
        upper_bound = segment_points.max(0)
        center = segment_points.mean(0)
        covariance = (
            np.cov(segment_points.T) if len(segment_points) > 2 else np.eye(3) * 1e-12
        )
        (eigenvalues, eigenvectors) = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        normals_here = normals[indices]
        normal_coherence = float(
            np.linalg.eigvalsh(normals_here.T @ normals_here / max(len(indices), 1))[-1]
        )
        rows.append(
            dict(
                segment_id=segment_id,
                n_points=len(indices),
                **{
                    f"centroid_{axis_name}": center[axis_index]
                    for (axis_index, axis_name) in enumerate("xyz")
                },
                **{
                    f"bbox_min_{axis_name}": lower_bound[axis_index]
                    for (axis_index, axis_name) in enumerate("xyz")
                },
                **{
                    f"bbox_max_{axis_name}": upper_bound[axis_index]
                    for (axis_index, axis_name) in enumerate("xyz")
                },
                **{
                    f"dimension_{axis_name}": upper_bound[axis_index]
                    - lower_bound[axis_index]
                    for (axis_index, axis_name) in enumerate("xyz")
                },
                linearity=float(
                    (eigenvalues[2] - eigenvalues[1]) / max(eigenvalues[2], 1e-12)
                ),
                planarity=float(
                    (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-12)
                ),
                scattering=float(eigenvalues[0] / max(eigenvalues[2], 1e-12)),
                pca_vertical_alignment=float(abs(eigenvectors[:, -1] @ vertical)),
                normal_coherence=normal_coherence,
                semantic=semantic.get(segment_id, "unknown"),
                justification=justifications.get(
                    segment_id,
                    "Evidencia insuficiente; región o ruido sin clase semántica fiable",
                ),
            )
        )
    table = pd.DataFrame(rows)
    from src.classification import NAMES as SEMANTIC_NAMES
    from src.visualization import palette

    table["semantic_id"] = table["semantic"].map(
        {name: i for (i, name) in enumerate(SEMANTIC_NAMES)}
    )
    colors = np.rint(palette(table.segment_id.to_numpy()) * 255).astype(np.uint8)
    for axis_index, name in enumerate(["display_red", "display_green", "display_blue"]):
        table[name] = colors[:, axis_index]
    return table
