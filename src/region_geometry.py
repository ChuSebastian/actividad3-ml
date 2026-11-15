"""Descriptores ponderados y contactos entre regiones de superficie."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def descriptors(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    segment_metadata,
):
    """Calcula forma, color y orientación por región ponderando puntos originales."""
    rows = []
    groups = {}
    order = np.argsort(labels, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(order)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        indices = order[start:stop]
        segment_id = int(labels[indices[0]])
        groups[segment_id] = indices
        region_points = points[indices]
        center = np.average(region_points, axis=0, weights=weights[indices])
        delta = region_points - center
        covariance = delta.T * weights[indices] @ delta / weights[indices].sum()
        (eigenvalues, eigenvectors) = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        if axis[np.argmax(abs(axis))] < 0:
            axis = -axis
        normal_eigenvalues = np.linalg.eigvalsh(
            normals[indices].T @ normals[indices] / len(indices)
        )
        principal_coordinates = delta @ eigenvectors
        spans = np.ptp(principal_coordinates, axis=0)
        row = dict(
            segment_id=segment_id,
            n_representatives=len(indices),
            n_points=int(weights[indices].sum()),
            semantic=segment_metadata.loc[segment_id, "semantic"],
            semantic_id=int(segment_metadata.loc[segment_id, "semantic_id"]),
            linearity=float(
                (eigenvalues[2] - eigenvalues[1]) / max(eigenvalues[2], 1e-12)
            ),
            planarity=float(
                (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-12)
            ),
            plane_rms=float(np.sqrt(max(eigenvalues[0], 0))),
            normal_coherence=float(normal_eigenvalues[-1]),
            length=float(spans[2]),
            cross_width=float(max(spans[:2])),
            center=center,
            axis=axis,
            normal=eigenvectors[:, 0],
            lo=region_points.min(0),
            hi=region_points.max(0),
            rgb=np.median(colors[indices], axis=0),
            mean_normal_abs_z=float(np.abs(normals[indices, 2]).mean()),
        )
        rows.append(row)
    return (rows, groups)


def flat_table(rows):
    """Convierte vectores geométricos en columnas de una tabla de regiones."""
    flattened = []
    for row in rows:
        record = {k: value for (k, value) in row.items() if np.isscalar(value)}
        for key in ["center", "axis", "normal", "lo", "hi", "rgb"]:
            for j, value in enumerate(row[key]):
                record[f"{key}_{j}"] = float(value)
        flattened.append(record)
    return pd.DataFrame(flattened)


def contacts(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    radius=0.24,
):
    """Contacto de pares de regiones: hasta 11 vecinos, orientaciones sin signo."""
    tree = cKDTree(points)
    (distances, neighbors) = tree.query(
        points, k=12, distance_upper_bound=radius, workers=2
    )
    source_indices = np.repeat(np.arange(len(points)), 11)
    target_indices = neighbors[:, 1:].ravel()
    edge_distances = distances[:, 1:].ravel()
    valid = (target_indices < len(points)) & (source_indices < target_indices)
    source_indices = source_indices[valid]
    target_indices = target_indices[valid]
    edge_distances = edge_distances[valid]
    valid = (
        (labels[source_indices] != labels[target_indices])
        & (labels[source_indices] >= 0)
        & (labels[target_indices] >= 0)
    )
    (source_indices, target_indices, edge_distances) = (
        source_indices[valid],
        target_indices[valid],
        edge_distances[valid],
    )
    first_segment = np.minimum(labels[source_indices], labels[target_indices])
    second_segment = np.maximum(labels[source_indices], labels[target_indices])
    normal_alignment = np.abs(
        np.einsum("ij,ij->i", normals[source_indices], normals[target_indices])
    )
    color = np.linalg.norm(colors[source_indices] - colors[target_indices], axis=1)
    delta = points[target_indices] - points[source_indices]
    tangent = np.maximum(
        abs(np.einsum("ij,ij->i", delta, normals[source_indices])),
        abs(np.einsum("ij,ij->i", delta, normals[target_indices])),
    )
    frame = pd.DataFrame(
        dict(
            a=first_segment,
            b=second_segment,
            distance=edge_distances,
            normal_dot=normal_alignment,
            color_delta=color,
            tangent=tangent,
        )
    )
    aggregated = (
        frame.groupby(["a", "b"])
        .agg(
            contact_edges=("distance", "size"),
            min_distance=("distance", "min"),
            median_distance=("distance", "median"),
            normal_dot_median=("normal_dot", "median"),
            color_median=("color_delta", "median"),
            tangent_median=("tangent", "median"),
        )
        .reset_index()
    )
    return aggregated


def region_descriptors(points: np.ndarray, labels: np.ndarray):
    """Describe la forma por representante y usa cotas robustas para los soportes."""
    rows = []
    for segment_id in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == segment_id)
        region_points = points[indices]
        (lower_bound, upper_bound) = np.quantile(region_points, [0.02, 0.98], axis=0)
        centroid = region_points.mean(0)
        (eigenvalues, eigenvectors) = np.linalg.eigh(np.cov(region_points.T))
        eigenvalues = np.maximum(eigenvalues, 0)
        rows.append(
            dict(
                id=int(segment_id),
                n=len(region_points),
                lo=lower_bound,
                hi=upper_bound,
                dim=upper_bound - lower_bound,
                center=centroid,
                axis=eigenvectors[:, -1],
                linearity=(eigenvalues[2] - eigenvalues[1])
                / max(eigenvalues[2], 1e-12),
            )
        )
    return rows


def analyze_neighbors(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    metadata,
):
    """Añade orientación, separación de ejes y hueco axial a los contactos."""
    (rows, groups) = descriptors(points, normals, colors, labels, weights, metadata)
    pairs = contacts(points, normals, colors, labels)
    by_id = {row["segment_id"]: row for row in rows}
    axis_angles = []
    center_offsets = []
    axial_gaps = []
    plane_angles = []
    smaller_sizes = []
    semantic_pairs = []
    for edge in pairs.itertuples():
        (first, second) = (by_id[edge.a], by_id[edge.b])
        axis = first["axis"]
        axis_angles.append(
            np.degrees(np.arccos(np.clip(abs(axis @ second["axis"]), 0, 1)))
        )
        delta = second["center"] - first["center"]
        center_offsets.append(float(np.linalg.norm(delta - axis * (delta @ axis))))
        first_positions = points[groups[edge.a]] @ axis
        second_positions = points[groups[edge.b]] @ axis
        axial_gaps.append(
            max(
                0,
                float(
                    max(first_positions.min(), second_positions.min())
                    - min(first_positions.max(), second_positions.max())
                ),
            )
        )
        plane_angles.append(
            float(
                np.degrees(
                    np.arccos(np.clip(abs(first["normal"] @ second["normal"]), 0, 1))
                )
            )
        )
        smaller_sizes.append(min(first["n_points"], second["n_points"]))
        semantic_pairs.append(first["semantic"] + " / " + second["semantic"])
    pairs["axis_angle_deg"] = axis_angles
    pairs["axis_center_offset"] = center_offsets
    pairs["axial_gap"] = axial_gaps
    pairs["plane_normal_angle_deg"] = plane_angles
    pairs["smaller_n_points"] = smaller_sizes
    pairs["classes"] = semantic_pairs
    return (rows, groups, pairs)
