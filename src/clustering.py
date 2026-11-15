"""Conectividad multiescala con restricciones de orientación, color y tangencia."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN


def dbscan(xyz: np.ndarray, eps, min_points=6):
    """Agrupa por densidad espacial y conserva el ruido como -1."""
    return (
        DBSCAN(eps=eps, min_samples=min_points, n_jobs=2)
        .fit_predict(xyz)
        .astype(np.int32)
    )


def compact(labels: np.ndarray, min_size=1, offset=0):
    """Filtra grupos pequeños y asigna IDs contiguos a los restantes."""
    (segment_ids, counts) = np.unique(labels[labels >= 0], return_counts=True)
    keep = segment_ids[counts >= min_size]
    result = np.full(len(labels), -1, np.int32)
    positions = np.searchsorted(keep, labels)
    valid = (labels >= 0) & (positions < len(keep))
    if len(keep):
        valid &= keep[np.minimum(positions, len(keep) - 1)] == labels
    else:
        valid[:] = False
    result[valid] = positions[valid] + offset
    return result


def graph_regions(
    xyz: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    curvature: np.ndarray,
    radius=0.18,
    angle=28,
    color=0.4,
    tangent=0.065,
    min_size=25,
    k=20,
):
    """Componente conexa de grafo kNN con aristas admisibles; no equivale a objeto completo."""
    if len(xyz) == 0:
        return np.empty(0, np.int32)
    k = min(k, len(xyz))
    (distances, neighbors) = cKDTree(xyz).query(xyz, k=k, workers=2)
    source_indices = np.repeat(np.arange(len(xyz)), k - 1)
    target_indices = neighbors[:, 1:].reshape(-1)
    edge_distances = distances[:, 1:].reshape(-1)
    normal_alignment = np.abs(
        np.einsum("ij,ij->i", normals[source_indices], normals[target_indices])
    )
    color_distance = np.linalg.norm(rgb[source_indices] - rgb[target_indices], axis=1)
    delta = xyz[target_indices] - xyz[source_indices]
    tangent_distance = np.maximum(
        np.abs(np.einsum("ij,ij->i", delta, normals[source_indices])),
        np.abs(np.einsum("ij,ij->i", delta, normals[target_indices])),
    )
    admissible = (
        (edge_distances <= radius)
        & (normal_alignment >= np.cos(np.deg2rad(angle)))
        & (color_distance <= color)
        & (tangent_distance <= tangent)
    )
    admissible &= (curvature[source_indices] < 0.15) & (
        curvature[target_indices] < 0.15
    )
    graph = coo_matrix(
        (
            np.ones(admissible.sum(), np.uint8),
            (source_indices[admissible], target_indices[admissible]),
        ),
        shape=(len(xyz), len(xyz)),
    ).tocsr()
    (_, labels) = connected_components(graph, directed=False)
    return compact(labels, min_size)


def fill_boundaries(
    xyz: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    radius=0.13,
    angle=38,
    color=0.5,
):
    """Una pasada conservadora: voto de vecinos ya asignados, sin propagación en cadena."""
    result = labels.copy()
    unknown = np.flatnonzero(labels < 0)
    known = np.flatnonzero(labels >= 0)
    if not len(unknown) or not len(known):
        return result
    (distances, neighbor_indices) = cKDTree(xyz[known]).query(
        xyz[unknown], k=min(5, len(known)), workers=2
    )
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
        distances = distances[:, None]
    neighbor_indices = known[neighbor_indices]
    normal_alignment = np.abs(
        (normals[unknown, None, :] * normals[neighbor_indices]).sum(2)
    )
    color_distance = np.linalg.norm(
        rgb[unknown, None, :] - rgb[neighbor_indices], axis=2
    )
    valid = (
        (distances <= radius)
        & (normal_alignment >= np.cos(np.deg2rad(angle)))
        & (color_distance <= color)
    )
    votes = labels[neighbor_indices]
    for j in range(neighbor_indices.shape[1]):
        support = ((votes == votes[:, j, None]) & valid).sum(1) >= 2
        take = (result[unknown] < 0) & valid[:, j] & support
        result[unknown[take]] = votes[take, j]
    return result


def combine_regions(planes, residual, indices):
    """Integra etiquetas del residuo manteniendo los IDs de los planos."""
    out = planes.copy()
    offset = int(planes.max()) + 1
    out[indices] = np.where(residual >= 0, residual + offset, -1)
    return out
