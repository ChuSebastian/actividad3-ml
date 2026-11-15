"""Refinamiento de regiones: uniones conservadoras y semántica geométrica parcial."""

from src.region_geometry import region_descriptors
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.clustering import fill_boundaries, graph_regions


def merge_vertical_faces(points: np.ndarray, labels: np.ndarray):
    """Une caras altas del mismo soporte; no une por proximidad cualquier estructura."""
    rows = region_descriptors(points, labels)
    candidates = []
    for region in rows:
        dimensions = region["dim"]
        if (
            dimensions[2] > 3.2
            and max(dimensions[:2]) < 1.25
            and (abs(region["axis"][2]) > 0.97)
            and (region["linearity"] > 0.86)
        ):
            candidates.append(region)
    out = labels.copy()
    merges = []
    if not candidates:
        return (out, merges)
    source_edges = []
    target_edges = []
    for i, region in enumerate(candidates):
        for j, other_region in enumerate(candidates[:i]):
            overlap = min(region["hi"][2], other_region["hi"][2]) - max(
                region["lo"][2], other_region["lo"][2]
            )
            combined_width = np.maximum(
                region["hi"][:2], other_region["hi"][:2]
            ) - np.minimum(region["lo"][:2], other_region["lo"][:2])
            if (
                np.linalg.norm(region["center"][:2] - other_region["center"][:2]) < 0.85
                and max(combined_width) < 1.3
                and (overlap > 0.6 * min(region["dim"][2], other_region["dim"][2]))
            ):
                source_edges.append(i)
                target_edges.append(j)
    graph = coo_matrix(
        (np.ones(len(source_edges)), (source_edges, target_edges)),
        shape=(len(candidates), len(candidates)),
    ).tocsr()
    (_, component) = connected_components(graph, directed=False)
    for component_id in np.unique(component):
        indices = [
            candidates[j]["id"] for j in np.flatnonzero(component == component_id)
        ]
        if len(indices) > 1:
            for segment_id in indices:
                out[labels == segment_id] = min(indices)
            merges.append(
                dict(
                    before=indices,
                    after=min(indices),
                    reason="Caras verticales esbeltas, proximidad XY y solapamiento vertical",
                )
            )
    return (out, merges)


def split_large_regions(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
):
    """Separa soportes por cortes horizontales y rompe cadenas de orientación en macroregiones."""
    from sklearn.cluster import MiniBatchKMeans, DBSCAN

    result = labels.copy()
    next_id = int(labels.max()) + 1
    logs = []
    for region in region_descriptors(points, labels):
        if region["n"] < 15000 or region["dim"][2] < 2.5 or min(region["dim"][:2]) < 3:
            continue
        segment_id = region["id"]
        indices = np.flatnonzero(labels == segment_id)
        region_points = points[indices]
        region_normals = normals[indices]
        slice_height = (region["lo"][2] + region["hi"][2]) / 2
        slab = (np.abs(region_points[:, 2] - slice_height) < 0.23) & (
            np.abs(region_normals[:, 2]) < 0.35
        )
        supports = []
        if slab.sum() >= 20:
            cross_section = region_points[slab, :2]
            slab_labels = DBSCAN(eps=0.24, min_samples=8).fit_predict(cross_section)
            for component_id in np.unique(slab_labels[slab_labels >= 0]):
                section_region = cross_section[slab_labels == component_id]
                lower_bound = section_region.min(0) - 0.055
                upper_bound = section_region.max(0) + 0.055
                if (
                    max(upper_bound - lower_bound) > 1.3
                    or min(upper_bound - lower_bound) < 0.14
                ):
                    continue
                member = np.all(
                    (region_points[:, :2] >= lower_bound)
                    & (region_points[:, :2] <= upper_bound),
                    axis=1,
                )
                heights = region_points[member, 2]
                if len(heights) < 300 or np.ptp(heights) < 4:
                    continue
                (hist, _) = np.histogram(
                    heights, bins=np.linspace(region["lo"][2], region["hi"][2], 11)
                )
                if (hist >= 5).sum() < 7:
                    continue
                chosen = indices[member & (result[indices] == segment_id)]
                if len(chosen) < 300:
                    continue
                result[chosen] = next_id
                supports.append(
                    dict(
                        new_id=next_id,
                        n_representatives=len(chosen),
                        xy_min=lower_bound.tolist(),
                        xy_max=upper_bound.tolist(),
                    )
                )
                next_id += 1
        remaining = indices[result[indices] == segment_id]
        remaining_normals = normals[remaining]
        features = np.column_stack(
            [
                remaining_normals[:, 0] ** 2,
                remaining_normals[:, 1] ** 2,
                remaining_normals[:, 2] ** 2,
                np.sqrt(2) * remaining_normals[:, 0] * remaining_normals[:, 1],
                np.sqrt(2) * remaining_normals[:, 0] * remaining_normals[:, 2],
                np.sqrt(2) * remaining_normals[:, 1] * remaining_normals[:, 2],
            ]
        )
        orientations = MiniBatchKMeans(
            n_clusters=10, random_state=42, n_init=3, batch_size=4096
        ).fit_predict(features)
        new_regions = []
        result[remaining] = -1
        for orient in np.unique(orientations):
            local_ids = remaining[orientations == orient]
            reg = graph_regions(
                points[local_ids],
                normals[local_ids],
                colors[local_ids],
                curvature[local_ids],
                radius=0.18,
                angle=18,
                color=0.4,
                tangent=0.04,
                min_size=25,
            )
            for sub in np.unique(reg[reg >= 0]):
                chosen = local_ids[reg == sub]
                result[chosen] = next_id
                new_regions.append(next_id)
                next_id += 1
        logs.append(
            dict(
                source_id=segment_id,
                source_representatives=len(indices),
                slice_height=float(slice_height),
                supports=supports,
                orientation_regions=len(new_regions),
                reason="Macroregión con gran extensión 3D y cadenas de contacto; partición local por soportes y orientación",
            )
        )
    return (result, logs)


def refine(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
):
    """Divide regiones extensas y recupera componentes y fronteras compatibles."""
    (split, split_log) = split_large_regions(points, normals, colors, curvature, labels)
    (merged, merges) = merge_vertical_faces(points, split)
    first = fill_boundaries(
        points, normals, colors, merged, radius=0.13, angle=38, color=0.5
    )
    unassigned_indices = np.flatnonzero(first < 0)
    result = first.copy()
    small = graph_regions(
        points[unassigned_indices],
        normals[unassigned_indices],
        colors[unassigned_indices],
        curvature[unassigned_indices],
        radius=0.14,
        angle=35,
        color=0.45,
        tangent=0.06,
        min_size=10,
    )
    offset = int(result.max()) + 1
    result[unassigned_indices] = np.where(small >= 0, small + offset, -1)
    final = fill_boundaries(
        points, normals, colors, result, radius=0.11, angle=32, color=0.45
    )
    return (
        final,
        dict(
            large_region_splits=split_log,
            vertical_merges=merges,
            boundary_assigned=int(((merged < 0) & (first >= 0)).sum()),
            small_assigned=int(((first < 0) & (result >= 0)).sum()),
            final_boundary_assigned=int(((result < 0) & (final >= 0)).sum()),
        ),
    )
