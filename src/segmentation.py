"""Extracción iterativa de superficies planas y orientación vertical basada en evidencia."""

import numpy as np
import open3d as o3d
from src.clustering import dbscan, compact


def extract_planes(
    xyz: np.ndarray,
    normals: np.ndarray,
    threshold=0.055,
    max_planes=18,
    min_support=700,
    iterations=1200,
    angle=22,
    seed=42,
):
    """Extrae soportes planos con orientación compatible y separa componentes."""
    o3d.utility.random.seed(seed)
    remaining = np.arange(len(xyz))
    labels = np.full(len(xyz), -1, np.int32)
    planes = []
    next_id = 0
    for step in range(max_planes):
        if len(remaining) < min_support:
            break
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz[remaining])
        (model, _) = point_cloud.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=iterations,
            probability=0.999,
        )
        model = np.asarray(model)
        model /= np.linalg.norm(model[:3])
        if model[np.argmax(np.abs(model[:3]))] < 0:
            model = -model
        distances = np.abs(xyz[remaining] @ model[:3] + model[3])
        normal_alignment = np.abs(normals[remaining] @ model[:3])
        mask = (distances < threshold) & (normal_alignment > np.cos(np.deg2rad(angle)))
        inlier_indices = remaining[mask]
        if len(inlier_indices) < min_support:
            break
        connected = dbscan(xyz[inlier_indices], eps=0.22, min_points=4)
        connected = compact(connected, min_size=100)
        segment_ids = []
        for component_id in np.unique(connected[connected >= 0]):
            region = inlier_indices[connected == component_id]
            labels[region] = next_id
            segment_ids.append(next_id)
            next_id += 1
        planes.append(
            dict(
                step=step,
                model=model.tolist(),
                support=len(inlier_indices),
                segment_ids=segment_ids,
                centroid=xyz[inlier_indices].mean(0).tolist(),
                dimensions=np.ptp(xyz[inlier_indices], axis=0).tolist(),
                residual_median=float(np.median(distances[mask])),
            )
        )
        remaining = remaining[~mask]
    return (labels, planes)


def infer_vertical(planes, xyz: np.ndarray):
    """Estima la vertical con el plano de mayor soporte por extensión."""
    candidates = []
    for plane in planes:
        normal = np.array(plane["model"][:3])
        extents = np.sort(plane["dimensions"])
        area = extents[-1] * extents[-2]
        candidates.append((plane["support"] * area, plane))
    best = max(candidates, key=lambda candidate: candidate[0])[1]
    normal = np.array(best["model"][:3])
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0:
        normal = -normal
    return dict(
        vector=normal.tolist(),
        nearest_axis="XYZ"[axis],
        angle_to_axis_degrees=float(np.rad2deg(np.arccos(abs(normal[axis])))),
        evidence_plane=best["step"],
        status="inferencia geométrica, no metadato del escáner",
    )
