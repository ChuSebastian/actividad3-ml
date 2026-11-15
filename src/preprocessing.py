"""Lectura PLY, reducción trazable y características locales. Unidades del escáner."""

import numpy as np
from scipy.spatial import cKDTree

PLY_TYPES = {
    "float": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "int": "<i4",
    "uint": "<u4",
    "float32": "<f4",
    "float64": "<f8",
    "uint8": "u1",
    "int32": "<i4",
}


def read_ply(path):
    """PLY binario little endian con propiedades escalares por vértice, sin caras."""
    with open(path, "rb") as stream:
        header = []
        properties = []
        count = None
        vertex = False
        while True:
            line = stream.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("format ") and "binary_little_endian" not in line:
                raise ValueError("Se requiere PLY binario little endian")
            if line.startswith("element "):
                vertex = line.split()[1] == "vertex"
                if vertex:
                    count = int(line.split()[2])
            if line.startswith("property ") and vertex:
                (_, property_type, name) = line.split()
                properties.append((name, PLY_TYPES[property_type]))
            if line == "end_header":
                break
            if not line:
                raise ValueError("Cabecera incompleta")
        records = np.fromfile(stream, dtype=np.dtype(properties), count=count)
    if len(records) != count:
        raise ValueError("PLY truncado")
    xyz = np.column_stack([records[k] for k in ["x", "y", "z"]]).astype(np.float64)
    rgb = (
        np.column_stack([records[k] for k in ["red", "green", "blue"]]).astype(
            np.float64
        )
        / 255
    )
    normals = np.column_stack([records[k] for k in ["nx", "ny", "nz"]]).astype(
        np.float64
    )
    return (xyz, rgb, normals, header)


def voxelize(xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray, size):
    """Promedia atributos, conserva inversa exacta original→voxel (no nearest neighbor)."""
    valid = np.isfinite(xyz).all(1)
    origin = xyz[valid].min(0)
    keys = np.floor((xyz[valid] - origin) / size).astype(np.int32)
    (voxel_keys, inverse_valid, counts) = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )

    def avg(a):
        return np.column_stack(
            [
                np.bincount(
                    inverse_valid, weights=a[valid, column], minlength=len(counts)
                )
                / counts
                for column in range(3)
            ]
        )

    points = avg(xyz)
    colors = avg(np.nan_to_num(rgb))
    unit_normals = normals[valid]
    unit_normals = unit_normals / np.maximum(
        np.linalg.norm(unit_normals, axis=1, keepdims=True), 1e-12
    )
    tensor = np.empty((len(counts), 3, 3))
    for row in range(3):
        for column in range(3):
            tensor[:, row, column] = (
                np.bincount(
                    inverse_valid,
                    weights=unit_normals[:, row] * unit_normals[:, column],
                    minlength=len(counts),
                )
                / counts
            )
    (_, vectors) = np.linalg.eigh(tensor)
    voxel_normals = vectors[:, :, -1]
    inverse = np.full(len(xyz), -1, np.int32)
    inverse[valid] = inverse_valid
    return dict(
        xyz=points,
        rgb=colors,
        normals=voxel_normals,
        inverse=inverse,
        counts=counts,
        origin=origin,
        size=np.array(size),
    )


def local_geometry(xyz: np.ndarray, k=20):
    """Describe la forma de cada vecindario de 20 puntos mediante PCA."""
    tree = cKDTree(xyz)
    (distances, neighbors) = tree.query(xyz, k=k, workers=2)
    curvature = np.empty(len(xyz))
    planarity = np.empty(len(xyz))
    linearity = np.empty(len(xyz))
    normals = np.empty_like(xyz)
    for start in range(0, len(xyz), 10000):
        stop_index = min(start + 10000, len(xyz))
        neighbor_points = xyz[neighbors[start:stop_index]]
        neighbor_points = neighbor_points - neighbor_points.mean(1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", neighbor_points, neighbor_points) / k
        (eigenvalues, eigenvectors) = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        curvature[start:stop_index] = eigenvalues[:, 0] / np.maximum(
            eigenvalues.sum(1), 1e-12
        )
        planarity[start:stop_index] = (
            eigenvalues[:, 1] - eigenvalues[:, 0]
        ) / np.maximum(eigenvalues[:, 2], 1e-12)
        linearity[start:stop_index] = (
            eigenvalues[:, 2] - eigenvalues[:, 1]
        ) / np.maximum(eigenvalues[:, 2], 1e-12)
        normals[start:stop_index] = eigenvectors[:, :, 0]
    return dict(
        neighbor_distances=distances,
        neighbors=neighbors,
        curvature=curvature,
        planarity=planarity,
        linearity=linearity,
        pca_normals=normals,
    )
