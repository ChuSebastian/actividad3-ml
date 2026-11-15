"""EDA ejecutable antes de decidir el método; sin supuestos métricos ni de verticalidad."""

import logging
import json
import os
import time
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fabrica-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.preprocessing import read_ply

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/FARO_Scan_Full_Sub1_down3_with_normals.ply"
LOGGER = logging.getLogger(__name__)


def analyze_cloud(points, colors, normals, header, source_file):
    start = time.perf_counter()
    random = np.random.default_rng(42)
    valid = np.isfinite(points).all(1)
    valid_points = points[valid]
    sample = random.choice(len(valid_points), min(60000, len(valid_points)), False)
    tree = cKDTree(valid_points)
    (distances, neighbors) = tree.query(valid_points[sample], k=17, workers=2)
    length = np.linalg.norm(normals, axis=1)
    unit = normals / np.maximum(length[:, None], 1e-12)
    normal_products = (unit[sample] * unit[neighbors[:, 1]]).sum(1)
    stats = dict(
        n_points=len(points),
        header=header,
        invalid_xyz=int((~valid).sum()),
        invalid_rgb=int((~np.isfinite(colors).all(1)).sum()),
        invalid_normals=int((~np.isfinite(normals).all(1)).sum()),
        zero_normals=int((length < 1e-08).sum()),
        duplicate_xyz=int(len(valid_points) - len(np.unique(valid_points, axis=0))),
        bbox_min=valid_points.min(0).tolist(),
        bbox_max=valid_points.max(0).tolist(),
        dimensions=np.ptp(valid_points, axis=0).tolist(),
        position_quantiles=np.quantile(
            valid_points, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1], axis=0
        ).tolist(),
        rgb_quantiles=np.quantile(colors, [0, 0.1, 0.5, 0.9, 1], axis=0).tolist(),
        normal_length_quantiles=np.quantile(length, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        abs_normal_axis_mean=np.abs(unit).mean(0).tolist(),
        neighbor_opposite_fraction=float((normal_products < 0).mean()),
        neighbor_abs_dot_quantiles=np.quantile(
            np.abs(normal_products), [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        distance_quantiles={
            str(k): np.quantile(
                distances[:, k], [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ).tolist()
            for k in [1, 4, 8, 16]
        },
        distance_quantile_levels=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        sample_seed=42,
        sample_n=len(sample),
        units="desconocidas: unidades de coordenadas (u)",
        sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )
    output_dir = ROOT / "outputs"
    (output_dir / "metrics/eda.json").write_text(json.dumps(stats, indent=2))
    (figure, axes) = plt.subplots(2, 3, figsize=(14, 8))
    for j in range(3):
        axes[0, j].hist(
            valid_points[:, j], bins=100, color=["#356b9b", "#b76d38", "#4d8068"][j]
        )
        axes[0, j].set_title("Coordenada " + "XYZ"[j] + " (u)")
        axes[1, j].hist(
            colors[:, j], bins=50, color=["red", "green", "blue"][j], alpha=0.65
        )
        axes[1, j].set_title("RGB " + "RGB"[j] + " / 255")
    figure.tight_layout()
    figure.savefig(output_dir / "figures/eda_distribuciones.png", dpi=180)
    plt.close(figure)
    (figure, axes) = plt.subplots(1, 3, figsize=(15, 4))
    for j in range(3):
        axes[0].hist(np.abs(unit[sample, j]), bins=50, histtype="step", label="XYZ"[j])
    axes[0].legend()
    axes[0].set_title("|Normal| por eje")
    for k in [1, 4, 8, 16]:
        axes[1].plot(np.sort(distances[:, k]), label=str(k) + " vecino")
    axes[1].set_ylim(0, np.quantile(distances[:, 16], 0.99))
    axes[1].legend()
    axes[1].set_title("Distancias ordenadas (u)")
    axes[2].hist(normal_products, bins=80)
    axes[2].set_title("Producto de normales vecinas")
    figure.tight_layout()
    figure.savefig(output_dir / "figures/eda_normales_vecinos.png", dpi=180)
    plt.close(figure)
    (figure, axes) = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        axis.scatter(
            valid_points[sample, i],
            valid_points[sample, j],
            c=colors[sample],
            s=0.3,
            rasterized=True,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("XYZ"[i] + " (u)")
        axis.set_ylabel("XYZ"[j] + " (u)")
    figure.tight_layout()
    figure.savefig(output_dir / "figures/eda_proyecciones.png", dpi=200)
    plt.close(figure)
    LOGGER.info(
        "EDA: %s puntos, %s coordenadas inválidas; %.2f s",
        len(points),
        stats["invalid_xyz"],
        time.perf_counter() - start,
    )
