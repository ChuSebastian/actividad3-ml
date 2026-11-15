"""Comparaciones registradas de parámetros y ajustes cilíndricos de control."""

import json
import time
import logging
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.optimize import least_squares
from src.clustering import dbscan, graph_regions, combine_regions
from src.segmentation import extract_planes
from src.region_geometry import descriptors, analyze_neighbors
from src.cylinders import basis, geometry
from src.evaluation import metrics
from src.paths import CACHE, EXPERIMENTS

LOGGER = logging.getLogger(__name__)


def baseline(xyz, eps, min_points=6):
    """Calcula dos planos y DBSCAN como referencia espacial reproducible."""
    o3d.utility.random.seed(42)
    left = np.arange(len(xyz))
    labels = np.full(len(xyz), -1, np.int32)
    for i in range(2):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz[left])
        (_, inliers) = pc.segment_plane(
            distance_threshold=0.055, ransac_n=3, num_iterations=1000, probability=0.999
        )
        mask = np.zeros(len(left), bool)
        mask[inliers] = True
        labels[left[mask]] = i
        left = left[~mask]
    b = dbscan(xyz[left], eps, min_points)
    labels[left] = np.where(b >= 0, b + 2, -1)
    return labels


def fit(points, normals, seed_indices, axis, threshold=0.025, iterations=180, seed=42):
    """Ajusta un cilindro parcial que debe explicar la región utilizada como semilla."""
    if len(points) < 60 or len(seed_indices) < 12:
        return None
    rng = np.random.default_rng(seed)
    (a, u, v) = basis(axis)
    origin = np.median(points[seed_indices], axis=0)
    d = points - origin
    xy = np.column_stack([d @ u, d @ v])
    pn = np.column_stack([normals @ u, normals @ v])
    pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
    tangent = np.column_stack([-pn[:, 1], pn[:, 0]])
    train = np.arange(len(points)) % 5 != 0
    hold = ~train
    seed_mask = np.zeros(len(points), bool)
    seed_mask[seed_indices] = True
    candidates = np.flatnonzero(train & (abs(normals @ a) < 0.3))
    if len(candidates) < 40:
        return None
    best = None
    best_score = -1
    for _ in range(iterations):
        pair = rng.choice(candidates, 2, replace=False)
        matrix = tangent[pair]
        if abs(np.linalg.det(matrix)) < 0.3:
            continue
        center = np.linalg.solve(matrix, np.einsum("ij,ij->i", matrix, xy[pair]))
        rr = np.linalg.norm(xy - center, axis=1)
        radius = float(np.mean(rr[pair]))
        if not 0.065 < radius < 0.55:
            continue
        radial = (xy - center) / np.maximum(rr[:, None], 1e-12)
        align = abs(np.einsum("ij,ij->i", radial, pn))
        mask = (
            (abs(rr - radius) < threshold)
            & (align > 0.88)
            & (abs(normals @ a) < 0.3)
            & train
        )
        seed_fraction = mask[seed_mask & train].mean()
        if seed_fraction < 0.45:
            continue
        score = mask.sum() + 2 * (mask & seed_mask).sum()
        if score > best_score:
            best = (center, radius, mask)
            best_score = score
    if best is None:
        return None
    (center, radius, mask) = best
    q = points[mask] - origin
    nn = normals[mask]

    def residuals(theta):
        aa = a + theta[2] * u + theta[3] * v
        aa /= np.linalg.norm(aa)
        cc = theta[0] * u + theta[1] * v
        delta = q - cc
        projection = delta @ aa
        rr = delta - projection[:, None] * aa
        length = np.linalg.norm(rr, axis=1)
        radial = rr / np.maximum(length[:, None], 1e-12)
        return np.r_[
            length - theta[4],
            0.03 * (1 - abs(np.einsum("ij,ij->i", nn, radial))),
            0.03 * (nn @ aa),
        ]

    opt = least_squares(
        residuals,
        [*center, 0, 0, radius],
        bounds=(
            [-np.inf, -np.inf, -0.12, -0.12, 0.065],
            [np.inf, np.inf, 0.12, 0.12, 0.55],
        ),
        loss="soft_l1",
        f_scale=0.012,
        max_nfev=70,
    )
    theta = opt.x
    aa = a + theta[2] * u + theta[3] * v
    aa /= np.linalg.norm(aa)
    cc = origin + theta[0] * u + theta[1] * v
    model = dict(axis=aa.tolist(), center=cc.tolist(), radius=float(theta[4]))
    (dist, align, axial, t, radial) = geometry(points, normals, model)
    inside = (dist < threshold) & (align > 0.88) & (axial < 0.25)
    if inside.sum() < 60:
        return None
    ra = radial[inside]
    angles = np.arctan2(ra @ v, ra @ u)
    angles = np.sort((angles + 2 * np.pi) % (2 * np.pi))
    angular_span = float(
        360 - np.rad2deg(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max())
    )
    bins = np.unique(np.floor(angles / (2 * np.pi) * 12).astype(int)).size
    interval = np.quantile(t[inside], [0.01, 0.99])
    length = float(np.diff(interval)[0])
    seed_fraction = float(inside[seed_mask].mean())
    train_fraction = float(inside[train].mean())
    hold_fraction = float(inside[hold].mean())
    (hist, _) = np.histogram(t[inside], bins=np.linspace(*interval, 13))
    occupancy = float((hist >= 2).mean())
    accepted = bool(
        seed_fraction >= 0.65
        and angular_span >= 120
        and (bins >= 5)
        and (length >= 1.2)
        and (length / model["radius"] >= 5)
        and (occupancy >= 0.8)
        and (hold_fraction >= 0.7 * train_fraction)
        and (np.median(dist[inside]) < 0.016)
    )
    model.update(
        accepted=accepted,
        seed_inlier_fraction=seed_fraction,
        n_inliers=int(inside.sum()),
        angular_span_deg=angular_span,
        angular_bins=bins,
        length=length,
        axial_min=float(interval[0]),
        axial_max=float(interval[1]),
        axial_occupancy=occupancy,
        train_inlier_fraction=train_fraction,
        holdout_inlier_fraction=hold_fraction,
        median_radial_residual=float(np.median(dist[inside])),
        p95_radial_residual=float(np.quantile(dist[inside], 0.95)),
        normal_alignment_p10=float(np.quantile(align[inside], 0.1)),
        threshold=threshold,
    )
    return model


def cross_section_checks(points, normals, model):
    """Rechaza círculos que solo explican franjas de una viga o engloban varias piezas."""
    (dist, align, axial, t, radial) = geometry(points, normals, model)
    delta = points - np.asarray(model["center"])
    rr = np.linalg.norm(delta - t[:, None] * np.asarray(model["axis"]), axis=1)
    inside = (dist < model["threshold"]) & (align > 0.88) & (axial < 0.25)
    purity = []
    interior = []
    radial_radii = []
    bounds = np.linspace(model["axial_min"], model["axial_max"], 13)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        local = (t >= lo) & (t < hi) & (rr < model["radius"] + 0.035)
        if local.sum() < 8:
            continue
        purity.append(float(inside[local].mean()))
        interior.append(float((rr[local] < model["radius"] - 0.045).mean()))
        if (local & inside).sum() > 4:
            radial_radii.append(float(np.median(rr[local & inside])))
    model = model.copy()
    model["accepted_radial_only"] = model["accepted"]
    model["section_purity_median"] = float(np.median(purity)) if purity else 0
    model["section_interior_median"] = float(np.median(interior)) if interior else 1
    model["section_bins_tested"] = len(purity)
    model["radius_axial_std"] = float(np.std(radial_radii)) if radial_radii else 1
    model["accepted"] = bool(
        model["accepted_radial_only"]
        and len(purity) >= 8
        and (model["section_purity_median"] >= 0.72)
        and (model["section_interior_median"] <= 0.12)
        and (model["radius_axial_std"] < 0.015)
    )
    return model


def run_experiments(original, colors, original_normals, working):
    """Repite únicamente las configuraciones ya registradas en la actividad."""
    (points, normals) = (working["xyz"], working["normals"])
    weights = working["counts"]
    rows = []
    folder = CACHE / "experiments"
    folder.mkdir(exist_ok=True)

    def record(name, labels, start, parameters, objective):
        rows.append(
            {
                "experiment": name,
                "objective": objective,
                "seconds": time.perf_counter() - start,
                **metrics(labels, weights),
                "parameters": json.dumps(parameters),
            }
        )
        np.save(folder / f"{name}_labels.npy", labels)

    for eps, minimum in [(0.12, 6), (0.16, 6), (0.2, 6), (0.16, 12)]:
        start = time.perf_counter()
        labels = baseline(points, eps, minimum)
        record(
            f"baseline_e{eps:.2f}_m{minimum}",
            labels,
            start,
            {
                "voxel": 0.08,
                "planes": 2,
                "ransac_threshold": 0.055,
                "iterations": 1000,
                "eps": eps,
                "min_points": minimum,
            },
            "Evaluar fusión espacial frente a ruido",
        )
    for threshold in [0.035, 0.055, 0.08]:
        start = time.perf_counter()
        (labels, models) = extract_planes(points, normals, threshold=threshold)
        record(
            f"planes_t{threshold:.3f}",
            labels,
            start,
            {
                "threshold": threshold,
                "max_planes": 18,
                "iterations": 1200,
                "normal_angle": 22,
                "connectivity_eps": 0.22,
            },
            "Sensibilidad del soporte y contaminación de planos",
        )
    planes = np.load(CACHE / "planes.npy")
    remaining = np.flatnonzero(planes < 0)
    start = time.perf_counter()
    coarse = combine_regions(planes, dbscan(points[remaining], 0.18, 6), remaining)
    record(
        "coarse",
        coarse,
        start,
        {"eps": 0.18, "min_points": 6},
        "Observar fusiones después de retirar planos",
    )
    for angle, color in [(18, 0.4), (28, 0.4), (40, 0.4), (28, 0.22), (28, 1.8)]:
        start = time.perf_counter()
        residual = graph_regions(
            points[remaining],
            normals[remaining],
            working["rgb"][remaining],
            working["curvature"][remaining],
            radius=0.18,
            angle=angle,
            color=color,
            tangent=0.065,
            min_size=25,
        )
        labels = combine_regions(planes, residual, remaining)
        record(
            f"regions_a{angle}_c{color:.2f}",
            labels,
            start,
            {
                "radius": 0.18,
                "angle": angle,
                "color": color,
                "tangent": 0.065,
                "min_size": 25,
                "k": 20,
            },
            "Separar contactos por orientación y color; medir fragmentación",
        )
    previous = pd.read_csv(EXPERIMENTS / "clustering.csv")
    refined = previous[previous.experiment == "region_refinement"]
    pd.concat([pd.DataFrame(rows), refined], ignore_index=True).to_csv(
        EXPERIMENTS / "clustering.csv", index=False
    )
    reproduce_cylinder_controls(working)
    LOGGER.info("Comparaciones de parámetros reproducidas")


def reproduce_cylinder_controls(working):
    """Reproduce los ensayos radiales descartados, sin usarlos para etiquetar."""
    (points, normals, weights) = (working["xyz"], working["normals"], working["counts"])
    labels = np.load(CACHE / "refined_regions.npy")
    preliminary_classes = np.load(CACHE / "preliminary_classes.npy")
    ids = np.unique(labels)
    from src.classification import NAMES

    metadata = pd.DataFrame(
        {
            "segment_id": ids,
            "semantic_id": [int(preliminary_classes[labels == sid][0]) for sid in ids],
        }
    ).set_index("segment_id")
    metadata["semantic"] = metadata.semantic_id.map(dict(enumerate(NAMES)))
    (regions, groups) = descriptors(
        points, normals, working["rgb"], labels, weights, metadata
    )
    crop_bounds = {
        "conductos_izquierda": [-1, 11, -8, -1, 0.5, 5.4],
        "conductos_derecha": [11, 24, -8, 2, 0.5, 5.4],
        "control_vigas": [2, 19, 4.2, 6.8, 5, 6.6],
    }
    all_records = []
    summaries = []
    for strict in [False, True]:
        fit_records = []
        for name, bounds in crop_bounds.items():
            limits = np.array(bounds)
            mask = np.all(
                (points >= limits[[0, 2, 4]]) & (points <= limits[[1, 3, 5]]), axis=1
            )
            candidates = [
                r
                for r in regions
                if r["segment_id"] >= 0
                and r["semantic_id"] not in [1, 2, 4]
                and (r["linearity"] > 0.93)
                and (r["length"] > 1.5)
                and (r["cross_width"] < 1.6)
                and (abs(r["axis"][2]) < 0.5)
                and (mask[groups[r["segment_id"]]].mean() > 0.6)
            ]
            candidates = sorted(candidates, key=lambda r: r["n_points"], reverse=True)[
                :28
            ]
            for threshold in [0.018, 0.028, 0.045]:
                shown = labels.copy()
                claimed = np.zeros(len(labels), bool)
                local_records = []
                for region in candidates:
                    segment_id = region["segment_id"]
                    indices = groups[segment_id]
                    axis = region["axis"]
                    offsets = points - region["center"]
                    axial = offsets @ axis
                    radial = np.linalg.norm(offsets - axial[:, None] * axis, axis=1)
                    extent = np.quantile(axial[indices], [0, 1])
                    pool = np.flatnonzero(
                        mask
                        & (radial < 0.68)
                        & (axial >= extent[0] - 0.28)
                        & (axial <= extent[1] + 0.28)
                    )
                    selected = np.flatnonzero(np.isin(pool, indices))
                    start = time.perf_counter()
                    model = fit(
                        points[pool],
                        normals[pool],
                        selected,
                        axis,
                        threshold=threshold,
                        seed=42 + segment_id,
                    )
                    fit_seconds = time.perf_counter() - start
                    row = {
                        "crop": name,
                        "threshold": threshold,
                        "seed_id": segment_id,
                        "seed_points": region["n_points"],
                        "roi_points": len(pool),
                        "status": "sin_modelo",
                    }
                    if model is not None:
                        if strict:
                            model = cross_section_checks(
                                points[pool], normals[pool], model
                            )
                        row.update(
                            {
                                key: value
                                for (key, value) in model.items()
                                if np.isscalar(value)
                            }
                        )
                        row["status"] = "aceptado" if model["accepted"] else "rechazado"
                        if model["accepted"]:
                            (dist, align, axial_alignment, positions, _) = geometry(
                                points[pool], normals[pool], model
                            )
                            chosen = pool[
                                (dist < threshold)
                                & (align > 0.88)
                                & (axial_alignment < 0.25)
                                & ~claimed[pool]
                            ]
                            shown[chosen] = segment_id
                            claimed[chosen] = True
                    row["seconds"] = fit_seconds
                    local_records.append(row)
                fit_records.extend(local_records)
                method = "ocupacion_interior" if strict else "radial_normales"
                summaries.append(
                    {
                        "experiment": method,
                        "crop": name,
                        "threshold": threshold,
                        "n_seed_fits": len(local_records),
                        "accepted_models": sum(
                            (r["status"] == "aceptado" for r in local_records)
                        ),
                        "fit_seconds": sum((r["seconds"] for r in local_records)),
                        "n_regions_before": metrics(labels[mask], weights[mask])[
                            "n_segments"
                        ],
                        **metrics(shown[mask], weights[mask]),
                        "changed_original_points": int(
                            weights[mask & (shown != labels)].sum()
                        ),
                        "visual_observation": "Ajuste parcial insuficiente para etiquetar conductos",
                        "objective": "Contrastar residuos parciales con los perfiles completos",
                    }
                )
        suffix = "_strict" if strict else ""
        pd.DataFrame(fit_records).to_csv(
            EXPERIMENTS / f"cylinder_crop_fits{suffix}.csv", index=False
        )
    pd.DataFrame(summaries).to_csv(EXPERIMENTS / "cylinder_controls.csv", index=False)
