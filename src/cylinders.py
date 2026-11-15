"""Perfiles circulares, ejes y transferencia de instancias cilíndricas."""

import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN


def basis(axis):
    """Construye una base ortonormal a partir del eje propuesto."""
    a = np.array(axis, dtype=float)
    a /= np.linalg.norm(a)
    u = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return (a, u, v)


def geometry(points: np.ndarray, normals: np.ndarray, model):
    """Calcula residuo radial, orientación y posición axial respecto a un cilindro."""
    axis = np.asarray(model["axis"])
    center = np.asarray(model["center"])
    offsets = points - center
    axial_positions = offsets @ axis
    radial_vectors = offsets - axial_positions[:, None] * axis
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radial = radial_vectors / np.maximum(radial_distances[:, None], 1e-12)
    normal_alignment = np.abs(np.einsum("ij,ij->i", normals, radial))
    normal_axis_alignment = abs(normals @ axis)
    residual = abs(radial_distances - model["radius"])
    return (residual, normal_alignment, normal_axis_alignment, axial_positions, radial)


def circle_profile(cross_section):
    """Ajusta un círculo robusto y comprueba el componente transversal completo."""
    center0 = cross_section.mean(0)
    centered_points = cross_section - center0
    algebraic_parameters = np.linalg.lstsq(
        np.column_stack([2 * centered_points, np.ones(len(centered_points))]),
        np.sum(centered_points * centered_points, axis=1),
        rcond=None,
    )[0]
    center = algebraic_parameters[:2] + center0
    radius = np.sqrt(
        max(
            0,
            algebraic_parameters[2]
            + algebraic_parameters[:2] @ algebraic_parameters[:2],
        )
    )
    if not 0.055 < radius < 0.55:
        return None
    solution = least_squares(
        lambda v: np.linalg.norm(cross_section - v[:2], axis=1) - v[2],
        [*center, radius],
        bounds=([-np.inf, -np.inf, 0.055], [np.inf, np.inf, 0.55]),
        loss="soft_l1",
        f_scale=0.008,
        max_nfev=80,
    )
    center = solution.x[:2]
    radius = float(solution.x[2])
    radial_residuals = abs(np.linalg.norm(cross_section - center, axis=1) - radius)
    angles = np.sort(
        np.arctan2(cross_section[:, 1] - center[1], cross_section[:, 0] - center[0])
        % (2 * np.pi)
    )
    arc = float(360 - np.rad2deg(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()))
    return dict(
        center2d=center,
        radius=radius,
        median=float(np.median(radial_residuals)),
        p90=float(np.quantile(radial_residuals, 0.9)),
        p95=float(np.quantile(radial_residuals, 0.95)),
        fraction=float((radial_residuals < 0.022).mean()),
        arc=arc,
        accepted=bool(
            np.quantile(radial_residuals, 0.95) < 0.028
            and (radial_residuals < 0.022).mean() > 0.9
            and (arc >= 115)
        ),
    )


def proposals(original_points, normals: np.ndarray, axis_index, locations, roi):
    """Busca perfiles circulares en las secciones originales seleccionadas."""
    cross_axes = [i for i in range(3) if i != axis_index]
    models = []
    records = []
    for location in locations:
        indices = np.flatnonzero(
            roi & (abs(original_points[:, axis_index] - location) < 0.35)
        )
        cross_section = original_points[indices][:, cross_axes]
        clusters = DBSCAN(eps=0.052, min_samples=3, n_jobs=2).fit_predict(cross_section)
        for component_id in np.unique(clusters[clusters >= 0]):
            local = indices[clusters == component_id]
            if len(local) < 70:
                continue
            fit_points = original_points[local][:, cross_axes]
            fit = circle_profile(fit_points)
            if fit is None:
                continue
            record = dict(
                axis_index=axis_index,
                location=location,
                component=int(component_id),
                points=len(fit_points),
                radius=fit["radius"],
                median_residual=fit["median"],
                p95_residual=fit["p95"],
                fraction=fit["fraction"],
                arc_deg=fit["arc"],
                accepted_profile=fit["accepted"],
            )
            records.append(record)
            if not fit["accepted"]:
                continue
            axis = np.eye(3)[axis_index]
            center = np.zeros(3)
            center[axis_index] = location
            center[cross_axes] = fit["center2d"]
            model = dict(
                axis=axis.tolist(),
                center=center.tolist(),
                radius=fit["radius"],
                source="seccion_original",
                section_location=location,
                profile=record,
            )
            centered_points = original_points[local] - center
            (a, u, v) = basis(axis)

            def residual(theta):
                refined_axis = a + theta[2] * u + theta[3] * v
                refined_axis /= np.linalg.norm(refined_axis)
                delta = centered_points - theta[0] * u - theta[1] * v
                t = delta @ refined_axis
                return (
                    np.linalg.norm(delta - t[:, None] * refined_axis, axis=1) - theta[4]
                )

            solution = least_squares(
                residual,
                [0, 0, 0, 0, model["radius"]],
                bounds=(
                    [-0.1, -0.1, -0.04, -0.04, 0.055],
                    [0.1, 0.1, 0.04, 0.04, 0.55],
                ),
                loss="soft_l1",
                f_scale=0.008,
                max_nfev=60,
            )
            parameters = solution.x
            refined_axis = a + parameters[2] * u + parameters[3] * v
            refined_axis /= np.linalg.norm(refined_axis)
            model["axis"] = refined_axis.tolist()
            model["center"] = (center + parameters[0] * u + parameters[1] * v).tolist()
            model["radius"] = float(parameters[4])
            models.append(model)
    return (models, records)


def validate_and_extend(
    points: np.ndarray,
    normals: np.ndarray,
    original_points,
    original_normals,
    roi,
    model,
    threshold=0.022,
):
    """El ajuste extendido debe conservar perfiles completos en varias secciones separadas."""
    (a, u, v) = basis(model["axis"])
    center = np.asarray(model["center"])
    centered_points = points - center
    t = centered_points @ a
    radial_distances = np.linalg.norm(centered_points - t[:, None] * a, axis=1)
    indices = np.flatnonzero(roi & (abs(radial_distances - model["radius"]) < 0.065))
    if len(indices) < 70:
        return None
    (radial_residuals, normal_alignment, axis_alignment, local_axial_positions, _) = (
        geometry(points[indices], normals[indices], model)
    )
    keep = (radial_residuals < 0.045) & (axis_alignment < 0.4)
    indices = indices[keep]
    if len(indices) < 70:
        return None
    fit_points = points[indices] - center

    def residual(theta):
        refined_axis = a + theta[2] * u + theta[3] * v
        refined_axis /= np.linalg.norm(refined_axis)
        delta = fit_points - theta[0] * u - theta[1] * v
        ts = delta @ refined_axis
        return np.linalg.norm(delta - ts[:, None] * refined_axis, axis=1) - theta[4]

    solution = least_squares(
        residual,
        [0, 0, 0, 0, model["radius"]],
        bounds=(
            [-0.1, -0.1, -0.025, -0.025, model["radius"] - 0.025],
            [0.1, 0.1, 0.025, 0.025, model["radius"] + 0.025],
        ),
        loss="soft_l1",
        f_scale=0.008,
        max_nfev=70,
    )
    parameters = solution.x
    refined_axis = a + parameters[2] * u + parameters[3] * v
    refined_axis /= np.linalg.norm(refined_axis)
    refined_center = center + parameters[0] * u + parameters[1] * v
    model = model.copy()
    model.update(
        axis=refined_axis.tolist(),
        center=refined_center.tolist(),
        radius=float(parameters[4]),
        threshold=threshold,
    )
    (radial_residuals, normal_alignment, axis_alignment, t, _) = geometry(
        points, normals, model
    )
    inlier_indices = np.flatnonzero(
        roi
        & (radial_residuals < threshold)
        & (normal_alignment > 0.84)
        & (axis_alignment < 0.3)
    )
    if len(inlier_indices) < 70:
        return None
    (lower_axial, upper_axial) = np.quantile(t[inlier_indices], [0.01, 0.99])
    length = upper_axial - lower_axial
    if length < 2:
        return None
    tests = []
    radii = []
    (refined_axis, u, v) = basis(model["axis"])
    original_offsets = original_points - refined_center
    original_axial = original_offsets @ refined_axis
    original_cross_section = np.column_stack(
        [original_offsets @ u, original_offsets @ v]
    )
    original_radial = np.linalg.norm(original_cross_section, axis=1)
    for section_position in np.linspace(lower_axial + 0.25, upper_axial - 0.25, 5):
        local = np.flatnonzero(
            (abs(original_axial - section_position) < 0.22)
            & (original_radial < model["radius"] + 0.25)
        )
        if len(local) < 70:
            continue
        cross = original_cross_section[local]
        component_labels = DBSCAN(eps=0.052, min_samples=3, n_jobs=2).fit_predict(cross)
        seed = abs(original_radial[local] - model["radius"]) < threshold
        candidate_labels = component_labels[seed & (component_labels >= 0)]
        if not len(candidate_labels):
            continue
        (component_ids, counts) = np.unique(candidate_labels, return_counts=True)
        best = component_ids[np.argmax(counts)]
        section_points = cross[component_labels == best]
        rad_error = abs(np.linalg.norm(section_points, axis=1) - model["radius"])
        fraction = float((rad_error < threshold).mean())
        profile = circle_profile(section_points)
        arc = 0 if profile is None else profile["arc"]
        pass_test = bool(
            fraction >= 0.86 and np.quantile(rad_error, 0.95) < 0.035 and (arc >= 110)
        )
        tests.append(
            dict(
                t=float(section_position),
                n_points=len(section_points),
                fraction=fraction,
                p95=float(np.quantile(rad_error, 0.95)),
                arc=arc,
                passed=pass_test,
            )
        )
        if profile and pass_test:
            radii.append(profile["radius"])
    passed = sum((t["passed"] for t in tests))
    accepted = bool(len(tests) >= 4 and passed >= 3 and (np.std(radii) < 0.018))
    model.update(
        accepted=accepted,
        profile_tests=tests,
        profile_passed=passed,
        profile_tested=len(tests),
        radius_section_std=float(np.std(radii)) if radii else None,
        length=float(length),
        axial_min=float(lower_axial),
        axial_max=float(upper_axial),
        median_radial_residual=float(np.median(radial_residuals[inlier_indices])),
        p95_radial_residual=float(np.quantile(radial_residuals[inlier_indices], 0.95)),
        n_representatives=len(inlier_indices),
        point_indices=inlier_indices.tolist(),
    )
    return model


def deduplicate(models):
    """Conserva un modelo por eje, radio y soporte geométrico coincidentes."""
    candidates = [
        dict(model, proposal_id=i)
        for (i, model) in enumerate(models)
        if model["accepted"] and model["crop"] == "conductos"
    ]
    kept = []
    decisions = []
    for model in sorted(
        candidates, key=lambda v: (-v["profile_passed"], v["p95_radial_residual"])
    ):
        duplicate = None
        for previous in kept:
            axis = np.array(previous["axis"])
            other_axis = np.array(model["axis"])
            delta = np.array(model["center"]) - np.array(previous["center"])
            offset = np.linalg.norm(delta - axis * (delta @ axis))
            shared = len(
                set(previous["point_indices"]) & set(model["point_indices"])
            ) / max(1, min(len(previous["point_indices"]), len(model["point_indices"])))
            if (
                abs(axis @ other_axis) > 0.9995
                and offset < 0.045
                and (abs(previous["radius"] - model["radius"]) < 0.02)
                and (shared > 0.65)
            ):
                duplicate = previous["proposal_id"]
                break
        decisions.append(
            dict(
                proposal_id=model["proposal_id"],
                duplicate_of=duplicate,
                kept=duplicate is None,
            )
        )
        if duplicate is None:
            kept.append(model)
    return (kept, decisions)


def apply_cylinders(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    models,
    max_gap=0.45,
):
    """Transfiere modelos comprobados sin alterar categorías estructurales protegidas."""
    result = labels.copy()
    result_classes = classes.copy()
    claimed = np.zeros(len(points), bool)
    next_id = int(labels.max()) + 1
    events = []
    model_records = []
    protected = np.isin(classes, [1, 2, 4])
    for model in models:
        (radial_residuals, normal_alignment, axial, axial_positions, _) = geometry(
            points, normals, model
        )
        roi = (
            (points[:, 0] >= -1)
            & (points[:, 0] <= 24)
            & (points[:, 1] >= -8)
            & (points[:, 1] <= 3)
            & (points[:, 2] >= 0.5)
            & (points[:, 2] <= 5.4)
        )
        eligible = (
            roi
            & ~protected
            & ~claimed
            & (axial_positions >= model["axial_min"] - 0.1)
            & (axial_positions <= model["axial_max"] + 0.1)
        )
        chosen = np.flatnonzero(
            eligible
            & (radial_residuals < 0.025)
            & (normal_alignment > 0.75)
            & (axial < 0.4)
        )
        if not len(chosen):
            continue
        absorbed = []
        for segment_id in np.unique(labels[chosen]):
            if segment_id < 0:
                continue
            indices = np.flatnonzero(labels == segment_id)
            if weights[indices].sum() >= 500:
                continue
            if (
                eligible[indices].all()
                and (radial_residuals[indices] < 0.04).all()
                and (np.mean(radial_residuals[indices] < 0.025) >= 0.9)
                and (np.mean(axial[indices] < 0.4) > 0.9)
            ):
                chosen = np.union1d(chosen, indices)
                absorbed.append(int(segment_id))
        order = chosen[np.argsort(axial_positions[chosen])]
        splits = np.r_[
            0, np.flatnonzero(np.diff(axial_positions[order]) > max_gap) + 1, len(order)
        ]
        instance_ids = []
        for start, stop in zip(splits[:-1], splits[1:]):
            indices = order[start:stop]
            if len(indices) < 35 or np.ptp(axial_positions[indices]) < 1.0:
                continue
            segment_id = next_id
            next_id += 1
            result[indices] = segment_id
            result_classes[indices] = 6
            claimed[indices] = True
            instance_ids.append(segment_id)
            (previous_ids, inverse_labels) = np.unique(
                labels[indices], return_inverse=True
            )
            point_counts = np.bincount(inverse_labels, weights=weights[indices])
            rank = np.argsort(point_counts)[::-1]
            events.append(
                dict(
                    new_id=segment_id,
                    proposal_id=model["proposal_id"],
                    original_points=int(weights[indices].sum()),
                    n_previous_regions=int((previous_ids >= 0).sum()),
                    dominant_previous_fraction=float(
                        point_counts[rank[0]] / point_counts.sum()
                    ),
                    previous_regions=[int(s) for s in previous_ids],
                    length=float(np.ptp(axial_positions[indices])),
                    radius=model["radius"],
                    median_radial_residual=float(np.median(radial_residuals[indices])),
                    p95_radial_residual=float(
                        np.quantile(radial_residuals[indices], 0.95)
                    ),
                    absorbed_complete_fragments=absorbed,
                    reason="Eje y radio comunes; perfiles originales en varias posiciones; sin salto axial >0,45 u; normales compatibles",
                )
            )
        record = {k: v for (k, v) in model.items() if k != "point_indices"}
        record["instance_ids"] = instance_ids
        model_records.append(record)
    assert np.array_equal(result[protected], labels[protected])
    assert np.array_equal(result_classes[protected], classes[protected])
    return (result, result_classes, events, model_records)
