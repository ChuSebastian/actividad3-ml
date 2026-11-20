"""Consistencia semántica con geometría verificable; no modifica instancias."""

import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.region_geometry import region_descriptors
from src.cylinders import geometry
from src.classification import NAMES


def describe(points, normals, colors, labels, classes, weights):
    rows = region_descriptors(points, labels)
    groups = {row["id"]: np.flatnonzero(labels == row["id"]) for row in rows}
    for row in rows:
        indices = groups[row["id"]]
        centered = points[indices] - points[indices].mean(0)
        values, vectors = np.linalg.eigh(np.cov(centered.T))
        spans = np.diff(np.quantile(centered @ vectors, [0.02, 0.98], axis=0), axis=0)[
            0
        ]
        row.update(
            class_id=int(classes[indices[0]]),
            n_points=int(weights[indices].sum()),
            rgb=np.median(colors[indices], axis=0),
            cross_width=float(max(spans[:2])),
            length=float(spans[2]),
            side_normal_fraction=float((abs(normals[indices, 2]) < 0.3).mean()),
        )
    return rows, groups


def consistency(points, normals, colors, labels, classes, weights, models):
    """Aplica reglas sobre regiones completas y conserva los casos ambiguos."""
    rows, groups = describe(points, normals, colors, labels, classes, weights)
    result = classes.copy()
    changes, audits, rejected = [], [], []
    anchors = [
        row
        for row in rows
        if row["class_id"] == 4 and row["dim"][2] > 5 and min(row["dim"][:2]) > 0.3
    ]
    upper_anchors = [
        row for row in anchors if row["center"][1] > 4 and row["hi"][2] > 7.5
    ]
    linear = [
        row
        for row in rows
        if row["class_id"] == 5
        and abs(row["axis"][2]) < 0.2
        and row["linearity"] > 0.97
        and np.linalg.norm(row["dim"]) > 3.5
    ]
    trees = {row["id"]: cKDTree(points[groups[row["id"]]]) for row in anchors + linear}
    for row in rows:
        sid = row["id"]
        indices = groups[sid]
        original_class = row["class_id"]
        audit = dict(
            segment_id=sid,
            n_points=row["n_points"],
            class_before=original_class,
            height=float(row["dim"][2]),
            width=float(max(row["dim"][:2])),
            vertical_alignment=float(abs(row["axis"][2])),
            linearity=float(row["linearity"]),
            cross_width=row["cross_width"],
            length=row["length"],
        )
        audit.update(
            n_representatives=len(indices),
            side_normal_fraction=row["side_normal_fraction"],
            median_red=float(row["rgb"][0]),
            median_green=float(row["rgb"][1]),
            median_blue=float(row["rgb"][2]),
            fails_linear_length=bool(np.linalg.norm(row["dim"]) <= 1.5),
            fails_planar_coherence=bool(
                np.linalg.eigvalsh(
                    normals[indices].T @ normals[indices] / len(indices)
                )[-1]
                <= 0.97
            ),
            fails_support_height=bool(row["dim"][2] <= 3.2),
            fails_support_width=bool(max(row["dim"][:2]) >= 1.3),
            fails_support_axis=bool(abs(row["axis"][2]) <= 0.97),
            fails_support_linearity=bool(row["linearity"] <= 0.86),
            fails_linear_linearity=bool(row["linearity"] <= 0.91),
            fails_linear_bbox=bool(np.partition(row["dim"], 1)[1] >= 0.85),
            insufficient_planar_support=bool(len(indices) < 100),
            normal_coherence=float(
                np.linalg.eigvalsh(
                    normals[indices].T @ normals[indices] / len(indices)
                )[-1]
            ),
        )
        for prefix, values in [
            ("robust_min", row["lo"]),
            ("robust_max", row["hi"]),
            ("axis", row["axis"]),
        ]:
            audit.update(
                {f"{prefix}_{name}": float(value) for name, value in zip("xyz", values)}
            )
        audits.append(audit)
        if original_class != 0:
            continue
        target, anchor, rule, reason = None, None, None, None
        # Plantilla estructural: no basta ser alto. Se requiere base, altura,
        # grosor, orientación y coincidencia de la retícula de soportes existentes.
        if original_class == 0 and upper_anchors:
            grid_x = min(abs(row["center"][0] - a["center"][0]) for a in upper_anchors)
            grid_y = min(abs(row["center"][1] - a["center"][1]) for a in upper_anchors)
            if (
                row["dim"][2] > 7
                and 0.3 < min(row["dim"][:2])
                and max(row["dim"][:2]) < 1.45
                and abs(row["axis"][2]) > 0.995
                and row["linearity"] > 0.95
                and row["side_normal_fraction"] > 0.9
                and row["lo"][2] < -0.8
                and row["hi"][2] > 7.5
                and row["center"][1] > 4
                and grid_x < 0.6
                and grid_y < 0.4
            ):
                target, rule = 4, "support_structural_template"
                reason = (
                    f'Soporte candidato: H={row["dim"][2]:.3f}, ancho={max(row["dim"][:2]):.3f}<1,45 u, '
                    f'L={row["linearity"]:.4f}, eje vertical={abs(row["axis"][2]):.4f}; '
                    f"base baja, cota superior y retícula de soportes (dx={grid_x:.3f}, dy={grid_y:.3f} u)."
                )
        if (
            target is None
            and original_class in [0, 3]
            and row["dim"][2] > 0.5
            and abs(row["axis"][2]) > 0.98
            and row["linearity"] > 0.8
            and row["side_normal_fraction"] > 0.8
        ):
            for reference in anchors:
                if not (
                    np.all(row["lo"][:2] >= reference["lo"][:2] - 0.12)
                    and np.all(row["hi"][:2] <= reference["hi"][:2] + 0.12)
                ):
                    continue
                if (
                    row["lo"][2] < reference["lo"][2] - 0.2
                    or row["hi"][2] > reference["hi"][2] + 0.2
                ):
                    continue
                distances, _ = trees[reference["id"]].query(points[indices], k=1)
                color_delta = np.linalg.norm(row["rgb"] - reference["rgb"])
                accepted = bool(
                    (distances < 0.18).sum() >= 5
                    and distances.min() < 0.15
                    and color_delta < 0.25
                )
                rejected.append(
                    dict(
                        segment_id=sid,
                        reference_id=reference["id"],
                        test="support_contact",
                        accepted=accepted,
                        min_distance=float(distances.min()),
                        color_delta=float(color_delta),
                    )
                )
                if accepted:
                    target, anchor, rule = (
                        4,
                        reference["id"],
                        "support_connected_fragment",
                    )
                    reason = f"Fragmento vertical dentro de la envolvente del soporte {anchor}; al menos 5 contactos <0,18 u, separación mínima {distances.min():.4f} u; orientación, normales y RGB compatibles."
                    break
        if (
            target is None
            and original_class in [0, 3]
            and len(indices) >= 40
            and row["linearity"] > 0.92
            and abs(row["axis"][2]) < 0.2
            and np.linalg.norm(row["dim"]) > 0.6
            and row["lo"][2] > 0
        ):
            for reference in linear:
                axis = reference["axis"]
                delta = row["center"] - reference["center"]
                offset = np.linalg.norm(delta - axis * (delta @ axis))
                if abs(axis @ row["axis"]) < 0.985 or offset > 0.25:
                    continue
                distances, _ = trees[reference["id"]].query(points[indices], k=1)
                if (
                    distances.min() > 0.16
                    or (distances < 0.18).sum() < 5
                    or np.linalg.norm(row["rgb"] - reference["rgb"]) > 0.25
                ):
                    continue
                joint = points[np.r_[indices, groups[reference["id"]]]]
                joint -= joint.mean(0)
                values, vectors = np.linalg.eigh(np.cov(joint.T))
                width = max(
                    np.diff(np.quantile(joint @ vectors, [0.02, 0.98], axis=0), axis=0)[
                        0
                    ][:2]
                )
                if (values[2] - values[1]) / values[2] < 0.96 or width > 0.85:
                    continue
                target, anchor, rule = 5, reference["id"], "linear_connected_fragment"
                reason = f"Continuidad con elemento lineal {anchor}: ejes compatibles, offset={offset:.4f} u, contacto y RGB; PCA conjunta L>0,96 y anchura transversal {width:.4f}<0,85 u. No se infiere función de viga."
                break
        # Solo regiones enteras que pertenezcan a un cilindro ya comprobado.
        if target is None and original_class in [0, 3, 5] and len(indices) >= 10:
            for model in models:
                residual, align, axial, positions, _ = geometry(
                    points[indices], normals[indices], model
                )
                if not (
                    positions.min() >= model["axial_min"] - 0.1
                    and positions.max() <= model["axial_max"] + 0.1
                ):
                    continue
                if (
                    np.quantile(residual, 0.95) < 0.035
                    and (align > 0.65).mean() > 0.9
                    and (axial < 0.4).mean() > 0.9
                ):
                    target, anchor, rule = (
                        6,
                        model["instance_ids"][0],
                        "cylinder_whole_fragment",
                    )
                    reason = f"Región completa compatible con cilindro comprobado {anchor}: residuo radial p95={np.quantile(residual,.95):.5f}<0,035 u, intervalo axial y normales compatibles. Se conserva el control de perfil original del modelo."
                    break
        if target is not None and target != original_class:
            result[indices] = target
            changes.append(
                dict(
                    segment_id=sid,
                    before_class=original_class,
                    after_class=target,
                    n_points=row["n_points"],
                    reference_id=anchor,
                    rule=rule,
                    justification=reason,
                )
            )
    return result, pd.DataFrame(changes), pd.DataFrame(audits), pd.DataFrame(rejected)


def review_candidates(table, changes):
    """Selecciona envolventes para revisión; las categorías propuestas no son etiquetas."""
    rows = []
    changed = set(changes.segment_id) if len(changes) else set()
    for record in table[table.segment_id >= 0].to_dict("records"):
        sid = record["segment_id"]
        problems = []
        if sid in changed:
            problems.append(
                "Revisar reclasificación automática y posibles accesorios incluidos"
            )
        if (
            record["semantic_id"] == 0
            and record["dimension_z"] > 2
            and record["pca_vertical_alignment"] > 0.95
        ):
            problems.append(
                "Soporte posible: vertical pero no satisface todas las reglas; puede incluir conexiones"
            )
        if (
            record["semantic_id"] in [0, 3]
            and record["linearity"] > 0.9
            and max(record["dimension_x"], record["dimension_y"]) > 2
        ):
            problems.append(
                "Elemento alargado con etiqueta ambigua; comprobar continuidad y sección"
            )
        # Solo propuesta para inspección: postura y función no se deducen de una caja.
        if (
            record["semantic_id"] in [0, 5]
            and record["n_points"] >= 100
            and 0.5 < record["dimension_z"] < 2.2
            and 0.25 < max(record["dimension_x"], record["dimension_y"]) < 1.8
            and max(record["dimension_x"], record["dimension_y"], record["dimension_z"])
            < 2.3
            and -1.9 < record["bbox_min_z"] < -0.7
        ):
            problems.append(
                "Envolvente compatible con figura humana compacta o equipo; persona no confirmada, incluida postura sentada"
            )
        if (
            record["semantic_id"] == 0
            and record["n_points"] > 1000
            and record["linearity"] < 0.85
        ):
            problems.append(
                "Conjunto complejo: maquinaria, válvula u otra estructura sin función identificada"
            )
        if problems:
            record.update(
                possible_problem="; ".join(problems),
                review_reason="Geometría y posición requieren interpretación humana independiente",
                human_label=None,
                reviewed=False,
            )
            rows.append(record)
    return pd.DataFrame(rows)


def save_analysis(original, colors, working):
    """Guarda diagnóstico, controles, candidatos y recortes sin anotación humana."""
    from src.paths import CACHE, METRICS, OUTPUTS
    from src.evaluation import metrics
    from src.visualization import plot_semantic_comparisons

    start = time.perf_counter()
    labels = np.load(CACHE / "final_labels.npy")
    baseline = np.load(CACHE / "classes_before_consistency.npy")
    models = json.loads((METRICS / "cylinder_models.json").read_text())
    corrected, changes, audit, controls = consistency(
        working["xyz"],
        working["normals"],
        working["rgb"],
        labels,
        baseline,
        working["counts"],
        models,
    )
    changes.to_csv(METRICS / "semantic_reclassifications.csv", index=False)
    audit.to_csv(METRICS / "semantic_rule_audit.csv", index=False)
    controls.to_csv(METRICS / "semantic_contact_controls.csv", index=False)
    unknown = audit[audit.class_before == 0]
    failures = []
    for column in [
        name
        for name in audit
        if name.startswith("fails_") or name.startswith("insufficient_")
    ]:
        selected = unknown[unknown[column]]
        failures.append(
            dict(
                criterion=column,
                n_regions=len(selected),
                n_points=int(selected.n_points.sum()),
                interpretation="Criterio actual no satisfecho; conteos superpuestos, no prueba de error",
            )
        )
    pd.DataFrame(failures).to_csv(METRICS / "unknown_rule_failures.csv", index=False)
    table = pd.read_csv(OUTPUTS / "segments.csv")
    candidates = review_candidates(table, changes)
    rejected_models = json.loads((CACHE / "cylinder_proposals.json").read_text())
    for model in rejected_models:
        if model["accepted"] or not model.get("point_indices"):
            continue
        ids, frequencies = np.unique(
            labels[np.array(model["point_indices"])], return_counts=True
        )
        sid = int(ids[np.argmax(frequencies)])
        record = table[table.segment_id == sid]
        if sid < 0 or record.empty:
            continue
        row = record.iloc[0].to_dict()
        row.update(
            possible_problem="Perfil cilíndrico propuesto pero no aceptado",
            review_reason=f"Pasan {model['profile_passed']}/{model['profile_tested']} secciones; no se asigna clase cilíndrica",
            human_label=None,
            reviewed=False,
        )
        candidates = pd.concat([candidates, pd.DataFrame([row])], ignore_index=True)
    # Una región aparece una sola vez, aunque tenga varias hipótesis de revisión.
    candidates = candidates.groupby("segment_id", sort=True, as_index=False).agg(
        {
            column: (
                (lambda values: "; ".join(dict.fromkeys(values.dropna().astype(str))))
                if column in ["possible_problem", "review_reason"]
                else "first"
            )
            for column in candidates
            if column != "segment_id"
        }
    )
    candidates.to_csv(OUTPUTS / "review_candidates.csv", index=False)
    class_rows = [
        dict(
            stage=stage,
            class_id=class_id,
            semantic=NAMES[class_id],
            n_points=int(working["counts"][cls == class_id].sum()),
        )
        for stage, cls in [
            ("before_consistency", baseline),
            ("after_consistency", corrected),
        ]
        for class_id in range(len(NAMES))
    ]
    pd.DataFrame(class_rows).to_csv(
        METRICS / "semantic_class_comparison.csv", index=False
    )
    size_rows = []
    for low, high in [
        (0, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, float("inf")),
    ]:
        selected_unknown = unknown[
            (unknown.n_points >= low) & (unknown.n_points < high)
        ]
        size_rows.append(
            dict(
                min_points=low,
                max_points=None if high == float("inf") else high,
                n_regions=len(selected_unknown),
                n_points=int(selected_unknown.n_points.sum()),
            )
        )
    pd.DataFrame(size_rows).to_csv(
        METRICS / "unknown_size_distribution.csv", index=False
    )
    stages_path = METRICS / "refinement_stages.csv"
    stages = pd.read_csv(stages_path)
    stages = stages[stages.stage != "semantic_consistency"]
    final_stage = dict(
        stage="semantic_consistency",
        **metrics(labels, working["counts"]),
        semantic_unknown_points=int(working["counts"][corrected == 0].sum()),
        semantic_unknown_pct=100
        * float(working["counts"][corrected == 0].sum() / working["counts"].sum()),
    )
    sizes = table[table.segment_id >= 0].n_points
    for threshold in [100, 200, 500]:
        final_stage[f"regions_lt{threshold}"] = int((sizes < threshold).sum())
        final_stage[f"points_in_regions_lt{threshold}"] = int(
            sizes[sizes < threshold].sum()
        )
    pd.concat([stages, pd.DataFrame([final_stage])], ignore_index=True).to_csv(
        stages_path, index=False
    )
    save_cylinder_checks(working, labels, baseline, models)
    summary = []
    for stage, cls in [
        ("before_consistency", baseline),
        ("after_consistency", corrected),
    ]:
        summary.append(
            dict(
                stage=stage,
                n_regions=len(np.unique(labels[labels >= 0])),
                instance_coverage_pct=100
                * float(working["counts"][labels >= 0].sum() / working["counts"].sum()),
                unknown_points=int(working["counts"][cls == 0].sum()),
                reclassified_regions=len(changes) if stage.startswith("after") else 0,
                changed_class_points=int(working["counts"][cls != baseline].sum()),
                split_regions=0,
                merged_regions=0,
            )
        )
    pd.DataFrame(summary).to_csv(METRICS / "semantic_comparison.csv", index=False)
    # Controles negativos sobre cajas complejas y elementos altos fuera de plantilla.
    negatives = audit[
        (audit.class_before == 0)
        & (audit.height > 2)
        & ((audit.width > 1.45) | (audit.linearity < 0.8))
    ].copy()
    negatives["reclassified"] = negatives.segment_id.isin(set(changes.segment_id))
    assert not negatives.reclassified.any()
    negatives.to_csv(METRICS / "semantic_negative_controls.csv", index=False)
    plot_semantic_comparisons(
        original, colors, working, labels, baseline, corrected, candidates
    )
    parameters = dict(
        support_width=1.45,
        support_min_height=7,
        support_min_linearity=0.95,
        support_axis=0.995,
        grid_dx=0.6,
        grid_dy=0.4,
        fragment_distance=0.15,
        min_contact_count=5,
        contact_radius=0.18,
        color_delta=0.25,
        linear_axis_dot=0.985,
        linear_offset=0.25,
        joint_linearity=0.96,
        joint_width=0.85,
        instance_modifications=False,
        scores_are_probabilities=False,
        human_review_performed=False,
    )
    parameters.update(
        only_unknown=True,
        support_min_thickness=0.3,
        support_max_base_z=-0.8,
        support_min_top_z=7.5,
        support_side_normals=0.9,
        support_grid_min_y=4,
        anchor_support_height=5,
        anchor_linear_length=3.5,
        anchor_linear_linearity=0.97,
        fragment_min_height=0.5,
        fragment_axis=0.98,
        fragment_linearity=0.8,
        fragment_side_normals=0.8,
        fragment_bbox_margin_xy=0.12,
        fragment_bbox_margin_z=0.2,
        linear_min_representatives=40,
        linear_min_linearity=0.92,
        linear_max_abs_axis_z=0.2,
        linear_min_extent=0.6,
        linear_min_z=0,
        linear_min_distance=0.16,
        cylinder_min_representatives=10,
        cylinder_max_p95=0.035,
        cylinder_min_radial_dot=0.65,
        cylinder_max_axis_dot=0.4,
        cylinder_min_fraction=0.9,
        cylinder_axial_margin=0.1,
        propagation_iterations=1,
    )
    (METRICS / "semantic_consistency_parameters.json").write_text(
        json.dumps(parameters, indent=2)
    )
    parameter_path = METRICS / "parameters.json"
    all_parameters = json.loads(parameter_path.read_text())
    all_parameters["semantic_consistency"] = parameters
    parameter_path.write_text(json.dumps(all_parameters, indent=2))
    run = dict(
        analysis_and_figures_seconds=time.perf_counter() - start,
        reviewed_by_humans=False,
        n_review_candidates=len(candidates),
        n_negative_controls=len(negatives),
        n_preserved_previously_known_points=int(working["counts"][baseline > 0].sum()),
    )
    (METRICS / "semantic_analysis.json").write_text(json.dumps(run, indent=2))


def save_cylinder_checks(working, labels, classes, models):
    """Registra por qué no se extiende un cilindro a regiones completas próximas."""
    from src.paths import METRICS

    records = []
    for sid in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == sid)
        class_id = int(classes[indices[0]])
        if class_id not in [0, 3, 5] or len(indices) < 10:
            continue
        for model in models:
            residual, alignment, axial_alignment, positions, _ = geometry(
                working["xyz"][indices], working["normals"][indices], model
            )
            overlap = (
                positions.max() >= model["axial_min"]
                and positions.min() <= model["axial_max"]
            )
            if not overlap or np.median(residual) > 0.5:
                continue
            axial_ok = bool(
                positions.min() >= model["axial_min"] - 0.1
                and positions.max() <= model["axial_max"] + 0.1
            )
            p95 = float(np.quantile(residual, 0.95))
            radial_fraction = float((alignment > 0.65).mean())
            tangent_fraction = float((axial_alignment < 0.4).mean())
            geometry_ok = (
                axial_ok
                and p95 < 0.035
                and radial_fraction > 0.9
                and tangent_fraction > 0.9
            )
            records.append(
                dict(
                    segment_id=int(sid),
                    class_before=class_id,
                    reference_id=model["instance_ids"][0],
                    n_points=int(working["counts"][indices].sum()),
                    p95_radial=p95,
                    radial_normals_fraction=radial_fraction,
                    tangent_normals_fraction=tangent_fraction,
                    axial_interval_ok=axial_ok,
                    whole_region_geometry_ok=geometry_ok,
                    automatically_accepted=bool(geometry_ok and class_id == 0),
                )
            )
    pd.DataFrame(records).to_csv(METRICS / "cylinder_extension_audit.csv", index=False)
