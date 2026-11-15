"""Uniones de fragmentos auditables; proximidad sola nunca autoriza una unión."""

import numpy as np
import pandas as pd


def merge_fragments(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    region_descriptors,
    indices_by_segment,
    pairs,
    threshold=200,
):
    """Une contactos solo si su forma conjunta satisface los criterios seleccionados."""
    descriptors_by_id = {r["segment_id"]: r for r in region_descriptors}
    parents = {k: k for k in indices_by_segment if k >= 0}
    member_ids = {k: [k] for k in parents}
    group_indices = {k: indices_by_segment[k].copy() for k in parents}
    group_sizes = {k: descriptors_by_id[k]["n_points"] for k in parents}
    decisions = []

    def find_root(current_segment):
        while parents[current_segment] != current_segment:
            parents[current_segment] = parents[parents[current_segment]]
            current_segment = parents[current_segment]
        return current_segment

    candidates = pairs[pairs.smaller_n_points < threshold].sort_values(
        ["min_distance", "contact_edges"], ascending=[True, False]
    )
    for edge in candidates.itertuples():
        (root_a, root_b) = (find_root(int(edge.a)), find_root(int(edge.b)))
        reason = None
        accepted = False
        kind = ""
        if root_a == root_b:
            continue
        (region_a, region_b) = (
            descriptors_by_id[int(edge.a)],
            descriptors_by_id[int(edge.b)],
        )
        if region_a["semantic_id"] in [1, 2, 4] or region_b["semantic_id"] in [1, 2, 4]:
            reason = "estructura_protegida"
        elif group_sizes[root_a] + group_sizes[root_b] > 2500:
            reason = "limite_extension_del_grupo"
        elif min(region_a["center"][1], region_b["center"][1]) >= 4:
            reason = "fuera_red_de_conductos"
        elif edge.contact_edges < 4 or edge.min_distance > 0.115:
            reason = "contacto_insuficiente"
        elif edge.color_median > 0.2:
            reason = "color_discontinuo"
        elif edge.normal_dot_median < 0.97 or edge.tangent_median > 0.025:
            reason = "normales_o_tangencia_incompatibles"
        elif edge.plane_normal_angle_deg > 10:
            reason = "orientacion_de_plano_distinta"
        else:
            indices = np.r_[group_indices[root_a], group_indices[root_b]]
            region_points = points[indices]
            center = region_points.mean(0)
            delta = region_points - center
            (eigenvalues, eigenvectors) = np.linalg.eigh(
                delta.T @ delta / len(region_points)
            )
            normal = eigenvectors[:, 0]
            plane_distances = abs(delta @ normal)
            linearity = (eigenvalues[2] - eigenvalues[1]) / max(eigenvalues[2], 1e-12)
            length = np.ptp(delta @ eigenvectors[:, -1])
            coherence = np.mean(abs(normals[indices] @ normal) > 0.97)
            if (
                np.quantile(plane_distances, 0.95) > 0.025
                or np.sqrt(eigenvalues[0]) > 0.016
                or coherence < 0.9
            ):
                reason = "modelo_conjunto_no_plano"
            elif min(region_a["linearity"], region_b["linearity"]) > 0.85 and (
                edge.axis_angle_deg > 12 or edge.axis_center_offset > 0.085
            ):
                reason = "ejes_paralelos_distintos_o_esquina"
            elif region_a["semantic_id"] != region_b["semantic_id"] and set(
                [region_a["semantic_id"], region_b["semantic_id"]]
            ) not in [{0, 3}, {0, 5}]:
                reason = "clases_incompatibles"
            else:
                accepted = True
                reason = "contacto_y_modelo_geometrico_conjunto"
                kind = (
                    "plano_continuo" if linearity < 0.95 else "franja_colineal_continua"
                )
                root = min(root_a, root_b)
                other = max(root_a, root_b)
                parents[other] = root
                group_indices[root] = indices
                member_ids[root] = member_ids[root_a] + member_ids[root_b]
                group_sizes[root] = group_sizes[root_a] + group_sizes[root_b]
        decisions.append(
            dict(
                a=int(edge.a),
                b=int(edge.b),
                current_a=root_a,
                current_b=root_b,
                accepted=accepted,
                reason=reason,
                kind=kind,
                min_distance=edge.min_distance,
                contact_edges=edge.contact_edges,
                normal_dot=edge.normal_dot_median,
                color_delta=edge.color_median,
                axis_angle=edge.axis_angle_deg,
                smaller_points=edge.smaller_n_points,
            )
        )
    result = labels.copy()
    for segment_id in parents:
        result[indices_by_segment[segment_id]] = find_root(segment_id)
    changes = [
        dict(
            target=root,
            sources=sorted(set(member_ids[root])),
            n_points=group_sizes[root],
        )
        for root in parents
        if find_root(root) == root and len(member_ids[root]) > 1
    ]
    return (result, pd.DataFrame(decisions), changes)
