"""Compara reglas exploratorias por ID sin modificar las etiquetas automáticas."""

from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/semantic_category_review"
REQUESTED_IDS = {182, 333, 418, 428, 430, 434, 454, 456, 472, 474, 494, 497}
DEVELOPMENT_EXCEPTIONS = {456, 472}
REFERENCE_CATEGORIES = {
    "Persona", "Vehículo", "Tubería o conducto", "Superficie plana", "Viga",
    "Columna o soporte", "Accesorio o conexión de tubería",
}


def prepare_features() -> tuple[pd.DataFrame, float]:
    """Relaciona observaciones existentes y descriptores de puntos originales."""
    segments = pd.read_csv(ROOT / "outputs/segments.csv")
    observations = pd.read_csv(
        ROOT / "outputs/observaciones_visuales_fabrica.csv",
        sep=";", encoding="utf-8-sig",
    ).rename(columns={"instance_id": "segment_id"})
    features = segments.merge(observations, on="segment_id")
    floor = segments[segments.semantic_id == 1]
    floor_z = float(np.average(floor.centroid_z, weights=floor.n_points))
    features["height"] = features.dimension_z
    features["width"] = features[["dimension_x", "dimension_y"]].max(axis=1)
    features["thin"] = features[["dimension_x", "dimension_y"]].min(axis=1)
    features["floor_gap"] = abs(features.bbox_min_z - floor_z)
    features["ratio"] = features.height / np.maximum(features.width, 0.01)
    clear_reference = (
        (features["Confianza preliminar"] == "Alta")
        & features["Categoría visual propuesta"].isin(REFERENCE_CATEGORIES)
    )
    features = features[clear_reference].copy()
    features["split"] = [
        "comprobacion" if (
            (int(segment_id) in REQUESTED_IDS - DEVELOPMENT_EXCEPTIONS)
            or (int(segment_id) % 3 == 0 and int(segment_id) not in REQUESTED_IDS)
        ) else "desarrollo"
        for segment_id in features.segment_id
    ]
    return features, floor_z


def candidate_rules(features: pd.DataFrame, category: str):
    """Genera las rejillas fijas de forma, tamaño y cota aproximada del piso."""
    if category == "Persona":
        for hmin, hmax, width, ratio, linearity, gap in product(
            [0.5, 0.8, 1.0], [1.8, 2.2, 2.6], [0.6, 1.0, 1.5],
            [0.5, 1.0, 1.5], [0.85, 0.97, 1.01], [0.25, 0.5],
        ):
            prediction = (
                (features.height >= hmin) & (features.height <= hmax)
                & (features.width <= width) & (features.ratio >= ratio)
                & (features.linearity < linearity) & (features.floor_gap < gap)
                & (features.n_points >= 100)
            ).to_numpy()
            parameters = dict(
                hmin=hmin, hmax=hmax, width=width, ratio=ratio,
                linearity_max=linearity, floor_gap=gap,
            )
            yield parameters, prediction
    else:
        for length, thin, hmax, gap in product(
            [1.5, 2.0, 3.0], [0.4, 0.8, 1.2], [1.8, 2.3, 2.8], [0.3, 0.6],
        ):
            prediction = (
                (features.width >= length) & (features.width <= 6.5)
                & (features.thin >= thin) & (features.height >= 0.5)
                & (features.height <= hmax) & (features.floor_gap < gap)
                & (features.n_points >= 500)
            ).to_numpy()
            parameters = dict(
                length_min=length, thin_min=thin, hmax=hmax, floor_gap=gap,
            )
            yield parameters, prediction


def main() -> None:
    """Selecciona solo con desarrollo y registra la comprobación separada por ID."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features, floor_z = prepare_features()
    development = (features.split == "desarrollo").to_numpy()
    results, choices, grid_records = [], [], []
    for category in ["Persona", "Vehículo"]:
        positive = (features["Categoría visual propuesta"] == category).to_numpy()
        candidates = []
        for parameters, prediction in candidate_rules(features, category):
            true_positives = int((prediction & positive & development).sum())
            false_positives = int((prediction & ~positive & development).sum())
            key = (false_positives, -true_positives, int((prediction & development).sum()))
            candidates.append((key, parameters, prediction))
            grid_records.append(dict(
                categoria=category, parametros=json.dumps(parameters),
                tp_desarrollo=true_positives, fp_desarrollo=false_positives,
            ))
        nonempty_rules = [item for item in candidates if item[0][1] < 0]
        _, parameters, prediction = min(nonempty_rules or candidates, key=lambda item: item[0])
        for split, mask in [("desarrollo", development), ("comprobacion", ~development)]:
            results.append(dict(
                categoria=category, particion=split,
                positivos=int((positive & mask).sum()),
                negativos=int((~positive & mask).sum()),
                tp=int((prediction & positive & mask).sum()),
                fp=int((prediction & ~positive & mask).sum()),
                fn=int((~prediction & positive & mask).sum()),
                tn=int((~prediction & ~positive & mask).sum()),
                parametros=json.dumps(parameters), configuraciones=len(candidates),
            ))
        for index, row in enumerate(features.itertuples()):
            choices.append(dict(
                segment_id=row.segment_id, categoria=category, particion=row.split,
                referencia_positiva=bool(positive[index]),
                regla_acepta=bool(prediction[index]), clase_automatica=row.semantic,
            ))
    pd.DataFrame(results).to_csv(OUTPUT / "results.csv", index=False)
    pd.DataFrame(choices).to_csv(OUTPUT / "cases.csv", index=False)
    pd.DataFrame(grid_records).to_csv(OUTPUT / "grid_results.csv", index=False)
    features.to_csv(OUTPUT / "features.csv", index=False)
    protocol = dict(
        split="IDs solicitados reservados salvo 456 y 472; otros IDs módulo 3",
        reference="Categorías humanas claras de confianza alta; mezclas excluidas",
        selection="Mínimos FP, después máximos VP, entre reglas con VP de desarrollo",
        selection_scope="Solo desarrollo; comprobación separada por ID, sin identidad física anotada",
        floor_z=floor_z, adopted=False, new_classes=[],
    )
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
