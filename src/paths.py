"""Rutas del proyecto, independientes del directorio de ejecución."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "FARO_Scan_Full_Sub1_down3_with_normals.ply"
OUTPUTS = ROOT / "outputs"
CACHE = OUTPUTS / "cache"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
EXPERIMENTS = OUTPUTS / "experiments"
TEMP = ROOT / "temp"


def ensure_directories() -> None:
    """Prepara las carpetas de resultados y trabajo regenerable."""
    for directory in (CACHE, FIGURES, METRICS, EXPERIMENTS, TEMP):
        directory.mkdir(parents=True, exist_ok=True)
