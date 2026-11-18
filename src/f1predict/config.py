"""Project paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

FIRST_SEASON = 2014  # start of the V6 turbo-hybrid era
LAST_SEASON = 2026
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]

# Constructor renames: map every Jolpica constructorId to one stable "lineage" so rolling
# team form does not reset when a team rebrands.
CONSTRUCTOR_LINEAGE = {
    "toro_rosso": "racing_bulls",
    "alphatauri": "racing_bulls",
    "rb": "racing_bulls",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    "lotus_f1": "alpine",
    "renault": "alpine",
    "alpine": "alpine",
    "sauber": "audi",
    "alfa": "audi",
    "audi": "audi",
    "marussia": "manor",
    "manor": "manor",
}

# Circuits run fully or largely on public roads / temporary layouts.
STREET_CIRCUITS = {
    "monaco",
    "baku",
    "marina_bay",
    "jeddah",
    "vegas",
    "miami",
    "albert_park",
    "villeneuve",
    "sochi",
    "madring",
}
