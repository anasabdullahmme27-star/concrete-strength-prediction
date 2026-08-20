"""Feature definitions, model registry and mix presets.

Everything the interface needs about the dataset was baked into
`artifacts/app_meta.json` by the build script, so this app never opens a
spreadsheet and boots in well under a second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
META_PATH = APP_ROOT / "artifacts" / "app_meta.json"
STATS_PATH = DATA_DIR / "stats.json"


@lru_cache(maxsize=1)
def meta() -> dict:
    with open(META_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Feature:
    key: str
    column: str      # exact column name the fitted scalers expect
    label: str
    unit: str
    icon: str
    step: float
    decimals: int


CEMENT = "Cement (component 1)(kg in a m^3 mixture)"
SLAG = "Blast Furnace Slag (component 2)(kg in a m^3 mixture)"
FLY_ASH = "Fly Ash (component 3)(kg in a m^3 mixture)"
WATER = "Water  (component 4)(kg in a m^3 mixture)"
SUPERPLASTICIZER = "Superplasticizer (component 5)(kg in a m^3 mixture)"
COARSE_AGG = "Coarse Aggregate  (component 6)(kg in a m^3 mixture)"
FINE_AGG = "Fine Aggregate (component 7)(kg in a m^3 mixture)"
AGE = "Age (day)"
TEMPERATURE = "Curing Temperature (Celsius)"

FEATURES: tuple[Feature, ...] = (
    Feature("cement", CEMENT, "Cement", "kg/m³", "▦", 5.0, 0),
    Feature("slag", SLAG, "Slag", "kg/m³", "◈", 5.0, 0),
    Feature("fly_ash", FLY_ASH, "Fly Ash", "kg/m³", "✦", 5.0, 0),
    Feature("water", WATER, "Water", "kg/m³", "💧", 2.0, 0),
    Feature("superplasticizer", SUPERPLASTICIZER, "Superplasticizer", "kg/m³", "🧪", 0.5, 1),
    Feature("coarse_agg", COARSE_AGG, "Coarse Aggregate", "kg/m³", "◍", 10.0, 0),
    Feature("fine_agg", FINE_AGG, "Fine Aggregate", "kg/m³", "◎", 10.0, 0),
    Feature("age", AGE, "Age", "days", "🗓", 1.0, 0),
    Feature("temperature", TEMPERATURE, "Curing Temperature", "°C", "🌡", 0.5, 1),
)

FEATURES_BY_KEY = {f.key: f for f in FEATURES}

NO_TEMP_COLUMNS = [f.column for f in FEATURES if f.key != "temperature"]
WITH_TEMP_COLUMNS = [f.column for f in FEATURES]

FAMILIES = {
    "with_temp": {"label": "With curing temperature", "inputs": 9},
    "no_temp": {"label": "Without curing temperature", "inputs": 8},
}


def feature_range(feature: Feature) -> tuple[float, float]:
    entry = meta()["features"][feature.key]
    return float(entry["min"]), float(entry["max"])


def feature_median(feature: Feature) -> float:
    return float(meta()["features"][feature.key]["median"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    model_path: str
    scaler_path: str
    r2: float
    rmse: float
    mae: float
    is_champion: bool
    requires_package: str | None

    @property
    def columns(self) -> list[str]:
        return WITH_TEMP_COLUMNS if self.family == "with_temp" else NO_TEMP_COLUMNS


@lru_cache(maxsize=1)
def model_registry() -> dict[str, ModelSpec]:
    registry: dict[str, ModelSpec] = {}
    for key, entry in meta()["models"].items():
        registry[key] = ModelSpec(
            key=key,
            name=entry["name"],
            family=entry["family"],
            model_path=entry["model_path"],
            scaler_path=entry["scaler_path"],
            r2=entry["r2"],
            rmse=entry["rmse"],
            mae=entry["mae"],
            is_champion=entry["is_champion"],
            requires_package=entry.get("requires_package"),
        )
    return registry


def models_for_family(family: str) -> list[ModelSpec]:
    ordered = sorted(
        (m for m in model_registry().values() if m.family == family),
        key=lambda m: m.r2,
        reverse=True,
    )
    return ordered


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, float]] = {
    "High Performance": {
        CEMENT: 450, SLAG: 100, FLY_ASH: 0, WATER: 160, SUPERPLASTICIZER: 8,
        COARSE_AGG: 1000, FINE_AGG: 700, AGE: 28, TEMPERATURE: 22,
    },
    "Standard C25": {
        CEMENT: 300, SLAG: 0, FLY_ASH: 0, WATER: 180, SUPERPLASTICIZER: 3,
        COARSE_AGG: 1000, FINE_AGG: 780, AGE: 28, TEMPERATURE: 23,
    },
    "Early Age (7d)": {
        CEMENT: 350, SLAG: 0, FLY_ASH: 0, WATER: 175, SUPERPLASTICIZER: 5,
        COARSE_AGG: 1000, FINE_AGG: 760, AGE: 7, TEMPERATURE: 25,
    },
    "Plain C15": {
        CEMENT: 200, SLAG: 0, FLY_ASH: 0, WATER: 190, SUPERPLASTICIZER: 0,
        COARSE_AGG: 1000, FINE_AGG: 800, AGE: 28, TEMPERATURE: 24,
    },
    "Slag Rich": {
        CEMENT: 200, SLAG: 200, FLY_ASH: 0, WATER: 175, SUPERPLASTICIZER: 5,
        COARSE_AGG: 950, FINE_AGG: 780, AGE: 56, TEMPERATURE: 23,
    },
    "High Fly Ash": {
        CEMENT: 220, SLAG: 0, FLY_ASH: 150, WATER: 175, SUPERPLASTICIZER: 6,
        COARSE_AGG: 950, FINE_AGG: 790, AGE: 56, TEMPERATURE: 24,
    },
}


# ---------------------------------------------------------------------------
# Strength classes (EN 206 / Eurocode 2)
# ---------------------------------------------------------------------------
STRENGTH_CLASSES: tuple[tuple[float, str, str], ...] = (
    (0, "Below C12/15", "Non-structural — very low strength"),
    (12, "C12/15", "Blinding and non-structural fill"),
    (16, "C16/20", "Light structural, dry conditions"),
    (20, "C20/25", "General reinforced concrete"),
    (25, "C25/30", "Standard structural — slabs, beams, columns"),
    (30, "C30/37", "Exposed elements, bridge decks"),
    (35, "C35/45", "High-durability structural work"),
    (40, "C40/50", "Heavily loaded columns, precast"),
    (45, "C45/55", "High-performance concrete"),
    (50, "C50/60", "High-strength concrete"),
    (55, "C55+ — Ultra-High Performance", "Special structures, nuclear, offshore"),
)


def strength_class(mpa: float) -> tuple[str, str]:
    chosen = STRENGTH_CLASSES[0]
    for entry in STRENGTH_CLASSES:
        if mpa >= entry[0]:
            chosen = entry
    return chosen[1], chosen[2]
