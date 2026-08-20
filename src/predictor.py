"""Model loading and prediction."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import joblib
import pandas as pd
import streamlit as st

from src import config


@lru_cache(maxsize=None)
def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def available_models(family: str) -> list[config.ModelSpec]:
    """Registered models whose file and optional dependency are both present."""
    usable = []
    for spec in config.models_for_family(family):
        if spec.requires_package and not _package_available(spec.requires_package):
            continue
        if not (config.APP_ROOT / spec.model_path).exists():
            continue
        usable.append(spec)
    return usable


@st.cache_resource(show_spinner=False)
def _load(path: str):
    return joblib.load(config.APP_ROOT / path)


def predict(spec: config.ModelSpec, values: dict[str, float]) -> float:
    """Predicted compressive strength in MPa for one raw, unscaled mix."""
    columns = spec.columns
    frame = pd.DataFrame([[values[c] for c in columns]], columns=columns)

    scaler = _load(spec.scaler_path)
    scaled = pd.DataFrame(scaler.transform(frame), columns=columns)

    model = _load(spec.model_path)
    return float(model.predict(scaled)[0])


def indicators(values: dict[str, float]) -> dict[str, float]:
    """The mix-design ratios shown on the result panel."""
    cement = values.get(config.CEMENT, 0.0)
    slag = values.get(config.SLAG, 0.0)
    fly_ash = values.get(config.FLY_ASH, 0.0)
    water = values.get(config.WATER, 0.0)
    coarse = values.get(config.COARSE_AGG, 0.0)
    fine = values.get(config.FINE_AGG, 0.0)

    binder = cement + slag + fly_ash
    paste = binder + water

    return {
        "wc_ratio": water / cement if cement else float("nan"),
        "total_binder": binder,
        "binder_ratio": binder / paste if paste else float("nan"),
        "total_aggregate": coarse + fine,
        "paste_volume": paste / 2400.0,
    }


def out_of_range(values: dict[str, float], columns: list[str]) -> list[str]:
    """Labels of the inputs that fall outside the training range."""
    flagged = []
    for feature in config.FEATURES:
        if feature.column not in columns:
            continue
        low, high = config.feature_range(feature)
        value = values.get(feature.column)
        if value is not None and (value < low or value > high):
            flagged.append(feature.label)
    return flagged
