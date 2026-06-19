from __future__ import annotations

"""Small helpers and configuration readers for PortOffsetCollect."""

import os
import numpy as np
from pathlib import Path

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}

def _env_mm(name: str, default_mm: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default_mm / 1000.0
    try:
        return float(value) / 1000.0
    except ValueError:
        return default_mm / 1000.0

def _env_optional_mm(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except ValueError:
        return None

def _env_mm_range(
    min_name: str,
    max_name: str,
    default_min_m: float,
    default_max_m: float,
) -> tuple[float, float]:
    low = _env_optional_mm(min_name)
    high = _env_optional_mm(max_name)
    if low is None:
        low = default_min_m
    if high is None:
        high = default_max_m
    if low > high:
        low, high = high, low
    return low, high

def _env_deg(name: str, default_deg: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return np.deg2rad(default_deg)
    try:
        return np.deg2rad(float(value))
    except ValueError:
        return np.deg2rad(default_deg)

def _env_optional_deg(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return np.deg2rad(float(value))
    except ValueError:
        return None

def _env_deg_range(
    min_name: str,
    max_name: str,
    default_min_rad: float,
    default_max_rad: float,
) -> tuple[float, float]:
    low = _env_optional_deg(min_name)
    high = _env_optional_deg(max_name)
    if low is None:
        low = default_min_rad
    if high is None:
        high = default_max_rad
    if low > high:
        low, high = high, low
    return low, high

def _default_dataset_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[5] / "data" / "ais_rpy_randomization"
    version = os.environ.get("AIC_RPY_DATASET_VERSION", "").strip()
    return base_dir / version if version else base_dir
