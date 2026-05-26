from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


SRC_ROOT = _resolve_src_root()
WS_ROOT = SRC_ROOT.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_range_m(name: str, default_mm: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None:
        return (default_mm[0] / 1000.0, default_mm[1] / 1000.0)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        return (default_mm[0] / 1000.0, default_mm[1] / 1000.0)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (default_mm[0] / 1000.0, default_mm[1] / 1000.0)


class SfpGrvsConfig:
    """SFP-only GRVS training configuration.

    The default fallback epsilon uses the SFP triangulation-error CI99 values
    from ``visualize_triangulation_error_SFP.ipynb``. The SC notebook follows
    the same percentile logic, but this package is intentionally SFP-only for
    now.
    """

    PACKAGE_NAME = "ais_ground_truth_guided_residual_visual_servoing"
    VERSION = os.environ.get("AIC_GRVS_VERSION", "v0")

    DATA_ROOT = Path(
        os.environ.get("AIC_GRVS_DATA_ROOT", WS_ROOT / "data" / PACKAGE_NAME / VERSION)
    ).expanduser()
    MODEL_ROOT = Path(
        os.environ.get("AIC_GRVS_MODEL_ROOT", WS_ROOT / "model" / PACKAGE_NAME / VERSION)
    ).expanduser()

    YOLO_DATASET_DIR = Path(
        os.environ.get("AIC_GRVS_YOLO_DATASET_DIR", DATA_ROOT / "yolo")
    ).expanduser()
    DISTANCE_DATASET_DIR = Path(
        os.environ.get("AIC_GRVS_DISTANCE_DATASET_DIR", DATA_ROOT / "distance_prediction")
    ).expanduser()
    ROTATION_DATASET_DIR = Path(
        os.environ.get("AIC_GRVS_ROTATION_DATASET_DIR", DATA_ROOT / "rotation_prediction")
    ).expanduser()

    YOLO_MODEL_DIR = Path(
        os.environ.get("AIC_GRVS_YOLO_MODEL_DIR", MODEL_ROOT / "yolo")
    ).expanduser()
    DISTANCE_MODEL_DIR = Path(
        os.environ.get("AIC_GRVS_DISTANCE_MODEL_DIR", MODEL_ROOT / "distance_prediction")
    ).expanduser()
    ROTATION_MODEL_DIR = Path(
        os.environ.get("AIC_GRVS_ROTATION_MODEL_DIR", MODEL_ROOT / "rotation_prediction")
    ).expanduser()
    BATCH_DIR = Path(
        os.environ.get("AIC_GRVS_BATCH_DIR", DATA_ROOT / "batches")
    ).expanduser()
    REPLAY_BUFFER_DIR = Path(
        os.environ.get("AIC_GRVS_REPLAY_BUFFER_DIR", DATA_ROOT / "replay_buffer")
    ).expanduser()
    METRICS_DIR = Path(
        os.environ.get("AIC_GRVS_METRICS_DIR", DATA_ROOT / "metrics")
    ).expanduser()

    RECORD_DISTANCE_SAMPLES = _env_bool("AIC_GRVS_RECORD_DISTANCE_SAMPLES", True)
    RECORD_ROTATION_SAMPLES = _env_bool("AIC_GRVS_RECORD_ROTATION_SAMPLES", True)
    ROTATION_SAMPLE_STRIDE = _env_int("AIC_GRVS_ROTATION_SAMPLE_STRIDE", 1)
    RECORD_ROTATION_TRAJECTORY = _env_bool(
        "AIC_GRVS_RECORD_ROTATION_TRAJECTORY",
        True,
    )
    RECORD_YOLO_FAILURES = _env_bool("AIC_GRVS_RECORD_YOLO_FAILURES", True)
    USE_GT_ALIGN_ACTIONS = _env_bool("AIC_GRVS_USE_GT_ALIGN_ACTIONS", True)
    GT_FALLBACK_EPSILON_ENABLED = _env_bool(
        "AIC_GRVS_GT_FALLBACK_EPSILON_ENABLED",
        True,
    )
    GT_FALLBACK_EPSILON_SEED = _env_int("AIC_GRVS_GT_FALLBACK_EPSILON_SEED", 42)

    # SFP notebook CI99 signed axis error in meters: pred - gt.
    GT_FALLBACK_EPSILON_X_RANGE_M = _env_range_m(
        "AIC_GRVS_GT_FALLBACK_EPSILON_X_RANGE_M",
        (-1.4305, 1.0263),
    )
    GT_FALLBACK_EPSILON_Y_RANGE_M = _env_range_m(
        "AIC_GRVS_GT_FALLBACK_EPSILON_Y_RANGE_M",
        (-2.1049, 0.9690),
    )
    GT_FALLBACK_EPSILON_Z_RANGE_M = _env_range_m(
        "AIC_GRVS_GT_FALLBACK_EPSILON_Z_RANGE_M",
        (-4.4473, 5.5253),
    )

    ALIGN_MAX_ATTEMPTS = _env_int("AIC_GRVS_ALIGN_MAX_ATTEMPTS", 10)
    ALIGN_XY_TOL_M = _env_float("AIC_GRVS_ALIGN_XY_TOL_M", 0.003)
    ALIGN_Z_TOL_M = _env_float("AIC_GRVS_ALIGN_Z_TOL_M", 0.003)
    ALIGN_STABLE_STEPS = _env_int("AIC_GRVS_ALIGN_STABLE_STEPS", 2)
    ALIGN_INSERT_STEP_M = _env_float("AIC_GRVS_ALIGN_INSERT_STEP_M", 0.0015)
    MAX_XY_STEP_M = _env_float("AIC_GRVS_MAX_XY_STEP_M", 0.008)
    MAX_Z_STEP_M = _env_float("AIC_GRVS_MAX_Z_STEP_M", 0.005)
    XY_GAIN = _env_float("AIC_GRVS_XY_GAIN", 1.0)
    Z_GAIN = _env_float("AIC_GRVS_Z_GAIN", 0.9)
    COMMAND_SETTLE_S = _env_float("AIC_GRVS_COMMAND_SETTLE_S", 0.15)
    FORCE_RETRY_ENABLED = _env_bool("AIC_GRVS_FORCE_RETRY_ENABLED", True)
    FORCE_LPF_ALPHA = _env_float("AIC_GRVS_FORCE_LPF_ALPHA", 0.30)
    FORCE_THRESHOLD_X_N = _env_float("AIC_GRVS_FORCE_THRESHOLD_X_N", 10.0)
    FORCE_THRESHOLD_Y_N = _env_float("AIC_GRVS_FORCE_THRESHOLD_Y_N", 10.0)
    FORCE_THRESHOLD_Z_N = _env_float("AIC_GRVS_FORCE_THRESHOLD_Z_N", 8.0)
    FORCE_FALLBACK_Z_M = _env_float("AIC_GRVS_FORCE_FALLBACK_Z_M", 0.010)
    FORCE_FALLBACK_XY_M = _env_float("AIC_GRVS_FORCE_FALLBACK_XY_M", 0.050)
    FORCE_RETRY_XY_GATE_M = _env_float("AIC_GRVS_FORCE_RETRY_XY_GATE_M", 0.003)
    GT_TF_WAIT_S = _env_float("AIC_GRVS_GT_TF_WAIT_S", 10.0)
    GT_TF_POLL_S = _env_float("AIC_GRVS_GT_TF_POLL_S", 0.10)
    USE_GT_ORIENTATION = _env_bool("AIC_GRVS_USE_GT_ORIENTATION", True)
    GT_ORIENTATION_TOLERANCE_RAD = _env_float(
        "AIC_GRVS_GT_ORIENTATION_TOLERANCE_RAD",
        0.02,
    )
    PRE_DETECT_GT_VIEW_ENABLED = _env_bool(
        "AIC_GRVS_PRE_DETECT_GT_VIEW_ENABLED",
        True,
    )
    PRE_DETECT_VIEW_Z_OFFSET_M = _env_float(
        "AIC_GRVS_PRE_DETECT_VIEW_Z_OFFSET_M",
        0.18,
    )
    PRE_DETECT_VIEW_STEPS = _env_int("AIC_GRVS_PRE_DETECT_VIEW_STEPS", 30)
    PRE_DETECT_VIEW_DT = _env_float("AIC_GRVS_PRE_DETECT_VIEW_DT", 0.04)
    PRE_DETECT_VIEW_SETTLE_S = _env_float("AIC_GRVS_PRE_DETECT_VIEW_SETTLE_S", 0.25)

    YOLO_VAL_RATIO = _env_float("AIC_GRVS_YOLO_VAL_RATIO", 0.10)
    YOLO_BBOX_MARGIN = _env_float("AIC_GRVS_YOLO_BBOX_MARGIN", 0.08)
    YOLO_RECORD_ERROR_THRESHOLD_M = _env_float(
        "AIC_GRVS_YOLO_RECORD_ERROR_THRESHOLD_M",
        0.006,
    )


def sample_epsilon_m(
    rng: np.random.Generator,
    config: type[SfpGrvsConfig] = SfpGrvsConfig,
) -> np.ndarray:
    if not config.GT_FALLBACK_EPSILON_ENABLED:
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [
            rng.uniform(*config.GT_FALLBACK_EPSILON_X_RANGE_M),
            rng.uniform(*config.GT_FALLBACK_EPSILON_Y_RANGE_M),
            rng.uniform(*config.GT_FALLBACK_EPSILON_Z_RANGE_M),
        ],
        dtype=np.float64,
    )
