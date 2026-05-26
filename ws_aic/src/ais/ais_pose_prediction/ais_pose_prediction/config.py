from __future__ import annotations

import os
from pathlib import Path


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


SRC_ROOT = _resolve_src_root()
WS_ROOT = SRC_ROOT.parent


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


def _env_cameras(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    cameras = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = set(cameras) - {"left", "center", "right"}
    return default if not cameras or invalid else cameras


def _resolve_model_path() -> str:
    env = os.environ.get("AIC_POSE_MODEL_PATH")
    if env:
        return str(Path(env).expanduser())
    return str(WS_ROOT / "model" / "ais_pose_prediction" / "pose_resnet50_v4.0" / "best.pt")


class PosePredictionConfig:
    CHECKPOINT_PATH: str = _resolve_model_path()
    DEVICE: str = os.environ.get("AIC_POSE_DEVICE", "auto")
    CAMERAS: tuple[str, ...] = _env_cameras("AIC_POSE_CAMERAS", ("left", "center", "right"))

    XY_GAIN: float = _env_float("AIC_POSE_XY_GAIN", 0.65)
    YAW_GAIN: float = _env_float("AIC_POSE_YAW_GAIN", 0.8)
    MAX_XY_STEP_M: float = _env_float("AIC_POSE_MAX_XY_STEP_M", 0.003)
    MAX_YAW_STEP_RAD: float = _env_float("AIC_POSE_MAX_YAW_STEP_RAD", 0.02)
    XY_TOL_M: float = _env_float("AIC_POSE_XY_TOL_M", 0.003)
    YAW_TOL_RAD: float = _env_float("AIC_POSE_YAW_TOL_RAD", 0.01)
    STABLE_STEPS: int = _env_int("AIC_POSE_STABLE_STEPS", 4)
    ALIGN_MAX_STEPS: int = _env_int("AIC_POSE_ALIGN_MAX_STEPS", 100)
    COMMAND_SETTLE_S: float = _env_float("AIC_POSE_COMMAND_SETTLE_S", 1.0)

    INSERT_STEP_M: float = _env_float("AIC_POSE_INSERT_STEP_M", 0.0006)
    INSERT_DT: float = _env_float("AIC_POSE_INSERT_DT", 0.08)
    INSERT_RETRY_MAX: int = _env_int("AIC_POSE_INSERT_RETRY_MAX", 8)
    INSERT_RETRY_SETTLE_S: float = _env_float("AIC_POSE_INSERT_RETRY_SETTLE_S", 0.25)
    INSERT_FORCE_DROP_LIMIT_N: float = _env_float("AIC_POSE_INSERT_FORCE_DROP_LIMIT_N", 4.0)
    INSERT_FORCE_RISE_LIMIT_N: float = _env_float("AIC_POSE_INSERT_FORCE_RISE_LIMIT_N", 12.0)
    INSERT_RETRY_LIFT_M: float = _env_float("AIC_POSE_INSERT_RETRY_LIFT_M", 0.004)
