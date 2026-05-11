"""Configuration for the distance prediction policy."""

import os
from pathlib import Path

from motion_planning_node.core.config import Stage1Config


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    return Path(__file__).resolve().parents[5]


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower()


def _env_cameras(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    cameras = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = set(cameras) - {"left", "center", "right"}
    if not cameras or invalid:
        return default
    return cameras


def _resolve_distance_model_path() -> str:
    env = os.environ.get("AIC_DISTANCE_MODEL_PATH")
    if env:
        env_path = Path(env).expanduser()
        if env_path.is_file():
            return str(env_path)

    candidates = [
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "sfp_distance_resnet50_left_center_right_concat"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "sfp_distance_resnet18_left_center_right_concat"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet50_left_center_right_concat"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet18_left_center_right_concat"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet50_left_center_right"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet18_left_center_right"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet50_center"
            / "best.pt"
        ),
        (
            WS_ROOT
            / "model"
            / "ais_distance_prediction"
            / "vision_offset_resnet18_center"
            / "best.pt"
        ),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


class DistancePredictionConfig:
    CHECKPOINT_PATH: str = _resolve_distance_model_path()
    DEVICE: str = os.environ.get("AIC_DISTANCE_DEVICE", "auto")
    CAMERAS: tuple[str, ...] = _env_cameras(
        "AIC_DISTANCE_CAMERAS",
        ("left", "center", "right"),
    )
    LABEL_PORT_FRAME_MODE: str = _env_str(
        "AIC_DISTANCE_LABEL_PORT_FRAME_MODE",
        "entrance",
    )

    APPROACH_VISION_RETRIES: int = _env_int("AIC_APPROACH_VISION_RETRIES", 8)
    APPROACH_RETRY_DT: float = _env_float("AIC_APPROACH_RETRY_DT", 0.1)
    APPROACH_Z_OFFSET_SFP_M: float = _env_float(
        "AIC_APPROACH_Z_OFFSET_SFP_M",
        Stage1Config.APPROACH_Z_OFFSET_SFP,
    )
    APPROACH_Z_OFFSET_SC_M: float = _env_float(
        "AIC_APPROACH_Z_OFFSET_SC_M",
        Stage1Config.APPROACH_Z_OFFSET_SC,
    )
    APPROACH_NEAR_Z_OFFSET_M: float = _env_float(
        "AIC_APPROACH_NEAR_Z_OFFSET_M",
        Stage1Config.TRIANGULATION_STOP_Z_OFFSET,
    )
    APPROACH_TCP_OFFSET_X_M: float = _env_float("AIC_APPROACH_TCP_OFFSET_X_M", 0.0)
    APPROACH_TCP_OFFSET_Y_M: float = _env_float("AIC_APPROACH_TCP_OFFSET_Y_M", 0.015)
    APPROACH_TCP_OFFSET_Z_M: float = _env_float("AIC_APPROACH_TCP_OFFSET_Z_M", 0.045)
    APPROACH_USE_TF_WRIST_ALIGNMENT: bool = _env_bool(
        "AIC_APPROACH_USE_TF_WRIST_ALIGNMENT", False
    )
    APPROACH_SFP_UPRIGHT_ENABLED: bool = _env_bool(
        "AIC_APPROACH_SFP_UPRIGHT_ENABLED", False
    )
    APPROACH_SFP_UPRIGHT_STEPS: int = _env_int("AIC_APPROACH_SFP_UPRIGHT_STEPS", 30)
    APPROACH_SFP_UPRIGHT_DT: float = _env_float("AIC_APPROACH_SFP_UPRIGHT_DT", 0.04)
    APPROACH_SFP_UPRIGHT_TOLERANCE_RAD: float = _env_float(
        "AIC_APPROACH_SFP_UPRIGHT_TOLERANCE_RAD", 0.02
    )
    APPROACH_SFP_UPRIGHT_MAX_ANGLE_RAD: float = _env_float(
        "AIC_APPROACH_SFP_UPRIGHT_MAX_ANGLE_RAD", 0.7
    )
    APPROACH_SFP_MANUAL_ROTATION_DEG: float = _env_float(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_DEG", -21.21
    )
    APPROACH_SFP_MANUAL_ROTATION_AXIS: str = _env_str(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_AXIS", "base_x"
    )
    APPROACH_SFP_MANUAL_ROTATION_STEPS: int = _env_int(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_STEPS", 30
    )
    APPROACH_SFP_MANUAL_ROTATION_DT: float = _env_float(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_DT", 0.04
    )
    INITIAL_LIFT_M: float = _env_float("AIC_DISTANCE_INITIAL_LIFT_M", 0.050)
    INITIAL_LIFT_STEPS: int = _env_int("AIC_DISTANCE_INITIAL_LIFT_STEPS", 40)
    INITIAL_LIFT_DT: float = _env_float("AIC_DISTANCE_INITIAL_LIFT_DT", 0.05)
    INITIAL_LIFT_SETTLE_S: float = _env_float(
        "AIC_DISTANCE_INITIAL_LIFT_SETTLE_S", 0.50
    )
    APPROACH_STEPS: int = _env_int("AIC_APPROACH_STEPS", 80)
    APPROACH_NEAR_STEPS: int = _env_int("AIC_APPROACH_NEAR_STEPS", 40)
    APPROACH_DT: float = _env_float("AIC_APPROACH_DT", 0.05)
    APPROACH_SETTLE_S: float = _env_float("AIC_APPROACH_SETTLE_S", 0.50)
    APPROACH_FORCE_DELTA_LIMIT_N: float = _env_float(
        "AIC_APPROACH_FORCE_DELTA_LIMIT_N", 15.0
    )
    APPROACH_STIFFNESS: tuple = (180.0, 180.0, 180.0, 45.0, 45.0, 45.0)
    APPROACH_DAMPING: tuple = (75.0, 75.0, 75.0, 18.0, 18.0, 18.0)
    APPROACH_NEAR_STIFFNESS: tuple = (140.0, 140.0, 140.0, 40.0, 40.0, 40.0)
    APPROACH_NEAR_DAMPING: tuple = (65.0, 65.0, 65.0, 16.0, 16.0, 16.0)

    MAX_STEPS: int = _env_int("AIC_DISTANCE_MAX_STEPS", 180)
    DT: float = _env_float("AIC_DISTANCE_DT", 0.05)
    SETTLE_AFTER_INSERT_S: float = _env_float("AIC_DISTANCE_SETTLE_S", 3.0)

    XY_GAIN: float = _env_float("AIC_DISTANCE_XY_GAIN", 0.65)
    Z_GAIN: float = _env_float("AIC_DISTANCE_Z_GAIN", 0.45)
    SMOOTHING_ALPHA: float = _env_float("AIC_DISTANCE_SMOOTHING_ALPHA", 0.6)
    MAX_XY_STEP_M: float = _env_float("AIC_DISTANCE_MAX_XY_STEP_M", 0.003)
    MAX_DOWN_STEP_M: float = _env_float("AIC_DISTANCE_MAX_DOWN_STEP_M", 0.0012)
    MAX_UP_STEP_M: float = _env_float("AIC_DISTANCE_MAX_UP_STEP_M", 0.0006)
    XY_DEADBAND_M: float = _env_float("AIC_DISTANCE_XY_DEADBAND_M", 0.00035)
    Z_DEADBAND_M: float = _env_float("AIC_DISTANCE_Z_DEADBAND_M", 0.00035)

    FINISH_DISTANCE_M: float = _env_float("AIC_DISTANCE_FINISH_M", 0.003)
    FINISH_STABLE_STEPS: int = _env_int("AIC_DISTANCE_FINISH_STABLE_STEPS", 4)
    MAX_INSERT_DEPTH_M: float = _env_float("AIC_DISTANCE_MAX_INSERT_DEPTH_M", 0.045)

    FORCE_LIMIT_N: float = _env_float("AIC_DISTANCE_FORCE_LIMIT_N", 18.0)
    STIFFNESS: tuple = (80.0, 80.0, 80.0, 45.0, 45.0, 45.0)
    DAMPING: tuple = (45.0, 45.0, 45.0, 18.0, 18.0, 18.0)
    ALIGN_MAX_STEPS: int = _env_int("AIC_DISTANCE_ALIGN_MAX_STEPS", 100)
    ALIGN_FINISH_XY_M: float = _env_float("AIC_DISTANCE_ALIGN_FINISH_XY_M", 0.002)
    ALIGN_STABLE_STEPS: int = _env_int("AIC_DISTANCE_ALIGN_STABLE_STEPS", 4)
    ALIGN_COMMAND_SETTLE_S: float = _env_float(
        "AIC_DISTANCE_ALIGN_COMMAND_SETTLE_S", 3.00
    )
    INSERT_MAX_STEPS: int = _env_int("AIC_DISTANCE_INSERT_MAX_STEPS", 120)
