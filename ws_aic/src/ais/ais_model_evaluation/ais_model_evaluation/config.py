from __future__ import annotations

import os
import math
import time
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_cameras(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    cameras = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = set(cameras) - {"left", "center", "right"}
    return default if not cameras or invalid else cameras


def _first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _orientation_checkpoint_path() -> Path:
    env = os.environ.get("AIC_ORIENTATION_MODEL_PATH")
    if env:
        return Path(env).expanduser()
    return _first_existing(
        (
            WS_ROOT
            / "model"
            / "ais_orientation_prediction"
            / "rpy_delta_actual_correction_resnet50_left_center_right_concat_prev_none__close_v2.3"
            / "best.pt",
            WS_ROOT
            / "model"
            / "ais_orientation_prediction"
            / "rpy_delta_resnet50_left_center_right_concat_prev_v2.1__close_v2.3"
            / "best.pt",
        )
    )


COMPETITION_GRASP_RPY_RAD = 0.04
COMPETITION_GRASP_RPY_DEG = math.degrees(COMPETITION_GRASP_RPY_RAD)


class ModelEvalConfig:
    RUN_ID: str = os.environ.get("AIC_MODEL_EVAL_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
    OUTPUT_DIR: Path = Path(
        os.environ.get(
            "AIC_MODEL_EVAL_OUTPUT_DIR",
            str(WS_ROOT / "data" / "ais_model_evaluation" / RUN_ID),
        )
    ).expanduser()
    METRICS_PATH: Path = OUTPUT_DIR / "metrics.jsonl"
    SUMMARY_PATH: Path = OUTPUT_DIR / "summary.json"

    TRIALS: int = _env_int("AIC_MODEL_EVAL_TRIALS", 30)
    SEED: int = _env_int("AIC_MODEL_EVAL_SEED", 42)
    TRIAL_INDEX: int = _env_int("AIC_MODEL_EVAL_TRIAL_INDEX", 0)

    CAMERAS: tuple[str, ...] = _env_cameras(
        "AIC_MODEL_EVAL_CAMERAS",
        ("left", "center", "right"),
    )
    MIN_VISIBLE_CAMERAS: int = _env_int("AIC_MODEL_EVAL_MIN_VISIBLE_CAMERAS", len(CAMERAS))
    VISIBILITY_MARGIN_PX: float = _env_float("AIC_MODEL_EVAL_VISIBILITY_MARGIN_PX", 8.0)
    MAX_VISIBILITY_ATTEMPTS: int = _env_int("AIC_MODEL_EVAL_MAX_VISIBILITY_ATTEMPTS", 20)
    ORIENTATION_CHECKPOINT_PATH: Path = _orientation_checkpoint_path()
    DEVICE: str = os.environ.get("AIC_MODEL_EVAL_DEVICE", "auto")
    RECORD_VIDEO: bool = _env_bool("AIC_MODEL_EVAL_RECORD_VIDEO", False)
    VIDEO_CAMERAS: tuple[str, ...] = _env_cameras("AIC_MODEL_EVAL_VIDEO_CAMERAS", CAMERAS)
    VIDEO_FPS: float = _env_float("AIC_MODEL_EVAL_VIDEO_FPS", 8.0)
    VIDEO_MAX_HEIGHT: int = _env_int("AIC_MODEL_EVAL_VIDEO_MAX_HEIGHT", 360)
    VIDEO_DIR: Path = OUTPUT_DIR / "videos"

    RANDOMIZE_INITIAL_POSE: bool = _env_bool("AIC_MODEL_EVAL_RANDOMIZE_INITIAL_POSE", True)
    INITIAL_SETTLE_S: float = _env_float("AIC_MODEL_EVAL_INITIAL_SETTLE_S", 0.20)
    INITIAL_WAIT_TIMEOUT_S: float = _env_float("AIC_MODEL_EVAL_INITIAL_WAIT_TIMEOUT_S", 8.0)
    INITIAL_POSITION_TOLERANCE_M: float = _env_float("AIC_MODEL_EVAL_INITIAL_POSITION_TOLERANCE_MM", 10.0) / 1000.0
    INITIAL_ORIENTATION_TOLERANCE_RAD: float = _env_float("AIC_MODEL_EVAL_INITIAL_ORIENTATION_TOLERANCE_DEG", 5.0) * 3.141592653589793 / 180.0
    ACTION_SETTLE_S: float = _env_float("AIC_MODEL_EVAL_ACTION_SETTLE_S", 0.50)
    ACTION_WAIT_TIMEOUT_S: float = _env_float("AIC_MODEL_EVAL_ACTION_WAIT_TIMEOUT_S", 3.0)
    ACTION_POSITION_TOLERANCE_M: float = _env_float("AIC_MODEL_EVAL_ACTION_POSITION_TOLERANCE_MM", 3.0) / 1000.0
    ACTION_ORIENTATION_TOLERANCE_RAD: float = _env_float("AIC_MODEL_EVAL_ACTION_ORIENTATION_TOLERANCE_DEG", 1.0) * 3.141592653589793 / 180.0
    TF_WAIT_S: float = _env_float("AIC_MODEL_EVAL_TF_WAIT_S", 3.0)
    TF_POLL_S: float = _env_float("AIC_MODEL_EVAL_TF_POLL_S", 0.05)

    DX_MIN_M: float = _env_float("AIC_MODEL_EVAL_DX_MIN_MM", -50.0) / 1000.0
    DX_MAX_M: float = _env_float("AIC_MODEL_EVAL_DX_MAX_MM", 50.0) / 1000.0
    DY_MIN_M: float = _env_float("AIC_MODEL_EVAL_DY_MIN_MM", -50.0) / 1000.0
    DY_MAX_M: float = _env_float("AIC_MODEL_EVAL_DY_MAX_MM", 50.0) / 1000.0
    DZ_MIN_M: float = _env_float("AIC_MODEL_EVAL_DZ_MIN_MM", 0.0) / 1000.0
    DZ_MAX_M: float = _env_float("AIC_MODEL_EVAL_DZ_MAX_MM", 100.0) / 1000.0
    ROLL_MIN_RAD: float = _env_float("AIC_MODEL_EVAL_ROLL_MIN_DEG", -COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    ROLL_MAX_RAD: float = _env_float("AIC_MODEL_EVAL_ROLL_MAX_DEG", COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    PITCH_MIN_RAD: float = _env_float("AIC_MODEL_EVAL_PITCH_MIN_DEG", -COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    PITCH_MAX_RAD: float = _env_float("AIC_MODEL_EVAL_PITCH_MAX_DEG", COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    YAW_MIN_RAD: float = _env_float("AIC_MODEL_EVAL_YAW_MIN_DEG", -COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    YAW_MAX_RAD: float = _env_float("AIC_MODEL_EVAL_YAW_MAX_DEG", COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0
    RPY_NORM_MAX_RAD: float = _env_float("AIC_MODEL_EVAL_RPY_NORM_MAX_DEG", COMPETITION_GRASP_RPY_DEG) * 3.141592653589793 / 180.0

    MAX_DISTANCE_ACTION_M: float = _env_float("AIC_MODEL_EVAL_MAX_DISTANCE_ACTION_M", 0.0)
    MAX_ORIENTATION_ACTION_RAD: float = _env_float("AIC_MODEL_EVAL_MAX_ORIENTATION_ACTION_DEG", 0.0) * 3.141592653589793 / 180.0

    DISTANCE_SUCCESS_M: float = _env_float("AIC_MODEL_EVAL_DISTANCE_SUCCESS_MM", 2.0) / 1000.0
    ORIENTATION_SUCCESS_RAD: float = _env_float("AIC_MODEL_EVAL_ORIENTATION_SUCCESS_DEG", 1.0) * 3.141592653589793 / 180.0

    STIFFNESS: tuple[float, ...] = (80.0, 80.0, 80.0, 45.0, 45.0, 45.0)
    DAMPING: tuple[float, ...] = (45.0, 45.0, 45.0, 18.0, 18.0, 18.0)
    TOOL0_TO_TCP_Z: float = _env_float("AIC_TOOL0_TO_TCP_Z", 0.1965)
