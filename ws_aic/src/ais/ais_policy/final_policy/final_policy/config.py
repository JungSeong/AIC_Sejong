"""FinalPolicy에서 사용하는 환경변수 기반 설정값을 한곳에 모아둔 모듈."""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    """환경변수를 float로 읽고, 값이 없거나 파싱에 실패하면 기본값을 반환한다."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """환경변수를 int로 읽고, 값이 없거나 파싱에 실패하면 기본값을 반환한다."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """환경변수를 소문자 문자열로 정규화해서 반환한다."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower()


def _env_cameras(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """카메라 목록 환경변수를 읽고 left/center/right 외 값이 있으면 기본값을 쓴다."""
    value = os.environ.get(name)
    if value is None:
        return default
    cameras = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = set(cameras) - {"left", "center", "right"}
    return default if not cameras or invalid else cameras


class FinalPolicyConfig:
    """FinalPolicy의 접근, 정렬, 삽입 단계에서 공유하는 튜닝 파라미터."""

    DEVICE: str = os.environ.get("AIC_POSE_DEVICE", "auto")
    CAMERAS: tuple[str, ...] = _env_cameras(
        "AIC_POSE_CAMERAS",
        ("left", "center", "right"),
    )

    TCP_OFFSET_X: float = _env_float("AIC_APPROACH_TCP_OFFSET_X_M", 0.0)
    TCP_OFFSET_Y: float = _env_float("AIC_APPROACH_TCP_OFFSET_Y_M", 0.015)
    TCP_OFFSET_Z: float = _env_float("AIC_APPROACH_TCP_OFFSET_Z_M", 0.045)

    APPROACH_VISION_RETRIES: int = _env_int("AIC_APPROACH_VISION_RETRIES", 20)
    APPROACH_RETRY_DT: float = _env_float("AIC_APPROACH_RETRY_DT", 0.1)
    APPROACH_NEAR_Z_OFFSET_M: float = _env_float("AIC_APPROACH_NEAR_Z_OFFSET_M", 0.020)
    APPROACH_SFP_MANUAL_ROTATION_DEG: float = _env_float(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_DEG",
        -21.21,
    )
    APPROACH_SC_MANUAL_ROTATION_DEG: float = _env_float(
        "AIC_APPROACH_SC_MANUAL_ROTATION_DEG",
        -25.21,
    )
    APPROACH_SFP_MANUAL_ROTATION_AXIS: str = _env_str(
        "AIC_APPROACH_SFP_MANUAL_ROTATION_AXIS",
        "base_x",
    )
    APPROACH_STEPS: int = _env_int("AIC_APPROACH_STEPS", 80)
    APPROACH_DT: float = _env_float("AIC_APPROACH_DT", 0.05)
    APPROACH_SETTLE_S: float = _env_float("AIC_APPROACH_SETTLE_S", 0.50)
    APPROACH_STIFFNESS: tuple = (180.0, 180.0, 180.0, 45.0, 45.0, 45.0)
    APPROACH_DAMPING: tuple = (75.0, 75.0, 75.0, 18.0, 18.0, 18.0)
    APPROACH_NEAR_STIFFNESS: tuple = (140.0, 140.0, 140.0, 40.0, 40.0, 40.0)
    APPROACH_NEAR_DAMPING: tuple = (65.0, 65.0, 65.0, 16.0, 16.0, 16.0)
    BOARD_CENTER: tuple = (-0.38, 0.22, 0.13)
    BOARD_RADIUS: float = 0.5
    Z_RANGE: tuple = (-0.1, 0.5)

    INITIAL_LIFT_M: float = _env_float("AIC_DISTANCE_INITIAL_LIFT_M", 0.050)
    INITIAL_LIFT_STEPS: int = _env_int("AIC_DISTANCE_INITIAL_LIFT_STEPS", 40)
    INITIAL_LIFT_DT: float = _env_float("AIC_DISTANCE_INITIAL_LIFT_DT", 0.05)
    INITIAL_LIFT_SETTLE_S: float = _env_float(
        "AIC_DISTANCE_INITIAL_LIFT_SETTLE_S",
        0.50,
    )

    DT: float = _env_float("AIC_DISTANCE_DT", 0.05)
    ALIGN_STIFFNESS: tuple = (80.0, 80.0, 80.0, 45.0, 45.0, 45.0)
    ALIGN_DAMPING: tuple = (45.0, 45.0, 45.0, 18.0, 18.0, 18.0)
    VISION_OFFSET_XY_TOL_M: float = _env_float(
        "AIC_VISION_OFFSET_XY_TOL_M",
        _env_float("AIC_VISION_OFFSET_XYZ_TOL_M", 0.003),
    )
    VISION_OFFSET_RPY_TOL_RAD: float = _env_float(
        "AIC_VISION_OFFSET_RPY_TOL_RAD",
        0.01,
    )
    VISION_OFFSET_MAX_ABS_POSITION_M: float = _env_float(
        "AIC_VISION_OFFSET_MAX_ABS_POSITION_M",
        0.08,
    )
    VISION_OFFSET_MAX_ABS_RPY_RAD: float = _env_float(
        "AIC_VISION_OFFSET_MAX_ABS_RPY_RAD",
        0.6,
    )

    STABLE_STEPS: int = _env_int("AIC_POSE_STABLE_STEPS", 4)
    ALIGN_MAX_STEPS: int = _env_int("AIC_POSE_ALIGN_MAX_STEPS", 100)
    COMMAND_SETTLE_S: float = _env_float("AIC_POSE_COMMAND_SETTLE_S", 1.0)

    INSERT_STEP_M: float = _env_float("AIC_POSE_INSERT_STEP_M", 0.0006)
    INSERT_DT: float = _env_float("AIC_POSE_INSERT_DT", 0.08)
    MAX_DOWN_STEP_M: float = _env_float("AIC_DISTANCE_MAX_DOWN_STEP_M", 0.0012)
    MAX_INSERT_DEPTH_M: float = _env_float("AIC_DISTANCE_MAX_INSERT_DEPTH_M", 0.045)
    INSERT_MAX_STEPS: int = _env_int("AIC_DISTANCE_INSERT_MAX_STEPS", 120)
    SETTLE_AFTER_INSERT_S: float = _env_float("AIC_DISTANCE_SETTLE_S", 3.0)
    SFP_INSERTION_STIFFNESS: tuple = (20.0, 20.0, 250.0, 10.0, 10.0, 40.0)
    SFP_INSERTION_DAMPING: tuple = (10.0, 10.0, 60.0, 5.0, 5.0, 15.0)
    SC_INSERTION_STIFFNESS: tuple = (51.0, 50.0, 300.0, 15.0, 15.0, 40.0)
    SC_INSERTION_DAMPING: tuple = (31.0, 30.0, 87.0, 8.0, 8.0, 15.0)
