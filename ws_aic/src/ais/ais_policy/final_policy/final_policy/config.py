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


def _env_bool(name: str, default: bool) -> bool:
    """환경변수를 bool로 읽고, 값이 없거나 알 수 없는 값이면 기본값을 반환한다."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
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
    PUBLISH_TRIANGULATED_PORT_XYZ: bool = _env_bool(
        "AIC_PUBLISH_TRIANGULATED_PORT_XYZ",
        False,
    )
    TRIANGULATED_PORT_XYZ_TOPIC: str = os.environ.get(
        "AIC_TRIANGULATED_PORT_XYZ_TOPIC",
        "/final_policy/triangulated_port_xyz",
    )
    TRIANGULATED_PORT_XYZ_FRAME_ID: str = os.environ.get(
        "AIC_TRIANGULATED_PORT_XYZ_FRAME_ID",
        "base_link",
    )
    TRIANGULATION_DEBUG_SAVE_ENABLED: bool = _env_bool(
        "AIC_TRIANGULATION_DEBUG_SAVE",
        True,
    )
    TRIANGULATION_EVAL_ONLY: bool = _env_bool(
        "AIC_TRIANGULATION_EVAL_ONLY",
        False,
    )
    TRIANGULATION_DEBUG_FIXED_FRAME: str = os.environ.get(
        "AIC_TRIANGULATION_DEBUG_FIXED_FRAME",
        "world",
    )
    TRIANGULATION_DEBUG_RVIZ_ENABLED: bool = _env_bool(
        "AIC_TRIANGULATION_DEBUG_RVIZ",
        True,
    )
    TRIANGULATION_DEBUG_IMAGE_TOPIC_PREFIX: str = os.environ.get(
        "AIC_TRIANGULATION_DEBUG_IMAGE_TOPIC_PREFIX",
        "/final_policy/triangulation_debug",
    )
    TRIANGULATION_DEBUG_MARKER_TOPIC: str = os.environ.get(
        "AIC_TRIANGULATION_DEBUG_MARKER_TOPIC",
        "/final_policy/triangulation_debug/markers",
    )
    DETECTION_DEBUG_RVIZ_ENABLED: bool = _env_bool(
        "AIC_DETECTION_DEBUG_RVIZ",
        True,
    )
    DETECTION_DEBUG_IMAGE_TOPIC_PREFIX: str = os.environ.get(
        "AIC_DETECTION_DEBUG_IMAGE_TOPIC_PREFIX",
        "/final_policy/detection_debug",
    )
    TRIANGULATION_SYNC_THRESHOLD_MS: float = _env_float(
        "AIC_TRIANGULATION_SYNC_THRESHOLD_MS",
        30.0,
    )
    SFP_YOLO_PORT_INDEX_FLIP: bool = _env_bool(
        "AIC_SFP_YOLO_PORT_INDEX_FLIP",
        True,
    )
    CAMERAS: tuple[str, ...] = _env_cameras(
        "AIC_POSE_CAMERAS",
        ("left", "center", "right"),
    )

    TCP_OFFSET_X: float = _env_float("AIC_APPROACH_TCP_OFFSET_X_M", 0.0)
    TCP_OFFSET_Y: float = _env_float("AIC_APPROACH_TCP_OFFSET_Y_M", 0.015)
    TCP_OFFSET_Z: float = _env_float("AIC_APPROACH_TCP_OFFSET_Z_M", 0.045)

    APPROACH_VISION_RETRIES: int = _env_int("AIC_APPROACH_VISION_RETRIES", 20)
    APPROACH_RETRY_DT: float = _env_float("AIC_APPROACH_RETRY_DT", 0.2)
    APPROACH_NEAR_Z_OFFSET_M: float = _env_float("AIC_APPROACH_NEAR_Z_OFFSET_M", 0.030)
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
    LIFT_DETECT_TO_APPROACH_SETTLE_S: float = _env_float(
        "AIC_LIFT_DETECT_TO_APPROACH_SETTLE_S",
        2.0,
    )

    DT: float = _env_float("AIC_DISTANCE_DT", 0.05)
    ALIGN_STIFFNESS: tuple = (80.0, 80.0, 80.0, 45.0, 45.0, 45.0)
    ALIGN_DAMPING: tuple = (45.0, 45.0, 45.0, 18.0, 18.0, 18.0)
    VISION_OFFSET_XY_TOL_M: float = _env_float(
        "AIC_VISION_OFFSET_XY_TOL_M",
        _env_float("AIC_VISION_OFFSET_XYZ_TOL_M", 0.010),
    )
    VISION_OFFSET_RPY_TOL_RAD: float = _env_float(
        "AIC_VISION_OFFSET_RPY_TOL_RAD",
        0.05236,
    )
    VISION_OFFSET_MAX_ABS_POSITION_M: float = _env_float(
        "AIC_VISION_OFFSET_MAX_ABS_POSITION_M",
        0.08,
    )
    VISION_OFFSET_MAX_ABS_RPY_RAD: float = _env_float(
        "AIC_VISION_OFFSET_MAX_ABS_RPY_RAD",
        0.6,
    )
    VISION_OFFSET_XY_MOVE_GAIN: float = _env_float(
        "AIC_VISION_OFFSET_XY_MOVE_GAIN",
        0.5,
    )
    VISION_OFFSET_MAX_XY_STEP_M: float = _env_float(
        "AIC_VISION_OFFSET_MAX_XY_STEP_M",
        0.003,
    )
    VISION_OFFSET_RPY_MOVE_GAIN: float = _env_float(
        "AIC_VISION_OFFSET_RPY_MOVE_GAIN",
        0.5,
    )
    VISION_OFFSET_MAX_RPY_STEP_RAD: float = _env_float(
        "AIC_VISION_OFFSET_MAX_RPY_STEP_RAD",
        0.008726646,
    )
    VISION_OFFSET_VALIDATION_STEPS: int = _env_int(
        "AIC_VISION_OFFSET_VALIDATION_STEPS",
        4,
    )
    VISION_OFFSET_XY_SPREAD_TOL_M: float = _env_float(
        "AIC_VISION_OFFSET_XY_SPREAD_TOL_M",
        0.004,
    )
    VISION_OFFSET_RPY_SPREAD_TOL_RAD: float = _env_float(
        "AIC_VISION_OFFSET_RPY_SPREAD_TOL_RAD",
        0.01745,
    )
    VISION_OFFSET_VALIDATION_DT: float = _env_float(
        "AIC_VISION_OFFSET_VALIDATION_DT",
        0.15,
    )
    ALIGN_TRIANGULATION_ENABLED: bool = _env_bool(
        "AIC_ALIGN_TRIANGULATION_ENABLED",
        True,
    )
    ALIGN_TRIANGULATION_REQUIRED: bool = _env_bool(
        "AIC_ALIGN_TRIANGULATION_REQUIRED",
        False,
    )
    ALIGN_TRIANGULATION_XY_TOL_M: float = _env_float(
        "AIC_ALIGN_TRIANGULATION_XY_TOL_M",
        0.006,
    )
    ALIGN_TRIANGULATION_MAX_STEP_M: float = _env_float(
        "AIC_ALIGN_TRIANGULATION_MAX_STEP_M",
        0.003,
    )
    ALIGN_TRIANGULATION_MOVE_GAIN: float = _env_float(
        "AIC_ALIGN_TRIANGULATION_MOVE_GAIN",
        0.5,
    )
    ALIGN_TRIANGULATION_RETRIES: int = _env_int(
        "AIC_ALIGN_TRIANGULATION_RETRIES",
        2,
    )
    ALIGN_TRIANGULATION_RETRY_DT: float = _env_float(
        "AIC_ALIGN_TRIANGULATION_RETRY_DT",
        0.1,
    )
    ALIGN_TRIANGULATION_ALLOW_TCP_TIP_FALLBACK: bool = _env_bool(
        "AIC_ALIGN_TRIANGULATION_ALLOW_TCP_TIP_FALLBACK",
        True,
    )
    ALIGN_TCP_FROM_TIP_X_M: float = _env_float(
        "AIC_ALIGN_TCP_FROM_TIP_X_M",
        TCP_OFFSET_X,
    )
    ALIGN_TCP_FROM_TIP_Y_M: float = _env_float(
        "AIC_ALIGN_TCP_FROM_TIP_Y_M",
        TCP_OFFSET_Y,
    )
    SFP_TIP_TARGET_CLASS_ID: int = _env_int("AIC_SFP_TIP_TARGET_CLASS_ID", 2)
    SC_TIP_TARGET_CLASS_ID: int = _env_int("AIC_SC_TIP_TARGET_CLASS_ID", 3)
    SFP_TRIANGULATION_TARGET_DX_M: float = _env_float(
        "AIC_SFP_TRIANGULATION_TARGET_DX_M",
        0.0,
    )
    SFP_TRIANGULATION_TARGET_DY_M: float = _env_float(
        "AIC_SFP_TRIANGULATION_TARGET_DY_M",
        0.0,
    )
    SC_TRIANGULATION_TARGET_DX_M: float = _env_float(
        "AIC_SC_TRIANGULATION_TARGET_DX_M",
        0.0,
    )
    SC_TRIANGULATION_TARGET_DY_M: float = _env_float(
        "AIC_SC_TRIANGULATION_TARGET_DY_M",
        0.0,
    )

    STABLE_STEPS: int = _env_int("AIC_POSE_STABLE_STEPS", 4)
    ALIGN_MAX_STEPS: int = _env_int("AIC_POSE_ALIGN_MAX_STEPS", 50)
    COMMAND_SETTLE_S: float = _env_float("AIC_POSE_COMMAND_SETTLE_S", 1.0)

    INSERT_STEP_M: float = _env_float("AIC_POSE_INSERT_STEP_M", 0.0006)
    INSERT_DT: float = _env_float("AIC_POSE_INSERT_DT", 0.08)
    MAX_DOWN_STEP_M: float = _env_float("AIC_DISTANCE_MAX_DOWN_STEP_M", 0.0012)
    MAX_INSERT_DEPTH_M: float = _env_float("AIC_DISTANCE_MAX_INSERT_DEPTH_M", 0.045)
    INSERT_MAX_STEPS: int = _env_int("AIC_DISTANCE_INSERT_MAX_STEPS", 120)
    SETTLE_AFTER_INSERT_S: float = _env_float("AIC_DISTANCE_SETTLE_S", 3.0)
    SFP_INSERTION_STIFFNESS: tuple = (80.0, 80.0, 250.0, 45.0, 45.0, 45.0)
    SFP_INSERTION_DAMPING: tuple = (45.0, 45.0, 60.0, 18.0, 18.0, 18.0)
    SC_INSERTION_STIFFNESS: tuple = (80.0, 80.0, 300.0, 45.0, 45.0, 45.0)
    SC_INSERTION_DAMPING: tuple = (45.0, 45.0, 87.0, 18.0, 18.0, 18.0)
