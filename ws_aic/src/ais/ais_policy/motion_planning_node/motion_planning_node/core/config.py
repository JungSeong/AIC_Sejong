"""Configuration for the staged motion-planning policy."""

import os
from pathlib import Path


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        sibling_src = parent / "src"
        if (sibling_src / "aic").is_dir() and (sibling_src / "ais").is_dir():
            return sibling_src
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    # Fallback: editable source layout
    return Path(__file__).resolve().parents[5]


# ws_aic/src/ 루트
_SRC_ROOT = _resolve_src_root()
_WS_ROOT = _SRC_ROOT.parent

_MODEL_ROOT = _WS_ROOT / "model"


def _env_or_default(env_key: str, default: Path) -> str:
    env = os.environ.get(env_key)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return str(p)
    return str(default)


class Stage1Config:
    # ═════════════════════════════════════════════════════════
    #  Stage 1: Single Approach
    # ═════════════════════════════════════════════════════════
    APPROACH_Z_OFFSET_SFP: float = 0.150
    APPROACH_Z_OFFSET_SC: float = 0.050
    TRIANGULATION_STOP_Z_OFFSET: float = 0.050
    TRIANGULATION_STOP_X_OFFSET: float = 0.005
    TRIANGULATION_STOP_Y_OFFSET: float = 0.005
    Z_OFFSET_TOLERANCE: float = 0.015
    XY_TOLERANCE: float = 0.025

    AXIS_TOLERANCE_RAD: float = 0.087
    ROLL_TOLERANCE_RAD: float = 0.175

    VEL_TOLERANCE_LIN: float = 0.01
    VEL_TOLERANCE_ANG: float = 0.1

    APPROACH_STEPS: int = 100
    DT: float = 0.05

    STIFFNESS: tuple = (200.0, 200.0, 200.0, 50.0, 50.0, 50.0)
    DAMPING:   tuple = (80.0,  80.0,  80.0,  20.0, 20.0, 20.0)

    # ═════════════════════════════════════════════════════════
    #  안정화 대기 (관성 떨림 대응)
    # ═════════════════════════════════════════════════════════
    USE_QUINTIC_HERMITE: bool = True     # 5차 Hermite → 가속도 끝값 0

    # ═════════════════════════════════════════════════════════
    #  안전 / 실패 처리
    # ═════════════════════════════════════════════════════════
    FORCE_DELTA_LIMIT_N: float = 15.0
    MAX_DURATION_S: float = 10.0
    VISION_ACQUIRE_TIMEOUT_S: float = 20.0
    VISION_ACQUIRE_DT: float = 0.05

    USE_WORLD_Z_APPROACH: bool = True

    # ═════════════════════════════════════════════════════════
    #  Vision 설정
    # ═════════════════════════════════════════════════════════
    DETECTION_MODEL_PATH: str = _env_or_default(
        "AIC_YOLO_MODEL_PATH",
        _MODEL_ROOT / "ais_yolo" / "weights" / "best.pt",
    )
    DETECTION_CONF_THRESH: float = 0.7

    DISTANCE_MODEL_PATH: str = _env_or_default(
        "AIC_DISTANCE_MODEL_PATH",
        _MODEL_ROOT / "distance_prediction_vision_offset" / "best.pt",
    )

    # 3D 타당성 검증 범위 (base_link)
    BOARD_CENTER: tuple = (-0.38, 0.22, 0.13)
    BOARD_RADIUS: float = 0.5            # 보드 중심 반경 50cm 이내
    Z_RANGE: tuple = (-0.1, 0.5)
