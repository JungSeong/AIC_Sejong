"""Configuration for the staged motion-planning policy."""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════
#  Stage 1 설정
# ═══════════════════════════════════════════════════════════

def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    # Fallback for the editable source layout.
    return Path(__file__).resolve().parents[5]


# ws_aic/src/ 루트
_SRC_ROOT = _resolve_src_root()


def _resolve_yolo_model_path() -> str:
    # 1순위: 환경 변수 (팀원마다 경로가 다를 수 있으므로 권장)
    env = os.environ.get("AIC_YOLO_MODEL_PATH")
    if env and os.path.isfile(env):
        return env

    # 2순위: 워크스페이스 기준 상대 경로
    candidates = [
        _SRC_ROOT / "model" / "ais_yolo" / "weight" / "best.pt",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    return str(candidates[0])


class Stage1Config:
    # ═════════════════════════════════════════════════════════
    #  Stage 1-A: Far Approach (10cm → 7cm 이상에서 접근)
    # ═════════════════════════════════════════════════════════
    # 포트 축선상 거리 (접근점 높이)
    Z_OFFSET: float = 0.07
    Z_OFFSET_TOLERANCE: float = 0.015
    XY_TOLERANCE: float = 0.025

    # 방향 사양
    AXIS_TOLERANCE_RAD: float = 0.087
    ROLL_TOLERANCE_RAD: float = 0.175

    # 속도 사양
    VEL_TOLERANCE_LIN: float = 0.01
    VEL_TOLERANCE_ANG: float = 0.1

    # Stage 1-A 동작
    N_STEPS: int = 80
    DT: float = 0.05  # 총 4초

    # 제어
    STIFFNESS: tuple = (200.0, 200.0, 200.0, 50.0, 50.0, 50.0)
    DAMPING: tuple = (80.0, 80.0, 80.0, 20.0, 20.0, 20.0)

    # ═════════════════════════════════════════════════════════
    #  Stage 1-B: Mid Approach (7cm → 3cm 하강, 정렬 포함)
    # ═════════════════════════════════════════════════════════
    ENABLE_STAGE1B: bool = True
    Z_OFFSET_MID: float = 0.03             # 7cm → 3cm 하강

    # [신규] Cable tension feedforward compensation (SFP only)
    # 근거: Stage 1-B 수렴 대기 루프에서 axial err 가 정확히 13.6mm 에서
    #       타임아웃 (Trial 1/2 재현). Hogan impedance steady-state err
    #       공식: Δx = F_ext / K.  13.6mm × 150 N/m ≈ 2N.
    #       → 케이블이 플러그를 2N 으로 위로 당기는 상태.
    # 해법: target z 를 14mm 아래로 하달 → compliance 평형에서 정확히 목표 도달.
    # SC 는 수렴 2.3mm 라 보상 불필요 (적용 시 overshoot 위험).
    SFP_CABLE_TENSION_COMPENSATION: float = 0.014  # 14mm (30mm 실험 결과: gripper z=0.2350m 동일, plug z=0.1771m 동일 → saturation 아니고 cable 평형점. 증량은 무의미하므로 14mm 유지)

    # [신규] 케이블 떨림(oscillation) 대응: 수렴 "안정성" 체크
    # 단순 "err < 임계" 아니라 "연속 N회 err < 임계" 로 변경.
    # 진동 중이면 err 가 오르락내리락 → stable_count 가 안 쌓임.
    STAGE1B_CONVERGENCE_TOL_M: float = 0.005         # 5mm (보상 후 기준)
    STAGE1B_STABLE_CONSECUTIVE: int = 3              # 0.15초 연속 안정
    STAGE1B_CONVERGENCE_MAX_WAIT_S: float = 2.0
    N_STEPS_MID: int = 40
    DT_MID: float = 0.05                    # 총 2초
    # 중간 접근은 조금 더 부드럽게 (낮은 stiffness)
    STIFFNESS_MID: tuple = (150.0, 150.0, 150.0, 40.0, 40.0, 40.0)
    DAMPING_MID: tuple = (70.0, 70.0, 70.0, 18.0, 18.0, 18.0)
    # [옵션 A] 수렴 대기 구간에서만 Z-방향 stiffness 부스트.
    # 진단: cable 평형점에서 F=2N, K=150 → Δx=13.6mm. K=500 으로 올리면
    # 이론상 Δx=4mm 로 축소. XY 는 150 유지(횡방향 순응성 보존),
    # rot 도 그대로. damping 은 sqrt(K_ratio)=sqrt(3.33)≈1.83 배로 Z 축만 증가.
    STIFFNESS_MID_BOOST: tuple = (150.0, 150.0, 500.0, 40.0, 40.0, 40.0)
    DAMPING_MID_BOOST: tuple = (70.0, 70.0, 130.0, 18.0, 18.0, 18.0)
    # 매 스텝 TF 재조회 (feedback)
    FEEDBACK_MID: bool = True

    # ═════════════════════════════════════════════════════════
    #  안정화 대기 (관성 떨림 대응)
    # ═════════════════════════════════════════════════════════
    SETTLE_AFTER_STAGE1A: float = 0.3       # 초
    SETTLE_AFTER_STAGE1B: float = 0.5       # 초
    # 5차 Hermite 사용 (가속도 끝값 0 → 떨림 감소)
    USE_QUINTIC_HERMITE: bool = True

    # ═════════════════════════════════════════════════════════
    #  안전 / 실패 처리
    # ═════════════════════════════════════════════════════════
    FORCE_DELTA_LIMIT_N: float = 15.0
    MAX_DURATION_S: float = 10.0           # Stage 1-A+B 포함 (기존 8 → 10)
    TF_RETRY: int = 10
    TF_RETRY_DT: float = 0.1

    # --- 접근 방향 ---
    USE_WORLD_Z_APPROACH: bool = True

    # --- Vision 설정 ---
    YOLO_MODEL_PATH: str = _resolve_yolo_model_path()
    YOLO_CONF_THRESH: float = 0.2
    # 3D 타당성 검증 범위 (base_link)
    BOARD_CENTER: tuple = (-0.38, 0.22, 0.13)
    BOARD_RADIUS: float = 0.5  # 보드 중심 반경 50cm 이내
    Z_RANGE: tuple = (-0.1, 0.5)  # z 좌표 범위

