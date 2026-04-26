"""
StagedPolicy v3: Vision 통합 3단계 State Machine 정책

Stage 1 (이동) — 모션 플래닝 + Vision (신찬희 담당)
  - 포트 좌표 획득: Ground truth TF 우선, 실패 시 YOLO+스테레오
  - 목표: 그리퍼를 포트 축선 위 10cm 지점까지 이동
  - 방식: S-curve 직선 보간

Stage 2/3: 임시 (ground_truth 기반, 추후 AI 교체)

환경별 동작:
  - ground_truth=true:  TF로 포트 좌표 직접 읽음 (오차 0)
  - ground_truth=false: YOLO 검출 + 스테레오 삼각측량 (오차 ~17mm)

실행:
  pixi reinstall ros-kilted-motion-planning-node
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=motion_planning_node.StagedPolicy
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


# ═══════════════════════════════════════════════════════════
#  Stage 1 설정
# ═══════════════════════════════════════════════════════════

# ws_aic/src/ 루트 (이 파일 기준 상위 4단계)
_SRC_ROOT = Path(__file__).resolve().parents[4]


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


@dataclass
class Stage1Result:
    success: bool
    final_pose: Optional[Pose]
    port_pose: Optional[Pose]
    port_axis: Optional[np.ndarray]
    elapsed_time: float
    failure_reason: Optional[str] = None
    port_source: str = "unknown"  # "tf" | "vision" | "fallback"


# ═══════════════════════════════════════════════════════════
#  쿼터니언 / 벡터 유틸리티
# ═══════════════════════════════════════════════════════════

def quat_to_tuple(q: Quaternion) -> tuple:
    return (q.w, q.x, q.y, q.z)


def tuple_to_quat(q: tuple) -> Quaternion:
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def quat_inverse(q: tuple) -> tuple:
    return (q[0], -q[1], -q[2], -q[3])


def rotate_vector_by_quat(v: np.ndarray, q: tuple) -> np.ndarray:
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    q_inv = quat_inverse(q)
    rotated = quaternion_multiply(quaternion_multiply(q, qv), q_inv)
    return np.array([rotated[1], rotated[2], rotated[3]])


def s_curve(t: float) -> float:
    """3차 Hermite: 시작/끝 속도 0."""
    return 3.0 * t * t - 2.0 * t * t * t


def s_curve_quintic(t: float) -> float:
    """5차 Hermite: 시작/끝에서 속도 + 가속도 모두 0.

    식: 10t³ - 15t⁴ + 6t⁵
    특성:
      t=0: pos=0, vel=0, acc=0
      t=1: pos=1, vel=0, acc=0
    → 가속도 연속 → 관성 떨림 최소화
    """
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    """보간 프로파일 선택."""
    return s_curve_quintic(t) if quintic else s_curve(t)


def _project_3d_to_pixel(point_3d_base, K, T_base_to_cam):
    """3D base 좌표를 카메라 이미지 픽셀로 투영."""
    p_homo = np.append(point_3d_base, 1.0)
    p_cam = T_base_to_cam @ p_homo
    x, y, z = p_cam[:3]
    if z < 1e-6:
        return -1.0, -1.0
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def transform_to_matrix(t) -> np.ndarray:
    """geometry_msgs/Transform → 4x4 matrix."""
    tx, ty, tz = t.translation.x, t.translation.y, t.translation.z
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1 - 2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1 - 2*(xx+yy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


# ═══════════════════════════════════════════════════════════
#  Vision 모듈 (`YOLO + Stereo)
# ═══════════════════════════════════════════════════════════

class VisionPortEstimator:
    """카메라 영상에서 포트 3D 좌표를 추정.

    첫 호출 시에만 YOLO 모델 로드 (지연 초기화).
    """

    CAMERAS = [("left", "left_camera/optical"),
               ("center", "center_camera/optical"),
               ("right", "right_camera/optical")]

    CLASS_NAMES = {0: "sfp_port", 1: "sc_port"}

    # 디버그 이미지 저장 디렉토리 (None이면 저장 안 함)
    # Docker: AIC_DEBUG_SAVE_DIR=/debug/yolo_detections (docker-compose에서 volume mount)
    # Native: ~/aic_debug/yolo_detections (기본값)
    DEBUG_SAVE_DIR: str = os.environ.get(
        "AIC_DEBUG_SAVE_DIR",
        str(Path.home() / "aic_debug" / "yolo_detections"),
    )

    def __init__(self, model_path: str, conf_thresh: float = 0.5, logger=None):
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._logger = logger
        self._model = None
        self._loaded = False
        self._debug_call_count = 0  # 호출 횟수 (파일명 중복 방지)

        # 디버그 디렉토리 생성
        if self.DEBUG_SAVE_DIR:
            try:
                os.makedirs(self.DEBUG_SAVE_DIR, exist_ok=True)
                if logger:
                    logger.info(f"[Vision Debug] 저장 경로: {self.DEBUG_SAVE_DIR}")
            except Exception as e:
                if logger:
                    logger.error(f"[Vision Debug] 디렉토리 생성 실패: {e}")
                # 저장 실패해도 동작은 계속 — 디버그 비활성화
                self.__class__.DEBUG_SAVE_DIR = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not os.path.isfile(self._model_path):
            if self._logger:
                self._logger.error(
                    f"YOLO 모델 파일 없음: {self._model_path}\n"
                    "  해결 방법:\n"
                    "  1) AIC_YOLO_MODEL_PATH 환경 변수로 경로 지정\n"
                    "  2) 패키지 share/models/port_detector.pt 에 번들\n"
                    "  3) ~/aic_yolo_runs/port_detector/weights/best.pt 에 배치"
                )
            return
        try:
            from ultralytics import YOLO
            import cv2  # noqa: F401
        except ImportError as e:
            if self._logger:
                self._logger.error(f"Vision 의존성 없음: {e}")
            return
        try:
            self._model = YOLO(self._model_path)
            self._loaded = True
            if self._logger:
                self._logger.info(f"YOLO 모델 로드: {self._model_path}")
        except Exception as e:
            if self._logger:
                self._logger.error(f"YOLO 로드 실패: {e}")

    @staticmethod
    def _image_from_msg(img_msg):
        import cv2
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3)
        if img_msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _detect(self, image: np.ndarray, cam_name: str = "") -> list:
        import cv2

        # conf=0.01로 raw 결과 전부 받아서 로깅 후, conf_thresh 적용
        results = self._model(image, verbose=False, conf=0.01)
        raw_dets = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_dets.append({
                    "class_id": cls,
                    "class_name": self.CLASS_NAMES.get(cls, "unknown"),
                    "conf": conf,
                    "u": float((x1 + x2) / 2),
                    "v": float((y1 + y2) / 2),
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                })

        # 진단 로그
        if self._logger:
            if raw_dets:
                for d in raw_dets:
                    flag = "✓" if d["conf"] >= self._conf_thresh else "✗(low conf)"
                    self._logger.info(
                        f"  [{cam_name}] {flag} {d['class_name']} "
                        f"conf={d['conf']:.3f} (thresh={self._conf_thresh}) "
                        f"uv=({d['u']:.0f},{d['v']:.0f})"
                    )
            else:
                self._logger.warn(f"  [{cam_name}] 검출 결과 0개 (모델이 아무것도 못 찾음)")

        # ── 디버그 이미지 저장 ────────────────────────────────
        if self.DEBUG_SAVE_DIR:
            debug_img = image.copy()
            # 클래스별 색상: sfp_port=초록, sc_port=파랑, unknown=빨강
            COLOR = {0: (0, 255, 0), 1: (255, 100, 0), -1: (0, 0, 255)}

            for d in raw_dets:
                x1, y1, x2, y2 = d["xyxy"]
                color = COLOR.get(d["class_id"], COLOR[-1])
                passed = d["conf"] >= self._conf_thresh

                # 통과한 것: 실선 두껍게, 실패한 것: 점선 얇게
                thickness = 3 if passed else 1
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)

                # 라벨
                label = f"{d['class_name']} {d['conf']:.2f}"
                label += " ✓" if passed else " ✗"
                cv2.putText(
                    debug_img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

            # threshold 선 표시 (이미지 상단에 텍스트)
            cv2.putText(
                debug_img,
                f"thresh={self._conf_thresh}  cam={cam_name}  dets={len(raw_dets)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            fname = (
                f"{self.DEBUG_SAVE_DIR}/"
                f"{self._debug_call_count:04d}_{cam_name}.jpg"
            )
            cv2.imwrite(fname, debug_img)

        self._debug_call_count += 1

        # conf_thresh 필터 적용
        return [d for d in raw_dets if d["conf"] >= self._conf_thresh]

    @staticmethod
    def _triangulate(u_a, v_a, K_a, T_base_to_a,
                     u_b, v_b, K_b, T_base_to_b) -> np.ndarray:
        import cv2
        P_a = K_a @ T_base_to_a[:3, :]
        P_b = K_b @ T_base_to_b[:3, :]
        pts_4d = cv2.triangulatePoints(
            P_a, P_b,
            np.array([[u_a], [v_a]], dtype=np.float64),
            np.array([[u_b], [v_b]], dtype=np.float64),
        )
        return (pts_4d[:3] / pts_4d[3]).flatten()

    def estimate(self, obs, tf_buffer, target_class_id: int,
                 port_hint: Optional[str] = None) -> Optional[np.ndarray]:
        """포트 3D 좌표 추정 (호환성 유지).

        단일 best 결과를 반환. 세밀한 선택이 필요하면 estimate_all() 사용.
        """
        candidates = self.estimate_all(obs, tf_buffer, target_class_id)
        if not candidates:
            return None

        # port_hint가 주어지면 그에 맞는 걸 선택
        if port_hint is not None:
            chosen = self.select_by_port_name(candidates, port_hint)
            if chosen is not None:
                return chosen

        # 기본: 보드 중심에 가장 가까운 것 (첫 번째)
        return candidates[0]["pos"]

    def estimate_all(
        self, obs, tf_buffer, target_class_id: int
    ) -> list:
        """가능한 모든 유효 후보 반환. score 낮은 순(= 보드 중심 가까운 순) 정렬.

        Returns:
            [{"pos": np.ndarray(3), "score": float, "conf_sum": float}, ...]
        """
        self._ensure_loaded()
        if not self._loaded:
            return []

        # 1. 카메라별 이미지 + 내부 파라미터 + 외부 TF
        images = {
            "left": self._image_from_msg(obs.left_image),
            "center": self._image_from_msg(obs.center_image),
            "right": self._image_from_msg(obs.right_image),
        }
        cam_infos = {
            "left": obs.left_camera_info,
            "center": obs.center_camera_info,
            "right": obs.right_camera_info,
        }
        cam_T_in_base = {}
        for name, frame in self.CAMERAS:
            try:
                tf = tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
            except TransformException:
                if self._logger:
                    self._logger.warn(f"Vision: 카메라 {name} TF 없음")
                return []

        # 2. 각 카메라에서 검출 (타겟 클래스만)
        if self._logger:
            self._logger.info(f"Vision: 검출 시작 (target_class_id={target_class_id})")
        detections = {}
        for name in ["left", "center", "right"]:
            dets = self._detect(images[name], cam_name=name)
            dets = [d for d in dets if d["class_id"] == target_class_id]
            detections[name] = dets

        cams_with_dets = [n for n, d in detections.items() if d]
        if len(cams_with_dets) < 2:
            if self._logger:
                self._logger.warn(
                    f"Vision: 2대 이상의 카메라에서 검출 실패 "
                    f"({cams_with_dets})"
                )
            return []

        # 3. 가능한 카메라 쌍 모두 사용 (3-view consistency 검증용)
        available_pairs = []
        for a, b in [("left", "center"), ("center", "right"),
                     ("left", "right")]:
            if a in cams_with_dets and b in cams_with_dets:
                available_pairs.append((a, b))

        if not available_pairs:
            return []

        # 카메라 정보 사전 캐시
        K = {n: np.array(cam_infos[n].k).reshape(3, 3) for n, _ in self.CAMERAS
             if n in cams_with_dets}
        T_base_to = {n: np.linalg.inv(cam_T_in_base[n]) for n, _ in self.CAMERAS
                     if n in cams_with_dets}

        # 4. 각 카메라 쌍으로 후보 삼각측량
        board_center = np.array(Stage1Config.BOARD_CENTER)

        # 각 detection (왼쪽 카메라 기준)마다 3D 추정
        #   → 다른 카메라들에서 "재투영" 해서 실제 검출된 bbox와 매칭되는지 확인
        # 이게 3-view consistency: 같은 포트의 진짜 3D라면 모든 카메라에서 일치
        PIXEL_MATCH_THRESH = 30.0  # 재투영된 점이 실제 검출과 이만큼 이내여야 OK

        candidates = []
        # 주 카메라 쌍은 베이스라인 큰 것 (left-right 우선)
        main_pair = available_pairs[-1]
        cam_a, cam_b = main_pair

        for da in detections[cam_a]:
            for db in detections[cam_b]:
                port_3d = self._triangulate(
                    da["u"], da["v"], K[cam_a], T_base_to[cam_a],
                    db["u"], db["v"], K[cam_b], T_base_to[cam_b],
                )
                dist = float(np.linalg.norm(port_3d - board_center))
                if dist > Stage1Config.BOARD_RADIUS:
                    continue
                if not (Stage1Config.Z_RANGE[0] <= port_3d[2]
                        <= Stage1Config.Z_RANGE[1]):
                    continue

                # 3-view consistency: 제3의 카메라(center)에 재투영했을 때
                # 그 카메라 검출과 실제로 일치하는가?
                third_cam = None
                for name in ["center", "left", "right"]:
                    if name not in (cam_a, cam_b) and name in detections:
                        third_cam = name
                        break

                consistent = True
                if third_cam and detections[third_cam]:
                    u_proj, v_proj = _project_3d_to_pixel(
                        port_3d, K[third_cam], T_base_to[third_cam]
                    )
                    # 제3카메라의 검출들 중 가장 가까운 것과의 거리
                    min_px_dist = min(
                        np.hypot(d["u"] - u_proj, d["v"] - v_proj)
                        for d in detections[third_cam]
                    )
                    if min_px_dist > PIXEL_MATCH_THRESH:
                        consistent = False

                if not consistent:
                    continue

                conf_sum = da["conf"] + db["conf"]
                score = dist - 0.1 * conf_sum
                candidates.append({
                    "pos": port_3d,
                    "score": score,
                    "conf_sum": conf_sum,
                })

        # score 낮은 순 정렬 (보드 중심 가깝고 conf 높을수록 앞)
        candidates.sort(key=lambda c: c["score"])

        # 중복 제거 (같은 3D 좌표로 수렴한 매칭들 — 1cm 이내면 같은 포트로 간주)
        unique = []
        for c in candidates:
            is_dup = False
            for u in unique:
                if np.linalg.norm(c["pos"] - u["pos"]) < 0.01:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)

        if self._logger and unique:
            self._logger.info(
                f"Vision: {len(unique)}개 후보 포트 3D 추정 "
                f"(best=({unique[0]['pos'][0]:+.3f}, "
                f"{unique[0]['pos'][1]:+.3f}, {unique[0]['pos'][2]:+.3f}))"
            )

        return unique

    @staticmethod
    def select_by_port_name(
        candidates: list, port_name: str
    ) -> Optional[np.ndarray]:
        """Task의 port_name으로 올바른 후보 선택.

        NIC 카드 SFP 포트의 경우:
          sfp_port_0 → x 좌표가 더 큰 쪽 (NIC 카드의 오른쪽)
          sfp_port_1 → x 좌표가 더 작은 쪽 (NIC 카드의 왼쪽)

        하나만 후보면 그걸 반환.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["pos"]

        # SFP 포트 인덱스 기반 구분
        if "sfp_port" in port_name:
            # 후보를 x 좌표 기준으로 정렬
            sorted_by_x = sorted(candidates, key=lambda c: c["pos"][0])
            # port_0: x가 큰 쪽 (리스트 끝)
            # port_1: x가 작은 쪽 (리스트 시작)
            if port_name.endswith("_0") or port_name.endswith("0_link"):
                return sorted_by_x[-1]["pos"]
            elif port_name.endswith("_1") or port_name.endswith("1_link"):
                return sorted_by_x[0]["pos"]

        # SC 포트나 기타는 보드 중심에 가장 가까운 것 (score 최소)
        return candidates[0]["pos"]


# ═══════════════════════════════════════════════════════════
#  StagedPolicy
# ═══════════════════════════════════════════════════════════

class StagedPolicy(Policy):
    """3단계 State Machine 정책 (Vision 통합)."""

    PORT_AXIS_LOCAL: np.ndarray = np.array([0.0, 0.0, 1.0])

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task: Optional[Task] = None

        # PI 적분기
        self._x_integrator = 0.0
        self._y_integrator = 0.0
        self._max_windup = 0.05
        self._i_gain = 0.15

        # Vision 모듈
        self._vision = VisionPortEstimator(
            model_path=Stage1Config.YOLO_MODEL_PATH,
            conf_thresh=Stage1Config.YOLO_CONF_THRESH,
            logger=self.get_logger(),
        )
        # ★ YOLO 모델 즉시 로드 (Stage 1 지연 방지)
        # __init__ 단계는 60초 여유 있음 (lifecycle configured 단계)
        self.get_logger().info("YOLO 모델 사전 로드 중...")
        self.get_logger().info(f"  YOLO model path: {Stage1Config.YOLO_MODEL_PATH}")
        self.get_logger().info(f"  YOLO conf threshold: {Stage1Config.YOLO_CONF_THRESH}")
        self._vision._ensure_loaded()
        self.get_logger().info(f"  YOLO loaded: {self._vision._loaded}")
        if not self._vision._loaded:
            self.get_logger().error(
                "  ✗ YOLO 모델 로드 실패! Vision fallback 불가.\n"
                "  해결: AIC_YOLO_MODEL_PATH 환경변수 설정 또는 모델 파일 경로 확인"
            )
        else:
            self.get_logger().info("  ✓ YOLO 사전 로드 완료")

    # ─────────────────────────────────────────────────────
    #  프레임 이름 / TF 조회
    # ─────────────────────────────────────────────────────

    def _port_frame(self) -> str:
        return (
            f"task_board/{self._task.target_module_name}"
            f"/{self._task.port_name}_link"
        )

    def _plug_frame(self) -> str:
        return f"{self._task.cable_name}/{self._task.plug_name}_link"

    def _wait_for_tf(self, frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    "base_link", frame, Time()
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"TF 대기 중: {frame} "
                        "(ground_truth:=true 환경이면 TF 제공됨)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().warn(f"TF 대기 시간 초과: {frame}")
        return False

    def _lookup_tf(self, frame: str) -> Optional[Transform]:
        for _ in range(Stage1Config.TF_RETRY):
            try:
                return self._parent_node._tf_buffer.lookup_transform(
                    "base_link", frame, Time()
                ).transform
            except TransformException:
                self.sleep_for(Stage1Config.TF_RETRY_DT)
        return None

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return Pose(
            position=Point(
                x=tf.translation.x, y=tf.translation.y, z=tf.translation.z,
            ),
            orientation=Quaternion(
                x=tf.rotation.x, y=tf.rotation.y,
                z=tf.rotation.z, w=tf.rotation.w,
            ),
        )

    # ─────────────────────────────────────────────────────
    #  포트 pose 획득: TF → Vision fallback
    # ─────────────────────────────────────────────────────

    def _get_port_pose(self, get_observation) -> tuple:
        """(port_pose, port_source) 반환.

        port_source:
          "tf": Ground truth TF 사용 (훈련 환경)
          "vision": YOLO + 스테레오 (평가 환경)
          None: 실패
        """
        # 1순위: TF
        port_tf = self._lookup_tf(self._port_frame())
        if port_tf is not None:
            return self._transform_to_pose(port_tf), "tf"

        # 2순위: Vision
        self.get_logger().warn(
            f"TF로 포트 좌표 못 얻음 → Vision 시도"
        )
        obs = get_observation()
        if obs is None:
            return None, None

        # 타겟 클래스 결정 (task.plug_name으로 판단)
        if "sc" in self._task.plug_name.lower():
            target_class_id = 1  # sc_port
        else:
            target_class_id = 0  # sfp_port

        # task.port_name으로 올바른 후보 선택 (SFP 포트 0/1 구분)
        # 예: "sfp_port_0", "sfp_port_1", "sc_port_base"
        port_name_hint = self._task.port_name or ""
        port_3d = self._vision.estimate(
            obs, self._parent_node._tf_buffer, target_class_id,
            port_hint=port_name_hint,
        )
        if port_3d is None:
            return None, None

        self.get_logger().info(
            f"Vision 선택 결과 (port_hint='{port_name_hint}'): "
            f"({port_3d[0]:+.3f}, {port_3d[1]:+.3f}, {port_3d[2]:+.3f})"
        )

        # Vision은 위치만 주고, 방향은 추정 안 함 → 단위 쿼터니언 사용
        # (월드 +z 접근 방식을 쓰므로 방향 정보가 덜 중요)
        pose = Pose(
            position=Point(
                x=float(port_3d[0]), y=float(port_3d[1]), z=float(port_3d[2]),
            ),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return pose, "vision"

    # ─────────────────────────────────────────────────────
    #  Stage 1: 이동 (Vision 통합)
    # ─────────────────────────────────────────────────────

    def _compute_approach_pose(
        self, port_pose: Pose, plug_tf: Optional[Transform],
        gripper_tf: Transform, port_source: str
    ) -> tuple[Pose, np.ndarray]:
        """접근점 pose 계산. (approach_pose, port_axis_world)

        port_source='vision'이면 plug_tf가 None일 수 있으므로 처리.
        """
        # 접근 축 (월드 +z 방향 기본)
        port_axis_world = np.array([0.0, 0.0, 1.0])

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        approach_pos = port_pos + port_axis_world * Stage1Config.Z_OFFSET

        # 플러그-그리퍼 오프셋 (TF 가능한 경우만)
        if plug_tf is not None:
            plug_pos = np.array([
                plug_tf.translation.x,
                plug_tf.translation.y,
                plug_tf.translation.z,
            ])
            gripper_pos = np.array([
                gripper_tf.translation.x,
                gripper_tf.translation.y,
                gripper_tf.translation.z,
            ])
            offset = gripper_pos - plug_pos
            gripper_target_pos = approach_pos + offset
        else:
            # Vision 모드: 플러그 TF 없음 → 대략적 오프셋 (SFP 기준 ~5cm)
            gripper_target_pos = approach_pos + np.array([0.0, 0.015, 0.045])

        # 방향
        if port_source == "tf" and plug_tf is not None:
            # TF 방식: 포트 방향에 맞춤
            q_port = quat_to_tuple(port_pose.orientation)
            q_plug = quat_to_tuple(self._transform_to_pose(plug_tf).orientation)
            q_gripper = quat_to_tuple(self._transform_to_pose(gripper_tf).orientation)
            q_diff = quaternion_multiply(q_port, quat_inverse(q_plug))
            q_target = quaternion_multiply(q_diff, q_gripper)
        else:
            # Vision 모드: 현재 그리퍼 방향 유지 (안전)
            q_target = quat_to_tuple(
                self._transform_to_pose(gripper_tf).orientation
            )

        approach_pose = Pose(
            position=Point(
                x=float(gripper_target_pos[0]),
                y=float(gripper_target_pos[1]),
                z=float(gripper_target_pos[2]),
            ),
            orientation=tuple_to_quat(q_target),
        )
        return approach_pose, port_axis_world

    def _check_stage1_termination(
        self,
        plug_tf: Optional[Transform],
        gripper_tf: Transform,
        port_axis: np.ndarray,
        port_pose: Pose,
        target_z: Optional[float] = None,
    ) -> tuple[bool, str]:
        """target_z가 주어지면 그 값을 목표로 검증 (기본은 Stage 1-A 목표).

        Stage 1-B까지 완료된 경우엔 target_z=Z_OFFSET_MID (3cm) 사용.
        """
        if target_z is None:
            target_z = Stage1Config.Z_OFFSET

        # 플러그 TF 있으면 플러그 기준, 없으면 그리퍼 기준
        if plug_tf is not None:
            ref = np.array([
                plug_tf.translation.x,
                plug_tf.translation.y,
                plug_tf.translation.z,
            ])
            ref_name = "plug"
        else:
            ref = np.array([
                gripper_tf.translation.x,
                gripper_tf.translation.y,
                gripper_tf.translation.z,
            ])
            ref = ref - np.array([0.0, 0.015, 0.045])
            ref_name = "gripper(offset)"

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        to_port = ref - port_pos
        axial = float(np.dot(to_port, port_axis))
        axial_err = abs(axial - target_z)
        radial = float(np.linalg.norm(to_port - axial * port_axis))

        info = (
            f"{ref_name} axial={axial*100:.1f}cm "
            f"(target {target_z*100:.0f}, err {axial_err*100:.1f}), "
            f"radial={radial*100:.1f}cm"
        )

        if axial_err > Stage1Config.Z_OFFSET_TOLERANCE:
            return False, f"axial_err too large: {info}"
        if radial > Stage1Config.XY_TOLERANCE:
            return False, f"radial_err too large: {info}"
        return True, info

    def _stage1_approach(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> Stage1Result:
        self.get_logger().info("━━━ Stage 1: 이동 시작 ━━━")
        send_feedback("Stage 1: approaching port axis")

        # 0. F/T baseline (포트 획득 전에 측정)
        baseline_force_mag = 0.0
        init_obs = get_observation()
        if init_obs is not None:
            f = init_obs.wrist_wrench.wrench.force
            baseline_force_mag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
            self.get_logger().info(
                f"  F/T baseline: {baseline_force_mag:.2f}N"
            )

        # 1. 포트 pose 획득 (TF → Vision fallback) — 시간 측정 전
        port_pose, port_source = self._get_port_pose(get_observation)
        if port_pose is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=None, port_axis=None,
                elapsed_time=0.0,
                failure_reason="포트 좌표 획득 실패 (TF/Vision 모두 실패)",
                port_source="none",
            )

        self.get_logger().info(
            f"  포트 좌표 소스: {port_source}\n"
            f"  포트 위치: ({port_pose.position.x:+.3f}, "
            f"{port_pose.position.y:+.3f}, {port_pose.position.z:+.3f})"
        )

        # ★ 시간 측정은 실제 이동 시작 전에 개시
        t0 = self.time_now()

        # 2. 플러그 / 그리퍼 TF (Vision 모드에서도 플러그 TF는 시도)
        plug_tf = self._lookup_tf(self._plug_frame())  # None 가능
        gripper_tf = self._lookup_tf("gripper/tcp")
        if gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=None, elapsed_time=0.0,
                failure_reason="gripper TF 조회 실패",
                port_source=port_source,
            )

        # 3. 접근점 계산
        approach_pose, port_axis = self._compute_approach_pose(
            port_pose, plug_tf, gripper_tf, port_source
        )
        self.get_logger().info(
            f"  접근점: ({approach_pose.position.x:+.3f}, "
            f"{approach_pose.position.y:+.3f}, {approach_pose.position.z:+.3f})"
        )

        # 4. S-curve 직선 보간
        start_pose = self._transform_to_pose(gripper_tf)
        q_start = quat_to_tuple(start_pose.orientation)
        q_end = quat_to_tuple(approach_pose.orientation)
        p_start = np.array([
            start_pose.position.x, start_pose.position.y, start_pose.position.z
        ])
        p_end = np.array([
            approach_pose.position.x,
            approach_pose.position.y,
            approach_pose.position.z,
        ])

        for i in range(Stage1Config.N_STEPS):
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed > Stage1Config.MAX_DURATION_S:
                return Stage1Result(
                    success=False, final_pose=None, port_pose=port_pose,
                    port_axis=port_axis, elapsed_time=elapsed,
                    failure_reason="timeout", port_source=port_source,
                )

            # 충돌 체크
            obs = get_observation()
            if obs is not None:
                f = obs.wrist_wrench.wrench.force
                fmag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
                fdelta = fmag - baseline_force_mag
                if fdelta > Stage1Config.FORCE_DELTA_LIMIT_N:
                    self.get_logger().warn(
                        f"충돌 감지: {fmag:.1f}N (baseline+{fdelta:.1f}N)"
                    )
                    return Stage1Result(
                        success=False, final_pose=None, port_pose=port_pose,
                        port_axis=port_axis, elapsed_time=elapsed,
                        failure_reason=f"collision (+{fdelta:.1f}N)",
                        port_source=port_source,
                    )

            # 5차 Hermite (옵션: 3차). 끝 가속도 0 → 관성 떨림 감소
            t_smooth = interp_profile(
                (i + 1) / Stage1Config.N_STEPS,
                quintic=Stage1Config.USE_QUINTIC_HERMITE,
            )
            pos = p_start * (1.0 - t_smooth) + p_end * t_smooth
            q = quaternion_slerp(q_start, q_end, t_smooth)

            waypoint = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=tuple_to_quat(q),
            )
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=waypoint,
                    stiffness=list(Stage1Config.STIFFNESS),
                    damping=list(Stage1Config.DAMPING),
                )
            except TransformException as ex:
                self.get_logger().warn(f"Stage 1 TF 오류: {ex}")

            self.sleep_for(Stage1Config.DT)

        # 4.5. Stage 1-A 후 안정화 대기 (관성 떨림 완화)
        if Stage1Config.SETTLE_AFTER_STAGE1A > 0:
            self.get_logger().info(
                f"  Stage 1-A 안정화 대기 {Stage1Config.SETTLE_AFTER_STAGE1A:.1f}s"
            )
            # 마지막 waypoint 재전송으로 holding (떨림 감소)
            last_pose = Pose(
                position=Point(x=float(p_end[0]), y=float(p_end[1]), z=float(p_end[2])),
                orientation=tuple_to_quat(q_end),
            )
            settle_end = time.time() + Stage1Config.SETTLE_AFTER_STAGE1A
            while time.time() < settle_end:
                try:
                    self.set_pose_target(
                        move_robot=move_robot, pose=last_pose,
                        stiffness=list(Stage1Config.STIFFNESS),
                        damping=list(Stage1Config.DAMPING),
                    )
                except TransformException:
                    pass
                self.sleep_for(0.05)

        # ═════════════════════════════════════════════════════════
        # 4-B. Stage 1-B: Mid Approach (7cm → 3cm 하강 + 정렬)
        # ═════════════════════════════════════════════════════════
        if Stage1Config.ENABLE_STAGE1B:
            self.get_logger().info("━━━ Stage 1-B: 중간 접근 시작 (7→3cm) ━━━")
            send_feedback("Stage 1-B: mid approach")

            # Stage 1-B 시작점 = 현재 그리퍼 (= 접근점 근처)
            mid_start_tf = self._lookup_tf("gripper/tcp")
            if mid_start_tf is None:
                self.get_logger().warn("Stage 1-B: gripper TF 조회 실패, 건너뜀")
            else:
                mid_start_pose = self._transform_to_pose(mid_start_tf)
                p_start_mid = np.array([
                    mid_start_pose.position.x,
                    mid_start_pose.position.y,
                    mid_start_pose.position.z,
                ])
                q_start_mid = quat_to_tuple(mid_start_pose.orientation)

                for i in range(Stage1Config.N_STEPS_MID):
                    elapsed = (self.time_now() - t0).nanoseconds / 1e9
                    if elapsed > Stage1Config.MAX_DURATION_S:
                        return Stage1Result(
                            success=False, final_pose=None,
                            port_pose=port_pose, port_axis=port_axis,
                            elapsed_time=elapsed,
                            failure_reason="timeout (stage 1-B)",
                            port_source=port_source,
                        )

                    # 충돌 체크
                    obs = get_observation()
                    if obs is not None:
                        f = obs.wrist_wrench.wrench.force
                        fmag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
                        fdelta = fmag - baseline_force_mag
                        if fdelta > Stage1Config.FORCE_DELTA_LIMIT_N:
                            self.get_logger().warn(
                                f"Stage 1-B 충돌 감지: {fmag:.1f}N "
                                f"(baseline+{fdelta:.1f}N)"
                            )
                            return Stage1Result(
                                success=False, final_pose=None,
                                port_pose=port_pose, port_axis=port_axis,
                                elapsed_time=elapsed,
                                failure_reason=f"stage1b collision (+{fdelta:.1f}N)",
                                port_source=port_source,
                            )

                    # FEEDBACK: 매 스텝 포트 TF 재조회 (있으면) — 드리프트 보정
                    if Stage1Config.FEEDBACK_MID and port_source == "tf":
                        fresh_port_tf = self._lookup_tf(self._port_frame())
                        fresh_gripper_tf = self._lookup_tf("gripper/tcp")
                        fresh_plug_tf = self._lookup_tf(self._plug_frame())
                        if fresh_port_tf is not None and fresh_gripper_tf is not None:
                            # 현재 포트 기준 접근점 재계산 (더 낮은 z_offset)
                            fresh_port_pose = self._transform_to_pose(fresh_port_tf)
                            # 임시로 Z_OFFSET을 Z_OFFSET_MID로 바꿔 계산
                            saved_z = Stage1Config.Z_OFFSET
                            Stage1Config.Z_OFFSET = Stage1Config.Z_OFFSET_MID
                            mid_target_pose, _ = self._compute_approach_pose(
                                fresh_port_pose, fresh_plug_tf,
                                fresh_gripper_tf, port_source,
                            )
                            Stage1Config.Z_OFFSET = saved_z
                            p_end_mid = np.array([
                                mid_target_pose.position.x,
                                mid_target_pose.position.y,
                                mid_target_pose.position.z,
                            ])
                            q_end_mid = quat_to_tuple(
                                mid_target_pose.orientation
                            )
                        else:
                            # fallback: 처음 p_end에서 z만 내림
                            p_end_mid = p_end.copy()
                            p_end_mid[2] -= (
                                Stage1Config.Z_OFFSET - Stage1Config.Z_OFFSET_MID
                            )
                            q_end_mid = q_end
                    else:
                        # Vision 모드: 처음 p_end에서 z만 내림
                        p_end_mid = p_end.copy()
                        p_end_mid[2] -= (
                            Stage1Config.Z_OFFSET - Stage1Config.Z_OFFSET_MID
                        )
                        q_end_mid = q_end

                    # [Cable tension compensation] SFP 만 14mm 더 내림
                    # 근거: 측정된 13.6mm steady-state err ≈ 2N 장력 / 150 N/m
                    plug_name = (self._task.plug_name or "").lower()
                    if "sfp" in plug_name:
                        z_before = p_end_mid[2]
                        p_end_mid[2] -= Stage1Config.SFP_CABLE_TENSION_COMPENSATION
                        if i == 0:
                            # 진단: 실제로 얼마나 낮게 명령되는지 확인
                            self.get_logger().info(
                                f"  [SFP] cable tension compensation applied: "
                                f"gripper z {z_before:.4f}m → {p_end_mid[2]:.4f}m "
                                f"(Δ=-{Stage1Config.SFP_CABLE_TENSION_COMPENSATION*1000:.0f}mm)"
                            )

                    t_norm = (i + 1) / Stage1Config.N_STEPS_MID
                    t_smooth = interp_profile(
                        t_norm, quintic=Stage1Config.USE_QUINTIC_HERMITE
                    )
                    pos_mid = p_start_mid * (1.0 - t_smooth) + p_end_mid * t_smooth
                    q_mid = quaternion_slerp(q_start_mid, q_end_mid, t_smooth)

                    waypoint_mid = Pose(
                        position=Point(
                            x=float(pos_mid[0]),
                            y=float(pos_mid[1]),
                            z=float(pos_mid[2]),
                        ),
                        orientation=tuple_to_quat(q_mid),
                    )
                    try:
                        self.set_pose_target(
                            move_robot=move_robot,
                            pose=waypoint_mid,
                            stiffness=list(Stage1Config.STIFFNESS_MID),
                            damping=list(Stage1Config.DAMPING_MID),
                        )
                    except TransformException as ex:
                        self.get_logger().warn(f"Stage 1-B TF 오류: {ex}")

                    self.sleep_for(Stage1Config.DT_MID)

                # ─ Stage 1-B 수렴 대기 (위치 + 안정성 기반) ─
                # 근거:
                #   1) compensation 으로 정적 offset 제거됨 → target 정확 도달 기대
                #   2) 그러나 cable 떨림(oscillation) 가능 → "err < tol 이
                #      연속 N회" 로 안정 상태 확인
                # 단순 "순간 err < tol" 이면 진동 중 우연히 낮은 순간에 통과할
                # 수 있음 → 연속 체크로 진짜 수렴만 인정.
                # 진단 로그: 명령 vs 목표
                expected_plug_z = None
                if port_pose is not None:
                    expected_plug_z = port_pose.position.z + Stage1Config.Z_OFFSET_MID
                self.get_logger().info(
                    f"  Stage 1-B 수렴 대기 진단: "
                    f"commanded gripper z = {p_end_mid[2]:.4f}m, "
                    f"desired plug z = {expected_plug_z:.4f}m "
                    f"(port_z={port_pose.position.z:.4f} + {Stage1Config.Z_OFFSET_MID:.3f})"
                )
                self.get_logger().info(
                    f"  수렴 기준: err ≤ {Stage1Config.STAGE1B_CONVERGENCE_TOL_M*1000:.0f}mm "
                    f"× {Stage1Config.STAGE1B_STABLE_CONSECUTIVE}회 연속 or "
                    f"{Stage1Config.STAGE1B_CONVERGENCE_MAX_WAIT_S:.1f}s"
                )
                hold_pose = Pose(
                    position=Point(
                        x=float(p_end_mid[0]),
                        y=float(p_end_mid[1]),
                        z=float(p_end_mid[2]),
                    ),
                    orientation=tuple_to_quat(q_end_mid),
                )
                convergence_tol = Stage1Config.STAGE1B_CONVERGENCE_TOL_M
                max_wait = Stage1Config.STAGE1B_CONVERGENCE_MAX_WAIT_S
                stable_needed = Stage1Config.STAGE1B_STABLE_CONSECUTIVE
                wait_end = time.time() + max_wait
                stable_count = 0
                last_err = None
                converged = False
                wait_start = time.time()
                last_log_time = 0.0
                # [옵션 A] 수렴 대기 구간에서 Z-stiffness 부스트 (150→500 N/m)
                # 케이블 평형점 극복 목적 — S-curve 본체는 낮은 K 그대로 유지.
                plug_is_sfp = "sfp" in (self._task.plug_name or "").lower()
                hold_stiffness = (
                    Stage1Config.STIFFNESS_MID_BOOST if plug_is_sfp
                    else Stage1Config.STIFFNESS_MID
                )
                hold_damping = (
                    Stage1Config.DAMPING_MID_BOOST if plug_is_sfp
                    else Stage1Config.DAMPING_MID
                )
                if plug_is_sfp:
                    self.get_logger().info(
                        f"  [SFP] 수렴 대기 Z-stiffness 부스트: "
                        f"{Stage1Config.STIFFNESS_MID[2]:.0f} → "
                        f"{Stage1Config.STIFFNESS_MID_BOOST[2]:.0f} N/m"
                    )
                while time.time() < wait_end:
                    try:
                        self.set_pose_target(
                            move_robot=move_robot, pose=hold_pose,
                            stiffness=list(hold_stiffness),
                            damping=list(hold_damping),
                        )
                    except TransformException:
                        pass
                    self.sleep_for(0.05)

                    # 수렴 체크: 실제 plug z vs TARGET z
                    cur_plug_tf = self._lookup_tf(self._plug_frame())
                    cur_gripper_tf = self._lookup_tf("gripper/tcp")
                    if cur_plug_tf is not None and port_pose is not None:
                        desired_plug_z = (
                            port_pose.position.z + Stage1Config.Z_OFFSET_MID
                        )
                        cur_axial = abs(
                            cur_plug_tf.translation.z - desired_plug_z
                        )
                        last_err = cur_axial

                        # [진단] 0.3초마다 실시간 상태 출력
                        elapsed = time.time() - wait_start
                        if elapsed - last_log_time >= 0.3:
                            gripper_z_str = (
                                f"{cur_gripper_tf.translation.z:.4f}"
                                if cur_gripper_tf else "N/A"
                            )
                            self.get_logger().info(
                                f"    [t={elapsed:.1f}s] "
                                f"plug_z={cur_plug_tf.translation.z:.4f}m, "
                                f"gripper_z={gripper_z_str}, "
                                f"err={cur_axial*1000:.1f}mm, "
                                f"stable={stable_count}"
                            )
                            last_log_time = elapsed

                        if cur_axial < convergence_tol:
                            stable_count += 1
                            if stable_count >= stable_needed:
                                converged = True
                                self.get_logger().info(
                                    f"  수렴 완료: axial err "
                                    f"{cur_axial*1000:.1f}mm × {stable_count}회 연속"
                                )
                                break
                        else:
                            stable_count = 0  # 떨림 중 — 카운터 reset
                if not converged and last_err is not None:
                    self.get_logger().warn(
                        f"  수렴 대기 타임아웃 ({max_wait:.1f}s): "
                        f"최종 err {last_err*1000:.1f}mm, "
                        f"stable_count={stable_count} (필요 {stable_needed})"
                    )

                self.get_logger().info("━━━ Stage 1-B: 중간 접근 완료 ━━━")

        # 5. 종료 검증
        final_plug_tf = self._lookup_tf(self._plug_frame())
        final_gripper_tf = self._lookup_tf("gripper/tcp")
        if final_gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=port_axis,
                elapsed_time=(self.time_now() - t0).nanoseconds / 1e9,
                failure_reason="final TF lookup failed",
                port_source=port_source,
            )

        # Stage 1-B가 실행됐으면 최종 목표는 Z_OFFSET_MID (3cm)
        final_target_z = (
            Stage1Config.Z_OFFSET_MID if Stage1Config.ENABLE_STAGE1B
            else Stage1Config.Z_OFFSET
        )
        ok, info = self._check_stage1_termination(
            final_plug_tf, final_gripper_tf, port_axis, port_pose,
            target_z=final_target_z,
        )
        elapsed = (self.time_now() - t0).nanoseconds / 1e9

        self.get_logger().info(
            f"━━━ Stage 1: 완료 ━━━ "
            f"(source={port_source}, {info}, elapsed {elapsed:.2f}s, ok={ok})"
        )

        return Stage1Result(
            success=ok,
            final_pose=self._transform_to_pose(final_gripper_tf),
            port_pose=port_pose,
            port_axis=port_axis,
            elapsed_time=elapsed,
            failure_reason=None if ok else f"spec not met: {info}",
            port_source=port_source,
        )

    # ─────────────────────────────────────────────────────
    #  Stage 2/3: 임시 (기존과 동일)
    # ─────────────────────────────────────────────────────

    def _reset_integrator(self):
        self._x_integrator = 0.0
        self._y_integrator = 0.0

    def _compute_stage23_pose(self, z_offset, use_integrator=False,
                              port_pose_vision: Optional[Pose] = None):
        """Stage 2/3용 목표 pose.

        port_pose_vision이 제공되면 Vision 결과 사용 (ground_truth=false 환경).
        아니면 TF 기반 CheatCode 방식 (ground_truth=true 환경).
        """
        gripper_tf = self._lookup_tf("gripper/tcp")
        if gripper_tf is None:
            return None

        # Vision 모드라면 TF lookup 건너뛰기 (무한 재시도 방지)
        if port_pose_vision is not None:
            port_tf = None
            plug_tf = None
        else:
            port_tf = self._lookup_tf(self._port_frame())
            plug_tf = self._lookup_tf(self._plug_frame())

        if port_tf is not None and plug_tf is not None:
            # ── TF 경로 (ground_truth=true) ──
            q_port = quat_to_tuple(self._transform_to_pose(port_tf).orientation)
            q_plug = quat_to_tuple(self._transform_to_pose(plug_tf).orientation)
            q_gripper = quat_to_tuple(
                self._transform_to_pose(gripper_tf).orientation
            )
            q_diff = quaternion_multiply(q_port, quat_inverse(q_plug))
            q_target = quaternion_multiply(q_diff, q_gripper)

            offset_z = gripper_tf.translation.z - plug_tf.translation.z
            tip_x_err = port_tf.translation.x - plug_tf.translation.x
            tip_y_err = port_tf.translation.y - plug_tf.translation.y

            if use_integrator:
                self._x_integrator = np.clip(
                    self._x_integrator + tip_x_err,
                    -self._max_windup, self._max_windup,
                )
                self._y_integrator = np.clip(
                    self._y_integrator + tip_y_err,
                    -self._max_windup, self._max_windup,
                )
                tx = port_tf.translation.x + self._i_gain * self._x_integrator
                ty = port_tf.translation.y + self._i_gain * self._y_integrator
            else:
                tx = port_tf.translation.x
                ty = port_tf.translation.y
            tz = port_tf.translation.z + z_offset + offset_z

            return Pose(
                position=Point(x=float(tx), y=float(ty), z=float(tz)),
                orientation=tuple_to_quat(q_target),
            )

        elif port_pose_vision is not None:
            # ── Vision 경로 (ground_truth=false) ──
            # 플러그 TF 없음 → 대략적 오프셋으로 그리퍼 목표 생성
            q_gripper = quat_to_tuple(
                self._transform_to_pose(gripper_tf).orientation
            )
            # 포트 위치 + z_offset + 그리퍼-플러그 대략 오프셋
            tx = port_pose_vision.position.x
            ty = port_pose_vision.position.y + 0.015  # 플러그-그리퍼 y 오프셋
            tz = port_pose_vision.position.z + z_offset + 0.045  # z 오프셋

            return Pose(
                position=Point(x=float(tx), y=float(ty), z=float(tz)),
                orientation=tuple_to_quat(q_gripper),
            )

        return None

    def _stage2_align(self, move_robot, send_feedback,
                      port_pose_vision=None):
        self.get_logger().info("━━━ Stage 2: 정렬 시작 ━━━")
        send_feedback("Stage 2: aligning to port")
        self._reset_integrator()

        Z_START, Z_END, N = 0.10, 0.005, 100
        for i in range(N):
            t = (i + 1) / N
            z = Z_START + t * (Z_END - Z_START)
            pose = self._compute_stage23_pose(
                z_offset=z, use_integrator=True,
                port_pose_vision=port_pose_vision,
            )
            if pose is not None:
                self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.05)

        self.get_logger().info("━━━ Stage 2: 정렬 완료 ━━━")

    def _stage3_insert(self, get_observation, move_robot, send_feedback,
                       port_pose_vision=None):
        self.get_logger().info("━━━ Stage 3: 삽입 시작 ━━━")
        send_feedback("Stage 3: inserting cable")

        FORCE_LIMIT = 18.0
        z = 0.005
        stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

        while z > -0.015:
            z -= 0.0005
            obs = get_observation()
            if obs is not None:
                f = obs.wrist_wrench.wrench.force
                if np.sqrt(f.x**2 + f.y**2 + f.z**2) > FORCE_LIMIT:
                    self.sleep_for(0.2)
                    continue

            pose = self._compute_stage23_pose(
                z_offset=z, use_integrator=True,
                port_pose_vision=port_pose_vision,
            )
            if pose is not None:
                self.set_pose_target(
                    move_robot=move_robot, pose=pose,
                    stiffness=stiffness, damping=damping,
                )
            self.sleep_for(0.05)

        self.sleep_for(3.0)
        self.get_logger().info("━━━ Stage 3: 삽입 완료 ━━━")

    # ─────────────────────────────────────────────────────
    #  메인
    # ─────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            f"StagedPolicy (Vision 통합) 시작\n"
            f"  cable={task.cable_name}, plug={task.plug_name}\n"
            f"  port={task.port_name}, target={task.target_module_name}"
        )
        self._task = task

        # [신규] Pre-Stage settle — Trial 1 초기 cable/physics 불안정 완화
        # 근거:
        #   Trial 1 은 cable 이 막 gripper 에 attach 된 직후 시작되어
        #   flexible cable 이 아직 흔들리는 상태. 관측: Trial 1 이 다른
        #   trial 대비 Stage 1 axial err 일관되게 크고 rim 걸림 많음.
        # 효과 가설:
        #   0.8 초 정지 대기 → cable 이 중력으로 안정화 → 첫 이동 시 tracking 향상.
        # 비용: 시간 점수 영향 미미 (max 점수의 ~3%).
        self.get_logger().info("Pre-Stage settle (0.8s) — cable 안정화 대기")
        self.sleep_for(0.8)

        # TF 대기 (training 모드에서만; 평가 모드에선 실패해도 Vision으로 진행)
        # 짧게 대기 (최대 1초) — 없으면 바로 Vision으로
        self._wait_for_tf(self._port_frame(), timeout_sec=1.0)

        # Stage 1 (Vision 자동 fallback)
        result = self._stage1_approach(get_observation, move_robot, send_feedback)
        self.get_logger().info(
            f"Stage 1 result: success={result.success}, "
            f"source={result.port_source}, "
            f"elapsed={result.elapsed_time:.2f}s, "
            f"reason={result.failure_reason}"
        )

        # TF/Vision 둘 다 실패한 경우 → Stage 2/3는 항상 None pose만 반환하므로
        # 시간만 낭비. 즉시 종료하여 평가 시스템이 다음 태스크로 넘어갈 수 있도록 함.
        if result.port_source == "none":
            self.get_logger().error(
                "포트 좌표 획득 완전 실패 (TF/Vision 모두 실패) → 조기 종료"
            )
            send_feedback("failed: port not detected (skipping stage 2/3)")
            return False

        # Stage 2/3 — Vision 모드면 Stage1 result의 port_pose를 전달
        # (TF 없어서 _compute_stage23_pose 내부에서 무한 대기 방지)
        port_pose_for_23 = None
        if result.port_source == "vision":
            port_pose_for_23 = result.port_pose

        try:
            self._stage2_align(move_robot, send_feedback,
                               port_pose_vision=port_pose_for_23)
            self._stage3_insert(get_observation, move_robot, send_feedback,
                                port_pose_vision=port_pose_for_23)
        except Exception as ex:
            self.get_logger().warn(f"Stage 2/3 실행 중 예외: {ex}")

        self.get_logger().info("StagedPolicy 완료")
        return True
