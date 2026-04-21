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
  pixi reinstall ros-kilted-my-policy-node
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=my_policy_node.StagedPolicy
"""

import os
from dataclasses import dataclass
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

def _resolve_yolo_model_path() -> str:
    # 1순위: 환경 변수 (팀원마다 경로가 다를 수 있으므로 권장)
    env = os.environ.get("AIC_YOLO_MODEL_PATH")
    if env and os.path.isfile(env):
        return env

    # 2순위: 홈 디렉토리 fallback (개발 편의용)
    return "/home/swlinux/aic_sejong/aic_data/model/yolo/weight/best.pt"


class Stage1Config:
    # --- 목표 위치 사양 ---
    # 접근점은 포트 축선상 거리 (팀 피드백 반영: 10cm → 7cm 로 하향)
    Z_OFFSET: float = 0.07
    Z_OFFSET_TOLERANCE: float = 0.015
    XY_TOLERANCE: float = 0.025

    # --- 방향 사양 ---
    AXIS_TOLERANCE_RAD: float = 0.087
    ROLL_TOLERANCE_RAD: float = 0.175

    # --- 속도 사양 ---
    VEL_TOLERANCE_LIN: float = 0.01
    VEL_TOLERANCE_ANG: float = 0.1

    # --- 동작 ---
    N_STEPS: int = 80
    DT: float = 0.05

    # --- 제어 ---
    STIFFNESS: tuple = (200.0, 200.0, 200.0, 50.0, 50.0, 50.0)
    DAMPING: tuple = (80.0, 80.0, 80.0, 20.0, 20.0, 20.0)

    # --- 안전 / 실패 처리 ---
    FORCE_DELTA_LIMIT_N: float = 15.0
    MAX_DURATION_S: float = 8.0
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
    return 3.0 * t * t - 2.0 * t * t * t


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
    # Native: /home/swlinux/aic_sejong/debug/yolo_detections
    DEBUG_SAVE_DIR: str = os.environ.get(
        "AIC_DEBUG_SAVE_DIR", "/home/swlinux/aic_sejong/debug/yolo_detections"
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

    def estimate(self, obs, tf_buffer, target_class_id: int) -> Optional[np.ndarray]:
        """포트 3D 좌표 추정.

        Args:
            obs: aic observation (left/center/right image + camera info 포함)
            tf_buffer: TF buffer (카메라 외부 파라미터용)
            target_class_id: 0=sfp, 1=sc

        Returns:
            (x, y, z) in base_link, or None if failed
        """
        self._ensure_loaded()
        if not self._loaded:
            return None

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
                return None

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
            return None

        # 3. 카메라 쌍 선택 (베이스라인 큰 순)
        pair = None
        for a, b in [("left", "right"), ("left", "center"), ("center", "right")]:
            if a in cams_with_dets and b in cams_with_dets:
                pair = (a, b)
                break
        if pair is None:
            return None

        cam_a, cam_b = pair
        K_a = np.array(cam_infos[cam_a].k).reshape(3, 3)
        K_b = np.array(cam_infos[cam_b].k).reshape(3, 3)
        T_base_to_a = np.linalg.inv(cam_T_in_base[cam_a])
        T_base_to_b = np.linalg.inv(cam_T_in_base[cam_b])

        # 4. 가능한 모든 매칭으로 삼각측량 → 타당성 검증
        board_center = np.array(Stage1Config.BOARD_CENTER)
        best = None
        best_score = float("inf")

        for da in detections[cam_a]:
            for db in detections[cam_b]:
                port_3d = self._triangulate(
                    da["u"], da["v"], K_a, T_base_to_a,
                    db["u"], db["v"], K_b, T_base_to_b,
                )
                # 보드 근처 + z 범위 검증
                dist = float(np.linalg.norm(port_3d - board_center))
                if dist > Stage1Config.BOARD_RADIUS:
                    continue
                if not (Stage1Config.Z_RANGE[0] <= port_3d[2]
                        <= Stage1Config.Z_RANGE[1]):
                    continue

                conf_sum = da["conf"] + db["conf"]
                score = dist - 0.1 * conf_sum
                if score < best_score:
                    best_score = score
                    best = port_3d

        if best is None:
            if self._logger:
                self._logger.warn("Vision: 타당한 매칭 없음 "
                                  "(3D 타당성 검증 실패)")
            return None

        if self._logger:
            self._logger.info(
                f"Vision: 포트 3D 추정 성공 "
                f"({best[0]:+.3f}, {best[1]:+.3f}, {best[2]:+.3f})"
            )
        return best


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

        port_3d = self._vision.estimate(
            obs, self._parent_node._tf_buffer, target_class_id
        )
        if port_3d is None:
            return None, None

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
    ) -> tuple[bool, str]:
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
            # 그리퍼 기준일 때는 오프셋 보정
            ref = ref - np.array([0.0, 0.015, 0.045])
            ref_name = "gripper(offset)"

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        to_port = ref - port_pos
        axial = float(np.dot(to_port, port_axis))
        axial_err = abs(axial - Stage1Config.Z_OFFSET)
        radial = float(np.linalg.norm(to_port - axial * port_axis))

        info = (
            f"{ref_name} axial={axial*100:.1f}cm "
            f"(target {Stage1Config.Z_OFFSET*100:.0f}, err {axial_err*100:.1f}), "
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

            t_smooth = s_curve((i + 1) / Stage1Config.N_STEPS)
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

        ok, info = self._check_stage1_termination(
            final_plug_tf, final_gripper_tf, port_axis, port_pose
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
