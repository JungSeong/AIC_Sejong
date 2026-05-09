"""
DataCollect2 policy (YOLO-guided Phase 1-A)
──────────────────────────
Phase 1-A 접근 정렬을 포트 TF 대신 YOLO bbox 기반 이미지 오차로 수행하는 정책 파일.
"""

import atexit
import json
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform, Vector3, Wrench, Pose

from .lib.recording import (
    LeRobotRecorder,
    LEROBOT_FEATURES,
    _LEROBOT_AVAILABLE,
)
from .lib.cheatcode import CheatCodePlanner
from tf2_ros import TransformException


# ── 유틸리티 함수 ────────────────────────────────────────────────────────

def s_curve_quintic(t: float) -> float:
    """0~1 진행률을 부드러운 S-curve 값으로 바꿔 로봇 이동 시작/끝을 완만하게 만든다."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    """입력 진행률 t를 로봇 pose 보간에 사용할 진행률로 변환한다."""
    return s_curve_quintic(t) if quintic else (3.0 * t**2 - 2.0 * t**3)


def _quat_to_matrix_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """ROS 형식 xyzw quaternion을 3x3 회전 행렬로 변환한다."""
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def _matrix_from_translation_quat(translation, quat_xyzw) -> np.ndarray:
    """translation과 xyzw quaternion을 하나의 4x4 좌표 변환 행렬로 묶는다."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_matrix_xyzw(*quat_xyzw)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _matrix_from_pose(pose: Pose) -> np.ndarray:
    """geometry_msgs/Pose를 행렬 곱에 사용할 수 있는 4x4 좌표 변환 행렬로 바꾼다."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_matrix_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def _quat_multiply_xyzw(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """xyzw quaternion 두 개를 곱해 회전을 합성한다."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_from_axis_angle_xyzw(axis_xyz: np.ndarray, angle_rad: float) -> tuple[float, float, float, float]:
    """base_link 기준 회전축과 각도를 xyzw quaternion으로 변환한다."""
    axis = np.asarray(axis_xyz, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    axis /= norm
    half = 0.5 * float(angle_rad)
    sin_half = float(np.sin(half))
    return (
        float(axis[0] * sin_half),
        float(axis[1] * sin_half),
        float(axis[2] * sin_half),
        float(np.cos(half)),
    )


class DataCollect2(Policy):
    """
    YOLO 기반 접근 정렬 데이터 수집 정책 클래스.
    """

    # ── 임피던스 설정 (성공률 최적화) ──────────────────────────────────────────
    _STIFFNESS_DEFAULT = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    _DAMPING_DEFAULT   = [40.0, 40.0, 40.0, 20.0, 20.0, 20.0]

    # ── 모션 플래닝 상수 ──────────────────────────────────────────────────
    current_approach_z: float = 0.00
    _TRIANGULATION_STOP_Z_OFFSET_DEFAULT: float = 0.020  # 2cm
    _TOOL0_TO_TCP_Z: float = 0.1965
    _TOOL0_TO_OPTICAL = {
        "left": (
            [-0.100516584, -0.058032593, -0.008935891],
            [-0.113039947, 0.065265728, -0.495722390, 0.858616135],
        ),
        "center": (
            [-0.000000001, -0.116079183, -0.008937891],
            [-0.130528330, 0.000001827, -0.000000288, 0.991444580],
        ),
        "right": (
            [0.100516583, -0.058032595, -0.008935891],
            [-0.113041775, -0.065262563, 0.495721890, 0.858616424],
        ),
    }

    def __init__(self, parent_node):
        """정책 실행에 필요한 planner, YOLO, 데이터셋, 토픽 구독, 종료 처리를 초기화한다."""
        super().__init__(parent_node)
        
        # 1. 상태 및 제어 변수 초기화
        self._task: Optional[Task] = None
        # 나선형 진동(Oscillation) 방지를 위해 적분 한계를 낮추고(안전 범위), 부족한 힘은 Stiffness로 보상
        self._max_integrator_windup = 0.15 
        
        # 2. 제어 주기 및 경로 설정
        fps = int(os.environ.get("AIC_LEROBOT_FPS", "0"))
        self.step_sleep_sec = 1.0 / (fps if fps > 0 else 20.0)
        self.capture_root = Path(os.environ.get("AIC_CAPTURE_DIR", "/tmp/aic_episodes"))
        
        # 3. 플래너 및 환경 설정
        self._planner = CheatCodePlanner(
            i_gain=float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15")),
            max_integrator_windup=self._max_integrator_windup,
        )
        self.approach_steps = int(os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_STEPS", "100"))
        self.collect_steps = int(os.environ.get("AIC_COLLECT_STEPS", "160"))
        self.collect_turns = float(os.environ.get("AIC_COLLECT_TURNS", "2.0"))
        self.collect_start_radius = float(os.environ.get("AIC_COLLECT_START_RADIUS", "0.020"))
        self.collect_end_radius = float(os.environ.get("AIC_COLLECT_END_RADIUS", "0.0"))
        self.collect_rotate_angle = float(os.environ.get("AIC_COLLECT_ROTATE_ANGLE", "0.0"))
        self.collect_pattern = os.environ.get("AIC_COLLECT_PATTERN", "spiral").strip().lower()
        if self.collect_pattern not in {"gaussian", "spiral"}:
            self.get_logger().warn(
                f"[DataCollect2] Invalid AIC_COLLECT_PATTERN={self.collect_pattern}; using spiral"
            )
            self.collect_pattern = "spiral"
        self.collect_gaussian_sigma = float(os.environ.get("AIC_COLLECT_GAUSSIAN_SIGMA", "0.006"))
        self.collect_gaussian_max_radius = float(os.environ.get("AIC_COLLECT_GAUSSIAN_MAX_RADIUS", str(self.collect_start_radius)))
        seed_text = os.environ.get("AIC_COLLECT_RANDOM_SEED", "").strip()
        seed = int(seed_text) if seed_text else None
        self._collect_rng = np.random.default_rng(seed)

        # 4. YOLO 및 데이터셋 설정
        self._yolo_model_path = str(Path(__file__).resolve().parents[6] / "model" / "ais_yolo" / "weights" / "best.pt")
        self._lerobot_dataset = None
        self._lerobot_full_repo_id = os.environ.get("AIC_LEROBOT_REPO_ID", "").strip()
        self._lerobot_version = os.environ.get("AIC_LEROBOT_VERSION", "master").strip()
        self._yolo_trigger_conf = float(os.environ.get("AIC_YOLO_TRIGGER_CONF", "0.7"))
        self._yolo_align_conf = float(os.environ.get("AIC_YOLO_ALIGN_CONF", str(self._yolo_trigger_conf)))
        self._yolo_align_gain_x = float(os.environ.get("AIC_YOLO_ALIGN_GAIN_X", "0.035"))
        self._yolo_align_gain_y = float(os.environ.get("AIC_YOLO_ALIGN_GAIN_Y", "0.035"))
        self._yolo_align_sign_x = float(os.environ.get("AIC_YOLO_ALIGN_SIGN_X", "1.0"))
        self._yolo_align_sign_y = float(os.environ.get("AIC_YOLO_ALIGN_SIGN_Y", "1.0"))
        self._yolo_align_max_step = float(os.environ.get("AIC_YOLO_ALIGN_MAX_STEP", "0.006"))
        self._yolo_model_wait_sec = float(os.environ.get("AIC_YOLO_MODEL_WAIT_SEC", "5.0"))
        self._yolo_search_interval_steps = max(1, int(os.environ.get("AIC_YOLO_SEARCH_INTERVAL_STEPS", "5")))
        self._yolo_track_interval_steps = max(1, int(os.environ.get("AIC_YOLO_TRACK_INTERVAL_STEPS", "1")))
        self._yolo_cache_max_age_sec = float(os.environ.get("AIC_YOLO_CACHE_MAX_AGE_SEC", "0.75"))
        self._triangulation_z_stop_enabled = os.environ.get("AIC_TRIANGULATION_Z_STOP", "1").lower() not in {"0", "false", "no"}
        self._triangulation_min_views = max(2, int(os.environ.get("AIC_TRIANGULATION_MIN_VIEWS", "2")))
        self._triangulation_stop_z_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_Z_OFFSET", str(self._TRIANGULATION_STOP_Z_OFFSET_DEFAULT)))
        self._triangulation_stop_x_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_X_OFFSET", "0.005"))
        self._triangulation_stop_y_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_Y_OFFSET", "0.005"))
        self._tool0_to_tcp_z = float(os.environ.get("AIC_TOOL0_TO_TCP_Z", str(self._TOOL0_TO_TCP_Z)))
        self._t_tool0_tcp = np.eye(4, dtype=float)
        self._t_tool0_tcp[2, 3] = self._tool0_to_tcp_z
        self._t_tool0_to_optical = {
            name: _matrix_from_translation_quat(translation, quat)
            for name, (translation, quat) in self._TOOL0_TO_OPTICAL.items()
        }
        self._yolo_control_camera = os.environ.get("AIC_YOLO_CONTROL_CAMERA", "center").strip().lower()
        if self._yolo_control_camera not in {"left", "center", "right"}:
            self.get_logger().warn(
                f"[DataCollect2] Invalid AIC_YOLO_CONTROL_CAMERA={self._yolo_control_camera}; using center"
            )
            self._yolo_control_camera = "center"
        self._yolo_debug_video_enabled = os.environ.get("AIC_YOLO_DEBUG_VIDEO", "1").lower() not in {"0", "false", "no"}
        self._yolo_debug_video_fps = float(os.environ.get("AIC_YOLO_DEBUG_VIDEO_FPS", "20.0"))
        self._yolo_debug_video_writer = None
        self._yolo_debug_video_path = None
        self._vision_offset_record_enabled = os.environ.get("AIC_VISION_OFFSET_RECORD", "1").lower() not in {"0", "false", "no"}
        self._yolo_lock = threading.Lock()
        self._yolo_request_event = threading.Event()
        self._yolo_request = None
        self._yolo_cache = {
            "detection": None,
            "results": None,
            "bgr": None,
            "detections_by_camera": {},
            "results_by_camera": {},
            "bgrs_by_camera": {},
            "updated_at": 0.0,
            "port_kw": "",
        }
        
        _lerobot_out = os.environ.get("AIC_LEROBOT_OUT_DIR", "lerobot")
        self._debug_image_dir = Path(_lerobot_out) / self._lerobot_version / "debug" / "image"
        self._debug_video_dir = Path(_lerobot_out) / self._lerobot_version / "debug" / "video"
        self._vision_offset_dataset_dir = Path(
            os.environ.get(
                "AIC_VISION_OFFSET_DATASET_DIR",
                str(Path(_lerobot_out) / self._lerobot_version / "vision_offset_dataset"),
            )
        )
        self._vision_offset_images_dir = self._vision_offset_dataset_dir / "images"
        self._vision_offset_samples_path = self._vision_offset_dataset_dir / "samples.jsonl"
        if self._vision_offset_record_enabled:
            self._vision_offset_images_dir.mkdir(parents=True, exist_ok=True)
        self._scenario_params_file = Path(os.environ.get("AIC_SCENARIO_PARAMS_FILE", "/tmp/aic_scenario_params.json"))

        # 5. 서브스크립션 및 스레드 시작
        self._init_lerobot_dataset()
        threading.Thread(target=self._init_yolo, daemon=True).start()
        threading.Thread(target=self._yolo_detection_worker, daemon=True).start()

        # 6. 종료 처리
        atexit.register(self._finalize_dataset)
        try: signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError: pass

        self._stop_file = Path(os.environ.get("AIC_STOP_FILE", "/tmp/aic_policy_stop"))
        threading.Thread(target=self._watch_stop_file, daemon=True).start()

        self.get_logger().info(f"[DataCollect2] YOLO-guided Policy Initialized. Root: {self.capture_root}")

    # ── 인프라 로직 ──────────────────────────────────────────────────────────

    def _init_yolo(self) -> None:
        """환경 변수에 지정된 YOLO weight 파일을 로드해서 포트 검출 모델을 준비한다."""
        model_path = Path(self._yolo_model_path)
        self._yolo_model = None
        if not model_path.exists(): return
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(model_path))
        except Exception: pass

    @staticmethod
    def _is_valid_dataset_root(root: Path) -> bool:
        """LeRobot dataset 디렉터리가 이어서 기록할 수 있는 정상 구조인지 확인한다."""
        return (root / "meta" / "info.json").exists() and (root / "meta" / "tasks.parquet").exists()

    def _init_lerobot_dataset(self) -> None:
        """LeRobot dataset을 새로 생성하거나 기존 dataset을 resume해서 기록 준비를 한다."""
        if not _LEROBOT_AVAILABLE or not self._lerobot_full_repo_id: return
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_root = Path(os.environ.get("AIC_LEROBOT_OUT_DIR", "lerobot")) / self._lerobot_version
            if self._is_valid_dataset_root(dataset_root):
                self._lerobot_dataset = LeRobotDataset.resume(repo_id=self._lerobot_full_repo_id, root=dataset_root)
            else:
                if dataset_root.exists(): shutil.rmtree(dataset_root)
                self._lerobot_dataset = LeRobotDataset.create(
                    repo_id=self._lerobot_full_repo_id, root=dataset_root, fps=20,
                    features=LEROBOT_FEATURES, use_videos=True,
                )
        except Exception: self._lerobot_dataset = None

    def _finalize_dataset(self) -> None:
        """프로세스 종료 전에 LeRobot dataset writer를 finalize해서 파일 손상을 막는다."""
        if self._lerobot_dataset is not None:
            try: self._lerobot_dataset.finalize()
            except Exception: pass
            finally: self._lerobot_dataset = None

    def _watch_stop_file(self) -> None:
        """AIC_STOP_FILE이 생기면 dataset을 finalize하고 즉시 프로세스를 종료한다."""
        while True:
            if self._stop_file.exists(): self._finalize_dataset(); os._exit(0)
            time.sleep(0.5)

    def _on_sigterm(self, signum, frame) -> None:
        """SIGTERM을 받았을 때 dataset을 finalize한 뒤 정상 종료한다."""
        self._finalize_dataset(); raise SystemExit(0)

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        """target_frame 기준 source_frame TF가 timeout 안에 조회 가능해질 때까지 기다린다."""
        start = self.time_now()
        while (self.time_now() - start) < Duration(seconds=timeout_sec):
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException: self.sleep_for(0.1)
        return False

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        """현재 시점의 target_frame 기준 source_frame transform을 조회한다."""
        return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform

    def _select_port_frame(self, task: Task) -> str:
        """포트 입구 frame이 있으면 사용하고, 없으면 기본 port link frame으로 fallback한다."""
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = f"{port_frame}_entrance"
        if self._wait_for_tf("base_link", entrance_frame, timeout_sec=2.0):
            self.get_logger().info(f"[DataCollect2] Using port entrance frame: {entrance_frame}")
            return entrance_frame
        self.get_logger().warn(f"[DataCollect2] Port entrance TF unavailable, falling back to: {port_frame}")
        return port_frame

    def _select_cable_tip_frame(self, task: Task) -> str:
        """케이블 끝단 제어를 위해 task 정보에서 사용 가능한 cable tip frame을 선택한다."""
        cable_prefix = task.cable_name
        plug_name = task.plug_name.strip()
        plug_type = task.plug_type.strip().lower()
        candidates = []

        if plug_name:
            candidates.append(f"{cable_prefix}/{plug_name}_link")
            if not plug_name.endswith("_tip"):
                candidates.append(f"{cable_prefix}/{plug_name}_tip_link")
        if plug_type:
            candidates.append(f"{cable_prefix}/{plug_type}_tip_link")

        for frame in dict.fromkeys(candidates):
            if self._wait_for_tf("base_link", frame, timeout_sec=0.5):
                self.get_logger().info(f"[DataCollect2] Using cable tip frame: {frame}")
                return frame

        fallback_frame = f"{cable_prefix}/{plug_name}_link"
        self.get_logger().warn(
            f"[DataCollect2] Cable tip TF unavailable, falling back to plug frame: {fallback_frame}"
        )
        return fallback_frame

    def _wait_for_yolo_model(self) -> bool:
        """YOLO 모델이 로드될 때까지 기다리고, 실패하면 TF-guided fallback 상태를 알린다."""
        start = time.time()
        while self._yolo_model is None and (time.time() - start) < self._yolo_model_wait_sec:
            self.sleep_for(0.1)
        if self._yolo_model is None:
            self.get_logger().warn("[DataCollect2] YOLO model is not available; Phase 1-A will remain TF-guided")
            return False
        return True

    def set_pose_target(self, move_robot, pose, stiffness=None, damping=None):
        """주어진 목표 pose와 impedance 값으로 controller에 MotionUpdate 명령을 보낸다."""
        _s = stiffness if stiffness is not None else self._STIFFNESS_DEFAULT
        _d = damping if damping is not None else self._DAMPING_DEFAULT
        mu = MotionUpdate(
            header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
            pose=pose, target_stiffness=np.diag(_s).flatten(), target_damping=np.diag(_d).flatten(),
            feedforward_wrench_at_tip=Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION),
        )
        try: move_robot(motion_update=mu)
        except Exception: pass

    def _motion_update_from_pose(self, pose, stiffness=None, damping=None) -> MotionUpdate:
        """데이터셋 action 기록에 사용할 MotionUpdate 메시지를 pose에서 생성한다."""
        mu = MotionUpdate()
        mu.header.frame_id, mu.header.stamp = "base_link", self.get_clock().now().to_msg()
        mu.pose = pose
        _s = stiffness if stiffness is not None else self._STIFFNESS_DEFAULT
        _d = damping if damping is not None else self._DAMPING_DEFAULT
        mu.target_stiffness, mu.target_damping = list(np.diag(_s).flatten()), list(np.diag(_d).flatten())
        mu.trajectory_generation_mode = TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION)
        return mu

    def _record_motion_step(self, recorder, phase, task, port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=None, damping=None):
        """현재 observation, action, TF, 부가 정보를 LeRobot recorder에 한 step 저장한다."""
        if recorder is None or obs is None: return
        action = self._motion_update_from_pose(pose, stiffness, damping)
        recorder.record_step(
            phase=phase, task=task, obs=obs, action=action,
            port_tf=port_tf, plug_tf=plug_tf, gripper_tf=gripper_tf, extras=extras,
            stiffness=stiffness, damping=damping
        )

    def _plug_tip_to_port_label(self, port_tf: Transform, plug_tf: Transform) -> dict[str, float]:
        """TF ground-truth로 plug tip 위치를 port 로컬 좌표계 기준 offset label로 계산한다."""
        port_xyz = np.array([port_tf.translation.x, port_tf.translation.y, port_tf.translation.z], dtype=float)
        plug_xyz = np.array([plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z], dtype=float)
        port_rotation = _quat_to_matrix_xyzw(
            port_tf.rotation.x,
            port_tf.rotation.y,
            port_tf.rotation.z,
            port_tf.rotation.w,
        )
        local_offset = port_rotation.T @ (plug_xyz - port_xyz)
        return {
            "x_m": float(local_offset[0]),
            "y_m": float(local_offset[1]),
            "z_m": float(local_offset[2]),
            "xy_m": float(np.linalg.norm(local_offset[:2])),
            "x_mm": float(local_offset[0] * 1000.0),
            "y_mm": float(local_offset[1] * 1000.0),
            "z_mm": float(local_offset[2] * 1000.0),
            "xy_mm": float(np.linalg.norm(local_offset[:2]) * 1000.0),
        }

    def _matrix_from_transform(self, transform: Transform) -> np.ndarray:
        """geometry_msgs/Transform을 상대 pose 계산용 4x4 행렬로 변환한다."""
        return _matrix_from_translation_quat(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
        )

    def _relative_transform_label(self, reference_tf: Transform, target_tf: Transform) -> dict[str, float]:
        """target pose를 reference frame 기준 xyz+quat label로 계산한다."""
        t_reference_target = np.linalg.inv(self._matrix_from_transform(reference_tf)) @ self._matrix_from_transform(target_tf)
        rotation = t_reference_target[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = float(np.sqrt(trace + 1.0) * 2.0)
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(rotation)))
            if idx == 0:
                s = float(np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0)
                qw = (rotation[2, 1] - rotation[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation[0, 1] + rotation[1, 0]) / s
                qz = (rotation[0, 2] + rotation[2, 0]) / s
            elif idx == 1:
                s = float(np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0)
                qw = (rotation[0, 2] - rotation[2, 0]) / s
                qx = (rotation[0, 1] + rotation[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation[1, 2] + rotation[2, 1]) / s
            else:
                s = float(np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0)
                qw = (rotation[1, 0] - rotation[0, 1]) / s
                qx = (rotation[0, 2] + rotation[2, 0]) / s
                qy = (rotation[1, 2] + rotation[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=float)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm > 1e-9:
            quat /= quat_norm
        translation = t_reference_target[:3, 3]
        return {
            "x_m": float(translation[0]),
            "y_m": float(translation[1]),
            "z_m": float(translation[2]),
            "xy_m": float(np.linalg.norm(translation[:2])),
            "x_mm": float(translation[0] * 1000.0),
            "y_mm": float(translation[1] * 1000.0),
            "z_mm": float(translation[2] * 1000.0),
            "xy_mm": float(np.linalg.norm(translation[:2]) * 1000.0),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3]),
        }

    def _lookup_optional_port_transform(self, module_name: str, port_name: str) -> tuple[Optional[str], Optional[Transform]]:
        """port entrance frame을 우선 조회하고 없으면 link frame으로 fallback한다."""
        base_frame = f"task_board/{module_name}/{port_name}_link"
        for frame in (f"{base_frame}_entrance", base_frame):
            try:
                return frame, self._lookup_transform("base_link", frame)
            except TransformException:
                continue
        return None, None

    def _candidate_port_specs(self, task: Task) -> list[tuple[str, str]]:
        """현재 task에서 비교 기록할 포트 후보 목록을 만든다."""
        if "sfp" in task.port_type.lower():
            return [(task.target_module_name, "sfp_port_0"), (task.target_module_name, "sfp_port_1")]
        if "sc" in task.port_type.lower():
            return [("sc_port_0", "sc_port_base"), ("sc_port_1", "sc_port_base")]
        return [(task.target_module_name, task.port_name)]

    def _all_ports_relative_label(self, task: Task, plug_tf: Transform) -> dict[str, Any]:
        """두 포트 후보 각각의 상대 위치를 plug/port 양쪽 기준으로 기록한다."""
        ports: dict[str, Any] = {}
        for module_name, port_name in self._candidate_port_specs(task):
            key = f"{module_name}/{port_name}"
            frame, port_tf = self._lookup_optional_port_transform(module_name, port_name)
            is_target = module_name == task.target_module_name and port_name == task.port_name
            if port_tf is None:
                ports[key] = {
                    "available": False,
                    "is_target": bool(is_target),
                    "frame": "",
                }
                continue
            ports[key] = {
                "available": True,
                "is_target": bool(is_target),
                "frame": frame,
                "plug_tip_in_port": self._plug_tip_to_port_label(port_tf, plug_tf),
                "port_in_plug_tip": self._relative_transform_label(plug_tf, port_tf),
            }
        return ports

    def _insertion_wrist_label(self, obs, plug_tf: Transform) -> dict[str, Any]:
        """현재 IK 결과 joint 값과 plug 축의 XY 평면 법선 기준 기울기를 기록한다."""
        label: dict[str, Any] = {
            "source": "joint_states_and_tf",
            "available": False,
        }
        if obs is None:
            label["reason"] = "missing_observation"
            return label

        joint_state = getattr(obs, "joint_states", None)
        names = list(getattr(joint_state, "name", []) or [])
        positions = [float(v) for v in list(getattr(joint_state, "position", []) or [])]
        joint_positions = {
            (names[idx] if idx < len(names) and names[idx] else f"joint_{idx}"): positions[idx]
            for idx in range(len(positions))
        }
        wrist_positions = {
            name: value
            for name, value in joint_positions.items()
            if name.startswith("wrist_")
        }
        if not wrist_positions and len(positions) >= 3:
            start_idx = len(positions) - 3
            wrist_positions = {
                (names[idx] if idx < len(names) and names[idx] else f"joint_{idx}"): positions[idx]
                for idx in range(start_idx, len(positions))
            }

        tcp_pose = getattr(getattr(obs, "controller_state", None), "tcp_pose", None)
        if tcp_pose is None:
            label.update({
                "joint_positions": joint_positions,
                "wrist_joint_positions": wrist_positions,
                "ik_result_joint_positions": joint_positions,
                "ik_result_wrist_joint_positions": wrist_positions,
                "reason": "missing_tcp_pose",
            })
            return label

        tcp_xyz = np.array([tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z], dtype=float)
        plug_xyz = np.array([plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z], dtype=float)
        plug_axis = plug_xyz - tcp_xyz
        axis_norm = float(np.linalg.norm(plug_axis))
        if axis_norm < 1e-9:
            label.update({
                "joint_positions": joint_positions,
                "wrist_joint_positions": wrist_positions,
                "ik_result_joint_positions": joint_positions,
                "ik_result_wrist_joint_positions": wrist_positions,
                "reason": "zero_tcp_to_plug_vector",
            })
            return label

        plug_axis /= axis_norm
        base_z = np.array([0.0, 0.0, 1.0], dtype=float)
        signed_dot = float(np.dot(plug_axis, base_z))
        target_axis = base_z if signed_dot >= 0.0 else -base_z
        correction_axis = np.cross(plug_axis, target_axis)
        correction_axis_norm = float(np.linalg.norm(correction_axis))
        if correction_axis_norm > 1e-9:
            correction_axis /= correction_axis_norm
        correction_angle = float(np.arctan2(correction_axis_norm, abs(signed_dot)))

        label.update({
            "available": True,
            "joint_positions": joint_positions,
            "wrist_joint_positions": wrist_positions,
            "ik_result_source": "observation.joint_states_after_controller_ik",
            "ik_result_joint_names": names,
            "ik_result_positions": positions,
            "ik_result_joint_positions": joint_positions,
            "ik_result_wrist_joint_positions": wrist_positions,
            "move_robot_joint_motion_update": {
                "target_state.positions": positions,
                "trajectory_generation_mode": "MODE_POSITION",
                "joint_order": names,
            },
            "plug_axis_base": {
                "x": float(plug_axis[0]),
                "y": float(plug_axis[1]),
                "z": float(plug_axis[2]),
            },
            "vertical_target_axis_base": {
                "x": float(target_axis[0]),
                "y": float(target_axis[1]),
                "z": float(target_axis[2]),
            },
            "tilt_from_xy_plane_normal_rad": correction_angle,
            "tilt_from_xy_plane_normal_deg": float(np.degrees(correction_angle)),
            "upright_correction_axis_base": {
                "x": float(correction_axis[0]),
                "y": float(correction_axis[1]),
                "z": float(correction_axis[2]),
            },
            "upright_correction_angle_rad": correction_angle,
            "upright_correction_angle_deg": float(np.degrees(correction_angle)),
        })
        return label

    def _json_safe(self, value):
        """numpy scalar/array가 섞인 metadata를 jsonl로 쓸 수 있는 기본 타입으로 바꾼다."""
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, np.ndarray):
            return [self._json_safe(v) for v in value.tolist()]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if value is None or isinstance(value, str):
            return value
        return str(value)

    def _vision_detection_metadata(self, detections_by_camera: dict[str, Optional[dict[str, Any]]]) -> dict[str, Any]:
        """camera별 YOLO bbox/confidence를 vision offset dataset metadata로 정리한다."""
        metadata = {}
        for camera_name in ("left", "center", "right"):
            det = detections_by_camera.get(camera_name) if detections_by_camera else None
            if det is None:
                metadata[camera_name] = {"detected": False}
                continue
            metadata[camera_name] = {
                "detected": True,
                "class_id": int(det.get("class_id", -1)),
                "class_name": str(det.get("class_name", "")),
                "confidence": float(det.get("confidence", 0.0)),
                "center_x": float(det.get("center_x", 0.0)),
                "center_y": float(det.get("center_y", 0.0)),
                "width_px": float(det.get("width_px", 0.0)),
                "height_px": float(det.get("height_px", 0.0)),
                "area_ratio": float(det.get("area_ratio", 0.0)),
            }
        return metadata

    def _save_vision_offset_sample(
        self,
        episode_name: str,
        task: Task,
        phase: str,
        step_idx: int,
        obs,
        port_tf: Transform,
        plug_tf: Transform,
        pose: Pose,
        extras: dict[str, Any],
        detections_by_camera: Optional[dict[str, Optional[dict[str, Any]]]] = None,
    ) -> None:
        """이미지 3장과 plug-tip-to-port offset label을 pure-vision 학습용 jsonl dataset으로 저장한다."""
        if not self._vision_offset_record_enabled or obs is None:
            return

        sample_id = f"{episode_name}_{phase}_{step_idx:06d}"
        image_paths: dict[str, str] = {}
        for camera_name in ("left", "center", "right"):
            img_msg = self._image_msg_for_camera(obs, camera_name)
            bgr = self._image_msg_to_bgr(img_msg, camera_name)
            if bgr is None:
                image_paths[camera_name] = ""
                continue
            camera_image_dir = self._vision_offset_images_dir / camera_name / episode_name
            camera_image_dir.mkdir(parents=True, exist_ok=True)
            image_path = camera_image_dir / f"{sample_id}_{camera_name}.png"
            cv2.imwrite(str(image_path), bgr)
            image_paths[camera_name] = str(image_path.relative_to(self._vision_offset_dataset_dir))

        label = self._plug_tip_to_port_label(port_tf, plug_tf)
        record = {
            "sample_id": sample_id,
            "episode_name": episode_name,
            "task_id": task.id,
            "task_type": "nic" if "sfp" in task.port_type.lower() else "sc",
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "phase": phase,
            "step_index": int(step_idx),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "images": image_paths,
            "label": {
                "source": "tf_base_link",
                "plug_tip_to_port": label,
                "ports": self._all_ports_relative_label(task, plug_tf),
                "insertion_wrist": self._insertion_wrist_label(obs, plug_tf),
            },
            "command": {
                "position": {
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "z": float(pose.position.z),
                },
                "orientation": {
                    "x": float(pose.orientation.x),
                    "y": float(pose.orientation.y),
                    "z": float(pose.orientation.z),
                    "w": float(pose.orientation.w),
                },
            },
            "collect": {
                "pattern": str(extras.get("collect_pattern", self.collect_pattern)),
                "progress": float(extras.get("collect_progress", 0.0)),
                "radius_m": float(extras.get("collect_radius", 0.0)),
                "theta_rad": float(extras.get("collect_theta", 0.0)),
                "local_x_m": float(extras.get("collect_local_x", 0.0)),
                "local_y_m": float(extras.get("collect_local_y", 0.0)),
                "spin_angle_rad": float(extras.get("collect_spin_angle", 0.0)),
            },
            "triangulation": {
                "valid": bool(extras.get("triangulated_tip_to_port_offsets_valid", False)),
                "x_m": float(extras.get("triangulated_x_offset", 0.0)),
                "y_m": float(extras.get("triangulated_y_offset", 0.0)),
                "z_m": float(extras.get("triangulated_z_offset", 0.0)),
                "xy_m": float(extras.get("triangulated_xy_offset", 0.0)),
                "port_views": int(extras.get("triangulated_port_views", 0)),
                "port_pairs": int(extras.get("triangulated_port_pairs", 0)),
            },
            "yolo": {
                "selected_camera": str(extras.get("yolo_selected_camera", "")),
                "multiview_detect_count": int(extras.get("yolo_multiview_detect_count", 0)),
                "cameras": self._vision_detection_metadata(detections_by_camera or {}),
            },
        }

        with self._vision_offset_samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._json_safe(record), ensure_ascii=False) + "\n")

    def _image_msg_to_bgr(self, img_msg, camera_name: str = "image") -> Optional[np.ndarray]:
        """ROS Image 메시지를 OpenCV에서 쓰는 BGR numpy 이미지로 변환한다."""
        if img_msg is None or img_msg.width == 0 or img_msg.height == 0:
            return None
        try:
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        except ValueError:
            self.get_logger().warn(f"[DataCollect2] Invalid {camera_name} buffer size")
            return None
        if img_msg.encoding == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    def _image_msg_for_camera(self, obs, camera_name: str):
        """Observation에서 camera_name에 해당하는 Image 메시지를 꺼낸다."""
        if camera_name == "left":
            return obs.left_image
        if camera_name == "right":
            return obs.right_image
        return obs.center_image

    def _camera_info_for_camera(self, obs, camera_name: str):
        """Observation에서 camera_name에 해당하는 CameraInfo 메시지를 꺼낸다."""
        if camera_name == "left":
            return obs.left_camera_info
        if camera_name == "right":
            return obs.right_camera_info
        return obs.center_camera_info

    def _camera_intrinsic_matrix(self, camera_info) -> Optional[np.ndarray]:
        """CameraInfo.k 배열을 3x3 intrinsic 행렬 K로 변환한다."""
        if camera_info is None or len(camera_info.k) < 9:
            return None
        k = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
        if abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9:
            return None
        return k

    def _base_to_camera_optical_matrix(self, obs, camera_name: str) -> np.ndarray:
        """tcp_pose와 고정 extrinsic으로 base_link 좌표를 camera optical 좌표로 보내는 행렬을 만든다."""
        t_base_tcp = _matrix_from_pose(obs.controller_state.tcp_pose)
        t_base_tool0 = t_base_tcp @ np.linalg.inv(self._t_tool0_tcp)
        t_base_optical = t_base_tool0 @ self._t_tool0_to_optical[camera_name]
        return np.linalg.inv(t_base_optical)

    def _triangulate_yolo_port(
        self,
        obs,
        detections: dict[str, Optional[dict[str, Any]]],
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """여러 카메라의 YOLO 포트 중심 픽셀을 이용해 base_link 기준 포트 3D 위치를 복원한다."""
        extras: dict[str, Any] = {
            "triangulated_port_valid": False,
            "triangulated_port_views": 0,
            "triangulated_port_pairs": 0,
        }
        if obs is None or not detections:
            return None, extras

        camera_measurements = []
        for camera_name in ("left", "center", "right"):
            detection = detections.get(camera_name)
            if detection is None or camera_name not in self._t_tool0_to_optical:
                continue
            k = self._camera_intrinsic_matrix(self._camera_info_for_camera(obs, camera_name))
            if k is None:
                continue
            try:
                t_cam_base = self._base_to_camera_optical_matrix(obs, camera_name)
            except Exception:
                continue
            camera_measurements.append(
                (
                    camera_name,
                    float(detection["center_x"]),
                    float(detection["center_y"]),
                    k,
                    t_cam_base,
                )
            )

        extras["triangulated_port_views"] = len(camera_measurements)
        if len(camera_measurements) < self._triangulation_min_views:
            return None, extras

        points = []
        for i in range(len(camera_measurements)):
            for j in range(i + 1, len(camera_measurements)):
                name_a, u_a, v_a, k_a, t_a = camera_measurements[i]
                name_b, u_b, v_b, k_b, t_b = camera_measurements[j]
                try:
                    p_a = k_a @ t_a[:3, :]
                    p_b = k_b @ t_b[:3, :]
                    pts_4d = cv2.triangulatePoints(
                        p_a,
                        p_b,
                        np.array([[u_a], [v_a]], dtype=np.float64),
                        np.array([[u_b], [v_b]], dtype=np.float64),
                    )
                    if abs(float(pts_4d[3, 0])) < 1e-12:
                        continue
                    point = (pts_4d[:3, 0] / pts_4d[3, 0]).astype(float)
                    if np.all(np.isfinite(point)):
                        points.append(point)
                        extras[f"triangulated_pair_{name_a}_{name_b}_x"] = float(point[0])
                        extras[f"triangulated_pair_{name_a}_{name_b}_y"] = float(point[1])
                        extras[f"triangulated_pair_{name_a}_{name_b}_z"] = float(point[2])
                except Exception as exc:
                    extras[f"triangulated_pair_{name_a}_{name_b}_error"] = str(exc)

        extras["triangulated_port_pairs"] = len(points)
        if not points:
            return None, extras

        point_arr = np.asarray(points, dtype=float)
        point_mean = point_arr.mean(axis=0)
        point_std = point_arr.std(axis=0) if len(points) > 1 else np.zeros(3, dtype=float)
        extras.update({
            "triangulated_port_valid": True,
            "triangulated_port_x": float(point_mean[0]),
            "triangulated_port_y": float(point_mean[1]),
            "triangulated_port_z": float(point_mean[2]),
            "triangulated_port_std_x": float(point_std[0]),
            "triangulated_port_std_y": float(point_std[1]),
            "triangulated_port_std_z": float(point_std[2]),
        })
        return point_mean, extras

    def _measure_triangulated_tip_to_port_offsets(
        self,
        obs,
        detections: dict[str, Optional[dict[str, Any]]],
        plug_tf: Transform,
        port_axis: Optional[dict[str, float]],
        port_transform: Optional[Transform] = None,
    ) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
        """YOLO triangulation 포트 위치 기준으로 cable tip의 포트 로컬 X/Y/Z offset을 계산한다."""
        port_point, extras = self._triangulate_yolo_port(obs, detections)
        extras["triangulated_z_stop_enabled"] = bool(self._triangulation_z_stop_enabled)
        extras["triangulated_z_stop_threshold"] = float(self._triangulation_stop_z_offset)
        extras["triangulated_x_stop_threshold"] = float(self._triangulation_stop_x_offset)
        extras["triangulated_y_stop_threshold"] = float(self._triangulation_stop_y_offset)
        if port_point is None or port_axis is None:
            extras["triangulated_tip_to_port_offsets_valid"] = False
            extras["triangulated_z_offset_valid"] = False
            extras["triangulated_xy_offset_valid"] = False
            return None, extras

        axis = np.array(
            [
                float(port_axis.get("x", 0.0)),
                float(port_axis.get("y", 0.0)),
                float(port_axis.get("z", 1.0)),
            ],
            dtype=float,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            extras["triangulated_tip_to_port_offsets_valid"] = False
            extras["triangulated_z_offset_valid"] = False
            extras["triangulated_xy_offset_valid"] = False
            return None, extras
        axis /= axis_norm

        if port_transform is not None:
            port_rotation = _quat_to_matrix_xyzw(
                port_transform.rotation.x,
                port_transform.rotation.y,
                port_transform.rotation.z,
                port_transform.rotation.w,
            )
            x_axis = port_rotation[:, 0].copy()
            y_axis = port_rotation[:, 1].copy()
        else:
            # Fallback: use base-frame components perpendicular to the approach axis.
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        for basis in (x_axis, y_axis):
            basis -= axis * float(np.dot(basis, axis))
            basis_norm = float(np.linalg.norm(basis))
            if basis_norm > 1e-9:
                basis /= basis_norm

        tip_point = np.array(
            [plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z],
            dtype=float,
        )
        tip_delta = tip_point - port_point
        x_offset = float(np.dot(tip_delta, x_axis))
        y_offset = float(np.dot(tip_delta, y_axis))
        xy_offset = float(np.linalg.norm([x_offset, y_offset]))
        z_offset = float(np.dot(tip_delta, axis)) # port에서 tip까지 거리가 approach 축 방향으로 얼마나 떨어져 있는지를 내적으로 측정
        xyz_within_threshold = (
            abs(x_offset) <= self._triangulation_stop_x_offset
            and abs(y_offset) <= self._triangulation_stop_y_offset
            and z_offset <= self._triangulation_stop_z_offset
        )
        offsets = {
            "x": x_offset,
            "y": y_offset,
            "z": z_offset,
            "xy": xy_offset,
            "within_threshold": bool(xyz_within_threshold),
        }
        extras.update({
            "triangulated_tip_to_port_offsets_valid": True,
            "triangulated_z_offset_valid": True,
            "triangulated_xy_offset_valid": True,
            "triangulated_xyz_within_threshold": bool(xyz_within_threshold),
            "triangulated_x_offset": x_offset,
            "triangulated_y_offset": y_offset,
            "triangulated_xy_offset": xy_offset,
            "triangulated_z_offset": z_offset,
            "triangulated_tip_x": float(tip_point[0]),
            "triangulated_tip_y": float(tip_point[1]),
            "triangulated_tip_z": float(tip_point[2]),
        })
        return offsets, extras

    def _port_local_xy_axes(self, port_transform: Transform, port_axis: Optional[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """포트 로컬 X/Y 축과 approach 축을 base_link 방향 벡터로 변환한다."""
        port_rotation = _quat_to_matrix_xyzw(
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
            port_transform.rotation.w,
        )
        x_axis = port_rotation[:, 0].copy()
        y_axis = port_rotation[:, 1].copy()
        if port_axis is None:
            z_axis = port_rotation[:, 2].copy()
        else:
            z_axis = np.array(
                [
                    float(port_axis.get("x", 0.0)),
                    float(port_axis.get("y", 0.0)),
                    float(port_axis.get("z", 1.0)),
                ],
                dtype=float,
            )
        z_norm = float(np.linalg.norm(z_axis))
        z_axis = z_axis / z_norm if z_norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=float)

        for basis in (x_axis, y_axis):
            basis -= z_axis * float(np.dot(basis, z_axis))
            basis_norm = float(np.linalg.norm(basis))
            if basis_norm > 1e-9:
                basis /= basis_norm
        return x_axis, y_axis, z_axis

    def _sample_collect_local_xy_offset(self, progress: float) -> tuple[float, float, float, float]:
        """COLLECT pattern에 맞춰 port-local x/y offset, radius, theta를 샘플링한다."""
        if self.collect_pattern == "gaussian":
            max_radius = max(1e-9, self.collect_gaussian_max_radius)
            sigma = max(1e-9, self.collect_gaussian_sigma)
            for _ in range(100):
                x_local = float(self._collect_rng.normal(0.0, sigma))
                y_local = float(self._collect_rng.normal(0.0, sigma))
                radius = float(np.hypot(x_local, y_local))
                if radius <= max_radius:
                    theta = float(np.arctan2(y_local, x_local)) if radius > 1e-12 else 0.0
                    return x_local, y_local, radius, theta
            theta = float(self._collect_rng.uniform(-np.pi, np.pi))
            radius = float(max_radius)
            return radius * float(np.cos(theta)), radius * float(np.sin(theta)), radius, theta

        theta = 2.0 * np.pi * self.collect_turns * progress
        radius = (1.0 - progress) * self.collect_start_radius + progress * self.collect_end_radius
        return (
            float(radius * np.cos(theta)),
            float(radius * np.sin(theta)),
            float(radius),
            float(theta),
        )

    def _apply_collect_offset(
        self,
        pose: Pose,
        port_transform: Transform,
        port_axis: Optional[dict[str, float]],
        step_idx: int,
    ) -> tuple[Pose, dict[str, float]]:
        """COLLECT pattern에서 뽑은 port-local XY 위치 offset을 목표 TCP pose에 더한다."""
        denom = float(max(1, self.collect_steps - 1))
        progress = float(np.clip(step_idx / denom, 0.0, 1.0))
        x_local, y_local, radius, theta = self._sample_collect_local_xy_offset(progress)
        x_axis, y_axis, approach_axis = self._port_local_xy_axes(port_transform, port_axis)
        offset = x_local * x_axis + y_local * y_axis

        pose.position.x += float(offset[0])
        pose.position.y += float(offset[1])
        pose.position.z += float(offset[2])

        spin_angle = self.collect_rotate_angle * progress
        if abs(spin_angle) > 1e-9:
            spin_quat = _quat_from_axis_angle_xyzw(approach_axis, spin_angle)
            base_quat = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            qx, qy, qz, qw = _quat_multiply_xyzw(spin_quat, base_quat)
            q_norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
            if q_norm > 1e-9:
                pose.orientation.x = float(qx / q_norm)
                pose.orientation.y = float(qy / q_norm)
                pose.orientation.z = float(qz / q_norm)
                pose.orientation.w = float(qw / q_norm)

        return pose, {
            "collect_pattern": self.collect_pattern,
            "collect_progress": progress,
            "collect_theta": float(theta),
            "collect_radius": float(radius),
            "collect_local_x": float(x_local),
            "collect_local_y": float(y_local),
            "collect_offset_x": float(offset[0]),
            "collect_offset_y": float(offset[1]),
            "collect_offset_z": float(offset[2]),
            "collect_spin_angle": float(spin_angle),
        }

    def _detect_port_from_bgr(self, bgr: np.ndarray, port_kw: str, conf: float) -> tuple[Optional[dict[str, Any]], Any]:
        """단일 BGR 이미지에서 YOLO로 목표 포트 타입을 검출하고 최고 confidence bbox를 반환한다."""
        if self._yolo_model is None or bgr is None:
            return None, None
        results = self._yolo_model(bgr, verbose=False, conf=conf)
        best_detection = None
        for result in results:
            names = getattr(result, "names", {})
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names.get(class_id, "").lower()
                if port_kw not in class_name:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": score,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_x": 0.5 * (x1 + x2),
                    "center_y": 0.5 * (y1 + y2),
                    "area_ratio": ((x2 - x1) * (y2 - y1)) / float(bgr.shape[0] * bgr.shape[1]),
                }
                if best_detection is None or detection["confidence"] > best_detection["confidence"]:
                    best_detection = detection
        return best_detection, results

    def _detect_port_from_obs(self, obs, port_kw: str, conf: float) -> tuple[Optional[dict[str, Any]], Any, Optional[np.ndarray]]:
        """Observation의 center camera 이미지에서 목표 포트를 검출한다."""
        if obs is None:
            return None, None, None
        bgr = self._image_msg_to_bgr(obs.center_image, "center_image")
        if bgr is None:
            return None, None, None
        detection, results = self._detect_port_from_bgr(bgr, port_kw, conf)
        return detection, results, bgr

    def _detect_ports_from_obs(self, obs, port_kw: str, conf: float) -> tuple[dict[str, Optional[dict[str, Any]]], dict[str, Any], dict[str, np.ndarray]]:
        """Observation의 left/center/right 이미지 각각에서 목표 포트를 검출한다."""
        detections = {}
        results_by_camera = {}
        bgrs = {}
        if obs is None:
            return detections, results_by_camera, bgrs
        for camera_name in ("left", "center", "right"):
            bgr = self._image_msg_to_bgr(
                self._image_msg_for_camera(obs, camera_name),
                f"{camera_name}_image",
            )
            if bgr is None:
                detections[camera_name] = None
                results_by_camera[camera_name] = None
                continue
            detection, results = self._detect_port_from_bgr(bgr, port_kw, conf)
            detections[camera_name] = detection
            results_by_camera[camera_name] = results
            bgrs[camera_name] = bgr
        return detections, results_by_camera, bgrs

    def _select_yolo_view(
        self,
        detections: dict[str, Optional[dict[str, Any]]],
        results_by_camera: dict[str, Any],
        bgrs_by_camera: dict[str, np.ndarray],
    ) -> tuple[Optional[dict[str, Any]], Any, Optional[np.ndarray], str]:
        """제어 기준 카메라에서 나온 YOLO 검출 결과를 선택한다."""
        detection = detections.get(self._yolo_control_camera)
        if detection is not None:
            return (
                detection,
                results_by_camera.get(self._yolo_control_camera),
                bgrs_by_camera.get(self._yolo_control_camera),
                self._yolo_control_camera,
            )
        return None, None, None, self._yolo_control_camera

    def _build_multiview_yolo_extras(self, detections: dict[str, Optional[dict[str, Any]]]) -> dict[str, Any]:
        """left/center/right YOLO 검출 상태와 중심점 통계를 recorder extras로 정리한다."""
        extras: dict[str, Any] = {}
        centers = []
        for camera_name in ("left", "center", "right"):
            detection = detections.get(camera_name)
            detected = detection is not None
            extras[f"yolo_{camera_name}_detected"] = detected
            extras[f"yolo_{camera_name}_confidence"] = detection["confidence"] if detected else 0.0
            extras[f"yolo_{camera_name}_center_x"] = detection["center_x"] if detected else 0.0
            extras[f"yolo_{camera_name}_center_y"] = detection["center_y"] if detected else 0.0
            extras[f"yolo_{camera_name}_area_ratio"] = detection["area_ratio"] if detected else 0.0
            if detected:
                centers.append((detection["center_x"], detection["center_y"]))
        extras["yolo_multiview_detect_count"] = len(centers)
        if len(centers) >= 2:
            arr = np.asarray(centers, dtype=float)
            extras["yolo_multiview_center_std_px"] = float(np.linalg.norm(arr.std(axis=0)))
        else:
            extras["yolo_multiview_center_std_px"] = 0.0
        return extras

    def _yolo_detection_worker(self) -> None:
        """메인 제어 루프를 막지 않도록 별도 스레드에서 YOLO 검출 요청을 처리한다."""
        while True:
            self._yolo_request_event.wait()
            with self._yolo_lock:
                request = self._yolo_request
                self._yolo_request = None
                self._yolo_request_event.clear()
            if request is None:
                continue

            detections = {}
            results_by_camera = {}
            for camera_name, bgr in request["bgrs"].items():
                detection, results = self._detect_port_from_bgr(
                    bgr, request["port_kw"], request["conf"]
                )
                detections[camera_name] = detection
                results_by_camera[camera_name] = results
            detection, results, bgr, selected_camera = self._select_yolo_view(
                detections, results_by_camera, request["bgrs"]
            )
            with self._yolo_lock:
                self._yolo_cache = {
                    "detection": detection,
                    "results": results,
                    "bgr": bgr,
                    "selected_camera": selected_camera,
                    "detections_by_camera": detections,
                    "results_by_camera": results_by_camera,
                    "bgrs_by_camera": request["bgrs"],
                    "updated_at": time.time(),
                    "port_kw": request["port_kw"],
                }

    def _submit_yolo_detection(self, obs, port_kw: str, conf: float) -> None:
        """현재 Observation 이미지를 YOLO worker가 처리하도록 최신 검출 요청으로 등록한다."""
        if obs is None:
            return
        bgrs = {}
        for camera_name in ("left", "center", "right"):
            bgr = self._image_msg_to_bgr(
                self._image_msg_for_camera(obs, camera_name),
                f"{camera_name}_image",
            )
            if bgr is not None:
                bgrs[camera_name] = bgr
        if not bgrs:
            return
        with self._yolo_lock:
            self._yolo_request = {"bgrs": bgrs, "port_kw": port_kw, "conf": conf}
            self._yolo_request_event.set()

    def _get_cached_yolo_detection(self, port_kw: str):
        """YOLO worker가 최근에 계산한 검출 결과를 cache 유효 시간 안에서 가져온다."""
        with self._yolo_lock:
            cache = dict(self._yolo_cache)
        if cache["port_kw"] != port_kw:
            return None, None, None, {}, ""
        if (time.time() - cache["updated_at"]) > self._yolo_cache_max_age_sec:
            return None, None, None, {}, ""
        return (
            cache["detection"],
            cache["results"],
            cache["bgr"],
            cache.get("detections_by_camera", {}),
            cache.get("selected_camera", self._yolo_control_camera),
        )

    def _log_yolo_detection(self, detection: dict[str, Any], prefix: str) -> None:
        """선택된 YOLO 검출 bbox, 중심점, confidence를 로그로 출력한다."""
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        self.get_logger().info(
            f"{prefix} class={detection['class_name']} "
            f"conf={detection['confidence']:.3f} "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
            f"center=({detection['center_x']:.1f},{detection['center_y']:.1f})"
        )

    def _save_yolo_debug_frame(
        self,
        bgr: np.ndarray,
        results,
        detection: dict[str, Any],
        task: Task,
        phase: str,
        step_idx: int,
    ) -> None:
        """YOLO handoff 순간의 검출 결과를 이미지와 JSON metadata로 저장한다."""
        try:
            self._debug_image_dir.mkdir(parents=True, exist_ok=True)
            annotated = bgr.copy()
            for result in results or []:
                names = getattr(result, "names", {})
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                    color = (0, 255, 0) if cls_name.lower() == detection["class_name"] else (0, 180, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            h, w = annotated.shape[:2]
            cx, cy = int(round(detection["center_x"])), int(round(detection["center_y"]))
            cv2.drawMarker(annotated, (w // 2, h // 2), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
            stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{task.id}_{phase}_{step_idx:04d}"
            image_path = self._debug_image_dir / f"{stem}.jpg"
            meta_path = self._debug_image_dir / f"{stem}.json"
            cv2.imwrite(str(image_path), annotated)
            meta_path.write_text(
                json.dumps(
                    {
                        "task_id": task.id,
                        "phase": phase,
                        "step": step_idx,
                        "image": image_path.name,
                        "detection": detection,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.get_logger().info(f"[DataCollect] YOLO debug frame saved: {image_path}")
        except Exception as exc:
            self.get_logger().warn(f"[DataCollect] Failed to save YOLO debug frame: {exc}")

    def _open_yolo_debug_video(self, task: Task, episode_name: str) -> None:
        """에피소드별 YOLO tracking 디버그 비디오를 기록할 경로를 준비한다."""
        if not self._yolo_debug_video_enabled:
            return
        self._debug_video_dir.mkdir(parents=True, exist_ok=True)
        self._yolo_debug_video_path = self._debug_video_dir / f"{episode_name}_yolo_tracking.mp4"
        self._yolo_debug_video_writer = None

    def _close_yolo_debug_video(self) -> None:
        """YOLO tracking 디버그 비디오 writer를 닫고 파일을 마무리한다."""
        writer = self._yolo_debug_video_writer
        self._yolo_debug_video_writer = None
        if writer is not None:
            try:
                writer.release()
                self.get_logger().info(f"[DataCollect2] YOLO debug video saved: {self._yolo_debug_video_path}")
            except Exception as exc:
                self.get_logger().warn(f"[DataCollect2] Failed to close YOLO debug video: {exc}")

    def _write_yolo_debug_video_frame(
        self,
        obs,
        results,
        detection: Optional[dict[str, Any]],
        phase: str,
        step_idx: int,
        guidance: str,
    ) -> None:
        """현재 center image 위에 YOLO bbox와 상태 문구를 그려 디버그 비디오에 한 프레임 추가한다."""
        if not self._yolo_debug_video_enabled or self._yolo_debug_video_path is None:
            return

        bgr = self._image_msg_to_bgr(obs.center_image) if obs is not None else None
        if bgr is None:
            return

        try:
            annotated = bgr.copy()
            for result in results or []:
                names = getattr(result, "names", {})
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                    is_selected = detection is not None and cls_name.lower() == detection["class_name"]
                    color = (0, 255, 0) if is_selected else (0, 180, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"{cls_name} {conf:.2f}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

            h, w = annotated.shape[:2]
            cv2.drawMarker(annotated, (w // 2, h // 2), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            if detection is not None:
                cx, cy = int(round(detection["center_x"])), int(round(detection["center_y"]))
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
                status = f"{phase} step={step_idx} guide={guidance} conf={detection['confidence']:.2f}"
            else:
                status = f"{phase} step={step_idx} guide={guidance} no-detection"

            cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated, "blue +=image center  red x=selected YOLO center", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self._yolo_debug_video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._yolo_debug_video_writer = cv2.VideoWriter(
                    str(self._yolo_debug_video_path),
                    fourcc,
                    self._yolo_debug_video_fps,
                    (w, h),
                )
                if not self._yolo_debug_video_writer.isOpened():
                    self.get_logger().warn(f"[DataCollect2] Failed to open YOLO debug video: {self._yolo_debug_video_path}")
                    self._yolo_debug_video_writer = None
                    return

            self._yolo_debug_video_writer.write(annotated)
        except Exception as exc:
            self.get_logger().warn(f"[DataCollect2] Failed to write YOLO debug video frame: {exc}")

    def _build_yolo_correction(
        self,
        detection: Optional[dict[str, Any]],
        image_shape: tuple[int, int, int],
        position_fraction: float,
    ) -> tuple[float, float, dict[str, Any]]:
        """YOLO 중심점과 이미지 중심의 차이를 base 좌표계 XY 보정량으로 변환한다."""
        x_error_norm = 0.0
        y_error_norm = 0.0
        area_ratio = 0.0
        dx = 0.0
        dy = 0.0

        if detection is not None:
            height, width = image_shape[:2]
            x_error_norm = ((width * 0.5) - detection["center_x"]) / max(width * 0.5, 1.0)
            y_error_norm = ((height * 0.5) - detection["center_y"]) / max(height * 0.5, 1.0)
            area_ratio = float(detection["area_ratio"])
            dx = float(np.clip(self._yolo_align_sign_x * self._yolo_align_gain_x * x_error_norm, -self._yolo_align_max_step, self._yolo_align_max_step))
            dy = float(np.clip(self._yolo_align_sign_y * self._yolo_align_gain_y * y_error_norm, -self._yolo_align_max_step, self._yolo_align_max_step))
            dx *= position_fraction
            dy *= position_fraction

        extras = {
            "yolo_detected": detection is not None,
            "yolo_class": detection["class_name"] if detection is not None else "",
            "yolo_confidence": detection["confidence"] if detection is not None else 0.0,
            "yolo_bbox_xyxy": detection["bbox_xyxy"] if detection is not None else [],
            "yolo_center_x": detection["center_x"] if detection is not None else 0.0,
            "yolo_center_y": detection["center_y"] if detection is not None else 0.0,
            "yolo_x_error_norm": float(x_error_norm),
            "yolo_y_error_norm": float(y_error_norm),
            "yolo_area_ratio": float(area_ratio),
            "yolo_dx": float(dx),
            "yolo_dy": float(dy),
            "position_fraction": float(position_fraction),
        }
        return dx, dy, extras

    def _apply_yolo_correction(
        self,
        pose: Pose,
        detection: Optional[dict[str, Any]],
        image_shape: tuple[int, int, int],
        position_fraction: float,
        port_axis: Optional[dict[str, float]] = None,
    ) -> tuple[Pose, dict[str, Any]]:
        """YOLO 기반 XY 보정량을 목표 pose에 적용하고 approach 축 방향 성분은 제거한다."""
        dx, dy, extras = self._build_yolo_correction(detection, image_shape, position_fraction)
        correction = np.array([dx, dy, 0.0], dtype=float)
        if port_axis is not None:
            axis = np.array(
                [
                    float(port_axis.get("x", 0.0)),
                    float(port_axis.get("y", 0.0)),
                    float(port_axis.get("z", 1.0)),
                ],
                dtype=float,
            )
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axis /= norm
                correction = correction - axis * float(np.dot(correction, axis))
        pose.position.x = float(pose.position.x + correction[0])
        pose.position.y = float(pose.position.y + correction[1])
        pose.position.z = float(pose.position.z + correction[2])
        extras["yolo_dx"] = float(correction[0])
        extras["yolo_dy"] = float(correction[1])
        extras["yolo_dz"] = float(correction[2])
        extras.update({
            "target_x": float(pose.position.x),
            "target_y": float(pose.position.y),
            "target_z": float(pose.position.z),
        })
        return pose, extras

    # ── 메인 에피소드 수집 로직 ───────────────────────────────────────────────

    def _finish_data_collection_episode(
        self,
        *,
        episode_dir: Path,
        task: Task,
        recorder,
        phase_step_counts: dict[str, int],
        status: str,
        detail: str = "",
    ) -> bool:
        """삽입 성공과 별개로 데이터 수집 task를 마무리하고 engine에는 완료를 알린다."""
        insertion_success = False
        if recorder is not None:
            try:
                recorder.save_episode(insertion_success=insertion_success)
            except Exception as exc:
                self.get_logger().warn(f"[DataCollect2] Failed to save LeRobot episode: {exc}")
        self._close_yolo_debug_video()
        self._write_episode_summary(
            episode_dir,
            {
                "task_id": task.id,
                "success": insertion_success,
                "insertion_success": insertion_success,
                "task_completed_for_engine": True,
                "status": status,
                "detail": detail,
                "mode": "lerobot",
                "approach_steps": int(phase_step_counts.get("approach", 0)),
                "collect_steps": int(phase_step_counts.get("collect", 0)),
                "collect_pattern": self.collect_pattern,
                "collect_start_radius": self.collect_start_radius,
                "collect_end_radius": self.collect_end_radius,
                "collect_turns": self.collect_turns,
                "collect_gaussian_sigma": self.collect_gaussian_sigma,
                "collect_gaussian_max_radius": self.collect_gaussian_max_radius,
            },
        )
        self.get_logger().info(
            f"DataCollect complete. status={status} "
            f"collect_steps={phase_step_counts.get('collect', 0)} "
            f"insertion_success={insertion_success} task_completed_for_engine=True"
        )
        return True

    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        """하나의 task에 대해 TF/YOLO 접근 후 포트 주변 나선형 COLLECT 데이터를 기록한다."""
        self._task = task; self._planner.reset(); send_feedback("data collect running")
        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name; episode_dir.mkdir(parents=True, exist_ok=True)
        phase_step_counts = {"approach": 0, "collect": 0}
        if self._lerobot_dataset is None and not self._vision_offset_record_enabled:
            self.get_logger().warn("[DataCollect2] No recorder is enabled: LeRobot unavailable and vision offset recording disabled")
            return self._finish_data_collection_episode(
                episode_dir=episode_dir,
                task=task,
                recorder=None,
                phase_step_counts=phase_step_counts,
                status="no_recorder",
                detail="LeRobot unavailable and vision offset recording disabled",
            )
        self._open_yolo_debug_video(task, episode_name)
        
        scenario_params_vec = self._load_scenario_params(task)
        recorder = LeRobotRecorder(self._lerobot_dataset, scenario_params_vec) if self._lerobot_dataset is not None else None
        
        port_frame = self._select_port_frame(task)
        cable_tip_frame = self._select_cable_tip_frame(task)
        if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf("base_link", cable_tip_frame):
            return self._finish_data_collection_episode(
                episode_dir=episode_dir,
                task=task,
                recorder=recorder,
                phase_step_counts=phase_step_counts,
                status="tf_unavailable",
                detail=f"Missing required TF: port_frame={port_frame}, cable_tip_frame={cable_tip_frame}",
            )
        self._wait_for_yolo_model()

        start_time = time.time()
        recording_started = False
        yolo_tracking_started = False
        _port_kw = "sfp" if "sfp" in task.port_type.lower() else "sc"

        # 포트 타입에 따른 파라미터 분기 (SFP는 기존 설정 유지, SC만 별도 튜닝)
        if _port_kw == "sc":
            # SC: 나선형 진동 방지를 위해 적분 이득/한계를 대폭 낮추고, 
            # 대신 접근 단계의 강성을 올려 물리적으로 케이블 장력을 이겨내도록 함.
            self._planner.i_gain = 0.07
            self._planner.max_integrator_windup = 0.06
            approach_stiffness = [280.0, 250.0, 250.0, 50.0, 50.0, 50.0]
            approach_damping = [87.0, 80.0, 80.0, 20.0, 20.0, 20.0]
            collect_stiffness = approach_stiffness
            collect_damping = approach_damping
        else:
            # SFP: 문제없이 작동하던 기존 설정 복구
            self._planner.i_gain = float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15"))
            self._planner.max_integrator_windup = 0.08
            approach_stiffness = self._STIFFNESS_DEFAULT
            approach_damping = self._DAMPING_DEFAULT
            collect_stiffness = approach_stiffness
            collect_damping = approach_damping

        # This is the desired port-to-cable-tip distance along the port frame's
        # outward local -Z approach axis. CheatCodePlanner converts it to a gripper TCP
        # target with the current gripper-to-plug offset.
        current_approach_z = 0.150 if _port_kw == "sfp" else 0.050

        def _check_and_start(obs, phase: str = "trigger", step_idx: int = 0, detection=None, results=None, bgr=None) -> bool:
            """YOLO가 처음 포트를 검출한 순간부터 episode recording을 시작한다."""
            nonlocal recording_started
            if recording_started or obs is None: return False
            if detection is None or results is None or bgr is None:
                detections, results_by_camera, bgrs_by_camera = self._detect_ports_from_obs(
                    obs, _port_kw, self._yolo_trigger_conf
                )
                detection, results, bgr, _ = self._select_yolo_view(
                    detections, results_by_camera, bgrs_by_camera
                )
            if detection is not None:
                self._save_yolo_debug_frame(bgr, results, detection, task, phase, step_idx)
                recording_started = True
                self._log_yolo_detection(
                    detection,
                    "[DataCollect2] YOLO DETECTION -> Recording Started",
                )
                return True
            return False

        def _detect_and_update_tracking(obs, phase: str, step_idx: int):
            """YOLO worker에 검출을 요청하고 최신 cache를 읽어 tracking 상태를 갱신한다."""
            nonlocal yolo_tracking_started
            interval = self._yolo_track_interval_steps if yolo_tracking_started else self._yolo_search_interval_steps
            if step_idx % interval == 0:
                self._submit_yolo_detection(obs, _port_kw, self._yolo_align_conf)

            detection, results, bgr, detections_by_camera, selected_camera = self._get_cached_yolo_detection(_port_kw)
            if detection is not None:
                if not yolo_tracking_started:
                    yolo_tracking_started = True
                    _check_and_start(obs, f"{phase}_yolo_handoff", step_idx, detection, results, bgr)
                    self.get_logger().info(
                        f"[DataCollect2] {phase} handoff: TF-guided -> YOLO-guided "
                        f"camera={selected_camera} views={sum(det is not None for det in detections_by_camera.values())}/3"
                    )
                elif not recording_started:
                    _check_and_start(obs, phase, step_idx, detection, results, bgr)
            return detection, results, bgr, detections_by_camera, selected_camera

        # Phase 1-A: TF alignment first, then YOLO-guided approach to the
        # triangulated stop offset once the port is visible.
        self.get_logger().info(
            f"━━━ Phase 1-A: TF -> YOLO Approach "
            f"({current_approach_z*100:.0f}cm -> triangulated "
            f"x/y/z <= {self._triangulation_stop_x_offset*1000:.1f}/"
            f"{self._triangulation_stop_y_offset*1000:.1f}/"
            f"{self._triangulation_stop_z_offset*1000:.1f}mm) ━━━"
        )
        approach_reached_triangulation_stop = False
        for t in range(self.approach_steps):
            t_norm = (t + 1) / float(self.approach_steps); t_smooth = interp_profile(t_norm, quintic=True)
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", cable_tip_frame), self._lookup_transform("base_link", "gripper/tcp")
                obs = get_observation()
                detection, results, bgr, detections_by_camera, selected_camera = _detect_and_update_tracking(obs, "approach", t)

                should_reset = (t < 60) and not yolo_tracking_started
                approach_z_offset = self._triangulation_stop_z_offset if yolo_tracking_started else current_approach_z
                pose, extras = self._planner.build_pose(
                    port_transform=current_port_tf, plug_transform=plug_tf, gripper_transform=gripper_tf,
                    slerp_fraction=t_smooth, position_fraction=t_smooth, z_offset=approach_z_offset,
                    reset_xy_integrator=should_reset
                )
                extras["z_offset"] = float(approach_z_offset)
                if yolo_tracking_started and detection is not None:
                    pose, yolo_extras = self._apply_yolo_correction(
                        pose=pose,
                        detection=detection,
                        image_shape=bgr.shape if bgr is not None else (1, 1, 3),
                        position_fraction=t_smooth,
                        port_axis=extras.get("port_axis"),
                    )
                    extras.update(yolo_extras)
                    extras.update(self._build_multiview_yolo_extras(detections_by_camera))
                    extras["yolo_selected_camera"] = selected_camera
                    measured_offsets, offset_extras = self._measure_triangulated_tip_to_port_offsets(
                        obs,
                        detections_by_camera,
                        plug_tf,
                        extras.get("port_axis"),
                        current_port_tf,
                    )
                    extras.update(offset_extras)
                    if (
                        self._triangulation_z_stop_enabled
                        and measured_offsets is not None
                        and measured_offsets["within_threshold"]
                    ):
                        approach_reached_triangulation_stop = True
                    extras["phase1a_guidance"] = "yolo"
                    measured_offset_text = (
                        f" tri_offsets_xyz: {measured_offsets['x']:+0.4f} "
                        f"{measured_offsets['y']:+0.4f} "
                        f"{measured_offsets['z']:0.4f} "
                        f"xy={measured_offsets['xy']:0.4f}"
                        if measured_offsets is not None
                        else " tri_offsets_xyz: n/a"
                    )
                    self.get_logger().info(
                        f"pfrac: {extras['position_fraction']:.3} guide: yolo "
                        f"z_target: {approach_z_offset:0.4} "
                        f"det: {extras['yolo_detected']} err: {extras['yolo_x_error_norm']:0.3} {extras['yolo_y_error_norm']:0.3} "
                        f"conf: {extras['yolo_confidence']:0.3} "
                        f"views: {extras['yolo_multiview_detect_count']}/3 "
                        f"cam: {selected_camera} corr: {extras['yolo_dx']:0.4} {extras['yolo_dy']:0.4}"
                        f"{measured_offset_text}"
                    )
                else:
                    extras["phase1a_guidance"] = "tf"
                    extras.update({
                        "triangulated_z_stop_enabled": bool(self._triangulation_z_stop_enabled),
                        "triangulated_tip_to_port_offsets_valid": False,
                        "triangulated_z_offset_valid": False,
                        "triangulated_xy_offset_valid": False,
                        "triangulated_z_stop_threshold": float(self._triangulation_stop_z_offset),
                        "triangulated_x_stop_threshold": float(self._triangulation_stop_x_offset),
                        "triangulated_y_stop_threshold": float(self._triangulation_stop_y_offset),
                    })
                    self.get_logger().info(
                        f"pfrac: {extras['position_fraction']:.3} guide: tf "
                        f"z_target: {approach_z_offset:0.4} "
                        f"xy_error: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} "
                        f"ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}"
                    )
                self._write_yolo_debug_video_frame(
                    obs,
                    results,
                    detection,
                    "approach",
                    t,
                    extras["phase1a_guidance"],
                )
                self.set_pose_target(move_robot, pose, stiffness=approach_stiffness, damping=approach_damping)
                if recording_started:
                    self._record_motion_step(recorder, "approach", task, current_port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=approach_stiffness, damping=approach_damping)
                    phase_step_counts["approach"] += 1
                if approach_reached_triangulation_stop:
                    self.get_logger().info(
                        f"[DataCollect2] Approach reached triangulated x/y/z thresholds: "
                        f"x<={self._triangulation_stop_x_offset:.4f}m "
                        f"y<={self._triangulation_stop_y_offset:.4f}m "
                        f"z<={self._triangulation_stop_z_offset:.4f}m"
                    )
                    break
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)
        if not approach_reached_triangulation_stop:
            self.get_logger().warn(
                "[DataCollect2] Approach did not reach triangulated x/y/z thresholds before COLLECT; "
                "COLLECT will still run from the planned port-relative pose."
            )
        if not recording_started:
            recording_started = True
            self.get_logger().warn(
                "[DataCollect2] YOLO trigger was not observed before COLLECT; "
                "recording COLLECT samples with yolo_lost metadata."
            )

        # Phase 1-B: Collect. Do not insert. Keep the cable tip near the
        # triangulated stop height and command a sampled port-local XY offset.
        self.get_logger().info(
            f"━━━ Phase 1-B: COLLECT {self.collect_pattern} "
            f"(max_radius={self.collect_gaussian_max_radius*1000:.1f}mm, "
            f"sigma={self.collect_gaussian_sigma*1000:.1f}mm, "
            f"spiral_radius={self.collect_start_radius*1000:.1f}->{self.collect_end_radius*1000:.1f}mm, "
            f"turns={self.collect_turns:.2f}, steps={self.collect_steps}, "
            f"z={self._triangulation_stop_z_offset:.4f}m) ━━━"
        )
        collect_steps = max(1, self.collect_steps)
        for collect_idx in range(collect_steps):
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", cable_tip_frame), self._lookup_transform("base_link", "gripper/tcp")

                pose, extras = self._planner.build_pose(
                    current_port_tf,
                    plug_tf,
                    gripper_tf,
                    z_offset=self._triangulation_stop_z_offset,
                    reset_xy_integrator=False,
                )
                extras["z_offset"] = float(self._triangulation_stop_z_offset)
                pose, collect_extras = self._apply_collect_offset(
                    pose,
                    current_port_tf,
                    extras.get("port_axis"),
                    collect_idx,
                )
                extras.update(collect_extras)
                obs = get_observation()
                detection, results, bgr, detections_by_camera, selected_camera = _detect_and_update_tracking(
                    obs,
                    "collect",
                    phase_step_counts["collect"],
                )
                if yolo_tracking_started and detection is not None:
                    _, yolo_extras = self._apply_yolo_correction(
                        pose=Pose(),
                        detection=detection,
                        image_shape=bgr.shape if bgr is not None else (1, 1, 3),
                        position_fraction=1.0,
                        port_axis=extras.get("port_axis"),
                    )
                    extras.update(yolo_extras)
                    extras.update(self._build_multiview_yolo_extras(detections_by_camera))
                    extras["yolo_selected_camera"] = selected_camera
                    measured_offsets, offset_extras = self._measure_triangulated_tip_to_port_offsets(
                        obs,
                        detections_by_camera,
                        plug_tf,
                        extras.get("port_axis"),
                        current_port_tf,
                    )
                    extras.update(offset_extras)
                    extras["collect_guidance"] = "spiral_yolo_observed"
                    measured_offset_text = (
                        f" tri_offsets_xyz: {measured_offsets['x']:+0.4f} "
                        f"{measured_offsets['y']:+0.4f} "
                        f"{measured_offsets['z']:0.4f} "
                        f"xy={measured_offsets['xy']:0.4f}"
                        if measured_offsets is not None
                        else " tri_offsets_xyz: n/a"
                    )
                    self.get_logger().info(
                        f"COLLECT step={collect_idx}/{collect_steps} "
                        f"radius={extras['collect_radius']*1000:.2f}mm "
                        f"theta={extras['collect_theta']:.3f} "
                        f"err: {extras['yolo_x_error_norm']:0.3} {extras['yolo_y_error_norm']:0.3} "
                        f"views: {extras['yolo_multiview_detect_count']}/3 "
                        f"cam: {selected_camera}"
                        f"{measured_offset_text}"
                    )
                else:
                    extras["collect_guidance"] = "spiral_yolo_lost"
                    extras.update({
                        "triangulated_z_stop_enabled": bool(self._triangulation_z_stop_enabled),
                        "triangulated_tip_to_port_offsets_valid": False,
                        "triangulated_z_offset_valid": False,
                        "triangulated_xy_offset_valid": False,
                        "triangulated_z_stop_threshold": float(self._triangulation_stop_z_offset),
                        "triangulated_x_stop_threshold": float(self._triangulation_stop_x_offset),
                        "triangulated_y_stop_threshold": float(self._triangulation_stop_y_offset),
                    })
                    self.get_logger().info(
                        f"COLLECT step={collect_idx}/{collect_steps} guide: yolo_lost "
                        f"radius={extras['collect_radius']*1000:.2f}mm "
                        f"xy_error: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} "
                        f"ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}"
                    )
                self._write_yolo_debug_video_frame(
                    obs,
                    results,
                    detection,
                    "collect",
                    phase_step_counts["collect"],
                    extras["collect_guidance"],
                )
                self.set_pose_target(move_robot, pose, stiffness=collect_stiffness, damping=collect_damping)
                if recording_started:
                    self._save_vision_offset_sample(
                        episode_name=episode_name,
                        task=task,
                        phase="collect",
                        step_idx=phase_step_counts["collect"],
                        obs=obs,
                        port_tf=current_port_tf,
                        plug_tf=plug_tf,
                        pose=pose,
                        extras=extras,
                        detections_by_camera=detections_by_camera,
                    )
                    self._record_motion_step(recorder, "collect", task, current_port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=collect_stiffness, damping=collect_damping)
                    phase_step_counts["collect"] += 1
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        self.sleep_for(0.5)
        return self._finish_data_collection_episode(
            episode_dir=episode_dir,
            task=task,
            recorder=recorder,
            phase_step_counts=phase_step_counts,
            status="ok",
        )

    def _write_episode_summary(self, episode_dir: Path, summary: dict) -> None:
        """에피소드 단위 요약 정보를 episode_summary.json 파일로 저장한다."""
        (episode_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _load_scenario_params(self, task) -> np.ndarray:
        """scenario parameter JSON에서 현재 task의 환경 파라미터를 읽어 11차원 벡터로 반환한다."""
        zero = np.zeros(11, dtype=np.float32)
        try:
            p = json.loads(self._scenario_params_file.read_text(encoding="utf-8")).get(task.id)
            if not p: return zero
            return np.array([p["trial_type"], p["rail_idx"], p["board_x"], p["board_y"], p["board_yaw"],
                             p["gripper_offset_x"], p["gripper_offset_y"], p["gripper_offset_z"],
                             p["nic_translation"], p["nic_yaw"], p["sc_translation"]], dtype=np.float32)
        except Exception: return zero


# 기존 로더가 파일 안의 DataCollect 심볼을 찾는 경우도 같이 지원한다.
DataCollect = DataCollect2
