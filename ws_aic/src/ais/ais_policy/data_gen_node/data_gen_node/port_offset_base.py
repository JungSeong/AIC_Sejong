from __future__ import annotations
"""
Shared runtime for PortOffsetCollect.
"""

import json
import os
import signal
import sys
import threading
import time
import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Transform, Vector3, Wrench
from data_gen_node.lib.cheatcode import CheatCodePlanner
from data_gen_node.port_offset_config import (
    DAMPING_DEFAULT,
    SFP_PLUG_REFERENCE_OFFSET_IN_CABLE_TIP_FRAME,
    STIFFNESS_DEFAULT,
    TOOL0_TO_OPTICAL,
    TOOL0_TO_TCP_Z,
    TRIANGULATION_STOP_Z_OFFSET_DEFAULT,
)
from data_gen_node.port_offset_geometry import (
    _matrix_to_rpy_xyz,
    _matrix_from_pose,
    _matrix_from_translation_quat,
    _quat_to_matrix_xyzw,
)
from tf2_ros import TransformException

def init_runtime(self, parent_node):
    """정책 실행에 필요한 planner, YOLO, 데이터셋, 토픽 구독, 종료 처리를 초기화한다."""
    
    # 1. 상태 및 제어 변수 초기화
    self._task: Optional[Task] = None
    # 나선형 진동(Oscillation) 방지를 위해 적분 한계를 낮추고(안전 범위), 부족한 힘은 Stiffness로 보상
    self._max_integrator_windup = 0.15 
    
    # 2. 제어 주기 및 경로 설정
    fps = int(os.environ.get("AIC_COLLECT_FPS", "0"))
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
            f"[PortOffsetCollect] Invalid AIC_COLLECT_PATTERN={self.collect_pattern}; using spiral"
        )
        self.collect_pattern = "spiral"
    self.collect_gaussian_sigma = float(os.environ.get("AIC_COLLECT_GAUSSIAN_SIGMA", "0.006"))
    self.collect_gaussian_max_radius = float(os.environ.get("AIC_COLLECT_GAUSSIAN_MAX_RADIUS", str(self.collect_start_radius)))
    self.collect_capture_settle_sec = float(os.environ.get("AIC_COLLECT_CAPTURE_SETTLE_SEC", "0.25"))
    seed_text = os.environ.get("AIC_COLLECT_RANDOM_SEED", "").strip()
    seed = int(seed_text) if seed_text else None
    self._collect_rng = np.random.default_rng(seed)

    # 4. YOLO 및 데이터셋 설정
    self._yolo_model_root = Path(
        os.environ.get(
            "AIC_APPROACH_YOLO_MODEL_ROOT",
            str(Path(__file__).resolve().parents[6] / "model" / "approach"),
        )
    ).expanduser()
    self._yolo_model_path = ""
    self._yolo_model = None
    self._yolo_model_port_kw = ""
    self._yolo_device = os.environ.get("AIC_YOLO_DEVICE", "").strip() or None
    self._yolo_trigger_conf = float(os.environ.get("AIC_YOLO_TRIGGER_CONF", "0.7"))
    self._yolo_align_conf = float(os.environ.get("AIC_YOLO_ALIGN_CONF", str(self._yolo_trigger_conf)))
    self._yolo_align_gain_x = float(os.environ.get("AIC_YOLO_ALIGN_GAIN_X", "0.035"))
    self._yolo_align_gain_y = float(os.environ.get("AIC_YOLO_ALIGN_GAIN_Y", "0.035"))
    self._yolo_align_sign_x = float(os.environ.get("AIC_YOLO_ALIGN_SIGN_X", "1.0"))
    self._yolo_align_sign_y = float(os.environ.get("AIC_YOLO_ALIGN_SIGN_Y", "1.0"))
    self._yolo_align_max_step = float(os.environ.get("AIC_YOLO_ALIGN_MAX_STEP", "0.006"))
    self._yolo_search_interval_steps = max(1, int(os.environ.get("AIC_YOLO_SEARCH_INTERVAL_STEPS", "5")))
    self._yolo_track_interval_steps = max(1, int(os.environ.get("AIC_YOLO_TRACK_INTERVAL_STEPS", "1")))
    self._yolo_cache_max_age_sec = float(os.environ.get("AIC_YOLO_CACHE_MAX_AGE_SEC", "0.75"))
    self._triangulation_z_stop_enabled = os.environ.get("AIC_TRIANGULATION_Z_STOP", "1").lower() not in {"0", "false", "no"}
    self._triangulation_min_views = max(2, int(os.environ.get("AIC_TRIANGULATION_MIN_VIEWS", "2")))
    self._triangulation_stop_z_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_Z_OFFSET", str(TRIANGULATION_STOP_Z_OFFSET_DEFAULT)))
    self._triangulation_stop_x_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_X_OFFSET", "0.005"))
    self._triangulation_stop_y_offset = float(os.environ.get("AIC_TRIANGULATION_STOP_Y_OFFSET", "0.005"))
    self._tool0_to_tcp_z = float(os.environ.get("AIC_TOOL0_TO_TCP_Z", str(TOOL0_TO_TCP_Z)))
    self._t_tool0_tcp = np.eye(4, dtype=float)
    self._t_tool0_tcp[2, 3] = self._tool0_to_tcp_z
    self._t_tool0_to_optical = {
        name: _matrix_from_translation_quat(translation, quat)
        for name, (translation, quat) in TOOL0_TO_OPTICAL.items()
    }
    self._yolo_control_camera = os.environ.get("AIC_YOLO_CONTROL_CAMERA", "center").strip().lower()
    if self._yolo_control_camera not in {"left", "center", "right"}:
        self.get_logger().warn(
            f"[PortOffsetCollect] Invalid AIC_YOLO_CONTROL_CAMERA={self._yolo_control_camera}; using center"
        )
        self._yolo_control_camera = "center"
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
    
    self._vision_offset_dataset_dir = Path(
        os.environ.get(
            "AIC_VISION_OFFSET_DATASET_DIR",
            str(self.capture_root / "vision_offset_dataset"),
        )
    )
    self._debug_image_dir = self._vision_offset_dataset_dir / "debug" / "image"
    self._vision_offset_images_dir = self._vision_offset_dataset_dir / "images"
    self._vision_offset_samples_path = self._vision_offset_dataset_dir / "samples.jsonl"
    if self._vision_offset_record_enabled:
        self._vision_offset_images_dir.mkdir(parents=True, exist_ok=True)
    self._scenario_params_file = Path(os.environ.get("AIC_SCENARIO_PARAMS_FILE", "/tmp/aic_scenario_params.json"))

    # 5. 서브스크립션 및 스레드 시작
    threading.Thread(target=self._yolo_detection_worker, daemon=True).start()

    # 6. 종료 처리
    try: signal.signal(signal.SIGTERM, self._on_sigterm)
    except ValueError: pass

    self._stop_file = Path(os.environ.get("AIC_STOP_FILE", "/tmp/aic_policy_stop"))
    threading.Thread(target=self._watch_stop_file, daemon=True).start()

    self.get_logger().info(f"[PortOffsetCollect] YOLO-guided Policy Initialized. Root: {self.capture_root}")

# ── 인프라 로직 ──────────────────────────────────────────────────────────

def _yolo_model_path_for_port(self, port_kw: str) -> Path:
    """포트 타입에 맞는 approach YOLO weight 경로를 반환한다."""
    override_key = "AIC_SFP_YOLO_MODEL_PATH" if port_kw == "sfp" else "AIC_SC_YOLO_MODEL_PATH"
    override_path = os.environ.get(override_key, "").strip()
    if override_path:
        return Path(override_path).expanduser()
    return self._yolo_model_root / port_kw.upper() / "weights" / "best.pt"


def _init_yolo(self, port_kw: str) -> bool:
    """포트 타입별 YOLO weight 파일을 로드해서 포트 검출 모델을 준비한다."""
    port_kw = "sfp" if str(port_kw).lower() == "sfp" else "sc"
    if self._yolo_model is not None and self._yolo_model_port_kw == port_kw:
        return True

    model_path = self._yolo_model_path_for_port(port_kw)
    self._yolo_model = None
    self._yolo_model_port_kw = ""
    self._yolo_model_path = str(model_path)
    if not model_path.exists():
        self.get_logger().warn(
            f"[PortOffsetCollect] {port_kw.upper()} YOLO model file not found: {model_path}; "
            "YOLO triangulation approach cannot run"
        )
        return False
    try:
        from ultralytics import YOLO
        self._yolo_model = YOLO(str(model_path))
        self._yolo_model_port_kw = port_kw
        self.get_logger().info(
            f"[PortOffsetCollect] {port_kw.upper()} YOLO model loaded: {model_path}"
        )
        if self._yolo_device:
            self.get_logger().info(f"[PortOffsetCollect] YOLO device override: {self._yolo_device}")
        return True
    except Exception as exc:
        self.get_logger().warn(f"[PortOffsetCollect] Failed to load YOLO model: {exc}")
        self._yolo_model = None
        self._yolo_model_port_kw = ""
        return False

def _watch_stop_file(self) -> None:
    """AIC_STOP_FILE이 생기면 즉시 프로세스를 종료한다."""
    while True:
        if self._stop_file.exists(): os._exit(0)
        time.sleep(0.5)

def _on_sigterm(self, signum, frame) -> None:
    """SIGTERM을 받았을 때 정상 종료한다."""
    raise SystemExit(0)

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
        self.get_logger().info(f"[PortOffsetCollect] Using port entrance frame: {entrance_frame}")
        return entrance_frame
    self.get_logger().warn(f"[PortOffsetCollect] Port entrance TF unavailable, falling back to: {port_frame}")
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
            self.get_logger().info(f"[PortOffsetCollect] Using cable tip frame: {frame}")
            return frame

    fallback_frame = f"{cable_prefix}/{plug_name}_link"
    self.get_logger().warn(
        f"[PortOffsetCollect] Cable tip TF unavailable, falling back to plug frame: {fallback_frame}"
    )
    return fallback_frame

def _wait_for_yolo_model(self, port_kw: str) -> bool:
    """현재 task 포트 타입에 맞는 YOLO 모델을 로드하고 사용 가능 여부를 반환한다."""
    if self._init_yolo(port_kw):
        return True
    if self._yolo_model is None:
        self.get_logger().warn(
            "[PortOffsetCollect] YOLO model is not available; "
            "YOLO triangulation approach cannot run"
        )
        return False
    return self._yolo_model_port_kw == port_kw

def set_pose_target(self, move_robot, pose, stiffness=None, damping=None):
    """주어진 목표 pose와 impedance 값으로 controller에 MotionUpdate 명령을 보낸다."""
    _s = stiffness if stiffness is not None else STIFFNESS_DEFAULT
    _d = damping if damping is not None else DAMPING_DEFAULT
    mu = MotionUpdate(
        header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
        pose=pose, target_stiffness=np.diag(_s).flatten(), target_damping=np.diag(_d).flatten(),
        feedforward_wrench_at_tip=Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)),
        wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION),
    )
    try: move_robot(motion_update=mu)
    except Exception: pass

def _transform_translation_array(self, transform: Transform) -> np.ndarray:
    return np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=float,
    )

def _transform_rotation_matrix(self, transform: Transform) -> np.ndarray:
    return _quat_to_matrix_xyzw(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )

def _shift_transform_origin(self, transform: Transform, local_offset_xyz: np.ndarray) -> Transform:
    """Return transform whose origin is shifted by local_offset_xyz in transform frame."""
    local_offset = np.asarray(local_offset_xyz, dtype=float)
    shifted_xyz = self._transform_translation_array(transform) + self._transform_rotation_matrix(transform) @ local_offset
    shifted = Transform()
    shifted.translation.x = float(shifted_xyz[0])
    shifted.translation.y = float(shifted_xyz[1])
    shifted.translation.z = float(shifted_xyz[2])
    shifted.rotation.x = float(transform.rotation.x)
    shifted.rotation.y = float(transform.rotation.y)
    shifted.rotation.z = float(transform.rotation.z)
    shifted.rotation.w = float(transform.rotation.w)
    return shifted

def _plug_location_label_in_base_frame(
    self,
    port_tf: Transform,
    plug_tf: Transform,
) -> dict[str, dict[str, float]]:
    """settle 후 실제 plug reference 위치와 정렬 correction을 base_link 기준으로 계산한다."""
    port_position = self._transform_translation_array(port_tf)
    port_rotation = self._transform_rotation_matrix(port_tf)
    plug_position = self._transform_translation_array(plug_tf)
    plug_rotation = self._transform_rotation_matrix(plug_tf)

    location_position = plug_position - port_position
    location_rotation = plug_rotation @ port_rotation.T
    location_roll, location_pitch, location_yaw = _matrix_to_rpy_xyz(location_rotation)

    label_position = port_position - plug_position
    label_rotation = port_rotation @ plug_rotation.T
    label_roll, label_pitch, label_yaw = _matrix_to_rpy_xyz(label_rotation)

    location = {
        "x_m": float(location_position[0]),
        "y_m": float(location_position[1]),
        "z_m": float(location_position[2]),
        "roll_rad": float(location_roll),
        "pitch_rad": float(location_pitch),
        "yaw_rad": float(location_yaw),
    }
    label = {
        "x_m": float(label_position[0]),
        "y_m": float(label_position[1]),
        "z_m": float(label_position[2]),
        "roll_rad": float(label_roll),
        "pitch_rad": float(label_pitch),
        "yaw_rad": float(label_yaw),
    }
    return {"location": location, "label": label}

def _plug_reference_offset_local(self, task: Task, cable_tip_frame: str) -> np.ndarray:
    """Offset from selected plug TF origin to the physical point used as label/control reference."""
    is_sfp = (
        "sfp" in str(task.plug_type).lower()
        or "sfp" in str(task.plug_name).lower()
        or "sfp" in cable_tip_frame.lower()
    )
    if not is_sfp:
        return np.zeros(3, dtype=float)
    return np.array(
        [
            float(os.environ.get("AIC_SFP_PLUG_REFERENCE_X", str(SFP_PLUG_REFERENCE_OFFSET_IN_CABLE_TIP_FRAME[0]))),
            float(os.environ.get("AIC_SFP_PLUG_REFERENCE_Y", str(SFP_PLUG_REFERENCE_OFFSET_IN_CABLE_TIP_FRAME[1]))),
            float(os.environ.get("AIC_SFP_PLUG_REFERENCE_Z", str(SFP_PLUG_REFERENCE_OFFSET_IN_CABLE_TIP_FRAME[2]))),
        ],
        dtype=float,
    )

def _plug_reference_metadata(
    self,
    task: Task,
    cable_tip_frame: str,
    offset_local_xyz: np.ndarray,
) -> dict[str, Any]:
    is_sfp = (
        "sfp" in str(task.plug_type).lower()
        or "sfp" in str(task.plug_name).lower()
        or "sfp" in cable_tip_frame.lower()
    )
    return {
        "plug_frame": cable_tip_frame,
        "point_name": "sfp_tip_top_center" if is_sfp else "plug_frame_origin",
        "local_offset_xyz_m": [float(value) for value in np.asarray(offset_local_xyz, dtype=float)],
        "description": (
            "SFP contact-collision top center; zero label means this point is on the port entrance frame."
            if is_sfp
            else "Selected plug frame origin."
        ),
    }

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

def _image_msg_to_bgr(self, img_msg, camera_name: str = "image") -> Optional[np.ndarray]:
    """ROS Image 메시지를 OpenCV에서 쓰는 BGR numpy 이미지로 변환한다."""
    if img_msg is None or img_msg.width == 0 or img_msg.height == 0:
        return None
    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
    except ValueError:
        self.get_logger().warn(f"[PortOffsetCollect] Invalid {camera_name} buffer size")
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


def _target_class_id_for_port(self, port_kw: str) -> int:
    """FinalPolicy와 같은 환경변수/기본값으로 포트 타입별 YOLO target class id를 결정한다."""
    if str(port_kw).lower() == "sc":
        return int(os.environ.get("AIC_DEBUG_SC_TARGET_CLASS_ID", "0"))
    return int(os.environ.get("AIC_DEBUG_SFP_TARGET_CLASS_ID", "0"))


def _target_port_index(self) -> Optional[int]:
    """task.port_name 끝 숫자를 SFP keypoint 그룹 선택용 포트 인덱스로 읽는다."""
    task = getattr(self, "_task", None)
    text = str(getattr(task, "port_name", "") or "")
    for token in reversed(text.split("_")):
        if token.isdigit():
            return int(token)
    return None


def _detect_port_from_bgr(self, bgr: np.ndarray, port_kw: str, conf: float) -> tuple[Optional[dict[str, Any]], Any]:
    """단일 BGR 이미지에서 FinalPolicy와 같은 class id 기준으로 목표 포트를 검출한다."""
    if self._yolo_model is None or bgr is None:
        return None, None
    predict_kwargs = {"verbose": False, "conf": conf}
    yolo_device = getattr(self, "_yolo_device", None)
    if yolo_device:
        predict_kwargs["device"] = yolo_device
    results = self._yolo_model(bgr, **predict_kwargs)
    target_class_id = self._target_class_id_for_port(port_kw)
    target_port_index = self._target_port_index()
    best_detection = None
    for result in results:
        names = getattr(result, "names", {})
        keypoints_xy = None
        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
        for box_idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            if class_id != target_class_id:
                continue
            class_name = names.get(class_id, "").lower()
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0

            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            point_name = "bbox_center"
            port_index = None
            if keypoints_xy is not None:
                if box_idx < len(keypoints_xy):
                    kpts = np.asarray(keypoints_xy[box_idx], dtype=float)
                    if port_kw == "sfp" and len(kpts) >= 8:
                        chosen_index = 0 if target_port_index is None else int(target_port_index)
                        chosen_index = 1 if chosen_index == 1 else 0
                        start = 4 * chosen_index
                        group = kpts[start:start + 4]
                        if np.all(np.isfinite(group)):
                            center_x = float(np.mean(group[:, 0]))
                            center_y = float(np.mean(group[:, 1]))
                            point_name = f"sfp_port_{chosen_index}"
                            port_index = chosen_index
                    elif port_kw == "sc" and len(kpts) >= 4:
                        group = kpts[:4]
                        if np.all(np.isfinite(group)):
                            center_x = float(np.mean(group[:, 0]))
                            center_y = float(np.mean(group[:, 1]))
                            point_name = "sc_port"

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "center_x": center_x,
                "center_y": center_y,
                "area_ratio": ((x2 - x1) * (y2 - y1)) / float(bgr.shape[0] * bgr.shape[1]),
                "point_name": point_name,
                "port_index": port_index,
                "target_class_id": target_class_id,
            }
            if best_detection is None or detection["confidence"] > best_detection["confidence"]:
                best_detection = detection
    return best_detection, results

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
    """left/center/right YOLO 검출 상태와 중심점 통계를 metadata extras로 정리한다."""
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
    """YOLO 검출 요청을 처리"""
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
        self.get_logger().info(f"[PortOffsetCollect] YOLO debug frame saved: {image_path}")
    except Exception as exc:
        self.get_logger().warn(f"[PortOffsetCollect] Failed to save YOLO debug frame: {exc}")

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

def _upload_vision_offset_dataset_to_hub(
    self,
    *,
    task: Task,
    status: str,
    collect_steps: int,
) -> dict[str, Any]:
    """수집 완료 후 vision-offset dataset 디렉터리를 Hugging Face dataset repo에 업로드한다."""
    if not getattr(self, "_rpy_push_to_hub", False):
        return {"enabled": False, "success": False, "reason": "disabled"}
    if status != "ok":
        return {"enabled": True, "success": False, "reason": f"skipped_status_{status}"}
    if collect_steps <= 0:
        return {"enabled": True, "success": False, "reason": "no_collect_samples"}
    upload_on_port_type = str(
        getattr(self, "_rpy_hf_upload_on_port_type", "") or ""
    ).strip().lower()
    task_port_type = str(getattr(task, "port_type", "") or "").strip().lower()
    if upload_on_port_type and task_port_type != upload_on_port_type:
        return {
            "enabled": True,
            "success": False,
            "reason": f"waiting_for_port_type_{upload_on_port_type}",
            "current_port_type": task_port_type,
        }

    repo_id = str(getattr(self, "_rpy_hf_repo_id", "") or "").strip()
    if not repo_id:
        reason = "AIC_VISION_OFFSET_REPO_ID is not set"
        self.get_logger().warn(f"[PortOffsetCollect] HF upload skipped: {reason}")
        return {"enabled": True, "success": False, "reason": reason}

    dataset_dir = Path(getattr(self, "_rpy_dataset_dir"))
    revision = str(getattr(self, "_rpy_hf_revision", "") or "main").strip() or "main"
    path_in_repo = str(getattr(self, "_rpy_hf_path_in_repo", "") or "").strip() or None
    private = bool(getattr(self, "_rpy_hf_private", True))
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            user_info = api.whoami()
        except Exception as exc:
            reason = (
                "Hugging Face authentication failed. Run `pixi run hf auth login` "
                "or set HF_TOKEN with write permission."
            )
            self.get_logger().error(f"[PortOffsetCollect] {reason} Details: {exc}")
            return {
                "enabled": True,
                "success": False,
                "repo_id": repo_id,
                "repo_type": "dataset",
                "revision": revision,
                "path_in_repo": path_in_repo or "",
                "reason": reason,
                "error": str(exc),
            }
        self.get_logger().info(
            "[PortOffsetCollect] Uploading vision-offset dataset to "
            f"https://huggingface.co/datasets/{repo_id}/tree/{revision} "
            f"as {user_info.get('name', 'unknown')}"
        )
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        if revision != "main":
            try:
                branches = [
                    branch.name
                    for branch in api.list_repo_refs(repo_id, repo_type="dataset").branches
                ]
                if revision not in branches:
                    api.create_branch(
                        repo_id=repo_id,
                        repo_type="dataset",
                        branch=revision,
                    )
            except Exception as exc:
                self.get_logger().warn(
                    f"[PortOffsetCollect] HF branch preparation warning: {exc}"
                )
        if path_in_repo:
            self.get_logger().warn(
                "[PortOffsetCollect] AIC_VISION_OFFSET_HF_PATH_IN_REPO is ignored "
                "when using upload_large_folder"
            )
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            folder_path=str(dataset_dir),
            ignore_patterns=["*.tmp", "*.lock", "__pycache__/*", ".DS_Store"],
            private=private,
        )
        url = f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"
        self.get_logger().info(f"[PortOffsetCollect] HF upload complete: {url}")
        return {
            "enabled": True,
            "success": True,
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "path_in_repo": path_in_repo or "",
            "url": url,
            "upload_method": "upload_large_folder",
        }
    except Exception as exc:
        self.get_logger().error(f"[PortOffsetCollect] HF upload failed: {exc}")
        return {
            "enabled": True,
            "success": False,
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "path_in_repo": path_in_repo or "",
            "reason": str(exc),
        }

# ── 메인 에피소드 수집 로직 ───────────────────────────────────────────────
def _finish_data_collection_episode(
    self,
    *,
    episode_dir: Path,
    task: Task,
    phase_step_counts: dict[str, int],
    status: str,
    detail: str = "",
) -> bool:
    """삽입 성공과 별개로 데이터 수집 task를 마무리하고 engine에는 완료를 알린다."""
    insertion_success = False
    summary = {
        "task_id": task.id,
        "success": insertion_success,
        "insertion_success": insertion_success,
        "task_completed_for_engine": True,
        "status": status,
        "detail": detail,
        "mode": "vision_offset",
        "lift_up_steps": int(phase_step_counts.get("lift_up", 0)),
        "approach_steps": int(phase_step_counts.get("approach", 0)),
        "collect_steps": int(phase_step_counts.get("collect", 0)),
        "collect_pattern": self.collect_pattern,
        "collect_start_radius": self.collect_start_radius,
        "collect_end_radius": self.collect_end_radius,
        "collect_turns": self.collect_turns,
        "collect_gaussian_sigma": self.collect_gaussian_sigma,
        "collect_gaussian_max_radius": self.collect_gaussian_max_radius,
    }
    self._write_episode_summary(episode_dir, summary)
    summary["hub_upload"] = self._upload_vision_offset_dataset_to_hub(
        task=task,
        status=status,
        collect_steps=int(phase_step_counts.get("collect", 0)),
    )
    self._write_episode_summary(episode_dir, summary)
    self.get_logger().info(
        f"DataCollect complete. status={status} "
        f"collect_steps={phase_step_counts.get('collect', 0)} "
        f"insertion_success={insertion_success} task_completed_for_engine=True"
    )
    return True

def _write_episode_summary(self, episode_dir: Path, summary: dict) -> None:
    """에피소드 단위 요약 정보를 episode_summary.json 파일로 저장한다."""
    (episode_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
