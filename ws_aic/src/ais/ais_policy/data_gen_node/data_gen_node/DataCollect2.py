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
from std_msgs.msg import Header, String

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform, Vector3, Wrench, Pose, Point, Quaternion

from .lib.recording import (
    LeRobotRecorder,
    LEROBOT_FEATURES,
    _LEROBOT_AVAILABLE,
)
from .lib.cheatcode import CheatCodePlanner
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


# ── 유틸리티 함수 ────────────────────────────────────────────────────────

def s_curve_quintic(t: float) -> float:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    return s_curve_quintic(t) if quintic else (3.0 * t**2 - 2.0 * t**3)


class DataCollect2(Policy):
    """
    YOLO 기반 접근 정렬 데이터 수집 정책 클래스.
    """

    # ── 임피던스 설정 (성공률 최적화) ──────────────────────────────────────────
    _STIFFNESS_DEFAULT = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    _DAMPING_DEFAULT   = [40.0, 40.0, 40.0, 20.0, 20.0, 20.0]

    # SFP 커넥터 (부드러운 삽입)
    _SFP_INSERT_STIFFNESS = [20.0, 20.0, 250.0, 10.0, 10.0, 40.0]
    _SFP_INSERT_DAMPING   = [10.0, 10.0, 60.0, 5.0, 5.0, 15.0]

    # SC 커넥터 (미끄러짐 유도를 위한 Compliance를 유지하되, 케이블 저항을 이길 수 있도록 강성/감쇠 상향)
    _SC_INSERT_STIFFNESS = [51.0, 50.0, 300.0, 15.0, 15.0, 40.0]
    _SC_INSERT_DAMPING   = [31.0, 30.0, 87.0, 8.0, 8.0, 15.0]

    # ── 모션 플래닝 상수 ──────────────────────────────────────────────────
    _SC_INSERT_MIN_Z_OFFSET: float = -0.025  
    _NEAR_APPROACH_Z_OFFSET: float = 0.010  # 1cm

    def __init__(self, parent_node):
        super().__init__(parent_node)
        
        # 1. 상태 및 제어 변수 초기화
        self._task: Optional[Task] = None
        self._latest_insertion_event: Optional[str] = None
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
        self.insert_z_step = float(os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_Z_STEP", "0.00025"))
        self.insert_min_z_offset = float(os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_MIN_Z_OFFSET", "-0.015"))
        self.stabilize_sec = float(os.environ.get("AIC_CAPTURE_CHEATCODE_STABILIZE_SEC", "5.0"))
        self.sc_insert_min_z_offset = float(os.environ.get("AIC_CAPTURE_SC_INSERT_MIN_Z_OFFSET", str(self._SC_INSERT_MIN_Z_OFFSET)))

        # 4. YOLO 및 데이터셋 설정
        self._yolo_model_path = str(Path(__file__).resolve().parents[5] / "src" / "model" / "ais_yolo-2" / "weights" / "best.pt")
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
        
        _lerobot_out = os.environ.get("AIC_LEROBOT_OUT_DIR", "lerobot")
        self._debug_image_dir = Path(_lerobot_out) / self._lerobot_version / "debug"
        self._scenario_params_file = Path(os.environ.get("AIC_SCENARIO_PARAMS_FILE", "/tmp/aic_scenario_params.json"))

        # 5. 서브스크립션 및 스레드 시작
        self._insertion_event_sub = self._parent_node.create_subscription(
            String, "/scoring/insertion_event", self._insertion_event_callback, 10
        )
        self._init_lerobot_dataset()
        threading.Thread(target=self._init_yolo, daemon=True).start()

        # 6. 종료 처리
        atexit.register(self._finalize_dataset)
        try: signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError: pass

        self._stop_file = Path(os.environ.get("AIC_STOP_FILE", "/tmp/aic_policy_stop"))
        threading.Thread(target=self._watch_stop_file, daemon=True).start()

        self.get_logger().info(f"[DataCollect2] YOLO-guided Policy Initialized. Root: {self.capture_root}")

    # ── 인프라 로직 ──────────────────────────────────────────────────────────

    def _init_yolo(self) -> None:
        model_path = Path(self._yolo_model_path)
        self._yolo_model = None
        if not model_path.exists(): return
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(model_path))
        except Exception: pass

    @staticmethod
    def _is_valid_dataset_root(root: Path) -> bool:
        return (root / "meta" / "info.json").exists() and (root / "meta" / "tasks.parquet").exists()

    def _init_lerobot_dataset(self) -> None:
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
        if self._lerobot_dataset is not None:
            try: self._lerobot_dataset.finalize()
            except Exception: pass
            finally: self._lerobot_dataset = None

    def _watch_stop_file(self) -> None:
        while True:
            if self._stop_file.exists(): self._finalize_dataset(); os._exit(0)
            time.sleep(0.5)

    def _on_sigterm(self, signum, frame) -> None: self._finalize_dataset(); raise SystemExit(0)

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event = msg.data.strip().strip("/")

    def _has_successful_insertion(self, task: Task) -> bool:
        if not self._latest_insertion_event: return False
        tokens = [t for t in self._latest_insertion_event.split("/") if t]
        return len(tokens) >= 2 and tokens[0] == task.target_module_name and tokens[1] == task.port_name

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        while (self.time_now() - start) < Duration(seconds=timeout_sec):
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException: self.sleep_for(0.1)
        return False

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform

    def _select_port_frame(self, task: Task) -> str:
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = f"{port_frame}_entrance"
        if self._wait_for_tf("base_link", entrance_frame, timeout_sec=2.0):
            self.get_logger().info(f"[DataCollect2] Using port entrance frame: {entrance_frame}")
            return entrance_frame
        self.get_logger().warn(f"[DataCollect2] Port entrance TF unavailable, falling back to: {port_frame}")
        return port_frame

    def _wait_for_yolo_model(self) -> bool:
        start = time.time()
        while self._yolo_model is None and (time.time() - start) < self._yolo_model_wait_sec:
            self.sleep_for(0.1)
        if self._yolo_model is None:
            self.get_logger().warn("[DataCollect2] YOLO model is not available; Phase 1-A will remain TF-guided")
            return False
        return True

    def set_pose_target(self, move_robot, pose, stiffness=None, damping=None):
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
        mu = MotionUpdate()
        mu.header.frame_id, mu.header.stamp = "base_link", self.get_clock().now().to_msg()
        mu.pose = pose
        _s = stiffness if stiffness is not None else self._STIFFNESS_DEFAULT
        _d = damping if damping is not None else self._DAMPING_DEFAULT
        mu.target_stiffness, mu.target_damping = list(np.diag(_s).flatten()), list(np.diag(_d).flatten())
        mu.trajectory_generation_mode = TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION)
        return mu

    def _record_motion_step(self, recorder, phase, task, port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=None, damping=None):
        if obs is None: return
        action = self._motion_update_from_pose(pose, stiffness, damping)
        recorder.record_step(
            phase=phase, task=task, obs=obs, action=action,
            port_tf=port_tf, plug_tf=plug_tf, gripper_tf=gripper_tf, extras=extras,
            stiffness=stiffness, damping=damping
        )

    def _image_msg_to_bgr(self, img_msg) -> Optional[np.ndarray]:
        if img_msg is None or img_msg.width == 0 or img_msg.height == 0:
            return None
        try:
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        except ValueError:
            self.get_logger().warn("[DataCollect2] Invalid center_image buffer size")
            return None
        if img_msg.encoding == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    def _detect_port_from_bgr(self, bgr: np.ndarray, port_kw: str, conf: float) -> tuple[Optional[dict[str, Any]], Any]:
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
        if obs is None:
            return None, None, None
        bgr = self._image_msg_to_bgr(obs.center_image)
        if bgr is None:
            return None, None, None
        detection, results = self._detect_port_from_bgr(bgr, port_kw, conf)
        return detection, results, bgr

    def _log_yolo_detection(self, detection: dict[str, Any], prefix: str) -> None:
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

    def _build_yolo_guided_pose(
        self,
        gripper_tf: Transform,
        detection: Optional[dict[str, Any]],
        image_shape: tuple[int, int, int],
        z_offset: float,
        position_fraction: float,
    ) -> tuple[Pose, dict[str, Any]]:
        gripper_xyz = gripper_tf.translation
        q = gripper_tf.rotation
        target_x, target_y = float(gripper_xyz.x), float(gripper_xyz.y)
        x_error_norm = 0.0
        y_error_norm = 0.0
        area_ratio = 0.0

        if detection is not None:
            height, width = image_shape[:2]
            x_error_norm = ((width * 0.5) - detection["center_x"]) / max(width * 0.5, 1.0)
            y_error_norm = ((height * 0.5) - detection["center_y"]) / max(height * 0.5, 1.0)
            area_ratio = float(detection["area_ratio"])
            dx = float(np.clip(self._yolo_align_sign_x * self._yolo_align_gain_x * x_error_norm, -self._yolo_align_max_step, self._yolo_align_max_step))
            dy = float(np.clip(self._yolo_align_sign_y * self._yolo_align_gain_y * y_error_norm, -self._yolo_align_max_step, self._yolo_align_max_step))
            target_x += dx
            target_y += dy

        target_z = float(gripper_xyz.z)
        pose = Pose(
            position=Point(
                x=float((1.0 - position_fraction) * gripper_xyz.x + position_fraction * target_x),
                y=float((1.0 - position_fraction) * gripper_xyz.y + position_fraction * target_y),
                z=target_z,
            ),
            orientation=Quaternion(w=float(q.w), x=float(q.x), y=float(q.y), z=float(q.z)),
        )
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
            "target_x": float(pose.position.x),
            "target_y": float(pose.position.y),
            "target_z": float(pose.position.z),
            "z_offset": float(z_offset),
            "position_fraction": float(position_fraction),
        }
        return pose, extras

    # ── 메인 에피소드 수집 로직 ───────────────────────────────────────────────

    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self._task = task; self._latest_insertion_event = None; self._planner.reset(); send_feedback("data collect running")
        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name; episode_dir.mkdir(parents=True, exist_ok=True)
        if self._lerobot_dataset is None: return False
        
        scenario_params_vec = self._load_scenario_params(task)
        recorder = LeRobotRecorder(self._lerobot_dataset, scenario_params_vec)
        
        port_frame, plug_frame = self._select_port_frame(task), f"{task.cable_name}/{task.plug_name}_link"
        if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf("base_link", plug_frame): return False
        self._wait_for_yolo_model()

        start_time = time.time(); phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}
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
            insert_stiffness = self._SC_INSERT_STIFFNESS
            insert_damping = self._SC_INSERT_DAMPING
        else:
            # SFP: 문제없이 작동하던 기존 설정 복구
            self._planner.i_gain = float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15"))
            self._planner.max_integrator_windup = 0.08
            approach_stiffness = self._STIFFNESS_DEFAULT
            approach_damping = self._DAMPING_DEFAULT
            insert_stiffness = self._SFP_INSERT_STIFFNESS
            insert_damping = self._SFP_INSERT_DAMPING

        current_approach_z = 0.180 if _port_kw == "sfp" else 0.050

        def _check_and_start(obs, phase: str = "trigger", step_idx: int = 0, detection=None, results=None, bgr=None) -> bool:
            nonlocal recording_started
            if recording_started or obs is None: return
            if detection is None or results is None or bgr is None:
                detection, results, bgr = self._detect_port_from_obs(obs, _port_kw, self._yolo_trigger_conf)
            if detection is not None:
                self._save_yolo_debug_frame(bgr, results, detection, task, phase, step_idx)
                recording_started = True
                self._log_yolo_detection(
                    detection,
                    "[DataCollect2] YOLO DETECTION -> Recording Started",
                )
                return True
            return False

        # Phase 1-A: TF alignment first, then YOLO bbox tracking after detection.
        self.get_logger().info(f"━━━ Phase 1-A: TF -> YOLO Alignment at {current_approach_z*100:.0f}cm ━━━")
        for t in range(self.approach_steps):
            t_norm = (t + 1) / float(self.approach_steps); t_smooth = interp_profile(t_norm, quintic=True)
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
                obs = get_observation()
                detection, results, bgr = self._detect_port_from_obs(obs, _port_kw, self._yolo_align_conf)
                if detection is not None and not yolo_tracking_started:
                    yolo_tracking_started = True
                    _check_and_start(obs, "approach_yolo_handoff", t, detection, results, bgr)
                    self.get_logger().info("[DataCollect2] Phase 1-A handoff: TF-guided -> YOLO-guided")

                if yolo_tracking_started:
                    pose, extras = self._build_yolo_guided_pose(
                        gripper_tf=gripper_tf,
                        detection=detection,
                        image_shape=bgr.shape if bgr is not None else (1, 1, 3),
                        z_offset=current_approach_z,
                        position_fraction=t_smooth,
                    )
                    extras["phase1a_guidance"] = "yolo"
                    self.get_logger().info(
                        f"pfrac: {extras['position_fraction']:.3} guide: yolo "
                        f"det: {extras['yolo_detected']} err: {extras['yolo_x_error_norm']:0.3} {extras['yolo_y_error_norm']:0.3} "
                        f"conf: {extras['yolo_confidence']:0.3}"
                    )
                else:
                    should_reset = (t < 60)
                    pose, extras = self._planner.build_pose(
                        port_transform=current_port_tf, plug_transform=plug_tf, gripper_transform=gripper_tf,
                        slerp_fraction=t_smooth, position_fraction=t_smooth, z_offset=current_approach_z,
                        reset_xy_integrator=should_reset
                    )
                    extras["phase1a_guidance"] = "tf"
                    self.get_logger().info(
                        f"pfrac: {extras['position_fraction']:.3} guide: tf "
                        f"xy_error: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} "
                        f"ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}"
                    )
                self.set_pose_target(move_robot, pose, stiffness=approach_stiffness, damping=approach_damping)
                _check_and_start(obs, "approach_yolo", t, detection, results, bgr)
                if recording_started:
                    self._record_motion_step(recorder, "approach", task, current_port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=approach_stiffness, damping=approach_damping)
                    phase_step_counts["approach"] += 1
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        # Phase 1-D: Continuous Descent (Fine alignment with Dynamic Tracking)
        self.get_logger().info(f"━━━ Phase 1-D: Continuous Descent ━━━")
        dist_to_descend = current_approach_z - self._NEAR_APPROACH_Z_OFFSET
        mid_steps = int(dist_to_descend * 100 * 20)
        for t in range(mid_steps):
            t_norm = (t + 1) / float(mid_steps); t_smooth = interp_profile(t_norm, quintic=True)
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
                cur_z_offset = current_approach_z * (1.0 - t_smooth) + self._NEAR_APPROACH_Z_OFFSET * t_smooth

                pose, extras = self._planner.build_pose(current_port_tf, plug_tf, gripper_tf, z_offset=cur_z_offset, reset_xy_integrator=False)
                self.get_logger().info(f"z_off: {cur_z_offset:0.4} xy_err: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}")
                self.set_pose_target(move_robot, pose, stiffness=approach_stiffness, damping=approach_damping)
                obs = get_observation(); _check_and_start(obs, "approach_descent", t)
                if recording_started:
                    self._record_motion_step(recorder, "approach", task, current_port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=approach_stiffness, damping=approach_damping)
                    phase_step_counts["approach"] += 1
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        # Phase 3: Insert (Compliance Optimized for Slipping)
        self.get_logger().info("━━━ Phase 3: Insert (Compliance ON) ━━━")
        z_limit = (self.sc_insert_min_z_offset if _port_kw == "sc" else self.insert_min_z_offset)
        z_offset = self._NEAR_APPROACH_Z_OFFSET
        while z_offset >= z_limit:
            if self._has_successful_insertion(task): break
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")

                pose, extras = self._planner.build_pose(current_port_tf, plug_tf, gripper_tf, z_offset=z_offset)
                self.get_logger().info(f"INSERT z: {z_offset:0.4} xy_err: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}")
                self.set_pose_target(move_robot, pose, stiffness=insert_stiffness, damping=insert_damping)
                obs = get_observation(); _check_and_start(obs, "insert", phase_step_counts["insert"])
                if recording_started:
                    self._record_motion_step(recorder, "insert", task, current_port_tf, plug_tf, gripper_tf, obs, pose, extras, stiffness=insert_stiffness, damping=insert_damping)
                    phase_step_counts["insert"] += 1
                z_offset -= self.insert_z_step
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        success = self._has_successful_insertion(task)
        self.sleep_for(0.5 if success else self.stabilize_sec)
        try:
            plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
            obs = get_observation()
            if obs and recording_started:
                recorder.record_terminal_step(
                    phase="stabilize", task=task, obs=obs, port_tf=current_port_tf,
                    plug_tf=plug_tf, gripper_tf=gripper_tf, extras={"z_offset": z_offset},
                    stiffness=insert_stiffness, damping=insert_damping
                )
                if "stabilize" not in phase_step_counts: phase_step_counts["stabilize"] = 0
                phase_step_counts["stabilize"] += 1
        except TransformException: pass

        success = self._has_successful_insertion(task)
        self.sleep_for(0.5 if success else self.stabilize_sec)
        recorder.save_episode(insertion_success=success)
        self._write_episode_summary(episode_dir, {"task_id": task.id, "success": success, "mode": "lerobot"})
        self.get_logger().info(f"DataCollect complete. Success: {success}")
        return True

    def _write_episode_summary(self, episode_dir: Path, summary: dict) -> None:
        (episode_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _load_scenario_params(self, task) -> np.ndarray:
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
