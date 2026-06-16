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


class LeRobot(Policy):
    """
    Collect basic LeRobot-format insertion episodes.
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
        self._yolo_model_path = str(Path(__file__).resolve().parents[6] / "model" / "ais_yolo" / "weights" / "best.pt")
        self._lerobot_dataset = None
        self._lerobot_full_repo_id = os.environ.get("AIC_LEROBOT_REPO_ID", "").strip()
        self._lerobot_version = os.environ.get("AIC_LEROBOT_VERSION", "master").strip()
        self._yolo_trigger_conf = float(os.environ.get("AIC_YOLO_TRIGGER_CONF", "0.7"))
        
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

        self.get_logger().info(f"[LeRobot] Unified Policy Initialized. Root: {self.capture_root}")

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
            self.get_logger().info(f"[LeRobot] Using port entrance frame: {entrance_frame}")
            return entrance_frame
        self.get_logger().warn(f"[LeRobot] Port entrance TF unavailable, falling back to: {port_frame}")
        return port_frame

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

    # ── 메인 에피소드 수집 로직 ───────────────────────────────────────────────

    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self._task = task; self._latest_insertion_event = None; self._planner.reset(); send_feedback("data collect running")
        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name; episode_dir.mkdir(parents=True, exist_ok=True)
        scenario_params_vec = self._load_scenario_params(task)
        recorder = None
        if self._lerobot_dataset is not None:
            recorder = LeRobotRecorder(self._lerobot_dataset, scenario_params_vec)
        else:
            self.get_logger().info("[LeRobot] LeRobot dataset disabled; running motion without episode recording.")
        
        port_frame, plug_frame = self._select_port_frame(task), f"{task.cable_name}/{task.plug_name}_link"
        if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf("base_link", plug_frame): return False

        start_time = time.time(); phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}
        recording_started = (self._yolo_model is None)
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

        def _check_and_start(obs) -> None:
            nonlocal recording_started
            if recording_started or obs is None: return
            img_msg = obs.center_image
            if img_msg.width == 0: return
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            bgr = img if img_msg.encoding != "rgb8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = self._yolo_model(bgr, verbose=False, conf=self._yolo_trigger_conf)
            best_detection = None
            for result in results:
                names = getattr(result, "names", {})
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = names.get(class_id, "").lower()
                    if _port_kw not in class_name:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                    detection = (confidence, class_name, x1, y1, x2, y2)
                    if best_detection is None or confidence > best_detection[0]:
                        best_detection = detection
            if best_detection is not None:
                confidence, class_name, x1, y1, x2, y2 = best_detection
                recording_started = True
                self.get_logger().info(
                    "[LeRobot] YOLO DETECTION -> Recording Started "
                    f"class={class_name} conf={confidence:.3f} "
                    f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                    f"center=({0.5 * (x1 + x2):.1f},{0.5 * (y1 + y2):.1f})"
                )

        # Phase 1-A: Alignment at High Z (Dynamic Tracking + Early Lock-on)
        self.get_logger().info(f"━━━ Phase 1-A: Alignment at {current_approach_z*100:.0f}cm ━━━")
        for t in range(self.approach_steps):
            t_norm = (t + 1) / float(self.approach_steps); t_smooth = interp_profile(t_norm, quintic=True)
            try:
                current_port_tf = self._lookup_transform("base_link", port_frame)
                plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
                
                # 조기 락온: 마지막 2초(60스텝 이후) 적분기 가동
                should_reset = (t < 60)

                pose, extras = self._planner.build_pose(
                    port_transform=current_port_tf, plug_transform=plug_tf, gripper_transform=gripper_tf, 
                    slerp_fraction=t_smooth, position_fraction=t_smooth, z_offset=current_approach_z,
                    reset_xy_integrator=should_reset
                )

                # 텔레메트리 로깅
                self.get_logger().info(f"pfrac: {extras['position_fraction']:.3} xy_error: {extras['tip_x_error']:0.3} {extras['tip_y_error']:0.3} ints: {extras['tip_x_error_integrator']:.3} , {extras['tip_y_error_integrator']:.3}")

                self.set_pose_target(move_robot, pose, stiffness=approach_stiffness, damping=approach_damping)
                obs = get_observation(); _check_and_start(obs)
                if recorder is not None and recording_started:
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
                obs = get_observation(); _check_and_start(obs)
                if recorder is not None and recording_started:
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
                obs = get_observation(); _check_and_start(obs)
                if recorder is not None and recording_started:
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
            if recorder is not None and obs and recording_started:
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
        if recorder is not None:
            recorder.save_episode(insertion_success=success)
        self._write_episode_summary(episode_dir, {"task_id": task.id, "success": success, "mode": "lerobot" if recorder is not None else "yolo_only"})
        self.get_logger().info(f"LeRobot complete. Success: {success}")
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
