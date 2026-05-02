"""
DataCollect policy (Unified)
──────────────────────────
모든 데이터 수집 로직(AutoCapture + LeRobot)이 통합된 단일 정책 파일.
StagedPolicy 기반의 안정적인 모션 플래닝과 LeRobot 포맷 저장을 수행한다.

실행:
  pixi run ros2 run aic_model aic_model --ros-args \
    -p policy:=data_gen_node.DataCollect
"""

import atexit
import json
import os
import shutil
import signal
import subprocess
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
    """5차 Hermite: 시작/끝에서 속도 + 가속도 모두 0."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    return s_curve_quintic(t) if quintic else (3.0 * t**2 - 2.0 * t**3)


class DataCollect(Policy):
    """
    통합 데이터 수집 정책 클래스.
    - Staged Approach (Far/Mid/Insert)
    - LeRobot Dataset Recording
    - YOLO Port Detection Trigger
    - Gaussian Offset Diversification
    """

    # ── 기본 경로 및 설정 ──────────────────────────────────────────────────
    _YOLO_MODEL_DEFAULT = str(
        Path(__file__).resolve().parents[5] / "src" / "model" / "ais_yolo-2" / "weights" / "best.pt"
    )
    _CAPTURE_DIR_DEFAULT = "/tmp/aic_episodes"
    _DEFAULT_STEP_HZ: float = 10.0

    # ── 임피던스 설정 (StagedPolicy 기준) ──────────────────────────────────
    # 기본(Approach) 단계
    _STIFFNESS_DEFAULT = [200.0, 200.0, 150.0, 50.0, 50.0, 50.0]
    _DAMPING_DEFAULT   = [80.0, 80.0, 80.0, 20.0, 20.0, 20.0]

    # SFP 커넥터 전용 삽입 파라미터
    _SFP_INSERT_STIFFNESS = [15.0, 15.0, 400.0, 15.0, 15.0, 50.0]
    _SFP_INSERT_DAMPING   = [30.0, 30.0, 100.0, 10.0, 10.0, 20.0]

    # SC 커넥터 전용 삽입 파라미터
    _SC_INSERT_STIFFNESS = [20.0, 20.0, 500.0, 20.0, 20.0, 50.0]
    _SC_INSERT_DAMPING   = [30.0, 30.0, 100.0, 10.0, 10.0, 20.0]

    # ── 모션 플래닝 상수 ──────────────────────────────────────────────────
    _SFP_STAGE1A_Z_OFFSET: float = 0.14       # 14cm (Far approach)
    _SC_STAGE1A_Z_OFFSET: float = 0.008       # 0.8cm (Far approach)

    # SC 케이블 전용 삽입 하한: 스프링 클립 ±45° 클릭 포인트 통과에 필요한 추가 5mm
    _SC_INSERT_MIN_Z_OFFSET: float = -0.020  # -20mm (SFP 기본 -15mm 대비 5mm 추가 하강) 
    _STAGE1B_CONVERGENCE_TOL_M: float = 0.005
    _STAGE1B_STABLE_CONSECUTIVE: int = 3
    _STAGE1B_CONVERGENCE_MAX_WAIT_S: float = 2.0

    WIGGLE_RAD = 0.0015  # 1.5mm 반지름
    WIGGLE_FREQ = 3.0    # 3Hz (초당 3바퀴)
    STUCK_THRESHOLD = 0.005 # 5mm 이상 차이나면 박힌 것으로 간주

    def __init__(self, parent_node):
        super().__init__(parent_node)
        
        # 1. 상태 및 제어 변수 초기화
        self._task: Optional[Task] = None
        self._latest_insertion_event: Optional[str] = None
        self._max_integrator_windup = 0.05
        
        # 2. 제어 주기 및 경로 설정
        fps = int(os.environ.get("AIC_LEROBOT_FPS", "0"))
        self.step_sleep_sec = 1.0 / (fps if fps > 0 else self._DEFAULT_STEP_HZ)
        self.capture_root = Path(os.environ.get("AIC_CAPTURE_DIR", self._CAPTURE_DIR_DEFAULT))
        
        # 3. 플래너 및 환경 설정
        self._planner = CheatCodePlanner(
            i_gain=float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15")),
            max_integrator_windup=self._max_integrator_windup,
        )
        self.approach_z_offset = float(os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_Z_OFFSET", "0.2"))
        self.approach_steps = int(os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_STEPS", "100"))
        self.insert_z_step = float(os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_Z_STEP", "0.0005"))
        self.insert_min_z_offset = float(os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_MIN_Z_OFFSET", "-0.015"))
        self.stabilize_sec = float(os.environ.get("AIC_CAPTURE_CHEATCODE_STABILIZE_SEC", "5.0"))
        self.sc_insert_min_z_offset = float(os.environ.get("AIC_CAPTURE_SC_INSERT_MIN_Z_OFFSET", str(self._SC_INSERT_MIN_Z_OFFSET)))

        # 4. YOLO 및 데이터셋 설정
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

        # 6. 종료 처리 (3중 안전장치)
        atexit.register(self._finalize_dataset)
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:
            pass

        self._stop_file = Path(os.environ.get("AIC_STOP_FILE", "/tmp/aic_policy_stop"))
        self._stop_file.unlink(missing_ok=True)
        threading.Thread(target=self._watch_stop_file, daemon=True).start()

        self.get_logger().info(f"[DataCollect] Unified Policy Initialized. Root: {self.capture_root}")

    # ── 인프라 로직 (YOLO, TF, Dataset, Events) ───────────────────────────────

    def _init_yolo(self) -> None:
        model_path = Path(os.environ.get("AIC_YOLO_MODEL_PATH", self._YOLO_MODEL_DEFAULT))
        self._yolo_model = None
        if not model_path.exists():
            self.get_logger().warn(f"[DataCollect] YOLO 모델 없음: {model_path}")
            return
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(model_path))
            self.get_logger().info(f"[DataCollect] YOLO 로드 완료: {model_path}")
        except Exception as e:
            self.get_logger().error(f"[DataCollect] YOLO 로드 실패: {e}")

    @staticmethod
    def _is_valid_dataset_root(root: Path) -> bool:
        """tasks.parquet까지 존재해야 유효한 dataset으로 간주."""
        return (root / "meta" / "info.json").exists() and \
               (root / "meta" / "tasks.parquet").exists()

    def _init_lerobot_dataset(self) -> None:
        if not _LEROBOT_AVAILABLE or not self._lerobot_full_repo_id:
            self.get_logger().warn(f"[DataCollect] LeRobot 설정 미비 (Available: {_LEROBOT_AVAILABLE}, RepoID: '{self._lerobot_full_repo_id}') → 수집 불가")
            return

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_root = Path(os.environ.get("AIC_LEROBOT_OUT_DIR", "lerobot")) / self._lerobot_version
            fps = int(os.environ.get("AIC_LEROBOT_FPS", "10"))

            self.get_logger().info(f"[DataCollect] Initializing LeRobotDataset: repo_id={self._lerobot_full_repo_id}, root={dataset_root}")

            if self._is_valid_dataset_root(dataset_root):
                self._lerobot_dataset = LeRobotDataset.resume(repo_id=self._lerobot_full_repo_id, root=dataset_root)
            else:
                if dataset_root.exists():
                    self.get_logger().warn(f"[DataCollect] 불완전한 dataset 감지 ({dataset_root}) → 삭제 후 재생성")
                    shutil.rmtree(dataset_root)
                self._lerobot_dataset = LeRobotDataset.create(
                    repo_id=self._lerobot_full_repo_id, root=dataset_root, fps=fps,
                    features=LEROBOT_FEATURES, use_videos=True,
                )
            self.get_logger().info(f"[DataCollect] Dataset Ready: {dataset_root}")
        except Exception as e:
            self.get_logger().error(f"[DataCollect] LeRobot Dataset initialization failed: {e}")
            self.get_logger().error("Tip: Check if repo_id is valid (user/repo) and you are logged into Hugging Face (huggingface-cli login).")
            self._lerobot_dataset = None

    def _finalize_dataset(self) -> None:
        if self._lerobot_dataset is not None:
            try:
                self._lerobot_dataset.finalize()
                self.get_logger().info("[DataCollect] Dataset finalized.")
            except Exception as e:
                self.get_logger().error(f"Finalize failed: {e}")
            finally:
                self._lerobot_dataset = None

    def _watch_stop_file(self) -> None:
        while True:
            if self._stop_file.exists():
                self._finalize_dataset()
                os._exit(0)  # sys.exit(0) → daemon thread에서는 프로세스가 안 죽음
            time.sleep(0.5)

    def _on_sigterm(self, signum, frame) -> None:
        self._finalize_dataset()
        raise SystemExit(0)

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event = msg.data.strip().strip("/")

    def _has_successful_insertion(self, task: Task) -> bool:
        if not self._latest_insertion_event: return False
        tokens = [t for t in self._latest_insertion_event.split("/") if t]
        return len(tokens) >= 2 and tokens[0] == task.target_module_name and tokens[1] == task.port_name

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                self.sleep_for(0.1)
        return False

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform

    def set_pose_target(self, move_robot, pose, frame_id="base_link", stiffness=None, damping=None, feedforward_wrench=None):
        _stiffness = stiffness if stiffness is not None else self._STIFFNESS_DEFAULT
        _damping   = damping   if damping   is not None else self._DAMPING_DEFAULT
        _wrench    = feedforward_wrench if feedforward_wrench is not None else Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(_stiffness).flatten(),
            target_damping=np.diag(_damping).flatten(),
            feedforward_wrench_at_tip=_wrench,
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    def _motion_update_from_pose(self, pose, stiffness=None, damping=None) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header.frame_id = "base_link"
        mu.header.stamp = self._parent_node.get_clock().now().to_msg()
        mu.pose = pose
        _s = stiffness if stiffness is not None else self._STIFFNESS_DEFAULT
        _d = damping if damping is not None else self._DAMPING_DEFAULT
        mu.target_stiffness = list(np.diag(_s).flatten())
        mu.target_damping = list(np.diag(_d).flatten())
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

    def insert_cable(
        self, task: Task, get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"DataCollect.insert_cable() task: {task.id}")
        self._task = task
        self._latest_insertion_event = None
        self._planner.reset()
        send_feedback("data collect running")

        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        scenario_params_vec = self._load_scenario_params(task)

        if self._lerobot_dataset is None:
            self.get_logger().error("[DataCollect] Dataset not initialized.")
            return False
            
        recorder = LeRobotRecorder(self._lerobot_dataset, scenario_params_vec)

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf("base_link", plug_frame):
            return False

        try:
            port_transform = self._lookup_transform("base_link", port_frame)
        except Exception as ex:
            self.get_logger().error(f"TF Lookup failed: {ex}")
            return False

        start_time = time.time()
        phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}
        
        recording_started = (self._yolo_model is None)
        _port_kw = "sfp" if "sfp" in task.port_type.lower() else "sc"

        insert_stiffness = self._STIFFNESS_DEFAULT
        insert_damping = self._DAMPING_DEFAULT

        def _check_and_start(obs) -> None:
            nonlocal recording_started
            if recording_started or obs is None: return
            img_msg = obs.center_image
            if img_msg.width == 0: return
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            bgr = img if img_msg.encoding != "rgb8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = self._yolo_model(bgr, verbose=False, conf=self._yolo_trigger_conf)
            if any(_port_kw in r.names.get(int(box.cls[0]), "").lower() for r in results for box in r.boxes):
                recording_started = True
                self.get_logger().info("[DataCollect] YOLO Detected Port -> Recording Started")
                try:
                    self._debug_image_dir.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    annotated = results[0].plot()
                    cv2.imwrite(str(self._debug_image_dir / f"yolo_trigger_{ts}.jpg"), annotated)
                except Exception as e:
                    self.get_logger().warn(f"[DataCollect] debug 이미지 저장 실패: {e}")

        # 9. Phase 1-A: Far Approach
        self.get_logger().info(f"━━━ Phase 1-A: Far Approach ({self.approach_steps // 2} steps) ━━━")
        z_offset = self.approach_z_offset

        if _port_kw == "sfp":
            target_z_1a = self._SFP_STAGE1A_Z_OFFSET
        elif _port_kw == "sc":
            target_z_1a = self._SC_STAGE1A_Z_OFFSET
        
        for t in range(self.approach_steps // 2):
            t_norm = (t + 1) / float(self.approach_steps // 2)
            t_smooth = interp_profile(t_norm, quintic=True)
            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                cur_z_offset = z_offset * (1.0 - t_smooth) + target_z_1a * t_smooth
                port_tf = Transform(rotation=port_transform.rotation)

                pose, extras = self._planner.build_pose(
                    port_transform=port_tf, plug_transform=plug_tf,
                    gripper_transform=gripper_tf, slerp_fraction=t_smooth,
                    position_fraction=t_smooth, z_offset=cur_z_offset,
                    reset_xy_integrator=(t == 0)
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                obs = get_observation()
                _check_and_start(obs)
                if recording_started:
                    self._record_motion_step(recorder, "approach", task, port_transform, plug_tf, gripper_tf, obs, pose, extras, stiffness=self._STIFFNESS_DEFAULT, damping=self._DAMPING_DEFAULT)
                    phase_step_counts["approach"] += 1
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        # 11. Convergence Wait
        self.get_logger().info(f"━━━ Phase 1-C: CONVERGENCE WAIT ━━━")
        wait_end = time.time() + self._STAGE1B_CONVERGENCE_MAX_WAIT_S
        stable_count = 0
        while time.time() < wait_end:
            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
                desired_plug_z = port_transform.translation.z + target_z_1a
                if abs(plug_tf.translation.z - desired_plug_z) < self._STAGE1B_CONVERGENCE_TOL_M:
                    stable_count += 1
                    if stable_count >= self._STAGE1B_STABLE_CONSECUTIVE: break
                else: stable_count = 0
            except TransformException: pass
            self.sleep_for(0.05)

        # 포트 타입에 따른 삽입 파라미터 선택
        if _port_kw == "sfp":
            insert_stiffness = self._SFP_INSERT_STIFFNESS
            insert_damping = self._SFP_INSERT_DAMPING
        else:
            insert_stiffness = self._SC_INSERT_STIFFNESS
            insert_damping = self._SC_INSERT_DAMPING

        # 12. Final stabilization & Insert
        self.sleep_for(1.0)
        self.get_logger().info("━━━ Phase 3: Insert started (Compliance ON) ━━━")
        # SC 케이블은 스프링 클립 클릭 포인트 통과를 위해 SFP보다 더 깊이 삽입
        effective_insert_min_z_offset = self.sc_insert_min_z_offset if _port_kw == "sc" else self.insert_min_z_offset
        self.get_logger().info(f"[DataCollect] insert_min_z_offset: {effective_insert_min_z_offset*1000:.1f}mm ({_port_kw})")
        z_offset = target_z_1a
        x_offset_noise, y_offset_noise = 0, 0

        while z_offset >= effective_insert_min_z_offset:
            if self._has_successful_insertion(task):
                self.get_logger().info(f"✨ [DataCollect] Insertion SUCCESS detected via callback! (Event:{self._latest_insertion_event})")
                break # 즉시 삽입 루프 탈출
            try:
                plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
                if _port_kw == "sc":
                    # 명령한 위치(port_z + z_offset)와 실제 위치(plug_tf.z) 비교
                    commanded_z = port_transform.translation.z + z_offset
                    if (plug_tf.translation.z - commanded_z) > self.STUCK_THRESHOLD:
                        self.get_logger().info("⚠️ [SC] Stuck detected! Lifting up for  retry...")
                        z_offset += 0.010  # 1cm 위로 급하게 들어올림
                        # 들어올릴 때 XY에 미세한 랜덤 오프셋을 주어 정렬을 다시 시도
                        x_offset_noise += np.random.uniform(-0.001, 0.001)
                        y_offset_noise += np.random.uniform(-0.001, 0.001)
                        self.sleep_for(0.2) # 잠시 대기 후 다시 하강 시작
                        continue

                target_x = port_transform.translation.x + x_offset_noise
                target_y = port_transform.translation.y + y_offset_noise

                if _port_kw == "sfp":
                    # 시간에 따른 원형 궤적 계산
                    t_phase = time.time() * self.WIGGLE_FREQ * 2.0 * np.pi
                    target_x += self.WIGGLE_RAD * np.cos(t_phase)
                    target_y += self.WIGGLE_RAD * np.sin(t_phase)

                noisy_port_tf = Transform(rotation=port_transform.rotation)
                noisy_port_tf.translation.x = target_x
                noisy_port_tf.translation.y = target_y
                noisy_port_tf.translation.z = port_transform.translation.z

                pose, extras = self._planner.build_pose(port_transform=noisy_port_tf, plug_transform=plug_tf, gripper_transform=gripper_tf, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=pose, stiffness=insert_stiffness, damping=insert_damping)
                obs = get_observation()
                _check_and_start(obs)
                if recording_started:
                    self._record_motion_step(recorder, "insert", task, port_transform, plug_tf, gripper_tf, obs, pose, extras, stiffness=insert_stiffness, damping=insert_damping)
                    phase_step_counts["insert"] += 1

                z_offset -= self.insert_z_step
            except TransformException: pass
            self.sleep_for(self.step_sleep_sec)

        # 13. Phase 4: Stabilize
        if self._has_successful_insertion(task):
            self.get_logger().info("[DataCollect] Fast exit triggered.")
            self.sleep_for(0.5) # 성공 시엔 0.5초만 안정화
        else:
            self.get_logger().warn("[DataCollect] Reached min Z without success event. Waiting full stabilize time.")
            self.sleep_for(self.stabilize_sec) # 실패 시엔 기존대로 대기
        try:
            plug_tf, gripper_tf = self._lookup_transform("base_link", plug_frame), self._lookup_transform("base_link", "gripper/tcp")
            obs = get_observation()
            if obs and recording_started:
                recorder.record_terminal_step(
                    phase="stabilize", task=task, obs=obs, port_tf=port_transform,
                    plug_tf=plug_tf, gripper_tf=gripper_tf, extras={"z_offset": z_offset},
                    stiffness=insert_stiffness, damping=insert_damping
                )
                if "stabilize" not in phase_step_counts: phase_step_counts["stabilize"] = 0
                phase_step_counts["stabilize"] += 1
        except TransformException: pass

        # 14. Finalize episode
        success = self._has_successful_insertion(task)
        recorder.save_episode(insertion_success=success)
        self._write_episode_summary(episode_dir, {
            "task_id": task.id, "status": "completed", "elapsed_sec": time.time() - start_time,
            "insertion_event_observed": success, "phase_step_counts": phase_step_counts, "mode": "lerobot"
        })
        self.get_logger().info(f"DataCollect complete. Success: {success}")
        return True

    @staticmethod
    def _write_episode_summary(episode_dir: Path, summary: dict) -> None:
        (episode_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _load_scenario_params(self, task) -> np.ndarray:
        zero = np.zeros(11, dtype=np.float32)
        if not self._scenario_params_file.exists(): return zero
        try:
            p = json.loads(self._scenario_params_file.read_text(encoding="utf-8")).get(task.id)
            if not p: return zero
            return np.array([p["trial_type"], p["rail_idx"], p["board_x"], p["board_y"], p["board_yaw"],
                             p["gripper_offset_x"], p["gripper_offset_y"], p["gripper_offset_z"],
                             p["nic_translation"], p["nic_yaw"], p["sc_translation"]], dtype=np.float32)
        except Exception: return zero
