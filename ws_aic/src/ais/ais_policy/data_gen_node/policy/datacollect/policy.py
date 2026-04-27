"""
DataCollect policy
──────────────────
AutoCapture 기반으로 에피소드 데이터를 LeRobot 포맷으로 직접 저장한다.
lerobot 패키지가 없으면 기존 raw 포맷(steps.jsonl)으로 fallback.

환경 변수:
  AIC_LEROBOT_REPO_ID      HuggingFace repo ID (예: aic-sejong/my-dataset)
  AIC_LEROBOT_OUT_DIR      로컬 저장 경로
  AIC_LEROBOT_RUN_ID       실행 구분자 (미지정 시 YYYYMMDD_HHMMSS 자동 생성)
  AIC_LEROBOT_FPS          fps (기본: 10)
  AIC_LEROBOT_PUSH_TO_HUB  "true" 이면 finalize 후 HF Hub 업로드

실행:
  pixi run ros2 run aic_model aic_model --ros-args \\
    -p policy:=data_gen_node.policy.datacollect
"""

import atexit
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from data_gen_node.policy.autocapture import AutoCapture
from data_gen_node.policy.autocapture.lib.recording import (
    AutoCaptureRecorder,
    LeRobotRecorder,
    LEROBOT_FEATURES,
    _LEROBOT_AVAILABLE,
)
from tf2_ros import TransformException

TARE_SERVICE = "/aic_controller/tare_force_torque_sensor"
TARE_TYPE    = "std_srvs/srv/Trigger"


class DataCollect(AutoCapture):
    """
    에피소드 시작 시 F/T 센서 영점 조절 후 LeRobot 포맷으로 직접 저장.
    AIC_LEROBOT_REPO_ID / AIC_LEROBOT_OUT_DIR 미설정 시 raw 포맷으로 fallback.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._lerobot_dataset = None
        self._lerobot_full_repo_id: Optional[str] = None
        self._init_lerobot_dataset()

        # 프로세스 종료 시 finalize (SIGTERM + 정상 종료 모두 대응)
        atexit.register(self._finalize_dataset)
        # signal.signal()은 main thread에서만 가능.
        # ROS2 라이프사이클 on_configure 콜백은 executor 스레드에서 실행되므로
        # non-main-thread일 경우 ValueError가 발생한다 — 조용히 건너뜀.
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:
            self.get_logger().warn(
                "[DataCollect] SIGTERM 핸들러 등록 불가 (non-main thread). "
                "atexit 핸들러로만 finalize 보장됨."
            )

    # ──────────────────────────────────────────
    # LeRobot dataset 초기화
    # ──────────────────────────────────────────
    def _init_lerobot_dataset(self) -> None:
        if not _LEROBOT_AVAILABLE:
            self.get_logger().warn(
                "[DataCollect] lerobot 패키지 없음 → raw 포맷으로 수집"
            )
            return

        repo_id = os.environ.get("AIC_LEROBOT_REPO_ID", "").strip()
        out_dir  = os.environ.get("AIC_LEROBOT_OUT_DIR", "").strip()
        if not repo_id or not out_dir:
            self.get_logger().warn(
                "[DataCollect] AIC_LEROBOT_REPO_ID 또는 AIC_LEROBOT_OUT_DIR 미설정 "
                "→ raw 포맷으로 수집"
            )
            return

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        run_id = (os.environ.get("AIC_LEROBOT_RUN_ID") or
                  datetime.now().strftime("%Y%m%d_%H%M%S"))
        # HF Hub 목적지: run_id 없이 고정 repo → 모든 실행의 에피소드가 누적됨
        self._lerobot_full_repo_id = repo_id
        fps = int(os.environ.get("AIC_LEROBOT_FPS", "10"))

        # 로컬 저장 경로: 고정 'master' 디렉터리에 모든 run의 에피소드를 누적.
        # → 재실행해도 에피소드 번호가 이어지고 push_to_hub 시 덮어쓰기 없음.
        dataset_root = Path(out_dir) / "master"

        if dataset_root.exists():
            # 같은 run 내 두 번째 이상 configure (trial 2~N) → 기존 dataset에 episode 추가
            self.get_logger().info(
                f"[DataCollect] 기존 LeRobot dataset 열기 (episode 추가): {dataset_root}"
            )
            self._lerobot_dataset = LeRobotDataset.resume(
                repo_id=self._lerobot_full_repo_id,
                root=dataset_root,
            )
        else:
            # 첫 번째 configure → 새 dataset 생성 (LeRobotDataset.create가 디렉터리도 생성)
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=self._lerobot_full_repo_id,
                root=dataset_root,
                fps=fps,
                features=LEROBOT_FEATURES,
                use_videos=True,
            )
        self.get_logger().info(
            f"[DataCollect] LeRobot dataset 초기화 완료. "
            f"local={dataset_root}  hub={self._lerobot_full_repo_id}"
        )



    def _finalize_dataset(self) -> None:
        if self._lerobot_dataset is None:
            return
        try:
            self._lerobot_dataset.finalize()
            self.get_logger().info("[DataCollect] LeRobot dataset finalize 완료.")
            if os.environ.get("AIC_LEROBOT_PUSH_TO_HUB", "true").lower() == "true":
                # push_to_hub 전 HF Hub에 repo가 없으면 자동 생성
                from huggingface_hub import HfApi
                HfApi().create_repo(
                    repo_id=self._lerobot_full_repo_id,
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,   # 이미 있으면 무시
                )
                self._lerobot_dataset.push_to_hub(private=True)

                self.get_logger().info(
                    f"[DataCollect] HF Hub 업로드 완료: {self._lerobot_full_repo_id}"
                )
        except Exception as e:
            self.get_logger().error(f"[DataCollect] finalize 실패: {e}")
        finally:
            self._lerobot_dataset = None

    def _on_sigterm(self, signum, frame) -> None:
        self._finalize_dataset()
        raise SystemExit(0)

    # ──────────────────────────────────────────
    # 에피소드 수집
    # ──────────────────────────────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        # 1. F/T 센서 Tare
        self._tare_sensor()

        # 2. 소프트웨어 Tare (Fz 보정)
        self.sleep_for(1.0)
        initial_obs = get_observation()
        fz_offset = 0.0
        if initial_obs and hasattr(initial_obs, "wrist_wrench"):
            fz_offset = initial_obs.wrist_wrench.wrench.force.z
            self.get_logger().info(
                f"[DataCollect] Fz offset captured: {fz_offset:.4f}"
            )

        def wrapped_get_observation() -> Observation:
            obs = get_observation()
            if obs and hasattr(obs, "wrist_wrench"):
                obs.wrist_wrench.wrench.force.z -= fz_offset
            return obs

        # 3. 에피소드 초기화
        self.get_logger().info(f"DataCollect.insert_cable() task: {task.id}")
        self._task = task
        self._latest_insertion_event = None
        self._planner.reset()
        send_feedback("data collect running")

        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        # 4. scenario params (task별 랜덤 파라미터 + gripper offset)
        scenario_params_vec = self._load_scenario_params(task)
        ground_truth_gripper_offset = {
            "x": float(scenario_params_vec[5]),
            "y": float(scenario_params_vec[6]),
            "z": float(scenario_params_vec[7]),
        }

        # 5. recorder 선택
        if self._lerobot_dataset is not None:
            recorder = LeRobotRecorder(self._lerobot_dataset, scenario_params_vec)
        else:
            recorder = AutoCaptureRecorder(episode_dir)

        # 6. port / plug TF 프레임 정의
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, plug_frame]:
            if not self._wait_for_tf("base_link", frame):
                self._write_episode_summary(episode_dir, {
                    "task_id": task.id,
                    "status": "setup_failed",
                    "missing_frame": frame,
                })
                return False

        # 7. TF 기반 gripper offset 측정 (분석용)
        try:
            port_transform = self._lookup_transform("base_link", port_frame)
            init_plug_tf   = self._lookup_transform("base_link", plug_frame)
            init_gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            initial_gripper_offset = {
                "x": init_gripper_tf.translation.x - init_plug_tf.translation.x,
                "y": init_gripper_tf.translation.y - init_plug_tf.translation.y,
                "z": init_gripper_tf.translation.z - init_plug_tf.translation.z,
            }
        except Exception as ex:
            self.get_logger().error(f"Initial setup failed: {ex}")
            return False

        # raw recorder에만 meta.json 기록
        if isinstance(recorder, AutoCaptureRecorder):
            recorder.write_meta({
                "task": recorder.task_to_dict(task),
                "selected_port_frame": port_frame,
                "plug_frame": plug_frame,
                "approach_z_offset": self.approach_z_offset,
                "approach_steps": self.approach_steps,
                "insert_z_step": self.insert_z_step,
                "insert_min_z_offset": self.insert_min_z_offset,
                "stabilize_sec": self.stabilize_sec,
                "i_gain": self._planner.i_gain,
                "initial_gripper_offset": initial_gripper_offset,
                "ground_truth_gripper_offset": ground_truth_gripper_offset,
                "fz_software_tare_offset": fz_offset,
            })

        start_time = time.time()
        phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}

        # 포트가 처음 보이는 순간부터 레코딩 시작 (phase 무관).
        # YOLO 없으면 즉시 시작, 있으면 trigger 임계값(AIC_YOLO_TRIGGER_CONF, 기본 0.1)으로 감지.
        recording_started = (self._yolo_model is None)

        def _check_and_start(obs) -> None:
            nonlocal recording_started
            if not recording_started and self._detect_plugs(obs, conf=self._yolo_trigger_conf):
                recording_started = True
                self.get_logger().info("[DataCollect] 포트 첫 검출 → 레코딩 시작")

        # 8. Approach
        z_offset = self.approach_z_offset
        for t in range(self.approach_steps):
            interp = t / float(self.approach_steps)
            try:
                plug_tf    = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                pose, extras = self._planner.build_pose(
                    port_transform=port_transform,
                    plug_transform=plug_tf,
                    gripper_transform=gripper_tf,
                    slerp_fraction=interp,
                    position_fraction=interp,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                obs = wrapped_get_observation()
                _check_and_start(obs)
                if recording_started:
                    self._record_motion_step(
                        recorder, "approach", task, port_transform,
                        plug_tf, gripper_tf, obs, pose, extras,
                    )
                    phase_step_counts["approach"] += 1
            except TransformException:
                pass
            self.sleep_for(self.step_sleep_sec)

        # 9. Insert
        while z_offset >= self.insert_min_z_offset:
            z_offset -= self.insert_z_step
            try:
                plug_tf    = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                pose, extras = self._planner.build_pose(
                    port_transform=port_transform,
                    plug_transform=plug_tf,
                    gripper_transform=gripper_tf,
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                obs = wrapped_get_observation()
                _check_and_start(obs)
                if recording_started:
                    self._record_motion_step(
                        recorder, "insert", task, port_transform,
                        plug_tf, gripper_tf, obs, pose, extras,
                    )
                    phase_step_counts["insert"] += 1
            except TransformException:
                pass
            self.sleep_for(self.step_sleep_sec)

        # 10. Stabilize
        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(self.stabilize_sec)

        try:
            plug_tf    = self._lookup_transform("base_link", plug_frame)
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            obs = wrapped_get_observation()
            if obs is not None and recording_started:
                recorder.record_terminal_step(
                    phase="stabilize", task=task, obs=obs,
                    port_tf=port_transform, plug_tf=plug_tf,
                    gripper_tf=gripper_tf, extras={"z_offset": z_offset},
                )
                phase_step_counts["stabilize"] = 1
        except TransformException:
            pass

        if not recording_started:
            self.get_logger().warn(
                "[DataCollect] 에피소드 전체에서 포트 미검출 → skip"
            )
            self._write_episode_summary(episode_dir, {
                "task_id": task.id,
                "status": "detection_timeout",
                "failure_reason": "port_not_detected_during_episode",
            })
            return False

        # 11. 에피소드 저장
        insertion_event_observed = self._has_successful_insertion(task)

        if isinstance(recorder, LeRobotRecorder):
            recorder.save_episode()
        else:
            recorder.write_summary({
                "task": recorder.task_to_dict(task),
                "status": "completed",
                "selected_port_frame": port_frame,
                "plug_frame": plug_frame,
                "elapsed_sec": time.time() - start_time,
                "insertion_event_observed": insertion_event_observed,
                "phase_step_counts": phase_step_counts,
                "initial_gripper_offset": initial_gripper_offset,
                "ground_truth_gripper_offset": ground_truth_gripper_offset,
                "fz_software_tare_offset": fz_offset,
            })

        # collect_data_aarch.py 에피소드 감지용 summary (LeRobot 모드에서도 기록)
        self._write_episode_summary(episode_dir, {
            "task_id": task.id,
            "status": "completed",
            "elapsed_sec": time.time() - start_time,
            "insertion_event_observed": insertion_event_observed,
            "phase_step_counts": phase_step_counts,
            "mode": "lerobot" if isinstance(recorder, LeRobotRecorder) else "raw",
        })

        self.get_logger().info(
            f"DataCollect complete. Event: {insertion_event_observed}"
        )
        send_feedback("data collect complete")
        return True

    @staticmethod
    def _write_episode_summary(episode_dir: Path, summary: dict) -> None:
        (episode_dir / "episode_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def _load_scenario_params(self, task) -> np.ndarray:
        """
        /tmp/aic_scenario_params.json 에서 task.id에 해당하는 랜덤 파라미터를 로드.
        파일 없거나 key 없으면 0 벡터 반환.

        벡터 레이아웃 (11차원):
          [0] trial_type  [1] rail_idx
          [2] board_x  [3] board_y  [4] board_yaw
          [5] gripper_offset_x  [6] gripper_offset_y  [7] gripper_offset_z
          [8] nic_translation  [9] nic_yaw  [10] sc_translation
        """
        params_file = Path(
            os.environ.get("AIC_SCENARIO_PARAMS_FILE", "/tmp/aic_scenario_params.json")
        )
        zero = np.zeros(11, dtype=np.float32)
        if not params_file.exists():
            self.get_logger().warn(
                f"[DataCollect] scenario_params 파일 없음({params_file}) → 0 벡터 사용"
            )
            return zero
        try:
            all_params = json.loads(params_file.read_text(encoding="utf-8"))
            p = all_params.get(task.id)
            if p is None:
                self.get_logger().warn(
                    f"[DataCollect] task.id={task.id} 파라미터 없음 → 0 벡터 사용"
                )
                return zero
            return np.array([
                p["trial_type"],       p["rail_idx"],
                p["board_x"],          p["board_y"],         p["board_yaw"],
                p["gripper_offset_x"], p["gripper_offset_y"], p["gripper_offset_z"],
                p["nic_translation"],  p["nic_yaw"],          p["sc_translation"],
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"[DataCollect] scenario_params 로드 실패: {e}")
            return zero

    def _tare_sensor(self) -> None:
        self.get_logger().info(
            f"[DataCollect] System Tare 서비스 호출: {TARE_SERVICE}"
        )
        try:
            result = subprocess.run(
                ["ros2", "service", "call", TARE_SERVICE, TARE_TYPE],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                self.get_logger().info("[DataCollect] System Tare 완료.")
            else:
                self.get_logger().warn(
                    f"[DataCollect] System Tare 실패 (code={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().error("[DataCollect] System Tare timeout (10s)")
        except Exception as e:
            self.get_logger().error(f"[DataCollect] System Tare 예외: {e}")
