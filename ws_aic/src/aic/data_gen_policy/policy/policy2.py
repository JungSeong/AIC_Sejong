"""
DataCollect policy
──────────────────
AutoCapture + F/T Sensor Tare (System/Software) + Detailed Gripper Offset Recording.
(No intentional perturbation, pure CheatCode alignment)

실행:
  pixi run ros2 run aic_model aic_model --ros-args \
    -p policy:=data_gen_policy.policy.policy2
"""

import os
import time
import subprocess
from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from tf2_ros import TransformException

from data_gen_policy.policy.autocapture import AutoCapture
from data_gen_policy.policy.autocapture.lib.recording import AutoCaptureRecorder

# F/T 센서 Tare 서비스 정보
TARE_SERVICE = "/aic_controller/tare_force_torque_sensor"
TARE_TYPE    = "std_srvs/srv/Trigger"


class DataCollect(AutoCapture):
    """
    AutoCapture를 기반으로 하며, 매 에피소드 시작 시 센서 영점 조절 및
    분석을 위한 상세 파지 데이터를 기록하는 정책입니다.
    """

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        # 1. 시스템 레벨 하드웨어 Tare 시도
        self._tare_sensor()
        
        # 2. 코드 레벨 소프트웨어 Tare (Fz 보정)
        # 센서 안정화를 위해 잠시 대기 후 현재 Fz 값을 오프셋으로 캡처
        self.sleep_for(1.0)
        initial_obs = get_observation()
        fz_offset = 0.0
        if initial_obs and hasattr(initial_obs, 'wrist_wrench'):
            fz_offset = initial_obs.wrist_wrench.wrench.force.z
            self.get_logger().info(f"[DataCollect] Initial Fz offset captured: {fz_offset:.4f}")

        # 모든 관측 데이터에서 초기 오프셋을 차감하는 래핑 함수 정의
        def wrapped_get_observation() -> Observation:
            obs = get_observation()
            if obs and hasattr(obs, 'wrist_wrench'):
                obs.wrist_wrench.wrench.force.z -= fz_offset
            return obs

        # 3. 에피소드 초기화 및 리코더 설정
        self.get_logger().info(f"DataCollect.insert_cable() task: {task.id}")
        self._task = task
        self._latest_insertion_event = None
        self._planner.reset()
        send_feedback("data collect running")

        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        recorder = AutoCaptureRecorder(episode_dir)

        # 포트 및 플러그 프레임 정의
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        # TF 대기
        for frame in [port_frame, plug_frame]:
            if not self._wait_for_tf("base_link", frame):
                recorder.write_summary({
                    "task": recorder.task_to_dict(task),
                    "status": "setup_failed",
                    "missing_frame": frame,
                })
                return False

        # 4. 분석용 파지 오프셋(Gripper Offset) 계산
        try:
            port_transform = self._lookup_transform("base_link", port_frame)
            init_plug_tf = self._lookup_transform("base_link", plug_frame)
            init_gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            
            # TF 기반 측정된 파지 편차
            initial_gripper_offset = {
                "x": init_gripper_tf.translation.x - init_plug_tf.translation.x,
                "y": init_gripper_tf.translation.y - init_plug_tf.translation.y,
                "z": init_gripper_tf.translation.z - init_plug_tf.translation.z,
            }
        except Exception as ex:
            self.get_logger().error(f"Initial setup failed: {ex}")
            return False

        # 시나리오 생성 시 주입된 그라운드 트루스(GT) 오프셋 (환경변수에서 획득)
        gt_offset_x = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_X", "0.0"))
        gt_offset_y = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_Y", "0.0"))
        gt_offset_z = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_Z", "0.0"))
        ground_truth_gripper_offset = {"x": gt_offset_x, "y": gt_offset_y, "z": gt_offset_z}

        # 메타데이터 기록
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

        # 5. 동작 페이즈 1: Approach (접근)
        z_offset = self.approach_z_offset
        for t in range(self.approach_steps):
            interp = t / float(self.approach_steps)
            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
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
                # 래핑된 관측값(wrapped_get_observation)으로 기록
                self._record_motion_step(
                    recorder, "approach", task, port_transform, 
                    plug_tf, gripper_tf, wrapped_get_observation(), pose, extras
                )
                phase_step_counts["approach"] += 1
            except TransformException:
                pass
            self.sleep_for(self.step_sleep_sec)

        # 6. 동작 페이즈 2: Insert (삽입)
        while z_offset >= self.insert_min_z_offset:
            z_offset -= self.insert_z_step
            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                pose, extras = self._planner.build_pose(
                    port_transform=port_transform,
                    plug_transform=plug_tf,
                    gripper_transform=gripper_tf,
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                self._record_motion_step(
                    recorder, "insert", task, port_transform, 
                    plug_tf, gripper_tf, wrapped_get_observation(), pose, extras
                )
                phase_step_counts["insert"] += 1
            except TransformException:
                pass
            self.sleep_for(self.step_sleep_sec)

        # 7. 동작 페이즈 3: Stabilize (안정화)
        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(self.stabilize_sec)

        try:
            plug_tf = self._lookup_transform("base_link", plug_frame)
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            obs = wrapped_get_observation()
            if obs is not None:
                recorder.record_terminal_step(
                    phase="stabilize", task=task, obs=obs, 
                    port_tf=port_transform, plug_tf=plug_tf, 
                    gripper_tf=gripper_tf, extras={"z_offset": z_offset}
                )
                phase_step_counts["stabilize"] = 1
        except TransformException:
            pass

        # 8. 에피소드 종료 및 요약 기록
        insertion_event_observed = self._has_successful_insertion(task)
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

        self.get_logger().info(f"DataCollect complete. Event: {insertion_event_observed}")
        send_feedback("data collect complete")
        return True

    def _tare_sensor(self) -> None:
        """F/T 센서 하드웨어 Tare 서비스 호출."""
        self.get_logger().info(f"[DataCollect] Calling system Tare service: {TARE_SERVICE}")
        try:
            result = subprocess.run(
                ["ros2", "service", "call", TARE_SERVICE, TARE_TYPE],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self.get_logger().info("[DataCollect] System Tare completed successfully.")
            else:
                self.get_logger().warn(
                    f"[DataCollect] System Tare failed (code={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().error("[DataCollect] System Tare service timeout (10s)")
        except Exception as e:
            self.get_logger().error(f"[DataCollect] System Tare exception: {e}")
