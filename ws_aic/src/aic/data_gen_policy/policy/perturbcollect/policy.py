"""
PerturbCollect policy
─────────────────────
DataCollect(= AutoCapture + Tare)에 XY 정렬 오차 perturbation을 추가한 정책.

목적:
  approach 이후 insert 진입 시 의도적으로 XY offset을 주입하여
  다양한 정렬 오차 조건의 데이터를 수집한다.
  → force_y bias 해소, 인코더 강건성 향상.

동작 흐름:
  1. F/T 센서 Tare (DataCollect 상속)
  2. approach 페이즈: CheatCode 그대로 (포트를 향해 정렬)
  3. insert 진입 시: (Δx, Δy) 샘플링 → 포트 목표 좌표에 offset 주입
  4. insert 내내 perturbation 유지 (decay 옵션으로 선형 감소 가능)
  5. extras에 perturbation 값 기록 → EDA에서 분석 가능

환경변수:
  AIC_CAPTURE_PERTURB_XY_RANGE   XY offset 최대 크기 [m] (기본 0.010 = 10mm)
  AIC_CAPTURE_PERTURB_XY_DIST    샘플링 분포: 'uniform' | 'gaussian' (기본 uniform)
  AIC_CAPTURE_PERTURB_DECAY      insert 진행에 따라 offset 선형 감소: '1' | '0' (기본 0)

실행:
  pixi run ros2 run aic_model aic_model --ros-args \\
    -p policy:=data_gen_policy.policy.perturbcollect
"""

import os
import time
from pathlib import Path

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Point, Quaternion, Transform
from tf2_ros import TransformException

from data_gen_policy.policy.datacollect import DataCollect
from data_gen_policy.policy.autocapture.lib.recording import AutoCaptureRecorder


class PerturbCollect(DataCollect):
    """
    insert 페이즈 진입 시 XY offset을 주입하여 다양한 정렬 오차 조건을 수집.

    DataCollect를 상속하므로 에피소드마다 F/T Tare가 자동 호출된다.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self.perturb_xy_range = float(
            os.environ.get("AIC_CAPTURE_PERTURB_XY_RANGE", "0.010")
        )
        self.perturb_xy_dist = os.environ.get(
            "AIC_CAPTURE_PERTURB_XY_DIST", "uniform"
        ).lower()
        self.perturb_decay = os.environ.get(
            "AIC_CAPTURE_PERTURB_DECAY", "0"
        ) == "1"

        self.get_logger().info(
            f"[PerturbCollect] xy_range={self.perturb_xy_range}m  "
            f"dist={self.perturb_xy_dist}  decay={self.perturb_decay}"
        )

    # ──────────────────────────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────────────────────────

    def _sample_perturbation(self) -> tuple[float, float]:
        """에피소드마다 한 번 호출. (Δx, Δy) [m] 반환."""
        r = self.perturb_xy_range
        if self.perturb_xy_dist == "gaussian":
            # 3σ = range 가 되도록 σ = range/3
            sigma = r / 3.0
            dx = float(np.clip(np.random.normal(0, sigma), -r, r))
            dy = float(np.clip(np.random.normal(0, sigma), -r, r))
        else:  # uniform
            dx = float(np.random.uniform(-r, r))
            dy = float(np.random.uniform(-r, r))
        return dx, dy

    def _apply_xy_offset(self, pose: Pose, dx: float, dy: float) -> Pose:
        """pose의 XY 위치에 offset을 더한 새 Pose 반환."""
        new_pose = Pose(
            position=Point(
                x=pose.position.x + dx,
                y=pose.position.y + dy,
                z=pose.position.z,
            ),
            orientation=Quaternion(
                w=pose.orientation.w,
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
            ),
        )
        return new_pose

    # ──────────────────────────────────────────────────────────────
    # 메인 로직 (AutoCapture.insert_cable 재구현)
    # ──────────────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        # DataCollect의 Tare 호출
        self._tare_sensor()

        self.get_logger().info("PerturbCollect.insert_cable() task: %s" % task)
        self._task = task
        self._latest_insertion_event = None
        self._planner.reset()
        send_feedback("perturb collect running")

        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        recorder = AutoCaptureRecorder(episode_dir)

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, plug_frame]:
            if not self._wait_for_tf("base_link", frame):
                recorder.write_summary({
                    "task": recorder.task_to_dict(task),
                    "status": "setup_failed",
                    "missing_frame": frame,
                })
                return False

        try:
            port_transform = self._lookup_transform("base_link", port_frame)
        except TransformException as ex:
            self.get_logger().error("Port TF lookup failed: %s" % ex)
            return False

        # insert 진입 시 사용할 XY perturbation 샘플링
        perturb_dx, perturb_dy = self._sample_perturbation()
        
        # [수정 이유] 에피소드 시작 시 실제 파지 편차(Grasp Deviation)를 기록하여
        # ~2mm 수준의 초기 오차가 정책 성공률에 미치는 영향을 분석하고 로버스트성을 확보하기 위함.
        try:
            init_plug_tf = self._lookup_transform("base_link", plug_frame)
            init_gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            initial_gripper_offset = {
                "x": init_gripper_tf.translation.x - init_plug_tf.translation.x,
                "y": init_gripper_tf.translation.y - init_plug_tf.translation.y,
                "z": init_gripper_tf.translation.z - init_plug_tf.translation.z,
            }
        except Exception:
            initial_gripper_offset = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.get_logger().info(
            f"[PerturbCollect] perturbation  dx={perturb_dx*1000:.1f}mm  "
            f"dy={perturb_dy*1000:.1f}mm  "
            f"initial_gripper_offset_z={initial_gripper_offset['z']*1000:.1f}mm"
        )

        # [수정 이유] 시나리오 생성 스크립트(collect_data.py)에서 랜덤하게 생성된 
        # 그라운드 트루스 파지 오프셋 값을 환경변수에서 읽어와 기록함.
        gt_offset_x = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_X", "0.0"))
        gt_offset_y = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_Y", "0.0"))
        gt_offset_z = float(os.environ.get("AIC_CAPTURE_GRIPPER_OFFSET_Z", "0.0"))
        ground_truth_gripper_offset = {"x": gt_offset_x, "y": gt_offset_y, "z": gt_offset_z}

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
            # [수정 사항] perturbation 및 파지 오프셋 기록 (robustness 분석용)
            "perturb_dx_m": perturb_dx,
            "perturb_dy_m": perturb_dy,
            "perturb_xy_range": self.perturb_xy_range,
            "perturb_xy_dist": self.perturb_xy_dist,
            "perturb_decay": self.perturb_decay,
            "initial_gripper_offset": initial_gripper_offset,  # TF 기반 측정값
            "ground_truth_gripper_offset": ground_truth_gripper_offset,  # 생성 스크립트 기반 실제값
        })

        start_time = time.time()
        phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}
        phase_exit_reason = {
            "approach": "not_entered",
            "insert": "not_entered",
            "stabilize": "not_entered",
        }

        # ── approach 페이즈: perturbation 없음 ──────────────────────
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
                obs = get_observation()
                self._record_motion_step(
                    recorder=recorder,
                    phase="approach",
                    task=task,
                    port_transform=port_transform,
                    plug_tf=plug_tf,
                    gripper_tf=gripper_tf,
                    obs=obs,
                    pose=pose,
                    extras={**extras, "perturb_dx_m": 0.0, "perturb_dy_m": 0.0},
                )
                phase_step_counts["approach"] += 1
            except TransformException as ex:
                self.get_logger().warn("TF lookup failed (approach): %s" % ex)
            self.sleep_for(self.step_sleep_sec)
        phase_exit_reason["approach"] = "completed_cheatcode_approach"

        # ── insert 페이즈: XY perturbation 주입 ─────────────────────
        # insert 전체 예상 스텝 수 (decay 계산용)
        total_insert_steps = int(
            (self.approach_z_offset - self.insert_min_z_offset) / self.insert_z_step
        )
        insert_step_idx = 0

        while True:
            if z_offset < self.insert_min_z_offset:
                phase_exit_reason["insert"] = "reached_cheatcode_min_z_offset"
                break

            z_offset -= self.insert_z_step

            # decay: insert 진행에 따라 perturbation 선형 감소 → 0
            if self.perturb_decay:
                decay_factor = max(0.0, 1.0 - insert_step_idx / total_insert_steps)
            else:
                decay_factor = 1.0

            effective_dx = perturb_dx * decay_factor
            effective_dy = perturb_dy * decay_factor

            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                pose, extras = self._planner.build_pose(
                    port_transform=port_transform,
                    plug_transform=plug_tf,
                    gripper_transform=gripper_tf,
                    z_offset=z_offset,
                )
                # XY offset 주입
                perturbed_pose = self._apply_xy_offset(pose, effective_dx, effective_dy)
                self.set_pose_target(move_robot=move_robot, pose=perturbed_pose)
                obs = get_observation()
                self._record_motion_step(
                    recorder=recorder,
                    phase="insert",
                    task=task,
                    port_transform=port_transform,
                    plug_tf=plug_tf,
                    gripper_tf=gripper_tf,
                    obs=obs,
                    pose=perturbed_pose,
                    extras={
                        **extras,
                        "perturb_dx_m": effective_dx,
                        "perturb_dy_m": effective_dy,
                        "perturb_decay_factor": decay_factor,
                    },
                )
                phase_step_counts["insert"] += 1
            except TransformException as ex:
                self.get_logger().warn("TF lookup failed (insert): %s" % ex)

            insert_step_idx += 1
            self.sleep_for(self.step_sleep_sec)

        # ── stabilize ────────────────────────────────────────────────
        self.sleep_for(self.stabilize_sec)
        try:
            plug_tf = self._lookup_transform("base_link", plug_frame)
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            obs = get_observation()
            if obs is not None:
                recorder.record_terminal_step(
                    phase="stabilize",
                    task=task,
                    obs=obs,
                    port_tf=port_transform,
                    plug_tf=plug_tf,
                    gripper_tf=gripper_tf,
                    extras={"z_offset": z_offset},
                )
                phase_step_counts["stabilize"] = 1
                phase_exit_reason["stabilize"] = "captured"
        except TransformException as ex:
            self.get_logger().warn("TF lookup failed (stabilize): %s" % ex)

        insertion_event_observed = self._has_successful_insertion(task)
        recorder.write_summary({
            "task": recorder.task_to_dict(task),
            "status": "completed",
            "selected_port_frame": port_frame,
            "plug_frame": plug_frame,
            "elapsed_sec": time.time() - start_time,
            "insertion_event_observed": insertion_event_observed,
            "phase_step_counts": phase_step_counts,
            "phase_exit_reason": phase_exit_reason,
            "initial_gripper_offset": initial_gripper_offset,  # TF 기반 측정값
            "ground_truth_gripper_offset": ground_truth_gripper_offset,  # 생성 스크립트 기반 실제값
            "perturb_dx_m": perturb_dx,
            "perturb_dy_m": perturb_dy,
        })

        send_feedback("perturb collect complete")
        return True
