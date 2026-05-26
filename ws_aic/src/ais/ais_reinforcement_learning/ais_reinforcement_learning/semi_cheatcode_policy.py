from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from data_gen_node.lib.cheatcode import CheatCodePlanner
from distance_prediction_policy.DebugSfpDistancePolicy import DebugSfpDistancePolicy
from distance_prediction_policy.config import DistancePredictionConfig
from geometry_msgs.msg import Pose, Transform, Vector3
from rclpy.time import Time

from ais_reinforcement_learning.config import SfpRlConfig
from ais_reinforcement_learning.schema import ActionSample


class SfpSemiCheatcodePolicy(DebugSfpDistancePolicy):
    """SFP-only data policy: learned approach, TF cheatcode align/insert.

    It records state-action pairs for supervised pretraining. The action target
    is the next TCP delta plus target quaternion.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._cheat_planner = CheatCodePlanner()
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._sample_index = 0
        self._rollout_path = Path(SfpRlConfig.ROLLOUT_DIR) / "samples.jsonl"
        if SfpRlConfig.RECORD_ROLLOUTS:
            self._rollout_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            "SfpSemiCheatcodePolicy ready: "
            f"record={SfpRlConfig.RECORD_ROLLOUTS}, path={self._rollout_path}"
        )

    @staticmethod
    def _pose_to_transform(pose: Pose) -> Transform:
        return Transform(
            translation=Vector3(
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
            ),
            rotation=pose.orientation,
        )

    @staticmethod
    def _pose_position(pose: Pose) -> np.ndarray:
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )

    @staticmethod
    def _transform_position(transform: Transform) -> np.ndarray:
        return np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_xyzw_from_pose(pose: Pose) -> list[float]:
        return [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ]

    def _lookup_first_transform(self, frames: Iterable[str]) -> Optional[tuple[str, Transform]]:
        buffer = getattr(self._parent_node, "_tf_buffer", None)
        if buffer is None:
            self.get_logger().error("semi-cheatcode failed: parent node has no TF buffer")
            return None

        for frame in frames:
            try:
                return frame, buffer.lookup_transform("base_link", frame, Time()).transform
            except Exception:
                continue
        return None

    def _port_frame_candidates(self, task: Task) -> tuple[str, ...]:
        module = str(task.target_module_name)
        port = str(task.port_name)
        if port.endswith("_link"):
            base_names = (port, f"{port}_entrance")
        else:
            base_names = (f"{port}_link_entrance", f"{port}_link", port)
        return tuple(f"task_board/{module}/{name}" for name in base_names)

    def _plug_frame_candidates(self, task: Task) -> tuple[str, ...]:
        cable = str(task.cable_name)
        plug = str(task.plug_name)
        if plug.endswith("_link"):
            return (f"{cable}/{plug}", plug)
        return (f"{cable}/{plug}_link", f"{cable}/{plug}", plug)

    def _make_state(
        self,
        *,
        observation,
        tcp_pose: Pose,
        z_offset: float,
        phase_code: float,
    ) -> list[float]:
        tcp_xyz = self._pose_position(tcp_pose)
        cached_port = self._cached_port_base
        if cached_port is None:
            cached_port = np.zeros(3, dtype=np.float64)
        offset_m = self._distance.predict_offset_m(observation, self._port_id())
        offset_valid = 1.0
        if offset_m is None:
            offset_m = np.zeros(3, dtype=np.float64)
            offset_valid = 0.0
        force = self._force_norm(observation)
        return [
            *[float(v) for v in tcp_xyz],
            *self._quat_xyzw_from_pose(tcp_pose),
            *[float(v) for v in cached_port],
            *[float(v) for v in offset_m],
            float(offset_valid),
            0.0 if force is None else float(force),
            float(z_offset),
            float(phase_code),
        ]

    def _make_action(self, *, tcp_pose: Pose, target_pose: Pose) -> list[float]:
        delta = self._pose_position(target_pose) - self._pose_position(tcp_pose)
        return [
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            *self._quat_xyzw_from_pose(target_pose),
        ]

    def _record_sample(
        self,
        *,
        task: Task,
        phase: str,
        step: int,
        state: list[float],
        action: list[float],
        extras: dict,
    ) -> None:
        if not SfpRlConfig.RECORD_ROLLOUTS:
            return

        sample = ActionSample(
            sample_id=f"{self._run_id}_{self._sample_index:06d}",
            run_id=self._run_id,
            task_target=str(task.target_module_name),
            task_port=str(task.port_name),
            task_cable=str(task.cable_name),
            task_plug=str(task.plug_name),
            phase=phase,
            step=int(step),
            state=state,
            action=action,
            extras=extras,
        )
        self._sample_index += 1
        with self._rollout_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(sample.to_json_dict(), sort_keys=True) + "\n")

    def _cheatcode_step(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        phase: str,
        step: int,
        z_offset: float,
    ) -> bool:
        obs = get_observation()
        tcp_pose = self._tcp_pose(obs)
        if tcp_pose is None:
            self.get_logger().error(f"{phase} failed: missing TCP pose")
            return False

        port = self._lookup_first_transform(self._port_frame_candidates(task))
        plug = self._lookup_first_transform(self._plug_frame_candidates(task))
        if port is None:
            self.get_logger().error(
                f"{phase} failed: no port TF from {self._port_frame_candidates(task)}"
            )
            return False
        if plug is None:
            self.get_logger().error(
                f"{phase} failed: no plug TF from {self._plug_frame_candidates(task)}"
            )
            return False

        target_pose, extras = self._cheat_planner.build_pose(
            port_transform=port[1],
            plug_transform=plug[1],
            gripper_transform=self._pose_to_transform(tcp_pose),
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=float(z_offset),
            reset_xy_integrator=(step == 0 and phase == "cheat_align"),
        )

        state = self._make_state(
            observation=obs,
            tcp_pose=tcp_pose,
            z_offset=float(z_offset),
            phase_code=0.0 if phase == "cheat_align" else 1.0,
        )
        action = self._make_action(tcp_pose=tcp_pose, target_pose=target_pose)
        extras.update(
            {
                "port_frame": port[0],
                "plug_frame": plug[0],
                "port_xyz_gt": {
                    "x": float(port[1].translation.x),
                    "y": float(port[1].translation.y),
                    "z": float(port[1].translation.z),
                },
                "plug_xyz_gt": {
                    "x": float(plug[1].translation.x),
                    "y": float(plug[1].translation.y),
                    "z": float(plug[1].translation.z),
                },
                "target_pose": {
                    "x": float(target_pose.position.x),
                    "y": float(target_pose.position.y),
                    "z": float(target_pose.position.z),
                    "qx": float(target_pose.orientation.x),
                    "qy": float(target_pose.orientation.y),
                    "qz": float(target_pose.orientation.z),
                    "qw": float(target_pose.orientation.w),
                },
            }
        )
        self._record_sample(
            task=task,
            phase=phase,
            step=step,
            state=state,
            action=action,
            extras=extras,
        )

        self.set_pose_target(
            move_robot=move_robot,
            pose=target_pose,
            stiffness=list(DistancePredictionConfig.STIFFNESS),
            damping=list(DistancePredictionConfig.DAMPING),
        )
        if step == 0 or step % 10 == 0:
            self.get_logger().info(
                f"{phase}[{step:03d}]: z_offset={z_offset * 1000:+.1f}mm, "
                f"tip_xy=({extras['tip_x_error'] * 1000:+.2f}, "
                f"{extras['tip_y_error'] * 1000:+.2f})mm, "
                f"axis={extras['tip_axis_distance'] * 1000:+.2f}mm"
            )
        self.sleep_for(SfpRlConfig.CHEAT_STEP_DT_S)
        return True

    def _stage_cheat_align_insert(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> bool:
        self.get_logger().info("[rl stage] semi-cheatcode align/insert start")
        self._cheat_planner.reset()

        align_offsets = np.linspace(
            SfpRlConfig.CHEAT_ALIGN_Z_START_M,
            SfpRlConfig.CHEAT_ALIGN_Z_END_M,
            max(1, int(SfpRlConfig.CHEAT_ALIGN_STEPS)),
        )
        for step, z_offset in enumerate(align_offsets):
            if not self._cheatcode_step(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                phase="cheat_align",
                step=step,
                z_offset=float(z_offset),
            ):
                return False

        insert_offsets = np.linspace(
            SfpRlConfig.CHEAT_ALIGN_Z_END_M,
            SfpRlConfig.CHEAT_INSERT_Z_END_M,
            max(1, int(SfpRlConfig.CHEAT_INSERT_STEPS)),
        )
        for step, z_offset in enumerate(insert_offsets):
            if not self._cheatcode_step(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                phase="cheat_insert",
                step=step,
                z_offset=float(z_offset),
            ):
                return False

        if SfpRlConfig.CHEAT_SETTLE_S > 0:
            self.sleep_for(SfpRlConfig.CHEAT_SETTLE_S)
        self.get_logger().info("[rl stage] semi-cheatcode align/insert done")
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if "sfp" not in str(task.port_name).lower():
            self.get_logger().warn(
                "SfpSemiCheatcodePolicy only supports SFP tasks; "
                f"got port={task.port_name}"
            )
            return False

        self._task = task
        self._cached_port_base = None
        self._target_orientation = None
        self.get_logger().info(
            "SfpSemiCheatcodePolicy start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )

        stages = (
            ("initial_lift", lambda: self._stage_initial_lift(get_observation, move_robot)),
            ("detect", lambda: self._stage_detect(get_observation)),
            ("approach", lambda: self._stage_approach(get_observation, move_robot)),
            (
                "semi_cheatcode_align_insert",
                lambda: self._stage_cheat_align_insert(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                ),
            ),
        )
        for name, stage in stages:
            send_feedback(f"sfp rl semi-cheatcode: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"SFP RL policy failed at stage: {name}")
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"SFP RL policy exception at {name}: {exc}")
                send_feedback(f"failed: {name} exception")
                return False

        self.get_logger().info("SfpSemiCheatcodePolicy done")
        send_feedback("sfp rl semi-cheatcode done")
        return True
