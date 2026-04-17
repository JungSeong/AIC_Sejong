import os
import time
from pathlib import Path
from typing import Optional

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException

from .lib.cheatcode import CheatCodePlanner
from .lib.recording import AutoCaptureRecorder


class AutoCapture(Policy):
    """CheatCode motion with lightweight recording."""

    def __init__(self, parent_node):
        self._task: Optional[Task] = None
        self._latest_insertion_event: Optional[str] = None
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        super().__init__(parent_node)

        self._insertion_event_sub = self._parent_node.create_subscription(
            String, "/scoring/insertion_event", self._insertion_event_callback, 10
        )

        self.capture_root = Path(
            os.environ.get(
                "AIC_CAPTURE_DIR",
                str(Path(__file__).resolve().parents[4] / "data" / "aic_captures"),
            )
        )
        self.approach_z_offset = float(
            os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_Z_OFFSET", "0.2")
        )
        self.approach_steps = int(
            os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_STEPS", "100")
        )
        self.insert_z_step = float(
            os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_Z_STEP", "0.0005")
        )
        self.insert_min_z_offset = float(
            os.environ.get("AIC_CAPTURE_CHEATCODE_INSERT_MIN_Z_OFFSET", "-0.015")
        )
        self.stabilize_sec = float(
            os.environ.get("AIC_CAPTURE_CHEATCODE_STABILIZE_SEC", "5.0")
        )
        self.step_sleep_sec = float(os.environ.get("AIC_CAPTURE_STEP_SLEEP_SEC", "0.05"))
        self._planner = CheatCodePlanner(
            i_gain=float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15")),
            max_integrator_windup=self._max_integrator_windup,
        )
        self.get_logger().info(
            "AutoCapture initialized. Output dir: %s" % str(self.capture_root)
        )

    def _normalize_event_namespace(self, namespace: str) -> str:
        return namespace.strip().strip("/")

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event = self._normalize_event_namespace(msg.data)

    def _has_successful_insertion(self, task: Task) -> bool:
        if not self._latest_insertion_event:
            return False
        tokens = [token for token in self._latest_insertion_event.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[0] == task.target_module_name and tokens[1] == task.port_name

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        "Waiting for transform '%s' -> '%s'... -- are you running eval with `ground_truth:=true`?"
                        % (source_frame, target_frame)
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            "Transform '%s' not available after %ss" % (source_frame, timeout_sec)
        )
        return False

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        ).transform

    def _motion_update_from_pose(self, pose) -> MotionUpdate:
        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.header.stamp = self._parent_node.get_clock().now().to_msg()
        motion_update.pose = pose
        motion_update.target_stiffness = [
            90.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            90.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            90.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
        ]
        motion_update.target_damping = [
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            20.0,
        ]
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        return motion_update

    def _record_motion_step(
        self,
        recorder: AutoCaptureRecorder,
        phase: str,
        task: Task,
        port_transform: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        obs,
        pose,
        extras: dict,
    ) -> None:
        if obs is None:
            return
        recorder.record_step(
            phase=phase,
            task=task,
            obs=obs,
            action=self._motion_update_from_pose(pose),
            port_tf=port_transform,
            plug_tf=plug_tf,
            gripper_tf=gripper_tf,
            extras=extras,
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info("AutoCapture.insert_cable() task: %s" % task)
        self._task = task
        self._latest_insertion_event = None
        self._planner.reset()
        send_feedback("auto capture running")

        episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
        episode_dir = self.capture_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        recorder = AutoCaptureRecorder(episode_dir)

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, plug_frame]:
            if not self._wait_for_tf("base_link", frame):
                recorder.write_summary(
                    {
                        "task": recorder.task_to_dict(task),
                        "status": "setup_failed",
                        "selected_port_frame": port_frame,
                        "plug_frame": plug_frame,
                        "missing_frame": frame,
                    }
                )
                return False

        try:
            port_transform = self._lookup_transform("base_link", port_frame)
        except TransformException as ex:
            self.get_logger().error("Could not look up port transform: %s" % ex)
            recorder.write_summary(
                {
                    "task": recorder.task_to_dict(task),
                    "status": "setup_failed",
                    "selected_port_frame": port_frame,
                    "plug_frame": plug_frame,
                    "failure_reason": "port_transform_lookup_failed",
                }
            )
            return False

        start_time = time.time()
        phase_step_counts = {"approach": 0, "insert": 0, "stabilize": 0}
        phase_exit_reason = {
            "approach": "not_entered",
            "insert": "not_entered",
            "stabilize": "not_entered",
        }

        recorder.write_meta(
            {
                "task": recorder.task_to_dict(task),
                "selected_port_frame": port_frame,
                "plug_frame": plug_frame,
                "capture_root": str(self.capture_root),
                "approach_z_offset": self.approach_z_offset,
                "approach_steps": self.approach_steps,
                "insert_z_step": self.insert_z_step,
                "insert_min_z_offset": self.insert_min_z_offset,
                "stabilize_sec": self.stabilize_sec,
                "i_gain": self._planner.i_gain,
            }
        )

        z_offset = self.approach_z_offset
        for t in range(0, self.approach_steps):
            interp_fraction = t / float(self.approach_steps)
            try:
                plug_tf = self._lookup_transform("base_link", plug_frame)
                gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                pose, extras = self._planner.build_pose(
                    port_transform=port_transform,
                    plug_transform=plug_tf,
                    gripper_transform=gripper_tf,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
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
                    extras=extras,
                )
                phase_step_counts["approach"] += 1
            except TransformException as ex:
                self.get_logger().warn(
                    "TF lookup failed during interpolation: %s" % ex
                )
            self.sleep_for(self.step_sleep_sec)
        phase_exit_reason["approach"] = "completed_cheatcode_approach"

        while True:
            if z_offset < self.insert_min_z_offset:
                phase_exit_reason["insert"] = "reached_cheatcode_min_z_offset"
                break

            z_offset -= self.insert_z_step
            self.get_logger().info("z_offset: %.5f" % z_offset)
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
                obs = get_observation()
                self._record_motion_step(
                    recorder=recorder,
                    phase="insert",
                    task=task,
                    port_transform=port_transform,
                    plug_tf=plug_tf,
                    gripper_tf=gripper_tf,
                    obs=obs,
                    pose=pose,
                    extras=extras,
                )
                phase_step_counts["insert"] += 1
            except TransformException as ex:
                self.get_logger().warn("TF lookup failed during insertion: %s" % ex)
            self.sleep_for(self.step_sleep_sec)

        self.get_logger().info("Waiting for connector to stabilize...")
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
            self.get_logger().warn("TF lookup failed during stabilize: %s" % ex)

        insertion_event_observed = self._has_successful_insertion(task)
        recorder.write_summary(
            {
                "task": recorder.task_to_dict(task),
                "status": "completed",
                "selected_port_frame": port_frame,
                "plug_frame": plug_frame,
                "elapsed_sec": time.time() - start_time,
                "insertion_event_observed": insertion_event_observed,
                "phase_step_counts": phase_step_counts,
                "phase_exit_reason": phase_exit_reason,
            }
        )

        self.get_logger().info("AutoCapture.insert_cable() exiting...")
        send_feedback("auto capture complete")
        return True
