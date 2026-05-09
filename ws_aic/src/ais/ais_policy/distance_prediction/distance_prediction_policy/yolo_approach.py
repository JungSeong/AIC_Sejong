"""YOLO triangulation approach for the distance prediction policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from distance_prediction_policy.config import DistancePredictionConfig
from motion_planning_node.core.geometry import (
    interp_profile,
    quat_inverse,
    quat_to_tuple,
    tuple_to_quat,
)


@dataclass
class YoloApproachResult:
    success: bool
    port_position: Optional[np.ndarray]
    elapsed_time: float
    failure_reason: Optional[str] = None


class YoloTriangulationApproach:
    """Approach the port using only YOLO detections triangulated from cameras."""

    def __init__(self, policy, vision, config=DistancePredictionConfig) -> None:
        self._policy = policy
        self._vision = vision
        self._config = config

    def get_logger(self):
        return self._policy.get_logger()

    def time_now(self):
        return self._policy.time_now()

    def sleep_for(self, duration_sec: float) -> None:
        self._policy.sleep_for(duration_sec)

    def set_pose_target(self, *args, **kwargs):
        return self._policy.set_pose_target(*args, **kwargs)

    def _target_class_id(self) -> int:
        task = self._policy._task
        tokens = " ".join(
            str(value or "").lower()
            for value in (
                getattr(task, "plug_name", ""),
                getattr(task, "port_name", ""),
                getattr(task, "task_type", ""),
            )
        )
        return 1 if "sc" in tokens else 0

    def _is_sfp_task(self) -> bool:
        task = self._policy._task
        tokens = " ".join(
            str(value or "").lower()
            for value in (
                getattr(task, "plug_name", ""),
                getattr(task, "port_name", ""),
                getattr(task, "port_type", ""),
                getattr(task, "task_type", ""),
            )
        )
        return "sfp" in tokens

    def _initial_approach_z_offset(self) -> float:
        if self._is_sfp_task():
            return float(self._config.APPROACH_Z_OFFSET_SFP_M)
        return float(self._config.APPROACH_Z_OFFSET_SC_M)

    def _copy_pose(self, pose: Pose) -> Pose:
        return Pose(
            position=Point(
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
            ),
            orientation=Quaternion(
                x=float(pose.orientation.x),
                y=float(pose.orientation.y),
                z=float(pose.orientation.z),
                w=float(pose.orientation.w),
            ),
        )

    def _copy_quaternion(self, orientation: Quaternion) -> Quaternion:
        return Quaternion(
            x=float(orientation.x),
            y=float(orientation.y),
            z=float(orientation.z),
            w=float(orientation.w),
        )

    def _tcp_pose(self, observation) -> Optional[Pose]:
        if observation is None:
            return None
        return self._copy_pose(observation.controller_state.tcp_pose)

    def _force_mag(self, observation) -> Optional[float]:
        if observation is None:
            return None
        force = observation.wrist_wrench.wrench.force
        return float(
            np.sqrt(force.x * force.x + force.y * force.y + force.z * force.z)
        )

    def _port_frame(self) -> str:
        task = self._policy._task
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

    def _plug_frame(self) -> str:
        task = self._policy._task
        return f"{task.cable_name}/{task.plug_name}_link"

    def _lookup_tf_once(self, frame: str):
        try:
            return self._policy._parent_node._tf_buffer.lookup_transform(
                "base_link", frame, Time()
            ).transform
        except TransformException:
            return None

    def _transform_orientation(self, transform) -> Quaternion:
        return Quaternion(
            x=float(transform.rotation.x),
            y=float(transform.rotation.y),
            z=float(transform.rotation.z),
            w=float(transform.rotation.w),
        )

    def _normalize_quat(self, quat: tuple[float, float, float, float]) -> tuple:
        q = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        q /= norm
        return tuple(float(v) for v in q)

    def _axis_angle_quat(
        self,
        axis: np.ndarray,
        angle: float,
    ) -> tuple[float, float, float, float]:
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        axis /= norm
        half = 0.5 * float(angle)
        sin_half = float(np.sin(half))
        return self._normalize_quat(
            (
                float(np.cos(half)),
                float(axis[0] * sin_half),
                float(axis[1] * sin_half),
                float(axis[2] * sin_half),
            )
        )

    def _manual_rotation_axis(self, tcp_pose: Pose) -> np.ndarray:
        axis_name = self._config.APPROACH_SFP_MANUAL_ROTATION_AXIS
        axes = {
            "base_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "base_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "base_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if axis_name in axes:
            return axes[axis_name]

        local_axes = {
            "tcp_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "tcp_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "tcp_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        local_axis = local_axes.get(axis_name, local_axes["tcp_z"])
        q_tcp = quat_to_tuple(tcp_pose.orientation)
        rotated = quaternion_multiply(
            quaternion_multiply(q_tcp, (0.0, *local_axis)),
            quat_inverse(q_tcp),
        )
        return np.array([rotated[1], rotated[2], rotated[3]], dtype=np.float64)

    def _manual_wrist_rotation_target_pose(
        self,
        tcp_pose: Pose,
    ) -> tuple[Optional[Pose], float]:
        angle_deg = self._config.APPROACH_SFP_MANUAL_ROTATION_DEG
        if abs(angle_deg) < 1e-9:
            return None, 0.0

        angle_rad = float(np.radians(angle_deg))
        q_delta = self._axis_angle_quat(
            self._manual_rotation_axis(tcp_pose),
            angle_rad,
        )
        q_tcp = quat_to_tuple(tcp_pose.orientation)
        q_target = self._normalize_quat(quaternion_multiply(q_delta, q_tcp))
        target_pose = self._copy_pose(tcp_pose)
        target_pose.orientation = tuple_to_quat(q_target)
        return target_pose, angle_rad

    def _run_manual_wrist_rotation(
        self,
        *,
        initial_observation,
        baseline_force: float,
        get_observation,
        move_robot,
    ) -> Optional[str]:
        if not self._is_sfp_task():
            return None

        start_pose = self._tcp_pose(initial_observation)
        if start_pose is None:
            self.get_logger().warn("Manual SFP wrist rotation skipped: no TCP pose")
            return None

        target_pose, angle_rad = self._manual_wrist_rotation_target_pose(start_pose)
        if target_pose is None:
            return None

        self.get_logger().info(
            "Manual SFP wrist rotation start: "
            f"axis={self._config.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
            f"angle={np.degrees(angle_rad):+.1f}deg"
        )
        failure = self._follow_path(
            start_pose=start_pose,
            target_pose=target_pose,
            steps=self._config.APPROACH_SFP_MANUAL_ROTATION_STEPS,
            stiffness=self._config.APPROACH_NEAR_STIFFNESS,
            damping=self._config.APPROACH_NEAR_DAMPING,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
            dt=self._config.APPROACH_SFP_MANUAL_ROTATION_DT,
        )
        if failure is not None:
            return failure
        self.get_logger().info("Manual SFP wrist rotation done")
        return None

    def _sfp_upright_target_pose(self, tcp_pose: Pose) -> tuple[Optional[Pose], float]:
        plug_tf = self._lookup_tf_once(self._plug_frame())
        if plug_tf is None:
            self.get_logger().warn(
                "SFP upright skipped: plug TF unavailable "
                f"({self._plug_frame()})"
            )
            return None, 0.0

        tcp_position = np.array(
            [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
            dtype=np.float64,
        )
        tip_position = np.array(
            [plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z],
            dtype=np.float64,
        )
        tip_vector = tip_position - tcp_position
        tip_norm = float(np.linalg.norm(tip_vector))
        if tip_norm < 1e-9:
            self.get_logger().warn("SFP upright skipped: tcp-to-tip vector is zero")
            return None, 0.0

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        horizontal = tip_vector - z_axis * float(np.dot(tip_vector, z_axis))
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-9:
            self.get_logger().warn(
                "SFP upright skipped: tcp-to-tip vector is nearly parallel to z"
            )
            return None, 0.0

        v = tip_vector / tip_norm
        h = horizontal / horizontal_norm
        cross = np.cross(v, h)
        angle = float(np.arctan2(np.linalg.norm(cross), np.dot(v, h)))
        if angle < self._config.APPROACH_SFP_UPRIGHT_TOLERANCE_RAD:
            self.get_logger().info(
                f"SFP upright already satisfied: tilt={np.degrees(angle):.2f}deg"
            )
            return None, angle

        max_angle = self._config.APPROACH_SFP_UPRIGHT_MAX_ANGLE_RAD
        clipped_angle = min(angle, max_angle)
        if clipped_angle < angle:
            self.get_logger().warn(
                "SFP upright correction clipped: "
                f"{np.degrees(angle):.1f}deg -> {np.degrees(clipped_angle):.1f}deg"
            )

        q_delta = self._axis_angle_quat(cross, clipped_angle)
        q_tcp = quat_to_tuple(tcp_pose.orientation)
        q_target = self._normalize_quat(quaternion_multiply(q_delta, q_tcp))
        target_pose = self._copy_pose(tcp_pose)
        target_pose.orientation = tuple_to_quat(q_target)
        return target_pose, clipped_angle

    def _run_sfp_upright(
        self,
        *,
        initial_observation,
        baseline_force: float,
        get_observation,
        move_robot,
    ) -> Optional[str]:
        if not self._config.APPROACH_SFP_UPRIGHT_ENABLED or not self._is_sfp_task():
            return None

        start_pose = self._tcp_pose(initial_observation)
        if start_pose is None:
            self.get_logger().warn("SFP upright skipped: missing initial TCP pose")
            return None

        target_pose, angle = self._sfp_upright_target_pose(start_pose)
        if target_pose is None:
            return None

        self.get_logger().info(
            "SFP upright start: rotate wrist so tcp->sfp_tip is perpendicular "
            f"to base z ({np.degrees(angle):.1f}deg)"
        )
        failure = self._follow_path(
            start_pose=start_pose,
            target_pose=target_pose,
            steps=self._config.APPROACH_SFP_UPRIGHT_STEPS,
            stiffness=self._config.APPROACH_NEAR_STIFFNESS,
            damping=self._config.APPROACH_NEAR_DAMPING,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
            dt=self._config.APPROACH_SFP_UPRIGHT_DT,
        )
        if failure is not None:
            return failure
        self.get_logger().info("SFP upright done")
        return None

    def _target_wrist_orientation(self, reference_pose: Pose) -> Quaternion:
        """Align plug orientation to the port while keeping YOLO as position source."""
        if not self._config.APPROACH_USE_TF_WRIST_ALIGNMENT:
            return self._copy_quaternion(reference_pose.orientation)

        port_tf = self._lookup_tf_once(self._port_frame())
        plug_tf = self._lookup_tf_once(self._plug_frame())
        if port_tf is None or plug_tf is None:
            missing = []
            if port_tf is None:
                missing.append(self._port_frame())
            if plug_tf is None:
                missing.append(self._plug_frame())
            self.get_logger().warn(
                "Wrist alignment TF unavailable; keeping current orientation "
                f"({', '.join(missing)})"
            )
            return self._copy_quaternion(reference_pose.orientation)

        q_port = quat_to_tuple(self._transform_orientation(port_tf))
        q_plug = quat_to_tuple(self._transform_orientation(plug_tf))
        q_tcp = quat_to_tuple(reference_pose.orientation)
        q_diff = quaternion_multiply(q_port, quat_inverse(q_plug))
        q_target = quaternion_multiply(q_diff, q_tcp)
        self.get_logger().info(
            "Wrist alignment enabled: "
            "position=YOLO triangulation, orientation=port/plug TF"
        )
        return tuple_to_quat(q_target)

    def _estimate_port(self, get_observation) -> tuple[Optional[np.ndarray], object]:
        target_class_id = self._target_class_id()
        port_hint = getattr(self._policy._task, "port_name", "") or ""

        for attempt in range(self._config.APPROACH_VISION_RETRIES):
            observation = get_observation()
            if observation is None:
                self.sleep_for(self._config.APPROACH_RETRY_DT)
                continue

            port_position = self._vision.estimate(
                observation,
                target_class_id,
                port_hint=port_hint,
            )
            if port_position is not None:
                if attempt > 0:
                    self.get_logger().info(
                        f"YOLO triangulation succeeded after {attempt + 1} tries"
                    )
                return port_position, observation

            self.sleep_for(self._config.APPROACH_RETRY_DT)

        return None, None

    def _approach_pose(
        self,
        port_position: np.ndarray,
        z_offset: float,
        target_orientation: Quaternion,
    ) -> Pose:
        tcp_offset = np.array(
            [
                self._config.APPROACH_TCP_OFFSET_X_M,
                self._config.APPROACH_TCP_OFFSET_Y_M,
                self._config.APPROACH_TCP_OFFSET_Z_M,
            ],
            dtype=np.float64,
        )
        target_position = (
            port_position.astype(np.float64)
            + np.array([0.0, 0.0, z_offset], dtype=np.float64)
            + tcp_offset
        )
        return Pose(
            position=Point(
                x=float(target_position[0]),
                y=float(target_position[1]),
                z=float(target_position[2]),
            ),
            orientation=self._copy_quaternion(target_orientation),
        )

    def _follow_path(
        self,
        *,
        start_pose: Pose,
        target_pose: Pose,
        steps: int,
        stiffness: tuple,
        damping: tuple,
        baseline_force: float,
        get_observation,
        move_robot,
        dt: Optional[float] = None,
    ) -> Optional[str]:
        start = np.array(
            [
                start_pose.position.x,
                start_pose.position.y,
                start_pose.position.z,
            ],
            dtype=np.float64,
        )
        target = np.array(
            [
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
            ],
            dtype=np.float64,
        )
        q_start = quat_to_tuple(start_pose.orientation)
        q_target = quat_to_tuple(target_pose.orientation)

        for idx in range(max(steps, 1)):
            observation = get_observation()
            force_mag = self._force_mag(observation)
            if force_mag is not None:
                force_delta = force_mag - baseline_force
                if force_delta > self._config.APPROACH_FORCE_DELTA_LIMIT_N:
                    return f"collision during YOLO approach (+{force_delta:.1f}N)"

            t = interp_profile((idx + 1) / max(steps, 1), quintic=True)
            position = start * (1.0 - t) + target * t
            orientation = quaternion_slerp(q_start, q_target, t)
            waypoint = Pose(
                position=Point(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                ),
                orientation=tuple_to_quat(orientation),
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=waypoint,
                stiffness=list(stiffness),
                damping=list(damping),
            )
            self.sleep_for(self._config.APPROACH_DT if dt is None else dt)

        return None

    def run(self, get_observation, move_robot, send_feedback) -> YoloApproachResult:
        self.get_logger().info("YOLO triangulation approach start")
        send_feedback("Approach: YOLO triangulation")
        start_time = self.time_now()

        initial_observation = get_observation()
        baseline_force = self._force_mag(initial_observation) or 0.0

        failure = self._run_manual_wrist_rotation(
            initial_observation=initial_observation,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
        )
        if failure is not None:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            return YoloApproachResult(
                success=False,
                port_position=None,
                elapsed_time=elapsed,
                failure_reason=failure,
            )

        initial_observation = get_observation()
        failure = self._run_sfp_upright(
            initial_observation=initial_observation,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
        )
        if failure is not None:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            return YoloApproachResult(
                success=False,
                port_position=None,
                elapsed_time=elapsed,
                failure_reason=failure,
            )

        port_position, observation = self._estimate_port(get_observation)
        if port_position is None:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            return YoloApproachResult(
                success=False,
                port_position=None,
                elapsed_time=elapsed,
                failure_reason="YOLO triangulation failed",
            )

        self.get_logger().info(
            "YOLO port position: "
            f"({port_position[0]:+.3f}, {port_position[1]:+.3f}, "
            f"{port_position[2]:+.3f})"
        )

        start_pose = self._tcp_pose(observation) or self._tcp_pose(
            initial_observation
        )
        if start_pose is None:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            return YoloApproachResult(
                success=False,
                port_position=port_position,
                elapsed_time=elapsed,
                failure_reason="missing TCP pose in observation",
            )

        target_orientation = self._target_wrist_orientation(start_pose)
        initial_z_offset = self._initial_approach_z_offset()
        self.get_logger().info(
            "YOLO approach z offsets: "
            f"initial={initial_z_offset:.3f}m, "
            f"near={self._config.APPROACH_NEAR_Z_OFFSET_M:.3f}m"
        )
        far_pose = self._approach_pose(
            port_position,
            initial_z_offset,
            target_orientation,
        )
        failure = self._follow_path(
            start_pose=start_pose,
            target_pose=far_pose,
            steps=self._config.APPROACH_STEPS,
            stiffness=self._config.APPROACH_STIFFNESS,
            damping=self._config.APPROACH_DAMPING,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
        )
        if failure is not None:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            return YoloApproachResult(False, port_position, elapsed, failure)

        near_start_observation = get_observation()
        near_start_pose = self._tcp_pose(near_start_observation) or far_pose
        near_pose = self._approach_pose(
            port_position,
            self._config.APPROACH_NEAR_Z_OFFSET_M,
            target_orientation,
        )
        failure = self._follow_path(
            start_pose=near_start_pose,
            target_pose=near_pose,
            steps=self._config.APPROACH_NEAR_STEPS,
            stiffness=self._config.APPROACH_NEAR_STIFFNESS,
            damping=self._config.APPROACH_NEAR_DAMPING,
            baseline_force=baseline_force,
            get_observation=get_observation,
            move_robot=move_robot,
        )
        elapsed = (self.time_now() - start_time).nanoseconds / 1e9
        if failure is not None:
            return YoloApproachResult(False, port_position, elapsed, failure)

        self.get_logger().info(
            f"YOLO triangulation approach done: elapsed={elapsed:.2f}s"
        )
        return YoloApproachResult(
            success=True,
            port_position=port_position,
            elapsed_time=elapsed,
        )
