from __future__ import annotations

import math
import os
import re
import numpy as np

from typing import TYPE_CHECKING, Optional
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from final_policy.config import FinalPolicyConfig
from final_policy.geometry import (
    interp_profile,
    quat_to_tuple,
    rotate_vector_by_quat,
    tuple_to_quat,
)
from final_policy.model_store import (
    POSE_MODEL,
    SC_YOLO_MODEL,
    SFP_YOLO_MODEL,
    resolve_model_path,
)
from final_policy.vision import VisionPortEstimator

if TYPE_CHECKING:
    from final_policy.pose_prediction import PosePredictor


class FinalPolicy(Policy):
    """Final Policy driven by the unified pose prediction model.
    Stages:
    1. Port Detection
    2. Approach to Port
    3. yaw rotation and align
    4. insert
    """

    TARGET_CLASS_ID_SFP = 0
    TARGET_CLASS_ID_SC = 0

    def __init__(self, parent_node):
        Policy.__init__(self, parent_node)
        self._task: Optional[Task] = None
        self._sfp_yolo_model_path: Optional[str] = None
        self._sc_yolo_model_path: Optional[str] = None
        self._yolo_model_path = self._sfp_yolo_model_path
        self._models_ready = False
        self._cached_port_base: Optional[np.ndarray] = None
        self._target_orientation = None
        self._fixed_target_orientation = None
        self._sfp_yolo_conf_thresh = float(
            os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.5")
        )
        self._sc_yolo_conf_thresh = float(
            os.environ.get(
                "AIC_DEBUG_SC_YOLO_CONF_THRESH",
                os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.5"),
            )
        )
        self._yolo_conf_thresh = self._sfp_yolo_conf_thresh
        self._vision_by_port_type = {}
        self._vision_debug_save_enabled = False
        self._pose_model_path: Optional[str] = None
        self._pose_predictor: Optional[PosePredictor] = None
        self._send_feedback: Optional[SendFeedbackCallback] = None
        self.get_logger().info(
            "FinalPolicy ready: "
            "yolo_models=lazy, "
            "pose_model=lazy"
        )

    @staticmethod
    def _copy_pose(pose: Pose) -> Pose:
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

    @staticmethod
    def _copy_quaternion(quat: Quaternion) -> Quaternion:
        return Quaternion(
            x=float(quat.x),
            y=float(quat.y),
            z=float(quat.z),
            w=float(quat.w),
        )

    @staticmethod
    def _normalize_quat(q):
        values = np.asarray(q, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        values /= norm
        return tuple(float(v) for v in values)

    @staticmethod
    def _force_norm(observation) -> Optional[float]:
        if observation is None:
            return None
        force = observation.wrist_wrench.wrench.force
        return float(math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z))

    @staticmethod
    def _force_vector(observation) -> Optional[np.ndarray]:
        if observation is None:
            return None
        force = observation.wrist_wrench.wrench.force
        return np.array(
            [float(force.x), float(force.y), float(force.z)],
            dtype=np.float64,
        )

    @staticmethod
    def _tcp_pose(observation) -> Optional[Pose]:
        if observation is None:
            return None
        return FinalPolicy._copy_pose(observation.controller_state.tcp_pose)

    @staticmethod
    def _axis_angle_quat(axis: np.ndarray, angle_rad: float):
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        axis /= norm
        half = 0.5 * float(angle_rad)
        sin_half = float(math.sin(half))
        return FinalPolicy._normalize_quat(
            (
                float(math.cos(half)),
                float(axis[0] * sin_half),
                float(axis[1] * sin_half),
                float(axis[2] * sin_half),
            )
        )

    def _port_id(self) -> int:
        text = str(getattr(self._task, "port_name", "") or "")
        match = re.search(r"_(\d+)$", text)
        return int(match.group(1)) if match is not None else 0

    def _port_type(self) -> str:
        tokens = " ".join(
            str(value or "").lower()
            for value in (
                getattr(self._task, "plug_name", ""),
                getattr(self._task, "port_name", ""),
                getattr(self._task, "port_type", ""),
                getattr(self._task, "task_type", ""),
            )
        )
        return "sc" if "sc" in tokens else "sfp"

    def _target_class_id(self, port_type: str) -> int:
        if port_type == "sc":
            return int(
                os.environ.get("AIC_DEBUG_SC_TARGET_CLASS_ID", self.TARGET_CLASS_ID_SC)
            )
        return int(
            os.environ.get("AIC_DEBUG_SFP_TARGET_CLASS_ID", self.TARGET_CLASS_ID_SFP)
        )

    def _ensure_yolo_models_ready(
        self,
        send_feedback: Optional[SendFeedbackCallback] = None,
    ) -> bool:
        if self._models_ready:
            return True

        if send_feedback is not None:
            send_feedback("Final Policy: prepare_models")
        self.get_logger().info("FinalPolicy model preparation start")

        if self._sfp_yolo_model_path is None:
            if send_feedback is not None:
                send_feedback("Final Policy: preparing SFP YOLO model")
            self._sfp_yolo_model_path = resolve_model_path(
                SFP_YOLO_MODEL,
                logger=self.get_logger(),
            )

        if self._sc_yolo_model_path is None:
            if send_feedback is not None:
                send_feedback("Final Policy: preparing SC YOLO model")
            self._sc_yolo_model_path = resolve_model_path(
                SC_YOLO_MODEL,
                logger=self.get_logger(),
            )

        self._yolo_model_path = self._sfp_yolo_model_path
        self._models_ready = True
        self.get_logger().info(
            "FinalPolicy model preparation done: "
            f"sfp_yolo={self._sfp_yolo_model_path}, "
            f"sc_yolo={self._sc_yolo_model_path}"
        )
        if send_feedback is not None:
            send_feedback("Final Policy: Models are ready")
        return True

    def _vision_for_port_type(self, port_type: str) -> VisionPortEstimator:
        self._ensure_yolo_models_ready()
        port_type = "sc" if port_type == "sc" else "sfp"
        if port_type not in self._vision_by_port_type:
            model_path = (
                self._sc_yolo_model_path
                if port_type == "sc"
                else self._sfp_yolo_model_path
            )
            conf_thresh = (
                self._sc_yolo_conf_thresh
                if port_type == "sc"
                else self._sfp_yolo_conf_thresh
            )
            self.get_logger().info(
                f"Loading {port_type.upper()} YOLO model: {model_path}"
            )
            vision = VisionPortEstimator(
                model_path=model_path,
                conf_thresh=conf_thresh,
                logger=self.get_logger(),
                debug_save_enabled=self._vision_debug_save_enabled,
                auto_start=False,
            )
            self._vision_by_port_type[port_type] = vision
        return self._vision_by_port_type[port_type]

    def _initial_approach_z_offset(self) -> float:
        if self._port_type() == "sc":
            return float(FinalPolicyConfig.APPROACH_Z_OFFSET_SC_M)
        return float(FinalPolicyConfig.APPROACH_Z_OFFSET_SFP_M)

    def _manual_rotation_deg(self) -> float:
        if self._port_type() == "sc":
            return float(FinalPolicyConfig.APPROACH_SC_MANUAL_ROTATION_DEG)
        return float(FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_DEG)

    def _insertion_stiffness(self) -> tuple:
        if self._port_type() == "sc":
            return FinalPolicyConfig.SC_INSERTION_STIFFNESS
        return FinalPolicyConfig.SFP_INSERTION_STIFFNESS

    def _insertion_damping(self) -> tuple:
        if self._port_type() == "sc":
            return FinalPolicyConfig.SC_INSERTION_DAMPING
        return FinalPolicyConfig.SFP_INSERTION_DAMPING

    def _align_correction_base(self, offset_m: np.ndarray) -> np.ndarray:
        offset = np.asarray(offset_m, dtype=np.float64)
        return np.array(
            [
                float(FinalPolicyConfig.ALIGN_CORRECTION_X_SIGN) * offset[0],
                float(FinalPolicyConfig.ALIGN_CORRECTION_Y_SIGN) * offset[1],
                -offset[2],
            ],
            dtype=np.float64,
        )

    def _align_retry_step_base(
        self,
        *,
        tcp_pose: Pose,
        force_delta: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if force_delta is None or not FinalPolicyConfig.ALIGN_RETRY_ENABLED:
            return None

        force_delta = np.asarray(force_delta, dtype=np.float64)
        xy_force = np.array(
            [
                float(FinalPolicyConfig.ALIGN_RETRY_FORCE_X_SIGN)
                * force_delta[0],
                float(FinalPolicyConfig.ALIGN_RETRY_FORCE_Y_SIGN)
                * force_delta[1],
                0.0,
            ],
            dtype=np.float64,
        )
        xy_norm = float(np.linalg.norm(xy_force[:2]))
        z_abs = abs(float(force_delta[2]))

        retry_step = np.zeros(3, dtype=np.float64)
        if xy_norm > float(FinalPolicyConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N):
            lateral = xy_force / max(xy_norm, 1e-9)
            lateral *= float(FinalPolicyConfig.ALIGN_RETRY_LATERAL_STEP_M)
            if FinalPolicyConfig.ALIGN_RETRY_USE_TCP_FRAME:
                lateral = rotate_vector_by_quat(lateral, quat_to_tuple(tcp_pose.orientation))
            retry_step[:2] = lateral[:2]

        if z_abs > float(FinalPolicyConfig.ALIGN_RETRY_FORCE_Z_THRESHOLD_N):
            retry_step[2] = float(FinalPolicyConfig.ALIGN_RETRY_LIFT_M)
        elif xy_norm > float(FinalPolicyConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N):
            retry_step[2] = 0.5 * float(FinalPolicyConfig.ALIGN_RETRY_LIFT_M)

        if float(np.linalg.norm(retry_step)) < 1e-9:
            return None
        return retry_step

    def _axis(self, pose: Pose) -> np.ndarray:
        axis_name = str(FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS)
        base_axes = {
            "base_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "base_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "base_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if axis_name in base_axes:
            return base_axes[axis_name]

        local_axes = {
            "tcp_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "tcp_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "tcp_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        local_axis = local_axes.get(axis_name, local_axes["tcp_z"])
        q = quat_to_tuple(pose.orientation)
        rotated = quaternion_multiply(
            quaternion_multiply(q, (0.0, *local_axis)),
            (q[0], -q[1], -q[2], -q[3]),
        )
        return np.array([rotated[1], rotated[2], rotated[3]], dtype=np.float64)

    def _follow_pose(
        self,
        *,
        move_robot,
        start_pose: Pose,
        target_pose: Pose,
        steps: int,
        stiffness: tuple,
        damping: tuple,
        dt: float,
        label: str,
    ) -> None:
        start = np.array(
            [start_pose.position.x, start_pose.position.y, start_pose.position.z],
            dtype=np.float64,
        )
        target = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z],
            dtype=np.float64,
        )
        q_start = quat_to_tuple(start_pose.orientation)
        q_target = quat_to_tuple(target_pose.orientation)

        step_count = max(1, int(steps))
        for index in range(step_count):
            t = interp_profile((index + 1) / step_count, quintic=True)
            pos = start * (1.0 - t) + target * t
            quat = quaternion_slerp(q_start, q_target, t)
            pose = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=tuple_to_quat(quat),
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=list(stiffness),
                damping=list(damping),
            )
            if index == 0 or index == step_count - 1:
                self.get_logger().info(
                    f"{label}: waypoint {index + 1}/{step_count} "
                    f"tcp=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
                )
            self.sleep_for(dt)

    def _target_wrist_orientation(self, start_pose: Pose) -> Quaternion:
        if self._fixed_target_orientation is not None:
            return self._copy_quaternion(self._fixed_target_orientation)

        angle_deg = self._manual_rotation_deg()
        if abs(angle_deg) < 1e-9:
            self._fixed_target_orientation = self._copy_quaternion(start_pose.orientation)
            return self._copy_quaternion(self._fixed_target_orientation)

        q_delta = self._axis_angle_quat(self._axis(start_pose), math.radians(angle_deg))
        q_target = self._normalize_quat(
            quaternion_multiply(q_delta, quat_to_tuple(start_pose.orientation))
        )
        self._fixed_target_orientation = tuple_to_quat(q_target)
        return self._copy_quaternion(self._fixed_target_orientation)

    def _estimate_port(self, get_observation) -> Optional[np.ndarray]:
        port_hint = str(getattr(self._task, "port_name", "") or "")
        target_module_name = str(getattr(self._task, "target_module_name", "") or "")
        port_type = self._port_type()
        target_class_id = self._target_class_id(port_type)
        vision = self._vision_for_port_type(port_type)
        for attempt in range(FinalPolicyConfig.APPROACH_VISION_RETRIES):
            obs = get_observation()
            port = vision.estimate(
                obs,
                target_class_id,
                port_hint=port_hint,
                target_module_name=target_module_name,
            )
            if port is not None:
                self.get_logger().info(
                    "YOLO port estimate: "
                    f"attempt={attempt + 1}, "
                    f"type={port_type}, "
                    f"target={target_module_name}, "
                    f"port={port_hint}, "
                    f"class_id={target_class_id}, "
                    f"base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
                )
                return port
            self.sleep_for(FinalPolicyConfig.APPROACH_RETRY_DT)
        return None

    def _stage_lift_up(self, get_observation, move_robot) -> bool:
        lift_m = float(FinalPolicyConfig.INITIAL_LIFT_M)
        self.get_logger().info(
            f"[stage 1/5] lift_up Start: dz={lift_m * 1000.0:.1f}mm"
        )
        if abs(lift_m) < 1e-9:
            self.get_logger().info("lift_up skipped: configured dz is 0")
            return True

        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("lift_up failed: missing TCP pose")
            return False

        target_pose = self._copy_pose(start_pose)
        target_pose.position.z += lift_m
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=target_pose,
            steps=FinalPolicyConfig.INITIAL_LIFT_STEPS,
            stiffness=FinalPolicyConfig.APPROACH_NEAR_STIFFNESS,
            damping=FinalPolicyConfig.APPROACH_NEAR_DAMPING,
            dt=FinalPolicyConfig.INITIAL_LIFT_DT,
            label="lift_up",
        )
        if FinalPolicyConfig.INITIAL_LIFT_SETTLE_S > 0:
            self.get_logger().info(
                f"lift_up settle: {FinalPolicyConfig.INITIAL_LIFT_SETTLE_S:.2f}s"
            )
            self.sleep_for(FinalPolicyConfig.INITIAL_LIFT_SETTLE_S)
        self.get_logger().info("[stage 1/5] lift_up Done")
        return True

    def _stage_detect(self, get_observation) -> bool:
        self.get_logger().info("[stage 2/5] Detection Start")
        self._vision_debug_save_enabled = True
        vision = self._vision_for_port_type(self._port_type())
        vision.set_debug_task_context(
            target_module_name=str(getattr(self._task, "target_module_name", "") or ""),
            port_name=str(getattr(self._task, "port_name", "") or ""),
            plug_name=str(getattr(self._task, "plug_name", "") or ""),
            cable_name=str(getattr(self._task, "cable_name", "") or ""),
            port_type=self._port_type(),
        )
        vision.start_detection(enable_debug_save=True, reset_counts=True)
        try:
            obs = get_observation()
            start_pose = self._tcp_pose(obs)
            if start_pose is None:
                self.get_logger().error("Detection failed: Missing TCP pose")
                return False

            port = self._estimate_port(get_observation)
            if port is None:
                self.get_logger().error("Detection failed: YOLO port estimate unavailable")
                return False

            self._cached_port_base = port
            self._target_orientation = self._target_wrist_orientation(start_pose)
            self.get_logger().info(
                "Detection cached: "
                f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
                f"axis={FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
                f"angle={self._manual_rotation_deg():+.2f}deg"
            )
            self.get_logger().info("[stage 2/5] Detection Done")
            return True
        finally:
            for estimator in self._vision_by_port_type.values():
                estimator.stop_detection()
                estimator.set_debug_save_enabled(False)
            self._vision_debug_save_enabled = False

    def _stage_approach(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 3/5] Approach Start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("Approach failed: missing TCP pose")
            return False

        port = self._cached_port_base
        if port is None:
            self.get_logger().error("Approach failed: missing cached YOLO port estimate")
            return False

        target_orientation = self._target_orientation
        if target_orientation is None:
            target_orientation = self._target_wrist_orientation(start_pose)
            self._target_orientation = target_orientation

        tcp_offset = np.array(
            [
                FinalPolicyConfig.APPROACH_TCP_OFFSET_X_M,
                FinalPolicyConfig.APPROACH_TCP_OFFSET_Y_M,
                FinalPolicyConfig.APPROACH_TCP_OFFSET_Z_M,
            ],
            dtype=np.float64,
        )
        target_z_offset = float(FinalPolicyConfig.APPROACH_NEAR_Z_OFFSET_M)

        def make_approach_pose(z_offset: float) -> tuple[Pose, np.ndarray]:
            target = port + np.array([0.0, 0.0, z_offset], dtype=np.float64)
            target = target + tcp_offset
            return (
                Pose(
                    position=Point(
                        x=float(target[0]),
                        y=float(target[1]),
                        z=float(target[2]),
                    ),
                    orientation=self._copy_quaternion(target_orientation),
                ),
                target,
            )

        approach_pose, approach_target = make_approach_pose(target_z_offset)
        self.get_logger().info(
            "approach target: "
            f"z_plus={target_z_offset*1000:.1f}mm, "
            f"tcp_offset=({tcp_offset[0]*1000:+.1f}, "
            f"{tcp_offset[1]*1000:+.1f}, {tcp_offset[2]*1000:+.1f})mm, "
            f"target_tcp=({approach_target[0]:+.4f}, "
            f"{approach_target[1]:+.4f}, {approach_target[2]:+.4f})"
        )
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=approach_pose,
            steps=FinalPolicyConfig.APPROACH_STEPS,
            stiffness=FinalPolicyConfig.APPROACH_STIFFNESS,
            damping=FinalPolicyConfig.APPROACH_DAMPING,
            dt=FinalPolicyConfig.APPROACH_DT,
            label="approach",
        )
        if FinalPolicyConfig.APPROACH_SETTLE_S > 0:
            self.get_logger().info(
                f"approach settle: {FinalPolicyConfig.APPROACH_SETTLE_S:.2f}s"
            )
            self.sleep_for(FinalPolicyConfig.APPROACH_SETTLE_S)
        self.get_logger().info("[stage 3/5] Approach Done")
        return True

    def _yaw_axis(self, pose) -> np.ndarray:
        q = quat_to_tuple(pose.orientation)
        rotated = quaternion_multiply(
            quaternion_multiply(q, (0.0, 0.0, 0.0, 1.0)),
            (q[0], -q[1], -q[2], -q[3]),
        )
        axis = np.array([rotated[1], rotated[2], rotated[3]], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return axis / norm

    def _select_offset(self, prediction: dict[str, object]) -> np.ndarray:
        key = "port1_position_m" if self._port_id() == 1 else "port0_position_m"
        return np.asarray(prediction[key], dtype=np.float64)

    def _pose_predictor_for_align(self) -> PosePredictor:
        if self._pose_predictor is None:
            from final_policy.pose_prediction import PosePredictor

            if self._send_feedback is not None:
                self._send_feedback("Final Policy: preparing pose model")
            self._pose_model_path = resolve_model_path(
                POSE_MODEL,
                logger=self.get_logger(),
            )
            self._pose_predictor = PosePredictor(
                checkpoint_path=self._pose_model_path,
                logger=self.get_logger(),
            )
            if self._send_feedback is not None:
                self._send_feedback("Final Policy: pose model ready")
        return self._pose_predictor

    def _stage_yaw_align(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 4/5] Yaw Alignment Start")
        pose_predictor = self._pose_predictor_for_align()
        baseline_force = self._force_vector(get_observation())
        stable_count = 0
        last_xy = None
        last_yaw = None

        for step in range(FinalPolicyConfig.ALIGN_MAX_STEPS):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if tcp_pose is None:
                self.sleep_for(FinalPolicyConfig.DT)
                continue

            force = self._force_vector(obs)
            force_delta = None
            if force is not None and baseline_force is not None:
                force_delta = force - baseline_force
            retry_step = self._align_retry_step_base(
                tcp_pose=tcp_pose,
                force_delta=force_delta,
            )
            if retry_step is not None:
                target_pose = self._copy_pose(tcp_pose)
                target_pose.position.x += float(retry_step[0])
                target_pose.position.y += float(retry_step[1])
                target_pose.position.z += float(retry_step[2])
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=target_pose,
                    stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                    damping=list(FinalPolicyConfig.ALIGN_DAMPING),
                )
                stable_count = 0
                self.sleep_for(FinalPolicyConfig.COMMAND_SETTLE_S)
                continue

            prediction = pose_predictor.predict(obs)
            if prediction is None:
                self.sleep_for(FinalPolicyConfig.DT)
                continue

            offset_m = self._select_offset(prediction)
            dyaw = float(prediction["dyaw_rad"])
            correction_base = self._align_correction_base(offset_m)
            xy_error = float(np.linalg.norm(offset_m[:2]))
            yaw_error = abs(dyaw)
            last_xy = xy_error
            last_yaw = yaw_error

            if xy_error < FinalPolicyConfig.XY_TOL_M and yaw_error < FinalPolicyConfig.YAW_TOL_RAD:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= FinalPolicyConfig.STABLE_STEPS:
                self.get_logger().info(
                    "yaw_align stable: "
                    f"xy={xy_error * 1000.0:.2f}mm, "
                    f"dyaw={math.degrees(yaw_error):.2f}deg"
                )
                return True

            step_xy = np.clip(
                correction_base[:2] * FinalPolicyConfig.XY_GAIN,
                -FinalPolicyConfig.MAX_XY_STEP_M,
                FinalPolicyConfig.MAX_XY_STEP_M,
            )
            yaw_step = float(
                np.clip(
                    dyaw * FinalPolicyConfig.YAW_GAIN,
                    -FinalPolicyConfig.MAX_YAW_STEP_RAD,
                    FinalPolicyConfig.MAX_YAW_STEP_RAD,
                )
            )

            target_pose = self._copy_pose(tcp_pose)
            target_pose.position.x += float(step_xy[0])
            target_pose.position.y += float(step_xy[1])
            q_delta = self._axis_angle_quat(self._yaw_axis(tcp_pose), yaw_step)
            q_target = self._normalize_quat(
                quaternion_multiply(q_delta, quat_to_tuple(tcp_pose.orientation))
            )
            target_pose.orientation = tuple_to_quat(q_target)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                damping=list(FinalPolicyConfig.ALIGN_DAMPING),
            )
            self.get_logger().info(
                f"yaw_align[{step:03d}]: "
                f"offset=({offset_m[0]*1000:+.2f}, {offset_m[1]*1000:+.2f}, {offset_m[2]*1000:+.2f})mm, "
                f"cmd_xy=({step_xy[0]*1000:+.2f}, {step_xy[1]*1000:+.2f})mm, "
                f"dyaw={math.degrees(dyaw):+.2f}deg, "
                f"cmd_yaw={math.degrees(yaw_step):+.2f}deg, "
                f"stable={stable_count}/{FinalPolicyConfig.STABLE_STEPS}"
            )
            self.sleep_for(FinalPolicyConfig.COMMAND_SETTLE_S)

        if last_xy is None or last_yaw is None:
            self.get_logger().error("yaw_align failed: no pose predictions")
            return False
        success = (
            last_xy < FinalPolicyConfig.XY_TOL_M * 1.5
            and last_yaw < FinalPolicyConfig.YAW_TOL_RAD * 1.5
        )
        self.get_logger().info(
            f"[stage 4/5] yaw_rotation_align done: success={success}, "
            f"last_xy={last_xy * 1000.0:.2f}mm, "
            f"last_dyaw={math.degrees(last_yaw):.2f}deg"
        )
        return success

    def _insert_force_blocked(
        self,
        *,
        force_norm: Optional[float],
        baseline_norm: Optional[float],
    ) -> tuple[bool, str]:
        if force_norm is None or baseline_norm is None:
            return False, ""
        delta = force_norm - baseline_norm
        drop = baseline_norm - force_norm
        if drop > FinalPolicyConfig.INSERT_FORCE_DROP_LIMIT_N:
            return True, (
                f"force_drop={drop:.2f}N "
                f"(force={force_norm:.2f}N, baseline={baseline_norm:.2f}N)"
            )
        if delta > FinalPolicyConfig.INSERT_FORCE_RISE_LIMIT_N:
            return True, (
                f"force_rise={delta:.2f}N "
                f"(force={force_norm:.2f}N, baseline={baseline_norm:.2f}N)"
            )
        return False, ""

    def _stage_insert(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 5/5] insert start")
        obs = get_observation()
        pose = self._tcp_pose(obs)
        if pose is None:
            self.get_logger().error("insert failed: missing TCP pose")
            return False

        baseline_norm = self._force_norm(obs)
        baseline_vector = self._force_vector(obs)
        if baseline_norm is None:
            self.get_logger().warn("insert force baseline unavailable; force guard disabled")
        else:
            self.get_logger().info(
                "insert force baseline: "
                f"{baseline_norm:.2f}N, "
                f"drop_limit={FinalPolicyConfig.INSERT_FORCE_DROP_LIMIT_N:.2f}N, "
                f"rise_limit={FinalPolicyConfig.INSERT_FORCE_RISE_LIMIT_N:.2f}N"
            )

        max_depth = float(FinalPolicyConfig.MAX_INSERT_DEPTH_M)
        step_m = min(
            float(FinalPolicyConfig.INSERT_STEP_M),
            float(FinalPolicyConfig.MAX_DOWN_STEP_M),
        )
        max_steps = min(
            int(math.ceil(max_depth / max(step_m, 1e-6))),
            int(FinalPolicyConfig.INSERT_MAX_STEPS),
        )
        start_z = float(pose.position.z)
        inserted_steps = 0
        retries = 0

        while inserted_steps < max_steps:
            obs = get_observation()
            current = self._tcp_pose(obs) or pose
            force_norm = self._force_norm(obs)
            blocked, reason = self._insert_force_blocked(
                force_norm=force_norm,
                baseline_norm=baseline_norm,
            )
            if blocked:
                retries += 1
                if retries > FinalPolicyConfig.INSERT_RETRY_MAX:
                    self.get_logger().error(
                        "insert failed: retry limit exceeded after "
                        f"{FinalPolicyConfig.INSERT_RETRY_MAX} retries; {reason}"
                    )
                    return False

                force_vector = self._force_vector(obs)
                force_delta = None
                if force_vector is not None and baseline_vector is not None:
                    force_delta = force_vector - baseline_vector
                retry_step = self._align_retry_step_base(
                    tcp_pose=current,
                    force_delta=force_delta,
                )
                if retry_step is None:
                    retry_step = np.array(
                        [0.0, 0.0, FinalPolicyConfig.INSERT_RETRY_LIFT_M],
                        dtype=np.float64,
                    )
                retry_step[2] = max(
                    float(retry_step[2]),
                    float(FinalPolicyConfig.INSERT_RETRY_LIFT_M),
                )

                target_pose = self._copy_pose(current)
                target_pose.position.x += float(retry_step[0])
                target_pose.position.y += float(retry_step[1])
                target_pose.position.z += float(retry_step[2])
                if self._target_orientation is not None:
                    target_pose.orientation = self._copy_quaternion(self._target_orientation)
                self.get_logger().warn(
                    "insert retry: "
                    f"{reason}, retry={retries}/{FinalPolicyConfig.INSERT_RETRY_MAX}, "
                    f"step=({retry_step[0]*1000:+.2f}, {retry_step[1]*1000:+.2f}, "
                    f"{retry_step[2]*1000:+.2f})mm"
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=target_pose,
                    stiffness=list(self._insertion_stiffness()),
                    damping=list(self._insertion_damping()),
                )
                self.sleep_for(FinalPolicyConfig.INSERT_RETRY_SETTLE_S)
                continue

            target_pose = self._copy_pose(current)
            target_pose.position.z = float(start_z - (inserted_steps + 1) * step_m)
            if self._target_orientation is not None:
                target_pose.orientation = self._copy_quaternion(self._target_orientation)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(self._insertion_stiffness()),
                damping=list(self._insertion_damping()),
            )
            if inserted_steps == 0 or inserted_steps % 10 == 0:
                force_text = ""
                if force_norm is not None and baseline_norm is not None:
                    force_text = f", force_delta={force_norm - baseline_norm:+.2f}N"
                self.get_logger().info(
                    f"insert[{inserted_steps:03d}]: "
                    f"dz={-(inserted_steps + 1) * step_m * 1000.0:.1f}mm"
                    f"{force_text}"
                )
            inserted_steps += 1
            self.sleep_for(FinalPolicyConfig.INSERT_DT)

        if FinalPolicyConfig.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(FinalPolicyConfig.SETTLE_AFTER_INSERT_S)
        self.get_logger().info("[stage 5/5] insert done")
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._task = task
        self._send_feedback = send_feedback
        self._cached_port_base = None
        self._target_orientation = None
        self._fixed_target_orientation = None
        self.get_logger().info(
            "FinalPolicy Start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )
        try:
            self._ensure_yolo_models_ready(send_feedback)
        except Exception as exc:
            self.get_logger().error(f"FinalPolicy model preparation failed: {exc}")
            send_feedback("failed: prepare_models exception")
            return False

        stages = (
            ("lift_up", lambda: self._stage_lift_up(get_observation, move_robot)),
            ("detect", lambda: self._stage_detect(get_observation)),
            ("approach", lambda: self._stage_approach(get_observation, move_robot)),
            # ("yaw_rotation_align", lambda: self._stage_yaw_align(get_observation, move_robot)),
            # ("insert", lambda: self._stage_insert(get_observation, move_robot)),
        )
        for name, stage in stages:
            send_feedback(f"Final Policy: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"FinalPolicy failed at stage: {name}")
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"FinalPolicy exception at {name}: {exc}")
                send_feedback(f"failed: {name} exception")
                return False
        send_feedback("Final Policy: done")
        self.get_logger().info("FinalPolicy done")
        return True
