"""Minimal SFP/SC policy for debugging YOLO approach and distance alignment."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from distance_prediction_policy.config import DistancePredictionConfig
from distance_prediction_policy.model_feedback import VisionOffsetPredictor
from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.geometry import (
    interp_profile,
    quat_to_tuple,
    rotate_vector_by_quat,
    tuple_to_quat,
)
from motion_planning_node.core.vision import VisionPortEstimator


SRC_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SFP_YOLO_MODEL_PATHS = (
    SRC_ROOT
    / "model"
    / "yolo-port-keypoint-detection"
    / "approach"
    / "SFP"
    / "weights"
    / "best.pt",
    SRC_ROOT / "model" / "ais_yolo" / "approach" / "SFP" / "weights" / "best.pt",
)
DEFAULT_SC_YOLO_MODEL_PATHS = (
    SRC_ROOT
    / "model"
    / "yolo-port-keypoint-detection"
    / "approach"
    / "SC"
    / "weights"
    / "best.pt",
    SRC_ROOT / "model" / "ais_yolo" / "approach" / "SC" / "weights" / "best.pt",
)


def _first_existing_model_path(paths: tuple[Path, ...]) -> Optional[str]:
    for path in paths:
        if path.is_file():
            return str(path)
    return None


def _resolve_sfp_yolo_model_path() -> str:
    env_path = os.environ.get("AIC_SFP_YOLO_MODEL_PATH")
    if env_path:
        return env_path
    default_path = _first_existing_model_path(DEFAULT_SFP_YOLO_MODEL_PATHS)
    if default_path is not None:
        return default_path
    return Stage1Config.DETECTION_MODEL_PATH


def _resolve_sc_yolo_model_path() -> str:
    env_path = os.environ.get("AIC_SC_YOLO_MODEL_PATH")
    if env_path:
        return env_path
    default_path = _first_existing_model_path(DEFAULT_SC_YOLO_MODEL_PATHS)
    if default_path is not None:
        return default_path
    return str(DEFAULT_SC_YOLO_MODEL_PATHS[0])


class DebugSfpDistancePolicy(Policy):
    """Four-stage policy with explicit logs and no ground-truth dependency.

    Stages:
    1. initial_lift: move upward before detection.
    2. detect: YOLO triangulation before changing wrist pose.
    3. approach: move to port z + clearance while rotating to insertion pose.
    4. align: distance model adjusts XY only while holding wrist pose.
    5. insert: blind downward motion while holding wrist pose.
    """

    TARGET_CLASS_ID_SFP = 0
    TARGET_CLASS_ID_SC = 0

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task: Optional[Task] = None
        self._sfp_yolo_model_path = _resolve_sfp_yolo_model_path()
        self._sc_yolo_model_path = _resolve_sc_yolo_model_path()
        self._yolo_model_path = self._sfp_yolo_model_path
        self._cached_port_base: Optional[np.ndarray] = None
        self._target_orientation: Optional[Quaternion] = None
        self._fixed_target_orientation: Optional[Quaternion] = None
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

        self._vision = self._vision_for_port_type("sfp")
        self._distance = VisionOffsetPredictor(logger=self.get_logger())

        self.get_logger().info(
            "DebugSfpDistancePolicy ready: "
            f"sfp_yolo={self._sfp_yolo_model_path}, "
            f"sc_yolo={self._sc_yolo_model_path}, "
            f"distance={DistancePredictionConfig.CHECKPOINT_PATH}, "
            f"sfp_yolo_conf_thresh={self._sfp_yolo_conf_thresh}, "
            f"sc_yolo_conf_thresh={self._sc_yolo_conf_thresh}"
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
        return DebugSfpDistancePolicy._copy_pose(observation.controller_state.tcp_pose)

    @staticmethod
    def _axis_angle_quat(axis: np.ndarray, angle_rad: float):
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        axis /= norm
        half = 0.5 * float(angle_rad)
        sin_half = float(math.sin(half))
        return DebugSfpDistancePolicy._normalize_quat(
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

    def _vision_for_port_type(self, port_type: str) -> VisionPortEstimator:
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
            )
            vision._ensure_loaded()
            self._vision_by_port_type[port_type] = vision
        return self._vision_by_port_type[port_type]

    def _initial_approach_z_offset(self) -> float:
        if self._port_type() == "sc":
            return float(DistancePredictionConfig.APPROACH_Z_OFFSET_SC_M)
        return float(DistancePredictionConfig.APPROACH_Z_OFFSET_SFP_M)

    def _manual_rotation_deg(self) -> float:
        if self._port_type() == "sc":
            return float(DistancePredictionConfig.APPROACH_SC_MANUAL_ROTATION_DEG)
        return float(DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_DEG)

    def _insertion_stiffness(self) -> tuple:
        if self._port_type() == "sc":
            return DistancePredictionConfig.SC_INSERTION_STIFFNESS
        return DistancePredictionConfig.SFP_INSERTION_STIFFNESS

    def _insertion_damping(self) -> tuple:
        if self._port_type() == "sc":
            return DistancePredictionConfig.SC_INSERTION_DAMPING
        return DistancePredictionConfig.SFP_INSERTION_DAMPING

    def _align_correction_base(self, offset_m: np.ndarray) -> np.ndarray:
        offset = np.asarray(offset_m, dtype=np.float64)
        # Evaluation policy must not query target-port TF. Interpret the learned
        # offset with fixed, tunable signs in the command/base frame.
        return np.array(
            [
                float(DistancePredictionConfig.ALIGN_CORRECTION_X_SIGN) * offset[0],
                float(DistancePredictionConfig.ALIGN_CORRECTION_Y_SIGN) * offset[1],
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
        if force_delta is None or not DistancePredictionConfig.ALIGN_RETRY_ENABLED:
            return None

        force_delta = np.asarray(force_delta, dtype=np.float64)
        xy_force = np.array(
            [
                float(DistancePredictionConfig.ALIGN_RETRY_FORCE_X_SIGN)
                * force_delta[0],
                float(DistancePredictionConfig.ALIGN_RETRY_FORCE_Y_SIGN)
                * force_delta[1],
                0.0,
            ],
            dtype=np.float64,
        )
        xy_norm = float(np.linalg.norm(xy_force[:2]))
        z_abs = abs(float(force_delta[2]))

        retry_step = np.zeros(3, dtype=np.float64)
        if xy_norm > float(DistancePredictionConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N):
            lateral = xy_force / max(xy_norm, 1e-9)
            lateral *= float(DistancePredictionConfig.ALIGN_RETRY_LATERAL_STEP_M)
            if DistancePredictionConfig.ALIGN_RETRY_USE_TCP_FRAME:
                lateral = rotate_vector_by_quat(lateral, quat_to_tuple(tcp_pose.orientation))
            retry_step[:2] = lateral[:2]

        if z_abs > float(DistancePredictionConfig.ALIGN_RETRY_FORCE_Z_THRESHOLD_N):
            retry_step[2] = float(DistancePredictionConfig.ALIGN_RETRY_LIFT_M)
        elif xy_norm > float(DistancePredictionConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N):
            retry_step[2] = 0.5 * float(DistancePredictionConfig.ALIGN_RETRY_LIFT_M)

        if float(np.linalg.norm(retry_step)) < 1e-9:
            return None
        return retry_step

    def _axis(self, pose: Pose) -> np.ndarray:
        axis_name = str(DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS)
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

        for index in range(max(1, int(steps))):
            t = interp_profile((index + 1) / max(1, int(steps)), quintic=True)
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
            if index == 0 or index == steps - 1:
                self.get_logger().info(
                    f"{label}: waypoint {index + 1}/{steps} "
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

    def _stage_initial_lift(self, get_observation, move_robot) -> bool:
        if self._port_type() == "sc":
            self.get_logger().info("[stage 1/5] initial_lift skipped for SC task")
            return True

        lift_m = float(DistancePredictionConfig.INITIAL_LIFT_M)
        self.get_logger().info(
            f"[stage 1/5] initial_lift start: dz={lift_m * 1000.0:.1f}mm"
        )
        if abs(lift_m) < 1e-9:
            self.get_logger().info("initial_lift skipped: configured dz is 0")
            return True

        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("initial_lift failed: missing TCP pose")
            return False

        target_pose = self._copy_pose(start_pose)
        target_pose.position.z += lift_m
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=target_pose,
            steps=DistancePredictionConfig.INITIAL_LIFT_STEPS,
            stiffness=DistancePredictionConfig.APPROACH_NEAR_STIFFNESS,
            damping=DistancePredictionConfig.APPROACH_NEAR_DAMPING,
            dt=DistancePredictionConfig.INITIAL_LIFT_DT,
            label="initial_lift",
        )
        if DistancePredictionConfig.INITIAL_LIFT_SETTLE_S > 0:
            self.get_logger().info(
                "initial_lift settle: "
                f"{DistancePredictionConfig.INITIAL_LIFT_SETTLE_S:.2f}s"
            )
            self.sleep_for(DistancePredictionConfig.INITIAL_LIFT_SETTLE_S)
        self.get_logger().info("[stage 1/5] initial_lift done")
        return True

    def _stage_detect(self, get_observation) -> bool:
        self.get_logger().info("[stage 2/5] detect start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("detect failed: missing TCP pose")
            return False

        port = self._estimate_port(get_observation)
        if port is None:
            self.get_logger().error("detect failed: YOLO port estimate unavailable")
            return False

        self._cached_port_base = port
        self._target_orientation = self._target_wrist_orientation(start_pose)
        self.get_logger().info(
            "detect cached: "
            f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
            f"axis={DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
            f"angle={self._manual_rotation_deg():+.2f}deg"
        )
        self.get_logger().info("[stage 2/5] detect done")
        return True

    def _estimate_port(self, get_observation) -> Optional[np.ndarray]:
        port_hint = str(getattr(self._task, "port_name", "") or "")
        port_type = self._port_type()
        target_class_id = self._target_class_id(port_type)
        vision = self._vision_for_port_type(port_type)
        for attempt in range(DistancePredictionConfig.APPROACH_VISION_RETRIES):
            obs = get_observation()
            port = vision.estimate(
                obs,
                target_class_id,
                port_hint=port_hint,
            )
            if port is not None:
                self.get_logger().info(
                    "YOLO port estimate: "
                    f"attempt={attempt + 1}, "
                    f"type={port_type}, "
                    f"class_id={target_class_id}, "
                    f"base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
                )
                return port
            self.sleep_for(DistancePredictionConfig.APPROACH_RETRY_DT)
        return None

    def _stage_approach(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 3/5] approach start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("approach failed: missing TCP pose")
            return False

        port = self._cached_port_base
        if port is None:
            self.get_logger().error("approach failed: missing cached YOLO port estimate")
            return False

        target_orientation = self._target_orientation
        if target_orientation is None:
            target_orientation = self._target_wrist_orientation(start_pose)
            self._target_orientation = target_orientation

        tcp_offset = np.array(
            [
                DistancePredictionConfig.APPROACH_TCP_OFFSET_X_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Y_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Z_M,
            ],
            dtype=np.float64,
        )
        initial_z_offset = self._initial_approach_z_offset()
        near_z_offset = float(DistancePredictionConfig.APPROACH_NEAR_Z_OFFSET_M)

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

        far_pose, far_target = make_approach_pose(initial_z_offset)
        near_pose, near_target = make_approach_pose(near_z_offset)
        self.get_logger().info(
            "approach targets: "
            f"initial_z_plus={initial_z_offset*1000:.1f}mm, "
            f"near_z_plus={near_z_offset*1000:.1f}mm, "
            f"tcp_offset=({tcp_offset[0]*1000:+.1f}, "
            f"{tcp_offset[1]*1000:+.1f}, {tcp_offset[2]*1000:+.1f})mm, "
            f"far_tcp=({far_target[0]:+.4f}, {far_target[1]:+.4f}, {far_target[2]:+.4f}), "
            f"near_tcp=({near_target[0]:+.4f}, {near_target[1]:+.4f}, {near_target[2]:+.4f})"
        )
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=far_pose,
            steps=DistancePredictionConfig.APPROACH_STEPS,
            stiffness=DistancePredictionConfig.APPROACH_STIFFNESS,
            damping=DistancePredictionConfig.APPROACH_DAMPING,
            dt=DistancePredictionConfig.APPROACH_DT,
            label="approach_far",
        )

        near_start_obs = get_observation()
        near_start_pose = self._tcp_pose(near_start_obs) or far_pose
        self._follow_pose(
            move_robot=move_robot,
            start_pose=near_start_pose,
            target_pose=near_pose,
            steps=DistancePredictionConfig.APPROACH_NEAR_STEPS,
            stiffness=DistancePredictionConfig.APPROACH_NEAR_STIFFNESS,
            damping=DistancePredictionConfig.APPROACH_NEAR_DAMPING,
            dt=DistancePredictionConfig.APPROACH_DT,
            label="approach_near",
        )
        if DistancePredictionConfig.APPROACH_SETTLE_S > 0:
            self.get_logger().info(
                f"approach settle: {DistancePredictionConfig.APPROACH_SETTLE_S:.2f}s"
            )
            self.sleep_for(DistancePredictionConfig.APPROACH_SETTLE_S)
        self.get_logger().info("[stage 3/5] approach done")
        return True

    def _stage_align(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 4/5] align start")
        stable_count = 0
        last_xy = None
        baseline_force = self._force_vector(get_observation())
        if baseline_force is None:
            self.get_logger().warn("align force baseline unavailable; retry disabled")
        else:
            self.get_logger().info(
                "align force baseline: "
                f"fx={baseline_force[0]:+.2f}N, "
                f"fy={baseline_force[1]:+.2f}N, "
                f"fz={baseline_force[2]:+.2f}N, "
                f"xy_threshold="
                f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N:.2f}N, "
                f"z_threshold="
                f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_Z_THRESHOLD_N:.2f}N"
            )
        for step in range(DistancePredictionConfig.ALIGN_MAX_STEPS):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if tcp_pose is None:
                self.sleep_for(DistancePredictionConfig.DT)
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
                stable_count = 0
                target_pose = self._copy_pose(tcp_pose)
                target_pose.position.x += float(retry_step[0])
                target_pose.position.y += float(retry_step[1])
                target_pose.position.z += float(retry_step[2])
                if self._target_orientation is not None:
                    target_pose.orientation = self._copy_quaternion(
                        self._target_orientation
                    )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=target_pose,
                    stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
                    damping=list(DistancePredictionConfig.ALIGN_DAMPING),
                )
                delta_text = ""
                if force_delta is not None:
                    delta_text = (
                        f"force_delta=({force_delta[0]:+.2f}, "
                        f"{force_delta[1]:+.2f}, "
                        f"{force_delta[2]:+.2f})N, "
                    )
                self.get_logger().warn(
                    f"align[{step:03d}] retry: "
                    f"{delta_text}"
                    f"cmd_base=({retry_step[0]*1000:+.2f}, "
                    f"{retry_step[1]*1000:+.2f}, "
                    f"{retry_step[2]*1000:+.2f})mm"
                )
                self.sleep_for(DistancePredictionConfig.ALIGN_COMMAND_SETTLE_S)
                continue

            offset_m = self._distance.predict_offset_m(obs, self._port_id())
            if offset_m is None:
                self.sleep_for(DistancePredictionConfig.DT)
                continue

            correction_base = self._align_correction_base(offset_m)
            offset_base = -correction_base
            xy_base = float(np.linalg.norm(offset_base[:2]))
            last_xy = xy_base
            if xy_base < DistancePredictionConfig.ALIGN_FINISH_XY_M:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= DistancePredictionConfig.ALIGN_STABLE_STEPS:
                self.get_logger().info(
                    f"align stable: xy_base={xy_base*1000:.2f}mm x {stable_count}"
                )
                return True

            step_xy = correction_base[:2] * DistancePredictionConfig.XY_GAIN
            step_xy = np.clip(
                step_xy,
                -DistancePredictionConfig.MAX_XY_STEP_M,
                DistancePredictionConfig.MAX_XY_STEP_M,
            )
            if np.linalg.norm(step_xy) < DistancePredictionConfig.XY_DEADBAND_M:
                step_xy[:] = 0.0

            target_pose = self._copy_pose(tcp_pose)
            target_pose.position.x += float(step_xy[0])
            target_pose.position.y += float(step_xy[1])
            if self._target_orientation is not None:
                target_pose.orientation = self._copy_quaternion(self._target_orientation)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
                damping=list(DistancePredictionConfig.ALIGN_DAMPING),
            )

            self.get_logger().info(
                f"align[{step:03d}]: "
                f"pred_base_est=({offset_base[0]*1000:+.2f}, "
                f"{offset_base[1]*1000:+.2f}, {offset_base[2]*1000:+.2f})mm, "
                f"cmd_base_xy=({step_xy[0]*1000:+.2f}, "
                f"{step_xy[1]*1000:+.2f})mm, "
                f"xy_base={xy_base*1000:.2f}mm, "
                f"z_base_est={offset_base[2]*1000:+.2f}mm, "
                f"stable={stable_count}/"
                f"{DistancePredictionConfig.ALIGN_STABLE_STEPS}"
            )
            self.sleep_for(DistancePredictionConfig.ALIGN_COMMAND_SETTLE_S)

        if last_xy is None:
            self.get_logger().error("align failed: no distance predictions")
            return False
        success = last_xy < DistancePredictionConfig.ALIGN_FINISH_XY_M * 1.5
        self.get_logger().info(
            f"[stage 4/5] align done: "
            f"success={success}, last_xy_base={last_xy*1000:.2f}mm"
        )
        return success

    def _stage_insert(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 5/5] insert start")
        obs = get_observation()
        pose = self._tcp_pose(obs)
        if pose is None:
            self.get_logger().error("insert failed: missing TCP pose")
            return False

        max_depth = float(DistancePredictionConfig.MAX_INSERT_DEPTH_M)
        step_m = float(DistancePredictionConfig.MAX_DOWN_STEP_M)
        steps = min(
            int(math.ceil(max_depth / max(step_m, 1e-6))),
            DistancePredictionConfig.INSERT_MAX_STEPS,
        )
        start_z = float(pose.position.z)
        baseline_force = self._force_norm(obs)
        if baseline_force is None:
            self.get_logger().warn("insert force baseline unavailable; force guard disabled")
        else:
            self.get_logger().info(
                f"insert force baseline: {baseline_force:.2f}N, "
                f"delta_limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
            )

        for step in range(steps):
            obs = get_observation()
            force = self._force_norm(obs)
            force_delta = None
            if force is not None and baseline_force is not None:
                force_delta = force - baseline_force
            if (
                force_delta is not None
                and force_delta > DistancePredictionConfig.FORCE_LIMIT_N
            ):
                self.get_logger().warn(
                    "insert force delta limit: "
                    f"force={force:.2f}N, baseline={baseline_force:.2f}N, "
                    f"delta={force_delta:.2f}N, "
                    f"limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
                )
                return False

            current = self._tcp_pose(obs) or pose
            target_pose = self._copy_pose(current)
            target_pose.position.z = float(start_z - (step + 1) * step_m)
            if self._target_orientation is not None:
                target_pose.orientation = self._copy_quaternion(self._target_orientation)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(self._insertion_stiffness()),
                damping=list(self._insertion_damping()),
            )
            if step == 0 or step % 10 == 0:
                force_text = ""
                if force is not None and force_delta is not None:
                    force_text = f", force_delta={force_delta:+.2f}N"
                self.get_logger().info(
                    f"insert[{step:03d}]: dz={-(step + 1) * step_m * 1000:.1f}mm"
                    f"{force_text}"
                )
            self.sleep_for(DistancePredictionConfig.DT)

        if DistancePredictionConfig.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(DistancePredictionConfig.SETTLE_AFTER_INSERT_S)
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
        self._cached_port_base = None
        self._target_orientation = None
        self._fixed_target_orientation = None
        self.get_logger().info(
            "DebugSfpDistancePolicy start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )

        stages = (
            ("initial_lift", lambda: self._stage_initial_lift(get_observation, move_robot)),
            ("detect", lambda: self._stage_detect(get_observation)),
            ("approach", lambda: self._stage_approach(get_observation, move_robot)),
            ("align", lambda: self._stage_align(get_observation, move_robot)),
            ("insert", lambda: self._stage_insert(get_observation, move_robot)),
        )
        for name, stage in stages:
            send_feedback(f"debug policy: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"Debug policy failed at stage: {name}")
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"Debug policy exception at {name}: {exc}")
                send_feedback(f"failed: {name} exception")
                return False

        self.get_logger().info("DebugSfpDistancePolicy done")
        send_feedback("debug policy done")
        return True
