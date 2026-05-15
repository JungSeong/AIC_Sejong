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
from distance_prediction_policy.phase.align import run_align_stage
from distance_prediction_policy.phase.approach import run_approach_stage
from distance_prediction_policy.phase.detection import (
    estimate_port as run_estimate_port,
    run_detect_stage,
)
from distance_prediction_policy.phase.initial_lift import run_initial_lift_stage
from distance_prediction_policy.phase.insert import run_insert_stage
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
WS_ROOT = SRC_ROOT.parent
DEFAULT_SFP_YOLO_MODEL_PATHS = (
    WS_ROOT / "model" / "ais_yolo" / "approach" / "SFP" / "weights" / "best.pt",
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
    WS_ROOT / "model" / "ais_yolo" / "approach" / "SC" / "weights" / "best.pt",
    WS_ROOT
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
        self._sfp_yolo_conf_thresh = max(
            0.8,
            float(os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.8")),
        )
        self._sc_yolo_conf_thresh = max(
            0.8,
            float(
                os.environ.get(
                    "AIC_DEBUG_SC_YOLO_CONF_THRESH",
                    os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.8"),
                )
            ),
        )
        self._yolo_conf_thresh = self._sfp_yolo_conf_thresh
        self._vision_by_port_type = {}
        self._vision_debug_save_enabled = False

        self._vision = self._vision_for_port_type("sfp")
        self._distance = VisionOffsetPredictor(logger=self.get_logger())
        try:
            self._distance.warmup(n_iter=2)
        except Exception as exc:
            self.get_logger().warn(f"distance model warmup error: {exc}")

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
                debug_save_enabled=self._vision_debug_save_enabled,
                auto_start=False,
            )
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
        return run_initial_lift_stage(self, get_observation, move_robot)

    def _stage_detect(self, get_observation) -> bool:
        return run_detect_stage(self, get_observation)

    def _estimate_port(self, get_observation) -> Optional[np.ndarray]:
        return run_estimate_port(self, get_observation)

    def _stage_approach(self, get_observation, move_robot) -> bool:
        return run_approach_stage(self, get_observation, move_robot)

    def _stage_align(self, get_observation, move_robot) -> bool:
        return run_align_stage(self, get_observation, move_robot)

    def _stage_insert(self, get_observation, move_robot) -> bool:
        return run_insert_stage(self, get_observation, move_robot)

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
