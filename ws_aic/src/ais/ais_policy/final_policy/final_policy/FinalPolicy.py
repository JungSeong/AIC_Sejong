from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from transforms3d._gohlketransforms import quaternion_multiply

from ais_pose_prediction.config import PosePredictionConfig
from ais_pose_prediction.predictor import PosePredictor
from distance_prediction_policy.DebugSfpDistancePolicy import (
    DebugSfpDistancePolicy,
    _resolve_sc_yolo_model_path,
    _resolve_sfp_yolo_model_path,
)
from distance_prediction_policy.config import DistancePredictionConfig
from motion_planning_node.core.geometry import quat_to_tuple, tuple_to_quat
from motion_planning_node.core.vision import VisionPortEstimator


class FinalPolicy(DebugSfpDistancePolicy):
    """Final policy driven by the unified pose prediction model.

    Stages:
    1. detect
    2. approach
    3. yaw rotation + align
    4. insert
    """

    def __init__(self, parent_node):
        Policy.__init__(self, parent_node)
        self._task: Optional[Task] = None
        self._sfp_yolo_model_path = _resolve_sfp_yolo_model_path()
        self._sc_yolo_model_path = _resolve_sc_yolo_model_path()
        self._yolo_model_path = self._sfp_yolo_model_path
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
        self._vision = self._vision_for_port_type("sfp")
        self._pose_predictor = PosePredictor(logger=self.get_logger())
        self.get_logger().info(
            "FinalPolicy ready: "
            f"pose_model={PosePredictionConfig.CHECKPOINT_PATH}, "
            f"sfp_yolo={self._sfp_yolo_model_path}, "
            f"sc_yolo={self._sc_yolo_model_path}"
        )

    def _vision_for_port_type(self, port_type: str) -> VisionPortEstimator:
        return super()._vision_for_port_type(port_type)

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

    def _stage_yaw_align(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 3/4] yaw_rotation_align start")
        baseline_force = self._force_vector(get_observation())
        stable_count = 0
        last_xy = None
        last_yaw = None

        for step in range(PosePredictionConfig.ALIGN_MAX_STEPS):
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
                target_pose = self._copy_pose(tcp_pose)
                target_pose.position.x += float(retry_step[0])
                target_pose.position.y += float(retry_step[1])
                target_pose.position.z += float(retry_step[2])
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=target_pose,
                    stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
                    damping=list(DistancePredictionConfig.ALIGN_DAMPING),
                )
                stable_count = 0
                self.sleep_for(PosePredictionConfig.COMMAND_SETTLE_S)
                continue

            prediction = self._pose_predictor.predict(obs)
            if prediction is None:
                self.sleep_for(DistancePredictionConfig.DT)
                continue

            offset_m = self._select_offset(prediction)
            dyaw = float(prediction["dyaw_rad"])
            correction_base = self._align_correction_base(offset_m)
            xy_error = float(np.linalg.norm(offset_m[:2]))
            yaw_error = abs(dyaw)
            last_xy = xy_error
            last_yaw = yaw_error

            if xy_error < PosePredictionConfig.XY_TOL_M and yaw_error < PosePredictionConfig.YAW_TOL_RAD:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= PosePredictionConfig.STABLE_STEPS:
                self.get_logger().info(
                    "yaw_align stable: "
                    f"xy={xy_error * 1000.0:.2f}mm, "
                    f"dyaw={math.degrees(yaw_error):.2f}deg"
                )
                return True

            step_xy = np.clip(
                correction_base[:2] * PosePredictionConfig.XY_GAIN,
                -PosePredictionConfig.MAX_XY_STEP_M,
                PosePredictionConfig.MAX_XY_STEP_M,
            )
            yaw_step = float(
                np.clip(
                    dyaw * PosePredictionConfig.YAW_GAIN,
                    -PosePredictionConfig.MAX_YAW_STEP_RAD,
                    PosePredictionConfig.MAX_YAW_STEP_RAD,
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
                stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
                damping=list(DistancePredictionConfig.ALIGN_DAMPING),
            )
            self.get_logger().info(
                f"yaw_align[{step:03d}]: "
                f"offset=({offset_m[0]*1000:+.2f}, {offset_m[1]*1000:+.2f}, {offset_m[2]*1000:+.2f})mm, "
                f"cmd_xy=({step_xy[0]*1000:+.2f}, {step_xy[1]*1000:+.2f})mm, "
                f"dyaw={math.degrees(dyaw):+.2f}deg, "
                f"cmd_yaw={math.degrees(yaw_step):+.2f}deg, "
                f"stable={stable_count}/{PosePredictionConfig.STABLE_STEPS}"
            )
            self.sleep_for(PosePredictionConfig.COMMAND_SETTLE_S)

        if last_xy is None or last_yaw is None:
            self.get_logger().error("yaw_align failed: no pose predictions")
            return False
        success = (
            last_xy < PosePredictionConfig.XY_TOL_M * 1.5
            and last_yaw < PosePredictionConfig.YAW_TOL_RAD * 1.5
        )
        self.get_logger().info(
            f"[stage 3/4] yaw_rotation_align done: success={success}, "
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
        if drop > PosePredictionConfig.INSERT_FORCE_DROP_LIMIT_N:
            return True, (
                f"force_drop={drop:.2f}N "
                f"(force={force_norm:.2f}N, baseline={baseline_norm:.2f}N)"
            )
        if delta > PosePredictionConfig.INSERT_FORCE_RISE_LIMIT_N:
            return True, (
                f"force_rise={delta:.2f}N "
                f"(force={force_norm:.2f}N, baseline={baseline_norm:.2f}N)"
            )
        return False, ""

    def _stage_insert(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 4/4] insert start")
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
                f"drop_limit={PosePredictionConfig.INSERT_FORCE_DROP_LIMIT_N:.2f}N, "
                f"rise_limit={PosePredictionConfig.INSERT_FORCE_RISE_LIMIT_N:.2f}N"
            )

        max_depth = float(DistancePredictionConfig.MAX_INSERT_DEPTH_M)
        step_m = min(
            float(PosePredictionConfig.INSERT_STEP_M),
            float(DistancePredictionConfig.MAX_DOWN_STEP_M),
        )
        max_steps = min(
            int(math.ceil(max_depth / max(step_m, 1e-6))),
            int(DistancePredictionConfig.INSERT_MAX_STEPS),
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
                if retries > PosePredictionConfig.INSERT_RETRY_MAX:
                    self.get_logger().error(
                        "insert failed: retry limit exceeded after "
                        f"{PosePredictionConfig.INSERT_RETRY_MAX} retries; {reason}"
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
                        [0.0, 0.0, PosePredictionConfig.INSERT_RETRY_LIFT_M],
                        dtype=np.float64,
                    )
                retry_step[2] = max(
                    float(retry_step[2]),
                    float(PosePredictionConfig.INSERT_RETRY_LIFT_M),
                )

                target_pose = self._copy_pose(current)
                target_pose.position.x += float(retry_step[0])
                target_pose.position.y += float(retry_step[1])
                target_pose.position.z += float(retry_step[2])
                if self._target_orientation is not None:
                    target_pose.orientation = self._copy_quaternion(self._target_orientation)
                self.get_logger().warn(
                    "insert retry: "
                    f"{reason}, retry={retries}/{PosePredictionConfig.INSERT_RETRY_MAX}, "
                    f"step=({retry_step[0]*1000:+.2f}, {retry_step[1]*1000:+.2f}, "
                    f"{retry_step[2]*1000:+.2f})mm"
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=target_pose,
                    stiffness=list(self._insertion_stiffness()),
                    damping=list(self._insertion_damping()),
                )
                self.sleep_for(PosePredictionConfig.INSERT_RETRY_SETTLE_S)
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
            self.sleep_for(PosePredictionConfig.INSERT_DT)

        if DistancePredictionConfig.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(DistancePredictionConfig.SETTLE_AFTER_INSERT_S)
        self.get_logger().info("[stage 4/4] insert done")
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
            "FinalPolicy start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )
        stages = (
            ("detect", lambda: self._stage_detect(get_observation)),
            ("approach", lambda: self._stage_approach(get_observation, move_robot)),
            ("yaw_rotation_align", lambda: self._stage_yaw_align(get_observation, move_robot)),
            ("insert", lambda: self._stage_insert(get_observation, move_robot)),
        )
        for name, stage in stages:
            send_feedback(f"final policy: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"FinalPolicy failed at stage: {name}")
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"FinalPolicy exception at {name}: {exc}")
                send_feedback(f"failed: {name} exception")
                return False
        send_feedback("final policy done")
        self.get_logger().info("FinalPolicy done")
        return True
