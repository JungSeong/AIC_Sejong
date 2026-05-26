from __future__ import annotations

import time
import os
import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from distance_prediction_policy.DebugSfpDistancePolicy import DebugSfpDistancePolicy
from distance_prediction_policy.config import DistancePredictionConfig

from ..core.config import SfpGrvsConfig
from ..core.geometry import plug_tip_to_port_label, pose3d_from_transform
from ..core.task_frames import sfp_plug_tip_frame_candidates, sfp_port_frame_candidates
from ..data.metrics import append_episode_metric


class SfpGrvsEvalPolicy(DebugSfpDistancePolicy):
    """SFP-only model policy for batch testing after training.

    This policy intentionally does not record data and does not use GT-guided
    align actions. It reuses the existing distance-prediction policy behavior
    so a test batch measures the currently configured YOLO and distance models.
    """

    def _lookup_first_transform(self, frames: tuple[str, ...]):
        buffer = getattr(self._parent_node, "_tf_buffer", None)
        if buffer is None:
            return None
        deadline = time.monotonic() + SfpGrvsConfig.GT_TF_WAIT_S
        while True:
            for frame in frames:
                try:
                    return frame, buffer.lookup_transform("base_link", frame, Time()).transform
                except Exception:
                    continue
            if time.monotonic() >= deadline:
                return None
            time.sleep(SfpGrvsConfig.GT_TF_POLL_S)
        return None

    def _target_port_transform(self, task: Task):
        return self._lookup_first_transform(sfp_port_frame_candidates(task))

    def _stage_pre_detect_view(self, get_observation, move_robot) -> bool:
        if not SfpGrvsConfig.PRE_DETECT_GT_VIEW_ENABLED:
            return True

        target = self._target_port_transform(self._task)
        if target is None:
            self.get_logger().warn("GRVS eval pre-detect view skipped: missing target port TF")
            return True

        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().warn("GRVS eval pre-detect view skipped: missing TCP pose")
            return True

        port_position = pose3d_from_transform(target[1]).position
        tcp_offset = np.array(
            [
                DistancePredictionConfig.APPROACH_TCP_OFFSET_X_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Y_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Z_M,
            ],
            dtype=np.float64,
        )
        view_position = port_position + tcp_offset
        view_position[2] += float(SfpGrvsConfig.PRE_DETECT_VIEW_Z_OFFSET_M)

        target_pose = self._copy_pose(start_pose)
        target_pose.position.x = float(view_position[0])
        target_pose.position.y = float(view_position[1])
        target_pose.position.z = float(view_position[2])

        self.get_logger().info(
            "GRVS eval pre-detect view start: "
            f"target=({view_position[0]:+.3f}, {view_position[1]:+.3f}, "
            f"{view_position[2]:+.3f})m"
        )
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=target_pose,
            steps=SfpGrvsConfig.PRE_DETECT_VIEW_STEPS,
            stiffness=DistancePredictionConfig.APPROACH_STIFFNESS,
            damping=DistancePredictionConfig.APPROACH_DAMPING,
            dt=SfpGrvsConfig.PRE_DETECT_VIEW_DT,
            label="eval_pre_detect_view",
        )
        if SfpGrvsConfig.PRE_DETECT_VIEW_SETTLE_S > 0:
            self.sleep_for(SfpGrvsConfig.PRE_DETECT_VIEW_SETTLE_S)
        self.get_logger().info("GRVS eval pre-detect view done")
        return True

    def _stage_initial_lift(self, get_observation, move_robot) -> bool:
        if not super()._stage_initial_lift(get_observation, move_robot):
            return False
        return self._stage_pre_detect_view(get_observation, move_robot)

    def _gt_xy_m(self, task: Task) -> float | None:
        port = self._lookup_first_transform(sfp_port_frame_candidates(task))
        plug = self._lookup_first_transform(sfp_plug_tip_frame_candidates(task))
        if port is None or plug is None:
            return None
        gt_label = plug_tip_to_port_label(port[1], plug[1])
        return float(np.hypot(gt_label["x_m"], gt_label["y_m"]))

    def _record_metric(self, task: Task, success: bool) -> None:
        final_xy = self._gt_xy_m(task)
        record = {
            "batch_id": os.environ.get("AIC_GRVS_BATCH_ID", ""),
            "phase": os.environ.get("AIC_GRVS_PHASE", "test"),
            "policy": self.__class__.__name__,
            "task_target": str(getattr(task, "target_module_name", "")),
            "task_port": str(getattr(task, "port_name", "")),
            "success": bool(success),
            "stage": "done" if success else "failed",
            "align_actions": None,
            "retry_count": None,
            "min_xy_m": final_xy,
            "final_xy_m": final_xy,
            "xy_tol_m": SfpGrvsConfig.ALIGN_XY_TOL_M,
        }
        append_episode_metric(record)
        xy_text = "N/A" if final_xy is None else f"{final_xy * 1000.0:.2f}mm"
        self.get_logger().info(
            f"GRVS eval metric: success={success}, final_xy={xy_text}"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if "sfp" not in str(task.port_name).lower():
            self.get_logger().warn(
                "SfpGrvsEvalPolicy only supports SFP tasks; "
                f"got port={task.port_name}"
            )
            return False
        success = bool(super().insert_cable(task, get_observation, move_robot, send_feedback))
        self._record_metric(task, success)
        return success
