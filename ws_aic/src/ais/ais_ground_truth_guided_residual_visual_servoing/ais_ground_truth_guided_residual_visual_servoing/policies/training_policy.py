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
from geometry_msgs.msg import Point, Pose
from rclpy.time import Time
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from distance_prediction_policy.DebugSfpDistancePolicy import DebugSfpDistancePolicy
from distance_prediction_policy.config import DistancePredictionConfig
from motion_planning_node.core.geometry import interp_profile, quat_to_tuple, tuple_to_quat

from ..core.config import SfpGrvsConfig, sample_epsilon_m
from ..core.geometry import plug_tip_to_port_label, pose3d_from_transform
from ..core.task_frames import (
    sfp_plug_tip_frame_candidates,
    sfp_port_frame_candidates,
    sfp_port_pair_frame_candidates,
)
from ..data.distance_dataset import SfpDistanceSampleRecorder
from ..data.metrics import append_episode_metric
from ..data.rotation_dataset import SfpRotationSampleRecorder
from ..data.yolo_dataset import SfpYoloFailureRecorder


class SfpGrvsTrainingPolicy(DebugSfpDistancePolicy):
    """SFP-only GRVS data policy.

    One rollout writes two datasets:
    - ``data/<package>/<version>/yolo`` for YOLO pose retraining.
    - ``data/<package>/<version>/distance_prediction`` for residual distance
      prediction fine-tuning.

    Ground truth is used for labels, success checks, and training-only fallback
    motion. It is not part of the learned runtime state.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._sample_index = 0
        self._episode_index = 0
        self._yolo_timestep_index = 0
        self._filtered_force = None
        self._force_baseline = None
        self._episode_metric = {}
        self._rotation_sample_observation = None
        self._rng = np.random.default_rng(SfpGrvsConfig.GT_FALLBACK_EPSILON_SEED)
        self._distance_recorder = SfpDistanceSampleRecorder(
            SfpGrvsConfig.DISTANCE_DATASET_DIR
        )
        self._rotation_recorder = SfpRotationSampleRecorder(
            SfpGrvsConfig.ROTATION_DATASET_DIR
        )
        self._yolo_recorder = SfpYoloFailureRecorder(
            SfpGrvsConfig.YOLO_DATASET_DIR,
            val_ratio=SfpGrvsConfig.YOLO_VAL_RATIO,
            bbox_margin=SfpGrvsConfig.YOLO_BBOX_MARGIN,
        )
        self.get_logger().info(
            "SfpGrvsTrainingPolicy ready: "
            f"version={SfpGrvsConfig.VERSION}, "
            f"data_yolo={SfpGrvsConfig.YOLO_DATASET_DIR}, "
            f"data_distance={SfpGrvsConfig.DISTANCE_DATASET_DIR}, "
            f"data_rotation={SfpGrvsConfig.ROTATION_DATASET_DIR}, "
            f"model_yolo={SfpGrvsConfig.YOLO_MODEL_DIR}, "
            f"model_distance={SfpGrvsConfig.DISTANCE_MODEL_DIR}, "
            f"model_rotation={SfpGrvsConfig.ROTATION_MODEL_DIR}"
        )

    def _next_sample_id(self, prefix: str) -> str:
        sample_id = f"{self._run_id}_{prefix}_{self._sample_index:06d}"
        self._sample_index += 1
        return sample_id

    def _lookup_first_transform(
        self,
        frames: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        label: str = "GT TF",
    ):
        buffer = getattr(self._parent_node, "_tf_buffer", None)
        if buffer is None:
            return None
        deadline = time.monotonic() + (
            SfpGrvsConfig.GT_TF_WAIT_S if timeout_s is None else max(0.0, timeout_s)
        )
        last_error = ""
        while True:
            for frame in frames:
                try:
                    transform = buffer.lookup_transform("base_link", frame, Time()).transform
                    return frame, transform
                except Exception as exc:
                    last_error = str(exc)
                    continue
            if time.monotonic() >= deadline:
                break
            time.sleep(SfpGrvsConfig.GT_TF_POLL_S)
        self.get_logger().warn(
            f"GRVS {label} unavailable: candidates={list(frames)}, last_error={last_error}"
        )
        return None

    def _target_port_transform(self, task: Task):
        return self._lookup_first_transform(
            sfp_port_frame_candidates(task),
            label="target port TF",
        )

    def _plug_tip_transform(self, task: Task):
        return self._lookup_first_transform(
            sfp_plug_tip_frame_candidates(task),
            label="plug tip TF",
        )

    def _port_pair_transforms(self, task: Task) -> list:
        transforms = []
        for candidates in sfp_port_pair_frame_candidates(task):
            found = self._lookup_first_transform(candidates, label="port-pair TF")
            if found is None:
                return []
            transforms.append(found[1])
        return transforms

    @staticmethod
    def _transform_quat_wxyz(transform) -> tuple[float, float, float, float]:
        return (
            float(transform.rotation.w),
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
        )

    @staticmethod
    def _quat_angle_rad(
        q_a: tuple[float, float, float, float],
        q_b: tuple[float, float, float, float],
    ) -> float:
        a = np.asarray(q_a, dtype=np.float64)
        b = np.asarray(q_b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        dot = abs(float(np.dot(a, b))) / denom
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def _gt_wrist_orientation(self, start_pose, port_transform, plug_transform):
        q_port = self._transform_quat_wxyz(port_transform)
        q_plug = self._transform_quat_wxyz(plug_transform)
        q_plug_inv = (q_plug[0], -q_plug[1], -q_plug[2], -q_plug[3])
        q_delta = quaternion_multiply(q_port, q_plug_inv)
        q_target = self._normalize_quat(
            quaternion_multiply(q_delta, quat_to_tuple(start_pose.orientation))
        )
        return tuple_to_quat(q_target)

    def _base_to_camera_matrices(self, observation) -> dict[str, np.ndarray]:
        matrices: dict[str, np.ndarray] = {}
        for camera in ("left", "center", "right"):
            try:
                matrix = self._vision._base_to_camera_optical_matrix(observation, camera)
            except Exception:
                matrix = None
            if matrix is not None:
                matrices[camera] = matrix
        return matrices

    @staticmethod
    def _force_vector(observation) -> np.ndarray | None:
        if observation is None:
            return None
        force = observation.wrist_wrench.wrench.force
        return np.array([force.x, force.y, force.z], dtype=np.float64)

    def _low_pass_force(self, observation) -> np.ndarray | None:
        current = self._force_vector(observation)
        if current is None:
            return None
        if self._filtered_force is None:
            self._filtered_force = current
        else:
            alpha = float(SfpGrvsConfig.FORCE_LPF_ALPHA)
            self._filtered_force = alpha * current + (1.0 - alpha) * self._filtered_force
        return self._filtered_force

    def _set_force_baseline(self, observation) -> None:
        baseline = self._force_vector(observation)
        if baseline is None:
            self._force_baseline = None
            self.get_logger().warn("GRVS force baseline skipped: missing observation")
            return
        self._force_baseline = baseline
        self._filtered_force = baseline.copy()
        self._episode_metric["force_baseline_n"] = baseline.tolist()
        self.get_logger().info(
            "GRVS force baseline after approach: "
            f"F=({baseline[0]:+.2f}, {baseline[1]:+.2f}, {baseline[2]:+.2f})N"
        )

    def _force_retry_action(
        self,
        observation,
        *,
        xy_m: float | None,
    ) -> tuple[np.ndarray | None, str]:
        if not SfpGrvsConfig.FORCE_RETRY_ENABLED:
            return None, ""
        if xy_m is None or xy_m > SfpGrvsConfig.FORCE_RETRY_XY_GATE_M:
            self._low_pass_force(observation)
            return None, ""
        filtered = self._low_pass_force(observation)
        if filtered is None:
            return None, ""
        if self._force_baseline is None:
            return None, ""

        dfx, dfy, dfz = filtered - self._force_baseline
        if abs(dfz) > SfpGrvsConfig.FORCE_THRESHOLD_Z_N:
            return (
                np.array(
                    [0.0, 0.0, self._z_command_delta(SfpGrvsConfig.FORCE_FALLBACK_Z_M)],
                    dtype=np.float64,
                ),
                f"dfz={dfz:+.2f}N",
            )
        if (
            abs(dfx) > SfpGrvsConfig.FORCE_THRESHOLD_X_N
            or abs(dfy) > SfpGrvsConfig.FORCE_THRESHOLD_Y_N
        ):
            return (
                np.array(
                    [0.0, 0.0, self._z_command_delta(SfpGrvsConfig.FORCE_FALLBACK_XY_M)],
                    dtype=np.float64,
                ),
                f"dfxy=({dfx:+.2f}, {dfy:+.2f})N",
            )
        return None, ""

    def _stage_approach_position(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[grvs stage 3/6] approach start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("GRVS approach failed: missing TCP pose")
            return False

        port = self._cached_port_base
        if port is None:
            self.get_logger().error("GRVS approach failed: missing cached port estimate")
            return False

        tcp_offset = np.array(
            [
                DistancePredictionConfig.APPROACH_TCP_OFFSET_X_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Y_M,
                DistancePredictionConfig.APPROACH_TCP_OFFSET_Z_M,
            ],
            dtype=np.float64,
        )
        initial_z_offset = float(DistancePredictionConfig.APPROACH_Z_OFFSET_SFP_M)
        near_z_offset = float(DistancePredictionConfig.APPROACH_NEAR_Z_OFFSET_M)

        def make_approach_pose(z_offset: float, reference_pose: Pose) -> tuple[Pose, np.ndarray]:
            target = port + np.array([0.0, 0.0, z_offset], dtype=np.float64)
            target = target + tcp_offset
            return (
                Pose(
                    position=Point(
                        x=float(target[0]),
                        y=float(target[1]),
                        z=float(target[2]),
                    ),
                    orientation=self._copy_quaternion(reference_pose.orientation),
                ),
                target,
            )

        far_pose, far_target = make_approach_pose(initial_z_offset, start_pose)
        near_pose, near_target = make_approach_pose(near_z_offset, start_pose)
        self.get_logger().info(
            "GRVS approach targets: "
            f"initial_z_plus={initial_z_offset*1000:.1f}mm, "
            f"near_z_plus={near_z_offset*1000:.1f}mm, "
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
        near_pose.orientation = self._copy_quaternion(near_start_pose.orientation)
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
            self.sleep_for(DistancePredictionConfig.APPROACH_SETTLE_S)
        self.get_logger().info("[grvs stage 3/6] approach done")
        return True

    def _stage_rotation_with_force_baseline(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[grvs stage 4/6] rotation start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("GRVS rotation failed: missing TCP pose")
            return False
        if self._target_orientation is None:
            self._target_orientation = self._target_wrist_orientation(start_pose)

        self._record_rotation_sample(observation=obs, stage="rotation_start")
        target_pose = self._copy_pose(start_pose)
        target_pose.orientation = self._copy_quaternion(self._target_orientation)
        self._rotation_sample_observation = get_observation
        try:
            self._follow_pose(
                move_robot=move_robot,
                start_pose=start_pose,
                target_pose=target_pose,
                steps=DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_STEPS,
                stiffness=DistancePredictionConfig.APPROACH_NEAR_STIFFNESS,
                damping=DistancePredictionConfig.APPROACH_NEAR_DAMPING,
                dt=DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_DT,
                label="rotation",
            )
        finally:
            self._rotation_sample_observation = None

        end_obs = get_observation()
        self._record_rotation_sample(observation=end_obs, stage="rotation_end")
        self._set_force_baseline(end_obs)
        self.get_logger().info("[grvs stage 4/6] rotation done")
        return True

    def _stage_pre_detect_view(self, get_observation, move_robot) -> bool:
        if not SfpGrvsConfig.PRE_DETECT_GT_VIEW_ENABLED:
            return True

        target = self._target_port_transform(self._task)
        if target is None:
            self.get_logger().warn("GRVS pre-detect view skipped: missing target port TF")
            return True

        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().warn("GRVS pre-detect view skipped: missing TCP pose")
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
            "[grvs stage 2/5] pre_detect_view start: "
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
            label="pre_detect_view",
        )
        if SfpGrvsConfig.PRE_DETECT_VIEW_SETTLE_S > 0:
            self.sleep_for(SfpGrvsConfig.PRE_DETECT_VIEW_SETTLE_S)
        self.get_logger().info("[grvs stage 2/5] pre_detect_view done")
        return True

    def _gt_xy_m(self, task: Task) -> float | None:
        port = self._target_port_transform(task)
        plug = self._plug_tip_transform(task)
        if port is None or plug is None:
            return None
        gt_label = plug_tip_to_port_label(port[1], plug[1])
        return float(np.hypot(gt_label["x_m"], gt_label["y_m"]))

    def _record_episode_metric(
        self,
        *,
        success: bool,
        stage: str,
        error: str = "",
    ) -> None:
        metric = dict(self._episode_metric)
        final_xy = self._gt_xy_m(self._task)
        if final_xy is not None:
            metric["final_xy_m"] = final_xy
            if metric.get("min_xy_m") is None:
                metric["min_xy_m"] = final_xy
        metric.update(
            {
                "batch_id": os.environ.get("AIC_GRVS_BATCH_ID", ""),
                "phase": os.environ.get("AIC_GRVS_PHASE", "collect"),
                "policy": self.__class__.__name__,
                "episode_index": self._episode_index,
                "task_target": str(getattr(self._task, "target_module_name", "")),
                "task_port": str(getattr(self._task, "port_name", "")),
                "success": bool(success),
                "stage": stage,
                "error": error,
            }
        )
        append_episode_metric(metric)
        final_text = "N/A"
        if metric.get("final_xy_m") is not None:
            final_text = f"{float(metric['final_xy_m']) * 1000.0:.2f}mm"
        min_text = "N/A"
        if metric.get("min_xy_m") is not None:
            min_text = f"{float(metric['min_xy_m']) * 1000.0:.2f}mm"
        self.get_logger().info(
            "GRVS episode metric: "
            f"success={success}, stage={stage}, final_xy={final_text}, "
            f"min_xy={min_text}, actions={metric.get('align_actions', 0)}, "
            f"retries={metric.get('retry_count', 0)}"
        )

    def _record_yolo_scene(self, *, observation, task: Task, reason: str) -> int:
        if not SfpGrvsConfig.RECORD_YOLO_FAILURES:
            return 0
        port_pair = self._port_pair_transforms(task)
        if len(port_pair) != 2:
            self.get_logger().warn("GRVS YOLO record skipped: missing SFP port-pair TF")
            return 0
        timestep_index = self._yolo_timestep_index
        self._yolo_timestep_index += 1
        sample_id = (
            f"{self._run_id}_ep{self._episode_index:04d}_"
            f"ts{timestep_index:06d}_yolo_{reason}"
        )
        saved = self._yolo_recorder.record(
            observation=observation,
            port_transforms=port_pair,
            base_to_camera=self._base_to_camera_matrices(observation),
            episode_id=self._episode_index,
            timestep_index=timestep_index,
            sample_id=sample_id,
            stem_prefix=reason,
        )
        if saved:
            self.get_logger().info(
                f"GRVS YOLO {reason} scene saved: images={saved}, "
                f"dir={SfpGrvsConfig.YOLO_DATASET_DIR}"
            )
        return saved

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
        stride = max(1, int(SfpGrvsConfig.ROTATION_SAMPLE_STRIDE))

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
            if (
                SfpGrvsConfig.RECORD_ROTATION_TRAJECTORY
                and self._rotation_sample_observation is not None
                and label == "rotation"
                and index % stride == 0
            ):
                self._record_rotation_sample(
                    observation=self._rotation_sample_observation(),
                    stage=f"{label}_{index + 1:02d}",
                )

    def _stage_detect(self, get_observation) -> bool:
        self.get_logger().info("[grvs stage 2/6] detect start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("GRVS detect failed: missing TCP pose")
            return False

        port = self._estimate_port(get_observation)
        gt_port = self._target_port_transform(self._task)
        if port is None:
            self._episode_metric["yolo_failed"] = True
            failure_obs = get_observation() or obs
            self._record_yolo_scene(
                observation=failure_obs,
                task=self._task,
                reason="yolo_failed",
            )
            if gt_port is None:
                self.get_logger().error("GRVS detect failed: YOLO and GT port unavailable")
                return False
            epsilon_m = sample_epsilon_m(self._rng, SfpGrvsConfig)
            port = pose3d_from_transform(gt_port[1]).position + epsilon_m
            self._episode_metric["yolo_fallback"] = True
            self.get_logger().warn(
                "GRVS YOLO fallback: using GT port + epsilon "
                f"epsilon=({epsilon_m[0]*1000:+.2f}, "
                f"{epsilon_m[1]*1000:+.2f}, {epsilon_m[2]*1000:+.2f})mm"
            )
        elif gt_port is not None:
            gt_xyz = pose3d_from_transform(gt_port[1]).position
            error_m = float(np.linalg.norm(np.asarray(port, dtype=np.float64) - gt_xyz))
            if error_m > SfpGrvsConfig.YOLO_RECORD_ERROR_THRESHOLD_M:
                self._episode_metric["yolo_high_error_m"] = error_m
                error_obs = get_observation() or obs
                self._record_yolo_scene(
                    observation=error_obs,
                    task=self._task,
                    reason=f"yolo_error_{int(round(error_m * 1000.0)):03d}mm",
                )
                self.get_logger().warn(
                    f"GRVS YOLO high error: {error_m*1000:.2f}mm "
                    f"(threshold={SfpGrvsConfig.YOLO_RECORD_ERROR_THRESHOLD_M*1000:.2f}mm)"
                )
                epsilon_m = sample_epsilon_m(self._rng, SfpGrvsConfig)
                port = gt_xyz + epsilon_m
                self._episode_metric["yolo_fallback"] = True
                self.get_logger().warn(
                    "GRVS YOLO high-error fallback: using GT port + epsilon "
                    f"epsilon=({epsilon_m[0]*1000:+.2f}, "
                    f"{epsilon_m[1]*1000:+.2f}, {epsilon_m[2]*1000:+.2f})mm"
                )

        self._cached_port_base = np.asarray(port, dtype=np.float64)
        gt_plug = self._plug_tip_transform(self._task) if gt_port is not None else None
        if SfpGrvsConfig.USE_GT_ORIENTATION and gt_port is not None and gt_plug is not None:
            target_orientation = self._gt_wrist_orientation(start_pose, gt_port[1], gt_plug[1])
            self._target_orientation = self._copy_quaternion(target_orientation)
            angle_rad = self._quat_angle_rad(
                quat_to_tuple(start_pose.orientation),
                quat_to_tuple(target_orientation),
            )
            self._episode_metric["gt_orientation_angle_rad"] = angle_rad
            if angle_rad > SfpGrvsConfig.GT_ORIENTATION_TOLERANCE_RAD:
                self.get_logger().info(
                    "GRVS GT orientation queued for rotation: "
                    f"angle={angle_rad:.3f}rad"
                )
        else:
            self._target_orientation = self._target_wrist_orientation(start_pose)
        self.get_logger().info(
            "GRVS detect cached: "
            f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
        )
        self.get_logger().info("[grvs stage 2/6] detect done")
        return True

    def _record_distance_sample(
        self,
        *,
        observation,
        task: Task,
        step_index: int,
        action_m: np.ndarray,
        success_gt: bool,
    ) -> None:
        if not SfpGrvsConfig.RECORD_DISTANCE_SAMPLES:
            return
        port = self._target_port_transform(task)
        plug = self._plug_tip_transform(task)
        if port is None or plug is None:
            self.get_logger().warn("GRVS distance sample skipped: missing GT TF")
            return

        backbone_offset = self._distance.predict_offset_m(observation, self._port_id())
        gt_label = plug_tip_to_port_label(port[1], plug[1])
        gt_offset = np.array(
            [gt_label["x_m"], gt_label["y_m"], gt_label["z_m"]],
            dtype=np.float64,
        )
        residual = None
        if backbone_offset is not None:
            residual = gt_offset - np.asarray(backbone_offset, dtype=np.float64)

        sample_id = self._next_sample_id("dist")
        episode_name = f"grvs_{self._run_id}_ep{self._episode_index:06d}"
        self._distance_recorder.record(
            observation=observation,
            task=task,
            sample_id=sample_id,
            episode_name=episode_name,
            step_index=step_index,
            port_tf=port[1],
            plug_tf=plug[1],
            port_frame=port[0],
            plug_frame=plug[0],
            extras={
                "backbone_offset_m": backbone_offset,
                "residual_target_m": residual,
                "action_m": action_m,
                "success_gt": success_gt,
            },
        )

    def _record_rotation_sample(self, *, observation, stage: str) -> None:
        if not SfpGrvsConfig.RECORD_ROTATION_SAMPLES:
            return
        if observation is None:
            return
        port = self._target_port_transform(self._task)
        plug = self._plug_tip_transform(self._task)
        if port is None or plug is None:
            self.get_logger().warn(f"GRVS rotation sample skipped: missing GT TF ({stage})")
            return

        sample_id = self._next_sample_id(f"rot_{stage}")
        episode_name = f"grvs_{self._run_id}_ep{self._episode_index:06d}"
        saved = self._rotation_recorder.record(
            observation=observation,
            task=self._task,
            sample_id=sample_id,
            episode_name=episode_name,
            stage=stage,
            port_tf=port[1],
            plug_tf=plug[1],
            port_frame=port[0],
            plug_frame=plug[0],
            extras={
                "gt_orientation_angle_rad": self._episode_metric.get(
                    "gt_orientation_angle_rad"
                ),
            },
        )
        if saved:
            self.get_logger().info(
                f"GRVS rotation sample saved: stage={stage}, "
                f"dir={SfpGrvsConfig.ROTATION_DATASET_DIR}"
            )

    def _stage_grvs_align(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[grvs stage 5/6] align start")
        stable_count = 0
        last_xy = None
        for step in range(SfpGrvsConfig.ALIGN_MAX_ATTEMPTS):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if tcp_pose is None:
                self.sleep_for(DistancePredictionConfig.DT)
                continue

            port = self._target_port_transform(self._task)
            plug = self._plug_tip_transform(self._task)
            if port is None or plug is None:
                self.get_logger().error("GRVS align failed: missing GT TF")
                return False

            gt_label = plug_tip_to_port_label(port[1], plug[1])
            gt_offset_port = np.array(
                [gt_label["x_m"], gt_label["y_m"], gt_label["z_m"]],
                dtype=np.float64,
            )
            port_rotation = pose3d_from_transform(port[1]).rotation_matrix
            correction_base = port_rotation @ (-gt_offset_port)
            xy = float(np.linalg.norm(gt_offset_port[:2]))
            z_error_base = float(correction_base[2])
            last_xy = xy
            self._episode_metric["align_actions"] = step + 1
            current_min_xy = self._episode_metric.get("min_xy_m")
            if current_min_xy is None or xy < float(current_min_xy):
                self._episode_metric["min_xy_m"] = xy
            z_ready = abs(z_error_base) < SfpGrvsConfig.ALIGN_Z_TOL_M
            success_gt = xy < SfpGrvsConfig.ALIGN_XY_TOL_M and z_ready
            stable_count = stable_count + 1 if success_gt else 0

            retry_action, retry_reason = self._force_retry_action(obs, xy_m=xy)
            if retry_action is not None:
                action_m = retry_action
                stable_count = 0
                self._episode_metric["retry_count"] = (
                    int(self._episode_metric.get("retry_count", 0)) + 1
                )
            else:
                step_xy = correction_base[:2] * SfpGrvsConfig.XY_GAIN
                step_xy = np.clip(
                    step_xy,
                    -SfpGrvsConfig.MAX_XY_STEP_M,
                    SfpGrvsConfig.MAX_XY_STEP_M,
                )
                command_z = 0.0
                z_step_actual = float(
                    np.clip(
                        z_error_base * SfpGrvsConfig.Z_GAIN,
                        -SfpGrvsConfig.MAX_Z_STEP_M,
                        SfpGrvsConfig.MAX_Z_STEP_M,
                    )
                )
                if not z_ready:
                    command_z = self._z_command_delta(z_step_actual)
                if success_gt:
                    command_z = self._z_command_delta(-SfpGrvsConfig.ALIGN_INSERT_STEP_M)
                action_m = np.array(
                    [step_xy[0], step_xy[1], command_z],
                    dtype=np.float64,
                )
            self._record_distance_sample(
                observation=obs,
                task=self._task,
                step_index=step,
                action_m=action_m,
                success_gt=success_gt,
            )
            stride = max(1, int(SfpGrvsConfig.ROTATION_SAMPLE_STRIDE))
            if step % stride == 0:
                self._record_rotation_sample(
                    observation=obs,
                    stage=f"align_{step:02d}",
                )

            if (
                retry_action is None
                and stable_count >= SfpGrvsConfig.ALIGN_STABLE_STEPS
            ):
                self.get_logger().info(
                    f"GRVS align GT success: xy={xy*1000:.2f}mm "
                    f"x {stable_count}, attempts={step + 1}"
                )
                self._episode_metric["final_xy_m"] = xy
                return True

            target_pose = self._copy_pose(tcp_pose)
            target_pose.position.x += float(action_m[0])
            target_pose.position.y += float(action_m[1])
            target_pose.position.z += float(action_m[2])
            if self._target_orientation is not None:
                target_pose.orientation = self._copy_quaternion(self._target_orientation)
            self._set_debug_pose_target(move_robot=move_robot, pose=target_pose)

            retry_text = f", retry={retry_reason}" if retry_reason else ""
            self.get_logger().info(
                f"grvs_align[{step:02d}]: "
                f"gt_offset=({gt_offset_port[0]*1000:+.2f}, "
                f"{gt_offset_port[1]*1000:+.2f}, "
                f"{gt_offset_port[2]*1000:+.2f})mm, "
                f"cmd=({action_m[0]*1000:+.2f}, {action_m[1]*1000:+.2f}, "
                f"{action_m[2]*1000:+.2f})mm, xy={xy*1000:.2f}mm, "
                f"base_z_err={z_error_base*1000:+.2f}mm"
                f"{retry_text}"
            )
            self.sleep_for(SfpGrvsConfig.COMMAND_SETTLE_S)

        success = last_xy is not None and last_xy < SfpGrvsConfig.ALIGN_XY_TOL_M
        self.get_logger().info(
            f"[grvs stage 5/6] align done: success={success}, "
            f"last_xy={(last_xy or 0.0)*1000:.2f}mm"
        )
        if last_xy is not None:
            self._episode_metric["final_xy_m"] = last_xy
        return success

    def _stage_insert_with_metric(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[grvs stage 6/6] insert start")
        success = bool(self._stage_insert(get_observation, move_robot))
        final_xy = self._gt_xy_m(self._task)
        if final_xy is not None:
            self._episode_metric["final_xy_m"] = final_xy
            current_min_xy = self._episode_metric.get("min_xy_m")
            if current_min_xy is None or final_xy < float(current_min_xy):
                self._episode_metric["min_xy_m"] = final_xy
        self.get_logger().info(f"[grvs stage 6/6] insert done: success={success}")
        return success

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if "sfp" not in str(task.port_name).lower():
            self.get_logger().warn(
                "SfpGrvsTrainingPolicy only supports SFP tasks; "
                f"got port={task.port_name}"
            )
            return False

        self._task = task
        self._cached_port_base = None
        self._target_orientation = None
        self._filtered_force = None
        self._force_baseline = None
        self._episode_metric = {
            "align_actions": 0,
            "retry_count": 0,
            "min_xy_m": None,
            "final_xy_m": None,
            "yolo_failed": False,
            "yolo_fallback": False,
        }
        self._episode_index += 1
        self._yolo_timestep_index = 0
        self.get_logger().info(
            "SfpGrvsTrainingPolicy start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )

        stages = (
            ("initial_lift", lambda: self._stage_initial_lift(get_observation, move_robot)),
            ("pre_detect_view", lambda: self._stage_pre_detect_view(get_observation, move_robot)),
            ("detect", lambda: self._stage_detect(get_observation)),
            ("approach", lambda: self._stage_approach_position(get_observation, move_robot)),
            ("rotation", lambda: self._stage_rotation_with_force_baseline(get_observation, move_robot)),
            ("grvs_align", lambda: self._stage_grvs_align(get_observation, move_robot)),
            ("insert", lambda: self._stage_insert_with_metric(get_observation, move_robot)),
        )
        for name, stage in stages:
            send_feedback(f"grvs training: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"GRVS training failed at stage: {name}")
                    self._record_episode_metric(success=False, stage=name)
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"GRVS training exception at {name}: {exc}")
                self._record_episode_metric(
                    success=False,
                    stage=name,
                    error=str(exc),
                )
                send_feedback(f"failed: {name} exception")
                return False

        self.get_logger().info("SfpGrvsTrainingPolicy done")
        self._record_episode_metric(success=True, stage="done")
        send_feedback("grvs training done")
        return True
