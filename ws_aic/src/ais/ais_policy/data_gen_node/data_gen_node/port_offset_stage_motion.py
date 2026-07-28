from __future__ import annotations

"""PortOffsetCollect의 lift-up, approach, collect motion stage."""

import time
from typing import Any

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
)
from tf2_ros import TransformException

from data_gen_node.port_offset_config import (
    APPROACH_DAMPING,
    APPROACH_DT,
    APPROACH_NEAR_DAMPING,
    APPROACH_NEAR_STIFFNESS,
    APPROACH_NEAR_Z_OFFSET_M,
    APPROACH_SETTLE_S,
    APPROACH_STEPS,
    APPROACH_STIFFNESS,
    INITIAL_LIFT_DT,
    INITIAL_LIFT_M,
    INITIAL_LIFT_SETTLE_S,
    INITIAL_LIFT_STEPS,
)
from data_gen_node.port_offset_stage_common import (
    _copy_pose,
    _follow_pose,
    _tcp_pose,
)


def _stage_lift_up(
    self,
    ctx: dict[str, Any],
    get_observation: GetObservationCallback,
    move_robot: MoveRobotCallback,
) -> bool:
    """FinalPolicy와 동일하게 초기 TCP를 위로 들어 전체 task board 관측을 확보한다."""
    lift_m = float(INITIAL_LIFT_M)
    self.get_logger().info(
        f"[PortOffsetCollect] lift_up start: dz={lift_m * 1000.0:.1f}mm"
    )
    if abs(lift_m) < 1e-9:
        self.get_logger().info("[PortOffsetCollect] lift_up skipped: dz is 0")
        return True

    start_pose = _tcp_pose(get_observation())
    if start_pose is None:
        self.get_logger().error("[PortOffsetCollect] lift_up failed: missing TCP pose")
        return False

    target_pose = _copy_pose(start_pose)
    target_pose.position.z = float(target_pose.position.z + lift_m)
    _follow_pose(
        self,
        move_robot=move_robot,
        start_pose=start_pose,
        target_pose=target_pose,
        steps=INITIAL_LIFT_STEPS,
        stiffness=ctx["lift_stiffness"],
        damping=ctx["lift_damping"],
        dt=INITIAL_LIFT_DT,
        label="lift_up",
    )
    if INITIAL_LIFT_SETTLE_S > 0:
        self.sleep_for(INITIAL_LIFT_SETTLE_S)
    ctx["phase_step_counts"]["lift_up"] += 1
    self.get_logger().info("[PortOffsetCollect] lift_up done")
    return True


def _stage_approach(
    self,
    ctx: dict[str, Any],
    get_observation: GetObservationCallback,
    move_robot: MoveRobotCallback,
) -> bool:
    """ROS 2 Ground Truth port/plug TF와 planner로 Near pose에 접근한다."""
    self.get_logger().info(
        "━━━ Phase 1-A: ROS 2 Ground Truth TF Approach "
        f"(near_z={APPROACH_NEAR_Z_OFFSET_M * 1000:.1f}mm) ━━━"
    )
    start_pose = _tcp_pose(get_observation())
    if start_pose is None:
        self.get_logger().error("[PortOffsetCollect] approach failed: missing TCP pose")
        return False

    try:
        port_tf = ctx["port_tf_snapshot"].transform
        raw_plug_tf = self._lookup_transform("base_link", ctx["cable_tip_frame"])
        plug_tf = self._shift_transform_origin(
            raw_plug_tf,
            ctx["plug_reference_offset_local"],
        )
        gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
        target_pose, planner_extras = self._planner.build_pose(
            port_tf,
            plug_tf,
            gripper_tf,
            z_offset=APPROACH_NEAR_Z_OFFSET_M,
            reset_xy_integrator=True,
        )
    except TransformException as exc:
        self.get_logger().error(
            f"[PortOffsetCollect] Ground Truth TF approach failed: {exc}"
        )
        return False

    ctx["recording_started"] = True
    self.get_logger().info(
        "[PortOffsetCollect] Ground Truth Approach target: "
        f"near_z={APPROACH_NEAR_Z_OFFSET_M * 1000:.1f}mm, "
        f"target_tcp=({planner_extras['target_x']:+.4f}, "
        f"{planner_extras['target_y']:+.4f}, "
        f"{planner_extras['target_z']:+.4f})"
    )
    _follow_pose(
        self,
        move_robot=move_robot,
        start_pose=start_pose,
        target_pose=target_pose,
        steps=APPROACH_STEPS,
        stiffness=ctx["approach_stiffness"],
        damping=ctx["approach_damping"],
        dt=APPROACH_DT,
        label="approach",
    )
    if APPROACH_SETTLE_S > 0:
        self.sleep_for(APPROACH_SETTLE_S)
    ctx["approach_reached_ground_truth_target"] = True
    ctx["phase_step_counts"]["approach"] += 1
    self.get_logger().info("[PortOffsetCollect] Ground Truth TF approach done")
    return True

def _stage_collect(
    self,
    ctx: dict[str, Any],
    get_observation: GetObservationCallback,
    move_robot: MoveRobotCallback,
) -> bool:
    """포트 주변 XYZ/RPY offset sample을 순회하며 이미지와 label을 저장한다."""
    self.get_logger().info(
        f"━━━ Phase 1-B: COLLECT {self.collect_pattern} "
        f"(max_radius={self.collect_gaussian_max_radius*1000:.1f}mm, "
        f"sigma={self.collect_gaussian_sigma*1000:.1f}mm, "
        f"spiral_radius={self.collect_start_radius*1000:.1f}->"
        f"{self.collect_end_radius*1000:.1f}mm, "
        f"turns={self.collect_turns:.2f}, steps={self.collect_steps}, "
        f"base_z={self.collect_base_z_offset_m:.4f}m) ━━━"
    )
    collect_steps = max(1, self.collect_steps)
    for collect_idx in range(collect_steps):
        try:
            current_port_tf = ctx["port_tf_snapshot"].transform
            raw_plug_tf = self._lookup_transform("base_link", ctx["cable_tip_frame"])
            plug_tf = self._shift_transform_origin(
                raw_plug_tf,
                ctx["plug_reference_offset_local"],
            )
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            pose, extras = self._planner.build_pose(
                current_port_tf,
                plug_tf,
                gripper_tf,
                z_offset=self.collect_base_z_offset_m,
                reset_xy_integrator=False,
            )
            extras["z_offset"] = float(self.collect_base_z_offset_m)
            extras["plug_reference"] = ctx["plug_reference_metadata"]
            pose, collect_extras = self._apply_collect_offset(
                pose,
                current_port_tf,
                extras.get("port_axis"),
                collect_idx,
            )
            extras.update(collect_extras)
            self.get_logger().info(
                f"COLLECT step={collect_idx}/{collect_steps} "
                f"offset=({extras['collect_local_x']*1000:+.2f}, "
                f"{extras['collect_local_y']*1000:+.2f}, "
                f"{extras['collect_local_z']*1000:+.2f})mm "
                f"rpy=({extras['collect_local_roll_deg']:+.2f}, "
                f"{extras['collect_local_pitch_deg']:+.2f}, "
                f"{extras['collect_local_yaw_deg']:+.2f})deg"
            )

            self.set_pose_target(
                move_robot,
                pose,
                stiffness=ctx["collect_stiffness"],
                damping=ctx["collect_damping"],
            )
            save_obs, timestamps = self._wait_for_synchronized_observation(
                get_observation
            )
            if save_obs is None:
                self.get_logger().warn(
                    self._collect_log_text(
                        "[PortOffsetCollect] Skipping sample: "
                        f"{timestamps.get('rejection_reason', 'timestamp_gating_failed')}, "
                        f"skew_ns={timestamps.get('skew_ns', {})}, "
                        f"tolerance_ns={self.collect_sync_tolerance_ns}",
                        "yellow",
                    )
                )
                self.sleep_for(self.step_sleep_sec)
                continue

            capture_stamp = save_obs.center_image.header.stamp
            tf_wait_start_ns = time.monotonic_ns()
            save_port_stamped = ctx["port_tf_snapshot"]
            save_raw_plug_stamped = self._lookup_transform_at(
                "base_link",
                ctx["cable_tip_frame"],
                capture_stamp,
            )
            timestamps.setdefault("wait_ns", {})["tf"] = (
                time.monotonic_ns() - tf_wait_start_ns
            )
            sync_valid, timestamps = self._tf_sync_metadata(
                timestamps,
                {
                    "port": save_port_stamped,
                    "plug": save_raw_plug_stamped,
                },
                static_sources={"port"},
            )
            if not sync_valid:
                self.get_logger().warn(
                    self._collect_log_text(
                        "[PortOffsetCollect] Skipping sample: "
                        f"{timestamps.get('rejection_reason', 'tf_timestamp_gating_failed')}, "
                        f"skew_ns={timestamps.get('skew_ns', {})}, "
                        f"tolerance_ns={self.collect_sync_tolerance_ns}",
                        "yellow",
                    )
                )
                self.sleep_for(self.step_sleep_sec)
                continue

            save_port_tf = save_port_stamped.transform
            save_raw_plug_tf = save_raw_plug_stamped.transform
            save_plug_tf = self._shift_transform_origin(
                save_raw_plug_tf,
                ctx["plug_reference_offset_local"],
            )
            extras["timestamps"] = timestamps
            extras.update(
                self._plug_location_label_in_base_frame(
                    save_port_tf,
                    save_plug_tf,
                )
            )
            if ctx["recording_started"]:
                saved = self._save_vision_offset_sample(
                    episode_name=ctx["episode_name"],
                    task=ctx["task"],
                    phase="collect",
                    step_idx=ctx["phase_step_counts"]["collect"],
                    obs=save_obs,
                    port_tf=save_port_tf,
                    plug_tf=save_plug_tf,
                    pose=pose,
                    extras=extras,
                    detections_by_camera={},
                )
                if saved:
                    ctx["phase_step_counts"]["collect"] += 1
                    self.get_logger().info(
                        self._collect_log_text(
                            "[PortOffsetCollect] Saved synchronized sample: "
                            f"capture_stamp_ns={timestamps['capture_stamp_ns']}, "
                            f"skew_ns={timestamps['skew_ns']}, "
                            f"wait_ns={timestamps['wait_ns']}",
                            "green",
                        )
                    )
        except TransformException as exc:
            self.get_logger().warn(
                self._collect_log_text(
                    "[PortOffsetCollect] Skipping sample: "
                    f"TF unavailable at capture timestamp after "
                    f"{self.collect_sync_wait_timeout_sec:.3f}s ({exc})",
                    "yellow",
                )
            )
        self.sleep_for(self.step_sleep_sec)
    return True
