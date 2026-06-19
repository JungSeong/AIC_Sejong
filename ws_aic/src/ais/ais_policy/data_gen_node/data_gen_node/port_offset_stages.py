from __future__ import annotations

"""Stage runner for PortOffsetCollect."""

import os
import time
from typing import Any

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from data_gen_node.port_offset_config import (
    APPROACH_DAMPING,
    APPROACH_DT,
    APPROACH_NEAR_DAMPING,
    APPROACH_NEAR_STIFFNESS,
    APPROACH_NEAR_Z_OFFSET_M,
    APPROACH_RETRY_DT,
    APPROACH_SETTLE_S,
    APPROACH_STEPS,
    APPROACH_STIFFNESS,
    APPROACH_TCP_OFFSET,
    APPROACH_VISION_RETRIES,
    DAMPING_DEFAULT,
    INITIAL_LIFT_DT,
    INITIAL_LIFT_M,
    INITIAL_LIFT_SETTLE_S,
    INITIAL_LIFT_STEPS,
    STIFFNESS_DEFAULT,
)
from data_gen_node.port_offset_geometry import interp_profile


def _copy_quaternion(quat: Quaternion) -> Quaternion:
    return Quaternion(
        x=float(quat.x),
        y=float(quat.y),
        z=float(quat.z),
        w=float(quat.w),
    )


def _copy_pose(pose: Pose) -> Pose:
    return Pose(
        position=Point(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
        ),
        orientation=_copy_quaternion(pose.orientation),
    )


def _tcp_pose(observation) -> Pose | None:
    if observation is None:
        return None
    return _copy_pose(observation.controller_state.tcp_pose)


def _follow_pose(
    self,
    *,
    move_robot: MoveRobotCallback,
    start_pose: Pose,
    target_pose: Pose,
    steps: int,
    stiffness: list[float],
    damping: list[float],
    dt: float,
    label: str,
) -> None:
    """현재 TCP pose에서 목표 pose까지 위치를 S-curve로 보간해 순차 명령한다."""
    start = np.array(
        [start_pose.position.x, start_pose.position.y, start_pose.position.z],
        dtype=float,
    )
    target = np.array(
        [target_pose.position.x, target_pose.position.y, target_pose.position.z],
        dtype=float,
    )
    step_count = max(1, int(steps))
    for index in range(step_count):
        fraction = interp_profile((index + 1) / float(step_count), quintic=True)
        position = start * (1.0 - fraction) + target * fraction
        pose = Pose(
            position=Point(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            orientation=_copy_quaternion(target_pose.orientation),
        )
        self.set_pose_target(
            move_robot,
            pose,
            stiffness=stiffness,
            damping=damping,
        )
        if index == 0 or index == step_count - 1:
            self.get_logger().info(
                f"{label}: waypoint {index + 1}/{step_count} "
                f"tcp=({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f})"
            )
        self.sleep_for(dt)


def _configure_port_collect_control(self, task: Task) -> dict[str, Any]:
    """포트 타입별 접근/수집 제어 파라미터를 설정하고 stage context 일부를 반환한다."""
    port_kw = "sfp" if "sfp" in task.port_type.lower() else "sc"
    if port_kw == "sc":
        self._planner.i_gain = 0.07
        self._planner.max_integrator_windup = 0.06
        approach_stiffness = [280.0, 250.0, 250.0, 50.0, 50.0, 50.0]
        approach_damping = [87.0, 80.0, 80.0, 20.0, 20.0, 20.0]
    else:
        self._planner.i_gain = float(
            os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15")
        )
        self._planner.max_integrator_windup = 0.08
        approach_stiffness = STIFFNESS_DEFAULT
        approach_damping = DAMPING_DEFAULT

    return {
        "port_kw": port_kw,
        "approach_stiffness": APPROACH_STIFFNESS,
        "approach_damping": APPROACH_DAMPING,
        "lift_stiffness": APPROACH_NEAR_STIFFNESS,
        "lift_damping": APPROACH_NEAR_DAMPING,
        "collect_stiffness": approach_stiffness,
        "collect_damping": approach_damping,
    }


def _check_and_start_recording(
    self,
    ctx: dict[str, Any],
    obs,
    phase: str = "trigger",
    step_idx: int = 0,
    detection=None,
    results=None,
    bgr=None,
) -> bool:
    """YOLO가 처음 포트를 검출한 순간부터 episode recording을 시작한다."""
    if ctx["recording_started"] or obs is None:
        return False
    if detection is None or results is None or bgr is None:
        detections, results_by_camera, bgrs_by_camera = self._detect_ports_from_obs(
            obs,
            ctx["port_kw"],
            self._yolo_trigger_conf,
        )
        detection, results, bgr, _ = self._select_yolo_view(
            detections,
            results_by_camera,
            bgrs_by_camera,
        )
    if detection is None:
        return False

    self._save_yolo_debug_frame(
        bgr,
        results,
        detection,
        ctx["task"],
        phase,
        step_idx,
    )
    ctx["recording_started"] = True
    self._log_yolo_detection(
        detection,
        "[PortOffsetCollect] YOLO DETECTION -> Recording Started",
    )
    return True


def _detect_and_update_tracking(
    self,
    ctx: dict[str, Any],
    obs,
    phase: str,
    step_idx: int,
):
    """YOLO worker에 검출을 요청하고 최신 cache를 읽어 tracking 상태를 갱신한다."""
    interval = (
        self._yolo_track_interval_steps
        if ctx["yolo_tracking_started"]
        else self._yolo_search_interval_steps
    )
    if step_idx % interval == 0:
        self._submit_yolo_detection(obs, ctx["port_kw"], self._yolo_align_conf)

    detection, results, bgr, detections_by_camera, selected_camera = (
        self._get_cached_yolo_detection(ctx["port_kw"])
    )
    if detection is not None:
        if not ctx["yolo_tracking_started"]:
            ctx["yolo_tracking_started"] = True
            self._check_and_start_recording(
                ctx,
                obs,
                f"{phase}_yolo_handoff",
                step_idx,
                detection,
                results,
                bgr,
            )
            self.get_logger().info(
                f"[PortOffsetCollect] {phase}: YOLO detection acquired "
                f"camera={selected_camera} "
                f"views={sum(det is not None for det in detections_by_camera.values())}/3"
            )
        elif not ctx["recording_started"]:
            self._check_and_start_recording(
                ctx,
                obs,
                phase,
                step_idx,
                detection,
                results,
                bgr,
            )
    return detection, results, bgr, detections_by_camera, selected_camera


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
    """YOLO multi-view triangulation으로 포트 위치를 구한 뒤 Near pose로 접근한다."""
    self.get_logger().info(
        f"━━━ Phase 1-A: YOLO Triangulation Approach "
        f"(near_z={APPROACH_NEAR_Z_OFFSET_M * 1000:.1f}mm, "
        f"tcp_offset=({APPROACH_TCP_OFFSET[0] * 1000:+.1f}, "
        f"{APPROACH_TCP_OFFSET[1] * 1000:+.1f}, "
        f"{APPROACH_TCP_OFFSET[2] * 1000:+.1f})mm) ━━━"
    )
    start_pose = _tcp_pose(get_observation())
    if start_pose is None:
        self.get_logger().error("[PortOffsetCollect] approach failed: missing TCP pose")
        return False

    port_point = None
    port_extras: dict[str, Any] = {}
    detections_by_camera = {}
    for attempt in range(max(1, APPROACH_VISION_RETRIES)):
        obs = get_observation()
        detection, results, bgr, detections_by_camera, selected_camera = (
            self._detect_and_update_tracking(ctx, obs, "approach", attempt)
        )
        if detection is None:
            self.sleep_for(APPROACH_RETRY_DT)
            continue
        if not ctx["recording_started"]:
            self._check_and_start_recording(
                ctx,
                obs,
                "approach_yolo_triangulation",
                attempt,
                detection,
                results,
                bgr,
            )
        port_point, port_extras = self._triangulate_yolo_port(obs, detections_by_camera)
        if port_point is not None:
            self.get_logger().info(
                "[PortOffsetCollect] YOLO triangulated port: "
                f"attempt={attempt + 1}, views={port_extras['triangulated_port_views']}, "
                f"pairs={port_extras['triangulated_port_pairs']}, "
                f"base=({port_point[0]:+.4f}, {port_point[1]:+.4f}, {port_point[2]:+.4f}), "
                f"camera={selected_camera}"
            )
            break
        self.sleep_for(APPROACH_RETRY_DT)

    if port_point is None:
        self.get_logger().error(
            "[PortOffsetCollect] approach failed: YOLO triangulated port unavailable"
        )
        return False

    target = (
        np.asarray(port_point, dtype=float)
        + np.array([0.0, 0.0, APPROACH_NEAR_Z_OFFSET_M], dtype=float)
        + APPROACH_TCP_OFFSET
    )
    target_pose = Pose(
        position=Point(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
        ),
        orientation=_copy_quaternion(start_pose.orientation),
    )
    self.get_logger().info(
        "[PortOffsetCollect] Approach target: "
        f"near_z={APPROACH_NEAR_Z_OFFSET_M * 1000:.1f}mm, "
        f"tcp_offset=({APPROACH_TCP_OFFSET[0] * 1000:+.1f}, "
        f"{APPROACH_TCP_OFFSET[1] * 1000:+.1f}, "
        f"{APPROACH_TCP_OFFSET[2] * 1000:+.1f})mm, "
        f"target_tcp=({target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f})"
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
    ctx["approach_reached_triangulation_stop"] = True
    ctx["last_triangulated_port"] = {
        "x": float(port_point[0]),
        "y": float(port_point[1]),
        "z": float(port_point[2]),
        **port_extras,
    }
    if not ctx["recording_started"]:
        ctx["recording_started"] = True
        self.get_logger().warn(
            "[PortOffsetCollect] recording started after triangulated approach without a debug handoff frame"
        )
    ctx["phase_step_counts"]["approach"] += 1
    self.get_logger().info("[PortOffsetCollect] YOLO triangulation approach done")
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
        f"z={self._triangulation_stop_z_offset:.4f}m) ━━━"
    )
    collect_steps = max(1, self.collect_steps)
    for collect_idx in range(collect_steps):
        try:
            current_port_tf = self._lookup_transform("base_link", ctx["port_frame"])
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
                z_offset=self._triangulation_stop_z_offset,
                reset_xy_integrator=False,
            )
            extras["z_offset"] = float(self._triangulation_stop_z_offset)
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
            self.sleep_for(max(self.step_sleep_sec, self.collect_capture_settle_sec))
            save_obs = get_observation()
            save_port_tf = self._lookup_transform("base_link", ctx["port_frame"])
            save_raw_plug_tf = self._lookup_transform(
                "base_link",
                ctx["cable_tip_frame"],
            )
            save_plug_tf = self._shift_transform_origin(
                save_raw_plug_tf,
                ctx["plug_reference_offset_local"],
            )
            extras.update(
                self._plug_location_label_in_base_frame(
                    save_port_tf,
                    save_plug_tf,
                )
            )
            if ctx["recording_started"]:
                self._save_vision_offset_sample(
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
                ctx["phase_step_counts"]["collect"] += 1
        except TransformException:
            pass
        self.sleep_for(self.step_sleep_sec)
    return True


def insert_cable(
    self,
    task: Task,
    get_observation: GetObservationCallback,
    move_robot: MoveRobotCallback,
    send_feedback: SendFeedbackCallback,
):
    self._task = task
    self._planner.reset()
    send_feedback("data collect running")

    episode_name = time.strftime("%Y%m%d_%H%M%S") + f"_{task.id}"
    episode_dir = self.capture_root / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    phase_step_counts = {"lift_up": 0, "approach": 0, "collect": 0}
    if not self._vision_offset_record_enabled:
        self.get_logger().warn(
            "[PortOffsetCollect] Vision offset recording disabled"
        )
        return self._finish_data_collection_episode(
            episode_dir=episode_dir,
            task=task,
            phase_step_counts=phase_step_counts,
            status="recording_disabled",
            detail="AIC_VISION_OFFSET_RECORD disabled",
        )

    port_frame = self._select_port_frame(task)
    cable_tip_frame = self._select_cable_tip_frame(task)
    if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf(
        "base_link",
        cable_tip_frame,
    ):
        return self._finish_data_collection_episode(
            episode_dir=episode_dir,
            task=task,
            phase_step_counts=phase_step_counts,
            status="tf_unavailable",
            detail=(
                f"Missing required TF: port_frame={port_frame}, "
                f"cable_tip_frame={cable_tip_frame}"
            ),
        )
    self.get_logger().info(
        f"[PortOffsetCollect] SELECTED FRAMES: port_frame={port_frame}, cable_tip_frame={cable_tip_frame}"
    )

    plug_reference_offset_local = self._plug_reference_offset_local(
        task,
        cable_tip_frame,
    )
    plug_reference_metadata = self._plug_reference_metadata(
        task,
        cable_tip_frame,
        plug_reference_offset_local,
    )
    self.get_logger().info(
        "[PortOffsetCollect] Plug reference point: "
        f"{plug_reference_metadata['point_name']} "
        f"frame={cable_tip_frame} "
        f"offset={plug_reference_metadata['local_offset_xyz_m']}"
    )
    control_ctx = self._configure_port_collect_control(task)
    if not self._wait_for_yolo_model(control_ctx["port_kw"]):
        return self._finish_data_collection_episode(
            episode_dir=episode_dir,
            task=task,
            phase_step_counts=phase_step_counts,
            status="yolo_unavailable",
            detail="YOLO model is required for triangulation approach",
        )

    ctx = {
        "task": task,
        "episode_name": episode_name,
        "episode_dir": episode_dir,
        "phase_step_counts": phase_step_counts,
        "port_frame": port_frame,
        "cable_tip_frame": cable_tip_frame,
        "plug_reference_offset_local": plug_reference_offset_local,
        "plug_reference_metadata": plug_reference_metadata,
        "recording_started": False,
        "yolo_tracking_started": False,
        "approach_reached_triangulation_stop": False,
    }
    ctx.update(control_ctx)

    stages = (
        ("lift_up", lambda: self._stage_lift_up(ctx, get_observation, move_robot)),
        ("approach", lambda: self._stage_approach(ctx, get_observation, move_robot)),
        ("collect", lambda: self._stage_collect(ctx, get_observation, move_robot)),
    )
    for stage_name, run_stage in stages:
        self.get_logger().info(f"[PortOffsetCollect] stage start: {stage_name}")
        if not run_stage():
            return self._finish_data_collection_episode(
                episode_dir=episode_dir,
                task=task,
                phase_step_counts=phase_step_counts,
                status=f"{stage_name}_failed",
            )

    self.sleep_for(0.5)
    return self._finish_data_collection_episode(
        episode_dir=episode_dir,
        task=task,
        phase_step_counts=phase_step_counts,
        status="ok",
    )
