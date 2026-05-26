"""Approach phase helpers for DebugSfpDistancePolicy."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Pose

from distance_prediction_policy.config import DistancePredictionConfig


def run_approach_stage(policy, get_observation, move_robot) -> bool:
    policy.get_logger().info("[stage 3/5] approach start")
    obs = get_observation()
    start_pose = policy._tcp_pose(obs)
    if start_pose is None:
        policy.get_logger().error("approach failed: missing TCP pose")
        return False

    port = policy._cached_port_base
    if port is None:
        policy.get_logger().error("approach failed: missing cached YOLO port estimate")
        return False

    target_orientation = policy._target_orientation
    if target_orientation is None:
        target_orientation = policy._target_wrist_orientation(start_pose)
        policy._target_orientation = target_orientation

    tcp_offset = np.array(
        [
            DistancePredictionConfig.APPROACH_TCP_OFFSET_X_M,
            DistancePredictionConfig.APPROACH_TCP_OFFSET_Y_M,
            DistancePredictionConfig.APPROACH_TCP_OFFSET_Z_M,
        ],
        dtype=np.float64,
    )
    initial_z_offset = policy._initial_approach_z_offset()
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
                orientation=policy._copy_quaternion(target_orientation),
            ),
            target,
        )

    far_pose, far_target = make_approach_pose(initial_z_offset)
    near_pose, near_target = make_approach_pose(near_z_offset)
    policy.get_logger().info(
        "approach targets: "
        f"initial_z_plus={initial_z_offset*1000:.1f}mm, "
        f"near_z_plus={near_z_offset*1000:.1f}mm, "
        f"tcp_offset=({tcp_offset[0]*1000:+.1f}, "
        f"{tcp_offset[1]*1000:+.1f}, {tcp_offset[2]*1000:+.1f})mm, "
        f"far_tcp=({far_target[0]:+.4f}, {far_target[1]:+.4f}, {far_target[2]:+.4f}), "
        f"near_tcp=({near_target[0]:+.4f}, {near_target[1]:+.4f}, {near_target[2]:+.4f})"
    )
    policy._follow_pose(
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
    near_start_pose = policy._tcp_pose(near_start_obs) or far_pose
    policy._follow_pose(
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
        policy.get_logger().info(
            f"approach settle: {DistancePredictionConfig.APPROACH_SETTLE_S:.2f}s"
        )
        policy.sleep_for(DistancePredictionConfig.APPROACH_SETTLE_S)
    policy.get_logger().info("[stage 3/5] approach done")
    return True
