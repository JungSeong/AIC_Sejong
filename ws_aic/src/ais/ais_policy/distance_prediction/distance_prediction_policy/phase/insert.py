"""Insert phase helpers for DebugSfpDistancePolicy."""

from __future__ import annotations

import math

from distance_prediction_policy.config import DistancePredictionConfig


def run_insert_stage(policy, get_observation, move_robot) -> bool:
    policy.get_logger().info("[stage 5/5] insert start")
    obs = get_observation()
    pose = policy._tcp_pose(obs)
    if pose is None:
        policy.get_logger().error("insert failed: missing TCP pose")
        return False

    max_depth = float(DistancePredictionConfig.MAX_INSERT_DEPTH_M)
    step_m = float(DistancePredictionConfig.MAX_DOWN_STEP_M)
    steps = min(
        int(math.ceil(max_depth / max(step_m, 1e-6))),
        DistancePredictionConfig.INSERT_MAX_STEPS,
    )
    start_z = float(pose.position.z)
    baseline_force = policy._force_norm(obs)
    if baseline_force is None:
        policy.get_logger().warn("insert force baseline unavailable; force guard disabled")
    else:
        policy.get_logger().info(
            f"insert force baseline: {baseline_force:.2f}N, "
            f"delta_limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
        )

    for step in range(steps):
        obs = get_observation()
        force = policy._force_norm(obs)
        force_delta = None
        if force is not None and baseline_force is not None:
            force_delta = force - baseline_force
        if (
            force_delta is not None
            and force_delta > DistancePredictionConfig.FORCE_LIMIT_N
        ):
            policy.get_logger().warn(
                "insert force delta limit: "
                f"force={force:.2f}N, baseline={baseline_force:.2f}N, "
                f"delta={force_delta:.2f}N, "
                f"limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
            )
            return False

        current = policy._tcp_pose(obs) or pose
        target_pose = policy._copy_pose(current)
        target_pose.position.z = float(start_z - (step + 1) * step_m)
        if policy._target_orientation is not None:
            target_pose.orientation = policy._copy_quaternion(policy._target_orientation)
        policy.set_pose_target(
            move_robot=move_robot,
            pose=target_pose,
            stiffness=list(policy._insertion_stiffness()),
            damping=list(policy._insertion_damping()),
        )
        if step == 0 or step % 10 == 0:
            force_text = ""
            if force is not None and force_delta is not None:
                force_text = f", force_delta={force_delta:+.2f}N"
            policy.get_logger().info(
                f"insert[{step:03d}]: dz={-(step + 1) * step_m * 1000:.1f}mm"
                f"{force_text}"
            )
        policy.sleep_for(DistancePredictionConfig.DT)

    if DistancePredictionConfig.SETTLE_AFTER_INSERT_S > 0:
        policy.sleep_for(DistancePredictionConfig.SETTLE_AFTER_INSERT_S)
    policy.get_logger().info("[stage 5/5] insert done")
    return True
