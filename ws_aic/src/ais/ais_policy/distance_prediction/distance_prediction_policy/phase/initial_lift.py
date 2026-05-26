"""Initial lift phase helpers for DebugSfpDistancePolicy."""

from __future__ import annotations

from distance_prediction_policy.config import DistancePredictionConfig


def run_initial_lift_stage(policy, get_observation, move_robot) -> bool:
    lift_m = float(DistancePredictionConfig.INITIAL_LIFT_M)
    policy.get_logger().info(
        f"[stage 1/5] initial_lift start: dz={lift_m * 1000.0:.1f}mm"
    )
    if abs(lift_m) < 1e-9:
        policy.get_logger().info("initial_lift skipped: configured dz is 0")
        return True

    obs = get_observation()
    start_pose = policy._tcp_pose(obs)
    if start_pose is None:
        policy.get_logger().error("initial_lift failed: missing TCP pose")
        return False

    target_pose = policy._copy_pose(start_pose)
    target_pose.position.z += lift_m
    policy._follow_pose(
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
        policy.get_logger().info(
            "initial_lift settle: "
            f"{DistancePredictionConfig.INITIAL_LIFT_SETTLE_S:.2f}s"
        )
        policy.sleep_for(DistancePredictionConfig.INITIAL_LIFT_SETTLE_S)
    policy.get_logger().info("[stage 1/5] initial_lift done")
    return True
