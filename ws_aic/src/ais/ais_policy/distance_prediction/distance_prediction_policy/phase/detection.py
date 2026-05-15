"""Detection phase helpers for DebugSfpDistancePolicy."""

from __future__ import annotations

from typing import Optional

import numpy as np

from distance_prediction_policy.config import DistancePredictionConfig


def run_detect_stage(policy, get_observation) -> bool:
    policy.get_logger().info("[stage 2/5] detect start")
    policy._vision_debug_save_enabled = True
    vision = policy._vision_for_port_type(policy._port_type())
    vision.set_debug_task_context(
        target_module_name=str(getattr(policy._task, "target_module_name", "") or ""),
        port_name=str(getattr(policy._task, "port_name", "") or ""),
        plug_name=str(getattr(policy._task, "plug_name", "") or ""),
        cable_name=str(getattr(policy._task, "cable_name", "") or ""),
        port_type=policy._port_type(),
    )
    vision.start_detection(enable_debug_save=True, reset_counts=True)
    try:
        obs = get_observation()
        start_pose = policy._tcp_pose(obs)
        if start_pose is None:
            policy.get_logger().error("detect failed: missing TCP pose")
            return False

        port = estimate_port(policy, get_observation)
        if port is None:
            policy.get_logger().error("detect failed: YOLO port estimate unavailable")
            return False

        policy._cached_port_base = port
        policy._target_orientation = policy._target_wrist_orientation(start_pose)
        policy.get_logger().info(
            "detect cached: "
            f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
            f"axis={DistancePredictionConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
            f"angle={policy._manual_rotation_deg():+.2f}deg"
        )
        policy.get_logger().info("[stage 2/5] detect done")
        return True
    finally:
        for estimator in policy._vision_by_port_type.values():
            estimator.stop_detection()
            estimator.set_debug_save_enabled(False)
        policy._vision_debug_save_enabled = False


def estimate_port(policy, get_observation) -> Optional[np.ndarray]:
    port_hint = str(getattr(policy._task, "port_name", "") or "")
    target_module_name = str(getattr(policy._task, "target_module_name", "") or "")
    port_type = policy._port_type()
    target_class_id = policy._target_class_id(port_type)
    vision = policy._vision_for_port_type(port_type)
    for attempt in range(DistancePredictionConfig.APPROACH_VISION_RETRIES):
        obs = get_observation()
        port = vision.estimate(
            obs,
            target_class_id,
            port_hint=port_hint,
            target_module_name=target_module_name,
        )
        if port is not None:
            policy.get_logger().info(
                "YOLO port estimate: "
                f"attempt={attempt + 1}, "
                f"type={port_type}, "
                f"target={target_module_name}, "
                f"port={port_hint}, "
                f"class_id={target_class_id}, "
                f"base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
            )
            return port
        policy.sleep_for(DistancePredictionConfig.APPROACH_RETRY_DT)
    return None
