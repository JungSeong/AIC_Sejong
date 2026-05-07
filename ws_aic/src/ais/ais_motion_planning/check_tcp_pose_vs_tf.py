#!/usr/bin/env python3
"""
Compare Observation.controller_state.tcp_pose against live TF.

This answers two practical questions:
  1. Is controller_state.tcp_pose the same as TF base_link -> gripper/tcp?
  2. If yes, can we recover base_link -> tool0 by removing the fixed
     tool0 -> gripper/tcp offset from controller_state.tcp_pose?

Run from the workspace source directory, for example:
  pixi run python ais/ais_motion_planning/check_tcp_pose_vs_tf.py
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from aic_model_interfaces.msg import Observation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion xyzw -> 3x3 rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 transform matrix."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_to_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def transform_to_matrix(transform) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4 transform matrix."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_to_matrix(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def translation_z_matrix(z: float) -> np.ndarray:
    """Fixed tool0 -> tcp transform when the only offset is local +Z."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[2, 3] = z
    return matrix


def position_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two transform origins, in millimeters."""
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest angle between two transform rotations, in degrees."""
    r_delta = a[:3, :3].T @ b[:3, :3]
    cos_theta = (np.trace(r_delta) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


class TcpPoseTfChecker(Node):
    def __init__(self) -> None:
        super().__init__("check_tcp_pose_vs_tf")

        self.declare_parameter("observation_topic", "observations")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "gripper/tcp")
        self.declare_parameter("tool0_frame", "tool0")
        self.declare_parameter("tool0_to_tcp_z", 0.1965)
        self.declare_parameter("log_every", 30)

        self.observation_topic = (
            self.get_parameter("observation_topic").get_parameter_value().string_value
        )
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.tcp_frame = self.get_parameter("tcp_frame").get_parameter_value().string_value
        self.tool0_frame = self.get_parameter("tool0_frame").get_parameter_value().string_value
        self.tool0_to_tcp_z = (
            self.get_parameter("tool0_to_tcp_z").get_parameter_value().double_value
        )
        self.log_every = max(
            1, self.get_parameter("log_every").get_parameter_value().integer_value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sample_count = 0
        self.last_tf_error = ""

        self.create_subscription(
            Observation,
            self.observation_topic,
            self.on_observation,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Listening on '%s'. Comparing controller_state.tcp_pose with TF %s -> %s. "
            "Assuming tool0 -> tcp is local +Z %.4f m."
            % (
                self.observation_topic,
                self.base_frame,
                self.tcp_frame,
                self.tool0_to_tcp_z,
            )
        )

    def lookup_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        return transform_to_matrix(tf_msg.transform)

    def on_observation(self, msg: Observation) -> None:
        self.sample_count += 1

        try:
            t_base_tcp_tf = self.lookup_matrix(self.base_frame, self.tcp_frame)
            t_base_tool0_tf = self.lookup_matrix(self.base_frame, self.tool0_frame)
        except TransformException as exc:
            error_text = str(exc)
            if error_text != self.last_tf_error or self.sample_count % self.log_every == 0:
                self.get_logger().warn(f"TF lookup failed: {error_text}")
                self.last_tf_error = error_text
            return

        t_base_tcp_obs = pose_to_matrix(msg.controller_state.tcp_pose)
        t_tool0_tcp = translation_z_matrix(self.tool0_to_tcp_z)
        t_base_tool0_from_obs = t_base_tcp_obs @ np.linalg.inv(t_tool0_tcp)

        tcp_pos_mm = position_error_mm(t_base_tcp_obs, t_base_tcp_tf)
        tcp_rot_deg = rotation_error_deg(t_base_tcp_obs, t_base_tcp_tf)
        tool0_pos_mm = position_error_mm(t_base_tool0_from_obs, t_base_tool0_tf)
        tool0_rot_deg = rotation_error_deg(t_base_tool0_from_obs, t_base_tool0_tf)

        if self.sample_count % self.log_every != 0:
            return

        obs_tcp_xyz = t_base_tcp_obs[:3, 3]
        tf_tcp_xyz = t_base_tcp_tf[:3, 3]
        obs_tool0_xyz = t_base_tool0_from_obs[:3, 3]
        tf_tool0_xyz = t_base_tool0_tf[:3, 3]

        self.get_logger().info(
            "\n"
            f"[sample {self.sample_count}]\n"
            f"  tcp_pose vs TF({self.base_frame}->{self.tcp_frame}): "
            f"pos_err={tcp_pos_mm:.3f} mm, rot_err={tcp_rot_deg:.4f} deg\n"
            f"    obs tcp xyz = [{obs_tcp_xyz[0]:+.5f}, {obs_tcp_xyz[1]:+.5f}, {obs_tcp_xyz[2]:+.5f}]\n"
            f"    tf  tcp xyz = [{tf_tcp_xyz[0]:+.5f}, {tf_tcp_xyz[1]:+.5f}, {tf_tcp_xyz[2]:+.5f}]\n"
            f"  derived tool0 vs TF({self.base_frame}->{self.tool0_frame}): "
            f"pos_err={tool0_pos_mm:.3f} mm, rot_err={tool0_rot_deg:.4f} deg\n"
            f"    obs-derived tool0 xyz = [{obs_tool0_xyz[0]:+.5f}, {obs_tool0_xyz[1]:+.5f}, {obs_tool0_xyz[2]:+.5f}]\n"
            f"    tf          tool0 xyz = [{tf_tool0_xyz[0]:+.5f}, {tf_tool0_xyz[1]:+.5f}, {tf_tool0_xyz[2]:+.5f}]"
        )


def main() -> None:
    rclpy.init()
    node = TcpPoseTfChecker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
