#!/usr/bin/env python3
"""Collect SFP distance-prediction samples near the 99 percent YOLO pose region.

Output layout:
  ws_aic/data/distance_prediction/SFP/
    samples.jsonl
    images/{left,center,right}/{episode_name}/{sample_id}_{camera}.png
    meta/generation_config_*.json

The label format matches ais_distance_prediction.model.dataset.VisionOffsetDataset:
``label.plug_tip_to_port`` stores the measured plug-tip offset in the target
port frame, with meter and millimeter fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from aic_control_interfaces.msg import ControllerState, MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3, Wrench
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from .sampling import (
        DEFAULT_APPROACH_OFFSET_MM,
        DEFAULT_CI99_X_MM,
        DEFAULT_CI99_Y_MM,
        DEFAULT_CI99_Z_ERROR_MM,
        LocalOffset,
        OffsetRangeMM,
        make_offset_grid,
        make_offset_step_grid,
        make_random_offsets,
        make_uniform_offsets,
    )
except ImportError:
    from sampling import (
        DEFAULT_APPROACH_OFFSET_MM,
        DEFAULT_CI99_X_MM,
        DEFAULT_CI99_Y_MM,
        DEFAULT_CI99_Z_ERROR_MM,
        LocalOffset,
        OffsetRangeMM,
        make_offset_grid,
        make_offset_step_grid,
        make_random_offsets,
        make_uniform_offsets,
    )


WS_AIC_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = WS_AIC_ROOT / "data" / "distance_prediction" / "SFP"
CAMERAS = ("left", "center", "right")
DEFAULT_CABLE_TIP_CANDIDATES = (
    "cable_0/sfp_tip_link",
    "cable_0/sfp_tip_tip_link",
    "cable_0/sfp_link",
)
DEFAULT_TCP_FRAME = "gripper/tcp"
SFP_PORT_NAMES = ("sfp_port_0", "sfp_port_1")
SAMPLING_MODE_CHOICES = ("step-grid", "uniform", "grid", "grid+uniform")
PORT_FRAME_MODE_CHOICES = ("entrance", "link", "auto")
DEFAULT_PORT_FRAME_MODE = "entrance"
TCP_POSE_SOURCE_CHOICES = ("controller_state", "tf")
DEFAULT_TCP_POSE_SOURCE = "controller_state"
DEFAULT_SAMPLES_PER_PORT = 125
DEFAULT_GRID_STEP_MM = 1.0
DEFAULT_MAX_PLACEMENT_AXIS_ERROR_MM = 2.0
DEFAULT_MAX_PLACEMENT_EUCLIDEAN_ERROR_MM = 3.0
DEFAULT_MAX_TCP_POSITION_ERROR_MM = -1.0
DEFAULT_MAX_CORRECTION_STEP_MM = 60.0


def parse_range_mm(value: str) -> OffsetRangeMM:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be LOW,HIGH in mm")
    low, high = float(parts[0]), float(parts[1])
    if high < low:
        raise argparse.ArgumentTypeError("range HIGH must be >= LOW")
    return OffsetRangeMM(low=low, high=high)


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return values


def quat_to_matrix_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
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


def transform_to_matrix(transform: Transform) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_to_matrix_xyzw(
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


def quat_multiply_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quat_inverse_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm_sq = sum(value * value for value in q)
    if norm_sq < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / norm_sq, -q[1] / norm_sq, -q[2] / norm_sq, -q[3] / norm_sq)


def quat_normalize_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value / norm) for value in q)


def transform_xyz(transform: Transform) -> np.ndarray:
    return np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ],
        dtype=np.float64,
    )


def pose_xyz(pose: Pose) -> np.ndarray:
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ],
        dtype=np.float64,
    )


def pose_to_transform(pose: Pose) -> Transform:
    return Transform(
        translation=Vector3(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
        ),
        rotation=Quaternion(
            x=float(pose.orientation.x),
            y=float(pose.orientation.y),
            z=float(pose.orientation.z),
            w=float(pose.orientation.w),
        ),
    )


def transform_quat_wxyz(transform: Transform) -> tuple[float, float, float, float]:
    return (
        float(transform.rotation.w),
        float(transform.rotation.x),
        float(transform.rotation.y),
        float(transform.rotation.z),
    )


def relative_transform_label(reference_tf: Transform, target_tf: Transform) -> dict[str, float]:
    reference_to_target = np.linalg.inv(transform_to_matrix(reference_tf)) @ transform_to_matrix(target_tf)
    translation = reference_to_target[:3, 3]
    rotation = reference_to_target[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag_index = int(np.argmax(np.diag(rotation)))
        if diag_index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif diag_index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm > 1e-12:
        quat /= quat_norm
    return {
        "x_m": float(translation[0]),
        "y_m": float(translation[1]),
        "z_m": float(translation[2]),
        "xy_m": float(np.linalg.norm(translation[:2])),
        "x_mm": float(translation[0] * 1000.0),
        "y_mm": float(translation[1] * 1000.0),
        "z_mm": float(translation[2] * 1000.0),
        "xy_mm": float(np.linalg.norm(translation[:2]) * 1000.0),
        "qx": float(quat[0]),
        "qy": float(quat[1]),
        "qz": float(quat[2]),
        "qw": float(quat[3]),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def image_msg_to_bgr(image_msg: Image) -> np.ndarray | None:
    if image_msg.width == 0 or image_msg.height == 0:
        return None

    height = int(image_msg.height)
    width = int(image_msg.width)
    encoding = getattr(image_msg, "encoding", "").lower()
    if encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        channels = 3

    flat = np.frombuffer(image_msg.data, dtype=np.uint8)
    step = int(getattr(image_msg, "step", 0))
    if step > 0 and flat.size >= height * step:
        rows = flat[: height * step].reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)
    else:
        expected = height * width * channels
        if flat.size < expected:
            return None
        image = flat[:expected].reshape(height, width, channels)

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image).copy()


class SfpDistanceCollector(Node):
    def __init__(self) -> None:
        super().__init__("sfp_distance_dataset_collector")
        self._latest_image: dict[str, np.ndarray] = {}
        self._camera_info: dict[str, CameraInfo] = {}
        self._latest_controller_state: ControllerState | None = None

        for camera in CAMERAS:
            self.create_subscription(
                CameraInfo,
                f"/{camera}_camera/camera_info",
                lambda msg, name=camera: self._on_camera_info(name, msg),
                10,
            )
            self.create_subscription(
                Image,
                f"/{camera}_camera/image",
                lambda msg, name=camera: self._on_image(name, msg),
                10,
            )

        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._motion_pub = self.create_publisher(
            MotionUpdate,
            "/aic_controller/pose_commands",
            10,
        )
        self.get_logger().info("SFP distance dataset collector ready.")

    def _on_camera_info(self, camera: str, msg: CameraInfo) -> None:
        self._camera_info[camera] = msg

    def _on_image(self, camera: str, msg: Image) -> None:
        image = image_msg_to_bgr(msg)
        if image is not None:
            self._latest_image[camera] = image

    def _on_controller_state(self, msg: ControllerState) -> None:
        self._latest_controller_state = msg

    def ready(self) -> bool:
        return len(self._camera_info) == len(CAMERAS) and len(self._latest_image) == len(CAMERAS)

    def lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        return self._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform

    def wait_for_tf(self, target_frame: str, source_frame: str, timeout_s: float) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                self.lookup_transform(target_frame, source_frame)
                return True
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_controller_state(self, timeout_s: float) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            if self._latest_controller_state is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def tcp_transform(self, tcp_frame: str, source: str) -> Transform:
        if source == "tf":
            return self.lookup_transform("base_link", tcp_frame)
        if self._latest_controller_state is None:
            raise RuntimeError("ControllerState not available yet.")
        return pose_to_transform(self._latest_controller_state.tcp_pose)

    def related_tf_frames(self, terms: tuple[str, ...]) -> list[str]:
        frames_text = self._tf_buffer.all_frames_as_string()
        frame_names = sorted(set(re.findall(r"Frame ([^ ]+) exists", frames_text)))
        return [name for name in frame_names if any(term in name for term in terms)]

    def log_tf_hints(self, missing_frame: str) -> None:
        terms = ("task_board", "nic_card_mount", "sfp_port")
        hints = self.related_tf_frames(terms)
        if not hints:
            self.get_logger().error(
                f"Missing TF: {missing_frame}. No task_board/NIC/SFP frames are visible."
            )
            return
        self.get_logger().error(
            f"Missing TF: {missing_frame}. Related visible TF frames: "
            + ", ".join(hints[:30])
        )

    def port_frame(self, module_name: str, port_name: str, mode: str = DEFAULT_PORT_FRAME_MODE) -> str:
        base_frame = f"task_board/{module_name}/{port_name}_link"
        entrance_frame = f"{base_frame}_entrance"
        if mode == "link":
            return base_frame
        if mode == "entrance":
            return entrance_frame
        if mode == "auto" and self.wait_for_tf("base_link", entrance_frame, timeout_s=0.5):
            return entrance_frame
        return base_frame

    def discover_module_name(
        self,
        max_mounts: int,
        port_names: tuple[str, ...],
        port_frame_mode: str,
    ) -> str | None:
        for mount_idx in range(max_mounts):
            module_name = f"nic_card_mount_{mount_idx}"
            for port_name in port_names:
                if self.wait_for_tf(
                    "base_link",
                    self.port_frame(module_name, port_name, port_frame_mode),
                    timeout_s=0.2,
                ):
                    return module_name
        return None

    def discover_cable_tip_frame(self, candidates: tuple[str, ...], timeout_s: float) -> str | None:
        for frame in candidates:
            if self.wait_for_tf("base_link", frame, timeout_s=timeout_s):
                return frame
        return None

    def move_robot_to(self, pose: Pose, stiffness: tuple[float, ...], damping: tuple[float, ...]) -> None:
        msg = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten().tolist(),
            target_damping=np.diag(damping).flatten().tolist(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        self._motion_pub.publish(msg)

    def plug_tip_to_port_label(self, port_tf: Transform, plug_tf: Transform) -> dict[str, float]:
        port_xyz = transform_xyz(port_tf)
        plug_xyz = transform_xyz(plug_tf)
        port_rotation = quat_to_matrix_xyzw(
            port_tf.rotation.x,
            port_tf.rotation.y,
            port_tf.rotation.z,
            port_tf.rotation.w,
        )
        local_offset = port_rotation.T @ (plug_xyz - port_xyz)
        return {
            "x_m": float(local_offset[0]),
            "y_m": float(local_offset[1]),
            "z_m": float(local_offset[2]),
            "xy_m": float(np.linalg.norm(local_offset[:2])),
            "x_mm": float(local_offset[0] * 1000.0),
            "y_mm": float(local_offset[1] * 1000.0),
            "z_mm": float(local_offset[2] * 1000.0),
            "xy_mm": float(np.linalg.norm(local_offset[:2]) * 1000.0),
        }

    def make_target_pose(
        self,
        port_tf: Transform,
        plug_tf: Transform,
        tcp_tf: Transform,
        local_offset: LocalOffset,
    ) -> Pose:
        port_matrix = transform_to_matrix(port_tf)
        target_plug_xyz = (port_matrix @ np.append(local_offset.vector_m(), 1.0))[:3]

        q_port = transform_quat_wxyz(port_tf)
        q_plug = transform_quat_wxyz(plug_tf)
        q_tcp = transform_quat_wxyz(tcp_tf)
        q_diff = quat_multiply_wxyz(q_port, quat_inverse_wxyz(q_plug))
        q_target_tcp = quat_normalize_wxyz(quat_multiply_wxyz(q_diff, q_tcp))

        tcp_rotation = transform_to_matrix(tcp_tf)[:3, :3]
        target_tcp_rotation = quat_to_matrix_xyzw(
            q_target_tcp[1],
            q_target_tcp[2],
            q_target_tcp[3],
            q_target_tcp[0],
        )
        tcp_to_plug_local = tcp_rotation.T @ (transform_xyz(plug_tf) - transform_xyz(tcp_tf))
        target_tcp_xyz = target_plug_xyz - target_tcp_rotation @ tcp_to_plug_local
        return Pose(
            position=Point(
                x=float(target_tcp_xyz[0]),
                y=float(target_tcp_xyz[1]),
                z=float(target_tcp_xyz[2]),
            ),
            orientation=Quaternion(
                w=float(q_target_tcp[0]),
                x=float(q_target_tcp[1]),
                y=float(q_target_tcp[2]),
                z=float(q_target_tcp[3]),
            ),
        )

    def save_sample(
        self,
        output_dir: Path,
        episode_name: str,
        sample_id: str,
        step_index: int,
        module_name: str,
        target_port_name: str,
        all_port_names: tuple[str, ...],
        target_port_frame: str,
        cable_tip_frame: str,
        tcp_frame: str,
        port_frame_mode: str,
        tcp_pose_source: str,
        command_pose: Pose,
        requested_offset: LocalOffset,
        placement_check: dict[str, Any],
    ) -> bool:
        port_tf = self.lookup_transform("base_link", target_port_frame)
        plug_tf = self.lookup_transform("base_link", cable_tip_frame)
        tcp_tf = self.tcp_transform(tcp_frame, tcp_pose_source)
        try:
            tcp_tf_frame = self.lookup_transform("base_link", tcp_frame)
            tcp_tf_diagnostic: dict[str, Any] = {
                "available": True,
                "x": float(tcp_tf_frame.translation.x),
                "y": float(tcp_tf_frame.translation.y),
                "z": float(tcp_tf_frame.translation.z),
            }
        except TransformException:
            tcp_tf_diagnostic = {"available": False}

        image_paths: dict[str, str] = {}
        for camera in CAMERAS:
            image = self._latest_image.get(camera)
            if image is None:
                return False
            image_dir = output_dir / "images" / camera / episode_name
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{sample_id}_{camera}.png"
            cv2.imwrite(str(image_path), image)
            image_paths[camera] = str(image_path.relative_to(output_dir))

        ports: dict[str, Any] = {}
        for port_name in all_port_names:
            key = f"{module_name}/{port_name}"
            frame = self.port_frame(module_name, port_name, port_frame_mode)
            try:
                candidate_port_tf = self.lookup_transform("base_link", frame)
            except TransformException:
                ports[key] = {"available": False, "is_target": port_name == target_port_name, "frame": ""}
                continue
            ports[key] = {
                "available": True,
                "is_target": port_name == target_port_name,
                "frame": frame,
                "plug_tip_in_port": self.plug_tip_to_port_label(candidate_port_tf, plug_tf),
                "port_in_plug_tip": relative_transform_label(plug_tf, candidate_port_tf),
            }

        label = self.plug_tip_to_port_label(port_tf, plug_tf)
        record = {
            "sample_id": sample_id,
            "episode_name": episode_name,
            "task_id": f"distance_sfp_{module_name}_{target_port_name}",
            "task_type": "nic",
            "port_type": "sfp",
            "port_name": target_port_name,
            "target_module_name": module_name,
            "phase": "distance_ci99",
            "step_index": int(step_index),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "images": image_paths,
            "label": {
                "frame": target_port_frame,
                "frame_mode": port_frame_mode,
                "coordinate": "plug_tip position expressed in the target SFP port frame",
                "source": "tf_base_link",
                "plug_tip_to_port": label,
                "ports": ports,
            },
            "frames": {
                "world": "base_link",
                "motion_command_frame": "base_link",
                "motion_command_target": tcp_frame,
                "motion_command_target_pose_source": tcp_pose_source,
                "requested_offset_frame": target_port_frame,
                "label_frame": target_port_frame,
                "plug_tip_frame": cable_tip_frame,
                "tcp_frame": tcp_frame,
            },
            "command": {
                "position": {
                    "x": float(command_pose.position.x),
                    "y": float(command_pose.position.y),
                    "z": float(command_pose.position.z),
                },
                "orientation": {
                    "x": float(command_pose.orientation.x),
                    "y": float(command_pose.orientation.y),
                    "z": float(command_pose.orientation.z),
                    "w": float(command_pose.orientation.w),
                },
            },
            "generator": {
                "name": "sfp_distance_ci99",
                "requested_port_local_offset": requested_offset.to_dict(),
                "requested_offset_frame": target_port_frame,
                "target_port_frame": target_port_frame,
                "port_frame_mode": port_frame_mode,
                "cable_tip_frame": cable_tip_frame,
                "tcp_frame": tcp_frame,
                "tcp_pose_source": tcp_pose_source,
                "placement_check": placement_check,
                "measured_tcp_in_base": {
                    "x": float(tcp_tf.translation.x),
                    "y": float(tcp_tf.translation.y),
                    "z": float(tcp_tf.translation.z),
                },
                "tf_tcp_in_base": tcp_tf_diagnostic,
            },
        }

        samples_path = output_dir / "samples.jsonl"
        with samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-module-name", type=str, default=None)
    parser.add_argument("--max-mounts", type=int, default=5)
    parser.add_argument("--port-names", type=parse_csv, default=SFP_PORT_NAMES)
    parser.add_argument("--all-port-names", type=parse_csv, default=SFP_PORT_NAMES)
    parser.add_argument(
        "--port-frame-mode",
        choices=PORT_FRAME_MODE_CHOICES,
        default=DEFAULT_PORT_FRAME_MODE,
        help=(
            "Frame used for requested offsets and labels. 'entrance' uses "
            "task_board/<module>/<port>_link_entrance; 'link' uses "
            "task_board/<module>/<port>_link; 'auto' prefers entrance when present."
        ),
    )
    parser.add_argument("--cable-tip-frame", type=str, default=None)
    parser.add_argument("--cable-tip-candidates", type=parse_csv, default=DEFAULT_CABLE_TIP_CANDIDATES)
    parser.add_argument("--tcp-frame", type=str, default=DEFAULT_TCP_FRAME)
    parser.add_argument(
        "--tcp-pose-source",
        choices=TCP_POSE_SOURCE_CHOICES,
        default=DEFAULT_TCP_POSE_SOURCE,
        help=(
            "Pose source used for the commanded TCP. 'controller_state' matches "
            "the MotionUpdate target frame; 'tf' uses live TF base_link->tcp_frame."
        ),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-mode", choices=SAMPLING_MODE_CHOICES, default="step-grid")
    parser.add_argument("--samples-per-port", type=int, default=DEFAULT_SAMPLES_PER_PORT)
    parser.add_argument("--grid-step-mm", type=float, default=DEFAULT_GRID_STEP_MM)
    parser.add_argument("--grid-per-axis", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-ci99-mm", type=parse_range_mm, default=OffsetRangeMM(*DEFAULT_CI99_X_MM))
    parser.add_argument("--y-ci99-mm", type=parse_range_mm, default=OffsetRangeMM(*DEFAULT_CI99_Y_MM))
    parser.add_argument("--z-ci99-error-mm", type=parse_range_mm, default=OffsetRangeMM(*DEFAULT_CI99_Z_ERROR_MM))
    parser.add_argument("--approach-offset-mm", type=float, default=DEFAULT_APPROACH_OFFSET_MM)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument("--tf-warmup-s", type=float, default=3.0)
    parser.add_argument("--move-settle-s", type=float, default=2.5)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--command-repeats", type=int, default=5)
    parser.add_argument("--placement-check-retries", type=int, default=4)
    parser.add_argument("--max-placement-axis-error-mm", type=float, default=DEFAULT_MAX_PLACEMENT_AXIS_ERROR_MM)
    parser.add_argument(
        "--max-placement-euclidean-error-mm",
        type=float,
        default=DEFAULT_MAX_PLACEMENT_EUCLIDEAN_ERROR_MM,
    )
    parser.add_argument(
        "--max-tcp-position-error-mm",
        type=float,
        default=DEFAULT_MAX_TCP_POSITION_ERROR_MM,
        help="Optional diagnostic TCP command error limit. Negative disables this as a save gate.",
    )
    parser.add_argument(
        "--allow-existing-samples",
        action="store_true",
        help="Allow appending to an existing samples.jsonl. Default is to fail to avoid mixing runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --run-id by skipping sample_ids already present in samples.jsonl.",
    )
    parser.add_argument("--max-correction-step-mm", type=float, default=DEFAULT_MAX_CORRECTION_STEP_MM)
    parser.add_argument("--stiffness", type=parse_csv, default=("200", "200", "200", "50", "50", "50"))
    parser.add_argument("--damping", type=parse_csv, default=("80", "80", "80", "20", "20", "20"))
    return parser.parse_args()


def wait_for_camera_data(node: SfpDistanceCollector, wait_s: float) -> bool:
    node.get_logger().info(f"Waiting for camera data up to {wait_s:.1f}s...")
    start = time.time()
    while not node.ready() and time.time() - start < wait_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    return node.ready()


def warmup_tf(node: SfpDistanceCollector, duration_s: float) -> None:
    node.get_logger().info(f"Warming TF buffer for {duration_s:.1f}s...")
    start = time.time()
    while time.time() - start < duration_s:
        rclpy.spin_once(node, timeout_sec=0.1)


def settle(node: SfpDistanceCollector, duration_s: float) -> None:
    start = time.time()
    while time.time() - start < duration_s:
        rclpy.spin_once(node, timeout_sec=0.1)


def load_completed_sample_ids(samples_path: Path, run_id: str) -> set[str]:
    if not samples_path.exists():
        return set()

    completed: set[str] = set()
    with samples_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {samples_path}:{line_number}: {exc}") from exc

            sample_id = record.get("sample_id")
            if isinstance(sample_id, str) and sample_id.startswith(f"{run_id}_"):
                completed.add(sample_id)
    return completed


def label_xyz_mm(label: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            label["x_mm"],
            label["y_mm"],
            label["z_mm"],
        ],
        dtype=np.float64,
    )


def requested_offset_xyz_mm(offset: LocalOffset) -> np.ndarray:
    return offset.vector_m() * 1000.0


def measure_placement(
    node: SfpDistanceCollector,
    target_port_frame: str,
    cable_tip_frame: str,
    tcp_frame: str,
    tcp_pose_source: str,
    command_pose: Pose,
    requested_offset: LocalOffset,
) -> dict[str, Any]:
    port_tf = node.lookup_transform("base_link", target_port_frame)
    plug_tf = node.lookup_transform("base_link", cable_tip_frame)
    tcp_tf = node.tcp_transform(tcp_frame, tcp_pose_source)
    label = node.plug_tip_to_port_label(port_tf, plug_tf)

    requested_mm = requested_offset_xyz_mm(requested_offset)
    measured_mm = label_xyz_mm(label)
    placement_error_mm = measured_mm - requested_mm
    tcp_position_error_mm = (transform_xyz(tcp_tf) - pose_xyz(command_pose)) * 1000.0

    return {
        "requested_offset_mm": requested_mm,
        "measured_offset_mm": measured_mm,
        "placement_error_mm": placement_error_mm,
        "placement_axis_error_abs_max_mm": float(np.max(np.abs(placement_error_mm))),
        "placement_error_norm_mm": float(np.linalg.norm(placement_error_mm)),
        "tcp_position_error_mm": tcp_position_error_mm,
        "tcp_position_error_abs_max_mm": float(np.max(np.abs(tcp_position_error_mm))),
        "tcp_position_error_norm_mm": float(np.linalg.norm(tcp_position_error_mm)),
    }


def placement_check_ok(check: dict[str, Any], args: argparse.Namespace) -> bool:
    placement_ok = (
        check["placement_axis_error_abs_max_mm"] <= args.max_placement_axis_error_mm
        and check["placement_error_norm_mm"] <= args.max_placement_euclidean_error_mm
    )
    tcp_limit = float(args.max_tcp_position_error_mm)
    if tcp_limit < 0.0:
        return placement_ok
    return placement_ok and check["tcp_position_error_norm_mm"] <= tcp_limit


def placement_check_summary(check: dict[str, Any]) -> str:
    requested = check["requested_offset_mm"]
    measured = check["measured_offset_mm"]
    placement_error = check["placement_error_mm"]
    tcp_error = check["tcp_position_error_mm"]
    return (
        "requested_mm="
        f"({requested[0]:+.3f},{requested[1]:+.3f},{requested[2]:+.3f}), "
        "measured_mm="
        f"({measured[0]:+.3f},{measured[1]:+.3f},{measured[2]:+.3f}), "
        "label_error_mm="
        f"({placement_error[0]:+.3f},{placement_error[1]:+.3f},{placement_error[2]:+.3f}), "
        f"label_norm={check['placement_error_norm_mm']:.3f}mm, "
        "tcp_error_mm="
        f"({tcp_error[0]:+.3f},{tcp_error[1]:+.3f},{tcp_error[2]:+.3f}), "
        f"tcp_norm={check['tcp_position_error_norm_mm']:.3f}mm"
    )


def correction_from_placement_error(
    port_tf: Transform,
    placement_check: dict[str, Any],
    max_step_mm: float,
) -> np.ndarray:
    correction_port_mm = -np.asarray(
        placement_check["placement_error_mm"],
        dtype=np.float64,
    )
    norm_mm = float(np.linalg.norm(correction_port_mm))
    if max_step_mm > 0.0 and norm_mm > max_step_mm:
        correction_port_mm *= max_step_mm / norm_mm

    port_rotation = quat_to_matrix_xyzw(
        port_tf.rotation.x,
        port_tf.rotation.y,
        port_tf.rotation.z,
        port_tf.rotation.w,
    )
    return port_rotation @ (correction_port_mm / 1000.0)


def apply_position_correction(pose: Pose, correction_base_m: np.ndarray) -> None:
    pose.position.x = float(pose.position.x + correction_base_m[0])
    pose.position.y = float(pose.position.y + correction_base_m[1])
    pose.position.z = float(pose.position.z + correction_base_m[2])


def build_offsets(args: argparse.Namespace) -> list[LocalOffset]:
    if args.samples_per_port < 0:
        raise ValueError("--samples-per-port must be >= 0")
    if args.random_samples < 0:
        raise ValueError("--random-samples must be >= 0")
    if args.grid_step_mm <= 0.0:
        raise ValueError("--grid-step-mm must be > 0")

    offsets: list[LocalOffset] = []
    if args.sampling_mode == "step-grid":
        offsets.extend(
            make_offset_step_grid(
                x_range=args.x_ci99_mm,
                y_range=args.y_ci99_mm,
                z_error_range=args.z_ci99_error_mm,
                approach_offset_mm=args.approach_offset_mm,
                step_mm=args.grid_step_mm,
            )
        )
    elif args.sampling_mode in {"grid", "grid+uniform"}:
        offsets.extend(
            make_offset_grid(
                x_range=args.x_ci99_mm,
                y_range=args.y_ci99_mm,
                z_error_range=args.z_ci99_error_mm,
                approach_offset_mm=args.approach_offset_mm,
                points_per_axis=args.grid_per_axis,
            )
        )

    if args.sampling_mode == "uniform":
        offsets.extend(
            make_uniform_offsets(
                x_range=args.x_ci99_mm,
                y_range=args.y_ci99_mm,
                z_error_range=args.z_ci99_error_mm,
                approach_offset_mm=args.approach_offset_mm,
                count=args.samples_per_port,
                seed=args.seed,
            )
        )
    elif args.sampling_mode == "grid+uniform" and args.random_samples > 0:
        offsets.extend(
            make_random_offsets(
                x_range=args.x_ci99_mm,
                y_range=args.y_ci99_mm,
                z_error_range=args.z_ci99_error_mm,
                approach_offset_mm=args.approach_offset_mm,
                count=args.random_samples,
                seed=args.seed,
            )
        )
    if not offsets:
        raise ValueError("No offsets generated. Increase --samples-per-port or choose grid mode.")
    return offsets


def write_generation_config(
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    offsets: list[LocalOffset],
) -> None:
    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "output_dir": str(output_dir.resolve()),
        "confidence_coverage": 0.99,
        "source_notebook": "ws_aic/src/ais/ais_yolo_train/notebook/visualize_triangulation_error_SFP.ipynb",
        "ci99_source_coordinate": (
            "YOLO triangulation CI99 was originally measured in the camera "
            "rig/tool0 frame, then converted to the plug-tip frame for "
            "sampling."
        ),
        "camera_rig_ci99_mm": {
            "x": {"low": -1.4305, "high": 1.0263},
            "y": {"low": -2.1049, "high": 0.9690},
            "z": {"low": -4.4473, "high": 5.5253},
        },
        "tip_frame_ci99_mm": {
            "x": args.x_ci99_mm.to_dict(),
            "y": args.y_ci99_mm.to_dict(),
            "raw_z": {"low": -4.5043, "high": 5.9138},
            "outward_z_error": args.z_ci99_error_mm.to_dict(),
        },
        "x_ci99_mm": args.x_ci99_mm.to_dict(),
        "y_ci99_mm": args.y_ci99_mm.to_dict(),
        "z_ci99_error_mm": args.z_ci99_error_mm.to_dict(),
        "approach_offset_mm": float(args.approach_offset_mm),
        "sampling_mode": args.sampling_mode,
        "samples_per_port": int(args.samples_per_port),
        "grid_step_mm": float(args.grid_step_mm),
        "grid_per_axis": int(args.grid_per_axis),
        "random_samples": int(args.random_samples),
        "seed": int(args.seed),
        "offset_count_per_port": len(offsets),
        "port_names": list(args.port_names),
        "all_port_names": list(args.all_port_names),
        "port_frame_mode": args.port_frame_mode,
        "tcp_pose_source": args.tcp_pose_source,
        "placement_validation": {
            "max_axis_error_mm": float(args.max_placement_axis_error_mm),
            "max_euclidean_error_mm": float(args.max_placement_euclidean_error_mm),
            "max_tcp_position_error_mm": float(args.max_tcp_position_error_mm),
            "max_correction_step_mm": float(args.max_correction_step_mm),
            "retries": int(args.placement_check_retries),
        },
        "frames": {
            "world": "base_link",
            "motion_command_frame": "base_link",
            "motion_command_target": args.tcp_frame,
            "motion_command_target_pose_source": args.tcp_pose_source,
            "requested_offset": (
                "task_board/<target_module_name>/<port_name>_link_entrance"
                if args.port_frame_mode == "entrance"
                else "task_board/<target_module_name>/<port_name>_link"
                if args.port_frame_mode == "link"
                else "auto: entrance if available, else link"
            ),
            "label": "same as requested_offset",
            "plug_tip": "resolved cable tip TF, usually cable_0/sfp_tip_link",
        },
        "coordinate_convention": (
            "requested_port_local_offset and label.plug_tip_to_port are both "
            "the plug tip position expressed in the selected SFP port frame. "
            "Sampling ranges are YOLO triangulation errors converted from the "
            "camera rig/tool0 frame into the plug-tip frame. Local -Z is the "
            "outward approach direction, so a 20 mm pre-align pose has "
            "z_m=-0.020 in that same frame. z_ci99_error_mm stores outward "
            "distance error, equal to -raw_tip_z_error_mm."
        ),
    }
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    config_path = meta_dir / f"generation_config_{run_id}.json"
    config_path.write_text(
        json.dumps(json_safe(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_for_port(
    node: SfpDistanceCollector,
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    module_name: str,
    port_name: str,
    all_port_names: tuple[str, ...],
    cable_tip_frame: str,
    offsets: list[LocalOffset],
    stiffness: tuple[float, ...],
    damping: tuple[float, ...],
    completed_sample_ids: set[str],
) -> int:
    target_port_frame = node.port_frame(module_name, port_name, args.port_frame_mode)
    if not node.wait_for_tf("base_link", target_port_frame, timeout_s=args.wait_s):
        node.log_tf_hints(target_port_frame)
        raise RuntimeError(f"Target port TF not available: {target_port_frame}")
    if not node.wait_for_tf("base_link", cable_tip_frame, timeout_s=args.wait_s):
        raise RuntimeError(f"Cable tip TF not available: {cable_tip_frame}")
    if args.tcp_pose_source == "tf" and not node.wait_for_tf("base_link", args.tcp_frame, timeout_s=args.wait_s):
        raise RuntimeError(f"TCP TF not available: {args.tcp_frame}")
    if args.tcp_pose_source == "controller_state" and not node.wait_for_controller_state(args.wait_s):
        raise RuntimeError("ControllerState not available: /aic_controller/controller_state")

    safe_module = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_name)
    safe_port = re.sub(r"[^A-Za-z0-9_.-]+", "_", port_name)
    episode_name = f"{run_id}_{safe_module}_{safe_port}_distance_ci99"
    saved = 0
    command_bias_base_m = np.zeros(3, dtype=np.float64)

    for step_index, offset in enumerate(offsets):
        sample_id = f"{episode_name}_{step_index:06d}"
        if sample_id in completed_sample_ids:
            node.get_logger().info(
                f"{port_name} sample {step_index + 1}/{len(offsets)} already saved; skipping"
            )
            continue

        port_tf = node.lookup_transform("base_link", target_port_frame)
        plug_tf = node.lookup_transform("base_link", cable_tip_frame)
        tcp_tf = node.tcp_transform(args.tcp_frame, args.tcp_pose_source)
        pose = node.make_target_pose(port_tf, plug_tf, tcp_tf, offset)
        apply_position_correction(pose, command_bias_base_m)

        node.get_logger().info(
            f"{port_name} sample {step_index + 1}/{len(offsets)} "
            f"offset_mm=({offset.x_error_mm:+.3f}, {offset.y_error_mm:+.3f}, "
            f"{offset.z_error_mm:+.3f}), "
            f"bias_base_mm=({command_bias_base_m[0] * 1000.0:+.1f}, "
            f"{command_bias_base_m[1] * 1000.0:+.1f}, "
            f"{command_bias_base_m[2] * 1000.0:+.1f})"
        )
        for _ in range(max(1, args.command_repeats)):
            node.move_robot_to(pose, stiffness=stiffness, damping=damping)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.1)
        settle(node, args.move_settle_s)

        placement_check: dict[str, Any] | None = None
        for attempt in range(args.placement_check_retries + 1):
            placement_check = measure_placement(
                node=node,
                target_port_frame=target_port_frame,
                cable_tip_frame=cable_tip_frame,
                tcp_frame=args.tcp_frame,
                tcp_pose_source=args.tcp_pose_source,
                command_pose=pose,
                requested_offset=offset,
            )
            if placement_check_ok(placement_check, args):
                node.get_logger().info(
                    "Placement check OK: " + placement_check_summary(placement_check)
                )
                break

            if attempt >= args.placement_check_retries:
                raise RuntimeError(
                    f"Placement validation failed for {sample_id}: "
                    + placement_check_summary(placement_check)
                )

            port_tf = node.lookup_transform("base_link", target_port_frame)
            correction_base_m = correction_from_placement_error(
                port_tf,
                placement_check,
                args.max_correction_step_mm,
            )
            apply_position_correction(pose, correction_base_m)
            command_bias_base_m += correction_base_m
            node.get_logger().warning(
                "Placement check failed; applying corrective command "
                f"base_m=({correction_base_m[0]:+.4f}, "
                f"{correction_base_m[1]:+.4f}, {correction_base_m[2]:+.4f}): "
                + placement_check_summary(placement_check)
            )
            for _ in range(max(1, args.command_repeats)):
                node.move_robot_to(pose, stiffness=stiffness, damping=damping)
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.1)
            settle(node, args.move_settle_s)

        if node.save_sample(
            output_dir=output_dir,
            episode_name=episode_name,
            sample_id=sample_id,
            step_index=step_index,
            module_name=module_name,
            target_port_name=port_name,
            all_port_names=all_port_names,
            target_port_frame=target_port_frame,
            cable_tip_frame=cable_tip_frame,
            tcp_frame=args.tcp_frame,
            port_frame_mode=args.port_frame_mode,
            tcp_pose_source=args.tcp_pose_source,
            command_pose=pose,
            requested_offset=offset,
            placement_check=placement_check or {},
        ):
            saved += 1
        time.sleep(args.interval_s)
    return saved


def main() -> None:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    if args.resume and args.run_id is None:
        raise ValueError("--resume requires --run-id so the script knows which run to continue")
    if samples_path.exists() and not args.allow_existing_samples and not args.resume:
        raise RuntimeError(
            f"{samples_path} already exists. Move it aside or pass "
            "--allow-existing-samples if you intentionally want to append."
        )
    if args.placement_check_retries < 0:
        raise ValueError("--placement-check-retries must be >= 0")
    if args.max_placement_axis_error_mm < 0.0:
        raise ValueError("--max-placement-axis-error-mm must be >= 0")
    if args.max_placement_euclidean_error_mm < 0.0:
        raise ValueError("--max-placement-euclidean-error-mm must be >= 0")
    if args.max_correction_step_mm < 0.0:
        raise ValueError("--max-correction-step-mm must be >= 0")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    offsets = build_offsets(args)
    write_generation_config(output_dir, run_id, args, offsets)
    completed_sample_ids = load_completed_sample_ids(samples_path, run_id) if args.resume else set()

    stiffness = tuple(float(value) for value in args.stiffness)
    damping = tuple(float(value) for value in args.damping)
    if len(stiffness) != 6 or len(damping) != 6:
        raise ValueError("--stiffness and --damping must each have 6 comma-separated values")

    rclpy.init()
    node = SfpDistanceCollector()
    try:
        if not wait_for_camera_data(node, args.wait_s):
            node.get_logger().error("Camera data not ready.")
            sys.exit(1)
        warmup_tf(node, args.tf_warmup_s)

        module_name = args.target_module_name
        if module_name is None:
            module_name = node.discover_module_name(
                args.max_mounts,
                args.port_names,
                args.port_frame_mode,
            )
        if module_name is None:
            node.log_tf_hints("task_board/nic_card_mount_*/sfp_port_*_link")
            node.get_logger().error("No SFP module TF found.")
            sys.exit(1)

        cable_tip_frame = args.cable_tip_frame
        if cable_tip_frame is None:
            cable_tip_frame = node.discover_cable_tip_frame(
                args.cable_tip_candidates,
                timeout_s=0.5,
            )
        if cable_tip_frame is None:
            node.get_logger().error(
                "No cable tip TF found. Pass --cable-tip-frame, for example "
                "cable_0/sfp_tip_link."
            )
            sys.exit(1)

        node.get_logger().info(
            f"Collecting distance dataset: output={output_dir}, "
            f"module={module_name}, ports={args.port_names}, "
            f"offsets_per_port={len(offsets)}, port_frame_mode={args.port_frame_mode}, "
            f"cable_tip={cable_tip_frame}, resume_skips={len(completed_sample_ids)}"
        )
        total_saved = 0
        for port_name in args.port_names:
            total_saved += collect_for_port(
                node=node,
                output_dir=output_dir,
                run_id=run_id,
                args=args,
                module_name=module_name,
                port_name=port_name,
                all_port_names=args.all_port_names,
                cable_tip_frame=cable_tip_frame,
                offsets=offsets,
                stiffness=stiffness,
                damping=damping,
                completed_sample_ids=completed_sample_ids,
            )
        node.get_logger().info(
            f"Done. saved_samples={total_saved}, samples_jsonl={output_dir / 'samples.jsonl'}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
