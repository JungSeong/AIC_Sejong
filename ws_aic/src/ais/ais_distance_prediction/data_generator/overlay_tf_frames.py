#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


DEFAULT_FRAMES = (
    "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
    "cable_0/sfp_tip_link",
)
DEFAULT_LABELS = ("port_entrance", "sfp_tip")
DEFAULT_OFFSETS = ("0:0:0", "0:0:0")
COLORS_BGR = (
    (0, 255, 255),
    (0, 80, 255),
    (80, 255, 80),
    (255, 160, 0),
    (255, 80, 255),
)


def quat_to_matrix_xyzw(quat: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


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


class TfFrameOverlay(Node):
    def __init__(
        self,
        camera: str,
        frames: list[str],
        labels: list[str],
        offsets: list[np.ndarray],
    ) -> None:
        super().__init__("tf_frame_overlay")
        self.camera = camera
        self.frames = frames
        self.labels = labels
        self.offsets = offsets
        self.latest_image: np.ndarray | None = None
        self.camera_info: CameraInfo | None = None

        self.create_subscription(
            CameraInfo,
            f"/{camera}_camera/camera_info",
            self._on_camera_info,
            10,
        )
        self.create_subscription(
            Image,
            f"/{camera}_camera/image",
            self._on_image,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info(
            f"overlay ready camera={camera} frames={', '.join(frames)}"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_image(self, msg: Image) -> None:
        image = image_msg_to_bgr(msg)
        if image is not None:
            self.latest_image = image

    def ready(self) -> bool:
        return self.latest_image is not None and self.camera_info is not None

    def camera_frame(self) -> str:
        assert self.camera_info is not None
        frame_id = self.camera_info.header.frame_id.strip()
        if frame_id:
            return frame_id
        return f"{self.camera}_camera/camera_optical_frame"

    def project_point(self, frame: str, offset_xyz: np.ndarray) -> tuple[float, float, float] | None:
        assert self.camera_info is not None
        camera_frame = self.camera_frame()
        tf = self.tf_buffer.lookup_transform(camera_frame, frame, Time()).transform
        translation = np.array(
            [tf.translation.x, tf.translation.y, tf.translation.z],
            dtype=np.float64,
        )
        rotation = quat_to_matrix_xyzw(
            (
                float(tf.rotation.x),
                float(tf.rotation.y),
                float(tf.rotation.z),
                float(tf.rotation.w),
            )
        )
        point = translation + rotation @ offset_xyz
        x, y, z = point.tolist()
        if z <= 1e-6:
            return None

        k = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        u = k[0, 0] * x / z + k[0, 2]
        v = k[1, 1] * y / z + k[1, 2]
        return float(u), float(v), z

    def matching_frame_lines(self, filters: list[str], limit: int) -> list[str]:
        tf_text = self.tf_buffer.all_frames_as_string()
        matches: list[str] = []
        for line in tf_text.splitlines():
            line_lower = line.lower()
            if any(token.lower() in line_lower for token in filters):
                matches.append(line)
            if len(matches) >= limit:
                break
        return matches

    def make_overlay(self) -> tuple[np.ndarray, list[str]]:
        assert self.latest_image is not None
        image = self.latest_image.copy()
        height, width = image.shape[:2]
        lines: list[str] = []

        for idx, (frame, label, offset) in enumerate(zip(self.frames, self.labels, self.offsets)):
            color = COLORS_BGR[idx % len(COLORS_BGR)]
            try:
                projected = self.project_point(frame, offset)
            except TransformException as exc:
                lines.append(f"{label}: TF unavailable frame={frame} ({exc})")
                continue

            if projected is None:
                lines.append(f"{label}: behind camera frame={frame}")
                continue

            u, v, depth = projected
            inside = 0 <= u < width and 0 <= v < height
            lines.append(
                f"{label}: frame={frame} offset=({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f}) "
                f"uv=({u:.1f}, {v:.1f}) "
                f"depth={depth:.4f}m inside={inside}"
            )
            if not inside:
                continue

            center = (int(round(u)), int(round(v)))
            cv2.drawMarker(
                image,
                center,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
            cv2.circle(image, center, 8, color, 2)
            cv2.putText(
                image,
                label,
                (center[0] + 10, max(20, center[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return image, lines


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_offset_csv(value: str) -> list[np.ndarray]:
    offsets: list[np.ndarray] = []
    for part in parse_csv(value):
        values = [float(item.strip()) for item in part.replace(";", ":").split(":")]
        if len(values) != 3:
            raise ValueError(f"offset must have 3 values separated by ':', got {part!r}")
        offsets.append(np.asarray(values, dtype=np.float64))
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project TF frame origins onto a camera image and save an overlay."
    )
    parser.add_argument("--camera", default="center", help="Camera prefix, e.g. left, center, right.")
    parser.add_argument(
        "--frames",
        default=",".join(DEFAULT_FRAMES),
        help="Comma-separated TF frames whose origins will be projected.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels for --frames.",
    )
    parser.add_argument(
        "--offsets",
        default=",".join(DEFAULT_OFFSETS),
        help="Comma-separated local XYZ offsets in meters for each frame, e.g. 0:0:-0.0458,0:0:0.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/tf_frame_overlay.png",
        help="Output image path.",
    )
    parser.add_argument("--wait-s", type=float, default=15.0, help="Max seconds to wait for image/info.")
    parser.add_argument("--retries", type=int, default=10, help="TF projection attempts after image is ready.")
    parser.add_argument("--retry-s", type=float, default=0.2, help="Seconds between TF projection attempts.")
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="Print TF frame lines matching --frame-filter.",
    )
    parser.add_argument(
        "--frame-filter",
        default="sfp,tip,cable,nic_card,task_board",
        help="Comma-separated substrings used by --print-frames.",
    )
    parser.add_argument("--frame-limit", type=int, default=80, help="Max matching TF frame lines to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = parse_csv(args.frames)
    labels = parse_csv(args.labels)
    offsets = parse_offset_csv(args.offsets)
    if not frames:
        raise SystemExit("--frames must not be empty")
    if len(labels) < len(frames):
        labels.extend(frames[len(labels) :])
    labels = labels[: len(frames)]
    if len(offsets) < len(frames):
        offsets.extend(np.zeros(3, dtype=np.float64) for _ in range(len(frames) - len(offsets)))
    offsets = offsets[: len(frames)]

    rclpy.init()
    node = TfFrameOverlay(args.camera, frames, labels, offsets)
    try:
        start = time.time()
        while rclpy.ok() and not node.ready() and time.time() - start < args.wait_s:
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.ready():
            node.get_logger().error(
                f"camera image/info not ready for /{args.camera}_camera within {args.wait_s}s"
            )
            return 1

        overlay = None
        lines: list[str] = []
        for _ in range(max(1, args.retries)):
            rclpy.spin_once(node, timeout_sec=0.1)
            overlay, lines = node.make_overlay()
            if any("uv=" in line for line in lines):
                break
            time.sleep(args.retry_s)

        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert overlay is not None
        if not cv2.imwrite(str(output_path), overlay):
            node.get_logger().error(f"failed to write overlay: {output_path}")
            return 1

        print(f"camera_frame: {node.camera_frame()}")
        for line in lines:
            print(line)
        if args.print_frames or any("TF unavailable" in line for line in lines):
            filters = parse_csv(args.frame_filter)
            print("matching TF frames:")
            matches = node.matching_frame_lines(filters, args.frame_limit)
            if matches:
                for line in matches:
                    print(line)
            else:
                print(f"(no frames matched filters: {', '.join(filters)})")
        print(f"saved: {output_path}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
