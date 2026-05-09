#!/usr/bin/env python3
"""
Collect YOLO labels for approach models.

SFP output label format:
  class x_center y_center width height
        port0_top_left_x port0_top_left_y ... port0_bottom_left_x port0_bottom_left_y
        port1_top_left_x port1_top_left_y ... port1_bottom_left_x port1_bottom_left_y

SC output label format:
  class x_center y_center width height
        sc_top_left_x sc_top_left_y ... sc_bottom_left_x sc_bottom_left_y

All coordinates are normalized to 0..1.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import rclpy

try:
    from .core.collector_node import NicPortPoseCollector
    from .core.dataset_format import (
        TARGET_CHOICES,
        default_output_for_target,
        normalize_target,
        split_for_episode,
        write_data_yaml,
    )
    from .core.geometry import make_viewpoints
except ImportError:
    from core.collector_node import NicPortPoseCollector
    from core.dataset_format import (
        TARGET_CHOICES,
        default_output_for_target,
        normalize_target,
        split_for_episode,
        write_data_yaml,
    )
    from core.geometry import make_viewpoints


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGET_CHOICES, default="SFP")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-viewpoints", type=int, default=15)
    parser.add_argument("--frames-per-viewpoint", type=int, default=None)
    parser.add_argument("--move-settle-s", type=float, default=2.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument("--tf-warmup-s", type=float, default=3.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--stem-prefix", type=str, default="")
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--debug-every", type=int, default=10)
    return parser.parse_args()


def wait_for_camera_data(node: NicPortPoseCollector, wait_s: float) -> bool:
    node.get_logger().info(f"Waiting for camera data up to {wait_s:.1f}s...")
    start = time.time()
    while not node.ready() and time.time() - start < wait_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    return node.ready()


def warmup_tf_buffer(node: NicPortPoseCollector, warmup_s: float) -> None:
    node.get_logger().info(f"Warming TF buffer for {warmup_s:.1f}s...")
    start = time.time()
    while time.time() - start < warmup_s:
        rclpy.spin_once(node, timeout_sec=0.1)


def move_to_viewpoint(node: NicPortPoseCollector, viewpoint, vp_idx: int, total: int, settle_s: float) -> None:
    x, y, z, roll, pitch, yaw = viewpoint
    node.get_logger().info(
        f"Viewpoint {vp_idx + 1}/{total}: "
        f"pos=({x:+.3f}, {y:+.3f}, {z:+.3f}) "
        f"rpy=({roll:+.2f}, {pitch:+.2f}, {yaw:+.2f})"
    )
    for _ in range(5):
        node.move_robot_to(x, y, z, roll, pitch, yaw)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)

    start = time.time()
    while time.time() - start < settle_s:
        rclpy.spin_once(node, timeout_sec=0.1)


def collect_frames(node: NicPortPoseCollector, args, output_dir: Path, debug_dir: Path | None) -> int:
    viewpoints = make_viewpoints(args.n_viewpoints) if args.n_viewpoints > 0 else [None]
    frames_per_viewpoint = args.frames_per_viewpoint
    if frames_per_viewpoint is None:
        frames_per_viewpoint = max(1, math.ceil(args.episodes / len(viewpoints)))

    saved_frames = 0
    episode_id = 0
    try:
        for vp_idx, viewpoint in enumerate(viewpoints):
            if episode_id >= args.episodes and args.frames_per_viewpoint is None:
                break

            if viewpoint is not None:
                move_to_viewpoint(node, viewpoint, vp_idx, len(viewpoints), args.move_settle_s)

            for _ in range(frames_per_viewpoint):
                if episode_id >= args.episodes and args.frames_per_viewpoint is None:
                    break

                rclpy.spin_once(node, timeout_sec=0.1)
                split = split_for_episode(episode_id, args.val_ratio)
                if node.collect_one_frame(
                    episode_id,
                    output_dir,
                    split,
                    args.stem_prefix,
                    args.min_depth,
                    args.max_depth,
                    args.bbox_margin,
                    debug_dir,
                    args.debug_every,
                ):
                    saved_frames += 1
                episode_id += 1
                time.sleep(args.interval_s)
    except KeyboardInterrupt:
        node.get_logger().info("Collection interrupted.")
    return saved_frames


def log_dataset_summary(node: NicPortPoseCollector, output_dir: Path, saved_frames: int) -> None:
    train_count = len(list((output_dir / "images" / "train").glob("*.jpg")))
    val_count = len(list((output_dir / "images" / "val").glob("*.jpg")))
    node.get_logger().info(
        f"Done. saved_frames={saved_frames}, train_images={train_count}, "
        f"val_images={val_count}, data_yaml={output_dir / 'data.yaml'}"
    )


def main() -> None:
    args = parse_args()
    target = normalize_target(args.target)
    default_output = default_output_for_target(target)
    output = args.output if args.output is not None else str(default_output)
    output_dir = Path(os.path.expanduser(output)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(os.path.expanduser(args.debug_dir)).resolve() if args.debug_dir else None

    rclpy.init()
    node = NicPortPoseCollector(target)

    try:
        if not wait_for_camera_data(node, args.wait_s):
            node.get_logger().error("Camera data not ready.")
            sys.exit(1)

        warmup_tf_buffer(node, args.tf_warmup_s)
        targets = node.discover_targets()
        if not targets:
            node.log_port_tf_diagnostics()
            node.get_logger().error(f"No {target} targets found. Is ground_truth enabled?")
            sys.exit(1)
        node.get_logger().info(f"Found {len(targets)} {target} target(s).")

        saved_frames = collect_frames(node, args, output_dir, debug_dir)
        write_data_yaml(output_dir, target)
        log_dataset_summary(node, output_dir, saved_frames)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
