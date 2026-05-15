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

TASK_BOARD output label format:
  class x_center y_center width height
        board_keypoint0_x board_keypoint0_y ... board_keypoint3_x board_keypoint3_y

MAGENTA output format:
  raw camera images plus metadata.csv. No YOLO label is generated.

All coordinates are normalized to 0..1.
"""

import argparse
import hashlib
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
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
    parser.add_argument("--target", choices=TARGET_CHOICES, default="SC")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-viewpoints", type=int, default=15)
    parser.add_argument("--frames-per-viewpoint", type=int, default=None)
    parser.add_argument("--move-settle-s", type=float, default=2.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument("--tf-warmup-s", type=float, default=3.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--stem-prefix", type=str, default="")
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument(
        "--magenta-split",
        choices=("train", "val"),
        default=None,
        help="Force all MAGENTA images from this run into one split.",
    )
    parser.add_argument(
        "--board-pose-id",
        type=str,
        default=None,
        help="Stable board-pose group id for MAGENTA split/metadata.",
    )
    parser.add_argument("--trial-group", type=str, default="")
    parser.add_argument("--board-x", type=float, default=None)
    parser.add_argument("--board-y", type=float, default=None)
    parser.add_argument("--board-yaw", type=float, default=None)
    parser.add_argument("--magenta-hue-min", type=int, default=125)
    parser.add_argument("--magenta-hue-max", type=int, default=179)
    parser.add_argument("--magenta-sat-min", type=int, default=70)
    parser.add_argument("--magenta-val-min", type=int, default=50)
    parser.add_argument("--magenta-min-area", type=float, default=80.0)
    parser.add_argument(
        "--magenta-viewpoint-mode",
        choices=("current-lift", "absolute"),
        default="current-lift",
        help=(
            "MAGENTA viewpoint source. current-lift preserves the current TCP "
            "orientation and lifts from the current TCP pose like DebugSFP."
        ),
    )
    parser.add_argument(
        "--magenta-lift-m",
        type=float,
        default=float(os.environ.get("AIC_DISTANCE_INITIAL_LIFT_M", "0.050")),
        help="Current-pose z lift for MAGENTA current-lift mode.",
    )
    parser.add_argument("--magenta-xy-jitter-m", type=float, default=0.015)
    parser.add_argument("--magenta-z-jitter-m", type=float, default=0.005)
    parser.add_argument("--magenta-viewpoint-seed", type=int, default=42)
    parser.add_argument(
        "--magenta-acquire-visible",
        dest="magenta_acquire_visible",
        action="store_true",
        default=True,
        help="Before MAGENTA collection, search bounded xy offsets until the blob is sufficiently visible.",
    )
    parser.add_argument(
        "--no-magenta-acquire-visible",
        dest="magenta_acquire_visible",
        action="store_false",
    )
    parser.add_argument("--magenta-acquire-xy-step-m", type=float, default=0.015)
    parser.add_argument("--magenta-acquire-xy-radius-m", type=float, default=0.060)
    parser.add_argument(
        "--magenta-acquire-first-axis-radius-m",
        type=float,
        default=None,
        help=(
            "Maximum distance to scan along --magenta-acquire-first-axis before "
            "falling back to the bounded xy search. Defaults to "
            "--magenta-acquire-xy-radius-m."
        ),
    )
    parser.add_argument(
        "--magenta-acquire-first-axis",
        choices=("+y", "-y", "+x", "-x", "none"),
        default="+y",
        help="Preferred first xy scan direction after the initial z lift.",
    )
    parser.add_argument("--magenta-acquire-min-area", type=float, default=300.0)
    parser.add_argument("--magenta-acquire-edge-margin-px", type=int, default=20)
    return parser.parse_args()


def wait_for_camera_data(node: NicPortPoseCollector, wait_s: float) -> bool:
    node.get_logger().info(f"Waiting for camera data up to {wait_s:.1f}s...")
    start = time.time()
    while not node.ready() and time.time() - start < wait_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    return node.ready()


def wait_for_controller_state(node: NicPortPoseCollector, wait_s: float) -> bool:
    node.get_logger().info(f"Waiting for controller_state up to {wait_s:.1f}s...")
    start = time.time()
    while not node.controller_state_ready() and time.time() - start < wait_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    return node.controller_state_ready()


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


def make_magenta_current_lift_offsets(
    n: int,
    lift_m: float,
    xy_jitter_m: float,
    z_jitter_m: float,
    seed: int,
) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []

    offsets = [(0.0, 0.0, lift_m)]
    xy = float(max(0.0, xy_jitter_m))
    for dx, dy in [
        (xy, 0.0),
        (-xy, 0.0),
        (0.0, xy),
        (0.0, -xy),
        (xy, xy),
        (xy, -xy),
        (-xy, xy),
        (-xy, -xy),
    ]:
        if len(offsets) >= n:
            return offsets[:n]
        offsets.append((dx, dy, lift_m))

    rng = np.random.default_rng(seed)
    while len(offsets) < n:
        dx = float(rng.uniform(-xy, xy)) if xy > 0 else 0.0
        dy = float(rng.uniform(-xy, xy)) if xy > 0 else 0.0
        dz = float(lift_m)
        if z_jitter_m > 0:
            dz += float(rng.uniform(-z_jitter_m, z_jitter_m))
        offsets.append((dx, dy, dz))
    return offsets


def make_magenta_acquisition_offsets(
    lift_m: float,
    xy_step_m: float,
    xy_radius_m: float,
    first_axis: str,
    first_axis_radius_m: float | None,
) -> list[tuple[float, float, float]]:
    step = float(max(0.0, xy_step_m))
    radius = float(max(0.0, xy_radius_m))
    first_axis_radius = (
        radius
        if first_axis_radius_m is None
        else float(max(0.0, first_axis_radius_m))
    )
    lift = float(lift_m)
    offsets = [(0.0, 0.0, lift)]
    seen = {(0.0, 0.0)}
    if step <= 1e-9 or max(radius, first_axis_radius) <= 1e-9:
        return offsets

    axis_sign = 1.0 if first_axis.startswith("+") else -1.0
    axis_name = first_axis[-1:] if first_axis != "none" else ""
    if axis_name in ("x", "y"):
        max_axis_steps = int(math.ceil(first_axis_radius / step))
        for i in range(1, max_axis_steps + 1):
            dx = axis_sign * i * step if axis_name == "x" else 0.0
            dy = axis_sign * i * step if axis_name == "y" else 0.0
            if math.hypot(dx, dy) <= first_axis_radius + 1e-9:
                key = (round(dx, 9), round(dy, 9))
                if key not in seen:
                    offsets.append((float(dx), float(dy), lift))
                    seen.add(key)

    if radius <= 1e-9:
        return offsets

    max_ring = int(math.ceil(radius / step))
    candidates = []
    for ring in range(1, max_ring + 1):
        for ix in range(-ring, ring + 1):
            for iy in range(-ring, ring + 1):
                if max(abs(ix), abs(iy)) != ring:
                    continue
                dx = float(ix * step)
                dy = float(iy * step)
                if math.hypot(dx, dy) <= radius + 1e-9:
                    candidates.append((math.hypot(dx, dy), dx, dy, lift))

    candidates.sort(key=lambda item: item[0])
    for _, dx, dy, dz in candidates:
        key = (round(dx, 9), round(dy, 9))
        if key in seen:
            continue
        offsets.append((dx, dy, dz))
        seen.add(key)
    return offsets


def move_to_current_lift_viewpoint(
    node: NicPortPoseCollector,
    base_pose,
    offset: tuple[float, float, float],
    vp_idx: int,
    total: int,
    settle_s: float,
) -> dict:
    dx, dy, dz = offset
    target_pose = node.offset_pose(base_pose, dx, dy, dz)
    node.get_logger().info(
        f"MAGENTA viewpoint {vp_idx + 1}/{total}: "
        f"offset=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) "
        f"target=({target_pose.position.x:+.3f}, "
        f"{target_pose.position.y:+.3f}, {target_pose.position.z:+.3f}) "
        "orientation=current_tcp"
    )
    for _ in range(5):
        node.move_robot_to_pose(target_pose)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)

    start = time.time()
    while time.time() - start < settle_s:
        rclpy.spin_once(node, timeout_sec=0.1)

    actual_pose = node.copy_current_tcp_pose() or target_pose
    return {
        "pose": actual_pose,
        "x": float(actual_pose.position.x),
        "y": float(actual_pose.position.y),
        "z": float(actual_pose.position.z),
        "qx": float(actual_pose.orientation.x),
        "qy": float(actual_pose.orientation.y),
        "qz": float(actual_pose.orientation.z),
        "qw": float(actual_pose.orientation.w),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
    }


def acquire_magenta_visible_pose(node: NicPortPoseCollector, start_pose, args):
    offsets = make_magenta_acquisition_offsets(
        args.magenta_lift_m,
        args.magenta_acquire_xy_step_m,
        args.magenta_acquire_xy_radius_m,
        args.magenta_acquire_first_axis,
        args.magenta_acquire_first_axis_radius_m,
    )
    best_metadata = None
    best_area = -1.0

    node.get_logger().info(
        "MAGENTA acquisition scan: "
        f"lift={args.magenta_lift_m * 1000.0:.1f}mm, "
        f"first_axis={args.magenta_acquire_first_axis}, "
        f"xy_step={args.magenta_acquire_xy_step_m * 1000.0:.1f}mm, "
        f"first_axis_radius={((args.magenta_acquire_first_axis_radius_m or args.magenta_acquire_xy_radius_m) * 1000.0):.1f}mm, "
        f"xy_radius={args.magenta_acquire_xy_radius_m * 1000.0:.1f}mm, "
        f"quality_area>={args.magenta_acquire_min_area:.1f}px"
    )

    for idx, offset in enumerate(offsets):
        metadata = move_to_current_lift_viewpoint(
            node,
            start_pose,
            offset,
            idx,
            len(offsets),
            args.move_settle_s,
        )
        visibility = node.detect_magenta_visibility(
            args.magenta_hue_min,
            args.magenta_hue_max,
            args.magenta_sat_min,
            args.magenta_val_min,
            args.magenta_min_area,
            args.magenta_acquire_min_area,
            args.magenta_acquire_edge_margin_px,
        )
        best = visibility["best"]
        area = float(best["area_px"]) if best is not None else 0.0
        if area > best_area:
            best_area = area
            best_metadata = metadata

        if best is None:
            node.get_logger().info(
                f"MAGENTA acquisition[{idx + 1}/{len(offsets)}]: no blob"
            )
        else:
            node.get_logger().info(
                f"MAGENTA acquisition[{idx + 1}/{len(offsets)}]: "
                f"cam={best['camera']} area={area:.0f}px "
                f"edge_ok={best['edge_ok']} quality={visibility['quality']}"
            )

        if visibility["quality"]:
            node.get_logger().info("MAGENTA acquisition succeeded.")
            return metadata

    if best_metadata is not None and best_area > 0:
        node.get_logger().warn(
            "MAGENTA acquisition did not reach quality threshold; "
            f"using best seen pose with area={best_area:.0f}px."
        )
        return best_metadata

    node.get_logger().warn(
        "MAGENTA acquisition did not detect any blob; using current lifted pose."
    )
    return move_to_current_lift_viewpoint(
        node,
        start_pose,
        (0.0, 0.0, args.magenta_lift_m),
        0,
        1,
        args.move_settle_s,
    )


def sanitize_stem_component(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return sanitized.strip("_.-") or "unknown"


def resolve_magenta_board_pose_id(args) -> str:
    if args.board_pose_id:
        return sanitize_stem_component(args.board_pose_id)

    if args.board_x is not None and args.board_y is not None and args.board_yaw is not None:
        trial = sanitize_stem_component(args.trial_group or "trial")
        return (
            f"{trial}_x{args.board_x:+.4f}_"
            f"y{args.board_y:+.4f}_yaw{args.board_yaw:+.4f}"
        ).replace("+", "p").replace("-", "m")

    return "unknown_board_pose"


def split_for_board_pose(board_pose_id: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    digest = hashlib.sha1(board_pose_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    return "val" if bucket < val_ratio else "train"


def resolve_magenta_split(args, board_pose_id: str) -> str:
    if args.magenta_split is not None:
        return args.magenta_split
    if board_pose_id == "unknown_board_pose":
        return "train"
    return split_for_board_pose(board_pose_id, args.val_ratio)


def collect_frames(node: NicPortPoseCollector, args, output_dir: Path, debug_dir: Path | None) -> int:
    magenta_current_lift = (
        node.target == "MAGENTA" and args.magenta_viewpoint_mode == "current-lift"
    )
    magenta_base_pose = None
    if magenta_current_lift and args.n_viewpoints > 0:
        start_pose = node.copy_current_tcp_pose()
        if start_pose is None:
            raise RuntimeError("controller_state is required for MAGENTA current-lift mode")
        if args.magenta_acquire_visible:
            acquired_metadata = acquire_magenta_visible_pose(node, start_pose, args)
            magenta_base_pose = acquired_metadata["pose"]
            base_lift_m = 0.0
        else:
            magenta_base_pose = start_pose
            base_lift_m = args.magenta_lift_m
        viewpoints = make_magenta_current_lift_offsets(
            args.n_viewpoints,
            base_lift_m,
            args.magenta_xy_jitter_m,
            args.magenta_z_jitter_m,
            args.magenta_viewpoint_seed,
        )
    else:
        viewpoints = make_viewpoints(args.n_viewpoints) if args.n_viewpoints > 0 else [None]
    frames_per_viewpoint = args.frames_per_viewpoint
    if frames_per_viewpoint is None:
        frames_per_viewpoint = max(1, math.ceil(args.episodes / len(viewpoints)))

    magenta_board_pose_id = None
    magenta_split = None
    if node.target == "MAGENTA":
        magenta_board_pose_id = resolve_magenta_board_pose_id(args)
        magenta_split = resolve_magenta_split(args, magenta_board_pose_id)

    saved_frames = 0
    episode_id = 0
    try:
        for vp_idx, viewpoint in enumerate(viewpoints):
            if episode_id >= args.episodes and args.frames_per_viewpoint is None:
                break

            viewpoint_metadata = viewpoint
            if magenta_current_lift and viewpoint is not None:
                viewpoint_metadata = move_to_current_lift_viewpoint(
                    node,
                    magenta_base_pose,
                    viewpoint,
                    vp_idx,
                    len(viewpoints),
                    args.move_settle_s,
                )
            elif viewpoint is not None:
                move_to_viewpoint(node, viewpoint, vp_idx, len(viewpoints), args.move_settle_s)

            for _ in range(frames_per_viewpoint):
                if episode_id >= args.episodes and args.frames_per_viewpoint is None:
                    break

                rclpy.spin_once(node, timeout_sec=0.1)
                if node.target == "MAGENTA":
                    saved_frames += node.collect_magenta_frame(
                        episode_id,
                        output_dir,
                        magenta_split,
                        args.stem_prefix,
                        magenta_board_pose_id,
                        args.trial_group,
                        args.board_x,
                        args.board_y,
                        args.board_yaw,
                        vp_idx,
                        viewpoint_metadata,
                        debug_dir,
                        args.debug_every,
                        args.magenta_hue_min,
                        args.magenta_hue_max,
                        args.magenta_sat_min,
                        args.magenta_val_min,
                        args.magenta_min_area,
                    )
                else:
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
        f"val_images={val_count}, output_dir={output_dir}"
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

        if target == "MAGENTA":
            if args.magenta_viewpoint_mode == "current-lift" and args.n_viewpoints > 0:
                if not wait_for_controller_state(node, args.wait_s):
                    node.get_logger().error("ControllerState not ready.")
                    sys.exit(1)
            board_pose_id = resolve_magenta_board_pose_id(args)
            split = resolve_magenta_split(args, board_pose_id)
            node.get_logger().info(
                "MAGENTA collection stores raw camera images only; "
                "TF target discovery and YOLO labels are skipped. "
                "Full-frame images are saved only when the magenta feature is detected."
            )
            node.get_logger().info(
                f"MAGENTA board_pose_id={board_pose_id}, split={split}, "
                f"metadata={output_dir / 'metadata.csv'}"
            )
            node.get_logger().info(
                "MAGENTA HSV filter: "
                f"h=[{args.magenta_hue_min}, {args.magenta_hue_max}], "
                f"s>={args.magenta_sat_min}, v>={args.magenta_val_min}, "
                f"area>={args.magenta_min_area:.1f}px"
            )
            if args.magenta_viewpoint_mode == "current-lift":
                node.get_logger().info(
                    "MAGENTA viewpoint mode=current-lift: "
                    "preserving current TCP orientation and using "
                    f"lift={args.magenta_lift_m * 1000.0:.1f}mm, "
                    f"xy_jitter=+/-{args.magenta_xy_jitter_m * 1000.0:.1f}mm, "
                    f"z_jitter=+/-{args.magenta_z_jitter_m * 1000.0:.1f}mm"
                )
                if args.magenta_acquire_visible:
                    node.get_logger().info(
                        "MAGENTA acquisition enabled: "
                        f"first_axis={args.magenta_acquire_first_axis}, "
                        f"xy_step={args.magenta_acquire_xy_step_m * 1000.0:.1f}mm, "
                        f"first_axis_radius={((args.magenta_acquire_first_axis_radius_m or args.magenta_acquire_xy_radius_m) * 1000.0):.1f}mm, "
                        f"xy_radius={args.magenta_acquire_xy_radius_m * 1000.0:.1f}mm, "
                        f"quality_area>={args.magenta_acquire_min_area:.1f}px, "
                        f"edge_margin={args.magenta_acquire_edge_margin_px}px"
                    )
            else:
                node.get_logger().warn(
                    "MAGENTA viewpoint mode=absolute uses the legacy absolute "
                    "viewpoint generator; this can command high z poses."
                )
            if board_pose_id == "unknown_board_pose" and args.magenta_split is None:
                node.get_logger().warn(
                    "No --board-pose-id or --board-x/--board-y/--board-yaw was given; "
                    "saving this MAGENTA run to train to avoid viewpoint-level leakage."
                )
        else:
            warmup_tf_buffer(node, args.tf_warmup_s)
            targets = node.discover_targets()
            if not targets:
                node.log_port_tf_diagnostics()
                node.get_logger().error(f"No {target} targets found. Is ground_truth enabled?")
                sys.exit(1)
            node.get_logger().info(f"Found {len(targets)} {target} target(s).")

        saved_frames = collect_frames(node, args, output_dir, debug_dir)
        if target != "MAGENTA":
            write_data_yaml(output_dir, target)
        log_dataset_summary(node, output_dir, saved_frames)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
