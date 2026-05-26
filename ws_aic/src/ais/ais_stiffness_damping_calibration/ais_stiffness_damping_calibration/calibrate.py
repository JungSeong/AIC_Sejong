#!/usr/bin/env python3
"""Headless Cartesian stiffness/damping sweep for the AIC controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


AXES = ("x", "y", "z")
DEFAULT_STIFFNESS = ("75,75,50", "100,100,50", "150,150,75", "200,200,100")
DEFAULT_DAMPING = ("30,30,20", "60,60,30", "90,90,45")


@dataclass(frozen=True)
class SweepCase:
    stiffness: tuple[float, float, float]
    damping: tuple[float, float, float]
    repeat: int


@dataclass(frozen=True)
class DeltaCase:
    index: int
    delta: np.ndarray


def parse_xyz(text: str) -> tuple[float, float, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return (values[0], values[1], values[2])
    raise argparse.ArgumentTypeError(f"expected k or x,y,z, got '{text}'")


def pose_xyz(pose: Pose) -> np.ndarray:
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)


def pose_quat(pose: Pose) -> Quaternion:
    return Quaternion(
        x=float(pose.orientation.x),
        y=float(pose.orientation.y),
        z=float(pose.orientation.z),
        w=float(pose.orientation.w),
    )


def make_pose(xyz: Sequence[float], quat: Quaternion) -> Pose:
    return Pose(
        position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
        orientation=Quaternion(x=quat.x, y=quat.y, z=quat.z, w=quat.w),
    )


def diag6(xyz: tuple[float, float, float], rotational: float) -> list[float]:
    return np.diag([xyz[0], xyz[1], xyz[2], rotational, rotational, rotational]).flatten().tolist()


def add_axis_values(row: dict[str, object], prefix: str, values: Sequence[object], suffix: str = "") -> None:
    for axis, value in zip(AXES, values):
        if isinstance(value, np.generic):
            value = value.item()
        row[f"{prefix}_{axis}{suffix}"] = value


class CalibrationNode(Node):
    def __init__(self, namespace: str) -> None:
        super().__init__("ais_stiffness_damping_calibration")
        self.namespace = namespace.strip("/")
        self.state: ControllerState | None = None
        self.state_seq = 0
        self.last_sample: tuple[float, np.ndarray, np.ndarray] | None = None

        self.create_subscription(
            ControllerState,
            f"/{self.namespace}/controller_state",
            self._on_state,
            10,
        )
        self.motion_pub = self.create_publisher(
            MotionUpdate,
            f"/{self.namespace}/pose_commands",
            10,
        )
        self.mode_client = self.create_client(
            ChangeTargetMode,
            f"/{self.namespace}/change_target_mode",
        )

    def _on_state(self, msg: ControllerState) -> None:
        self.state = msg
        self.state_seq += 1
        self.last_sample = (
            time.monotonic(),
            pose_xyz(msg.tcp_pose),
            np.array(
                [
                    msg.tcp_velocity.linear.x,
                    msg.tcp_velocity.linear.y,
                    msg.tcp_velocity.linear.z,
                ],
                dtype=float,
            ),
        )

    def wait_ready(self, timeout_s: float, need_mode_service: bool) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if need_mode_service:
                self.mode_client.wait_for_service(timeout_sec=0.05)
            ready = (
                self.state is not None
                and self.motion_pub.get_subscription_count() > 0
                and (not need_mode_service or self.mode_client.service_is_ready())
            )
            if ready:
                return
            rclpy.spin_once(self, timeout_sec=0.05)

        missing = []
        if self.state is None:
            missing.append(f"/{self.namespace}/controller_state")
        if self.motion_pub.get_subscription_count() == 0:
            missing.append(f"/{self.namespace}/pose_commands subscriber")
        if need_mode_service and not self.mode_client.service_is_ready():
            missing.append(f"/{self.namespace}/change_target_mode")
        raise RuntimeError("Timed out waiting for " + ", ".join(missing))

    def switch_to_cartesian(self, timeout_s: float) -> None:
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self.mode_client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError("Failed to switch controller to Cartesian mode.")

    def current_pose(self) -> Pose:
        if self.state is None:
            raise RuntimeError("ControllerState is not available yet.")
        return make_pose(pose_xyz(self.state.tcp_pose), pose_quat(self.state.tcp_pose))

    def publish_pose(
        self,
        pose: Pose,
        *,
        stiffness: tuple[float, float, float],
        damping: tuple[float, float, float],
        rot_stiffness: float,
        rot_damping: float,
        frame_id: str,
        publish_count: int,
        publish_period_s: float,
    ) -> None:
        msg = MotionUpdate()
        msg.header.frame_id = frame_id
        msg.pose = pose
        msg.target_stiffness = diag6(stiffness, rot_stiffness)
        msg.target_damping = diag6(damping, rot_damping)
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        for _ in range(max(1, publish_count)):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.motion_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=publish_period_s)

    def collect(self, duration_s: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
        samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        deadline = time.monotonic() + duration_s
        last_seq = self.state_seq
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.state_seq != last_seq and self.last_sample is not None:
                samples.append(self.last_sample)
                last_seq = self.state_seq
        return samples


def build_cases(
    stiffness_values: Sequence[tuple[float, float, float]],
    damping_values: Sequence[tuple[float, float, float]],
    repeats: int,
) -> list[SweepCase]:
    return [
        SweepCase(stiffness=stiffness, damping=damping, repeat=repeat)
        for repeat in range(1, repeats + 1)
        for stiffness in stiffness_values
        for damping in damping_values
    ]


def build_delta_cases(args: argparse.Namespace, baseline_xyz: np.ndarray) -> list[DeltaCase]:
    if args.target_xyz is not None:
        return [DeltaCase(index=1, delta=np.array(args.target_xyz, dtype=float) - baseline_xyz)]
    if args.delta_m_list is not None:
        return [
            DeltaCase(index=index, delta=np.array(delta, dtype=float))
            for index, delta in enumerate(args.delta_m_list, start=1)
        ]
    if args.distance_m is not None:
        axis_index = AXES.index(args.axis)
        cases = []
        for index, distance_m in enumerate(args.distance_m, start=1):
            delta = np.zeros(3, dtype=float)
            delta[axis_index] = float(distance_m)
            cases.append(DeltaCase(index=index, delta=delta))
        return cases
    return [DeltaCase(index=1, delta=np.array(args.delta_m, dtype=float))]


def compute_row(
    *,
    case: SweepCase,
    delta_case: DeltaCase,
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    command_time_s: float,
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    rot_stiffness: float,
    rot_damping: float,
    tail_window_s: float,
) -> dict[str, object]:
    times = np.array([stamp - command_time_s for stamp, _, _ in samples], dtype=float)
    xyz = np.vstack([sample_xyz for _, sample_xyz, _ in samples])
    velocity = np.vstack([sample_velocity for _, _, sample_velocity in samples])

    commanded_delta = target_xyz - start_xyz
    reported_delta = xyz[-1] - start_xyz
    final_error_mm = (target_xyz - xyz[-1]) * 1000.0
    delta_error_mm = (reported_delta - commanded_delta) * 1000.0
    speed = np.linalg.norm(velocity, axis=1)

    overshoot_mm = []
    for axis_index, commanded in enumerate(commanded_delta):
        axis_delta = xyz[:, axis_index] - start_xyz[axis_index]
        if abs(float(commanded)) < 1e-9:
            overshoot = float(np.max(np.abs(axis_delta)))
        else:
            signed_delta = axis_delta * math.copysign(1.0, float(commanded))
            overshoot = max(0.0, float(np.max(signed_delta) - abs(float(commanded))))
        overshoot_mm.append(overshoot * 1000.0)

    tail_mask = times >= max(float(times[-1]) - tail_window_s, 0.0)
    tail_xyz = xyz[tail_mask]
    tail_p2p_mm = (np.max(tail_xyz, axis=0) - np.min(tail_xyz, axis=0)) * 1000.0

    row: dict[str, object] = {
        "delta_index": delta_case.index,
        "requested_delta_norm_m": float(np.linalg.norm(delta_case.delta)),
        "repeat": case.repeat,
        "rot_stiffness": rot_stiffness,
        "rot_damping": rot_damping,
        "sample_count": len(samples),
        "duration_s": float(times[-1]),
        "final_error_norm_mm": float(np.linalg.norm(final_error_mm)),
        "tail_xy_peak_to_peak_mm": float(np.linalg.norm(tail_p2p_mm[:2])),
        "peak_speed_mps": float(np.max(speed)),
        "peak_abs_velocity_z_mps": float(np.max(np.abs(velocity[:, 2]))),
    }

    ratios: list[float | str] = []
    for reported, commanded in zip(reported_delta, commanded_delta):
        ratios.append("" if abs(float(commanded)) < 1e-9 else float(reported / commanded))

    add_axis_values(row, "requested_delta", delta_case.delta, "_m")
    add_axis_values(row, "stiffness", case.stiffness)
    add_axis_values(row, "damping", case.damping)
    add_axis_values(row, "start", start_xyz, "_m")
    add_axis_values(row, "target", target_xyz, "_m")
    add_axis_values(row, "commanded_delta", commanded_delta, "_m")
    add_axis_values(row, "reported_tcp_delta", reported_delta, "_m")
    add_axis_values(row, "tracking_ratio", ratios)
    add_axis_values(row, "delta_error", delta_error_mm, "_mm")
    add_axis_values(row, "final_error", final_error_mm, "_mm")
    add_axis_values(row, "max_overshoot", overshoot_mm, "_mm")
    add_axis_values(row, "tail_peak_to_peak", tail_p2p_mm, "_mm")
    return row


def write_results(output_dir: Path, rows: list[dict[str, object]], config: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stiffness_damping_sweep.csv"
    json_path = output_dir / "stiffness_damping_sweep.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump({"config": config, "results": rows}, json_file, indent=2)
        json_file.write("\n")

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Cartesian stiffness/damping and compare commanded vs reported TCP delta."
    )
    parser.add_argument("--controller-namespace", default="aic_controller")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--delta-m", nargs=3, type=float, default=[0.02, 0.0, 0.0])
    parser.add_argument(
        "--delta-m-list",
        nargs="+",
        type=parse_xyz,
        help="Sweep arbitrary Cartesian deltas. Example: --delta-m-list 0.02,0,0 0.10,0,0",
    )
    parser.add_argument(
        "--distance-m",
        nargs="+",
        type=float,
        help="Sweep distances along --axis. Example: --axis x --distance-m 0.02 0.05 0.10 0.20",
    )
    parser.add_argument("--axis", choices=AXES, default="x")
    parser.add_argument("--target-xyz", nargs=3, type=float)
    parser.add_argument(
        "--stiffness-xyz",
        nargs="+",
        type=parse_xyz,
        default=[parse_xyz(value) for value in DEFAULT_STIFFNESS],
    )
    parser.add_argument(
        "--damping-xyz",
        nargs="+",
        type=parse_xyz,
        default=[parse_xyz(value) for value in DEFAULT_DAMPING],
    )
    parser.add_argument("--rot-stiffness", type=float, default=50.0)
    parser.add_argument("--rot-damping", type=float, default=20.0)
    parser.add_argument("--return-stiffness-xyz", type=parse_xyz, default=parse_xyz("100,100,100"))
    parser.add_argument("--return-damping-xyz", type=parse_xyz, default=parse_xyz("60,60,60"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--return-s", type=float, default=1.5)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--publish-period-s", type=float, default=0.05)
    parser.add_argument("--oscillation-window-s", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ais/ais_stiffness_damping_calibration/outputs"),
    )
    parser.add_argument("--skip-target-mode-switch", action="store_true")
    args, _ = parser.parse_known_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.target_xyz is not None and (args.distance_m is not None or args.delta_m_list is not None):
        parser.error("--target-xyz cannot be combined with --distance-m or --delta-m-list")
    if args.distance_m is not None and args.delta_m_list is not None:
        parser.error("--distance-m and --delta-m-list are mutually exclusive")
    return args


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    node = CalibrationNode(args.controller_namespace)
    try:
        need_mode_service = not args.skip_target_mode_switch
        node.wait_ready(args.wait_s, need_mode_service=need_mode_service)
        if need_mode_service:
            node.switch_to_cartesian(args.wait_s)

        baseline_pose = node.current_pose()
        baseline_xyz = pose_xyz(baseline_pose)
        delta_cases = build_delta_cases(args, baseline_xyz)
        cases = build_cases(args.stiffness_xyz, args.damping_xyz, args.repeats)
        total_trials = len(delta_cases) * len(cases)
        rows: list[dict[str, object]] = []

        node.get_logger().info(
            "Starting %d trials. baseline=%s deltas=%s"
            % (
                total_trials,
                np.array2string(baseline_xyz, precision=4),
                [delta_case.delta.tolist() for delta_case in delta_cases],
            )
        )

        trial_index = 0
        for delta_case in delta_cases:
            target_xyz = baseline_xyz + delta_case.delta
            target_pose = make_pose(target_xyz, pose_quat(baseline_pose))
            for case in cases:
                trial_index += 1
                node.publish_pose(
                    baseline_pose,
                    stiffness=args.return_stiffness_xyz,
                    damping=args.return_damping_xyz,
                    rot_stiffness=args.rot_stiffness,
                    rot_damping=args.rot_damping,
                    frame_id=args.frame_id,
                    publish_count=args.publish_count,
                    publish_period_s=args.publish_period_s,
                )
                node.collect(args.return_s)
                if node.last_sample is None:
                    raise RuntimeError("No controller_state sample before trial.")

                command_time_s = time.monotonic()
                start_xyz = node.last_sample[1]
                start_velocity = node.last_sample[2]
                node.publish_pose(
                    target_pose,
                    stiffness=case.stiffness,
                    damping=case.damping,
                    rot_stiffness=args.rot_stiffness,
                    rot_damping=args.rot_damping,
                    frame_id=args.frame_id,
                    publish_count=args.publish_count,
                    publish_period_s=args.publish_period_s,
                )
                samples = [(command_time_s, start_xyz, start_velocity)] + node.collect(args.duration_s)
                row = compute_row(
                    case=case,
                    delta_case=delta_case,
                    start_xyz=start_xyz,
                    target_xyz=target_xyz,
                    command_time_s=command_time_s,
                    samples=samples,
                    rot_stiffness=args.rot_stiffness,
                    rot_damping=args.rot_damping,
                    tail_window_s=args.oscillation_window_s,
                )
                rows.append(row)

                node.get_logger().info(
                    "[%d/%d] delta=%s k=%s d=%s final=%.2fmm tail_xy=%.2fmm "
                    "overshoot=(%.2f, %.2f, %.2f)mm z_peak=%.4fm/s"
                    % (
                        trial_index,
                        total_trials,
                        np.array2string(delta_case.delta, precision=3),
                        case.stiffness,
                        case.damping,
                        float(row["final_error_norm_mm"]),
                        float(row["tail_xy_peak_to_peak_mm"]),
                        float(row["max_overshoot_x_mm"]),
                        float(row["max_overshoot_y_mm"]),
                        float(row["max_overshoot_z_mm"]),
                        float(row["peak_abs_velocity_z_mps"]),
                    )
                )

        node.publish_pose(
            baseline_pose,
            stiffness=args.return_stiffness_xyz,
            damping=args.return_damping_xyz,
            rot_stiffness=args.rot_stiffness,
            rot_damping=args.rot_damping,
            frame_id=args.frame_id,
            publish_count=args.publish_count,
            publish_period_s=args.publish_period_s,
        )

        output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        config = {
            "frame_id": args.frame_id,
            "baseline_xyz_m": baseline_xyz.tolist(),
            "requested_delta_m": [delta_case.delta.tolist() for delta_case in delta_cases],
            "stiffness_xyz": [list(value) for value in args.stiffness_xyz],
            "damping_xyz": [list(value) for value in args.damping_xyz],
            "duration_s": args.duration_s,
            "return_s": args.return_s,
        }
        csv_path, json_path = write_results(output_dir, rows, config)
        node.get_logger().info(f"Wrote CSV: {csv_path}")
        node.get_logger().info(f"Wrote JSON: {json_path}")
        return csv_path, json_path
    finally:
        node.destroy_node()


def main() -> None:
    args = parse_args()
    try:
        rclpy.init()
        run(args)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
