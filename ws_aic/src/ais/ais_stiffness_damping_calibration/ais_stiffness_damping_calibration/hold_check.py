#!/usr/bin/env python3
"""Repeated no-op Cartesian target check for TCP hold drift."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from ais_stiffness_damping_calibration.calibrate import (
    CalibrationNode,
    DeltaCase,
    SweepCase,
    add_axis_values,
    compute_row,
    make_pose,
    parse_xyz,
    pose_quat,
    pose_xyz,
)


def vector_norm(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(values, dtype=float)))


def add_hold_metrics(row: dict[str, object]) -> None:
    drift_mm = [
        float(row["reported_tcp_delta_x_m"]) * 1000.0,
        float(row["reported_tcp_delta_y_m"]) * 1000.0,
        float(row["reported_tcp_delta_z_m"]) * 1000.0,
    ]
    row["hold_drift_norm_mm"] = vector_norm(drift_mm)
    add_axis_values(row, "hold_drift", drift_mm, "_mm")
    row["overshoot_norm_mm"] = vector_norm(
        [
            row["max_overshoot_x_mm"],
            row["max_overshoot_y_mm"],
            row["max_overshoot_z_mm"],
        ]
    )
    row["tail_peak_to_peak_norm_mm"] = vector_norm(
        [
            row["tail_peak_to_peak_x_mm"],
            row["tail_peak_to_peak_y_mm"],
            row["tail_peak_to_peak_z_mm"],
        ]
    )


def percentile(values: Sequence[float], percent: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), percent))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    drift = [float(row["hold_drift_norm_mm"]) for row in rows]
    drift_x = [float(row["hold_drift_x_mm"]) for row in rows]
    drift_y = [float(row["hold_drift_y_mm"]) for row in rows]
    drift_z = [float(row["hold_drift_z_mm"]) for row in rows]
    final = [float(row["final_error_norm_mm"]) for row in rows]
    overshoot = [float(row["overshoot_norm_mm"]) for row in rows]
    tail_xy = [float(row["tail_xy_peak_to_peak_mm"]) for row in rows]
    z_peak = [float(row["peak_abs_velocity_z_mps"]) * 1000.0 for row in rows]
    return {
        "count": len(rows),
        "drift_mean_mm": float(np.mean(drift)),
        "drift_p95_mm": percentile(drift, 95),
        "drift_max_mm": float(np.max(drift)),
        "drift_x_mean_mm": float(np.mean(drift_x)),
        "drift_y_mean_mm": float(np.mean(drift_y)),
        "drift_z_mean_mm": float(np.mean(drift_z)),
        "drift_z_p95_abs_mm": percentile(np.abs(drift_z), 95),
        "drift_z_max_abs_mm": float(np.max(np.abs(drift_z))),
        "final_error_mean_mm": float(np.mean(final)),
        "final_error_max_mm": float(np.max(final)),
        "overshoot_mean_mm": float(np.mean(overshoot)),
        "overshoot_max_mm": float(np.max(overshoot)),
        "tail_xy_mean_mm": float(np.mean(tail_xy)),
        "tail_xy_max_mm": float(np.max(tail_xy)),
        "z_peak_mean_mmps": float(np.mean(z_peak)),
        "z_peak_max_mmps": float(np.max(z_peak)),
    }


def markdown_table(rows: list[dict[str, object]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_results(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    config: dict[str, object],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hold_check.csv"
    json_path = output_dir / "hold_check.json"
    summary_path = output_dir / "hold_check_summary.md"

    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump({"config": config, "summary": summary, "results": rows}, json_file, indent=2)
        json_file.write("\n")

    summary_path.write_text(
        "\n".join(
            [
                "# Hold Check Summary",
                "",
                f"CSV: `{csv_path}`",
                f"JSON: `{json_path}`",
                "",
                "## Aggregate",
                "",
                markdown_table([summary], summary.keys()),
                "",
                "## Per Command",
                "",
                markdown_table(
                    rows,
                    [
                        "hold_index",
                        "hold_drift_x_mm",
                        "hold_drift_y_mm",
                        "hold_drift_z_mm",
                        "hold_drift_norm_mm",
                        "final_error_norm_mm",
                        "overshoot_norm_mm",
                        "tail_xy_peak_to_peak_mm",
                        "peak_abs_velocity_z_mps",
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, json_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send repeated current-TCP no-op pose commands and measure hold drift."
    )
    parser.add_argument("--controller-namespace", default="aic_controller")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--stiffness-xyz", type=parse_xyz, default=parse_xyz("200,200,100"))
    parser.add_argument("--damping-xyz", type=parse_xyz, default=parse_xyz("120,120,60"))
    parser.add_argument("--rot-stiffness", type=float, default=50.0)
    parser.add_argument("--rot-damping", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--pre-settle-s", type=float, default=0.5)
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--publish-period-s", type=float, default=0.05)
    parser.add_argument("--oscillation-window-s", type=float, default=0.75)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ais/ais_stiffness_damping_calibration/outputs"),
    )
    parser.add_argument("--skip-target-mode-switch", action="store_true")
    args, _ = parser.parse_known_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be > 0")
    if args.pre_settle_s < 0.0:
        parser.error("--pre-settle-s must be >= 0")
    return args


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    node = CalibrationNode(args.controller_namespace)
    try:
        need_mode_service = not args.skip_target_mode_switch
        node.wait_ready(args.wait_s, need_mode_service=need_mode_service)
        if need_mode_service:
            node.switch_to_cartesian(args.wait_s)

        case = SweepCase(stiffness=args.stiffness_xyz, damping=args.damping_xyz, repeat=1)
        initial_pose = node.current_pose()
        initial_xyz = pose_xyz(initial_pose)
        rows: list[dict[str, object]] = []

        node.get_logger().info(
            "Starting hold check: count=%d duration=%.2fs start=%s k=%s d=%s"
            % (
                args.count,
                args.duration_s,
                np.array2string(initial_xyz, precision=4),
                args.stiffness_xyz,
                args.damping_xyz,
            )
        )

        for hold_index in range(1, args.count + 1):
            if args.pre_settle_s > 0.0:
                node.collect(args.pre_settle_s)
            if node.last_sample is None:
                raise RuntimeError("No controller_state sample before hold command.")

            current_pose = node.current_pose()
            command_time_s = time.monotonic()
            start_xyz = pose_xyz(current_pose)
            start_velocity = node.last_sample[2]
            target_xyz = np.array(start_xyz, dtype=float)
            node.publish_pose(
                make_pose(target_xyz, pose_quat(current_pose)),
                stiffness=args.stiffness_xyz,
                damping=args.damping_xyz,
                rot_stiffness=args.rot_stiffness,
                rot_damping=args.rot_damping,
                frame_id=args.frame_id,
                publish_count=args.publish_count,
                publish_period_s=args.publish_period_s,
            )
            samples = [(command_time_s, start_xyz, start_velocity)] + node.collect(args.duration_s)
            metrics = compute_row(
                case=case,
                delta_case=DeltaCase(index=hold_index, delta=np.zeros(3, dtype=float)),
                start_xyz=start_xyz,
                target_xyz=target_xyz,
                command_time_s=command_time_s,
                samples=samples,
                rot_stiffness=args.rot_stiffness,
                rot_damping=args.rot_damping,
                tail_window_s=args.oscillation_window_s,
            )
            add_hold_metrics(metrics)

            row: dict[str, object] = {"hold_index": hold_index}
            add_axis_values(row, "initial", initial_xyz, "_m")
            row.update(metrics)
            rows.append(row)

            node.get_logger().info(
                "[%d/%d] drift=(%.3f, %.3f, %.3f)mm norm=%.3fmm "
                "tail_xy=%.3fmm z_peak=%.3fmm/s"
                % (
                    hold_index,
                    args.count,
                    float(row["hold_drift_x_mm"]),
                    float(row["hold_drift_y_mm"]),
                    float(row["hold_drift_z_mm"]),
                    float(row["hold_drift_norm_mm"]),
                    float(row["tail_xy_peak_to_peak_mm"]),
                    float(row["peak_abs_velocity_z_mps"]) * 1000.0,
                )
            )

        output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = summarize_rows(rows)
        config = {
            "frame_id": args.frame_id,
            "initial_xyz_m": initial_xyz.tolist(),
            "count": args.count,
            "stiffness_xyz": list(args.stiffness_xyz),
            "damping_xyz": list(args.damping_xyz),
            "orientation_source": "current_tcp_pose_each_hold",
            "rot_stiffness": args.rot_stiffness,
            "rot_damping": args.rot_damping,
            "duration_s": args.duration_s,
            "pre_settle_s": args.pre_settle_s,
        }
        csv_path, json_path, summary_path = write_results(
            output_dir,
            rows=rows,
            summary=summary,
            config=config,
        )
        node.get_logger().info(
            "Hold summary: drift mean/p95/max = %.3f/%.3f/%.3fmm, "
            "z drift mean/max_abs = %.3f/%.3fmm"
            % (
                float(summary["drift_mean_mm"]),
                float(summary["drift_p95_mm"]),
                float(summary["drift_max_mm"]),
                float(summary["drift_z_mean_mm"]),
                float(summary["drift_z_max_abs_mm"]),
            )
        )
        node.get_logger().info(f"Wrote CSV: {csv_path}")
        node.get_logger().info(f"Wrote JSON: {json_path}")
        node.get_logger().info(f"Wrote summary: {summary_path}")
        return csv_path, json_path, summary_path
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
