#!/usr/bin/env python3
"""Random-source random-walk Cartesian tracking check."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def sample_delta(
    rng: np.random.Generator,
    *,
    min_step_m: float,
    max_step_m: float,
    z_step_scale: float,
) -> np.ndarray:
    direction = rng.normal(size=3)
    direction[2] *= z_step_scale
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        direction /= norm
    length = float(rng.uniform(min_step_m, max_step_m))
    return direction * length


def sample_point_in_box(
    rng: np.random.Generator,
    center: np.ndarray,
    half_range: np.ndarray,
) -> np.ndarray:
    return center + rng.uniform(-half_range, half_range)


def sample_walk_target(
    rng: np.random.Generator,
    *,
    current_xyz: np.ndarray,
    center_xyz: np.ndarray,
    half_range: np.ndarray,
    min_step_m: float,
    max_step_m: float,
    z_step_scale: float,
) -> np.ndarray:
    lower = center_xyz - half_range
    upper = center_xyz + half_range
    for _ in range(200):
        target_xyz = current_xyz + sample_delta(
            rng,
            min_step_m=min_step_m,
            max_step_m=max_step_m,
            z_step_scale=z_step_scale,
        )
        if bool(np.all(target_xyz >= lower) and np.all(target_xyz <= upper)):
            return target_xyz

    for _ in range(200):
        target_xyz = sample_point_in_box(rng, center_xyz, half_range)
        distance = float(np.linalg.norm(target_xyz - current_xyz))
        if min_step_m <= distance <= max_step_m:
            return target_xyz

    target_xyz = np.clip(sample_point_in_box(rng, center_xyz, half_range), lower, upper)
    delta = target_xyz - current_xyz
    distance = float(np.linalg.norm(delta))
    if distance > max_step_m:
        target_xyz = current_xyz + delta * (max_step_m / distance)
    return np.clip(target_xyz, lower, upper)


def move_to_xyz(
    node: CalibrationNode,
    *,
    target_xyz: np.ndarray,
    quat,
    stiffness: tuple[float, float, float],
    damping: tuple[float, float, float],
    rot_stiffness: float,
    rot_damping: float,
    frame_id: str,
    publish_count: int,
    publish_period_s: float,
    max_segment_m: float,
    segment_s: float,
) -> None:
    if node.last_sample is None:
        raise RuntimeError("No controller_state sample before setup move.")
    start_xyz = node.last_sample[1]
    delta = target_xyz - start_xyz
    segment_count = max(1, int(math.ceil(float(np.linalg.norm(delta)) / max_segment_m)))
    for segment_index in range(1, segment_count + 1):
        fraction = segment_index / segment_count
        waypoint_xyz = start_xyz + delta * fraction
        node.publish_pose(
            make_pose(waypoint_xyz, quat),
            stiffness=stiffness,
            damping=damping,
            rot_stiffness=rot_stiffness,
            rot_damping=rot_damping,
            frame_id=frame_id,
            publish_count=publish_count,
            publish_period_s=publish_period_s,
        )
        node.collect(segment_s)


def measured_step(
    node: CalibrationNode,
    *,
    target_xyz: np.ndarray,
    quat,
    case: SweepCase,
    delta_index: int,
    duration_s: float,
    rot_stiffness: float,
    rot_damping: float,
    frame_id: str,
    publish_count: int,
    publish_period_s: float,
    tail_window_s: float,
) -> dict[str, object]:
    if node.last_sample is None:
        raise RuntimeError("No controller_state sample before random-walk step.")

    command_time_s = time.monotonic()
    start_xyz = node.last_sample[1]
    start_velocity = node.last_sample[2]
    target_pose = make_pose(target_xyz, quat)
    node.publish_pose(
        target_pose,
        stiffness=case.stiffness,
        damping=case.damping,
        rot_stiffness=rot_stiffness,
        rot_damping=rot_damping,
        frame_id=frame_id,
        publish_count=publish_count,
        publish_period_s=publish_period_s,
    )
    samples = [(command_time_s, start_xyz, start_velocity)] + node.collect(duration_s)
    return compute_row(
        case=case,
        delta_case=DeltaCase(index=delta_index, delta=target_xyz - start_xyz),
        start_xyz=start_xyz,
        target_xyz=target_xyz,
        command_time_s=command_time_s,
        samples=samples,
        rot_stiffness=rot_stiffness,
        rot_damping=rot_damping,
        tail_window_s=tail_window_s,
    )


def add_norm_metrics(row: dict[str, object]) -> None:
    row["commanded_delta_norm_m"] = vector_norm(
        [
            row["commanded_delta_x_m"],
            row["commanded_delta_y_m"],
            row["commanded_delta_z_m"],
        ]
    )
    row["delta_error_norm_mm"] = vector_norm(
        [
            row["delta_error_x_mm"],
            row["delta_error_y_mm"],
            row["delta_error_z_mm"],
        ]
    )
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


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {"all": rows}
    for row in rows:
        groups.setdefault(f"source_{row['source_index']}", []).append(row)

    summary_rows = []
    for group_name, group_rows in groups.items():
        final = [float(row["final_error_norm_mm"]) for row in group_rows]
        delta_error = [float(row["delta_error_norm_mm"]) for row in group_rows]
        overshoot = [float(row["overshoot_norm_mm"]) for row in group_rows]
        tail_xy = [float(row["tail_xy_peak_to_peak_mm"]) for row in group_rows]
        z_peak = [float(row["peak_abs_velocity_z_mps"]) * 1000.0 for row in group_rows]
        commanded = [float(row["commanded_delta_norm_m"]) * 1000.0 for row in group_rows]
        summary_rows.append(
            {
                "group": group_name,
                "count": len(group_rows),
                "commanded_mean_mm": float(np.mean(commanded)),
                "commanded_max_mm": float(np.max(commanded)),
                "final_error_mean_mm": float(np.mean(final)),
                "final_error_p95_mm": percentile(final, 95),
                "final_error_max_mm": float(np.max(final)),
                "delta_error_mean_mm": float(np.mean(delta_error)),
                "delta_error_p95_mm": percentile(delta_error, 95),
                "delta_error_max_mm": float(np.max(delta_error)),
                "overshoot_mean_mm": float(np.mean(overshoot)),
                "overshoot_p95_mm": percentile(overshoot, 95),
                "overshoot_max_mm": float(np.max(overshoot)),
                "tail_xy_mean_mm": float(np.mean(tail_xy)),
                "tail_xy_p95_mm": percentile(tail_xy, 95),
                "tail_xy_max_mm": float(np.max(tail_xy)),
                "z_peak_mean_mmps": float(np.mean(z_peak)),
                "z_peak_p95_mmps": percentile(z_peak, 95),
                "z_peak_max_mmps": float(np.max(z_peak)),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    summary_rows: list[dict[str, object]],
    config: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "random_walk.csv"
    json_path = output_dir / "random_walk.json"
    summary_csv_path = output_dir / "random_walk_summary.csv"
    summary_md_path = output_dir / "random_walk_summary.md"

    write_csv(csv_path, rows)
    write_csv(summary_csv_path, summary_rows)
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump({"config": config, "summary": summary_rows, "results": rows}, json_file, indent=2)
        json_file.write("\n")

    worst_final = sorted(rows, key=lambda row: float(row["final_error_norm_mm"]), reverse=True)[:10]
    summary_md_path.write_text(
        "\n".join(
            [
                "# Random Walk Tracking Summary",
                "",
                f"Steps: {len(rows)}",
                f"CSV: `{csv_path}`",
                f"JSON: `{json_path}`",
                "",
                "## Aggregate",
                "",
                markdown_table(
                    summary_rows,
                    [
                        "group",
                        "count",
                        "commanded_mean_mm",
                        "commanded_max_mm",
                        "final_error_mean_mm",
                        "final_error_p95_mm",
                        "final_error_max_mm",
                        "overshoot_mean_mm",
                        "overshoot_p95_mm",
                        "overshoot_max_mm",
                        "z_peak_p95_mmps",
                    ],
                ),
                "",
                "## Worst Final Error Steps",
                "",
                markdown_table(
                    worst_final,
                    [
                        "source_index",
                        "walk_step",
                        "commanded_delta_norm_m",
                        "final_error_norm_mm",
                        "delta_error_norm_mm",
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
    return csv_path, json_path, summary_csv_path, summary_md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random-source random-walk Cartesian tracking checks."
    )
    parser.add_argument("--controller-namespace", default="aic_controller")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--source-count", type=int, default=5)
    parser.add_argument("--steps-per-source", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-half-range-m", type=parse_xyz, default=parse_xyz("0.12,0.12,0.04"))
    parser.add_argument("--min-step-m", type=float, default=0.02)
    parser.add_argument("--max-step-m", type=float, default=0.10)
    parser.add_argument("--z-step-scale", type=float, default=0.35)
    parser.add_argument("--stiffness-xyz", type=parse_xyz, default=parse_xyz("200,200,100"))
    parser.add_argument("--damping-xyz", type=parse_xyz, default=parse_xyz("120,120,60"))
    parser.add_argument("--rot-stiffness", type=float, default=50.0)
    parser.add_argument("--rot-damping", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--source-segment-s", type=float, default=3.0)
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

    if args.source_count < 1:
        parser.error("--source-count must be >= 1")
    if args.steps_per_source < 1:
        parser.error("--steps-per-source must be >= 1")
    if args.min_step_m <= 0.0:
        parser.error("--min-step-m must be > 0")
    if args.max_step_m < args.min_step_m:
        parser.error("--max-step-m must be >= --min-step-m")
    if args.z_step_scale < 0.0:
        parser.error("--z-step-scale must be >= 0")
    if min(args.workspace_half_range_m) <= 0.0:
        parser.error("--workspace-half-range-m values must be > 0")
    return args


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    node = CalibrationNode(args.controller_namespace)
    try:
        need_mode_service = not args.skip_target_mode_switch
        node.wait_ready(args.wait_s, need_mode_service=need_mode_service)
        if need_mode_service:
            node.switch_to_cartesian(args.wait_s)

        rng = np.random.default_rng(args.seed)
        baseline_pose = node.current_pose()
        baseline_xyz = pose_xyz(baseline_pose)
        baseline_quat = pose_quat(baseline_pose)
        workspace_half_range = np.array(args.workspace_half_range_m, dtype=float)
        case = SweepCase(stiffness=args.stiffness_xyz, damping=args.damping_xyz, repeat=1)
        rows: list[dict[str, object]] = []
        step_index = 0

        node.get_logger().info(
            "Starting random walk: sources=%d steps/source=%d seed=%d baseline=%s "
            "workspace_half_range=%s max_step=%.3fm k=%s d=%s"
            % (
                args.source_count,
                args.steps_per_source,
                args.seed,
                np.array2string(baseline_xyz, precision=4),
                np.array2string(workspace_half_range, precision=4),
                args.max_step_m,
                args.stiffness_xyz,
                args.damping_xyz,
            )
        )

        for source_index in range(1, args.source_count + 1):
            source_target_xyz = sample_point_in_box(rng, baseline_xyz, workspace_half_range)
            move_to_xyz(
                node,
                target_xyz=source_target_xyz,
                quat=baseline_quat,
                stiffness=args.stiffness_xyz,
                damping=args.damping_xyz,
                rot_stiffness=args.rot_stiffness,
                rot_damping=args.rot_damping,
                frame_id=args.frame_id,
                publish_count=args.publish_count,
                publish_period_s=args.publish_period_s,
                max_segment_m=args.max_step_m,
                segment_s=args.source_segment_s,
            )
            if node.last_sample is None:
                raise RuntimeError("No controller_state sample after source setup.")
            source_actual_xyz = node.last_sample[1]
            source_setup_error_mm = (source_target_xyz - source_actual_xyz) * 1000.0

            for walk_step in range(1, args.steps_per_source + 1):
                step_index += 1
                current_xyz = node.last_sample[1]
                target_xyz = sample_walk_target(
                    rng,
                    current_xyz=current_xyz,
                    center_xyz=baseline_xyz,
                    half_range=workspace_half_range,
                    min_step_m=args.min_step_m,
                    max_step_m=args.max_step_m,
                    z_step_scale=args.z_step_scale,
                )
                metrics = measured_step(
                    node,
                    target_xyz=target_xyz,
                    quat=baseline_quat,
                    case=case,
                    delta_index=step_index,
                    duration_s=args.duration_s,
                    rot_stiffness=args.rot_stiffness,
                    rot_damping=args.rot_damping,
                    frame_id=args.frame_id,
                    publish_count=args.publish_count,
                    publish_period_s=args.publish_period_s,
                    tail_window_s=args.oscillation_window_s,
                )
                add_norm_metrics(metrics)

                row: dict[str, object] = {
                    "source_index": source_index,
                    "walk_step": walk_step,
                    "global_step": step_index,
                    "seed": args.seed,
                    "source_setup_error_norm_mm": vector_norm(source_setup_error_mm),
                }
                add_axis_values(row, "source_target", source_target_xyz, "_m")
                add_axis_values(row, "source_actual", source_actual_xyz, "_m")
                add_axis_values(row, "source_setup_error", source_setup_error_mm, "_mm")
                row.update(metrics)
                rows.append(row)

                node.get_logger().info(
                    "[%d/%d] source=%d step=%d cmd=%.1fmm final=%.2fmm "
                    "delta_err=%.2fmm overshoot=%.2fmm tail_xy=%.2fmm z_peak=%.2fmm/s"
                    % (
                        step_index,
                        args.source_count * args.steps_per_source,
                        source_index,
                        walk_step,
                        float(row["commanded_delta_norm_m"]) * 1000.0,
                        float(row["final_error_norm_mm"]),
                        float(row["delta_error_norm_mm"]),
                        float(row["overshoot_norm_mm"]),
                        float(row["tail_xy_peak_to_peak_mm"]),
                        float(row["peak_abs_velocity_z_mps"]) * 1000.0,
                    )
                )

        move_to_xyz(
            node,
            target_xyz=baseline_xyz,
            quat=baseline_quat,
            stiffness=args.stiffness_xyz,
            damping=args.damping_xyz,
            rot_stiffness=args.rot_stiffness,
            rot_damping=args.rot_damping,
            frame_id=args.frame_id,
            publish_count=args.publish_count,
            publish_period_s=args.publish_period_s,
            max_segment_m=args.max_step_m,
            segment_s=args.source_segment_s,
        )

        output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_rows = summarize_rows(rows)
        config = {
            "frame_id": args.frame_id,
            "baseline_xyz_m": baseline_xyz.tolist(),
            "source_count": args.source_count,
            "steps_per_source": args.steps_per_source,
            "seed": args.seed,
            "workspace_half_range_m": workspace_half_range.tolist(),
            "min_step_m": args.min_step_m,
            "max_step_m": args.max_step_m,
            "z_step_scale": args.z_step_scale,
            "stiffness_xyz": list(args.stiffness_xyz),
            "damping_xyz": list(args.damping_xyz),
            "rot_stiffness": args.rot_stiffness,
            "rot_damping": args.rot_damping,
            "duration_s": args.duration_s,
            "source_segment_s": args.source_segment_s,
        }
        csv_path, json_path, summary_csv_path, summary_md_path = write_results(
            output_dir,
            rows=rows,
            summary_rows=summary_rows,
            config=config,
        )
        overall = summary_rows[0]
        node.get_logger().info(
            "Random walk summary: final mean/p95/max = %.2f/%.2f/%.2fmm, "
            "overshoot mean/p95/max = %.2f/%.2f/%.2fmm"
            % (
                float(overall["final_error_mean_mm"]),
                float(overall["final_error_p95_mm"]),
                float(overall["final_error_max_mm"]),
                float(overall["overshoot_mean_mm"]),
                float(overall["overshoot_p95_mm"]),
                float(overall["overshoot_max_mm"]),
            )
        )
        node.get_logger().info(f"Wrote CSV: {csv_path}")
        node.get_logger().info(f"Wrote JSON: {json_path}")
        node.get_logger().info(f"Wrote summary CSV: {summary_csv_path}")
        node.get_logger().info(f"Wrote summary MD: {summary_md_path}")
        return csv_path, json_path, summary_csv_path, summary_md_path
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
