#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - reported at runtime with a clearer message.
    yaml = None


FIELDNAMES = [
    "timestamp",
    "case_name",
    "port_type",
    "target_module_name",
    "port_name",
    "target_frame",
    "base_frame",
    "source",
    "sample_id",
    "gt_x_m",
    "gt_y_m",
    "gt_z_m",
    "pred_x_m",
    "pred_y_m",
    "pred_z_m",
    "dx_m",
    "dy_m",
    "dz_m",
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "abs_dx_mm",
    "abs_dy_mm",
    "abs_dz_mm",
    "error_3d_m",
    "error_3d_mm",
]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    port_type: str
    port_name: str
    target_module_name: str
    target_frame: str


def _default_cases_path() -> Path:
    return Path(__file__).resolve().with_name("triangulation_cases.yaml")


def _default_output_dir() -> Path:
    return Path(__file__).resolve().with_name("results")


def load_cases(path: Path) -> dict[str, CaseSpec]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read triangulation_cases.yaml.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    trials = data.get("trials", {})
    cases: dict[str, CaseSpec] = {}
    for case_name, case in trials.items():
        tasks = case.get("tasks", {})
        if not tasks:
            continue
        task = tasks.get("task_1") or next(iter(tasks.values()))
        port_type = str(task.get("port_type", "") or task.get("plug_type", "")).lower()
        port_name = str(task.get("port_name", ""))
        module = str(task.get("target_module_name", ""))
        cases[case_name] = CaseSpec(
            name=case_name,
            port_type=port_type,
            port_name=port_name,
            target_module_name=module,
            target_frame=target_frame_from_task(port_type, port_name, module),
        )
    return cases


def target_frame_from_task(port_type: str, port_name: str, target_module_name: str) -> str:
    port_type = str(port_type or "").lower()
    if port_type == "sc":
        return f"task_board/{target_module_name}/sc_port_base_link_entrance"
    if port_name.endswith("_link"):
        return f"task_board/{target_module_name}/{port_name}_entrance"
    return f"task_board/{target_module_name}/{port_name}_link_entrance"


def manual_case(name: str, target_frame: str) -> CaseSpec:
    return CaseSpec(
        name=name,
        port_type="manual",
        port_name="manual",
        target_module_name="manual",
        target_frame=target_frame,
    )


def as_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing numeric value")
    return float(value)


def parse_xyz_value(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.replace(";", ",").split(",")]
        value = parsed
    if isinstance(value, dict):
        return as_float(value["x"]), as_float(value["y"]), as_float(value["z"])
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return as_float(value[0]), as_float(value[1]), as_float(value[2])
    return None


def extract_xyz(
    record: dict[str, Any],
    field_sets: Iterable[tuple[str, str, str]],
    vector_keys: Iterable[str],
) -> tuple[float, float, float] | None:
    for key in vector_keys:
        parsed = parse_xyz_value(record.get(key))
        if parsed is not None:
            return parsed
    for keys in field_sets:
        if all(key in record and record[key] not in (None, "") for key in keys):
            return as_float(record[keys[0]]), as_float(record[keys[1]]), as_float(record[keys[2]])
    return None


def prediction_xyz_from_record(record: dict[str, Any]) -> tuple[float, float, float] | None:
    return extract_xyz(
        record,
        (
            ("pred_x_m", "pred_y_m", "pred_z_m"),
            ("prediction_x_m", "prediction_y_m", "prediction_z_m"),
            ("triangulated_port_x", "triangulated_port_y", "triangulated_port_z"),
            ("x_m", "y_m", "z_m"),
            ("x", "y", "z"),
        ),
        ("pred_xyz_m", "prediction_xyz_m", "triangulated_port_xyz", "xyz_m", "xyz"),
    )


def gt_xyz_from_record(record: dict[str, Any]) -> tuple[float, float, float] | None:
    return extract_xyz(
        record,
        (
            ("gt_x_m", "gt_y_m", "gt_z_m"),
            ("ground_truth_x_m", "ground_truth_y_m", "ground_truth_z_m"),
            ("target_x_m", "target_y_m", "target_z_m"),
        ),
        ("gt_xyz_m", "ground_truth_xyz_m", "target_xyz_m"),
    )


def make_row(
    case: CaseSpec,
    gt_xyz: tuple[float, float, float],
    pred_xyz: tuple[float, float, float],
    *,
    base_frame: str,
    source: str,
    sample_id: str = "",
    timestamp: float | None = None,
) -> dict[str, Any]:
    dx = pred_xyz[0] - gt_xyz[0]
    dy = pred_xyz[1] - gt_xyz[1]
    dz = pred_xyz[2] - gt_xyz[2]
    error_3d_m = math.sqrt(dx * dx + dy * dy + dz * dz)
    return {
        "timestamp": f"{time.time() if timestamp is None else timestamp:.6f}",
        "case_name": case.name,
        "port_type": case.port_type,
        "target_module_name": case.target_module_name,
        "port_name": case.port_name,
        "target_frame": case.target_frame,
        "base_frame": base_frame,
        "source": source,
        "sample_id": sample_id,
        "gt_x_m": gt_xyz[0],
        "gt_y_m": gt_xyz[1],
        "gt_z_m": gt_xyz[2],
        "pred_x_m": pred_xyz[0],
        "pred_y_m": pred_xyz[1],
        "pred_z_m": pred_xyz[2],
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "dx_mm": dx * 1000.0,
        "dy_mm": dy * 1000.0,
        "dz_mm": dz * 1000.0,
        "abs_dx_mm": abs(dx * 1000.0),
        "abs_dy_mm": abs(dy * 1000.0),
        "abs_dz_mm": abs(dz * 1000.0),
        "error_3d_m": error_3d_m,
        "error_3d_mm": error_3d_m * 1000.0,
    }


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "triangulation_xyz_results.csv"
    jsonl_path = output_dir / "triangulation_xyz_results.jsonl"
    summary_path = output_dir / "triangulation_xyz_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_path.write_text(
        json.dumps(summarize(rows), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = _numeric_values(rows, "error_3d_mm")
    summary: dict[str, Any] = {
        "count": len(errors),
        "mean_error_3d_mm": statistics.fmean(errors) if errors else None,
        "median_error_3d_mm": statistics.median(errors) if errors else None,
        "max_error_3d_mm": max(errors) if errors else None,
        "mean_abs_dx_mm": statistics.fmean(_numeric_values(rows, "abs_dx_mm")) if errors else None,
        "mean_abs_dy_mm": statistics.fmean(_numeric_values(rows, "abs_dy_mm")) if errors else None,
        "mean_abs_dz_mm": statistics.fmean(_numeric_values(rows, "abs_dz_mm")) if errors else None,
        "cases": {},
    }
    for case_name in sorted({str(row.get("case_name", "")) for row in rows}):
        case_rows = [row for row in rows if str(row.get("case_name", "")) == case_name]
        case_errors = _numeric_values(case_rows, "error_3d_mm")
        summary["cases"][case_name] = {
            "count": len(case_errors),
            "mean_error_3d_mm": statistics.fmean(case_errors) if case_errors else None,
            "max_error_3d_mm": max(case_errors) if case_errors else None,
        }
    return summary


def read_prediction_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(record) for record in data]
        if isinstance(data, dict):
            for key in ("records", "results", "predictions"):
                if isinstance(data.get(key), list):
                    return [dict(record) for record in data[key]]
            return [data]
    raise ValueError(f"Unsupported prediction file format: {path}")


def case_for_record(record: dict[str, Any], cases: dict[str, CaseSpec]) -> CaseSpec:
    case_name = str(record.get("case_name") or record.get("case") or "").strip()
    if case_name:
        if case_name not in cases:
            raise KeyError(f"Unknown case_name in prediction file: {case_name}")
        return cases[case_name]
    target_frame = str(record.get("target_frame") or "").strip()
    if target_frame:
        return manual_case("manual", target_frame)
    raise KeyError("Prediction record needs case_name or target_frame.")


def evaluate_prediction_file(args: argparse.Namespace) -> None:
    cases = load_cases(args.cases)
    rows = [] if args.overwrite else read_existing_rows(args.output_dir / "triangulation_xyz_results.csv")
    for index, record in enumerate(read_prediction_file(args.predictions), start=1):
        case = case_for_record(record, cases)
        pred_xyz = prediction_xyz_from_record(record)
        gt_xyz = gt_xyz_from_record(record)
        if pred_xyz is None:
            raise ValueError(f"Prediction XYZ missing at record {index}: {record}")
        if gt_xyz is None:
            raise ValueError(
                f"GT XYZ missing at record {index}. "
                "Offline mode needs gt_x_m/gt_y_m/gt_z_m or gt_xyz_m in the file."
            )
        rows.append(
            make_row(
                case,
                gt_xyz,
                pred_xyz,
                base_frame=args.base_frame,
                source=str(args.predictions),
                sample_id=str(record.get("sample_id") or record.get("id") or index),
                timestamp=float(record["timestamp"]) if "timestamp" in record else None,
            )
        )
    write_outputs(rows, args.output_dir)
    print_saved_summary(args.output_dir, rows)


def print_saved_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize(rows)
    mean_error = summary["mean_error_3d_mm"]
    max_error = summary["max_error_3d_mm"]
    mean_text = "nan" if mean_error is None else f"{mean_error:.3f}"
    max_text = "nan" if max_error is None else f"{max_error:.3f}"
    print(
        "saved "
        f"count={summary['count']} "
        f"mean_3d_mm={mean_text} "
        f"max_3d_mm={max_text} "
        f"dir={output_dir}"
    )


def select_runtime_case(args: argparse.Namespace, cases: dict[str, CaseSpec]) -> CaseSpec:
    if args.target_frame:
        return manual_case(args.case_name or "manual", args.target_frame)
    if not args.case_name:
        raise SystemExit("--case-name is required unless --target-frame is provided.")
    if args.case_name not in cases:
        available = ", ".join(sorted(cases))
        raise SystemExit(f"Unknown case: {args.case_name}. Available cases: {available}")
    return cases[args.case_name]


def xyz_from_msg(msg: Any, message_type: str) -> tuple[float, float, float]:
    if message_type == "pose":
        point = msg.pose.position
    else:
        point = msg.point
    return float(point.x), float(point.y), float(point.z)


def lookup_gt_xyz(
    buffer: Any,
    node: Any,
    base_frame: str,
    target_frame: str,
    timeout_s: float,
    *,
    spin_during_wait: bool = False,
) -> tuple[float, float, float]:
    from rclpy.duration import Duration
    from rclpy.time import Time

    deadline = time.monotonic() + max(timeout_s, 0.0)
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            tf = buffer.lookup_transform(
                base_frame,
                target_frame,
                Time(),
                timeout=Duration(seconds=min(0.2, max(timeout_s, 0.0))),
            ).transform
            return float(tf.translation.x), float(tf.translation.y), float(tf.translation.z)
        except Exception as exc:  # pragma: no cover - depends on ROS runtime.
            last_error = exc
            if spin_during_wait:
                import rclpy

                rclpy.spin_once(node, timeout_sec=0.05)
            else:
                time.sleep(0.05)
    raise RuntimeError(f"GT TF lookup failed: {base_frame} <- {target_frame}: {last_error}")


def run_ros(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from geometry_msgs.msg import PointStamped, PoseStamped
    from tf2_ros import Buffer, TransformListener

    cases = load_cases(args.cases)
    case = select_runtime_case(args, cases)
    rows = [] if args.overwrite else read_existing_rows(args.output_dir / "triangulation_xyz_results.csv")

    rclpy.init()
    node = rclpy.create_node("triangulation_xyz_evaluator")
    buffer = Buffer()
    TransformListener(buffer, node)

    if args.pred_xyz is not None:
        gt_xyz = lookup_gt_xyz(
            buffer,
            node,
            args.base_frame,
            case.target_frame,
            args.tf_timeout,
            spin_during_wait=True,
        )
        rows.append(
            make_row(
                case,
                gt_xyz,
                tuple(args.pred_xyz),
                base_frame=args.base_frame,
                source="--pred-xyz",
                sample_id="manual",
            )
        )
        write_outputs(rows, args.output_dir)
        print_saved_summary(args.output_dir, rows)
        node.destroy_node()
        rclpy.shutdown()
        return

    msg_type = PoseStamped if args.message_type == "pose" else PointStamped

    def on_prediction(msg: Any) -> None:
        try:
            pred_xyz = xyz_from_msg(msg, args.message_type)
            gt_xyz = lookup_gt_xyz(
                buffer,
                node,
                args.base_frame,
                case.target_frame,
                args.tf_timeout,
            )
            timestamp = node.get_clock().now().nanoseconds / 1e9
            row = make_row(
                case,
                gt_xyz,
                pred_xyz,
                base_frame=args.base_frame,
                source=args.prediction_topic,
                sample_id=str(len(rows) + 1),
                timestamp=timestamp,
            )
            rows.append(row)
            write_outputs(rows, args.output_dir)
            print(
                f"[{case.name}] "
                f"err={float(row['error_3d_mm']):.3f}mm "
                f"dxyz=({float(row['dx_mm']):+.3f}, "
                f"{float(row['dy_mm']):+.3f}, "
                f"{float(row['dz_mm']):+.3f})mm"
            )
            if args.once:
                rclpy.shutdown()
        except Exception as exc:  # pragma: no cover - depends on ROS runtime.
            node.get_logger().error(str(exc))

    node.create_subscription(msg_type, args.prediction_topic, on_prediction, 10)
    node.get_logger().info(
        "triangulation evaluator listening: "
        f"case={case.name}, target_frame={case.target_frame}, "
        f"topic={args.prediction_topic}"
    )
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare triangulated port XYZ against target-frame ground truth."
    )
    parser.add_argument("--cases", type=Path, default=_default_cases_path())
    parser.add_argument("--case-name", default="")
    parser.add_argument("--target-frame", default="")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--prediction-topic", default="/final_policy/triangulated_port_xyz")
    parser.add_argument("--message-type", choices=("point", "pose"), default="point")
    parser.add_argument("--pred-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--tf-timeout", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predictions is not None:
        evaluate_prediction_file(args)
        return
    run_ros(args)


if __name__ == "__main__":
    main()
