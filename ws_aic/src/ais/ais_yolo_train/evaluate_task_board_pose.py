#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

try:
    from .core.task_board_pose_eval import (
        evaluate_task_board_pose_dataset,
        summarize_pose_eval,
    )
except ImportError:
    from core.task_board_pose_eval import (
        evaluate_task_board_pose_dataset,
        summarize_pose_eval,
    )


def find_src_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pixi.toml").exists() and (path / "ais").exists():
            return path
        nested_src = path / "ws_aic" / "src"
        if (nested_src / "pixi.toml").exists() and (nested_src / "ais").exists():
            return nested_src
    raise RuntimeError("Could not find ws_aic/src root.")


def parse_device(value: str | None) -> str | int | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    if value.lower() == "cpu":
        return "cpu"
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    src_root = find_src_root(Path.cwd().resolve())
    ws_root = src_root.parent
    parser = argparse.ArgumentParser(
        description="Evaluate TASK_BOARD YOLO keypoint PnP pose quality."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ws_root / "data" / "yolo" / "approach" / "TASK_BOARD",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ws_root / "model" / "ais_yolo" / "approach" / "TASK_BOARD" / "weights" / "best.pt",
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help='e.g. "0", "cpu", or empty for Ultralytics default')
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--records-csv", type=Path, default=None)
    parser.add_argument("--consistency-csv", type=Path, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    counts = summary.get("counts", {})
    print(
        "records="
        f"{int(counts.get('records', 0))} "
        f"ok={int(counts.get('ok_records', 0))} "
        f"failed={int(counts.get('failed_records', 0))} "
        f"episodes={int(counts.get('consistency_records', 0))}"
    )
    for key, stats in summary.items():
        if key == "counts":
            continue
        print(
            f"{key:34s} "
            f"mean={stats['mean']:.3f} "
            f"p50={stats['p50']:.3f} "
            f"p90={stats['p90']:.3f} "
            f"p95={stats['p95']:.3f} "
            f"max={stats['max']:.3f}"
        )


def main() -> None:
    args = parse_args()
    records, consistency_records = evaluate_task_board_pose_dataset(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        split=args.split,
        conf=args.conf,
        imgsz=args.imgsz,
        device=parse_device(args.device),
        max_episodes=args.max_episodes,
    )
    summary = summarize_pose_eval(records, consistency_records)
    print_summary(summary)

    failed = [row for row in records if not row.get("ok")]
    if failed:
        print("\nfirst failures:")
        for row in failed[:10]:
            print(f"  {row.get('episode')} {row.get('camera')} {row.get('reason')}: {row.get('image')}")

    if args.records_csv is not None:
        write_csv(args.records_csv, records)
        print(f"\nrecords_csv={args.records_csv}")
    if args.consistency_csv is not None:
        write_csv(args.consistency_csv, consistency_records)
        print(f"consistency_csv={args.consistency_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
