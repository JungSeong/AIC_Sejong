from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..core.config import SfpGrvsConfig
from ..data.metrics import format_metric_summary, read_episode_metrics, summarize_metrics
from .collect_batch import run_collection
from .test_batch import run_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRVS batch cycles. One cycle is collect -> train -> test."
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--collect-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument("--engine-setup", default="/ws_aic/install/setup.bash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--distance-model", default="")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Stop before starting a new cycle after this many hours. 0 disables it.",
    )
    parser.add_argument("--no-model-snapshots", action="store_true")
    parser.add_argument(
        "--continue-on-collect-error",
        action="store_true",
        help="Continue to train/test when collection returns non-zero.",
    )
    parser.add_argument(
        "--continue-on-test-error",
        action="store_true",
        help="Continue to the next cycle when policy testing returns non-zero.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _train(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "ais_ground_truth_guided_residual_visual_servoing.training.train_joint",
    ]
    if args.skip_yolo:
        cmd.append("--skip-yolo")
    if args.skip_distance:
        cmd.append("--skip-distance")
    if args.skip_rotation:
        cmd.append("--skip-rotation")
    if args.dry_run:
        cmd.append("--dry-run")
        print("[dry-run] train:")
        print("  " + " ".join(cmd))
        return 0
    return subprocess.run(cmd).returncode


def _trained_yolo_model() -> str:
    return str(SfpGrvsConfig.YOLO_MODEL_DIR / "weights" / "best.pt")


def _trained_distance_model() -> str:
    return str(SfpGrvsConfig.DISTANCE_MODEL_DIR / "best.pt")


def _cycle_batch_id(prefix: str, cycle_number: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_cycle{cycle_number:03d}_{stamp}"


def _snapshot_file(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _snapshot_models(cycle_number: int) -> dict[str, str | None]:
    root = SfpGrvsConfig.MODEL_ROOT / "snapshots" / f"cycle_{cycle_number:03d}"
    snapshots = {
        "distance_best": _snapshot_file(
            SfpGrvsConfig.DISTANCE_MODEL_DIR / "best.pt",
            root / "distance_prediction" / "best.pt",
        ),
        "yolo_best": _snapshot_file(
            SfpGrvsConfig.YOLO_MODEL_DIR / "weights" / "best.pt",
            root / "yolo" / "best.pt",
        ),
        "rotation_best": _snapshot_file(
            SfpGrvsConfig.ROTATION_MODEL_DIR / "best.pt",
            root / "rotation_prediction" / "best.pt",
        ),
    }
    print(
        "GRVS model snapshots: "
        f"distance={snapshots['distance_best'] or 'N/A'}, "
        f"yolo={snapshots['yolo_best'] or 'N/A'}, "
        f"rotation={snapshots['rotation_best'] or 'N/A'}"
    )
    return snapshots


def _print_batch_summary(label: str, batch_id: str) -> None:
    records = read_episode_metrics(batch_id=batch_id)
    summary = summarize_metrics(records)
    print(format_metric_summary(label, summary))


def _run_cycle(
    args: argparse.Namespace,
    *,
    cycle_index: int,
    yolo_model: str,
    distance_model: str,
) -> tuple[int, str, str]:
    cycle_number = cycle_index + 1
    print(f"GRVS cycle {cycle_number}/{args.cycles}: collect -> train -> test")
    collect_batch_id = _cycle_batch_id("collect", cycle_number)
    collect_args = argparse.Namespace(
        episodes=args.collect_episodes,
        seed=args.seed + cycle_index,
        batch_id=collect_batch_id,
        domain_id=args.domain_id,
        distrobox=args.distrobox,
        engine_setup=args.engine_setup,
        time_limit_s=600,
        fixed=args.fixed,
        policy_start_wait_s=5.0,
        yolo_model=yolo_model,
        distance_model=distance_model,
        dry_run=args.dry_run,
    )
    code = run_collection(collect_args)
    _print_batch_summary(f"GRVS collect cycle {cycle_number}", collect_batch_id)
    if code != 0 and not args.continue_on_collect_error:
        return code, yolo_model, distance_model

    code = _train(args)
    if code != 0:
        return code, yolo_model, distance_model

    if not args.skip_yolo:
        yolo_model = _trained_yolo_model()
    if not args.skip_distance:
        distance_model = _trained_distance_model()
    if not args.dry_run and not args.no_model_snapshots:
        _snapshot_models(cycle_number)

    test_batch_id = _cycle_batch_id("test", cycle_number)
    test_args = argparse.Namespace(
        episodes=args.test_episodes,
        seed=args.seed + 10000 + cycle_index,
        batch_id=test_batch_id,
        domain_id=args.domain_id,
        distrobox=args.distrobox,
        engine_setup=args.engine_setup,
        time_limit_s=180,
        fixed=args.fixed,
        policy_start_wait_s=5.0,
        yolo_model=yolo_model,
        distance_model=distance_model,
        dry_run=args.dry_run,
    )
    code = run_test(test_args)
    _print_batch_summary(f"GRVS test cycle {cycle_number}", test_batch_id)
    if code != 0 and not args.continue_on_test_error:
        return code, yolo_model, distance_model
    return 0, yolo_model, distance_model


def main() -> None:
    args = parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")

    yolo_model = args.yolo_model
    distance_model = args.distance_model
    exit_code = 0
    started_at = time.monotonic()
    for cycle_index in range(args.cycles):
        if args.max_runtime_hours > 0:
            elapsed_hours = (time.monotonic() - started_at) / 3600.0
            if elapsed_hours >= args.max_runtime_hours:
                print(
                    "GRVS max runtime reached before next cycle: "
                    f"{elapsed_hours:.2f}h >= {args.max_runtime_hours:.2f}h"
                )
                break
        exit_code, yolo_model, distance_model = _run_cycle(
            args,
            cycle_index=cycle_index,
            yolo_model=yolo_model,
            distance_model=distance_model,
        )
        if exit_code != 0:
            break
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
