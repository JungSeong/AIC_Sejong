from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import yaml


CLASS_ORDER = (
    "complete_insert",
    "partial_insert",
    "side_wall_contact",
    "top_surface_contact",
)


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def make_scenario(class_name: str, idx: int, rng: random.Random) -> dict:
    if class_name == "complete_insert":
        params = {
            "xy_offset_mm": _uniform(rng, -1.0, 1.0),
            "z_offset_mm": _uniform(rng, -1.0, 1.0),
            "yaw_offset_deg": _uniform(rng, -2.0, 2.0),
            "pitch_offset_deg": _uniform(rng, -2.0, 2.0),
            "insertion_depth_target_mm": _uniform(rng, 9.0, 14.0),
            "push_force_hint_n": _uniform(rng, 6.0, 10.0),
        }
    elif class_name == "partial_insert":
        params = {
            "xy_offset_mm": _uniform(rng, -1.5, 1.5),
            "z_offset_mm": _uniform(rng, -1.0, 1.0),
            "yaw_offset_deg": _uniform(rng, 3.0, 9.0) * rng.choice([-1, 1]),
            "pitch_offset_deg": _uniform(rng, 2.0, 7.0) * rng.choice([-1, 1]),
            "insertion_depth_target_mm": _uniform(rng, 2.5, 7.5),
            "push_force_hint_n": _uniform(rng, 8.0, 16.0),
        }
    elif class_name == "side_wall_contact":
        params = {
            "xy_offset_mm": _uniform(rng, 4.0, 10.0) * rng.choice([-1, 1]),
            "z_offset_mm": _uniform(rng, -1.5, 1.5),
            "yaw_offset_deg": _uniform(rng, -5.0, 5.0),
            "pitch_offset_deg": _uniform(rng, -4.0, 4.0),
            "insertion_depth_target_mm": _uniform(rng, 0.0, 3.0),
            "push_force_hint_n": _uniform(rng, 6.0, 14.0),
        }
    elif class_name == "top_surface_contact":
        params = {
            "xy_offset_mm": _uniform(rng, 2.0, 8.0) * rng.choice([-1, 1]),
            "z_offset_mm": _uniform(rng, 3.0, 9.0),
            "yaw_offset_deg": _uniform(rng, -4.0, 4.0),
            "pitch_offset_deg": _uniform(rng, 4.0, 12.0),
            "insertion_depth_target_mm": _uniform(rng, 0.0, 3.0),
            "push_force_hint_n": _uniform(rng, 4.0, 12.0),
        }
    else:
        raise ValueError(f"unknown class_name: {class_name}")

    return {
        "scenario_id": f"{class_name}_{idx:05d}",
        "class_name": class_name,
        "binary_success": 1 if class_name == "complete_insert" else 0,
        **params,
    }


def build_plan(episodes_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    scenarios: list[dict] = []
    for class_name in CLASS_ORDER:
        for idx in range(episodes_per_class):
            scenarios.append(make_scenario(class_name, idx, rng))
    rng.shuffle(scenarios)
    return scenarios


def write_plan(scenarios: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scenario_plan.csv"
    yaml_path = output_dir / "scenario_plan.yaml"
    if scenarios:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
            writer.writeheader()
            writer.writerows(scenarios)
    yaml_path.write_text(
        yaml.safe_dump({"scenarios": scenarios}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"scenario plan written: {csv_path}")
    print(f"scenario plan written: {yaml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced 4-class SFP retry classifier scenario plan."
    )
    parser.add_argument("--episodes-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=f"data/retry_classifier/plans/{time.strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_plan(build_plan(args.episodes_per_class, args.seed), Path(args.output_dir))


if __name__ == "__main__":
    main()
