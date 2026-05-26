from __future__ import annotations

import argparse
from pathlib import Path

from ais_reinforcement_learning.config import RL_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless SFP RL fine-tuning entry point.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Supervised action model checkpoint to initialize from.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=RL_ROOT / "rl_runs",
        help="Directory for RL run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "RL fine-tuning scaffold is installed, but the simulator reset/step "
        "adapter is not wired yet. First collect semi-cheatcode rollouts and "
        "train the supervised checkpoint, then connect this entry point to the "
        "headless aic_engine trial loop."
    )


if __name__ == "__main__":
    main()
