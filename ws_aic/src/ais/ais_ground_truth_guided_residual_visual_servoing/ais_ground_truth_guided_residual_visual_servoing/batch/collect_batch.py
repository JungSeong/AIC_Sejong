from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..core.config import SfpGrvsConfig
from ..data.replay_buffer import append_replay_event
from .engine_config import make_sfp_batch_config, write_engine_config
from .runner import TRAINING_POLICY, run_engine, start_policy, stop_policy, wait_for_policy_start


def default_batch_id() -> str:
    return time.strftime("collect_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one SFP-only GRVS batch into the replay buffer."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument("--engine-setup", default="/ws_aic/install/setup.bash")
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--policy-start-wait-s", type=float, default=5.0)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--distance-model", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_collection(args: argparse.Namespace) -> int:
    batch_id = args.batch_id or default_batch_id()
    config_path = SfpGrvsConfig.BATCH_DIR / "collect" / f"{batch_id}.yaml"
    config = make_sfp_batch_config(
        episodes=args.episodes,
        seed=args.seed,
        diversify=not args.fixed,
        time_limit_s=args.time_limit_s,
    )
    write_engine_config(config, config_path)
    if not args.dry_run:
        append_replay_event(
            phase="collect_config",
            batch_id=batch_id,
            episodes=args.episodes,
            config_path=config_path,
            extra={"seed": args.seed, "fixed": bool(args.fixed)},
        )

    policy = start_policy(
        policy_module=TRAINING_POLICY,
        domain_id=args.domain_id,
        batch_id=batch_id,
        phase="collect",
        yolo_model=args.yolo_model or None,
        distance_model=args.distance_model or None,
        dry_run=args.dry_run,
    )
    try:
        wait_for_policy_start(args.policy_start_wait_s, args.dry_run)
        return_code = run_engine(
            config_path=config_path,
            domain_id=args.domain_id,
            distrobox=args.distrobox,
            engine_setup=args.engine_setup,
            dry_run=args.dry_run,
        )
    finally:
        stop_policy(policy)

    if not args.dry_run:
        append_replay_event(
            phase="collect_done",
            batch_id=batch_id,
            episodes=args.episodes,
            config_path=config_path,
            extra={
                "return_code": return_code,
                "yolo_model": args.yolo_model,
                "distance_model": args.distance_model,
            },
        )
    print(f"collect batch done: batch_id={batch_id}, return_code={return_code}")
    print(f"replay buffer: {SfpGrvsConfig.REPLAY_BUFFER_DIR}")
    return int(return_code)


def main() -> None:
    raise SystemExit(run_collection(parse_args()))


if __name__ == "__main__":
    main()
