from __future__ import annotations

import os
from pathlib import Path


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


SRC_ROOT = _resolve_src_root()
WS_ROOT = SRC_ROOT.parent
RL_ROOT = WS_ROOT / "data" / "reinforcement_learning" / "SFP"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class SfpRlConfig:
    ROLLOUT_DIR = Path(
        os.environ.get("AIC_RL_ROLLOUT_DIR", RL_ROOT / "semi_cheatcode_rollouts")
    )
    RECORD_ROLLOUTS = env_bool("AIC_RL_RECORD_ROLLOUTS", True)

    CHEAT_ALIGN_Z_START_M = env_float("AIC_RL_CHEAT_ALIGN_Z_START_M", 0.100)
    CHEAT_ALIGN_Z_END_M = env_float("AIC_RL_CHEAT_ALIGN_Z_END_M", 0.005)
    CHEAT_INSERT_Z_END_M = env_float("AIC_RL_CHEAT_INSERT_Z_END_M", -0.015)
    CHEAT_ALIGN_STEPS = env_int("AIC_RL_CHEAT_ALIGN_STEPS", 50)
    CHEAT_INSERT_STEPS = env_int("AIC_RL_CHEAT_INSERT_STEPS", 40)
    CHEAT_STEP_DT_S = env_float("AIC_RL_CHEAT_STEP_DT_S", 0.05)
    CHEAT_SETTLE_S = env_float("AIC_RL_CHEAT_SETTLE_S", 0.30)

    SUPERVISED_DATA = Path(
        os.environ.get("AIC_RL_SUPERVISED_DATA", ROLLOUT_DIR / "samples.jsonl")
    )
    SUPERVISED_OUTPUT = Path(
        os.environ.get(
            "AIC_RL_SUPERVISED_OUTPUT",
            WS_ROOT / "model" / "ais_reinforcement_learning" / "sfp_action_mlp.pt",
        )
    )
