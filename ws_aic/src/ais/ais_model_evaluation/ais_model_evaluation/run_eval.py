#!/usr/bin/env python3
"""Run ground-truth simulator evaluation for trained AIS models.

The simulator is expected to be running already with task/cable spawning
disabled. This runner creates one randomized GRVS-style engine config per
trial, starts the selected evaluation policy, lets aic_engine spawn the trial,
and records model-vs-ground-truth metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _resolve_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ws_aic" / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[5]


ROOT = _resolve_repo_root()
WS_ROOT = ROOT / "ws_aic"
WS_SRC = WS_ROOT / "src"
CONFIG_DIR = Path("/tmp/ais_model_evaluation")
CONFIG_PATH = CONFIG_DIR / "current_engine_config.yaml"
SCENARIO_PARAMS_PATH = CONFIG_DIR / "current_scenario_params.json"
POLICY_STOP_FILE = CONFIG_DIR / "policy_stop"
ENGINE_SETUP = "/ws_aic/install/setup.bash"

POLICY_MODULES = {
    "distance": "ais_model_evaluation.DistanceModelEvalPolicy",
    "orientation": "ais_model_evaluation.OrientationModelEvalPolicy",
}
COMPETITION_GRASP_RPY_DEG = math.degrees(0.04)


def _load_collect_module():
    module_name = "_ais_collect_rpy_randomization"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = WS_SRC / "ais" / "ais_auto_capture" / "collect_rpy_randomization.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _terminate_process_group(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def _terminate_process(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()


def _policy_module(args: argparse.Namespace) -> str:
    return args.policy or POLICY_MODULES[args.model_kind]


def cleanup_stale_processes(args: argparse.Namespace) -> None:
    try:
        POLICY_STOP_FILE.write_text("stop\n", encoding="utf-8")
    except OSError:
        pass

    policy = _policy_module(args)
    patterns = [
        rf"aic_model .*policy:={re.escape(policy)}",
        rf"aic_model .*{re.escape(policy)}",
        rf"aic_engine .*config_file_path:={re.escape(str(CONFIG_PATH))}",
        rf"distrobox enter .*aic_engine .*{re.escape(str(CONFIG_PATH))}",
    ]
    print("[cleanup] stale ais_model_evaluation/aic_engine processes 정리 중...")
    for pattern in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
    time.sleep(1.0)
    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)

    try:
        POLICY_STOP_FILE.unlink()
    except OSError:
        pass


def write_inputs(config: dict, scenario_params: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    SCENARIO_PARAMS_PATH.write_text(
        json.dumps(scenario_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _set_optional_env(env: dict[str, str], name: str, value: object | None) -> None:
    if value is not None:
        env[name] = str(value)


def start_policy(args: argparse.Namespace, trial_index: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["AIC_SCENARIO_PARAMS_FILE"] = str(SCENARIO_PARAMS_PATH)
    env["AIC_STOP_FILE"] = str(POLICY_STOP_FILE)
    env["AIC_MODEL_EVAL_RUN_ID"] = args.run_id
    env["AIC_MODEL_EVAL_OUTPUT_DIR"] = str(args.output_dir)
    env["AIC_MODEL_EVAL_TRIALS"] = str(args.trials)
    env["AIC_MODEL_EVAL_SEED"] = str(args.seed)
    env["AIC_MODEL_EVAL_TRIAL_INDEX"] = str(trial_index)
    env["AIC_MODEL_EVAL_CAMERAS"] = args.cameras
    env["AIC_DISTANCE_CAMERAS"] = args.cameras
    env["AIC_MODEL_EVAL_DEVICE"] = args.device
    env["AIC_DISTANCE_DEVICE"] = args.device
    env["AIC_MODEL_EVAL_MIN_VISIBLE_CAMERAS"] = str(args.min_visible_cameras)
    env["AIC_MODEL_EVAL_VISIBILITY_MARGIN_PX"] = str(args.visibility_margin_px)
    env["AIC_MODEL_EVAL_MAX_VISIBILITY_ATTEMPTS"] = str(args.max_visibility_attempts)
    env["AIC_MODEL_EVAL_RECORD_VIDEO"] = "1" if args.record_video else "0"
    env["AIC_MODEL_EVAL_VIDEO_CAMERAS"] = args.video_cameras or args.cameras
    env["AIC_MODEL_EVAL_VIDEO_FPS"] = str(args.video_fps)
    env["AIC_MODEL_EVAL_VIDEO_MAX_HEIGHT"] = str(args.video_max_height)
    env["AIC_MODEL_EVAL_INITIAL_SETTLE_S"] = str(args.initial_settle_s)
    env["AIC_MODEL_EVAL_INITIAL_WAIT_TIMEOUT_S"] = str(args.initial_wait_timeout_s)
    env["AIC_MODEL_EVAL_INITIAL_POSITION_TOLERANCE_MM"] = str(args.initial_position_tolerance_mm)
    env["AIC_MODEL_EVAL_INITIAL_ORIENTATION_TOLERANCE_DEG"] = str(args.initial_orientation_tolerance_deg)
    env["AIC_MODEL_EVAL_ACTION_SETTLE_S"] = str(args.action_settle_s)
    env["AIC_MODEL_EVAL_ACTION_WAIT_TIMEOUT_S"] = str(args.action_wait_timeout_s)
    env["AIC_MODEL_EVAL_ACTION_POSITION_TOLERANCE_MM"] = str(args.action_position_tolerance_mm)
    env["AIC_MODEL_EVAL_ACTION_ORIENTATION_TOLERANCE_DEG"] = str(args.action_orientation_tolerance_deg)
    env["AIC_MODEL_EVAL_DX_MIN_MM"] = str(args.dx_min_mm)
    env["AIC_MODEL_EVAL_DX_MAX_MM"] = str(args.dx_max_mm)
    env["AIC_MODEL_EVAL_DY_MIN_MM"] = str(args.dy_min_mm)
    env["AIC_MODEL_EVAL_DY_MAX_MM"] = str(args.dy_max_mm)
    env["AIC_MODEL_EVAL_DZ_MIN_MM"] = str(args.dz_min_mm)
    env["AIC_MODEL_EVAL_DZ_MAX_MM"] = str(args.dz_max_mm)
    env["AIC_MODEL_EVAL_ROLL_MIN_DEG"] = str(args.roll_min_deg)
    env["AIC_MODEL_EVAL_ROLL_MAX_DEG"] = str(args.roll_max_deg)
    env["AIC_MODEL_EVAL_PITCH_MIN_DEG"] = str(args.pitch_min_deg)
    env["AIC_MODEL_EVAL_PITCH_MAX_DEG"] = str(args.pitch_max_deg)
    env["AIC_MODEL_EVAL_YAW_MIN_DEG"] = str(args.yaw_min_deg)
    env["AIC_MODEL_EVAL_YAW_MAX_DEG"] = str(args.yaw_max_deg)
    env["AIC_MODEL_EVAL_RPY_NORM_MAX_DEG"] = str(args.rpy_norm_max_deg)
    env["AIC_MODEL_EVAL_MAX_DISTANCE_ACTION_M"] = str(args.max_distance_action_mm / 1000.0)
    env["AIC_MODEL_EVAL_MAX_ORIENTATION_ACTION_DEG"] = str(args.max_orientation_action_deg)
    env["AIC_MODEL_EVAL_DISTANCE_SUCCESS_MM"] = str(args.distance_success_mm)
    env["AIC_MODEL_EVAL_ORIENTATION_SUCCESS_DEG"] = str(args.orientation_success_deg)
    env["AIC_LEROBOT_REPO_ID"] = ""
    env["AIC_YOLO_DEBUG_VIDEO"] = "0"
    _set_optional_env(env, "AIC_DISTANCE_MODEL_PATH", args.distance_model_path)
    _set_optional_env(env, "AIC_ORIENTATION_MODEL_PATH", args.orientation_model_path)
    if POLICY_STOP_FILE.exists():
        POLICY_STOP_FILE.unlink()

    cmd = [
        "pixi",
        "run",
        "ros2",
        "run",
        "aic_model",
        "aic_model",
        "--ros-args",
        "-p",
        "use_sim_time:=true",
        "-p",
        f"policy:={_policy_module(args)}",
    ]
    print("[policy] " + shlex.join(cmd))
    return subprocess.Popen(cmd, cwd=WS_SRC, env=env, start_new_session=True)


def stop_policy(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        POLICY_STOP_FILE.write_text("stop\n", encoding="utf-8")
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
    finally:
        try:
            POLICY_STOP_FILE.unlink()
        except OSError:
            pass


def run_engine(args: argparse.Namespace) -> int:
    exports = [
        "export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}",
        "export ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}",
    ]
    inner = " && ".join(
        [
            f"source {shlex.quote(args.engine_setup)}",
            *exports,
            "ros2 run aic_engine aic_engine "
            "--ros-args "
            f"-p config_file_path:={shlex.quote(str(CONFIG_PATH))} "
            "-p ground_truth:=true "
            "-p use_sim_time:=true",
        ]
    )
    cmd = ["distrobox", "enter"]
    if not args.rootless_distrobox:
        cmd.append("-r")
    cmd += [args.distrobox, "--", "bash", "-lc", inner]
    print("[engine] " + shlex.join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_process(proc)
        raise


def _default_run_id(model_kind: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{model_kind}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AIS distance/orientation models in simulator with ground truth."
    )
    parser.add_argument("--model-kind", required=True, choices=("distance", "orientation"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument(
        "--rootless-distrobox",
        action="store_true",
        help="Use 'distrobox enter' without '-r'.",
    )
    parser.add_argument("--engine-setup", default=ENGINE_SETUP)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--policy-start-wait-s", type=float, default=5.0)
    parser.add_argument("--robot-joint-noise-deg", type=float, default=4.0)
    parser.add_argument("--cable-rpy-noise-deg", type=float, default=COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--cameras", default="left,center,right")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--min-visible-cameras",
        type=int,
        default=0,
        help="0 means all selected cameras must see the target port.",
    )
    parser.add_argument("--visibility-margin-px", type=float, default=8.0)
    parser.add_argument("--max-visibility-attempts", type=int, default=20)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="",
        help="Comma-separated cameras to compose in the video. Empty means --cameras.",
    )
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--video-max-height", type=int, default=360)
    parser.add_argument("--distance-model-path", default=None)
    parser.add_argument("--orientation-model-path", default=None)
    parser.add_argument("--initial-settle-s", type=float, default=0.2)
    parser.add_argument("--initial-wait-timeout-s", type=float, default=8.0)
    parser.add_argument("--initial-position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--initial-orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--action-settle-s", type=float, default=0.5)
    parser.add_argument("--action-wait-timeout-s", type=float, default=3.0)
    parser.add_argument("--action-position-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--action-orientation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--dx-min-mm", type=float, default=-50.0)
    parser.add_argument("--dx-max-mm", type=float, default=50.0)
    parser.add_argument("--dy-min-mm", type=float, default=-50.0)
    parser.add_argument("--dy-max-mm", type=float, default=50.0)
    parser.add_argument("--dz-min-mm", type=float, default=0.0)
    parser.add_argument("--dz-max-mm", type=float, default=100.0)
    parser.add_argument("--roll-min-deg", type=float, default=-COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--roll-max-deg", type=float, default=COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--pitch-min-deg", type=float, default=-COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--pitch-max-deg", type=float, default=COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--yaw-min-deg", type=float, default=-COMPETITION_GRASP_RPY_DEG)
    parser.add_argument("--yaw-max-deg", type=float, default=COMPETITION_GRASP_RPY_DEG)
    parser.add_argument(
        "--rpy-norm-max-deg",
        type=float,
        default=COMPETITION_GRASP_RPY_DEG,
        help="0 disables the sampled RPY norm cap.",
    )
    parser.add_argument(
        "--max-distance-action-mm",
        type=float,
        default=0.0,
        help="0 means no norm clipping.",
    )
    parser.add_argument(
        "--max-orientation-action-deg",
        type=float,
        default=0.0,
        help="0 means no norm clipping.",
    )
    parser.add_argument("--distance-success-mm", type=float, default=2.0)
    parser.add_argument("--orientation-success-deg", type=float, default=1.0)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing metrics.jsonl. By default existing outputs are rejected.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete an existing output directory before running.",
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.append and args.overwrite_output:
        parser.error("--append and --overwrite-output are mutually exclusive")
    args.run_id = args.run_id or _default_run_id(args.model_kind)
    args.output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else WS_ROOT / "data" / "ais_model_evaluation" / args.run_id
    )
    return args


def main() -> int:
    args = parse_args()
    if args.cleanup or args.cleanup_only:
        cleanup_stale_processes(args)
        if args.cleanup_only:
            return 0

    if not args.dry_run:
        metrics_path = args.output_dir / "metrics.jsonl"
        if args.overwrite_output and args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        elif metrics_path.exists() and not args.append:
            print(
                "[error] output already contains metrics.jsonl: "
                f"{metrics_path}\n"
                "Use a new --run-id, pass --overwrite-output to replace it, "
                "or pass --append if you intentionally want to append."
            )
            return 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
    collect = _load_collect_module()
    rng = random.Random(args.seed)
    policy_proc = None
    print(
        "[eval] "
        f"model_kind={args.model_kind}, trials={args.trials}, seed={args.seed}, "
        f"output={args.output_dir}"
    )

    try:
        for index in range(args.trials):
            config, scenario_params = collect.make_trial_config(index, rng, args)
            write_inputs(config, scenario_params)
            task_id = next(iter(scenario_params))
            print(f"\n=== eval trial {index + 1}/{args.trials}: {task_id} ===")

            if args.dry_run:
                print(CONFIG_PATH.read_text(encoding="utf-8"))
                continue

            policy_proc = start_policy(args, index)
            try:
                time.sleep(max(0.0, args.policy_start_wait_s))
                ret = run_engine(args)
                if ret != 0:
                    print(f"[warn] aic_engine exited with returncode={ret}")
            finally:
                stop_policy(policy_proc)
                policy_proc = None
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[interrupt] evaluation interrupted; cleaning policy/engine processes...")
        stop_policy(policy_proc)
        cleanup_stale_processes(args)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
