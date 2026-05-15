#!/usr/bin/env python3
"""Run sparse RPY randomization trials against an already-running simulator.

Start the simulator once with task/cable spawning disabled. This runner then
starts the PortOffsetCollect policy, asks aic_engine to spawn one randomized
GRVS-style trial, records sparse samples, tears the policy down, and repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
WS_SRC = ROOT / "ws_aic" / "src"
DATASET_ROOT = ROOT / "ws_aic" / "data" / "ais_rpy_randomization"
CONFIG_DIR = Path("/tmp/ais_rpy_randomization")
CONFIG_PATH = CONFIG_DIR / "current_engine_config.yaml"
SCENARIO_PARAMS_PATH = Path("/tmp/aic_scenario_params.json")
POLICY_STOP_FILE = Path("/tmp/aic_policy_stop")

POLICY_MODULE = "data_gen_node.PortOffsetCollect"
ENGINE_SETUP = "/ws_aic/install/setup.bash"


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


def cleanup_stale_processes(args: argparse.Namespace) -> None:
    """Clean only processes started by this collection flow."""
    try:
        POLICY_STOP_FILE.write_text("stop\n", encoding="utf-8")
    except OSError:
        pass

    patterns = [
        rf"aic_model .*policy:={re.escape(args.policy)}",
        rf"aic_model .*{re.escape(args.policy)}",
        rf"aic_engine .*config_file_path:={re.escape(str(CONFIG_PATH))}",
        rf"distrobox enter .*aic_engine .*{re.escape(str(CONFIG_PATH))}",
    ]

    print("[cleanup] stale PortOffsetCollect/aic_engine processes 정리 중...")
    for pattern in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
    time.sleep(1.0)
    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)

    try:
        POLICY_STOP_FILE.unlink()
    except OSError:
        pass


def dataset_dir(args: argparse.Namespace) -> Path:
    version = args.dataset_version.strip()
    return DATASET_ROOT / version if version else DATASET_ROOT


def set_optional_env(env: dict[str, str], name: str, value: float | None) -> None:
    if value is not None:
        env[name] = str(value)


BASE_ROBOT_HOME = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

LIMITS = {
    "nic_translation": (-0.0215, 0.0234),
    "nic_yaw": (-math.radians(10.0), math.radians(10.0)),
    "sc_translation": (-0.06, 0.055),
    "board_x": (0.13, 0.17),
    "board_y": (-0.25, -0.20),
    "board_yaw": (0.55, 0.80),
    "gripper_offset_noise": (-0.002, 0.002),
    "gripper_offset_x": 0.0,
    "gripper_offset_y": 0.015385,
    "gripper_offset_z": 0.04245,
    "cable_roll": 0.4432,
    "cable_pitch": -0.4838,
    "cable_yaw": 1.3303,
}


def _scoring_section() -> dict:
    return {
        "topics": [
            {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
            {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {
                "topic": {
                    "name": "/tf_static",
                    "type": "tf2_msgs/msg/TFMessage",
                    "latched": True,
                }
            },
            {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {
                "topic": {
                    "name": "/aic/gazebo/contacts/off_limit",
                    "type": "ros_gz_interfaces/msg/Contacts",
                }
            },
            {
                "topic": {
                    "name": "/fts_broadcaster/wrench",
                    "type": "geometry_msgs/msg/WrenchStamped",
                }
            },
            {
                "topic": {
                    "name": "/aic_controller/joint_commands",
                    "type": "aic_control_interfaces/msg/JointMotionUpdate",
                }
            },
            {
                "topic": {
                    "name": "/aic_controller/pose_commands",
                    "type": "aic_control_interfaces/msg/MotionUpdate",
                }
            },
            {
                "topic": {
                    "name": "/scoring/insertion_event",
                    "type": "std_msgs/msg/String",
                }
            },
            {
                "topic": {
                    "name": "/aic_controller/controller_state",
                    "type": "aic_control_interfaces/msg/ControllerState",
                }
            },
        ]
    }


def _task_board_limits_section() -> dict:
    return {
        "nic_rail": {
            "min_translation": LIMITS["nic_translation"][0],
            "max_translation": LIMITS["nic_translation"][1],
        },
        "sc_rail": {
            "min_translation": LIMITS["sc_translation"][0],
            "max_translation": LIMITS["sc_translation"][1],
        },
        "mount_rail": {"min_translation": -0.09425, "max_translation": 0.09425},
    }


def _robot_section(rng: random.Random, joint_noise_deg: float) -> dict:
    noise = math.radians(joint_noise_deg)
    return {
        "home_joint_positions": {
            name: value + rng.uniform(-noise, noise)
            for name, value in BASE_ROBOT_HOME.items()
        }
    }


def _board_pose(rng: random.Random) -> dict:
    return {
        "x": rng.uniform(*LIMITS["board_x"]),
        "y": rng.uniform(*LIMITS["board_y"]),
        "z": 1.14,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": rng.uniform(*LIMITS["board_yaw"]),
    }


def _nic_rails(active_rail: int, translation: float, yaw: float) -> dict:
    rails = {}
    for index in range(5):
        if index == active_rail:
            rails[f"nic_rail_{index}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{active_rail}",
                "entity_pose": {
                    "translation": translation,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": yaw,
                },
            }
        else:
            rails[f"nic_rail_{index}"] = {"entity_present": False}
    return rails


def _background_sc_rails(rng: random.Random) -> dict:
    return {
        "sc_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {
                "translation": rng.uniform(*LIMITS["sc_translation"]),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sc_rail_1": {"entity_present": False},
    }


def _mount_rails() -> dict:
    def present(name: str) -> dict:
        return {
            "entity_present": True,
            "entity_name": name,
            "entity_pose": {
                "translation": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        }

    return {
        "lc_mount_rail_0": present("lc_mount_0"),
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0": present("sc_mount_0"),
        "lc_mount_rail_1": present("lc_mount_1"),
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }


def make_trial_config(index: int, rng: random.Random, args: argparse.Namespace) -> tuple[dict, dict]:
    nic_rail = rng.randrange(5)
    port_index = rng.choice((0, 1))
    port_name = f"sfp_port_{port_index}"
    task_id = f"rpy_trial_{index:04d}_rail{nic_rail}_{port_name}"

    board = _board_pose(rng)
    nic_translation = rng.uniform(*LIMITS["nic_translation"])
    nic_yaw = rng.uniform(*LIMITS["nic_yaw"])
    gripper_offset = {
        axis: LIMITS[f"gripper_offset_{axis}"] + rng.uniform(*LIMITS["gripper_offset_noise"])
        for axis in ("x", "y", "z")
    }
    cable_rpy_noise = math.radians(args.cable_rpy_noise_deg)
    cable_roll = LIMITS["cable_roll"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise)
    cable_pitch = LIMITS["cable_pitch"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise)
    cable_yaw = LIMITS["cable_yaw"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise)
    robot = _robot_section(rng, args.robot_joint_noise_deg)

    task_board = {"pose": board}
    task_board.update(_nic_rails(nic_rail, nic_translation, nic_yaw))
    task_board.update(_background_sc_rails(rng))
    task_board.update(_mount_rails())

    trial = {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": gripper_offset,
                        "roll": cable_roll,
                        "pitch": cable_pitch,
                        "yaw": cable_yaw,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            task_id: {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": port_name,
                "target_module_name": f"nic_card_mount_{nic_rail}",
                "time_limit": int(args.time_limit_s),
            }
        },
    }
    config = {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "trials": {f"trial_{index:04d}": trial},
        "robot": robot,
    }
    scenario_params = {
        task_id: {
            "trial_type": 0,
            "rail_idx": nic_rail,
            "board_x": board["x"],
            "board_y": board["y"],
            "board_yaw": board["yaw"],
            "gripper_offset_x": gripper_offset["x"],
            "gripper_offset_y": gripper_offset["y"],
            "gripper_offset_z": gripper_offset["z"],
            "nic_translation": nic_translation,
            "nic_yaw": nic_yaw,
            "sc_translation": task_board["sc_rail_0"]["entity_pose"]["translation"],
            "sfp_port_idx": port_index,
            "cable_roll": cable_roll,
            "cable_pitch": cable_pitch,
            "cable_yaw": cable_yaw,
            "robot_home_joint_positions": robot["home_joint_positions"],
        }
    }
    return config, scenario_params


def write_inputs(config: dict, scenario_params: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    SCENARIO_PARAMS_PATH.write_text(
        json.dumps(scenario_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def start_policy(args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["AIC_SCENARIO_PARAMS_FILE"] = str(SCENARIO_PARAMS_PATH)
    env["AIC_STOP_FILE"] = str(POLICY_STOP_FILE)
    env["AIC_COLLECT_STEPS"] = str(args.samples_per_trial)
    env["AIC_RPY_DATASET_VERSION"] = args.dataset_version.strip()
    env["AIC_VISION_OFFSET_DATASET_DIR"] = str(dataset_dir(args))
    env["AIC_PORT_COLLECT_XY_LIMIT_MM"] = str(args.port_xy_limit_mm)
    env["AIC_PORT_COLLECT_Z_LIMIT_MM"] = str(args.port_z_limit_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DX_MIN_MM", args.dx_min_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DX_MAX_MM", args.dx_max_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DY_MIN_MM", args.dy_min_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DY_MAX_MM", args.dy_max_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DZ_MIN_MM", args.dz_min_mm)
    set_optional_env(env, "AIC_PORT_COLLECT_DZ_MAX_MM", args.dz_max_mm)
    env["AIC_PORT_COLLECT_ROLL_LIMIT_DEG"] = str(args.port_roll_limit_deg)
    env["AIC_PORT_COLLECT_PITCH_LIMIT_DEG"] = str(args.port_pitch_limit_deg)
    env["AIC_PORT_COLLECT_YAW_LIMIT_DEG"] = str(args.port_yaw_limit_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_ROLL_MIN_DEG", args.roll_min_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_ROLL_MAX_DEG", args.roll_max_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_PITCH_MIN_DEG", args.pitch_min_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_PITCH_MAX_DEG", args.pitch_max_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_YAW_MIN_DEG", args.yaw_min_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_YAW_MAX_DEG", args.yaw_max_deg)
    set_optional_env(env, "AIC_PORT_COLLECT_RPY_NORM_MAX_RAD", args.rpy_norm_max_rad)
    set_optional_env(
        env,
        "AIC_PORT_ACTUAL_RPY_NORM_MAX_RAD",
        args.actual_rpy_norm_max_rad,
    )
    env["AIC_RPY_MIN_VISIBLE_CAMERAS"] = str(args.min_visible_cameras)
    env["AIC_RPY_VISIBILITY_MARGIN_PX"] = str(args.visibility_margin_px)
    env["AIC_TRIANGULATION_STOP_Z_OFFSET"] = str(args.base_z_offset_mm / 1000.0)
    env["AIC_COLLECT_CAPTURE_SETTLE_SEC"] = str(args.capture_settle_s)
    env["AIC_LEROBOT_REPO_ID"] = ""
    env["AIC_YOLO_DEBUG_VIDEO"] = "0"
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
        f"policy:={args.policy}",
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


def run_engine(args: argparse.Namespace, config_path: Path) -> int:
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
            f"-p config_file_path:={shlex.quote(str(config_path))} "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sparse RPY-randomized samples from random GRVS-style trials."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-trial", type=int, default=24)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument(
        "--rootless-distrobox",
        action="store_true",
        help="Use 'distrobox enter' without '-r'. Default keeps the previous rootful '-r' behavior.",
    )
    parser.add_argument("--engine-setup", default=ENGINE_SETUP)
    parser.add_argument("--policy", default=POLICY_MODULE)
    parser.add_argument("--policy-start-wait-s", type=float, default=5.0)
    parser.add_argument("--robot-joint-noise-deg", type=float, default=4.0)
    parser.add_argument("--cable-rpy-noise-deg", type=float, default=20.0)
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Save under data/ais_rpy_randomization/{version}. Empty keeps the base directory.",
    )
    parser.add_argument("--port-xy-limit-mm", type=float, default=50.0)
    parser.add_argument("--port-z-limit-mm", type=float, default=100.0)
    parser.add_argument("--dx-min-mm", type=float, default=-50.0)
    parser.add_argument("--dx-max-mm", type=float, default=50.0)
    parser.add_argument("--dy-min-mm", type=float, default=-50.0)
    parser.add_argument("--dy-max-mm", type=float, default=50.0)
    parser.add_argument("--dz-min-mm", type=float, default=0.0)
    parser.add_argument("--dz-max-mm", type=float, default=100.0)
    parser.add_argument("--port-roll-limit-deg", type=float, default=25.0)
    parser.add_argument("--port-pitch-limit-deg", type=float, default=25.0)
    parser.add_argument("--port-yaw-limit-deg", type=float, default=35.0)
    parser.add_argument("--roll-min-deg", type=float, default=None)
    parser.add_argument("--roll-max-deg", type=float, default=None)
    parser.add_argument("--pitch-min-deg", type=float, default=None)
    parser.add_argument("--pitch-max-deg", type=float, default=None)
    parser.add_argument("--yaw-min-deg", type=float, default=None)
    parser.add_argument("--yaw-max-deg", type=float, default=None)
    parser.add_argument(
        "--rpy-norm-max-rad",
        type=float,
        default=None,
        help="Cap sampled port-local RPY vector magnitude in radians. Omit or use <=0 to disable.",
    )
    parser.add_argument(
        "--actual-rpy-norm-max-rad",
        type=float,
        default=None,
        help=(
            "Skip samples whose saved target plug-port quaternion angle exceeds this "
            "radian limit. Defaults to --rpy-norm-max-rad inside the policy."
        ),
    )
    parser.add_argument("--base-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--capture-settle-s", type=float, default=0.25)
    parser.add_argument("--min-visible-cameras", type=int, default=1)
    parser.add_argument("--visibility-margin-px", type=float, default=8.0)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove stale policy/engine processes from previous collection runs before starting.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Remove stale policy/engine processes and exit without collecting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cleanup or args.cleanup_only:
        cleanup_stale_processes(args)
        if args.cleanup_only:
            return 0

    rng = random.Random(args.seed)
    policy_proc = None
    try:
        for index in range(args.trials):
            config, scenario_params = make_trial_config(index, rng, args)
            write_inputs(config, scenario_params, CONFIG_PATH)
            task_id = next(iter(scenario_params))
            print(f"\n=== trial {index + 1}/{args.trials}: {task_id} ===")

            if args.dry_run:
                print(CONFIG_PATH.read_text(encoding="utf-8"))
                continue

            policy_proc = start_policy(args)
            try:
                time.sleep(max(0.0, args.policy_start_wait_s))
                ret = run_engine(args, CONFIG_PATH)
                if ret != 0:
                    print(f"[warn] aic_engine exited with returncode={ret}")
            finally:
                stop_policy(policy_proc)
                policy_proc = None
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[interrupt] collection interrupted; cleaning policy/engine processes...")
        stop_policy(policy_proc)
        cleanup_stale_processes(args)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
