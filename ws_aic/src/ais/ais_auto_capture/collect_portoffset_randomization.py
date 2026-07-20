#!/usr/bin/env python3
"""Run randomized PortOffsetCollect trials with an auto-started Gazebo simulator.

For each trial, this runner creates a randomized AIC engine config, creates a
randomized Gazebo world file for lighting variation, starts Gazebo + aic_engine,
starts the PortOffsetCollect policy, waits for collection to finish, and repeats.
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
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WS_SRC = ROOT / "ws_aic" / "src"
DATASET_ROOT = ROOT / "ws_aic" / "data" / "ais_portoffset_randomization"
CONFIG_DIR = Path("/tmp/ais_portoffset_randomization")
CONFIG_PATH = CONFIG_DIR / "current_engine_config.yaml"
SCENARIO_PARAMS_PATH = Path("/tmp/aic_scenario_params.json")
WORLD_TEMPLATE_PATH = ROOT / "ws_aic" / "src" / "aic" / "aic_description" / "world" / "aic.sdf"
WORLD_PATH = CONFIG_DIR / "current_randomized_world.sdf"
EPISODE_TRACKING_DIR = Path("/tmp/aic_episodes")
POLICY_STOP_FILE = Path("/tmp/aic_policy_stop")

POLICY_MODULE = "data_gen_node.PortOffsetCollect"
ENGINE_SETUP = "/ws_aic/install/setup.bash"
GAZEBO_HEAD_START_S = 5.0


def _terminate_process_group(proc: subprocess.Popen | None, timeout_s: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
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
        rf"distrobox enter .*entrypoint.sh .*aic_engine_config_file:={re.escape(str(CONFIG_PATH))}",
        rf"gz sim .*{re.escape(str(WORLD_PATH))}",
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


ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "blue": "\033[34m",
}


def _color(args: argparse.Namespace, text: str, color: str, *, bold: bool = True) -> str:
    if not getattr(args, "color_log", True) or os.environ.get("NO_COLOR"):
        return text
    prefix = ANSI_COLORS.get(color, "")
    if bold:
        prefix = ANSI_COLORS["bold"] + prefix
    return f"{prefix}{text}{ANSI_COLORS['reset']}"


def _mm(value_m: float) -> str:
    return f"{value_m * 1000.0:+.1f}mm"


def _deg(value_rad: float) -> str:
    return f"{math.degrees(value_rad):+.1f}deg"


def _vec_mm(values: tuple[float, float, float] | list[float]) -> str:
    return f"({_mm(float(values[0]))}, {_mm(float(values[1]))}, {_mm(float(values[2]))})"


def _vec_deg(values: tuple[float, float, float] | list[float]) -> str:
    return f"({_deg(float(values[0]))}, {_deg(float(values[1]))}, {_deg(float(values[2]))})"


def log_trial_randomization(
    *,
    index: int,
    total: int,
    task_id: str,
    scenario: dict,
    lighting: dict,
    args: argparse.Namespace,
) -> None:
    port_type = str(scenario.get("port_type", ""))
    rail_idx = int(scenario.get("rail_idx", -1))
    board_xyz = (
        float(scenario.get("board_x", 0.0)),
        float(scenario.get("board_y", 0.0)),
        1.14,
    )
    gripper_xyz = (
        float(scenario.get("gripper_offset_x", 0.0)),
        float(scenario.get("gripper_offset_y", 0.0)),
        float(scenario.get("gripper_offset_z", 0.0)),
    )
    cable_rpy = (
        float(scenario.get("cable_roll", 0.0)),
        float(scenario.get("cable_pitch", 0.0)),
        float(scenario.get("cable_yaw", 0.0)),
    )
    robot_home = scenario.get("robot_home_joint_positions", {}) or {}
    joint_delta_deg = {
        name: math.degrees(float(robot_home.get(name, base)) - float(base))
        for name, base in BASE_ROBOT_HOME.items()
    }

    print(_color(args, f"\n=== Trial {index + 1}/{total}: {task_id} ===", "blue"))
    print(
        _color(args, "[Task Board]", "cyan")
        + f" xyz={_vec_mm(board_xyz)} yaw={_deg(float(scenario.get('board_yaw', 0.0)))}"
    )

    if port_type == "sc":
        port_detail = (
            f"type=SC rail={rail_idx} target=sc_port_{rail_idx}/sc_port_base "
            f"sc_translation={_mm(float(scenario.get('sc_translation', 0.0)))}"
        )
    else:
        port_detail = (
            f"type=SFP rail={rail_idx} port=sfp_port_{int(scenario.get('sfp_port_idx', -1))} "
            f"nic_translation={_mm(float(scenario.get('nic_translation', 0.0)))} "
            f"nic_yaw={_deg(float(scenario.get('nic_yaw', 0.0)))} "
            f"background_sc_translation={_mm(float(scenario.get('sc_translation', 0.0)))}"
        )
    print(_color(args, "[Port]", "yellow") + " " + port_detail)
    print(
        _color(args, "[Cable / Robot]", "magenta")
        + f" gripper_offset={_vec_mm(gripper_xyz)} cable_rpy={_vec_deg(cable_rpy)} "
        + "joint_noise_deg={"
        + ", ".join(f"{name}:{value:+.1f}" for name, value in joint_delta_deg.items())
        + "}"
    )

    sim_parts = [
        f"headless={bool(args.headless)}",
        f"port_order={args.port_order}",
        f"samples={args.samples_per_trial}",
    ]
    if lighting.get("enabled"):
        sim_parts.append(f"world={lighting.get('world_file', '')}")
        if "ambient" in lighting:
            sim_parts.append(f"ambient={float(lighting['ambient']):.3f}")
        if "background" in lighting:
            sim_parts.append(f"background={float(lighting['background']):.3f}")
        light_summary = []
        for name, info in (lighting.get("lights") or {}).items():
            rgb = info.get("diffuse_rgb") or (1.0, 1.0, 1.0)
            pose = info.get("pose") or []
            pose_xyz = _vec_mm(pose[:3]) if len(pose) >= 3 else "n/a"
            light_summary.append(
                f"{name}:intensity={float(info.get('intensity', 0.0)):.2f} "
                f"scale={float(info.get('scale', 0.0)):.2f} "
                f"rgb=({float(rgb[0]):.2f},{float(rgb[1]):.2f},{float(rgb[2]):.2f}) "
                f"xyz={pose_xyz}"
            )
        print(_color(args, "[Simulator / Lighting]", "green") + " " + "; ".join(sim_parts))
        for item in light_summary:
            print("  " + _color(args, "light", "green", bold=False) + " " + item)
    else:
        sim_parts.append("lighting=randomization_disabled")
        reason = lighting.get("reason")
        if reason:
            sim_parts.append(f"reason={reason}")
        print(_color(args, "[Simulator / Lighting]", "green") + " " + "; ".join(sim_parts))


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
    "sfp_board_x": (0.13, 0.17),
    "sfp_board_y": (-0.25, -0.20),
    "sfp_board_yaw": (0.55, 0.80),
    "sc_board_x": (0.15, 0.19),
    "sc_board_y": (-0.05, 0.05),
    "sc_board_yaw": (0.0, 3.1415),
    "gripper_offset_noise": (-0.002, 0.002),
    "sfp_gripper_offset_x": 0.0,
    "sfp_gripper_offset_y": 0.015385,
    "sfp_gripper_offset_z": 0.04245,
    "sc_gripper_offset_x": 0.0,
    "sc_gripper_offset_y": 0.015385,
    "sc_gripper_offset_z": 0.04045,
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


def _board_pose(rng: random.Random, port_type: str) -> dict:
    prefix = "sc" if port_type == "sc" else "sfp"
    return {
        "x": rng.uniform(*LIMITS[f"{prefix}_board_x"]),
        "y": rng.uniform(*LIMITS[f"{prefix}_board_y"]),
        "z": 1.14,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": rng.uniform(*LIMITS[f"{prefix}_board_yaw"]),
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


def _sc_rails_sc(active_rail: int, translation: float) -> dict:
    rails = {}
    for index in range(2):
        if index == active_rail:
            rails[f"sc_rail_{index}"] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{active_rail}",
                "entity_pose": {
                    "translation": translation,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            }
        else:
            rails[f"sc_rail_{index}"] = {"entity_present": False}
    return rails


def _mount_rails_nic() -> dict:
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


def _mount_rails_sc() -> dict:
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
        "lc_mount_rail_0": {"entity_present": False},
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0": present("sc_mount_2"),
        "lc_mount_rail_1": present("lc_mount_1"),
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }


def _gripper_offset(rng: random.Random, port_type: str) -> dict:
    prefix = "sc" if port_type == "sc" else "sfp"
    return {
        axis: LIMITS[f"{prefix}_gripper_offset_{axis}"]
        + rng.uniform(*LIMITS["gripper_offset_noise"])
        for axis in ("x", "y", "z")
    }


def _cable_rpy(rng: random.Random, args: argparse.Namespace) -> tuple[float, float, float]:
    cable_rpy_noise = math.radians(args.cable_rpy_noise_deg)
    return (
        LIMITS["cable_roll"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise),
        LIMITS["cable_pitch"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise),
        LIMITS["cable_yaw"] + rng.uniform(-cable_rpy_noise, cable_rpy_noise),
    )


def _make_sfp_trial(index: int, rng: random.Random, args: argparse.Namespace) -> tuple[str, dict, dict]:
    nic_rail = rng.randrange(5)
    port_index = rng.choice((0, 1))
    port_name = f"sfp_port_{port_index}"
    task_id = f"portoffset_sfp_{index:04d}_rail{nic_rail}_{port_name}"

    board = _board_pose(rng, "sfp")
    nic_translation = rng.uniform(*LIMITS["nic_translation"])
    nic_yaw = rng.uniform(*LIMITS["nic_yaw"])
    gripper_offset = _gripper_offset(rng, "sfp")
    cable_roll, cable_pitch, cable_yaw = _cable_rpy(rng, args)

    task_board = {"pose": board}
    task_board.update(_nic_rails(nic_rail, nic_translation, nic_yaw))
    task_board.update(_background_sc_rails(rng))
    task_board.update(_mount_rails_nic())

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
    scenario_params = {
        task_id: {
            "trial_type": 0,
            "port_type": "sfp",
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
        }
    }
    return task_id, trial, scenario_params


def _make_sc_trial(index: int, rng: random.Random, args: argparse.Namespace) -> tuple[str, dict, dict]:
    sc_rail = rng.randrange(2)
    task_id = f"portoffset_sc_{index:04d}_rail{sc_rail}"

    board = _board_pose(rng, "sc")
    sc_translation = rng.uniform(*LIMITS["sc_translation"])
    gripper_offset = _gripper_offset(rng, "sc")
    cable_roll, cable_pitch, cable_yaw = _cable_rpy(rng, args)

    task_board = {"pose": board}
    for rail_idx in range(5):
        task_board[f"nic_rail_{rail_idx}"] = {"entity_present": False}
    task_board.update(_sc_rails_sc(sc_rail, sc_translation))
    task_board.update(_mount_rails_sc())

    trial = {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_1": {
                    "pose": {
                        "gripper_offset": gripper_offset,
                        "roll": cable_roll,
                        "pitch": cable_pitch,
                        "yaw": cable_yaw,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable_reversed",
                }
            },
        },
        "tasks": {
            task_id: {
                "cable_type": "sfp_sc",
                "cable_name": "cable_1",
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{sc_rail}",
                "time_limit": int(args.time_limit_s),
            }
        },
    }
    scenario_params = {
        task_id: {
            "trial_type": 1,
            "port_type": "sc",
            "rail_idx": sc_rail,
            "board_x": board["x"],
            "board_y": board["y"],
            "board_yaw": board["yaw"],
            "gripper_offset_x": gripper_offset["x"],
            "gripper_offset_y": gripper_offset["y"],
            "gripper_offset_z": gripper_offset["z"],
            "nic_translation": 0.0,
            "nic_yaw": 0.0,
            "sc_translation": sc_translation,
            "sfp_port_idx": -1,
            "cable_roll": cable_roll,
            "cable_pitch": cable_pitch,
            "cable_yaw": cable_yaw,
        }
    }
    return task_id, trial, scenario_params


def _enabled_port_types(args: argparse.Namespace) -> list[str]:
    values = [token.strip().lower() for token in args.port_types.split(",")]
    port_types = [value for value in values if value in {"sfp", "sc"}]
    return port_types or ["sfp", "sc"]


def make_trial_config(index: int, rng: random.Random, args: argparse.Namespace) -> tuple[dict, dict]:
    port_types = _enabled_port_types(args)
    port_type = port_types[index % len(port_types)] if args.port_order == "round_robin" else rng.choice(port_types)
    if port_type == "sc":
        task_id, trial, scenario_params = _make_sc_trial(index, rng, args)
    else:
        task_id, trial, scenario_params = _make_sfp_trial(index, rng, args)
    robot = _robot_section(rng, args.robot_joint_noise_deg)
    config = {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "trials": {f"trial_{index:04d}_{port_type}": trial},
        "robot": robot,
    }
    scenario_params[task_id]["robot_home_joint_positions"] = robot["home_joint_positions"]
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
    env["AIC_CAPTURE_DIR"] = str(EPISODE_TRACKING_DIR)
    env["AIC_STOP_FILE"] = str(POLICY_STOP_FILE)
    env["AIC_COLLECT_STEPS"] = str(args.samples_per_trial)
    env["AIC_RPY_DATASET_VERSION"] = args.dataset_version.strip()
    env["AIC_VISION_OFFSET_DATASET_DIR"] = str(dataset_dir(args))
    env["AIC_VISION_OFFSET_PUSH_TO_HUB"] = "true" if args.push_to_hub else "false"
    if args.vision_offset_repo_id:
        env["AIC_VISION_OFFSET_REPO_ID"] = args.vision_offset_repo_id
    if args.vision_offset_hf_revision:
        env["AIC_VISION_OFFSET_HF_REVISION"] = args.vision_offset_hf_revision
    if args.vision_offset_hf_path_in_repo:
        env["AIC_VISION_OFFSET_HF_PATH_IN_REPO"] = args.vision_offset_hf_path_in_repo
    env["AIC_VISION_OFFSET_UPLOAD_ON_PORT_TYPE"] = args.upload_on_port_type
    env["AIC_VISION_OFFSET_HF_PRIVATE"] = "true" if args.hf_private else "false"
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
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    env["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
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


def _format_sdf_float(value: float) -> str:
    return f"{value:.6g}"


def _randomized_color(rng: random.Random, jitter: float) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, 1.0 + rng.uniform(-jitter, jitter))) for _ in range(3))


def write_randomized_world(index: int, rng: random.Random, args: argparse.Namespace) -> tuple[Path | None, dict]:
    if not args.randomize_lighting:
        return None, {"enabled": False}
    if not WORLD_TEMPLATE_PATH.exists():
        print(f"[warn] world template not found: {WORLD_TEMPLATE_PATH}")
        return None, {"enabled": False, "reason": "template_missing"}

    tree = ET.parse(WORLD_TEMPLATE_PATH)
    root = tree.getroot()
    metadata = {
        "enabled": True,
        "world_file": str(WORLD_PATH),
        "lights": {},
    }

    world = root.find("world")
    if world is None:
        raise RuntimeError(f"Invalid SDF world file: missing <world> in {WORLD_TEMPLATE_PATH}")

    scene = world.find("scene")
    if scene is not None:
        ambient = scene.find("ambient")
        if ambient is not None:
            ambient_level = rng.uniform(args.ambient_min, args.ambient_max)
            ambient.text = " ".join(_format_sdf_float(ambient_level) for _ in range(3))
            metadata["ambient"] = ambient_level
        background = scene.find("background")
        if background is not None:
            bg = rng.uniform(args.background_min, args.background_max)
            background.text = f"{_format_sdf_float(bg)} {_format_sdf_float(bg)} {_format_sdf_float(bg)} 1"
            metadata["background"] = bg

    for light in world.findall("light"):
        name = light.attrib.get("name", "")
        intensity = light.find("intensity")
        base_intensity = float(intensity.text.strip()) if intensity is not None and intensity.text else 1.0
        scale = rng.uniform(args.light_intensity_scale_min, args.light_intensity_scale_max)
        new_intensity = max(0.0, base_intensity * scale)
        if intensity is not None:
            intensity.text = _format_sdf_float(new_intensity)

        color = _randomized_color(rng, args.light_color_jitter)
        diffuse = light.find("diffuse")
        if diffuse is not None:
            diffuse.text = f"{_format_sdf_float(color[0])} {_format_sdf_float(color[1])} {_format_sdf_float(color[2])} 1"

        pose = light.find("pose")
        pose_values = []
        if pose is not None and pose.text:
            pose_values = [float(token) for token in pose.text.split()]
            if len(pose_values) >= 3:
                pose_values[0] += rng.uniform(-args.light_pose_xy_jitter_m, args.light_pose_xy_jitter_m)
                pose_values[1] += rng.uniform(-args.light_pose_xy_jitter_m, args.light_pose_xy_jitter_m)
                pose_values[2] = max(0.5, pose_values[2] + rng.uniform(-args.light_pose_z_jitter_m, args.light_pose_z_jitter_m))
                pose.text = " ".join(_format_sdf_float(v) for v in pose_values)

        metadata["lights"][name] = {
            "base_intensity": base_intensity,
            "intensity": new_intensity,
            "scale": scale,
            "diffuse_rgb": color,
            "pose": pose_values,
        }

    WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree.write(WORLD_PATH, encoding="utf-8", xml_declaration=True)
    return WORLD_PATH, metadata


def start_gazebo(args: argparse.Namespace, config_path: Path, world_path: Path | None) -> subprocess.Popen | None:
    launch_args = [
        "spawn_task_board:=false",
        "spawn_cable:=false",
        "ground_truth:=true",
        "start_aic_engine:=true",
        f"aic_engine_config_file:={config_path}",
    ]
    if world_path is not None:
        launch_args.append(f"world_file:={world_path}")
    if args.headless:
        launch_args += ["gazebo_gui:=false", "launch_rviz:=false"]

    args_str = " ".join(shlex.quote(value) for value in launch_args)
    exports = [
        'export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"',
        'export ZENOH_CONFIG_OVERRIDE="transport/shared_memory/enabled=false"',
    ]
    inner = " && ".join([*exports, f"/entrypoint.sh {args_str}"])
    cmd = ["distrobox", "enter"]
    if not args.rootless_distrobox:
        cmd.append("-r")
    cmd += [args.distrobox, "--", "bash", "-lc", inner]
    print("[gazebo] " + shlex.join(cmd))
    if args.dry_run:
        return None
    return subprocess.Popen(cmd, stderr=subprocess.STDOUT, start_new_session=True)


def _known_episode_summaries() -> set[Path]:
    return set(EPISODE_TRACKING_DIR.glob("*/episode_summary.json"))


def _summary_matches_task(path: Path, task_id: str) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(data.get("task_id", "")) == task_id


def wait_for_trial_summary(
    task_id: str,
    known_summaries: set[Path],
    timeout_s: float,
    watch_procs: list[subprocess.Popen | None],
) -> bool:
    deadline = time.monotonic() + max(1.0, timeout_s)
    print(f"[wait] episode summary 대기: task_id={task_id}, timeout={timeout_s:.1f}s")
    while time.monotonic() < deadline:
        for summary_path in _known_episode_summaries() - known_summaries:
            if _summary_matches_task(summary_path, task_id):
                print(f"[done] episode summary saved: {summary_path}")
                return True
        failed = [proc for proc in watch_procs if proc is not None and proc.poll() is not None]
        if failed:
            print(f"[warn] watched process exited before summary: returncode={failed[0].returncode}")
            return False
        time.sleep(1.0)
    print(f"[warn] timeout waiting for task summary: {task_id}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PortOffsetCollect samples from random GRVS-style trials."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-types", default="sfp,sc", help="Comma-separated target port types: sfp, sc, or sfp,sc.")
    parser.add_argument("--port-order", choices=("round_robin", "random"), default="round_robin")
    parser.add_argument("--color-log", dest="color_log", action="store_true", default=True)
    parser.add_argument("--no-color-log", dest="color_log", action="store_false")
    parser.add_argument("--samples-per-trial", type=int, default=24)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--trial-timeout-s", type=float, default=None, help="Wall-clock timeout per trial. Defaults to time-limit-s + 180s.")
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument("--headless", action="store_true", help="Start Gazebo without GUI/RViz.")
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
        help="Save under data/ais_portoffset_randomization/{version}. Empty keeps the base directory.",
    )
    parser.add_argument(
        "--push-to-hub",
        dest="push_to_hub",
        action="store_true",
        default=False,
        help="Upload the vision-offset dataset to Hugging Face after collection. Default is off.",
    )
    parser.add_argument(
        "--no-push-to-hub",
        dest="push_to_hub",
        action="store_false",
        help="Disable Hugging Face upload explicitly.",
    )
    parser.add_argument(
        "--vision-offset-repo-id",
        default="aic-sejong-team/aic-vision-offset-dataset",
        help="Hugging Face dataset repo id for vision-offset upload.",
    )
    parser.add_argument(
        "--vision-offset-hf-revision",
        default="main",
        help="Hugging Face revision/branch for vision-offset upload.",
    )
    parser.add_argument(
        "--vision-offset-hf-path-in-repo",
        default="",
        help="Optional path inside the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--upload-on-port-type",
        choices=("", "sfp", "sc"),
        default="",
        help="Only upload after this port type. Empty means no port-type filter.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        default=False,
        help="Create/use the Hugging Face dataset repo as private when uploading.",
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
    parser.add_argument("--randomize-lighting", dest="randomize_lighting", action="store_true", default=True)
    parser.add_argument("--no-randomize-lighting", dest="randomize_lighting", action="store_false")
    parser.add_argument("--light-intensity-scale-min", type=float, default=0.65)
    parser.add_argument("--light-intensity-scale-max", type=float, default=1.35)
    parser.add_argument("--light-color-jitter", type=float, default=0.12)
    parser.add_argument("--light-pose-xy-jitter-m", type=float, default=0.25)
    parser.add_argument("--light-pose-z-jitter-m", type=float, default=0.20)
    parser.add_argument("--ambient-min", type=float, default=0.0)
    parser.add_argument("--ambient-max", type=float, default=0.08)
    parser.add_argument("--background-min", type=float, default=0.08)
    parser.add_argument("--background-max", type=float, default=0.20)
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
    gazebo_proc = None
    trial_timeout_s = args.trial_timeout_s or (float(args.time_limit_s) + 180.0)
    try:
        for index in range(args.trials):
            config, scenario_params = make_trial_config(index, rng, args)
            task_id = next(iter(scenario_params))
            world_path, lighting_metadata = write_randomized_world(index, rng, args)
            scenario_params[task_id]["lighting"] = lighting_metadata
            write_inputs(config, scenario_params, CONFIG_PATH)
            known_summaries = _known_episode_summaries()
            log_trial_randomization(
                index=index,
                total=args.trials,
                task_id=task_id,
                scenario=scenario_params[task_id],
                lighting=lighting_metadata,
                args=args,
            )

            if args.dry_run:
                print(CONFIG_PATH.read_text(encoding="utf-8"))
                if world_path is not None:
                    print(f"[dry-run] randomized world: {world_path}")
                    print(json.dumps(lighting_metadata, indent=2, ensure_ascii=False))
                continue

            gazebo_proc = start_gazebo(args, CONFIG_PATH, world_path)
            try:
                print(f"[wait] Gazebo/aic_engine head start: {GAZEBO_HEAD_START_S:.1f}s")
                time.sleep(GAZEBO_HEAD_START_S)
                policy_proc = start_policy(args)
                wait_for_trial_summary(
                    task_id,
                    known_summaries,
                    trial_timeout_s,
                    [gazebo_proc, policy_proc],
                )
            finally:
                stop_policy(policy_proc)
                policy_proc = None
                _terminate_process_group(gazebo_proc)
                gazebo_proc = None
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[interrupt] collection interrupted; cleaning policy/engine/Gazebo processes...")
        stop_policy(policy_proc)
        _terminate_process_group(gazebo_proc)
        cleanup_stale_processes(args)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
