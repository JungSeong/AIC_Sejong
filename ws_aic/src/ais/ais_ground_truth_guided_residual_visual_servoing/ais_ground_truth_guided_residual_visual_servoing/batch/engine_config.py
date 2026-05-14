from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import yaml

from ..core.config import SfpGrvsConfig


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
        "mount_rail": {
            "min_translation": -0.09425,
            "max_translation": 0.09425,
        },
    }


def _robot_section() -> dict:
    return {
        "home_joint_positions": {
            "shoulder_pan_joint": -0.1597,
            "shoulder_lift_joint": -1.3542,
            "elbow_joint": -1.6648,
            "wrist_1_joint": -1.6933,
            "wrist_2_joint": 1.5710,
            "wrist_3_joint": 1.4110,
        }
    }


def _board_pose(rng: random.Random, diversify: bool) -> dict:
    return {
        "x": rng.uniform(*LIMITS["board_x"]) if diversify else 0.15,
        "y": rng.uniform(*LIMITS["board_y"]) if diversify else -0.2,
        "z": 1.14,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": rng.uniform(*LIMITS["board_yaw"]) if diversify else 0.70,
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


def make_sfp_trial(
    *,
    index: int,
    rng: random.Random,
    diversify: bool,
    time_limit_s: int,
) -> dict:
    nic_rail = index % 5
    port_index = rng.choice((0, 1))
    port_name = f"sfp_port_{port_index}"
    task_board = {"pose": _board_pose(rng, diversify)}
    task_board.update(
        _nic_rails(
            nic_rail,
            rng.uniform(*LIMITS["nic_translation"]),
            rng.uniform(*LIMITS["nic_yaw"]),
        )
    )
    task_board.update(_background_sc_rails(rng))
    task_board.update(_mount_rails())

    noise = LIMITS["gripper_offset_noise"]
    gripper_offset = {
        "x": LIMITS["gripper_offset_x"] + rng.uniform(*noise),
        "y": LIMITS["gripper_offset_y"] + rng.uniform(*noise),
        "z": LIMITS["gripper_offset_z"] + rng.uniform(*noise),
    }
    return {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": gripper_offset,
                        "roll": 0.4432,
                        "pitch": -0.4838,
                        "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            f"sfp_batch_{index:04d}_{port_name}": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": port_name,
                "target_module_name": f"nic_card_mount_{nic_rail}",
                "time_limit": int(time_limit_s),
            }
        },
    }


def make_sfp_batch_config(
    *,
    episodes: int,
    seed: int,
    diversify: bool = True,
    time_limit_s: int = 600,
) -> dict:
    rng = random.Random(seed)
    return {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "trials": {
            f"trial_sfp_{index:04d}": make_sfp_trial(
                index=index,
                rng=rng,
                diversify=diversify,
                time_limit_s=time_limit_s,
            )
            for index in range(int(episodes))
        },
        "robot": _robot_section(),
    }


def write_engine_config(config: dict, path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an SFP-only GRVS batch config.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=SfpGrvsConfig.BATCH_DIR / "sfp_batch.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_sfp_batch_config(
        episodes=args.episodes,
        seed=args.seed,
        diversify=not args.fixed,
        time_limit_s=args.time_limit_s,
    )
    path = write_engine_config(config, args.output)
    print(path)


if __name__ == "__main__":
    main()
