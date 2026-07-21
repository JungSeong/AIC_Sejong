"""PortOffset 무작위 수집 모듈이 공유하는 경로와 물리 상수."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WS_SRC = ROOT / "ws_aic" / "src"
DATASET_ROOT = ROOT / "ws_aic" / "data" / "ais_portoffset_randomization"
CONFIG_DIR = Path("/tmp/ais_portoffset_randomization")
WORLD_TEMPLATE_PATH = (
    ROOT / "ws_aic" / "src" / "aic" / "aic_description" / "world" / "aic.sdf"
)
EPISODE_TRACKING_DIR = Path("/tmp/aic_episodes")
LEGACY_POLICY_STOP_FILE = Path("/tmp/aic_policy_stop")

POLICY_MODULE = "data_gen_node.PortOffsetCollect"
ENGINE_SETUP = "/ws_aic/install/setup.bash"
RUN_MARKER_ENV = "AIC_PORTOFFSET_RUN_ID"
REGISTRY_FILENAME = "owned_process_groups.json"

ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "blue": "\033[34m",
}

SFP_NIC_RAIL_COUNT = 5
SFP_PORT_COUNT = 2
SC_RAIL_COUNT = 2

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
