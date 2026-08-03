#!/usr/bin/env python3
"""PortOffset 범위의 triangulation case YAML을 만들고 AIC simulator를 실행한다."""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


TRIANGULATION_ROOT = Path(__file__).resolve().parent
AIS_ROOT = TRIANGULATION_ROOT.parent
AUTO_CAPTURE_ROOT = AIS_ROOT / "ais_auto_capture"
if str(AUTO_CAPTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_CAPTURE_ROOT))

from portoffset_randomization.constants import CLI_DEFAULTS
from portoffset_randomization.scenario import make_trial_config


BOLD_MAGENTA = "\033[1;35m"
BOLD_GREEN = "\033[1;32m"
ANSI_RESET = "\033[0m"

CAMERA_WIDTH = 1152
CAMERA_HEIGHT = 1024
CAMERA_HORIZONTAL_FOV_RAD = 0.8718
CAMERA_NEAR_M = 0.07
CAMERA_FAR_M = 20.0
DEFAULT_VISIBILITY_MARGIN_PX = 64.0
DEFAULT_MIN_VISIBLE_CAMERAS = 2
MAX_VISIBILITY_ATTEMPTS_PER_CASE = 10_000
DEFAULT_ROBOT_SPAWN = (-0.2, 0.2, 1.14, 0.0, 0.0, -3.141)

# base_link -> optical at BASE_ROBOT_HOME. Values come from the expanded
# ur_gz.urdf.xacro chain used by the simulator.
FIXED_HOME_CAMERA_OPTICAL_IN_BASE = {
    "left": np.array(
        [
            [0.499915949638, 0.836592857950, 0.224045605456, -0.471896872997],
            [0.866073897294, -0.482833595251, -0.129567448550, 0.252840246715],
            [-0.000218456897, 0.258812884773, -0.965927452220, 0.534876195104],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "center": np.array(
        [
            [0.999999988449, 0.000124277708, -0.000087504200, -0.371385883430],
            [0.000097372977, -0.965876306529, -0.259003766391, 0.310896517535],
            [-0.000116706628, 0.259003754879, -0.965876307473, 0.534855257450],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "right": np.array(
        [
            [0.500084644929, -0.836437458748, -0.224249248625, -0.270863708282],
            [-0.865976522094, -0.482994914026, -0.129617036714, 0.252859612625],
            [0.000105298239, 0.259014074188, -0.965873541560, 0.534852019966],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
}


def _bold_log(color: str, message: str) -> None:
    """중요 runner 상태를 ANSI bold/color로 출력한다."""
    print(f"{color}{message}{ANSI_RESET}", flush=True)


def _parse_bool(value: str) -> bool:
    """CLI의 true/false 문자열을 boolean으로 변환한다."""
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_sim_arg(value: str) -> tuple[str, str]:
    """추가 simulator launch argument를 NAME=VALUE 튜플로 파싱한다."""
    token = str(value).strip()
    separator = ":=" if ":=" in token else "="
    if separator not in token:
        raise argparse.ArgumentTypeError("expected NAME=VALUE or NAME:=VALUE")
    name, raw_value = token.split(separator, 1)
    name = name.strip()
    raw_value = raw_value.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError(f"invalid launch argument name: {name!r}")
    if not raw_value:
        raise argparse.ArgumentTypeError("simulator launch argument value is empty")
    return name, raw_value


def _ros_value(value: Any) -> str:
    """Python boolean은 ROS launch 형식의 소문자 문자열로 변환한다."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _pose_matrix(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """XYZ/RPY pose를 homogeneous transform으로 변환한다."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    matrix[:3, 3] = (x, y, z)
    return matrix


def target_entrance_in_base(
    trial: dict[str, Any],
    robot_spawn: tuple[float, float, float, float, float, float] = DEFAULT_ROBOT_SPAWN,
) -> np.ndarray:
    """case YAML transform만으로 target entrance의 base_link XYZ를 계산한다."""
    task = next(iter(trial["tasks"].values()))
    board = trial["scene"]["task_board"]
    pose = board["pose"]
    target = _pose_matrix(
        pose["x"],
        pose["y"],
        pose["z"],
        pose["roll"],
        pose["pitch"],
        pose["yaw"],
    )

    if task["port_type"] == "sfp":
        rail_index = int(task["target_module_name"].removeprefix("nic_card_mount_"))
        rail = board[f"nic_rail_{rail_index}"]["entity_pose"]
        port_index = int(task["port_name"].removeprefix("sfp_port_"))
        port_x = 0.01295 if port_index == 0 else -0.01025
        target = (
            target
            @ _pose_matrix(
                -0.081418 + rail["translation"],
                -0.1745 + 0.04 * rail_index,
                0.012,
                yaw=rail["yaw"],
            )
            @ _pose_matrix(-0.002, -0.01785, 0.0899, roll=-1.57)
            @ _pose_matrix(port_x, -0.031572, 0.00501, roll=4.69895)
            @ _pose_matrix(0.0, 0.0, -0.0458)
        )
    else:
        rail_index = int(task["target_module_name"].removeprefix("sc_port_"))
        rail = board[f"sc_rail_{rail_index}"]["entity_pose"]
        target = (
            target
            @ _pose_matrix(
                -0.075 + rail["translation"],
                0.0295 + 0.041 * rail_index,
                0.0165,
                roll=1.57,
                yaw=1.57 + rail["yaw"],
            )
            @ _pose_matrix(0.0, -0.002, 0.0, roll=1.5708, pitch=3.14159)
            @ _pose_matrix(0.0, 0.0, -0.01564)
        )

    world_from_base = _pose_matrix(*robot_spawn)
    point_base = np.linalg.inv(world_from_base) @ target @ np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=float,
    )
    return point_base[:3]


def target_camera_projections(
    trial: dict[str, Any],
    *,
    robot_spawn: tuple[float, float, float, float, float, float] = DEFAULT_ROBOT_SPAWN,
    margin_px: float = DEFAULT_VISIBILITY_MARGIN_PX,
) -> dict[str, dict[str, float | bool]]:
    """고정 home에서 target entrance의 camera별 pixel/depth/가시성을 계산한다."""
    point_base = np.append(target_entrance_in_base(trial, robot_spawn), 1.0)
    focal = CAMERA_WIDTH / (2.0 * math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))
    projections: dict[str, dict[str, float | bool]] = {}
    for camera, base_from_camera in FIXED_HOME_CAMERA_OPTICAL_IN_BASE.items():
        point_camera = np.linalg.inv(base_from_camera) @ point_base
        depth = float(point_camera[2])
        if depth <= 0.0:
            u_px = v_px = float("nan")
        else:
            u_px = float(focal * point_camera[0] / depth + CAMERA_WIDTH / 2.0)
            v_px = float(focal * point_camera[1] / depth + CAMERA_HEIGHT / 2.0)
        visible = (
            CAMERA_NEAR_M <= depth <= CAMERA_FAR_M
            and margin_px <= u_px < CAMERA_WIDTH - margin_px
            and margin_px <= v_px < CAMERA_HEIGHT - margin_px
        )
        projections[camera] = {
            "visible": bool(visible),
            "u_px": u_px,
            "v_px": v_px,
            "depth_m": depth,
        }
    return projections


def _robot_spawn_from_sim_args(
    sim_args: Sequence[tuple[str, str]],
) -> tuple[float, float, float, float, float, float]:
    """launch의 robot pose override를 visibility 계산에 동일하게 반영한다."""
    names = ("robot_x", "robot_y", "robot_z", "robot_roll", "robot_pitch", "robot_yaw")
    values = dict(zip(names, DEFAULT_ROBOT_SPAWN))
    for name, raw_value in sim_args:
        if name in values:
            values[name] = float(raw_value)
    return tuple(float(values[name]) for name in names)


def _scenario_args(args: argparse.Namespace) -> argparse.Namespace:
    """PortOffset generator가 사용하는 CLI 값 중 scene 관련 값을 전달한다."""
    return argparse.Namespace(
        port_types=args.port_types,
        port_order=args.port_order,
        time_limit_s=args.time_limit_s,
        robot_joint_noise_deg=args.robot_joint_noise_deg,
        cable_rpy_noise_deg=args.cable_rpy_noise_deg,
    )


def generate_cases(
    *,
    seed: int,
    num_cases: int,
    scenario_args: argparse.Namespace,
    min_visible_cameras: int = DEFAULT_MIN_VISIBLE_CAMERAS,
    visibility_margin_px: float = DEFAULT_VISIBILITY_MARGIN_PX,
    robot_spawn: tuple[
        float, float, float, float, float, float
    ] = DEFAULT_ROBOT_SPAWN,
) -> dict[str, Any]:
    """고정 home의 common FOV를 통과한 randomized multi-trial config를 만든다."""
    if num_cases <= 0:
        raise ValueError("num_cases must be greater than zero")
    if min_visible_cameras not in (2, 3):
        raise ValueError("min_visible_cameras must be 2 or 3")
    if visibility_margin_px < 0.0:
        raise ValueError("visibility_margin_px must be non-negative")
    if abs(float(scenario_args.robot_joint_noise_deg)) > 1e-12:
        raise ValueError(
            "common-FOV guarantee requires --robot-joint-noise-deg 0"
        )

    rng = random.Random(seed)
    generated = []
    for index in range(num_cases):
        for _attempt in range(MAX_VISIBILITY_ATTEMPTS_PER_CASE):
            config, metadata = make_trial_config(index, rng, scenario_args)
            trial = next(iter(config["trials"].values()))
            projections = target_camera_projections(
                trial,
                robot_spawn=robot_spawn,
                margin_px=visibility_margin_px,
            )
            if sum(
                bool(projection["visible"])
                for projection in projections.values()
            ) >= min_visible_cameras:
                generated.append((config, metadata))
                break
        else:
            raise RuntimeError(
                "failed to sample a common-FOV case: "
                f"index={index}, attempts={MAX_VISIBILITY_ATTEMPTS_PER_CASE}"
            )
    first_config = generated[0][0]
    trials: dict[str, Any] = {}
    for config, _metadata in generated:
        trials.update(config["trials"])

    # AIC multi-trial schema has one global robot section. The first PortOffset
    # draw is therefore shared by every trial; all board/module/cable values are
    # still sampled independently by the original PortOffset generator.
    return {
        "scoring": first_config["scoring"],
        "task_board_limits": first_config["task_board_limits"],
        "trials": trials,
        "robot": first_config["robot"],
    }


def default_output_path(today: date | None = None) -> Path:
    """날짜 기반 기본 YAML 경로를 반환한다."""
    stamp = (today or date.today()).strftime("%Y%m%d")
    return TRIANGULATION_ROOT / "cases" / f"{stamp}_triangulation_cases.yaml"


def write_cases(
    path: Path,
    config: dict[str, Any],
    *,
    seed: int,
    num_cases: int,
    min_visible_cameras: int = DEFAULT_MIN_VISIBLE_CAMERAS,
    visibility_margin_px: float = DEFAULT_VISIBILITY_MARGIN_PX,
) -> None:
    """재현 정보가 포함된 AIC engine YAML을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by run_triangulation_cases.py\n"
        f"# seed: {seed}\n"
        f"# num_cases: {num_cases}\n"
        "# robot pose: fixed BASE_ROBOT_HOME, shared by the multi-trial engine config\n"
        f"# common FOV: cameras>={min_visible_cameras}, "
        f"margin_px={visibility_margin_px:g}\n"
    )
    path.write_text(
        header + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def simulator_command(args: argparse.Namespace, cases_path: Path) -> list[str]:
    """생성 YAML을 실행할 Distrobox 명령을 구성한다."""
    launch_arguments = [
        ("ground_truth", args.ground_truth),
        ("start_aic_engine", args.start_aic_engine),
        ("gazebo_gui", args.gazebo_gui),
        ("launch_rviz", args.launch_rviz),
        ("spawn_task_board", args.spawn_task_board),
        ("spawn_cable", args.spawn_cable),
    ]
    launch_arguments.extend(args.sim_arg)
    launch_arguments.append(
        ("aic_engine_config_file", str(cases_path.resolve()))
    )
    return [
        "distrobox",
        "enter",
        args.distrobox,
        "--",
        "/entrypoint.sh",
        *(f"{name}:={_ros_value(value)}" for name, value in launch_arguments),
    ]


def run_simulator(args: argparse.Namespace, cases_path: Path) -> None:
    """Docker backend의 Distrobox에서 AIC simulator를 실행한다."""
    command = simulator_command(args, cases_path)
    env = os.environ.copy()
    env["DBX_CONTAINER_MANAGER"] = "docker"
    _bold_log(BOLD_GREEN, f"[simulator start] {shlex.join(command)}")
    subprocess.run(command, env=env, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PortOffset-range triangulation cases and run them in aic_eval."
        )
    )
    parser.add_argument("--seed", type=int, default=CLI_DEFAULTS["seed"])
    parser.add_argument(
        "--num-cases",
        "--num_cases",
        dest="num_cases",
        type=int,
        default=CLI_DEFAULTS["trials"],
        help="number of randomized cases (must be > 0)",
    )
    parser.add_argument("--port-types", default=CLI_DEFAULTS["port_types"])
    parser.add_argument(
        "--port-order",
        choices=("round_robin", "random"),
        default=CLI_DEFAULTS["port_order"],
    )
    parser.add_argument(
        "--time-limit-s",
        type=int,
        default=CLI_DEFAULTS["time_limit_s"],
    )
    parser.add_argument(
        "--robot-joint-noise-deg",
        type=float,
        default=0.0,
        help="must be 0 because common-FOV projection assumes fixed observation pose",
    )
    parser.add_argument(
        "--cable-rpy-noise-deg",
        type=float,
        default=CLI_DEFAULTS["cable_rpy_noise_deg"],
    )
    parser.add_argument(
        "--min-visible-cameras",
        type=int,
        choices=(2, 3),
        default=DEFAULT_MIN_VISIBLE_CAMERAS,
        help="required cameras containing the target entrance at fixed home",
    )
    parser.add_argument(
        "--visibility-margin-px",
        type=float,
        default=DEFAULT_VISIBILITY_MARGIN_PX,
        help="safe image-border margin used by the common-FOV filter",
    )
    parser.add_argument("--distrobox", default=CLI_DEFAULTS["distrobox"])
    parser.add_argument(
        "--headless",
        action="store_true",
        help="legacy shortcut for --gazebo_gui false",
    )
    parser.add_argument(
        "--ground-truth",
        "--ground_truth",
        dest="ground_truth",
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--start-aic-engine",
        "--start_aic_engine",
        dest="start_aic_engine",
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--gazebo-gui",
        "--gazebo_gui",
        dest="gazebo_gui",
        type=_parse_bool,
        default=False,
        metavar="{true, false}",
    )
    parser.add_argument(
        "--launch-rviz",
        "--launch_rviz",
        dest="launch_rviz",
        type=_parse_bool,
        default=False,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--spawn-task-board",
        "--spawn_task_board",
        dest="spawn_task_board",
        type=_parse_bool,
        default=False,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--spawn-cable",
        "--spawn_cable",
        dest="spawn_cable",
        type=_parse_bool,
        default=False,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--sim-arg",
        action="append",
        type=_parse_sim_arg,
        default=[],
        metavar="NAME=VALUE",
        help="additional /entrypoint.sh launch argument; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output YAML (default: cases/YYYYMMDD_triangulation_cases.yaml)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="write YAML without launching Distrobox",
    )
    args = parser.parse_args(argv)
    if args.num_cases <= 0:
        parser.error("--num-cases/--num_cases must be greater than zero")
    if abs(args.robot_joint_noise_deg) > 1e-12:
        parser.error("--robot-joint-noise-deg must be 0 for common-FOV guarantee")
    if args.visibility_margin_px < 0.0:
        parser.error("--visibility-margin-px must be non-negative")
    if args.headless and args.gazebo_gui is True:
        parser.error("--headless conflicts with --gazebo-gui/--gazebo_gui true")
    if args.headless:
        args.gazebo_gui = False
    elif args.gazebo_gui is None:
        args.gazebo_gui = True

    reserved_sim_args = {
        "ground_truth",
        "start_aic_engine",
        "gazebo_gui",
        "launch_rviz",
        "spawn_task_board",
        "spawn_cable",
        "aic_engine_config_file",
    }
    duplicated = sorted(
        name for name, _value in args.sim_arg if name in reserved_sim_args
    )
    if duplicated:
        parser.error(
            "--sim-arg duplicates explicit option(s): " + ", ".join(duplicated)
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = (args.output or default_output_path()).expanduser().resolve()
    config = generate_cases(
        seed=args.seed,
        num_cases=args.num_cases,
        scenario_args=_scenario_args(args),
        min_visible_cameras=args.min_visible_cameras,
        visibility_margin_px=args.visibility_margin_px,
        robot_spawn=_robot_spawn_from_sim_args(args.sim_arg),
    )
    write_cases(
        output,
        config,
        seed=args.seed,
        num_cases=args.num_cases,
        min_visible_cameras=args.min_visible_cameras,
        visibility_margin_px=args.visibility_margin_px,
    )
    _bold_log(
        BOLD_MAGENTA,
        f"[generated] cases={args.num_cases} seed={args.seed} yaml={output}",
    )
    if not args.generate_only:
        run_simulator(args, output)


if __name__ == "__main__":
    main()
