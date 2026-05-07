#!/usr/bin/env python3
"""
Run collect_dataset_v2.py without distrobox.

This runner mirrors the source-built ROS2 flow used by collect_data_aarch.py:
  1. start rmw_zenohd if needed
  2. start Gazebo/AIC bringup from ws_aic/install/setup.bash
  3. run collect_dataset_v2.py in the same ROS2/Zenoh environment
"""

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]  # AIC_Sejong/
WS_SRC = ROOT / "ws_aic" / "src"
WS_AIC_SETUP = ROOT / "ws_aic" / "install" / "setup.bash"
COLLECTOR_SCRIPT = Path(__file__).with_name("collect_dataset_v2.py")
ENGINE_CONFIG_TMP = Path("/tmp/aic_yolo_dataset_config.yaml")
EPISODE_TRACKING_DIR = Path("/tmp/aic_yolo_dataset_episodes")
SCENARIO_PARAMS_TMP = Path("/tmp/aic_yolo_dataset_scenario_params.json")
SPAWN_HOLD_STOP_FILE = Path("/tmp/aic_spawn_hold_stop")

LEROBOT_VENV = ROOT / "lerobot_venv"
_venv_site = next((LEROBOT_VENV / "lib").glob("python3.*/site-packages"), None)
LEROBOT_VENV_SITE = _venv_site or (LEROBOT_VENV / "lib" / "python3.12" / "site-packages")

LIMITS = {
    "nic_translation": (-0.0215, 0.0234),
    "nic_yaw": (-0.1745, 0.1745),
    "sc_translation": (-0.06, 0.055),
    "mount_translation": (-0.09625, 0.09625),
    "board_yaw_trial12": (0.0, 3.1415),
    "board_yaw_trial3": (0.0, 3.1415),
    "board_x_trial12": (0.13, 0.17),
    "board_y_trial12": (-0.25, -0.15),
    "board_x_trial3": (0.15, 0.19),
    "board_y_trial3": (-0.05, 0.05),
    "nic_gripper_offset_y": 0.015385,
    "nic_gripper_offset_z": 0.04245,
    "sc_gripper_offset_y": 0.015385,
    "sc_gripper_offset_z": 0.04045,
}


def rnd(low: float, high: float) -> float:
    return random.uniform(low, high)


def _scoring_section() -> dict:
    return {"topics": [
        {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
        {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "latched": True}},
        {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
        {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
        {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
        {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
        {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
        {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
    ]}


def _task_board_limits_section() -> dict:
    return {
        "nic_rail": {"min_translation": LIMITS["nic_translation"][0],
                     "max_translation": LIMITS["nic_translation"][1]},
        "sc_rail": {"min_translation": LIMITS["sc_translation"][0],
                    "max_translation": LIMITS["sc_translation"][1]},
        "mount_rail": {"min_translation": LIMITS["mount_translation"][0],
                       "max_translation": LIMITS["mount_translation"][1]},
    }


def _robot_section() -> dict:
    return {"home_joint_positions": {
        "shoulder_pan_joint": -0.1597,
        "shoulder_lift_joint": -1.3542,
        "elbow_joint": -1.6648,
        "wrist_1_joint": -1.6933,
        "wrist_2_joint": 1.5710,
        "wrist_3_joint": 1.4110,
    }}


def _board_pose(trial_type: str, diversify: bool) -> dict:
    if trial_type == "nic":
        x = rnd(*LIMITS["board_x_trial12"]) if diversify else 0.15
        y = rnd(*LIMITS["board_y_trial12"]) if diversify else -0.2
        yaw = rnd(*LIMITS["board_yaw_trial12"]) if diversify else 3.1415
    else:
        x = rnd(*LIMITS["board_x_trial3"]) if diversify else 0.17
        y = rnd(*LIMITS["board_y_trial3"]) if diversify else 0.0
        yaw = rnd(*LIMITS["board_yaw_trial3"]) if diversify else 3.0
    return {"x": x, "y": y, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": yaw}


def _nic_rails(active_rail: int) -> dict:
    rails = {}
    for i in range(5):
        rails[f"nic_rail_{i}"] = (
            {
                "entity_present": True,
                "entity_name": f"nic_card_{active_rail}",
                "entity_pose": {
                    "translation": rnd(*LIMITS["nic_translation"]),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rnd(*LIMITS["nic_yaw"]),
                },
            }
            if i == active_rail else {"entity_present": False}
        )
    return rails


def _sc_rails_nic() -> dict:
    return {
        "sc_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {"translation": rnd(*LIMITS["sc_translation"]),
                            "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
        "sc_rail_1": {"entity_present": False},
    }


def _sc_rails_sc(active_rail: int) -> dict:
    rails = {}
    for i in range(2):
        rails[f"sc_rail_{i}"] = (
            {
                "entity_present": True,
                "entity_name": f"sc_mount_{active_rail}",
                "entity_pose": {"translation": rnd(*LIMITS["sc_translation"]),
                                "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            }
            if i == active_rail else {"entity_present": False}
        )
    return rails


def _mount_rails_nic() -> dict:
    def present(name: str) -> dict:
        return {"entity_present": True, "entity_name": name,
                "entity_pose": {"translation": rnd(*LIMITS["mount_translation"]),
                                "roll": 0.0, "pitch": 0.0, "yaw": 0.0}}
    absent = {"entity_present": False}
    return {
        "lc_mount_rail_0": present("lc_mount_0"),
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0": present("sc_mount_0"),
        "lc_mount_rail_1": present("lc_mount_1"),
        "sfp_mount_rail_1": absent,
        "sc_mount_rail_1": absent,
    }


def _mount_rails_sc() -> dict:
    def present(name: str) -> dict:
        return {"entity_present": True, "entity_name": name,
                "entity_pose": {"translation": rnd(*LIMITS["mount_translation"]),
                                "roll": 0.0, "pitch": 0.0, "yaw": 0.0}}
    absent = {"entity_present": False}
    return {
        "lc_mount_rail_0": absent,
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0": present("sc_mount_2"),
        "lc_mount_rail_1": present("lc_mount_1"),
        "sfp_mount_rail_1": absent,
        "sc_mount_rail_1": absent,
    }


def make_nic_trial_config(nic_rail: int, diversify: bool) -> dict:
    task_board = {"pose": _board_pose("nic", diversify)}
    task_board.update(_nic_rails(nic_rail))
    task_board.update(_sc_rails_nic())
    task_board.update(_mount_rails_nic())
    return {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "robot": _robot_section(),
        "trials": {
            f"trial_nic{nic_rail}": {
                "scene": {
                    "task_board": task_board,
                    "cables": {"cable_0": {
                        "pose": {"gripper_offset": {"x": 0.0,
                                                    "y": LIMITS["nic_gripper_offset_y"],
                                                    "z": LIMITS["nic_gripper_offset_z"]},
                                 "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
                        "attach_cable_to_gripper": True,
                        "cable_type": "sfp_sc_cable",
                    }},
                },
                "tasks": {f"nic_rail{nic_rail}_task_1": {
                    "cable_type": "sfp_sc", "cable_name": "cable_0",
                    "plug_type": "sfp", "plug_name": "sfp_tip",
                    "port_type": "sfp", "port_name": "sfp_port_0",
                    "target_module_name": f"nic_card_mount_{nic_rail}",
                    "time_limit": 180,
                }},
            }
        },
    }


def make_sc_trial_config(sc_rail: int, diversify: bool) -> dict:
    task_board = {"pose": _board_pose("sc", diversify)}
    for i in range(5):
        task_board[f"nic_rail_{i}"] = {"entity_present": False}
    task_board.update(_sc_rails_sc(sc_rail))
    task_board.update(_mount_rails_sc())
    return {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "robot": _robot_section(),
        "trials": {
            f"trial_sc{sc_rail}": {
                "scene": {
                    "task_board": task_board,
                    "cables": {"cable_1": {
                        "pose": {"gripper_offset": {"x": 0.0,
                                                    "y": LIMITS["sc_gripper_offset_y"],
                                                    "z": LIMITS["sc_gripper_offset_z"]},
                                 "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
                        "attach_cable_to_gripper": True,
                        "cable_type": "sfp_sc_cable_reversed",
                    }},
                },
                "tasks": {f"sc_rail{sc_rail}_task_1": {
                    "cable_type": "sfp_sc", "cable_name": "cable_1",
                    "plug_type": "sc", "plug_name": "sc_tip",
                    "port_type": "sc", "port_name": "sc_port_base",
                    "target_module_name": f"sc_port_{sc_rail}",
                    "time_limit": 180,
                }},
            }
        },
    }


def save_engine_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _ros2_env() -> dict:
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
    env["ZENOH_CONFIG_OVERRIDE"] = (
        "transport/shared_memory/enabled=true;"
        "transport/shared_memory/transport_optimization/pool_size=536870912"
    )

    extra_paths = [
        str(WS_SRC / "ais" / "ais_policy" / "data_gen_node"),
        str(LEROBOT_VENV_SITE),
    ]
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing_py] if existing_py else []))
    return env


def _is_zenoh_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "rmw_zenohd"], capture_output=True)
    return result.returncode == 0


def start_zenoh(dry_run: bool = False):
    cmd = f"source {WS_AIC_SETUP} && ros2 run rmw_zenoh_cpp rmw_zenohd"
    if dry_run:
        print("[DRY-RUN] Zenoh:")
        print(f"  {cmd}")
        return None
    if _is_zenoh_running():
        print("[Zenoh] already running; reusing existing router.")
        return None
    print("[Zenoh] starting rmw_zenohd...")
    return subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", env=_ros2_env(), stderr=subprocess.STDOUT
    )


def start_gazebo(
    headless: bool = False,
    dry_run: bool = False,
    config_path: Path | None = None,
):
    if config_path is None:
        launch_args = [
            "spawn_task_board:=true",
            "nic_card_mount_0_present:=true",
            "sc_port_0_present:=true",
            "spawn_cable:=true",
            "cable_type:=sfp_sc_cable",
            "attach_cable_to_gripper:=true",
            "ground_truth:=true",
            "start_aic_engine:=false",
        ]
    else:
        launch_args = [
            "spawn_task_board:=false",
            "spawn_cable:=false",
            "ground_truth:=true",
            "start_aic_engine:=true",
            f"aic_engine_config_file:={config_path}",
        ]
    if headless:
        launch_args += ["gazebo_gui:=false", "launch_rviz:=false"]

    cmd = (
        f"source {WS_AIC_SETUP} && "
        f"ros2 launch aic_bringup aic_gz_bringup.launch.py "
        + " ".join(launch_args)
    )
    if dry_run:
        print("[DRY-RUN] Gazebo:")
        print(f"  {cmd}")
        return None
    print("[Gazebo] starting source-built bringup...")
    return subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", env=_ros2_env(), stderr=subprocess.STDOUT
    )


def start_aic_model(args, dry_run: bool = False):
    env = _ros2_env()
    
    if getattr(args, "lerobot_out_dir", None) and getattr(args, "lerobot_repo_id", None):
        env["AIC_LEROBOT_OUT_DIR"] = str(Path(args.lerobot_out_dir).resolve())
        env["AIC_LEROBOT_REPO_ID"] = args.lerobot_repo_id
        env["AIC_LEROBOT_VERSION"] = getattr(args, "lerobot_version", "master")
        env["AIC_LEROBOT_FPS"] = "20"
    else:
        env.pop("AIC_LEROBOT_REPO_ID", None)
        env.pop("AIC_LEROBOT_OUT_DIR", None)
        
    env["AIC_LEROBOT_PUSH_TO_HUB"] = "false"
    env["AIC_CAPTURE_DIR"] = str(EPISODE_TRACKING_DIR)
    env["AIC_SCENARIO_PARAMS_FILE"] = str(SCENARIO_PARAMS_TMP)
    env["AIC_CAPTURE_STEP_SLEEP_SEC"] = "0.1"
    env["AIC_SPAWN_HOLD_STOP_FILE"] = str(SPAWN_HOLD_STOP_FILE)
    env["AIC_SPAWN_HOLD_MAX_SEC"] = "3600"
    
    if not SCENARIO_PARAMS_TMP.exists():
        SCENARIO_PARAMS_TMP.write_text("{}", encoding="utf-8")
    if SPAWN_HOLD_STOP_FILE.exists():
        SPAWN_HOLD_STOP_FILE.unlink()

    policy = getattr(args, "data_policy", "SpawnHold")
    cmd = (
        f"source {WS_AIC_SETUP} && "
        f"ros2 run aic_model aic_model "
        f"--ros-args -p policy:=data_gen_node.{policy}"
    )
    if dry_run:
        print("[DRY-RUN] aic_model:")
        print(f"  {cmd}")
        return None

    print(f"[aic_model] starting {policy} policy...")
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", env=env, stderr=subprocess.STDOUT
    )
    time.sleep(1)
    if proc.poll() is not None:
        print(f"[error] aic_model exited early (returncode={proc.returncode}).")
    return proc


def push_dataset_to_hub(
    lerobot_out_dir: Path,
    lerobot_repo_id: str,
    lerobot_version: str = "master",
) -> None:
    dataset_root = Path(lerobot_out_dir).resolve() / lerobot_version
    if not (dataset_root / "meta" / "info.json").exists():
        print(f"[HF Hub] LeRobot dataset missing at {dataset_root}")
        return

    print(f"\n[HF Hub] Pushing standard LeRobot dataset to {lerobot_repo_id} (branch: {lerobot_version})")
    
    inline_script = (
        f"from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"ds = LeRobotDataset.resume(repo_id='{lerobot_repo_id}', root='{str(dataset_root)}')\n"
        f"ds.push_to_hub(branch='{lerobot_version}')\n"
    )
    
    lerobot_python = str(LEROBOT_VENV / "bin" / "python3")
    
    result = subprocess.run(
        [lerobot_python, "-c", inline_script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
        print(f"[HF Hub] Successfully uploaded LeRobot format to https://huggingface.co/datasets/{lerobot_repo_id}/tree/{lerobot_version}")
    else:
        print(f"[HF Hub] Upload failed:\n{result.stderr.strip()}")


def run_collector(
    args,
    dry_run: bool = False,
    episodes: int | None = None,
    stem_prefix: str = "",
):
    collector_args = [
        f"--episodes {episodes if episodes is not None else args.episodes}",
        f"--n_viewpoints {args.n_viewpoints}",
        f"--output {args.output}",
        f"--val_ratio {args.val_ratio}",
        f"--move_settle_s {args.move_settle_s}",
    ]
    if stem_prefix:
        collector_args.append(f"--stem_prefix {stem_prefix}")
    if args.frames_per_viewpoint is not None:
        collector_args.append(f"--frames_per_viewpoint {args.frames_per_viewpoint}")

    cmd = (
        f"source {WS_AIC_SETUP} && "
        f"python3 {COLLECTOR_SCRIPT} "
        + " ".join(collector_args)
    )
    if dry_run:
        print("[DRY-RUN] Collector:")
        print(f"  PYTHONPATH(add)={LEROBOT_VENV_SITE}")
        print(f"  {cmd}")
        return 0

    print("[Collector] starting collect_dataset_v2.py...")
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", env=_ros2_env(), stderr=subprocess.STDOUT
    )
    return proc.wait()


def terminate_processes(*procs, cleanup_gazebo: bool = False, stop_zenoh: bool = False) -> None:
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        print(f"[Stop] terminating PID {proc.pid}...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    patterns = []
    if cleanup_gazebo:
        patterns += [
            "gz sim",
            "gz_server",
            "gzserver",
            "ruby.*gz",
            "robot_state_publisher",
            "ros_gz_bridge",
            "ros2_control_node",
            "controller_manager",
            "component_container",
            "ros2.*spawner",
            "rviz2",
        ]
    if stop_zenoh:
        patterns += ["zenoh", "rmw_zenohd"]
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def iter_scenarios(n_sets: int, diversify: bool):
    for set_idx in range(n_sets):
        for rail in range(5):
            label = f"s{set_idx + 1:03d}_nic{rail}"
            yield label, make_nic_trial_config(rail, diversify)
        for rail in range(2):
            label = f"s{set_idx + 1:03d}_sc{rail}"
            yield label, make_sc_trial_config(rail, diversify)


def run_static_scene(args, zenoh_proc):
    gazebo_proc = None
    policy_proc = None
    try:
        if not args.no_start_gazebo:
            gazebo_proc = start_gazebo(headless=args.headless, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"[Wait] Gazebo initialization: {args.gazebo_wait}s")
                time.sleep(args.gazebo_wait)
                
        if args.data_policy != "SpawnHold":
            policy_proc = start_aic_model(args, dry_run=args.dry_run)
                
        if not args.no_run_collector:
            return run_collector(args, dry_run=args.dry_run)
        else:
            if policy_proc and not args.dry_run:
                try:
                    policy_proc.wait()
                except KeyboardInterrupt:
                    pass
            return 0
    finally:
        if not args.dry_run:
            terminate_processes(
                gazebo_proc,
                policy_proc,
                zenoh_proc,
                cleanup_gazebo=gazebo_proc is not None,
                stop_zenoh=args.stop_zenoh_on_exit,
            )


def run_generated_scenarios(args, zenoh_proc):
    scenarios = list(iter_scenarios(args.scenario_sets, args.diversify))
    if not scenarios:
        print("[Scenarios] no generated scenario requested.")
        return 0

    episodes_per_scenario = args.episodes_per_scenario
    if episodes_per_scenario is None:
        episodes_per_scenario = max(1, args.episodes // len(scenarios))

    print(
        f"[Scenarios] {len(scenarios)} scenarios "
        f"({args.scenario_sets} set(s) x 7), "
        f"{episodes_per_scenario} requested frames per scenario"
    )

    final_ret = 0
    for idx, (label, config) in enumerate(scenarios, start=1):
        config_path = ENGINE_CONFIG_TMP.with_name(f"{ENGINE_CONFIG_TMP.stem}_{label}.yaml")
        save_engine_config(config, config_path)

        print(f"\n=== Scenario {idx}/{len(scenarios)}: {label} ===")
        print(f"  config: {config_path}")

        gazebo_proc = None
        policy_proc = None
        try:
            gazebo_proc = start_gazebo(
                headless=args.headless,
                dry_run=args.dry_run,
                config_path=config_path,
            )
            if not args.dry_run:
                pre_wait = min(args.aic_model_pre_wait, args.gazebo_wait)
                print(f"[Wait] Gazebo infrastructure: {pre_wait}s")
                time.sleep(pre_wait)

            policy_proc = start_aic_model(args, dry_run=args.dry_run)

            if not args.dry_run:
                remaining = max(0, args.gazebo_wait - args.aic_model_pre_wait)
                print(f"[Wait] scene spawn + TF/camera stabilization: {remaining}s")
                time.sleep(remaining)

            if not args.no_run_collector:
                ret = run_collector(
                    args,
                    dry_run=args.dry_run,
                    episodes=episodes_per_scenario,
                    stem_prefix=f"{label}_",
                )
                if ret != 0:
                    print(f"[Scenarios] collector failed for {label} (returncode={ret})")
                    final_ret = ret
                    break
            else:
                if not args.dry_run:
                    # Collect dataset automatically via policy, wait for it
                    try:
                        print("[Scenarios] Waiting for policy to complete...")
                        # Assume data policy completes after some time or is killed by user
                        # You may want to stop it gracefully if it runs forever
                        time.sleep(EPISODE_TIMEOUT if 'EPISODE_TIMEOUT' in globals() else 60)
                    except KeyboardInterrupt:
                        final_ret = 1
                        break
        finally:
            if not args.dry_run:
                terminate_processes(
                    gazebo_proc,
                    policy_proc,
                    cleanup_gazebo=gazebo_proc is not None,
                )
                time.sleep(args.scenario_cooldown)

    if not args.dry_run and args.stop_zenoh_on_exit:
        terminate_processes(zenoh_proc, stop_zenoh=True)
    return final_ret


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO dataset collection without distrobox."
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n_viewpoints", type=int, default=15)
    parser.add_argument("--output", type=str, default="../../data/yolo")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--move_settle_s", type=float, default=2.5)
    parser.add_argument("--frames_per_viewpoint", type=int, default=None)
    parser.add_argument("--gazebo-wait", type=int, default=25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-start-gazebo", action="store_true")
    parser.add_argument("--no-start-zenoh", action="store_true")
    parser.add_argument("--stop-zenoh-on-exit", action="store_true")
    parser.add_argument("--scenario-sets", type=int, default=0,
                        help="Generate diverse NIC/SC scenarios. 0 keeps the original single static scene.")
    parser.add_argument("--episodes-per-scenario", type=int, default=None,
                        help="Frames per generated scenario. Default: episodes / scenario_count.")
    parser.add_argument("--diversify", action="store_true",
                        help="Randomize board position as well as rail/module positions.")
    parser.add_argument("--scenario-cooldown", type=float, default=5.0,
                        help="Delay between generated scenarios after Gazebo cleanup.")
    parser.add_argument("--aic-model-pre-wait", type=int, default=5,
                        help="Seconds to wait after Gazebo launch before starting aic_model in generated scenarios.")
    parser.add_argument("--dry-run", action="store_true")
    
    # Newly added arguments for DataCollect policy and LeRobot HF hub upload
    parser.add_argument("--data-policy", type=str, default="SpawnHold",
                        help="Policy to use for data collection (e.g. DataCollect2). Defaults to SpawnHold.")
    parser.add_argument("--no-run-collector", action="store_true",
                        help="Do not run the collect_dataset_v2.py script (useful if using DataCollect2).")
    parser.add_argument("--lerobot-out-dir", type=str, default=None,
                        help="Output directory for LeRobot dataset format.")
    parser.add_argument("--lerobot-repo-id", type=str, default=None,
                        help="HuggingFace Hub repository ID for LeRobot dataset.")
    parser.add_argument("--lerobot-version", type=str, default="master",
                        help="LeRobot dataset version (branch).")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub at the end.")

    args = parser.parse_args()

    print("=== YOLO / Policy dataset collection (source-built ROS2, no distrobox) ===")
    print(f"  ws setup : {WS_AIC_SETUP}")
    print(f"  collector: {COLLECTOR_SCRIPT if not args.no_run_collector else 'Skipped'}")
    print(f"  policy   : {args.data_policy}")
    print(f"  output   : {Path(args.output).expanduser()}")
    print(f"  scenarios: {'generated' if args.scenario_sets > 0 else 'static'}")

    zenoh_proc = None
    try:
        if not args.no_start_zenoh:
            zenoh_proc = start_zenoh(dry_run=args.dry_run)
            if not args.dry_run:
                time.sleep(3)

        if args.scenario_sets > 0:
            if args.no_start_gazebo:
                raise SystemExit("--scenario-sets requires starting Gazebo from this runner.")
            ret = run_generated_scenarios(args, zenoh_proc)
        else:
            ret = run_static_scene(args, zenoh_proc)
            
        if args.push_to_hub and args.lerobot_out_dir and args.lerobot_repo_id and not args.dry_run:
            push_dataset_to_hub(args.lerobot_out_dir, args.lerobot_repo_id, args.lerobot_version)
            
        if ret != 0:
            raise SystemExit(ret)
    except KeyboardInterrupt:
        print("\n[Interrupted] stopping processes...")


if __name__ == "__main__":
    main()