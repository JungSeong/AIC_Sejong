#!/usr/bin/env python3
"""
collect_data_native.py
──────────────────────
collect_data.py 와 동일하지만, distrobox 대신 소스 빌드된 ROS2 환경을
직접 사용한다 (Ubuntu 24.04 + ROS2 Kilted 소스 빌드 기준).

흐름:
  for N 세트:
    1. Trial 1·2·3 각각의 랜덤 파라미터를 담은 aic_engine config YAML 생성
    2. /tmp/aic_custom_config.yaml 로 저장
    3. Zenoh 라우터(rmw_zenohd) 시작
    4. ros2 launch aic_bringup aic_gz_bringup.launch.py 로 Gazebo 시작
       (spawn_task_board:=false, spawn_cable:=false → 엔진이 YAML에서 직접 스폰)
    5. Gazebo 초기화 대기
    6. aic_model + DataCollect 정책 시작
    7. AIC_CAPTURE_DIR에서 episode_summary.json 수로 완료 감지
    8. 세 프로세스 모두 종료 → 다음 세트

사용법:
  python3 collect_data_native.py                               # 기본: 10 세트 × 7 에피소드
  python3 collect_data_native.py --sets 50 --diversify         # 50세트, 보드 위치도 랜덤화
  python3 collect_data_native.py --sets 5 --dry-run            # 명령어만 출력 (실행 X)
  python3 collect_data_native.py --seed 42                     # 재현 가능한 시드 (단일)
  python3 collect_data_native.py --seed-start 0   --seed-end 99   # 협업 A: 시드 0~99 (100 세트)
  python3 collect_data_native.py --seed-start 100 --seed-end 199  # 협업 B: 시드 100~199 (100 세트)
  python3 collect_data_native.py --gazebo-wait 60              # Gazebo 초기화 대기를 60초로
  python3 collect_data_native.py --headless                    # Gazebo GUI·RViz 없이 백그라운드 실행

협업 모드 (--seed-start / --seed-end):
  작업자마다 겹치지 않는 시드 범위를 할당해 독립적으로 데이터를 수집할 수 있습니다.
  세트 수는 (seed-end - seed-start + 1) 으로 자동 결정되며, --sets 는 무시됩니다.
  세트 i(0-based)의 시드 = seed-start + i  →  완전히 결정론적·재현 가능합니다.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# generate_scenario.py 를 라이브러리로 임포트
sys.path.insert(0, str(Path(__file__).parent))
from generate_scenario import LIMITS, rnd

# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────

PIXI_WS = Path.home() / "LLM_TUNE" / "AIC_Sejong" / "ws_aic" / "src" / "aic"
DEFAULT_CAPTURE_DIR = Path.home() / "LLM_TUNE" / "AIC_Sejong" / "aic_data" / "captures"
DEFAULT_SCENARIO_DIR = Path.home() / "LLM_TUNE" / "AIC_Sejong" / "aic_data" / "scenarios"
ENGINE_CONFIG_TMP = Path("/tmp/aic_custom_config.yaml")

# 소스 빌드된 ROS2 워크스페이스
WS_AIC_SETUP = Path.home() / "LLM_TUNE" / "AIC_Sejong" / "ws_aic" / "install" / "setup.bash"

GAZEBO_INIT_WAIT = 60   # Gazebo 초기화 대기 (초) — Zenoh peer 안정화 포함
EPISODE_TIMEOUT  = 1200  # 세트당 최대 대기 시간 (초)

# ──────────────────────────────────────────
# aic_engine config YAML 생성
# ──────────────────────────────────────────

def _scoring_section() -> dict:
    """scoring.topics: 고정 값 (sample_config.yaml 기준)."""
    return {
        "topics": [
            {"topic": {"name": "/joint_states",
                       "type": "sensor_msgs/msg/JointState"}},
            {"topic": {"name": "/tf",
                       "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/tf_static",
                       "type": "tf2_msgs/msg/TFMessage",
                       "latched": True}},
            {"topic": {"name": "/scoring/tf",
                       "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/aic/gazebo/contacts/off_limit",
                       "type": "ros_gz_interfaces/msg/Contacts"}},
            {"topic": {"name": "/fts_broadcaster/wrench",
                       "type": "geometry_msgs/msg/WrenchStamped"}},
            {"topic": {"name": "/aic_controller/joint_commands",
                       "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
            {"topic": {"name": "/aic_controller/pose_commands",
                       "type": "aic_control_interfaces/msg/MotionUpdate"}},
            {"topic": {"name": "/scoring/insertion_event",
                       "type": "std_msgs/msg/String"}},
            {"topic": {"name": "/aic_controller/controller_state",
                       "type": "aic_control_interfaces/msg/ControllerState"}},
        ]
    }


def _task_board_limits_section() -> dict:
    """task_board_limits: LIMITS와 동일하게 고정."""
    return {
        "nic_rail":   {"min_translation": LIMITS["nic_translation"][0],
                       "max_translation": LIMITS["nic_translation"][1]},
        "sc_rail":    {"min_translation": LIMITS["sc_translation"][0],
                       "max_translation": LIMITS["sc_translation"][1]},
        "mount_rail": {"min_translation": LIMITS["mount_translation"][0],
                       "max_translation": LIMITS["mount_translation"][1]},
    }


def _robot_section() -> dict:
    """robot.home_joint_positions: 고정 값."""
    return {
        "home_joint_positions": {
            "shoulder_pan_joint":  -0.1597,
            "shoulder_lift_joint": -1.3542,
            "elbow_joint":         -1.6648,
            "wrist_1_joint":       -1.6933,
            "wrist_2_joint":        1.5710,
            "wrist_3_joint":        1.4110,
        }
    }


def _board_pose(trial: int, diversify: bool) -> dict:
    """task_board pose 랜덤 생성.

    trial 1 = SFP/NIC (구 trial_1·2 통합)
    trial 2 = SC       (구 trial_3)

    yaw 값은 모두 동일하게 0~360 diversify
    """
    if trial == 1:
        x   = rnd(*LIMITS["board_x_trial12"]) if diversify else 0.15
        y   = rnd(*LIMITS["board_y_trial12"]) if diversify else -0.2
        yaw = rnd(*LIMITS["board_yaw_trial12"])
    else:
        x   = rnd(*LIMITS["board_x_trial3"]) if diversify else 0.17
        y   = rnd(*LIMITS["board_y_trial3"]) if diversify else 0.0
        yaw = rnd(*LIMITS["board_yaw_trial3"])
    return {"x": x, "y": y, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": yaw}


def _nic_rails(active_rail: int) -> dict:
    """nic_rail_0 ~ nic_rail_4: active_rail 위치에만 NIC 카드 배치."""
    rails = {}
    for i in range(5):
        if i == active_rail:
            rails[f"nic_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{active_rail}",
                "entity_pose": {
                    "translation": rnd(*LIMITS["nic_translation"]),
                    "roll": 0.0, "pitch": 0.0,
                    "yaw": rnd(*LIMITS["nic_yaw"]),
                },
            }
        else:
            rails[f"nic_rail_{i}"] = {"entity_present": False}
    return rails


def _sc_rails_nic() -> dict:
    """NIC trial용 sc_rail: sc_rail_0에 배경 SC 마운트, sc_rail_1 absent."""
    return {
        "sc_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {
                "translation": rnd(*LIMITS["sc_translation"]),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        },
        "sc_rail_1": {"entity_present": False},
    }


def _sc_rails_sc(active_rail: int) -> dict:
    """SC trial용 sc_rail: active_rail 위치에 삽입 대상 SC 마운트, 나머지 absent."""
    rails = {}
    for i in range(2):
        if i == active_rail:
            rails[f"sc_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{active_rail}",
                "entity_pose": {
                    "translation": rnd(*LIMITS["sc_translation"]),
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
            }
        else:
            rails[f"sc_rail_{i}"] = {"entity_present": False}
    return rails


def _mount_rails_nic() -> dict:
    """NIC trial용 mount rail 배치."""
    def present(name: str) -> dict:
        return {
            "entity_present": True,
            "entity_name": name,
            "entity_pose": {
                "translation": rnd(*LIMITS["mount_translation"]),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        }
    def absent() -> dict:
        return {"entity_present": False}

    return {
        "lc_mount_rail_0":  present("lc_mount_0"),
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0":  present("sc_mount_0"),
        "lc_mount_rail_1":  present("lc_mount_1"),
        "sfp_mount_rail_1": absent(),
        "sc_mount_rail_1":  absent(),
    }


def _mount_rails_sc() -> dict:
    """SC trial용 mount rail 배치."""
    def present(name: str) -> dict:
        return {
            "entity_present": True,
            "entity_name": name,
            "entity_pose": {
                "translation": rnd(*LIMITS["mount_translation"]),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        }
    def absent() -> dict:
        return {"entity_present": False}

    return {
        "lc_mount_rail_0":  absent(),
        "sfp_mount_rail_0": present("sfp_mount_0"),
        "sc_mount_rail_0":  present("sc_mount_2"),
        "lc_mount_rail_1":  present("lc_mount_1"),
        "sfp_mount_rail_1": absent(),
        "sc_mount_rail_1":  absent(),
    }


def _make_nic_trial(nic_rail: int, diversify: bool) -> dict:
    """NIC trial: nic_rail_N에 NIC 카드 삽입.

    task.id → episode 디렉토리: YYYYMMDD_HHMMSS_nic_rail{N}_task_1
    """
    task_board = {"pose": _board_pose(1, diversify)}
    task_board.update(_nic_rails(nic_rail))
    task_board.update(_sc_rails_nic())
    task_board.update(_mount_rails_nic())

    return {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04245},
                        "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            f"nic_rail{nic_rail}_task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": f"nic_card_mount_{nic_rail}",
                "time_limit": 180,
            }
        },
    }


def _make_sc_trial(sc_rail: int, diversify: bool) -> dict:
    """SC trial: sc_rail_N에 SC 케이블 삽입.

    task.id → episode 디렉토리: YYYYMMDD_HHMMSS_sc_rail{N}_task_1
    """
    task_board = {"pose": _board_pose(2, diversify)}
    # SC trial에서는 NIC rail 전부 absent
    for i in range(5):
        task_board[f"nic_rail_{i}"] = {"entity_present": False}
    task_board.update(_sc_rails_sc(sc_rail))
    task_board.update(_mount_rails_sc())

    return {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_1": {
                    "pose": {
                        "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04045},
                        "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable_reversed",
                }
            },
        },
        "tasks": {
            f"sc_rail{sc_rail}_task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_1",
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{sc_rail}",
                "time_limit": 180,
            }
        },
    }


def generate_engine_config(diversify: bool = True) -> dict:
    """
    sample_config.yaml 구조에 맞는 커스텀 aic_engine config 딕셔너리 생성.

    세트당 7개 trial:
      nic_rail0 ~ nic_rail4 : NIC 카드 삽입 (rail 0~4 순서대로)
      sc_rail0  ~ sc_rail1  : SC 케이블 삽입 (rail 0~1 순서대로)
    """
    trials = {}
    for rail in range(5):
        trials[f"trial_nic{rail}"] = _make_nic_trial(rail, diversify)
    for rail in range(2):
        trials[f"trial_sc{rail}"] = _make_sc_trial(rail, diversify)

    return {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "trials": trials,
        "robot": _robot_section(),
    }


def save_engine_config(config: dict, path: Path) -> None:
    """config 딕셔너리를 YAML 파일로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)


# ──────────────────────────────────────────
# 에피소드 감시
# ──────────────────────────────────────────

def count_completed_episodes(capture_dir: Path) -> int:
    """episode_summary.json 파일 수로 완료된 에피소드 수를 반환."""
    return len(list(capture_dir.glob("*/episode_summary.json")))


def wait_for_episodes(
    capture_dir: Path,
    episodes_before: int,
    target_count: int,
    timeout: int = EPISODE_TIMEOUT,
) -> int:
    """
    target_count 개의 새 에피소드 완료까지 대기.

    Returns:
        실제로 완료된 새 에피소드 수
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = count_completed_episodes(capture_dir)
        new_eps = current - episodes_before
        elapsed = int(time.time() - (deadline - timeout))
        print(
            f"\r[대기] 에피소드 {new_eps}/{target_count} 완료  ({elapsed}s 경과)",
            end="", flush=True,
        )
        if new_eps >= target_count:
            print()
            return new_eps
        time.sleep(5)

    print()
    new_eps = count_completed_episodes(capture_dir) - episodes_before
    print(f"[경고] {timeout}초 내에 {target_count}개 에피소드가 완료되지 않았습니다. "
          f"(실제 완료: {new_eps}개)")
    return new_eps


# ──────────────────────────────────────────
# 프로세스 관리
# ──────────────────────────────────────────

def _ros2_env() -> dict:
    """ROS2 소스 빌드 환경 변수 반환."""
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
    env["ZENOH_CONFIG_OVERRIDE"] = (
        "transport/shared_memory/enabled=true;"
        "transport/shared_memory/transport_optimization/pool_size=536870912"
    )
    return env


def start_zenoh(dry_run: bool = False) -> "subprocess.Popen | None":
    """
    Zenoh 라우터(rmw_zenohd) 시작.

    summary_eval-setup-no-docker 기준 Terminal 1 역할.
    ros2 launch 보다 먼저 실행되어야 한다.
    """
    cmd = f"source {WS_AIC_SETUP} && ros2 run rmw_zenoh_cpp rmw_zenohd"

    if dry_run:
        print("[DRY-RUN] Zenoh 명령어:")
        print(f"  {cmd}")
        return None

    print("[Zenoh] 라우터 시작...")
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        env=_ros2_env(), stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    if proc.poll() is not None:
        print(f"[에러] Zenoh 프로세스가 즉시 종료됨 (returncode={proc.returncode}).")
    return proc


def start_gazebo(
    config_path: Path,
    headless: bool = False,
    dry_run: bool = False,
) -> "subprocess.Popen | None":
    """
    소스 빌드 환경에서 ros2 launch로 Gazebo + AIC engine 시작.

    - spawn_task_board:=false  → launch 파일의 보드 스폰 비활성
    - spawn_cable:=false       → launch 파일의 케이블 스폰 비활성
    - aic_engine_config_file   → 우리가 생성한 커스텀 YAML 사용
    - headless=True 시 gazebo_gui:=false launch_rviz:=false 추가
    """
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
        mode = "헤드리스" if headless else "GUI"
        print(f"[DRY-RUN] Gazebo 명령어 ({mode}):")
        print(f"  {cmd}")
        return None

    mode = "헤드리스(GUI 없음)" if headless else "GUI"
    print(f"[Gazebo] ros2 launch 시작... (config: {config_path}, 모드: {mode})")
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        env=_ros2_env(), stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    if proc.poll() is not None:
        print(f"[에러] Gazebo 프로세스가 즉시 종료됨 (returncode={proc.returncode}). "
              "터미널 위 출력을 확인하세요.")
    return proc


def start_policy(
    capture_dir: Path,
    step_hz: float = 10.0,
    dry_run: bool = False,
) -> "subprocess.Popen | None":
    """
    aic_model + DataCollect 정책 노드 시작.

    DataCollect = AutoCapture + 에피소드 시작 전 F/T 센서 Tare.
    """
    env = os.environ.copy()
    env["AIC_CAPTURE_DIR"]            = str(capture_dir)
    env["AIC_CAPTURE_STEP_SLEEP_SEC"] = str(1.0 / step_hz)

    cmd = (
        f"cd {PIXI_WS} && pixi run ros2 run aic_model aic_model "
        "--ros-args -p policy:=data_gen_policy.policy.perturbcollect"
    )

    if dry_run:
        print("[DRY-RUN] Policy 명령어:")
        print(f"  AIC_CAPTURE_DIR={capture_dir}")
        print(f"  AIC_CAPTURE_STEP_SLEEP_SEC={env['AIC_CAPTURE_STEP_SLEEP_SEC']}  ({step_hz}Hz)")
        print(f"  {cmd}")
        return None

    print(f"[Policy] DataCollect 시작. 저장 경로: {capture_dir} / {step_hz}Hz")
    proc = subprocess.Popen(cmd, shell=True, env=env, stderr=subprocess.STDOUT)
    time.sleep(2)
    if proc.poll() is not None:
        print(f"[에러] aic_model 프로세스가 즉시 종료됨 (returncode={proc.returncode}). "
              "policy 초기화 에러일 수 있습니다.")
    return proc


def terminate_processes(*procs):
    """
    프로세스 트리 전체를 완전히 정리.

    Python Popen 종료 후 OS 레벨에서 관련 프로세스를 모두 정리한다.
    """
    # 1단계: Python이 직접 들고 있는 Popen 종료
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        print(f"[종료] PID {proc.pid} 종료 중...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[강제종료] PID {proc.pid} kill")
            proc.kill()

    # 2단계: ros2 launch에서 파생된 프로세스들 정리
    _GZ_PATTERNS = [
        "gz sim",
        "gz_server",
        "gzserver",
        "ruby.*gz",
    ]
    _ROS_PATTERNS = [
        "aic_engine",
        "aic_model",
        "aic_adapter",
        "robot_state_publisher",
        "ros_gz_bridge",
        "ros2_control",
        "ros2.*controller_manager",
        "ros2.*spawner",
        "rviz2",
        "static_transform_publisher",
        "topic_tools",
    ]
    _ZENOH_PATTERNS = [
        "zenoh",
        "rmw_zenohd",
    ]

    print("[정리] Gazebo·ROS2·Zenoh 잔존 프로세스 종료 중...")
    for pattern in _GZ_PATTERNS + _ROS_PATTERNS + _ZENOH_PATTERNS:
        subprocess.run(["sudo", "pkill", "-9", "-f", pattern], capture_output=True)

    # 3단계: 소켓·GPU 컨텍스트 반환 대기
    time.sleep(5)


# ──────────────────────────────────────────
# HuggingFace Hub 업로드
# ──────────────────────────────────────────

def upload_to_hub(
    capture_dir: Path,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = True,
    path_in_repo: str = "captures",
    num_workers: int = 4,
    new_episode_dirs: list | None = None,
    commit_batch_size: int = 500,
) -> None:
    """
    HuggingFace Hub 데이터셋 레포에 에피소드를 업로드.

    new_episode_dirs 가 주어지면 해당 디렉토리만, 없으면 capture_dir 전체를 업로드.
    커밋당 최대 commit_batch_size 개 에피소드씩 나눠 업로드해 25k 파일 한도를 회피.

    Args:
        capture_dir       : 로컬 에피소드 저장 경로 (e.g. ~/aic_data/captures)
        repo_id           : Hub 레포 ID (e.g. "aic-sejong-team/aic-dataset")
        repo_type         : "dataset" (기본) 또는 "model"
        private           : True 이면 비공개 레포로 생성
        path_in_repo      : Hub 레포 내 저장 경로 prefix
        num_workers       : (미사용, 하위 호환용)
        new_episode_dirs  : 이번 세트에서 새로 생긴 에피소드 디렉토리 목록
        commit_batch_size : 커밋당 최대 에피소드 수 (기본 500)
    """
    if not HF_HUB_AVAILABLE:
        print("[에러] huggingface_hub 패키지가 설치되어 있지 않습니다.")
        print("       pip install huggingface_hub  후 다시 시도하세요.")
        return

    if not capture_dir.exists():
        print(f"[경고] capture_dir 이 존재하지 않습니다: {capture_dir}")
        return

    api = HfApi()

    print(f"[Hub] 레포 확인/생성 중: {repo_id} (type={repo_type}, private={private})")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    # 업로드할 디렉토리 목록 결정
    if new_episode_dirs:
        target_dirs = sorted(new_episode_dirs)
    else:
        target_dirs = sorted(d for d in capture_dir.iterdir() if d.is_dir())

    if not target_dirs:
        print("[Hub] 업로드할 에피소드 디렉토리가 없습니다.")
        return

    # commit_batch_size 단위로 나눠 업로드 (25k 파일 한도 회피)
    total_dirs = len(target_dirs)
    for batch_start in range(0, total_dirs, commit_batch_size):
        batch = target_dirs[batch_start:batch_start + commit_batch_size]
        batch_end = min(batch_start + commit_batch_size, total_dirs)
        print(
            f"[Hub] 업로드 중 ({batch_start + 1}–{batch_end}/{total_dirs}): "
            f"{len(batch)}개 에피소드 → {repo_id}/{path_in_repo}"
        )
        for ep_dir in batch:
            ep_path_in_repo = f"{path_in_repo}/{ep_dir.name}"
            api.upload_folder(
                folder_path=str(ep_dir),
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=ep_path_in_repo,
                ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
            )

    print(f"[Hub] 업로드 완료! https://huggingface.co/datasets/{repo_id}")


# ──────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────

def run_collection_loop(
    n_sets: int,
    trials_per_set: int,
    capture_dir: Path,
    scenario_dir: Path,
    seed: int | None,
    diversify: bool,
    gazebo_wait: int,
    step_hz: float,
    headless: bool,
    dry_run: bool,
    hub_repo_id: str | None = None,
    hub_private: bool = True,
    hub_path_in_repo: str | None = None,
    seed_start: int | None = None,
):
    capture_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    range_mode = seed_start is not None

    if hub_path_in_repo is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        hub_path_in_repo = f"runs/{run_id}"

    print("=== AIC 데이터 수집 시작 (native / 소스 빌드 환경) ===")
    print(f"  세트 수        : {n_sets}")
    print(f"  세트당 에피소드: {trials_per_set}")
    print(f"  저장 경로      : {capture_dir}")
    print(f"  engine config  : {ENGINE_CONFIG_TMP}")
    print(f"  ws setup       : {WS_AIC_SETUP}")
    print(f"  diversify      : {diversify}")
    print(f"  headless       : {headless}")
    print(f"  dry-run        : {dry_run}")
    if range_mode:
        print(f"  시드 범위      : {seed_start} ~ {seed_start + n_sets - 1}  (협업 모드)")
    elif seed is not None:
        print(f"  시드           : {seed}  (단일 고정 시드)")
    if hub_repo_id:
        print(f"  hub 레포       : {hub_repo_id}")
        print(f"  hub 경로       : {hub_path_in_repo}")
        print(f"  hub private    : {hub_private}")

    if not dry_run:
        # 시작 전 잔존 프로세스 정리 (이전 수동 실행 또는 비정상 종료 대비)
        print("[정리] 잔존 ROS2/Zenoh 프로세스 정리 중...")
        terminate_processes()
        print("[정리] 완료\n")

        # pixi 환경 워밍업
        print("[pixi] 환경 워밍업 중 (최초 1회)...")
        subprocess.run(
            f"cd {PIXI_WS} && pixi run python3 -c 'import data_gen_policy'",
            shell=True, check=False,
        )
        print("[pixi] 워밍업 완료\n")

    total_collected = 0

    for set_idx in range(1, n_sets + 1):
        print(f"\n{'='*60}")
        print(f"  세트 {set_idx} / {n_sets}   (누적 에피소드: {total_collected})")
        print(f"{'='*60}")

        if range_mode:
            current_seed = seed_start + (set_idx - 1)
            random.seed(current_seed)
            print(f"  [시드] {current_seed}  (범위 {seed_start}~{seed_start + n_sets - 1} 중 {set_idx}번째)")
        elif seed is not None:
            random.seed(seed + set_idx)

        # 1. aic_engine config YAML 생성
        config = generate_engine_config(diversify=diversify)

        # 2. YAML 저장
        if not dry_run:
            save_engine_config(config, ENGINE_CONFIG_TMP)
            print(f"[저장] engine config → {ENGINE_CONFIG_TMP}")

        # 3. JSON 저장 (기록용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = scenario_dir / f"set_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "set_idx":    set_idx,
                        "timestamp":  timestamp,
                        "seed":       seed,
                        "diversify":  diversify,
                    },
                    "engine_config": config,
                },
                f, indent=2, ensure_ascii=False,
            )
        print(f"[저장] JSON → {json_path}")

        # 4. 에피소드 기준점 기록
        episodes_before = count_completed_episodes(capture_dir)
        episode_dirs_before = {d for d in capture_dir.iterdir() if d.is_dir()} if capture_dir.exists() else set()

        # 5. Zenoh 라우터 → Gazebo → Policy 순으로 시작
        ZENOH_WAIT      = 3   # Zenoh 안정화 대기
        GAZEBO_HEAD_START = 15  # Gazebo/aic_engine 선행 기동 시간

        zenoh_proc = start_zenoh(dry_run=dry_run)
        if not dry_run:
            print(f"[대기] Zenoh 안정화 대기 중... ({ZENOH_WAIT}초)")
            time.sleep(ZENOH_WAIT)

        gazebo_proc = start_gazebo(ENGINE_CONFIG_TMP, headless=headless, dry_run=dry_run)
        if not dry_run:
            print(f"[대기] Gazebo 초기화 대기 중... ({GAZEBO_HEAD_START}초)")
            time.sleep(GAZEBO_HEAD_START)

        policy_proc = start_policy(capture_dir, step_hz=step_hz, dry_run=dry_run)
        if not dry_run:
            remaining = max(0, gazebo_wait - GAZEBO_HEAD_START)
            print(f"[대기] aic_model 안정화 대기 중... ({remaining}초 추가)")
            time.sleep(remaining)

        if dry_run:
            print("[DRY-RUN] 실제 실행 없이 종료.")
            continue

        # 6. 에피소드 완료 대기
        try:
            completed = wait_for_episodes(
                capture_dir, episodes_before, trials_per_set,
                timeout=EPISODE_TIMEOUT,
            )
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C 감지. 프로세스 종료 중...")
            terminate_processes(policy_proc, gazebo_proc, zenoh_proc)
            print(f"[중단] 총 수집 에피소드: {total_collected}")
            sys.exit(0)

        total_collected += completed
        print(f"[완료] 세트 {set_idx}: {completed}개 에피소드 수집 (누적: {total_collected})")

        # 7. 프로세스 종료
        terminate_processes(policy_proc, gazebo_proc, zenoh_proc)

        # 8. 세트 완료 후 즉시 HuggingFace Hub 업로드 (이번 세트 신규 에피소드만)
        if hub_repo_id and not dry_run:
            set_path_in_repo = f"{hub_path_in_repo}/set_{set_idx:04d}"
            new_dirs = sorted({d for d in capture_dir.iterdir() if d.is_dir()} - episode_dirs_before)
            print(f"\n[Hub] 세트 {set_idx} 업로드 시작 ({len(new_dirs)}개 에피소드) → {hub_repo_id}/{set_path_in_repo}")
            upload_to_hub(
                capture_dir=capture_dir,
                repo_id=hub_repo_id,
                private=hub_private,
                path_in_repo=set_path_in_repo,
                new_episode_dirs=new_dirs,
            )
            print(f"[Hub] 세트 {set_idx} 업로드 완료. 다음 세트로 진행합니다.")
        elif hub_repo_id and dry_run:
            print(f"[DRY-RUN] 세트 {set_idx} Hub 업로드 건너뜀 (repo_id={hub_repo_id})")

        if set_idx < n_sets:
            print("[대기] 다음 세트 준비 중... (5초)")
            time.sleep(5)

    print(f"\n{'='*60}")
    print(f"=== 수집 완료: {n_sets} 세트, 총 {total_collected} 에피소드 ===")
    print(f"{'='*60}")


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AIC 자동 데이터 수집 루프 (native / 소스 빌드 환경)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        예시:
        python3 collect_data_native.py                               # 10 세트 × 7 에피소드
        python3 collect_data_native.py --sets 50 --diversify         # 50세트, 보드 위치도 랜덤화
        python3 collect_data_native.py --sets 5 --dry-run            # 명령어만 출력
        python3 collect_data_native.py --seed 42                     # 단일 시드 고정 (재현용)
        python3 collect_data_native.py --gazebo-wait 60              # Gazebo 초기화 대기 60초
        python3 collect_data_native.py --headless                    # Gazebo & RViz GUI 없이 백그라운드 실행
        python3 collect_data_native.py --hub-repo-id                 # 수집한 데이터 세트를 올릴 리포지토리 선택

        협업 모드 (시드 범위 분할):
        python3 collect_data_native.py --seed-start 0   --seed-end 99   # A: 시드 0~99   (100 세트)
        python3 collect_data_native.py --seed-start 100 --seed-end 199  # B: 시드 100~199 (100 세트)
        python3 collect_data_native.py --seed-start 200 --seed-end 299  # C: 시드 200~299 (100 세트)
""",
    )
    parser.add_argument("--sets",             type=int,  default=10,
                        help="수집할 세트 수 (기본: 10)")
    parser.add_argument("--episodes-per-set", type=int,  default=7,
                        help="세트당 에피소드 수 / 완료 감지 기준 (기본: 7 = NIC×5 + SC×2)")
    parser.add_argument("--capture-dir",      type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="에피소드 저장 경로")
    parser.add_argument("--scenario-dir",     type=Path, default=DEFAULT_SCENARIO_DIR,
                        help="JSON 시나리오 저장 경로")
    parser.add_argument("--seed",             type=int,  default=None,
                        help="랜덤 시드 (단일 고정, 재현용). --seed-start 와 동시에 사용 불가")
    parser.add_argument("--seed-start",      type=int,  default=None,
                        help="협업 모드 시드 범위 시작값 (예: A=0, B=100). "
                             "--seed-end 와 함께 사용하면 세트 수가 자동으로 결정됩니다")
    parser.add_argument("--seed-end",        type=int,  default=None,
                        help="협업 모드 시드 범위 끝값 (포함). "
                             "세트 수 = seed-end - seed-start + 1 으로 자동 결정 (--sets 무시)")
    parser.add_argument("--diversify",        action="store_true",
                        help="보드 위치/yaw도 범위 내에서 랜덤화")
    parser.add_argument("--gazebo-wait",      type=int,  default=GAZEBO_INIT_WAIT,
                        help=f"Gazebo 초기화 대기 시간(초, 기본: {GAZEBO_INIT_WAIT})")
    parser.add_argument("--step-hz",          type=float, default=10.0,
                        help="스텝 샘플링 주파수 Hz (기본: 10Hz)")
    parser.add_argument("--headless",          action="store_true",
                        help="Gazebo GUI·RViz 없이 백그라운드 실행 (gazebo_gui:=false launch_rviz:=false)")
    parser.add_argument("--dry-run",          action="store_true",
                        help="명령어만 출력하고 실제 실행하지 않음")
    parser.add_argument("--hub-repo-id",      type=str,  default="aic-sejong-team/aic-dataset",
                        help="수집 완료 후 업로드할 HuggingFace Hub 레포 ID "
                             "(예: 'aic-sejong-team/aic-dataset'). "
                             "지정하지 않으면 업로드하지 않음")
    parser.add_argument("--hub-no-private",      action="store_true",
                        help="Hub 레포를 공개로 생성 (기본: 비공개)")
    parser.add_argument("--hub-path-in-repo",  type=str,  default=None,
                        help="Hub 레포 내 저장 경로 prefix."
                             " 미지정 시 'runs/YYYYMMDD_HHMMSS' 형태로 자동 생성.")

    args = parser.parse_args()

    # ── 협업 범위 모드 검증 ──────────────────────────────────────────
    seed_start = None
    n_sets = args.sets

    if args.seed_start is not None or args.seed_end is not None:
        if args.seed is not None:
            parser.error("--seed 와 --seed-start/--seed-end 는 동시에 사용할 수 없습니다")
        if args.seed_start is None or args.seed_end is None:
            parser.error("--seed-start 와 --seed-end 는 반드시 함께 지정해야 합니다")
        if args.seed_end < args.seed_start:
            parser.error(f"--seed-end ({args.seed_end}) 는 --seed-start ({args.seed_start}) 이상이어야 합니다")
        seed_start = args.seed_start
        n_sets = args.seed_end - args.seed_start + 1

    run_collection_loop(
        n_sets           = n_sets,
        trials_per_set   = args.episodes_per_set,
        capture_dir      = args.capture_dir,
        scenario_dir     = args.scenario_dir,
        seed             = args.seed,
        diversify        = args.diversify,
        gazebo_wait      = args.gazebo_wait,
        step_hz          = args.step_hz,
        headless         = args.headless,
        dry_run          = args.dry_run,
        hub_repo_id      = args.hub_repo_id,
        hub_private      = not args.hub_no_private,
        hub_path_in_repo = args.hub_path_in_repo,
        seed_start       = seed_start,
    )


if __name__ == "__main__":
    main()
