#!/usr/bin/env python3
"""
collect_data.py
───────────────
랜덤 시나리오를 반복 생성하며 AutoCapture로 데이터 수집.

흐름:
  for N 세트:
    1. Trial 1·2·3 각각의 랜덤 파라미터를 담은 aic_engine config YAML 생성
    2. /tmp/aic_custom_config.yaml 로 저장 (distrobox 컨테이너와 /tmp 공유)
    3. distrobox + start_aic_engine:=true + aic_engine_config_file:=... 로 Gazebo 시작
       (spawn_task_board:=false, spawn_cable:=false → 엔진이 YAML에서 직접 스폰)
    4. Gazebo 초기화 대기
    5. aic_model + DataCollect 정책 시작
    6. AIC_CAPTURE_DIR에서 episode_summary.json 수로 완료 감지
    7. 두 프로세스 모두 종료 → 다음 세트

사용법:
  python3 collect_data.py                               # 기본: 10 세트 × 7 에피소드
  python3 collect_data.py --sets 50 --diversify         # 50세트, 보드 위치도 랜덤화
  python3 collect_data.py --sets 5 --dry-run            # 명령어만 출력 (실행 X)
  python3 collect_data.py --gazebo-wait 60              # Gazebo 초기화 대기를 60초로
  python3 collect_data.py --headless                    # Gazebo GUI·RViz 없이 백그라운드 실행
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

LIMITS = {
    "nic_translation":   (-0.0215, 0.0234),
    "nic_yaw":           (-math.radians(10), math.radians(10)),
    "sc_translation":    (-0.06, 0.055),
    "board_yaw_trial12": (0.0, 3.1415),
    "board_yaw_trial3":  (0.0, 3.1415),
    "board_x_trial12":   (0.13, 0.17),
    "board_y_trial12":   (-0.25, -0.15),
    "board_x_trial3":    (0.15, 0.19),
    "board_y_trial3":    (-0.05, 0.05),
    # gripper offset 기준값 및 노이즈
    "gripper_offset_noise": (-0.002, 0.002),
    "nic_gripper_offset_y": 0.015385,
    "nic_gripper_offset_z": 0.04245,
    "sc_gripper_offset_y":  0.015385,
    "sc_gripper_offset_z":  0.04045,
}

def rnd(low: float, high: float) -> float:
    return random.uniform(low, high)

GAZEBO_INIT_WAIT = 60   # Gazebo 초기화 대기 (초) — Zenoh peer 안정화 포함
EPISODE_TIMEOUT  = 1200  # 세트당 최대 대기 시간 (초)

# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[4]  # AIC_Sejong/

PIXI_WS              = ROOT / "ws_aic" / "src" / "aic"
ENGINE_CONFIG_TMP    = Path("/tmp/aic_custom_config.yaml")
SCENARIO_PARAMS_TMP  = Path("/tmp/aic_scenario_params.json")
EPISODE_TRACKING_DIR = Path("/tmp/aic_episodes")

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
        "nic_rail": {"min_translation": LIMITS["nic_translation"][0],
                     "max_translation": LIMITS["nic_translation"][1]},
        "sc_rail":  {"min_translation": LIMITS["sc_translation"][0],
                     "max_translation": LIMITS["sc_translation"][1]},
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


def _nic_rails(active_rail: int, translation: float, yaw: float) -> dict:
    """nic_rail_0 ~ nic_rail_4: active_rail 위치에만 NIC 카드 배치."""
    rails = {}
    for i in range(5):
        if i == active_rail:
            rails[f"nic_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{active_rail}",
                "entity_pose": {
                    "translation": translation,
                    "roll": 0.0, "pitch": 0.0,
                    "yaw": yaw,
                },
            }
        else:
            rails[f"nic_rail_{i}"] = {"entity_present": False}
    return rails


def _sc_rails_nic(translation: float) -> dict:
    """NIC trial용 sc_rail: sc_rail_0에 배경 SC 마운트, sc_rail_1 absent."""
    return {
        "sc_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {
                "translation": translation,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        },
        "sc_rail_1": {"entity_present": False},
    }


def _sc_rails_sc(active_rail: int, translation: float) -> dict:
    """SC trial용 sc_rail: active_rail 위치에 삽입 대상 SC 마운트, 나머지 absent."""
    rails = {}
    for i in range(2):
        if i == active_rail:
            rails[f"sc_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{active_rail}",
                "entity_pose": {
                    "translation": translation,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
            }
        else:
            rails[f"sc_rail_{i}"] = {"entity_present": False}
    return rails


def _mount_rails_nic() -> dict:
    """NIC trial용 mount rail 배치. translation 고정 (평가 대상 아님)."""
    def present(name: str) -> dict:
        return {
            "entity_present": True,
            "entity_name": name,
            "entity_pose": {"translation": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
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
    """SC trial용 mount rail 배치. translation 고정 (평가 대상 아님)."""
    def present(name: str) -> dict:
        return {
            "entity_present": True,
            "entity_name": name,
            "entity_pose": {"translation": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
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


def _make_nic_trial(nic_rail: int, diversify: bool) -> tuple[dict, dict]:
    """NIC trial: nic_rail_N에 NIC 카드 삽입.

    Returns:
        (engine_config, scenario_params) — scenario_params는 LeRobot 프레임에 저장될 랜덤 파라미터.
    """
    board   = _board_pose(1, diversify)
    nic_t   = rnd(*LIMITS["nic_translation"])
    nic_y   = rnd(*LIMITS["nic_yaw"])
    sc_bg_t = rnd(*LIMITS["sc_translation"])

    offset_x = rnd(*LIMITS["gripper_offset_noise"])
    offset_y = LIMITS["nic_gripper_offset_y"] + rnd(*LIMITS["gripper_offset_noise"])
    offset_z = LIMITS["nic_gripper_offset_z"] + rnd(*LIMITS["gripper_offset_noise"])

    task_board = {"pose": board}
    task_board.update(_nic_rails(nic_rail, nic_t, nic_y))
    task_board.update(_sc_rails_nic(sc_bg_t))
    task_board.update(_mount_rails_nic())

    scenario_params = {
        "trial_type":       0,         # 0 = NIC
        "rail_idx":         nic_rail,
        "board_x":          board["x"],
        "board_y":          board["y"],
        "board_yaw":        board["yaw"],
        "gripper_offset_x": offset_x,
        "gripper_offset_y": offset_y,
        "gripper_offset_z": offset_z,
        "nic_translation":  nic_t,
        "nic_yaw":          nic_y,
        "sc_translation":   sc_bg_t,
    }

    config = {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": {"x": offset_x, "y": offset_y, "z": offset_z},
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
    return config, scenario_params


def _make_sc_trial(sc_rail: int, diversify: bool) -> tuple[dict, dict]:
    """SC trial: sc_rail_N에 SC 케이블 삽입.

    Returns:
        (engine_config, scenario_params) — scenario_params는 LeRobot 프레임에 저장될 랜덤 파라미터.
    """
    board  = _board_pose(2, diversify)
    sc_t   = rnd(*LIMITS["sc_translation"])

    offset_x = rnd(*LIMITS["gripper_offset_noise"])
    offset_y = LIMITS["sc_gripper_offset_y"] + rnd(*LIMITS["gripper_offset_noise"])
    offset_z = LIMITS["sc_gripper_offset_z"] + rnd(*LIMITS["gripper_offset_noise"])

    task_board = {"pose": board}
    for i in range(5):
        task_board[f"nic_rail_{i}"] = {"entity_present": False}
    task_board.update(_sc_rails_sc(sc_rail, sc_t))
    task_board.update(_mount_rails_sc())

    scenario_params = {
        "trial_type":       1,         # 1 = SC
        "rail_idx":         sc_rail,
        "board_x":          board["x"],
        "board_y":          board["y"],
        "board_yaw":        board["yaw"],
        "gripper_offset_x": offset_x,
        "gripper_offset_y": offset_y,
        "gripper_offset_z": offset_z,
        "nic_translation":  0.0,       # SC trial에서는 해당 없음
        "nic_yaw":          0.0,
        "sc_translation":   sc_t,
    }

    config = {
        "scene": {
            "task_board": task_board,
            "cables": {
                "cable_1": {
                    "pose": {
                        "gripper_offset": {"x": offset_x, "y": offset_y, "z": offset_z},
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
    return config, scenario_params


def generate_engine_config(diversify: bool = True) -> tuple[dict, dict]:
    """
    sample_config.yaml 구조에 맞는 커스텀 aic_engine config 딕셔너리 생성.

    세트당 7개 trial:
      nic_rail0 ~ nic_rail4 : NIC 카드 삽입 (rail 0~4 순서대로)
      sc_rail0  ~ sc_rail1  : SC 케이블 삽입 (rail 0~1 순서대로)

    Returns:
        (engine_config, all_scenario_params)
        all_scenario_params: task_id → scenario_params dict (LeRobot 기록용)
    """
    trials: dict = {}
    all_scenario_params: dict = {}

    for rail in range(5):
        cfg, params = _make_nic_trial(rail, diversify)
        trials[f"trial_nic{rail}"] = cfg
        all_scenario_params[f"nic_rail{rail}_task_1"] = params

    for rail in range(2):
        cfg, params = _make_sc_trial(rail, diversify)
        trials[f"trial_sc{rail}"] = cfg
        all_scenario_params[f"sc_rail{rail}_task_1"] = params

    config = {
        "scoring": _scoring_section(),
        "task_board_limits": _task_board_limits_section(),
        "trials": trials,
        "robot": _robot_section(),
    }
    return config, all_scenario_params


def save_engine_config(config: dict, path: Path) -> None:
    """config 딕셔너리를 YAML 파일로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)


# ──────────────────────────────────────────
# 에피소드 감시
# ──────────────────────────────────────────

def count_completed_episodes() -> int:
    """episode_summary.json 파일 수로 완료된 에피소드 수를 반환."""
    return len(list(EPISODE_TRACKING_DIR.glob("*/episode_summary.json")))


def wait_for_episodes(
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
        current = count_completed_episodes()
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
    new_eps = count_completed_episodes() - episodes_before
    print(f"[경고] {timeout}초 내에 {target_count}개 에피소드가 완료되지 않았습니다. "
          f"(실제 완료: {new_eps}개)")
    return new_eps


# ──────────────────────────────────────────
# 프로세스 관리
# ──────────────────────────────────────────

def start_gazebo(
    config_path: Path,
    headless: bool = False,
    dry_run: bool = False,
) -> "subprocess.Popen | None":
    """
    distrobox에서 Gazebo + AIC engine 시작.

    - spawn_task_board:=false  → launch 파일의 보드 스폰 비활성
    - spawn_cable:=false       → launch 파일의 케이블 스폰 비활성
    - aic_engine_config_file   → 우리가 생성한 커스텀 YAML 사용
    - headless=True 시 gazebo_gui:=false launch_rviz:=false 추가
      → Gazebo 렌더링 창·RViz 없이 시뮬레이터만 백그라운드 실행
    """
    cmd = [
        "distrobox", "enter", "root", "-r", "aic_eval", "--", "/entrypoint.sh",
        "spawn_task_board:=false",
        "spawn_cable:=false",
        "ground_truth:=true",
        "start_aic_engine:=true",
        f"aic_engine_config_file:={config_path}",
    ]
    if headless:
        cmd += ["gazebo_gui:=false", "launch_rviz:=false"]

    if dry_run:
        mode = "헤드리스" if headless else "GUI"
        print(f"[DRY-RUN] Gazebo 명령어 ({mode}):")
        print("  " + " \\\n  ".join(cmd))
        return None

    mode = "헤드리스(GUI 없음)" if headless else "GUI"
    print(f"[Gazebo] distrobox 시작... (config: {config_path}, 모드: {mode})")
    proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT)
    # distrobox 시작 직후 즉시 종료됐으면 에러 출력
    time.sleep(1)
    if proc.poll() is not None:
        print(f"[에러] distrobox 프로세스가 즉시 종료됨 (returncode={proc.returncode}). "
              "터미널 위 출력을 확인하세요.")
    return proc


def start_policy(
    step_hz: float = 10.0,
    dry_run: bool = False,
) -> "subprocess.Popen | None":
    """aic_model + DataCollect 정책 노드 시작."""
    env = os.environ.copy()
    env["AIC_CAPTURE_DIR"]            = str(EPISODE_TRACKING_DIR)
    env["AIC_CAPTURE_STEP_SLEEP_SEC"] = str(1.0 / step_hz)
    env["AIC_SCENARIO_PARAMS_FILE"]   = str(SCENARIO_PARAMS_TMP)

    cmd = (
        f"cd {PIXI_WS} && pixi run ros2 run aic_model aic_model "
        "--ros-args -p policy:=data_gen_node.policy.datacollect"
    )

    if dry_run:
        print("[DRY-RUN] Policy 명령어:")
        print(f"  AIC_CAPTURE_DIR={EPISODE_TRACKING_DIR}")
        print(f"  AIC_CAPTURE_STEP_SLEEP_SEC={env['AIC_CAPTURE_STEP_SLEEP_SEC']}  ({step_hz}Hz)")
        print(f"  {cmd}")
        return None

    print(f"[Policy] DataCollect 시작. {step_hz}Hz")
    return subprocess.Popen(cmd, shell=True, env=env)


def terminate_processes(*procs):
    """
    프로세스 트리 전체를 완전히 정리.

    distrobox enter 프로세스만 죽이면 내부에서 뜬 Gazebo·ROS2 노드들이
    좀비로 남아 다음 세션의 포트·Zenoh·GPU 리소스를 점유한다.
    따라서 Python Popen 종료 후 OS 레벨에서 관련 프로세스를 모두 정리한다.
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

    # 2단계: distrobox 내부에서 실행된 프로세스들 정리
    # entrypoint.sh → ros2 launch → gz sim / rviz2 / aic_engine 등
    # distrobox는 /tmp, /proc 등을 호스트와 공유하므로 호스트에서 pkill 가능
    _GZ_PATTERNS = [
        "gz sim",           # Gazebo 서버
        "gz_server",        # ros_gz_sim 컨테이너 노드
        "gzserver",
        "ruby.*gz",         # Gazebo Ruby 헬퍼
    ]
    _ROS_PATTERNS = [
        "aic_engine",
        "aic_model",
        "aic_adapter",
        "robot_state_publisher",
        "ros_gz_bridge",
        "ros2_control",                  # ros2_control 메인 프로세스
        "ros2.*controller_manager",      # ros2_control controller_manager (경로 포함)
        "ros2.*spawner",                 # ros2_control spawner (경로 포함)
        "rviz2",
        "static_transform_publisher",
        "topic_tools",
    ]
    _ZENOH_PATTERNS = [
        "zenoh",            # Zenoh 라우터/데몬
    ]

    # pkill -9 는 다른 사용자 프로세스도 종료해야 하므로 sudo 필요.
    print("[정리] Gazebo·ROS2·Zenoh 잔존 프로세스 종료 중...")
    for pattern in _GZ_PATTERNS + _ROS_PATTERNS + _ZENOH_PATTERNS:
        subprocess.run(["sudo", "pkill", "-9", "-f", pattern], capture_output=True)

    # 3단계: 소켓·GPU 컨텍스트 반환 대기
    time.sleep(5)

# ──────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────

def run_collection_loop(
    n_sets: int,
    trials_per_set: int,
    diversify: bool,
    gazebo_wait: int,
    step_hz: float,
    headless: bool,
    dry_run: bool,
):
    EPISODE_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    print("=== AIC 데이터 수집 시작 ===")
    print(f"  세트 수        : {n_sets}")
    print(f"  세트당 에피소드: {trials_per_set}")
    print(f"  engine config  : {ENGINE_CONFIG_TMP}")
    print(f"  diversify      : {diversify}")
    print(f"  headless       : {headless}")
    print(f"  dry-run        : {dry_run}")

    if not dry_run:
        # pixi 환경 워밍업
        print("[pixi] 환경 워밍업 중 (최초 1회)...")
        subprocess.run(
            f"cd {PIXI_WS} && pixi run python3 -c 'import data_gen_node'",
            shell=True, check=False,
        )
        print("[pixi] 워밍업 완료\n")

    total_collected = 0

    for set_idx in range(1, n_sets + 1):
        print(f"\n{'='*60}")
        print(f"  세트 {set_idx} / {n_sets}   (누적 에피소드: {total_collected})")
        print(f"{'='*60}")

        # 1. aic_engine config YAML 생성
        config, all_scenario_params = generate_engine_config(diversify=diversify)

        # 2. YAML → /tmp (distrobox와 공유)
        if not dry_run:
            save_engine_config(config, ENGINE_CONFIG_TMP)
            print(f"[저장] engine config → {ENGINE_CONFIG_TMP}")

        # 3. scenario_params JSON 저장 (/tmp — policy가 task별 랜덤 파라미터 읽기용)
        SCENARIO_PARAMS_TMP.write_text(
            json.dumps(all_scenario_params, ensure_ascii=False), encoding="utf-8"
        )

        # 4. 에피소드 기준점 기록
        episodes_before = count_completed_episodes()

        # 6. Gazebo 먼저 시작, 이후 policy 시작
        GAZEBO_HEAD_START = 20  # Gazebo/aic_engine 선행 기동 시간

        gazebo_proc = start_gazebo(ENGINE_CONFIG_TMP, headless=headless, dry_run=dry_run)
        if not dry_run:
            print(f"[대기] Gazebo 초기화 대기 중... ({GAZEBO_HEAD_START}초)")
            time.sleep(GAZEBO_HEAD_START)

        policy_proc = start_policy(step_hz=step_hz, dry_run=dry_run)
        if not dry_run:
            remaining = max(0, gazebo_wait - GAZEBO_HEAD_START)
            print(f"[대기] aic_model 안정화 대기 중... ({remaining}초 추가)")
            time.sleep(remaining)

        if dry_run:
            print("[DRY-RUN] 실제 실행 없이 종료.")
            continue

        # 7. 에피소드 완료 대기
        try:
            completed = wait_for_episodes(
                episodes_before, trials_per_set,
                timeout=EPISODE_TIMEOUT,
            )
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C 감지. 프로세스 종료 중...")
            terminate_processes(policy_proc, gazebo_proc)
            sys.exit(0)

        total_collected += completed
        print(f"[완료] 세트 {set_idx}: {completed}개 에피소드 수집 (누적: {total_collected})")

        # 8. 프로세스 종료
        terminate_processes(policy_proc, gazebo_proc)


        # 다음 세트 전 짧은 대기
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
        description="AIC 자동 데이터 수집 루프",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        예시:
        python3 collect_data.py                               # 10 세트 × 7 에피소드
        python3 collect_data.py --sets 50 --diversify         # 50세트, 보드 위치도 랜덤화
        python3 collect_data.py --sets 5 --dry-run            # 명령어만 출력
        python3 collect_data.py --gazebo-wait 60              # Gazebo 초기화 대기 60초
        python3 collect_data.py --headless                    # Gazebo & RViz GUI 없이 백그라운드 실행
""",
    )
    parser.add_argument("--sets",             type=int,  default=10,
                        help="수집할 세트 수 (기본: 10)")
    parser.add_argument("--episodes-per-set", type=int,  default=7,
                        help="세트당 에피소드 수 / 완료 감지 기준 (기본: 7 = NIC×5 + SC×2)")
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

    args = parser.parse_args()

    run_collection_loop(
        n_sets         = args.sets,
        trials_per_set = args.episodes_per_set,
        diversify      = args.diversify,
        gazebo_wait    = args.gazebo_wait,
        step_hz        = args.step_hz,
        headless       = args.headless,
        dry_run        = args.dry_run,
    )


if __name__ == "__main__":
    main()
