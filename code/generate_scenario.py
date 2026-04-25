#!/usr/bin/env python3
"""
generate_scenario.py
────────────────────
Trial 1 / 2 / 3 에 맞는 랜덤 파라미터로 Gazebo 월드를 생성하고,
사용된 파라미터 값이 무엇이었는지 JSON으로 저장하는 스크립트.

사용법:
    python3 generate_scenario.py 1              # Trial 1, 명령어만 출력
    python3 generate_scenario.py 2 --run        # Trial 2, 실제 실행
    python3 generate_scenario.py 3 --seed 42    # Trial 3, 시드 고정 (재현용)
    python3 generate_scenario.py 1 --diversify  # 훈련 다양화 모드 (보드 yaw도 랜덤)
    python3 generate_scenario.py 1 --mode pixi  # eval 컨테이너 대신 pixi로 실행
    python3 generate_scenario.py 1 --set nic_card_mount_0_yaw=0.0  # 특정 파라미터 고정

출력:
    - JSON 파일: ~/aic_sejong/aic_data/scenarios/trial{N}_{timestamp}.json
    - SDF 파일:  ~/aic_sejong/aic_data/scenarios/world/trial{N}_{timestamp}.sdf  (Gazebo 실행 후 자동)
    - 콘솔: 실행 명령어 + SDF 수동 복사 명령어 출력

소스 파라미터 범위 근거:
    - task_board_description.md: NIC ±10°, SC 없음, mount ±60°
    - sample_config.yaml: NIC translation [-0.0215, 0.0234], SC [-0.06, 0.055], mount [-0.09425, 0.09425]
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# Gazebo가 월드를 내보내는 고정 경로
WORLD_SDF_SRC = Path("/tmp/aic.sdf")


# ──────────────────────────────────────────
# 파라미터 범위 정의
# ──────────────────────────────────────────

LIMITS = {
    "nic_translation":   (-0.0215, 0.0234),   # NIC 카드 레일 이동 범위 (m)
    "nic_yaw":           (-math.radians(10), math.radians(10)),  # NIC 카드 회전 ±10°
    "sc_translation":    (-0.06, 0.055),       # SC 포트 레일 이동 범위 (m)
    "mount_translation": (-0.09425, 0.09425),  # 픽 위치 마운트 이동 범위 (m)
    "mount_yaw":         (-math.radians(60), math.radians(60)),  # 픽 위치 마운트 회전 ±60°
    # 훈련 다양화 전용 (평가 시에는 고정)
    "board_yaw_trial12": (0.0, 3.1415),        # Trial 1-2 보드 yaw 변동 범위
    "board_yaw_trial3":  (0.0, 3.1415),        # Trial 3 보드 yaw 변동 범위
    "board_x_trial12":   (0.13, 0.17),
    "board_y_trial12":   (-0.25, -0.15),
    "board_x_trial3":    (0.15, 0.19),
    "board_y_trial3":    (-0.05, 0.05),
}


def rnd(low: float, high: float) -> float:
    """균등 분포 랜덤 샘플링."""
    return random.uniform(low, high)


# ──────────────────────────────────────────
# Trial별 파라미터 생성
# ──────────────────────────────────────────

def _base_pick_mounts(trial: int) -> dict:
    """
    Zone 3 & 4 픽 위치 마운트 파라미터.
    Trial 1&2 와 3 의 기본 구성을 sample_config 기준으로 설정하고 translation 랜덤화.
    """
    if trial in (1, 2):
        return {
            # Zone 3 (rail 0)
            "lc_mount_rail_0_present":     "true",
            "lc_mount_rail_0_translation": rnd(*LIMITS["mount_translation"]),
            "lc_mount_rail_0_roll":        0.0,
            "lc_mount_rail_0_pitch":       0.0,
            "lc_mount_rail_0_yaw":         0.0,

            "sfp_mount_rail_0_present":     "true",
            "sfp_mount_rail_0_translation": rnd(*LIMITS["mount_translation"]),
            "sfp_mount_rail_0_roll":        0.0,
            "sfp_mount_rail_0_pitch":       0.0,
            "sfp_mount_rail_0_yaw":         0.0,

            "sc_mount_rail_0_present":     "true",
            "sc_mount_rail_0_translation": rnd(*LIMITS["mount_translation"]),
            "sc_mount_rail_0_roll":        0.0,
            "sc_mount_rail_0_pitch":       0.0,
            "sc_mount_rail_0_yaw":         0.0,

            # Zone 4 (rail 1)
            "lc_mount_rail_1_present":     "true",
            "lc_mount_rail_1_translation": rnd(*LIMITS["mount_translation"]),
            "lc_mount_rail_1_roll":        0.0,
            "lc_mount_rail_1_pitch":       0.0,
            "lc_mount_rail_1_yaw":         0.0,

            "sfp_mount_rail_1_present":    "false",
            "sfp_mount_rail_1_translation": 0.0,
            "sfp_mount_rail_1_roll":        0.0,
            "sfp_mount_rail_1_pitch":       0.0,
            "sfp_mount_rail_1_yaw":         0.0,

            "sc_mount_rail_1_present":     "false",
            "sc_mount_rail_1_translation": 0.0,
            "sc_mount_rail_1_roll":        0.0,
            "sc_mount_rail_1_pitch":       0.0,
            "sc_mount_rail_1_yaw":         0.0,
        }
    else:  # trial 3
        return {
            "lc_mount_rail_0_present":     "false",
            "lc_mount_rail_0_translation": 0.0,
            "lc_mount_rail_0_roll":        0.0,
            "lc_mount_rail_0_pitch":       0.0,
            "lc_mount_rail_0_yaw":         0.0,

            "sfp_mount_rail_0_present":     "true",
            "sfp_mount_rail_0_translation": rnd(*LIMITS["mount_translation"]),
            "sfp_mount_rail_0_roll":        0.0,
            "sfp_mount_rail_0_pitch":       0.0,
            "sfp_mount_rail_0_yaw":         0.0,

            "sc_mount_rail_0_present":     "true",
            "sc_mount_rail_0_translation": rnd(*LIMITS["mount_translation"]),
            "sc_mount_rail_0_roll":        0.0,
            "sc_mount_rail_0_pitch":       0.0,
            "sc_mount_rail_0_yaw":         0.0,

            "lc_mount_rail_1_present":     "true",
            "lc_mount_rail_1_translation": rnd(*LIMITS["mount_translation"]),
            "lc_mount_rail_1_roll":        0.0,
            "lc_mount_rail_1_pitch":       0.0,
            "lc_mount_rail_1_yaw":         0.0,

            "sfp_mount_rail_1_present":    "false",
            "sfp_mount_rail_1_translation": 0.0,
            "sfp_mount_rail_1_roll":        0.0,
            "sfp_mount_rail_1_pitch":       0.0,
            "sfp_mount_rail_1_yaw":         0.0,

            "sc_mount_rail_1_present":     "false",
            "sc_mount_rail_1_translation": 0.0,
            "sc_mount_rail_1_roll":        0.0,
            "sc_mount_rail_1_pitch":       0.0,
            "sc_mount_rail_1_yaw":         0.0,
        }


def _nic_all_absent() -> dict:
    """NIC 카드 마운트 0~4 전부 비활성."""
    result = {}
    for i in range(5):
        result[f"nic_card_mount_{i}_present"]     = "false"
        result[f"nic_card_mount_{i}_translation"] = 0.0
        result[f"nic_card_mount_{i}_roll"]        = 0.0
        result[f"nic_card_mount_{i}_pitch"]       = 0.0
        result[f"nic_card_mount_{i}_yaw"]         = 0.0
    return result


def _sc_ports_absent() -> dict:
    """SC 포트 0~1 전부 비활성."""
    result = {}
    for i in range(2):
        result[f"sc_port_{i}_present"]     = "false"
        result[f"sc_port_{i}_translation"] = 0.0
        result[f"sc_port_{i}_roll"]        = 0.0
        result[f"sc_port_{i}_pitch"]       = 0.0
        result[f"sc_port_{i}_yaw"]         = 0.0
    return result


def generate_params(trial: int, diversify: bool = False) -> dict:
    """
    Trial 번호에 맞는 랜덤 파라미터를 생성.

    Args:
        trial:     1, 2, 3 중 하나
        diversify: True이면 보드 위치/yaw도 범위 내에서 랜덤화 (훈련 다양화용)
    """
    if trial not in (1, 2, 3):
        raise ValueError(f"Trial은 1, 2, 3 중 하나여야 합니다 (입력값: {trial})")

    params = {}

    # ── 1. 태스크 보드 자세 ──────────────────────────
    if trial in (1, 2):
        if diversify:
            params["task_board_x"]     = rnd(*LIMITS["board_x_trial12"])
            params["task_board_y"]     = rnd(*LIMITS["board_y_trial12"])
            params["task_board_yaw"]   = rnd(*LIMITS["board_yaw_trial12"])
        else:
            params["task_board_x"]     = 0.15
            params["task_board_y"]     = -0.2
            params["task_board_yaw"]   = rnd(*LIMITS["board_yaw_trial12"])
        params["task_board_z"]     = 1.14
        params["task_board_roll"]  = 0.0
        params["task_board_pitch"] = 0.0

    else:  # trial 3
        if diversify:
            params["task_board_x"]     = rnd(*LIMITS["board_x_trial3"])
            params["task_board_y"]     = rnd(*LIMITS["board_y_trial3"])
            params["task_board_yaw"]   = rnd(*LIMITS["board_yaw_trial3"])
        else:
            params["task_board_x"]     = 0.17
            params["task_board_y"]     = 0.0
            params["task_board_yaw"]   = rnd(*LIMITS["board_yaw_trial3"])
        params["task_board_z"]     = 1.14
        params["task_board_roll"]  = 0.0
        params["task_board_pitch"] = 0.0

    # ── 2. NIC 카드 마운트 ───────────────────────────
    params.update(_nic_all_absent())

    if trial == 1:
        params["nic_card_mount_0_present"]     = "true"
        params["nic_card_mount_0_translation"] = rnd(*LIMITS["nic_translation"])
        params["nic_card_mount_0_roll"]        = 0.0
        params["nic_card_mount_0_pitch"]       = 0.0
        params["nic_card_mount_0_yaw"]         = rnd(*LIMITS["nic_yaw"])

    elif trial == 2:
        params["nic_card_mount_1_present"]     = "true"
        params["nic_card_mount_1_translation"] = rnd(*LIMITS["nic_translation"])
        params["nic_card_mount_1_roll"]        = 0.0
        params["nic_card_mount_1_pitch"]       = 0.0
        params["nic_card_mount_1_yaw"]         = rnd(*LIMITS["nic_yaw"])

    # ── 3. SC 포트 ──────────────────────────────────
    params.update(_sc_ports_absent())

    if trial in (1, 2):
        params["sc_port_0_present"]     = "true"
        params["sc_port_0_translation"] = rnd(*LIMITS["sc_translation"])
        params["sc_port_0_roll"]        = 0.0
        params["sc_port_0_pitch"]       = 0.0
        params["sc_port_0_yaw"]         = 0.0

    else:  # trial 3 — SC 포트 1이 삽입 대상
        params["sc_port_1_present"]     = "true"
        params["sc_port_1_translation"] = rnd(*LIMITS["sc_translation"])
        params["sc_port_1_roll"]        = 0.0
        params["sc_port_1_pitch"]       = 0.0
        params["sc_port_1_yaw"]         = 0.0

    # ── 4. 픽 위치 마운트 (Zone 3 & 4) ──────────────
    params.update(_base_pick_mounts(trial))

    # ── 5. 케이블 & 공통 파라미터 ────────────────────
    params["cable_type"]              = "sfp_sc_cable" if trial in (1, 2) else "sfp_sc_cable_reversed"
    params["attach_cable_to_gripper"] = "true"
    params["spawn_task_board"]        = "true"
    params["spawn_cable"]             = "true"
    params["ground_truth"]            = "true"
    params["start_aic_engine"]        = "false"

    return params


# ──────────────────────────────────────────
# 명령어 빌드
# ──────────────────────────────────────────

def _fmt_val(v) -> str:
    """파라미터 값을 ROS2 launch 인자 형식 문자열로 변환."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def build_entrypoint_cmd(params: dict) -> str:
    """eval 컨테이너용 명령어: distrobox enter -r aic_eval -- /entrypoint.sh ..."""
    args = "  \\\n  ".join(f"{k}:={_fmt_val(v)}" for k, v in params.items())
    return f"distrobox enter -r aic_eval -- /entrypoint.sh \\\n  {args}"


def build_pixi_cmd(params: dict) -> str:
    """pixi 소스 빌드용 명령어: pixi run ros2 launch aic_bringup ..."""
    args = "  \\\n  ".join(f"{k}:={_fmt_val(v)}" for k, v in params.items())
    return (
        "cd ~/aic_sejong/ws_aic/src/aic && "
        "pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py \\\n  " + args
    )


# ──────────────────────────────────────────
# JSON 저장
# ──────────────────────────────────────────

def save_json(trial: int, params: dict, seed, diversify: bool, out_dir: Path,
              overrides: dict | None = None) -> Path:
    """파라미터를 JSON 파일로 저장하고 경로를 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trial{trial}_{timestamp}.json"
    path = out_dir / filename

    payload = {
        "meta": {
            "trial":     trial,
            "timestamp": timestamp,
            "seed":      seed,
            "diversify": diversify,
            "overrides": overrides or {},
        },
        "params": params,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


# ──────────────────────────────────────────
# SDF 파일 감시 & 저장
# ──────────────────────────────────────────

def watch_and_save_sdf(dest_path: Path, timeout: int = 90, prev_mtime: float = 0.0):
    """
    /tmp/aic.sdf 가 생성되거나 갱신될 때까지 백그라운드에서 대기한 뒤
    dest_path 로 복사한다.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    print(f"[SDF 감시] /tmp/aic.sdf 생성 대기 중... (최대 {timeout}초)", flush=True)

    while time.time() < deadline:
        if WORLD_SDF_SRC.exists():
            cur_mtime = WORLD_SDF_SRC.stat().st_mtime
            if cur_mtime > prev_mtime:
                shutil.copy2(WORLD_SDF_SRC, dest_path)
                # distrobox 컨테이너 내부 경로(/ws_aic/)를 호스트 실제 경로로 치환
                ws_host = str(Path.home() / "aic_sejong" / "ws_aic")
                content = dest_path.read_text()
                content = content.replace("/ws_aic/", f"{ws_host}/")
                dest_path.write_text(content)
                print(f"[SDF 저장] {dest_path}", flush=True)
                return
        time.sleep(1.0)

    print(f"[경고] {timeout}초 내에 /tmp/aic.sdf 가 생성/갱신되지 않았습니다.", flush=True)


def _prev_sdf_mtime() -> float:
    """실행 전 /tmp/aic.sdf 의 mtime 반환 (없으면 0.0)."""
    return WORLD_SDF_SRC.stat().st_mtime if WORLD_SDF_SRC.exists() else 0.0


# ──────────────────────────────────────────
# 실행 (subprocess)
# ──────────────────────────────────────────

def run_command(params: dict, mode: str) -> subprocess.Popen:
    """파라미터로 Gazebo를 실행하고 Popen 객체를 반환."""
    flat_args = [f"{k}:={_fmt_val(v)}" for k, v in params.items()]

    if mode == "eval":
        cmd = ["distrobox", "enter", "-r", "aic_eval", "--", "/entrypoint.sh"] + flat_args
        return subprocess.Popen(cmd)
    else:  # pixi
        ws = Path.home() / "aic_sejong" / "ws_aic" / "src" / "aic"
        cmd = (
            f"cd {ws} && pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py "
            + " ".join(flat_args)
        )
        return subprocess.Popen(cmd, shell=True)


# ──────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AIC Trial 시나리오 랜덤 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        예시:
        python3 generate_scenario.py 1                     # Trial 1, 명령어 출력만
        python3 generate_scenario.py 2 --run               # Trial 2, 실제 실행
        python3 generate_scenario.py 3 --seed 42           # 시드 고정 (재현 가능)
        python3 generate_scenario.py 1 --diversify         # 보드 위치/yaw도 랜덤화
        python3 generate_scenario.py 1 --mode pixi         # pixi 소스 빌드 모드
        python3 generate_scenario.py 1 --output-dir /tmp   # JSON 저장 위치 지정
        python3 generate_scenario.py 1 --set nic_card_mount_0_yaw=0.0 task_board_x=0.15
    """,
    )
    parser.add_argument(
        "trial",
        type=int,
        choices=[1, 2, 3],
        help="평가 Trial 번호 (1=SFP/NIC-rail0, 2=SFP/NIC-rail1, 3=SC)",
    )
    parser.add_argument("--seed",       type=int,  default=None,
                        help="랜덤 시드 (같은 시드면 동일 파라미터 생성, 재현용)")
    parser.add_argument("--diversify",  action="store_true",
                        help="훈련 다양화 모드: 보드 위치·yaw도 범위 내에서 랜덤화")
    parser.add_argument("--run",        action="store_true",
                        help="명령어를 실제로 실행 (기본: 출력만)")
    parser.add_argument("--mode",       choices=["eval", "pixi"], default="eval",
                        help="실행 모드: eval=distrobox eval 컨테이너(기본), pixi=소스 빌드")
    parser.add_argument("--output-dir", type=Path, default=Path("../aic_data/scenarios/"),
                        help="JSON 저장 폴더 (기본: ~/aic_sejong/aic_data/scenarios/)")
    parser.add_argument("--set",        nargs="+", metavar="KEY=VALUE", default=[],
                        help="파라미터 오버라이드 (예: --set nic_card_mount_0_yaw=0.0)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[시드] {args.seed} 고정")

    params = generate_params(args.trial, diversify=args.diversify)

    overrides = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set 형식 오류: '{item}' (올바른 형식: KEY=VALUE)")
        key, val = item.split("=", 1)
        if key not in params:
            print(f"[경고] '{key}'는 생성된 파라미터에 없습니다. 그래도 추가합니다.")
        if key in params and isinstance(params[key], float):
            overrides[key] = float(val)
        else:
            overrides[key] = val
    if overrides:
        params.update(overrides)
        print(f"[오버라이드] {len(overrides)}개 파라미터 고정: {list(overrides.keys())}")

    out_dir = args.output_dir or (Path.home() / "aic_sejong" / "aic_data" / "scenarios")
    json_path = save_json(args.trial, params, args.seed, args.diversify, out_dir, overrides)
    stem = json_path.stem
    sdf_path = out_dir / "world" / f"{stem}.sdf"

    print(f"[저장] JSON  → {json_path}")
    print(f"[예정] SDF   → {sdf_path}\n")

    print("=" * 60)
    print(f"  Trial {args.trial}  |  diversify={args.diversify}")
    print("=" * 60)
    _print_summary(params, args.trial)
    print()

    if args.mode == "eval":
        cmd_str = build_entrypoint_cmd(params)
    else:
        cmd_str = build_pixi_cmd(params)

    print("[명령어]")
    print(cmd_str)
    print()

    print("[SDF 복사 명령어]  ← Gazebo 실행 후 별도 터미널에서 실행")
    print(f"  mkdir -p {sdf_path.parent} && cp /tmp/aic.sdf {sdf_path}")
    print()

    if args.run:
        prev_mtime = _prev_sdf_mtime()
        print("[실행 중...] Ctrl+C 로 종료")

        watcher = threading.Thread(
            target=watch_and_save_sdf,
            args=(sdf_path,),
            kwargs={"timeout": 120, "prev_mtime": prev_mtime},
            daemon=True,
        )
        watcher.start()

        proc = run_command(params, args.mode)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n[종료] 프로세스를 종료했습니다.")

        watcher.join(timeout=5)


def _print_summary(params: dict, trial: int):
    """핵심 파라미터만 보기 좋게 출력."""
    keys_of_interest = [
        "task_board_x", "task_board_y", "task_board_yaw",
        "cable_type",
    ]
    if trial == 1:
        keys_of_interest += ["nic_card_mount_0_translation", "nic_card_mount_0_yaw"]
    elif trial == 2:
        keys_of_interest += ["nic_card_mount_1_translation", "nic_card_mount_1_yaw"]
    else:
        keys_of_interest += ["sc_port_1_translation"]

    for k in keys_of_interest:
        v = params.get(k, "—")
        if isinstance(v, float):
            print(f"  {k:45s} = {v:.6f}")
        else:
            print(f"  {k:45s} = {v}")


if __name__ == "__main__":
    main()
