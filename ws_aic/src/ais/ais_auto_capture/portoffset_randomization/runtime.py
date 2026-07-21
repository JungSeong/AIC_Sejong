"""PortOffset policy와 Gazebo launch 실행 및 trial 완료 감시."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

import yaml

from .constants import (
    DATASET_ROOT,
    EPISODE_TRACKING_DIR,
    RUN_MARKER_ENV,
    WS_SRC,
)
from .lifecycle import (
    OwnedProcessGroup,
    terminate_owned_group,
    wait_group_exit,
)


def dataset_dir(args: argparse.Namespace) -> Path:
    """dataset version을 반영한 로컬 저장 디렉터리를 반환한다."""
    version = args.dataset_version.strip()
    return DATASET_ROOT / version if version else DATASET_ROOT


def _set_optional_env(
    env: dict[str, str],
    name: str,
    value: float | None,
) -> None:
    """선택 CLI 값이 제공된 경우에만 policy 환경변수로 전달한다."""
    if value is not None:
        env[name] = str(value)


def write_inputs(
    config: dict,
    scenario_params: dict,
    config_path: Path,
    scenario_params_path: Path,
) -> None:
    """trial별 engine YAML과 추적용 scenario JSON을 고유 경로에 저장한다."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    scenario_params_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_params_path.write_text(
        json.dumps(scenario_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _policy_environment(
    args: argparse.Namespace,
    *,
    scenario_params_path: Path,
    stop_file: Path,
    run_id: str,
) -> dict[str, str]:
    """PortOffsetCollect가 사용할 ROS 2 및 데이터 수집 환경변수를 구성한다."""
    env = os.environ.copy()
    env["AIC_SCENARIO_PARAMS_FILE"] = str(scenario_params_path)
    env["AIC_CAPTURE_DIR"] = str(EPISODE_TRACKING_DIR)
    env["AIC_STOP_FILE"] = str(stop_file)
    env[RUN_MARKER_ENV] = run_id
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
    _set_optional_env(env, "AIC_PORT_COLLECT_DX_MIN_MM", args.dx_min_mm)
    _set_optional_env(env, "AIC_PORT_COLLECT_DX_MAX_MM", args.dx_max_mm)
    _set_optional_env(env, "AIC_PORT_COLLECT_DY_MIN_MM", args.dy_min_mm)
    _set_optional_env(env, "AIC_PORT_COLLECT_DY_MAX_MM", args.dy_max_mm)
    _set_optional_env(env, "AIC_PORT_COLLECT_DZ_MIN_MM", args.dz_min_mm)
    _set_optional_env(env, "AIC_PORT_COLLECT_DZ_MAX_MM", args.dz_max_mm)
    env["AIC_PORT_COLLECT_ROLL_LIMIT_DEG"] = str(args.port_roll_limit_deg)
    env["AIC_PORT_COLLECT_PITCH_LIMIT_DEG"] = str(args.port_pitch_limit_deg)
    env["AIC_PORT_COLLECT_YAW_LIMIT_DEG"] = str(args.port_yaw_limit_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_ROLL_MIN_DEG", args.roll_min_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_ROLL_MAX_DEG", args.roll_max_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_PITCH_MIN_DEG", args.pitch_min_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_PITCH_MAX_DEG", args.pitch_max_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_YAW_MIN_DEG", args.yaw_min_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_YAW_MAX_DEG", args.yaw_max_deg)
    _set_optional_env(env, "AIC_PORT_COLLECT_RPY_NORM_MAX_RAD", args.rpy_norm_max_rad)
    _set_optional_env(
        env,
        "AIC_PORT_ACTUAL_RPY_NORM_MAX_RAD",
        args.actual_rpy_norm_max_rad,
    )

    env["AIC_RPY_MIN_VISIBLE_CAMERAS"] = str(args.min_visible_cameras)
    env["AIC_RPY_VISIBILITY_MARGIN_PX"] = str(args.visibility_margin_px)
    env["AIC_PORT_COLLECT_BASE_Z_OFFSET_M"] = str(args.base_z_offset_mm / 1000.0)
    env["AIC_COLLECT_CAPTURE_SETTLE_SEC"] = str(args.capture_settle_s)
    env["AIC_COLLECT_STABILITY_TIMEOUT_SEC"] = str(args.stability_timeout_s)
    env["AIC_COLLECT_STABLE_SAMPLES"] = str(args.stable_samples)
    env["AIC_COLLECT_STABILITY_POLL_SEC"] = str(args.stability_poll_s)
    env["AIC_COLLECT_LINEAR_SPEED_TOL_MPS"] = str(
        args.linear_speed_tol_mm_s / 1000.0
    )
    env["AIC_COLLECT_ANGULAR_SPEED_TOL_RADPS"] = str(
        math.radians(args.angular_speed_tol_deg_s)
    )
    env["AIC_LEROBOT_REPO_ID"] = ""
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    env["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
    return env


def start_policy(
    args: argparse.Namespace,
    *,
    scenario_params_path: Path,
    stop_file: Path,
    run_id: str,
) -> subprocess.Popen:
    """PortOffsetCollect ROS 2 node를 독립 session/PGID로 실행한다."""
    env = _policy_environment(
        args,
        scenario_params_path=scenario_params_path,
        stop_file=stop_file,
        run_id=run_id,
    )
    try:
        stop_file.unlink()
    except FileNotFoundError:
        pass
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


def stop_policy(
    group: OwnedProcessGroup | None,
    stop_file: Path,
    args: argparse.Namespace,
) -> bool:
    """policy에 정상 stop을 요청하고 timeout 시 소유 PGID만 강제 종료한다."""
    if group is None:
        return True
    try:
        stop_file.write_text("stop\n", encoding="utf-8")
        if wait_group_exit(group.pgid, args.policy_stop_grace_s):
            print(f"[cleanup] policy graceful stop: PGID {group.pgid} 종료 확인")
            return True
        return terminate_owned_group(
            group,
            args,
            graceful_ros_shutdown=False,
        )
    finally:
        try:
            stop_file.unlink()
        except OSError:
            pass


def start_gazebo(
    args: argparse.Namespace,
    config_path: Path,
    world_path: Path | None,
    run_id: str,
) -> subprocess.Popen:
    """AIC Gazebo launch stack을 Distrobox의 독립 session/PGID로 실행한다."""
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
        f"export {RUN_MARKER_ENV}={shlex.quote(run_id)}",
    ]
    inner = " && ".join([*exports, f"/entrypoint.sh {args_str}"])
    cmd = ["distrobox", "enter"]
    if not args.rootless_distrobox:
        cmd.append("-r")
    cmd += [args.distrobox, "--", "bash", "-lc", inner]
    print("[gazebo] " + shlex.join(cmd))
    return subprocess.Popen(cmd, stderr=subprocess.STDOUT, start_new_session=True)


def known_episode_summaries() -> set[Path]:
    """trial 시작 전에 이미 존재하던 episode summary 경로를 수집한다."""
    return set(EPISODE_TRACKING_DIR.glob("*/episode_summary.json"))


def _summary_matches_task(path: Path, task_id: str) -> bool:
    """episode summary가 현재 task ID에 해당하는지 검증한다."""
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
    """현재 task summary를 기다리며 policy와 simulator 조기 종료도 감시한다."""
    deadline = time.monotonic() + max(1.0, timeout_s)
    print(f"[wait] episode summary 대기: task_id={task_id}, timeout={timeout_s:.1f}s")
    while time.monotonic() < deadline:
        for summary_path in known_episode_summaries() - known_summaries:
            if _summary_matches_task(summary_path, task_id):
                print(f"[done] episode summary saved: {summary_path}")
                return True
        failed = [
            proc
            for proc in watch_procs
            if proc is not None and proc.poll() is not None
        ]
        if failed:
            print(
                "[warn] watched process exited before summary: "
                f"returncode={failed[0].returncode}"
            )
            return False
        time.sleep(1.0)
    print(f"[warn] timeout waiting for task summary: {task_id}")
    return False
