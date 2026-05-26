from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

from ..core.config import SRC_ROOT


TRAINING_POLICY = (
    "ais_ground_truth_guided_residual_visual_servoing.SfpGrvsTrainingPolicy"
)
EVAL_POLICY = "ais_ground_truth_guided_residual_visual_servoing.SfpGrvsEvalPolicy"


_ROS_LINE = re.compile(r"\[(INFO|WARN|ERROR|FATAL|DEBUG)\]\s+\[[^\]]+\]\s+\[([^\]]*)\]:\s*(.*)")
_NOISY_PATTERNS = (
    "TF_OLD_DATA ignoring data from the past",
    "SubscriberCallback triggered over",
    "failed to initialize wait set: the given context is not valid",
    "insert_cable execute loop",
    "Vision: 검출 시작",
    "검출 결과 0개",
    "Vision: 2대 이상의 카메라에서 검출 실패",
    "Checking participant model readiness",
    "Found 1 node(s) with name",
    "Service '/aic_model/get_state' is available",
    "Lifecycle node 'aic_model'",
    "Configuring lifecycle node",
    "Transitioning model node",
    "Successfully transition model node",
    "Checking required endpoints",
    "All required endpoints are available",
    "Spawning task board",
    "Task board spawned successfully",
    "Spawning cable",
    "Cable cable_0 spawned successfully",
    "Waiting for robot arm to stabilize",
    "Started recording to",
    "on_configure",
    "on_activate",
    "Policy.__init__",
    "Instantiating policy",
    "Setting cartesian mode",
    "Successfully set target mode",
    "Goal accepted",
    "Entering insert_cable_execute_callback",
)
_IMPORTANT_PATTERNS = (
    "Total Trials",
    "Trial ",
    "Model Ready",
    "Simulator Ready",
    "Scoring Ready",
    "Sending InsertCable goal",
    "TrialState:",
    "SfpGrvsTrainingPolicy start",
    "SfpGrvsTrainingPolicy done",
    "SfpGrvsEvalPolicy",
    "DebugSfpDistancePolicy ready",
    "SfpGrvsTrainingPolicy ready",
    "YOLO 모델 파일 없음",
    "YOLO 모델 로드",
    "Distance model loaded",
    "[stage 1/5] initial_lift start",
    "[stage 1/5] initial_lift done",
    "[stage 3/5] approach start",
    "[stage 3/5] approach done",
    "[grvs stage 2/6] detect start",
    "[grvs stage 2/6] detect done",
    "[grvs stage 3/6] approach start",
    "[grvs stage 3/6] approach done",
    "[grvs stage 4/6] rotation start",
    "[grvs stage 4/6] rotation done",
    "[grvs stage 5/6] align start",
    "[grvs stage 5/6] align done",
    "[grvs stage 6/6] insert start",
    "[grvs stage 6/6] insert done",
    "GRVS YOLO",
    "GRVS align",
    "grvs_align",
    "GRVS force baseline",
    "failed",
    "Failed",
    "timeout",
    "timed out",
    "Score:",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _log_filter_enabled() -> bool:
    return _env_bool("AIC_GRVS_LOG_FILTER", True) and not _env_bool(
        "AIC_GRVS_LOG_VERBOSE",
        False,
    )


def _compact_line(line: str) -> str:
    text = line.rstrip()
    match = _ROS_LINE.search(text)
    if match is None:
        return text
    level, node, message = match.groups()
    return f"[{level}] [{node}] {message}"


def _should_print_line(line: str) -> bool:
    if not line.strip():
        return False
    if any(pattern in line for pattern in _NOISY_PATTERNS):
        return False
    if "[ERROR]" in line or "[FATAL]" in line:
        return True
    if "[WARN]" in line:
        return True
    return any(pattern in line for pattern in _IMPORTANT_PATTERNS)


def _print_filtered_line(line: str) -> None:
    if _should_print_line(line):
        print(_compact_line(line), flush=True)


def _drain_output(proc: subprocess.Popen) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        _print_filtered_line(line)


def _attach_filter_thread(proc: subprocess.Popen) -> None:
    thread = threading.Thread(target=_drain_output, args=(proc,), daemon=True)
    thread.start()
    setattr(proc, "_grvs_log_thread", thread)


def _send_process_signal(proc: subprocess.Popen, sig: signal.Signals) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        proc.send_signal(sig)


def _terminate_process(
    proc: subprocess.Popen,
    *,
    interrupt_timeout_s: float,
    terminate_timeout_s: float = 3.0,
    kill_timeout_s: float = 2.0,
) -> None:
    if proc.poll() is not None:
        return
    _send_process_signal(proc, signal.SIGINT)
    try:
        proc.wait(timeout=interrupt_timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        _send_process_signal(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            _send_process_signal(proc, signal.SIGKILL)
            proc.wait(timeout=kill_timeout_s)
        raise SystemExit(130) from None

    _send_process_signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=terminate_timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    _send_process_signal(proc, signal.SIGKILL)
    proc.wait(timeout=kill_timeout_s)


def _with_common_env(
    *,
    domain_id: int | None,
    extra: dict[str, str | None],
) -> dict[str, str]:
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    env["ZENOH_CONFIG_OVERRIDE"] = env.get(
        "ZENOH_CONFIG_OVERRIDE",
        "transport/shared_memory/enabled=false",
    )
    if domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(domain_id)
    for key, value in extra.items():
        if value is not None:
            env[key] = str(value)
    return env


def start_policy(
    *,
    policy_module: str,
    domain_id: int | None,
    batch_id: str,
    phase: str,
    yolo_model: str | None,
    distance_model: str | None,
    dry_run: bool,
) -> subprocess.Popen | None:
    env = _with_common_env(
        domain_id=domain_id,
        extra={
            "AIC_GRVS_BATCH_ID": batch_id,
            "AIC_GRVS_PHASE": phase,
            "AIC_YOLO_MODEL_PATH": yolo_model,
            "AIC_DISTANCE_MODEL_PATH": distance_model,
        },
    )
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
        f"policy:={policy_module}",
    ]
    if dry_run:
        print("[dry-run] policy:")
        print("  " + shlex.join(cmd))
        return None
    if not _log_filter_enabled():
        return subprocess.Popen(cmd, cwd=SRC_ROOT, env=env, start_new_session=True)
    proc = subprocess.Popen(
        cmd,
        cwd=SRC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    _attach_filter_thread(proc)
    return proc


def run_engine(
    *,
    config_path: Path,
    domain_id: int | None,
    distrobox: str,
    engine_setup: str,
    dry_run: bool,
) -> int:
    exports = [
        "export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}",
        "export ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}",
    ]
    if domain_id is not None:
        exports.append(f"export ROS_DOMAIN_ID={int(domain_id)}")
    inner = " && ".join(
        [
            f"source {shlex.quote(engine_setup)}",
            *exports,
            "ros2 run aic_engine aic_engine "
            "--ros-args "
            f"-p config_file_path:={shlex.quote(str(config_path))} "
            "-p ground_truth:=true "
            "-p use_sim_time:=true",
        ]
    )
    cmd = ["distrobox", "enter", "-r", distrobox, "--", "bash", "-lc", inner]
    if dry_run:
        print("[dry-run] engine:")
        print("  " + shlex.join(cmd))
        return 0
    # Keep stdin/stdout/stderr and the controlling TTY attached. Rootful
    # distrobox enters through sudo/podman and can require an interactive TTY.
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
        raise SystemExit(130) from None


def stop_policy(proc: subprocess.Popen | None, *, timeout_s: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    _terminate_process(proc, interrupt_timeout_s=timeout_s)
    thread = getattr(proc, "_grvs_log_thread", None)
    if thread is not None:
        thread.join(timeout=1.0)


def wait_for_policy_start(seconds: float, dry_run: bool) -> None:
    if not dry_run and seconds > 0:
        time.sleep(seconds)
