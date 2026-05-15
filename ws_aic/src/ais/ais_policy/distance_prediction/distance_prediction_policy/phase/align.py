"""Align phase helpers for DebugSfpDistancePolicy."""

from __future__ import annotations

import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

import numpy as np

from distance_prediction_policy.config import DistancePredictionConfig


_CSV_COLUMNS = [
    "wall_time",
    "step",
    "t_obs_ms",
    "t_pred_ms",
    "t_cmd_ms",
    "drift_mm",
    "drift_dx_mm",
    "drift_dy_mm",
    "drift_dz_mm",
    "xy_base_mm",
    "z_base_est_mm",
    "stable_count",
    "pred_x_mm",
    "pred_y_mm",
    "pred_z_mm",
    "pred_raw_x_mm",
    "pred_raw_y_mm",
    "pred_raw_z_mm",
    "bias_x_mm",
    "bias_y_mm",
    "bias_z_mm",
    "cmd_xy_x_mm",
    "cmd_xy_y_mm",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_at_cmd_x_m",
    "tcp_at_cmd_y_m",
    "tcp_at_cmd_z_m",
]


def _sanitize_task_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return token[:40]


def _build_metrics_label(task) -> str:
    parts: list[str] = []
    for attr in ("target_module_name", "port_name"):
        v = _sanitize_task_token(str(getattr(task, attr, "") or ""))
        if v:
            parts.append(v)
    return "_".join(parts) or "task_unknown"


def _open_metrics_csv(policy) -> tuple[Optional[TextIO], Optional[object]]:
    try:
        env_dir = os.environ.get("AIC_DEBUG_ALIGN_METRICS_DIR")
        base_dir = Path(env_dir) if env_dir else Path.home() / "aic_debug" / "align_metrics"
        base_dir.mkdir(parents=True, exist_ok=True)
        label = _build_metrics_label(getattr(policy, "_task", None))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base_dir / f"align_{stamp}_{label}.csv"
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        f.flush()
        policy.get_logger().info(f"[align metrics] CSV: {path}")
        return f, w
    except Exception as exc:
        policy.get_logger().warn(f"[align metrics] CSV open failed: {exc}")
        return None, None


def run_align_stage(policy, get_observation, move_robot) -> bool:
    policy.get_logger().info("[stage 4/5] align start")
    csv_file, csv_writer = _open_metrics_csv(policy)
    try:
        return _run_align_stage_inner(
            policy, get_observation, move_robot, csv_writer
        )
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass


def _run_align_stage_inner(
    policy, get_observation, move_robot, csv_writer
) -> bool:
    # Bias 가 0 이 아니면 align 시작 시 한 번 알림.
    bias_mm = (
        float(DistancePredictionConfig.ALIGN_BIAS_X_MM),
        float(DistancePredictionConfig.ALIGN_BIAS_Y_MM),
        float(DistancePredictionConfig.ALIGN_BIAS_Z_MM),
    )
    if any(abs(b) > 1e-9 for b in bias_mm):
        policy.get_logger().info(
            f"align bias (subtracted from pred): "
            f"x={bias_mm[0]:+.2f}mm, y={bias_mm[1]:+.2f}mm, z={bias_mm[2]:+.2f}mm"
        )

    pre_settle_s = float(DistancePredictionConfig.ALIGN_PRE_SETTLE_S)
    baseline_samples: list[np.ndarray] = []
    if pre_settle_s > 0:
        n_samples = 5
        policy.get_logger().info(
            f"align pre-settle: {pre_settle_s:.2f}s (force samples={n_samples})"
        )
        sample_dt = pre_settle_s / max(1, n_samples)
        for i in range(n_samples):
            policy.sleep_for(sample_dt)
            f = policy._force_vector(get_observation())
            if f is not None:
                baseline_samples.append(f)
                policy.get_logger().info(
                    f"  pre-settle sample {i+1}/{n_samples} "
                    f"(t=+{sample_dt*(i+1):.2f}s): "
                    f"fx={f[0]:+.2f}N, fy={f[1]:+.2f}N, fz={f[2]:+.2f}N"
                )

    stable_count = 0
    last_xy = None

    # Baseline = pre-settle 동안 모은 sample 들의 median.
    # outlier 1~2개가 섞여도 영향 없음 (mean 대비 강건). sample 없으면 단발 측정.
    if baseline_samples:
        baseline_arr = np.stack(baseline_samples)
        baseline_force = np.median(baseline_arr, axis=0)
        baseline_std = np.std(baseline_arr, axis=0)
    else:
        baseline_force = policy._force_vector(get_observation())
        baseline_std = None

    if baseline_force is None:
        policy.get_logger().warn("align force baseline unavailable; retry disabled")
    elif baseline_std is not None:
        policy.get_logger().info(
            f"align force baseline (median of {len(baseline_samples)} samples): "
            f"fx={baseline_force[0]:+.2f}±{baseline_std[0]:.2f}N, "
            f"fy={baseline_force[1]:+.2f}±{baseline_std[1]:.2f}N, "
            f"fz={baseline_force[2]:+.2f}±{baseline_std[2]:.2f}N, "
            f"xy_threshold="
            f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N:.2f}N, "
            f"z_threshold="
            f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_Z_THRESHOLD_N:.2f}N"
        )
    else:
        policy.get_logger().info(
            "align force baseline: "
            f"fx={baseline_force[0]:+.2f}N, "
            f"fy={baseline_force[1]:+.2f}N, "
            f"fz={baseline_force[2]:+.2f}N, "
            f"xy_threshold="
            f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_XY_THRESHOLD_N:.2f}N, "
            f"z_threshold="
            f"{DistancePredictionConfig.ALIGN_RETRY_FORCE_Z_THRESHOLD_N:.2f}N"
        )
    for step in range(DistancePredictionConfig.ALIGN_MAX_STEPS):
        t_obs_start = time.perf_counter()
        obs = get_observation()
        t_obs_end = time.perf_counter()
        tcp_pose = policy._tcp_pose(obs)
        if tcp_pose is None:
            policy.sleep_for(DistancePredictionConfig.DT)
            continue

        force = policy._force_vector(obs)
        force_delta = None
        if force is not None and baseline_force is not None:
            force_delta = force - baseline_force
        retry_step = policy._align_retry_step_base(
            tcp_pose=tcp_pose,
            force_delta=force_delta,
        )
        if retry_step is not None:
            stable_count = 0
            target_pose = policy._copy_pose(tcp_pose)
            target_pose.position.x += float(retry_step[0])
            target_pose.position.y += float(retry_step[1])
            target_pose.position.z += float(retry_step[2])
            if policy._target_orientation is not None:
                target_pose.orientation = policy._copy_quaternion(
                    policy._target_orientation
                )
            policy.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
                damping=list(DistancePredictionConfig.ALIGN_DAMPING),
            )
            delta_text = ""
            if force_delta is not None:
                delta_text = (
                    f"force_delta=({force_delta[0]:+.2f}, "
                    f"{force_delta[1]:+.2f}, "
                    f"{force_delta[2]:+.2f})N, "
                )
            policy.get_logger().warn(
                f"align[{step:03d}] retry: "
                f"{delta_text}"
                f"cmd_base=({retry_step[0]*1000:+.2f}, "
                f"{retry_step[1]*1000:+.2f}, "
                f"{retry_step[2]*1000:+.2f})mm"
            )
            policy.sleep_for(DistancePredictionConfig.ALIGN_COMMAND_SETTLE_S)
            continue

        t_pred_start = time.perf_counter()
        offset_m = policy._distance.predict_offset_m(obs, policy._port_id())
        t_pred_end = time.perf_counter()
        if offset_m is None:
            policy.sleep_for(DistancePredictionConfig.DT)
            continue

        correction_base = policy._align_correction_base(offset_m)
        offset_base_raw = -correction_base

        # Systematic bias 보정 (base frame, mm → m 변환).
        # 모델이 항상 일정한 편차로 잘못 예측하는 경우 이걸로 상쇄.
        bias_base_m = np.array(
            [
                float(DistancePredictionConfig.ALIGN_BIAS_X_MM) / 1000.0,
                float(DistancePredictionConfig.ALIGN_BIAS_Y_MM) / 1000.0,
                float(DistancePredictionConfig.ALIGN_BIAS_Z_MM) / 1000.0,
            ],
            dtype=np.float64,
        )
        offset_base = offset_base_raw - bias_base_m
        correction_base = -offset_base
        xy_base = float(np.linalg.norm(offset_base[:2]))
        last_xy = xy_base
        if xy_base < DistancePredictionConfig.ALIGN_FINISH_XY_M:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= DistancePredictionConfig.ALIGN_STABLE_STEPS:
            policy.get_logger().info(
                f"align stable: xy_base={xy_base*1000:.2f}mm x {stable_count}"
            )
            return True

        step_xy = correction_base[:2] * DistancePredictionConfig.XY_GAIN
        step_xy = np.clip(
            step_xy,
            -DistancePredictionConfig.MAX_XY_STEP_M,
            DistancePredictionConfig.MAX_XY_STEP_M,
        )
        if np.linalg.norm(step_xy) < DistancePredictionConfig.XY_DEADBAND_M:
            step_xy[:] = 0.0

        target_pose = policy._copy_pose(tcp_pose)
        target_pose.position.x += float(step_xy[0])
        target_pose.position.y += float(step_xy[1])
        if policy._target_orientation is not None:
            target_pose.orientation = policy._copy_quaternion(policy._target_orientation)

        # Drift 측정: obs 시점 tcp 와 명령 발행 직전 tcp 의 차이
        # = 추론(+후처리) Δt 동안 로봇이 실제로 움직인 거리.
        obs_at_cmd = get_observation()
        tcp_at_cmd = policy._tcp_pose(obs_at_cmd)
        if tcp_at_cmd is not None:
            drift_vec = np.array(
                [
                    tcp_at_cmd.position.x - tcp_pose.position.x,
                    tcp_at_cmd.position.y - tcp_pose.position.y,
                    tcp_at_cmd.position.z - tcp_pose.position.z,
                ],
                dtype=np.float64,
            )
            drift_mm = float(np.linalg.norm(drift_vec)) * 1000.0
            drift_xyz_mm = drift_vec * 1000.0
        else:
            drift_mm = float("nan")
            drift_xyz_mm = np.array([float("nan")] * 3)

        t_cmd_start = time.perf_counter()
        policy.set_pose_target(
            move_robot=move_robot,
            pose=target_pose,
            stiffness=list(DistancePredictionConfig.ALIGN_STIFFNESS),
            damping=list(DistancePredictionConfig.ALIGN_DAMPING),
        )
        t_cmd_end = time.perf_counter()

        # TCP 절대 위치 (의심되는 stuck 진단용)
        tcp_at_cmd_xyz = (
            (float(tcp_at_cmd.position.x),
             float(tcp_at_cmd.position.y),
             float(tcp_at_cmd.position.z))
            if tcp_at_cmd is not None
            else (float("nan"), float("nan"), float("nan"))
        )
        policy.get_logger().info(
            f"align[{step:03d}]: "
            f"pred_base_est=({offset_base[0]*1000:+.2f}, "
            f"{offset_base[1]*1000:+.2f}, {offset_base[2]*1000:+.2f})mm, "
            f"cmd_base_xy=({step_xy[0]*1000:+.2f}, "
            f"{step_xy[1]*1000:+.2f})mm, "
            f"xy_base={xy_base*1000:.2f}mm, "
            f"z_base_est={offset_base[2]*1000:+.2f}mm, "
            f"stable={stable_count}/"
            f"{DistancePredictionConfig.ALIGN_STABLE_STEPS}, "
            f"tcp_abs=({tcp_pose.position.x:+.4f}, "
            f"{tcp_pose.position.y:+.4f}, "
            f"{tcp_pose.position.z:+.4f})m, "
            f"t_obs={(t_obs_end-t_obs_start)*1000:.1f}ms, "
            f"t_pred={(t_pred_end-t_pred_start)*1000:.1f}ms, "
            f"t_cmd={(t_cmd_end-t_cmd_start)*1000:.1f}ms, "
            f"drift={drift_mm:.2f}mm "
            f"(dx={drift_xyz_mm[0]:+.2f}, "
            f"dy={drift_xyz_mm[1]:+.2f}, "
            f"dz={drift_xyz_mm[2]:+.2f})mm"
        )

        if csv_writer is not None:
            try:
                csv_writer.writerow([
                    time.time(),
                    step,
                    (t_obs_end - t_obs_start) * 1000.0,
                    (t_pred_end - t_pred_start) * 1000.0,
                    (t_cmd_end - t_cmd_start) * 1000.0,
                    drift_mm,
                    float(drift_xyz_mm[0]),
                    float(drift_xyz_mm[1]),
                    float(drift_xyz_mm[2]),
                    xy_base * 1000.0,
                    float(offset_base[2]) * 1000.0,
                    stable_count,
                    float(offset_base[0]) * 1000.0,
                    float(offset_base[1]) * 1000.0,
                    float(offset_base[2]) * 1000.0,
                    float(offset_base_raw[0]) * 1000.0,
                    float(offset_base_raw[1]) * 1000.0,
                    float(offset_base_raw[2]) * 1000.0,
                    bias_mm[0],
                    bias_mm[1],
                    bias_mm[2],
                    float(step_xy[0]) * 1000.0,
                    float(step_xy[1]) * 1000.0,
                    float(tcp_pose.position.x),
                    float(tcp_pose.position.y),
                    float(tcp_pose.position.z),
                    tcp_at_cmd_xyz[0],
                    tcp_at_cmd_xyz[1],
                    tcp_at_cmd_xyz[2],
                ])
            except Exception as exc:
                policy.get_logger().warn(f"[align metrics] row write failed: {exc}")

        policy.sleep_for(DistancePredictionConfig.ALIGN_COMMAND_SETTLE_S)

    if last_xy is None:
        policy.get_logger().error("align failed: no distance predictions")
        return False
    success = last_xy < DistancePredictionConfig.ALIGN_FINISH_XY_M * 1.5
    policy.get_logger().info(
        f"[stage 4/5] align done: "
        f"success={success}, last_xy_base={last_xy*1000:.2f}mm"
    )
    return success
