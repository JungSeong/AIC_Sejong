#!/usr/bin/env python3
"""Collect SFP distance samples while perturbing the commanded gripper RPY.

This variant keeps the existing SFP distance label:
``label.plug_tip_to_port`` is still the measured plug-tip position expressed in
the selected SFP port frame.  In addition, each sample records the nominal
gripper-offset RPY, the sampled RPY jitter, and the current RPY values measured
from TF/controller state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_sfp_distance_dataset as base


DEFAULT_OUTPUT_DIR = base.WS_AIC_ROOT / "data" / "distance_prediction" / "SFP_RPY"
DEFAULT_GRIPPER_OFFSET_M = (0.0, 0.015385, 0.04245)
DEFAULT_GRIPPER_RPY_RAD = (0.4432, -0.4838, 1.3303)
DEFAULT_NIC_TRANSLATION_RANGE_M = (-0.0215, 0.0234)
DEFAULT_NIC_YAW_RANGE_RAD = (-math.radians(10.0), math.radians(10.0))
RPY_NAMES = ("roll", "pitch", "yaw")


def parse_rpy_axes(value: str) -> tuple[str, ...]:
    axes = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = set(axes) - set(RPY_NAMES)
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid RPY axes: {sorted(invalid)}")
    if not axes:
        raise argparse.ArgumentTypeError("must contain at least one of roll,pitch,yaw")
    return axes


def rpy_dict(values: tuple[float, float, float] | np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(RPY_NAMES)}


def xyz_dict(values: tuple[float, float, float] | np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(("x", "y", "z"))}


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return base.quat_normalize_wxyz(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def transform_rpy(transform: base.Transform) -> tuple[float, float, float]:
    return matrix_to_rpy(base.transform_to_matrix(transform)[:3, :3])


def pose_rpy(pose: base.Pose) -> tuple[float, float, float]:
    transform = base.pose_to_transform(pose)
    return transform_rpy(transform)


def relative_pose_info(parent_tf: base.Transform, child_tf: base.Transform) -> dict[str, Any]:
    parent_to_child = np.linalg.inv(base.transform_to_matrix(parent_tf)) @ base.transform_to_matrix(child_tf)
    return {
        "translation_m": xyz_dict(parent_to_child[:3, 3]),
        "rpy_rad": rpy_dict(matrix_to_rpy(parent_to_child[:3, :3])),
    }


def transform_pose_info(transform: base.Transform) -> dict[str, Any]:
    return {
        "position_m": {
            "x": float(transform.translation.x),
            "y": float(transform.translation.y),
            "z": float(transform.translation.z),
        },
        "orientation_xyzw": {
            "x": float(transform.rotation.x),
            "y": float(transform.rotation.y),
            "z": float(transform.rotation.z),
            "w": float(transform.rotation.w),
        },
        "rpy_rad": rpy_dict(transform_rpy(transform)),
    }


def scene_randomization_metadata(args: argparse.Namespace, module_name: str) -> dict[str, Any]:
    provided = args.nic_translation_m is not None or args.nic_yaw_rad is not None
    return {
        "source": "launch_args" if provided else "not_provided_to_collector",
        "target_module_name": module_name,
        "nic_translation_m": None if args.nic_translation_m is None else float(args.nic_translation_m),
        "nic_yaw_rad": None if args.nic_yaw_rad is None else float(args.nic_yaw_rad),
        "configured_range": {
            "nic_translation_m": {
                "low": DEFAULT_NIC_TRANSLATION_RANGE_M[0],
                "high": DEFAULT_NIC_TRANSLATION_RANGE_M[1],
            },
            "nic_yaw_rad": {
                "low": DEFAULT_NIC_YAW_RANGE_RAD[0],
                "high": DEFAULT_NIC_YAW_RANGE_RAD[1],
            },
        },
        "note": (
            "NIC translation/yaw must be applied when the task board is spawned. "
            "The per-sample target_port_in_base pose below is the TF ground truth "
            "of the actual spawned port."
        ),
    }


def stable_jitter_seed(seed: int, module_name: str, port_name: str, step_index: int) -> int:
    value = int(seed) & 0xFFFFFFFF
    for text in (module_name, port_name):
        for char in text:
            value = ((value * 131) + ord(char)) & 0xFFFFFFFF
    value = (value + int(step_index) * 1_000_003) & 0xFFFFFFFF
    return value


def sample_rpy_jitter(args: argparse.Namespace, module_name: str, port_name: str, step_index: int) -> np.ndarray:
    jitter = np.zeros(3, dtype=np.float64)
    if args.rpy_jitter_rad <= 0.0:
        return jitter
    rng = np.random.default_rng(stable_jitter_seed(args.seed, module_name, port_name, step_index))
    enabled_axes = set(args.rpy_jitter_axes)
    for axis_index, axis_name in enumerate(RPY_NAMES):
        if axis_name in enabled_axes:
            jitter[axis_index] = rng.uniform(-args.rpy_jitter_rad, args.rpy_jitter_rad)
    return jitter


def make_target_pose_with_rpy_jitter(
    port_tf: base.Transform,
    plug_tf: base.Transform,
    tcp_tf: base.Transform,
    local_offset: base.LocalOffset,
    jitter_rpy_rad: np.ndarray,
) -> tuple[base.Pose, dict[str, Any]]:
    port_matrix = base.transform_to_matrix(port_tf)
    target_plug_xyz = (port_matrix @ np.append(local_offset.vector_m(), 1.0))[:3]

    q_port = base.transform_quat_wxyz(port_tf)
    q_plug = base.transform_quat_wxyz(plug_tf)
    q_tcp = base.transform_quat_wxyz(tcp_tf)
    q_diff = base.quat_multiply_wxyz(q_port, base.quat_inverse_wxyz(q_plug))
    q_nominal_tcp = base.quat_normalize_wxyz(base.quat_multiply_wxyz(q_diff, q_tcp))
    q_jitter = rpy_to_quat_wxyz(
        float(jitter_rpy_rad[0]),
        float(jitter_rpy_rad[1]),
        float(jitter_rpy_rad[2]),
    )
    # Apply jitter in the local TCP frame after the nominal alignment command.
    q_target_tcp = base.quat_normalize_wxyz(base.quat_multiply_wxyz(q_nominal_tcp, q_jitter))

    tcp_rotation = base.transform_to_matrix(tcp_tf)[:3, :3]
    target_tcp_rotation = base.quat_to_matrix_xyzw(
        q_target_tcp[1],
        q_target_tcp[2],
        q_target_tcp[3],
        q_target_tcp[0],
    )
    nominal_tcp_rotation = base.quat_to_matrix_xyzw(
        q_nominal_tcp[1],
        q_nominal_tcp[2],
        q_nominal_tcp[3],
        q_nominal_tcp[0],
    )
    tcp_to_plug_local = tcp_rotation.T @ (base.transform_xyz(plug_tf) - base.transform_xyz(tcp_tf))
    target_tcp_xyz = target_plug_xyz - target_tcp_rotation @ tcp_to_plug_local
    nominal_tcp_xyz = target_plug_xyz - nominal_tcp_rotation @ tcp_to_plug_local

    pose = base.Pose(
        position=base.Point(
            x=float(target_tcp_xyz[0]),
            y=float(target_tcp_xyz[1]),
            z=float(target_tcp_xyz[2]),
        ),
        orientation=base.Quaternion(
            w=float(q_target_tcp[0]),
            x=float(q_target_tcp[1]),
            y=float(q_target_tcp[2]),
            z=float(q_target_tcp[3]),
        ),
    )
    nominal_pose = base.Pose(
        position=base.Point(
            x=float(nominal_tcp_xyz[0]),
            y=float(nominal_tcp_xyz[1]),
            z=float(nominal_tcp_xyz[2]),
        ),
        orientation=base.Quaternion(
            w=float(q_nominal_tcp[0]),
            x=float(q_nominal_tcp[1]),
            y=float(q_nominal_tcp[2]),
            z=float(q_nominal_tcp[3]),
        ),
    )
    metadata = {
        "jitter_rpy_rad": rpy_dict(jitter_rpy_rad),
        "nominal_tcp_command_rpy_rad": rpy_dict(pose_rpy(nominal_pose)),
        "jittered_tcp_command_rpy_rad": rpy_dict(pose_rpy(pose)),
        "tcp_to_plug_local_m": xyz_dict(tcp_to_plug_local),
    }
    return pose, metadata


def make_rpy_metadata(
    node: base.SfpDistanceCollector,
    args: argparse.Namespace,
    cable_tip_frame: str,
    command_pose: base.Pose,
    command_rpy_metadata: dict[str, Any],
) -> dict[str, Any]:
    plug_tf = node.lookup_transform("base_link", cable_tip_frame)
    source_tcp_tf = node.tcp_transform(args.tcp_frame, args.tcp_pose_source)
    try:
        tf_tcp = node.lookup_transform("base_link", args.tcp_frame)
        tf_tcp_available = True
        measured_tcp_tf = tf_tcp
        measured_tcp_source = "tf"
    except base.TransformException:
        tf_tcp_available = False
        measured_tcp_tf = source_tcp_tf
        measured_tcp_source = args.tcp_pose_source

    tcp_minus_plug_base_m = base.transform_xyz(measured_tcp_tf) - base.transform_xyz(plug_tf)
    current_tcp_rpy = transform_rpy(measured_tcp_tf)
    current_plug_rpy = transform_rpy(plug_tf)
    nominal_rpy = np.asarray(
        [args.nominal_gripper_roll, args.nominal_gripper_pitch, args.nominal_gripper_yaw],
        dtype=np.float64,
    )
    jitter = np.asarray(
        [command_rpy_metadata["jitter_rpy_rad"][name] for name in RPY_NAMES],
        dtype=np.float64,
    )
    return {
        "nominal_config": {
            "gripper_offset_m": xyz_dict(
                np.asarray(
                    [
                        args.nominal_gripper_offset_x,
                        args.nominal_gripper_offset_y,
                        args.nominal_gripper_offset_z,
                    ],
                    dtype=np.float64,
                )
            ),
            "rpy_rad": rpy_dict(nominal_rpy),
        },
        "sampled": {
            "jitter_rpy_rad": command_rpy_metadata["jitter_rpy_rad"],
            "nominal_plus_jitter_rpy_rad": rpy_dict(nominal_rpy + jitter),
            "nominal_tcp_command_rpy_rad": command_rpy_metadata["nominal_tcp_command_rpy_rad"],
            "jittered_tcp_command_rpy_rad": command_rpy_metadata["jittered_tcp_command_rpy_rad"],
            "tcp_to_plug_local_m_before_command": command_rpy_metadata["tcp_to_plug_local_m"],
        },
        "current": {
            "tcp_source": measured_tcp_source,
            "tf_tcp_available": bool(tf_tcp_available),
            "commanded_tcp_in_base_rpy_rad": rpy_dict(pose_rpy(command_pose)),
            "measured_tcp_in_base_rpy_rad": rpy_dict(current_tcp_rpy),
            "plug_tip_in_base_rpy_rad": rpy_dict(current_plug_rpy),
            "tcp_minus_plug_tip_base_m": xyz_dict(tcp_minus_plug_base_m),
            "tcp_to_plug_tip_tf": relative_pose_info(measured_tcp_tf, plug_tf),
            "plug_tip_to_tcp_tf": relative_pose_info(plug_tf, measured_tcp_tf),
        },
    }


def save_sample_with_rpy(
    node: base.SfpDistanceCollector,
    output_dir: Path,
    episode_name: str,
    sample_id: str,
    step_index: int,
    module_name: str,
    target_port_name: str,
    all_port_names: tuple[str, ...],
    target_port_frame: str,
    cable_tip_frame: str,
    tcp_frame: str,
    port_frame_mode: str,
    tcp_pose_source: str,
    command_pose: base.Pose,
    requested_offset: base.LocalOffset,
    placement_check: dict[str, Any],
    rpy_metadata: dict[str, Any],
    scene_randomization: dict[str, Any],
) -> bool:
    port_tf = node.lookup_transform("base_link", target_port_frame)
    plug_tf = node.lookup_transform("base_link", cable_tip_frame)
    tcp_tf = node.tcp_transform(tcp_frame, tcp_pose_source)
    try:
        tcp_tf_frame = node.lookup_transform("base_link", tcp_frame)
        tcp_tf_diagnostic: dict[str, Any] = {
            "available": True,
            "x": float(tcp_tf_frame.translation.x),
            "y": float(tcp_tf_frame.translation.y),
            "z": float(tcp_tf_frame.translation.z),
            "rpy_rad": rpy_dict(transform_rpy(tcp_tf_frame)),
        }
    except base.TransformException:
        tcp_tf_diagnostic = {"available": False}

    image_paths: dict[str, str] = {}
    for camera in base.CAMERAS:
        image = node._latest_image.get(camera)
        if image is None:
            return False
        image_dir = output_dir / "images" / camera / episode_name
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{sample_id}_{camera}.png"
        base.cv2.imwrite(str(image_path), image)
        image_paths[camera] = str(image_path.relative_to(output_dir))

    ports: dict[str, Any] = {}
    for port_name in all_port_names:
        key = f"{module_name}/{port_name}"
        frame = node.port_frame(module_name, port_name, port_frame_mode)
        try:
            candidate_port_tf = node.lookup_transform("base_link", frame)
        except base.TransformException:
            ports[key] = {"available": False, "is_target": port_name == target_port_name, "frame": ""}
            continue
        ports[key] = {
            "available": True,
            "is_target": port_name == target_port_name,
            "frame": frame,
            "plug_tip_in_port": node.plug_tip_to_port_label(candidate_port_tf, plug_tf),
            "port_in_plug_tip": base.relative_transform_label(plug_tf, candidate_port_tf),
        }

    label = node.plug_tip_to_port_label(port_tf, plug_tf)
    record = {
        "sample_id": sample_id,
        "episode_name": episode_name,
        "task_id": f"distance_sfp_rpy_{module_name}_{target_port_name}",
        "task_type": "nic",
        "port_type": "sfp",
        "port_name": target_port_name,
        "target_module_name": module_name,
        "phase": "distance_ci99_rpy",
        "step_index": int(step_index),
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "images": image_paths,
        "label": {
            "frame": target_port_frame,
            "frame_mode": port_frame_mode,
            "coordinate": "plug_tip position expressed in the target SFP port frame",
            "source": "tf_base_link",
            "plug_tip_to_port": label,
            "ports": ports,
        },
        "frames": {
            "world": "base_link",
            "motion_command_frame": "base_link",
            "motion_command_target": tcp_frame,
            "motion_command_target_pose_source": tcp_pose_source,
            "requested_offset_frame": target_port_frame,
            "label_frame": target_port_frame,
            "plug_tip_frame": cable_tip_frame,
            "tcp_frame": tcp_frame,
        },
        "poses": {
            "target_port_in_base": transform_pose_info(port_tf),
            "plug_tip_in_base": transform_pose_info(plug_tf),
            "tcp_in_base": transform_pose_info(tcp_tf),
            "target_port_to_plug_tip": relative_pose_info(port_tf, plug_tf),
            "plug_tip_to_target_port": relative_pose_info(plug_tf, port_tf),
        },
        "command": {
            "position": {
                "x": float(command_pose.position.x),
                "y": float(command_pose.position.y),
                "z": float(command_pose.position.z),
            },
            "orientation": {
                "x": float(command_pose.orientation.x),
                "y": float(command_pose.orientation.y),
                "z": float(command_pose.orientation.z),
                "w": float(command_pose.orientation.w),
            },
            "rpy_rad": rpy_dict(pose_rpy(command_pose)),
        },
        "generator": {
            "name": "sfp_distance_ci99_rpy",
            "requested_port_local_offset": requested_offset.to_dict(),
            "requested_offset_frame": target_port_frame,
            "target_port_frame": target_port_frame,
            "port_frame_mode": port_frame_mode,
            "cable_tip_frame": cable_tip_frame,
            "tcp_frame": tcp_frame,
            "tcp_pose_source": tcp_pose_source,
            "placement_check": placement_check,
            "gripper_offset_rpy": rpy_metadata,
            "scene_randomization": scene_randomization,
            "measured_tcp_in_base": {
                "x": float(tcp_tf.translation.x),
                "y": float(tcp_tf.translation.y),
                "z": float(tcp_tf.translation.z),
                "rpy_rad": rpy_dict(transform_rpy(tcp_tf)),
            },
            "tf_tcp_in_base": tcp_tf_diagnostic,
        },
    }

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(base.json_safe(record), ensure_ascii=False) + "\n")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-module-name", type=str, default=None)
    parser.add_argument("--max-mounts", type=int, default=5)
    parser.add_argument("--port-names", type=base.parse_csv, default=base.SFP_PORT_NAMES)
    parser.add_argument("--all-port-names", type=base.parse_csv, default=base.SFP_PORT_NAMES)
    parser.add_argument("--port-frame-mode", choices=base.PORT_FRAME_MODE_CHOICES, default=base.DEFAULT_PORT_FRAME_MODE)
    parser.add_argument("--cable-tip-frame", type=str, default=None)
    parser.add_argument("--cable-tip-candidates", type=base.parse_csv, default=base.DEFAULT_CABLE_TIP_CANDIDATES)
    parser.add_argument("--tcp-frame", type=str, default=base.DEFAULT_TCP_FRAME)
    parser.add_argument("--tcp-pose-source", choices=base.TCP_POSE_SOURCE_CHOICES, default=base.DEFAULT_TCP_POSE_SOURCE)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-mode", choices=base.SAMPLING_MODE_CHOICES, default="step-grid")
    parser.add_argument("--samples-per-port", type=int, default=base.DEFAULT_SAMPLES_PER_PORT)
    parser.add_argument("--grid-step-mm", type=float, default=base.DEFAULT_GRID_STEP_MM)
    parser.add_argument("--grid-per-axis", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-ci99-mm", type=base.parse_range_mm, default=base.OffsetRangeMM(*base.DEFAULT_CI99_X_MM))
    parser.add_argument("--y-ci99-mm", type=base.parse_range_mm, default=base.OffsetRangeMM(*base.DEFAULT_CI99_Y_MM))
    parser.add_argument("--z-ci99-error-mm", type=base.parse_range_mm, default=base.OffsetRangeMM(*base.DEFAULT_CI99_Z_ERROR_MM))
    parser.add_argument("--approach-offset-mm", type=float, default=base.DEFAULT_APPROACH_OFFSET_MM)
    parser.add_argument("--wait-s", type=float, default=15.0)
    parser.add_argument("--tf-warmup-s", type=float, default=3.0)
    parser.add_argument("--move-settle-s", type=float, default=2.5)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--command-repeats", type=int, default=5)
    parser.add_argument("--placement-check-retries", type=int, default=4)
    parser.add_argument("--max-placement-axis-error-mm", type=float, default=base.DEFAULT_MAX_PLACEMENT_AXIS_ERROR_MM)
    parser.add_argument("--max-placement-euclidean-error-mm", type=float, default=base.DEFAULT_MAX_PLACEMENT_EUCLIDEAN_ERROR_MM)
    parser.add_argument("--max-tcp-position-error-mm", type=float, default=base.DEFAULT_MAX_TCP_POSITION_ERROR_MM)
    parser.add_argument("--allow-existing-samples", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-correction-step-mm", type=float, default=base.DEFAULT_MAX_CORRECTION_STEP_MM)
    parser.add_argument("--stiffness", type=base.parse_csv, default=("200", "200", "200", "50", "50", "50"))
    parser.add_argument("--damping", type=base.parse_csv, default=("80", "80", "80", "20", "20", "20"))
    parser.add_argument("--rpy-jitter-rad", type=float, default=0.04)
    parser.add_argument("--rpy-jitter-axes", type=parse_rpy_axes, default=RPY_NAMES)
    parser.add_argument("--nominal-gripper-offset-x", type=float, default=DEFAULT_GRIPPER_OFFSET_M[0])
    parser.add_argument("--nominal-gripper-offset-y", type=float, default=DEFAULT_GRIPPER_OFFSET_M[1])
    parser.add_argument("--nominal-gripper-offset-z", type=float, default=DEFAULT_GRIPPER_OFFSET_M[2])
    parser.add_argument("--nominal-gripper-roll", type=float, default=DEFAULT_GRIPPER_RPY_RAD[0])
    parser.add_argument("--nominal-gripper-pitch", type=float, default=DEFAULT_GRIPPER_RPY_RAD[1])
    parser.add_argument("--nominal-gripper-yaw", type=float, default=DEFAULT_GRIPPER_RPY_RAD[2])
    parser.add_argument("--nic-translation-m", type=float, default=None)
    parser.add_argument("--nic-yaw-rad", type=float, default=None)
    return parser.parse_args()


def write_generation_config(output_dir: Path, run_id: str, args: argparse.Namespace, offsets: list[base.LocalOffset]) -> None:
    base.write_generation_config(output_dir, run_id, args, offsets)
    config_path = output_dir / "meta" / f"generation_config_{run_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["generator_name"] = "sfp_distance_ci99_rpy"
    config["rpy_jitter"] = {
        "range_rad": float(args.rpy_jitter_rad),
        "axes": list(args.rpy_jitter_axes),
        "sampling": "uniform_per_axis",
        "application_frame": "local_tcp_after_nominal_alignment",
        "deterministic_seed": int(args.seed),
    }
    config["nominal_gripper_offset"] = {
        "translation_m": xyz_dict(
            np.asarray(
                [
                    args.nominal_gripper_offset_x,
                    args.nominal_gripper_offset_y,
                    args.nominal_gripper_offset_z,
                ],
                dtype=np.float64,
            )
        ),
        "rpy_rad": rpy_dict(
            np.asarray(
                [
                    args.nominal_gripper_roll,
                    args.nominal_gripper_pitch,
                    args.nominal_gripper_yaw,
                ],
                dtype=np.float64,
            )
        ),
    }
    config["nic_randomization"] = {
        "range": {
            "nic_translation_m": {
                "low": DEFAULT_NIC_TRANSLATION_RANGE_M[0],
                "high": DEFAULT_NIC_TRANSLATION_RANGE_M[1],
            },
            "nic_yaw_rad": {
                "low": DEFAULT_NIC_YAW_RANGE_RAD[0],
                "high": DEFAULT_NIC_YAW_RANGE_RAD[1],
            },
        },
        "applied_nic_translation_m": args.nic_translation_m,
        "applied_nic_yaw_rad": args.nic_yaw_rad,
        "application": "launch-time task_board nic_card_mount_* arguments",
    }
    config_path.write_text(json.dumps(base.json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")


def validate_resume_config(output_dir: Path, run_id: str, args: argparse.Namespace, offsets: list[base.LocalOffset]) -> None:
    base.validate_resume_config(output_dir, run_id, args, offsets)
    config_path = output_dir / "meta" / f"generation_config_{run_id}.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "generator_name": "sfp_distance_ci99_rpy",
        "rpy_jitter": {
            "range_rad": float(args.rpy_jitter_rad),
            "axes": list(args.rpy_jitter_axes),
            "sampling": "uniform_per_axis",
            "application_frame": "local_tcp_after_nominal_alignment",
            "deterministic_seed": int(args.seed),
        },
    }
    mismatches = []
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            mismatches.append(f"{key}: previous={config.get(key)!r}, current={expected_value!r}")
    if mismatches:
        raise ValueError("Resume RPY configuration does not match.\n" + "\n".join(mismatches))


def collect_for_port(
    node: base.SfpDistanceCollector,
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    module_name: str,
    port_name: str,
    all_port_names: tuple[str, ...],
    cable_tip_frame: str,
    offsets: list[base.LocalOffset],
    stiffness: tuple[float, ...],
    damping: tuple[float, ...],
    completed_step_indices: set[int] | None = None,
) -> int:
    target_port_frame = node.port_frame(module_name, port_name, args.port_frame_mode)
    if not node.wait_for_tf("base_link", target_port_frame, timeout_s=args.wait_s):
        node.log_tf_hints(target_port_frame)
        raise RuntimeError(f"Target port TF not available: {target_port_frame}")
    if not node.wait_for_tf("base_link", cable_tip_frame, timeout_s=args.wait_s):
        raise RuntimeError(f"Cable tip TF not available: {cable_tip_frame}")
    if args.tcp_pose_source == "tf" and not node.wait_for_tf("base_link", args.tcp_frame, timeout_s=args.wait_s):
        raise RuntimeError(f"TCP TF not available: {args.tcp_frame}")
    if args.tcp_pose_source == "controller_state" and not node.wait_for_controller_state(args.wait_s):
        raise RuntimeError("ControllerState not available: /aic_controller/controller_state")

    safe_module = base.safe_name(module_name)
    safe_port = base.safe_name(port_name)
    episode_name = f"{run_id}_{safe_module}_{safe_port}_distance_ci99_rpy"
    saved = 0
    command_bias_base_m = np.zeros(3, dtype=np.float64)
    scene_randomization = scene_randomization_metadata(args, module_name)
    completed_step_indices = completed_step_indices or set()
    skipped = sum(1 for index in completed_step_indices if 0 <= index < len(offsets))
    if skipped:
        node.get_logger().info(
            f"Resuming {port_name}: skipping {skipped}/{len(offsets)} already saved samples."
        )

    for step_index, offset in enumerate(offsets):
        if step_index in completed_step_indices:
            continue
        port_tf = node.lookup_transform("base_link", target_port_frame)
        plug_tf = node.lookup_transform("base_link", cable_tip_frame)
        tcp_tf = node.tcp_transform(args.tcp_frame, args.tcp_pose_source)
        jitter_rpy = sample_rpy_jitter(args, module_name, port_name, step_index)
        pose, command_rpy_metadata = make_target_pose_with_rpy_jitter(
            port_tf,
            plug_tf,
            tcp_tf,
            offset,
            jitter_rpy,
        )
        base.apply_position_correction(pose, command_bias_base_m)

        node.get_logger().info(
            f"{port_name} sample {step_index + 1}/{len(offsets)} "
            f"offset_mm=({offset.x_error_mm:+.3f}, {offset.y_error_mm:+.3f}, "
            f"{offset.z_error_mm:+.3f}), "
            f"rpy_jitter_rad=({jitter_rpy[0]:+.3f}, {jitter_rpy[1]:+.3f}, {jitter_rpy[2]:+.3f}), "
            f"bias_base_mm=({command_bias_base_m[0] * 1000.0:+.1f}, "
            f"{command_bias_base_m[1] * 1000.0:+.1f}, "
            f"{command_bias_base_m[2] * 1000.0:+.1f})"
        )
        for _ in range(max(1, args.command_repeats)):
            node.move_robot_to(pose, stiffness=stiffness, damping=damping)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.1)
        base.settle(node, args.move_settle_s)

        sample_id = f"{episode_name}_{step_index:06d}"
        placement_check: dict[str, Any] | None = None
        for attempt in range(args.placement_check_retries + 1):
            placement_check = base.measure_placement(
                node=node,
                target_port_frame=target_port_frame,
                cable_tip_frame=cable_tip_frame,
                tcp_frame=args.tcp_frame,
                tcp_pose_source=args.tcp_pose_source,
                command_pose=pose,
                requested_offset=offset,
            )
            if base.placement_check_ok(placement_check, args):
                node.get_logger().info(
                    "Placement check OK: " + base.placement_check_summary(placement_check)
                )
                break

            if attempt >= args.placement_check_retries:
                raise RuntimeError(
                    f"Placement validation failed for {sample_id}: "
                    + base.placement_check_summary(placement_check)
                )

            port_tf = node.lookup_transform("base_link", target_port_frame)
            correction_base_m = base.correction_from_placement_error(
                port_tf,
                placement_check,
                args.max_correction_step_mm,
            )
            base.apply_position_correction(pose, correction_base_m)
            command_bias_base_m += correction_base_m
            node.get_logger().warning(
                "Placement check failed; applying corrective command "
                f"base_m=({correction_base_m[0]:+.4f}, "
                f"{correction_base_m[1]:+.4f}, {correction_base_m[2]:+.4f}): "
                + base.placement_check_summary(placement_check)
            )
            for _ in range(max(1, args.command_repeats)):
                node.move_robot_to(pose, stiffness=stiffness, damping=damping)
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.1)
            base.settle(node, args.move_settle_s)

        rpy_metadata = make_rpy_metadata(
            node=node,
            args=args,
            cable_tip_frame=cable_tip_frame,
            command_pose=pose,
            command_rpy_metadata=command_rpy_metadata,
        )
        if save_sample_with_rpy(
            node=node,
            output_dir=output_dir,
            episode_name=episode_name,
            sample_id=sample_id,
            step_index=step_index,
            module_name=module_name,
            target_port_name=port_name,
            all_port_names=all_port_names,
            target_port_frame=target_port_frame,
            cable_tip_frame=cable_tip_frame,
            tcp_frame=args.tcp_frame,
            port_frame_mode=args.port_frame_mode,
            tcp_pose_source=args.tcp_pose_source,
            command_pose=pose,
            requested_offset=offset,
            placement_check=placement_check or {},
            rpy_metadata=rpy_metadata,
            scene_randomization=scene_randomization,
        ):
            saved += 1
        time.sleep(args.interval_s)
    return saved


def main() -> None:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    if args.resume and not samples_path.exists():
        raise RuntimeError(f"--resume requires an existing {samples_path}")
    if samples_path.exists() and not (args.allow_existing_samples or args.resume):
        raise RuntimeError(
            f"{samples_path} already exists. Move it aside or pass "
            "--allow-existing-samples if you intentionally want to append, "
            "or --resume to continue an interrupted run."
        )
    if args.placement_check_retries < 0:
        raise ValueError("--placement-check-retries must be >= 0")
    if args.max_placement_axis_error_mm < 0.0:
        raise ValueError("--max-placement-axis-error-mm must be >= 0")
    if args.max_placement_euclidean_error_mm < 0.0:
        raise ValueError("--max-placement-euclidean-error-mm must be >= 0")
    if args.max_correction_step_mm < 0.0:
        raise ValueError("--max-correction-step-mm must be >= 0")
    if args.rpy_jitter_rad < 0.0:
        raise ValueError("--rpy-jitter-rad must be >= 0")

    run_id = args.run_id or (base.infer_latest_run_id(samples_path) if args.resume else datetime.now().strftime("%Y%m%d_%H%M%S"))
    offsets = base.build_offsets(args)
    if args.resume:
        validate_resume_config(output_dir, run_id, args, offsets)
    else:
        write_generation_config(output_dir, run_id, args, offsets)
    completed_by_port = (
        base.load_completed_step_indices(samples_path, run_id) if args.resume else {}
    )

    stiffness = tuple(float(value) for value in args.stiffness)
    damping = tuple(float(value) for value in args.damping)
    if len(stiffness) != 6 or len(damping) != 6:
        raise ValueError("--stiffness and --damping must each have 6 comma-separated values")

    rclpy.init()
    node = base.SfpDistanceCollector()
    try:
        if not base.wait_for_camera_data(node, args.wait_s):
            node.get_logger().error("Camera data not ready.")
            sys.exit(1)
        base.warmup_tf(node, args.tf_warmup_s)

        module_name = args.target_module_name
        if module_name is None:
            module_name = node.discover_module_name(
                args.max_mounts,
                args.port_names,
                args.port_frame_mode,
            )
        if module_name is None:
            node.log_tf_hints("task_board/nic_card_mount_*/sfp_port_*_link")
            node.get_logger().error("No SFP module TF found.")
            sys.exit(1)

        cable_tip_frame = args.cable_tip_frame
        if cable_tip_frame is None:
            cable_tip_frame = node.discover_cable_tip_frame(
                args.cable_tip_candidates,
                timeout_s=0.5,
            )
        if cable_tip_frame is None:
            node.get_logger().error(
                "No cable tip TF found. Pass --cable-tip-frame, for example cable_0/sfp_tip_link."
            )
            sys.exit(1)

        node.get_logger().info(
            f"Collecting RPY distance dataset: output={output_dir}, "
            f"module={module_name}, ports={args.port_names}, run_id={run_id}, "
            f"offsets_per_port={len(offsets)}, rpy_jitter_rad=+/-{args.rpy_jitter_rad}, "
            f"rpy_axes={args.rpy_jitter_axes}, resume={args.resume}, "
            f"port_frame_mode={args.port_frame_mode}, cable_tip={cable_tip_frame}"
        )
        total_saved = 0
        for port_name in args.port_names:
            total_saved += collect_for_port(
                node=node,
                output_dir=output_dir,
                run_id=run_id,
                args=args,
                module_name=module_name,
                port_name=port_name,
                all_port_names=args.all_port_names,
                cable_tip_frame=cable_tip_frame,
                offsets=offsets,
                stiffness=stiffness,
                damping=damping,
                completed_step_indices=completed_by_port.get((module_name, port_name)),
            )
        node.get_logger().info(
            f"Done. saved_samples={total_saved}, samples_jsonl={output_dir / 'samples.jsonl'}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
