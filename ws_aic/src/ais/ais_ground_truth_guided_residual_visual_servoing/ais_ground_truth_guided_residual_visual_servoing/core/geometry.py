from __future__ import annotations

import numpy as np
from ais_transform import Pose3D
from transforms3d._gohlketransforms import quaternion_multiply


def pose3d_from_transform(transform) -> Pose3D:
    return Pose3D.from_xyz_quat(
        [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ],
    )


def pose3d_from_pose(pose) -> Pose3D:
    return Pose3D.from_xyz_quat(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
    )


def plug_tip_to_port_label(port_tf, plug_tf) -> dict[str, float]:
    port_pose = pose3d_from_transform(port_tf)
    plug_pose = pose3d_from_transform(plug_tf)
    local_offset = port_pose.rotation_matrix.T @ (
        plug_pose.position - port_pose.position
    )
    return {
        "x_m": float(local_offset[0]),
        "y_m": float(local_offset[1]),
        "z_m": float(local_offset[2]),
        "xy_m": float(np.linalg.norm(local_offset[:2])),
        "x_mm": float(local_offset[0] * 1000.0),
        "y_mm": float(local_offset[1] * 1000.0),
        "z_mm": float(local_offset[2] * 1000.0),
        "xy_mm": float(np.linalg.norm(local_offset[:2]) * 1000.0),
    }


def _quat_wxyz_from_transform(transform) -> tuple[float, float, float, float]:
    return (
        float(transform.rotation.w),
        float(transform.rotation.x),
        float(transform.rotation.y),
        float(transform.rotation.z),
    )


def _normalize_quat_wxyz(quat) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return values / norm


def _quat_to_rotvec_rad(quat_wxyz) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    if quat[0] < 0.0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(sin_half, w))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = xyz / sin_half
    return axis * angle


def plug_to_port_rotation_label(port_tf, plug_tf) -> dict[str, float | list[float]]:
    q_port = _quat_wxyz_from_transform(port_tf)
    q_plug = _quat_wxyz_from_transform(plug_tf)
    q_plug_inv = (q_plug[0], -q_plug[1], -q_plug[2], -q_plug[3])
    delta = _normalize_quat_wxyz(quaternion_multiply(q_port, q_plug_inv))
    rotvec = _quat_to_rotvec_rad(delta)
    angle = float(np.linalg.norm(rotvec))
    return {
        "coordinate": "base_link axis-angle rotation that maps plug orientation to target port orientation",
        "quat_wxyz": [float(v) for v in delta],
        "rx_rad": float(rotvec[0]),
        "ry_rad": float(rotvec[1]),
        "rz_rad": float(rotvec[2]),
        "angle_rad": angle,
        "angle_deg": float(np.degrees(angle)),
    }


def project_to_camera(point_3d_base: np.ndarray, k: np.ndarray, base_to_cam: np.ndarray):
    point_cam = base_to_cam @ np.append(point_3d_base, 1.0)
    x, y, z = point_cam[:3]
    if z < 1e-6:
        return None
    u = k[0, 0] * x / z + k[0, 2]
    v = k[1, 1] * y / z + k[1, 2]
    return float(u), float(v), float(z)


def make_bbox_from_points(points: np.ndarray, image_w: int, image_h: int, margin: float):
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    pad_x = (x_max - x_min) * margin
    pad_y = (y_max - y_min) * margin
    x_min = np.clip(x_min - pad_x, 0, image_w - 1)
    y_min = np.clip(y_min - pad_y, 0, image_h - 1)
    x_max = np.clip(x_max + pad_x, 0, image_w - 1)
    y_max = np.clip(y_max + pad_y, 0, image_h - 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    return (
        ((x_min + x_max) / 2.0) / image_w,
        ((y_min + y_max) / 2.0) / image_h,
        (x_max - x_min) / image_w,
        (y_max - y_min) / image_h,
    )


def port_corners_in_frame(port_size_m: tuple[float, float] = (0.014, 0.010)) -> np.ndarray:
    width, height = port_size_m
    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float64,
    )


def order_image_corners(points: np.ndarray) -> np.ndarray:
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmax(diffs)],
            points[np.argmax(sums)],
            points[np.argmin(diffs)],
        ],
        dtype=np.float64,
    )
