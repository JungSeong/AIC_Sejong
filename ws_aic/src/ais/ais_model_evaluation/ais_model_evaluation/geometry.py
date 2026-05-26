from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3


SFP_TIP_TOP_CENTER_OFFSET_M = np.array([0.0, 0.0021125, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class TransformState:
    translation: np.ndarray
    rotation: np.ndarray

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_xyzw_to_matrix(self.rotation)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.translation
        return matrix


def normalize_quat_xyzw(quat: Any) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return values / norm


def quat_xyzw_to_matrix(quat: Any) -> np.ndarray:
    qx, qy, qz, qw = normalize_quat_xyzw(quat)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0) * 2.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return normalize_quat_xyzw([qx, qy, qz, qw])


def rpy_to_matrix(rpy: Any) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    sy = float(np.linalg.norm(rotation[:2, 0]))
    if sy >= 1e-9:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_angle_rad(matrix: np.ndarray) -> float:
    value = (float(np.trace(matrix)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def transform_state_from_tf(transform: Transform) -> TransformState:
    return TransformState(
        translation=np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float64,
        ),
        rotation=np.array(
            [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
            dtype=np.float64,
        ),
    )


def transform_state_from_pose(pose: Pose) -> TransformState:
    return TransformState(
        translation=np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
        rotation=np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=np.float64,
        ),
    )


def pose_from_state(state: TransformState) -> Pose:
    qx, qy, qz, qw = normalize_quat_xyzw(state.rotation)
    return Pose(
        position=Point(
            x=float(state.translation[0]),
            y=float(state.translation[1]),
            z=float(state.translation[2]),
        ),
        orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
    )


def shift_transform_origin(state: TransformState, local_offset_m: np.ndarray) -> TransformState:
    return TransformState(
        translation=state.translation + state.rotation_matrix @ np.asarray(local_offset_m, dtype=np.float64),
        rotation=state.rotation,
    )


def local_distance_label(port: TransformState, plug: TransformState) -> dict[str, float]:
    local = port.rotation_matrix.T @ (plug.translation - port.translation)
    return {
        "x_m": float(local[0]),
        "y_m": float(local[1]),
        "z_m": float(local[2]),
        "norm_m": float(np.linalg.norm(local)),
        "xy_m": float(np.linalg.norm(local[:2])),
        "x_mm": float(local[0] * 1000.0),
        "y_mm": float(local[1] * 1000.0),
        "z_mm": float(local[2] * 1000.0),
        "norm_mm": float(np.linalg.norm(local) * 1000.0),
        "xy_mm": float(np.linalg.norm(local[:2]) * 1000.0),
    }


def orientation_correction_label(port: TransformState, plug: TransformState) -> dict[str, Any]:
    delta = plug.rotation_matrix.T @ port.rotation_matrix
    rpy = matrix_to_rpy(delta)
    angle = rotation_angle_rad(delta)
    return {
        "roll_rad": float(rpy[0]),
        "pitch_rad": float(rpy[1]),
        "yaw_rad": float(rpy[2]),
        "roll_deg": float(np.degrees(rpy[0])),
        "pitch_deg": float(np.degrees(rpy[1])),
        "yaw_deg": float(np.degrees(rpy[2])),
        "angular_rad": float(np.linalg.norm(rpy)),
        "angular_deg": float(np.degrees(np.linalg.norm(rpy))),
        "geodesic_rad": angle,
        "geodesic_deg": float(np.degrees(angle)),
    }


def transform_inverse(matrix: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ matrix[:3, 3]
    return inv


def state_from_matrix(matrix: np.ndarray) -> TransformState:
    return TransformState(
        translation=np.asarray(matrix[:3, 3], dtype=np.float64),
        rotation=matrix_to_quat_xyzw(matrix[:3, :3]),
    )


def pose_dict(pose: Pose) -> dict[str, float]:
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
    }


def vector3_from_array(values: np.ndarray) -> Vector3:
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))
