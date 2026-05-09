"""Small numeric helpers for ``ais_transform``."""

from __future__ import annotations

from typing import Sequence

import numpy as np


VectorLike = Sequence[float] | np.ndarray


def _vec3(value: VectorLike) -> np.ndarray:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"xyz는 shape (3,)이어야 합니다. 현재 shape={vec.shape}")
    return vec


def _quat_xyzw(value: VectorLike) -> np.ndarray:
    quat = np.asarray(value, dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"quaternion은 xyzw shape (4,)이어야 합니다. 현재 shape={quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm은 0이면 안 됩니다.")
    return quat / norm


def _quat_multiply_xyzw(left: VectorLike, right: VectorLike) -> np.ndarray:
    """xyzw quaternion 곱셈. parent 회전 뒤 child 회전을 적용할 때 사용."""

    lx, ly, lz, lw = _quat_xyzw(left)
    rx, ry, rz, rw = _quat_xyzw(right)
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=float,
    )


def _rotation_matrix_xyzw(quat: VectorLike) -> np.ndarray:
    """xyzw quaternion을 3x3 회전행렬로 변환."""

    x, y, z, w = _quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


__all__ = [
    "VectorLike",
    "_quat_multiply_xyzw",
    "_quat_xyzw",
    "_rotation_matrix_xyzw",
    "_vec3",
]
