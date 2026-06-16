from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Quaternion
from transforms3d._gohlketransforms import quaternion_multiply


def quat_to_tuple(q: Quaternion) -> tuple:
    return (q.w, q.x, q.y, q.z)


def tuple_to_quat(q: tuple) -> Quaternion:
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def quat_inverse(q: tuple) -> tuple:
    return (q[0], -q[1], -q[2], -q[3])


def rotate_vector_by_quat(v: np.ndarray, q: tuple) -> np.ndarray:
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    q_inv = quat_inverse(q)
    rotated = quaternion_multiply(quaternion_multiply(q, qv), q_inv)
    return np.array([rotated[1], rotated[2], rotated[3]])


def s_curve(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def s_curve_quintic(t: float) -> float:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    return s_curve_quintic(t) if quintic else s_curve(t)


def project_3d_to_pixel(point_3d_base, k_matrix, t_base_to_cam):
    point_h = np.append(point_3d_base, 1.0)
    point_cam = t_base_to_cam @ point_h
    x, y, z = point_cam[:3]
    if z < 1e-6:
        return -1.0, -1.0
    u = k_matrix[0, 0] * x / z + k_matrix[0, 2]
    v = k_matrix[1, 1] * y / z + k_matrix[1, 2]
    return float(u), float(v)
