"""FinalPolicy에서 쓰는 쿼터니언, 보간, 카메라 투영 기하 유틸리티."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Quaternion
from transforms3d._gohlketransforms import quaternion_multiply


def quat_to_tuple(q: Quaternion) -> tuple:
    """ROS Quaternion(x, y, z, w)을 transforms3d가 쓰는 (w, x, y, z) 튜플로 바꾼다."""
    return (q.w, q.x, q.y, q.z)


def tuple_to_quat(q: tuple) -> Quaternion:
    """(w, x, y, z) 튜플을 ROS Quaternion 메시지로 변환한다."""
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def quat_inverse(q: tuple) -> tuple:
    """단위 쿼터니언의 역원을 반환한다."""
    return (q[0], -q[1], -q[2], -q[3])


def rotate_vector_by_quat(v: np.ndarray, q: tuple) -> np.ndarray:
    """3D 벡터를 쿼터니언 회전으로 변환한다."""
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    q_inv = quat_inverse(q)
    rotated = quaternion_multiply(quaternion_multiply(q, qv), q_inv)
    return np.array([rotated[1], rotated[2], rotated[3]])


def s_curve(t: float) -> float:
    """0~1 구간에서 시작/끝 속도가 부드러운 3차 S-curve 값을 계산한다."""
    return 3.0 * t * t - 2.0 * t * t * t


def s_curve_quintic(t: float) -> float:
    """0~1 구간에서 가속도까지 부드러운 5차 S-curve 값을 계산한다."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    """경로 보간에 사용할 진행률 프로파일을 선택한다."""
    return s_curve_quintic(t) if quintic else s_curve(t)


def project_3d_to_pixel(point_3d_base, k_matrix, t_camera_base):
    """base 좌표계의 3D 점을 T_camera_base와 K로 특정 카메라 픽셀에 투영한다."""
    point_h = np.append(point_3d_base, 1.0)
    point_cam = t_camera_base @ point_h
    x, y, z = point_cam[:3]
    if z < 1e-6:
        return -1.0, -1.0
    u = k_matrix[0, 0] * x / z + k_matrix[0, 2]
    v = k_matrix[1, 1] * y / z + k_matrix[1, 2]
    return float(u), float(v)
