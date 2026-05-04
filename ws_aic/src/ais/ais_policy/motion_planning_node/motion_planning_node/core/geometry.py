"""Geometry helpers for staged motion planning."""

import numpy as np
from geometry_msgs.msg import Quaternion
from transforms3d._gohlketransforms import quaternion_multiply


# ═══════════════════════════════════════════════════════════
#  쿼터니언 / 벡터 유틸리티
# ═══════════════════════════════════════════════════════════

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
    """3차 Hermite: 시작/끝 속도 0."""
    return 3.0 * t * t - 2.0 * t * t * t


def s_curve_quintic(t: float) -> float:
    """5차 Hermite: 시작/끝에서 속도 + 가속도 모두 0.

    식: 10t³ - 15t⁴ + 6t⁵
    특성:
      t=0: pos=0, vel=0, acc=0
      t=1: pos=1, vel=0, acc=0
    → 가속도 연속 → 관성 떨림 최소화
    """
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def interp_profile(t: float, quintic: bool = True) -> float:
    """보간 프로파일 선택."""
    return s_curve_quintic(t) if quintic else s_curve(t)


def _project_3d_to_pixel(point_3d_base, K, T_base_to_cam):
    """3D base 좌표를 카메라 이미지 픽셀로 투영."""
    p_homo = np.append(point_3d_base, 1.0)
    p_cam = T_base_to_cam @ p_homo
    x, y, z = p_cam[:3]
    if z < 1e-6:
        return -1.0, -1.0
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def transform_to_matrix(t) -> np.ndarray:
    """geometry_msgs/Transform → 4x4 matrix."""
    tx, ty, tz = t.translation.x, t.translation.y, t.translation.z
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1 - 2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1 - 2*(xx+yy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


