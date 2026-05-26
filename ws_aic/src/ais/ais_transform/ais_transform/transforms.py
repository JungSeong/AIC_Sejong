"""Coordinate transforms for AIC plug/port tasks.

원칙:
- 모든 입력/출력 pose는 먼저 ``base_link`` 기준으로 맞춘다.
- plug->port 이동량도 기본적으로 ``base_link`` 기준 벡터로 계산한다.
- YOLO/삼각측량 경로는 현재 camera->port가 없으므로,
  "port 상단 중앙점 -> 실제 port 원점" 보정만 검증한다.
"""

from __future__ import annotations

from ._math import VectorLike, _quat_multiply_xyzw, _quat_xyzw, _rotation_matrix_xyzw, _vec3
from .poses import PlugToPort, Pose3D


def pose_in_baselink(
    parent_pose_in_baselink: Pose3D,
    child_position_in_parent: VectorLike,
    child_orientation_in_parent: VectorLike = (0.0, 0.0, 0.0, 1.0),
) -> Pose3D:
    """parent 기준 child pose를 ``base_link`` 기준 pose로 변환.

    TF의 parent->child fixed transform을 base_link로 올릴 때 쓰는 공통 함수다.
    plug, port, camera 모두 이 규칙으로 base_link 기준 pose가 될 수 있다.
    """

    child_pos = _vec3(child_position_in_parent)
    child_quat = _quat_xyzw(child_orientation_in_parent)
    base_pos = parent_pose_in_baselink.position + parent_pose_in_baselink.rotation_matrix @ child_pos
    base_quat = _quat_multiply_xyzw(parent_pose_in_baselink.orientation, child_quat)
    return Pose3D.from_xyz_quat(base_pos, base_quat)


def plug_in_baselink_from_gripper(
    gripper_pose_in_baselink: Pose3D,
    plug_tip_gripper_offset_in_baselink: VectorLike,
    plug_orientation_in_baselink: VectorLike | None = None,
) -> Pose3D:
    """cheatcode 관례로 plug pose를 ``base_link`` 기준으로 계산.

    cheatcode.py는 ``plug_tip_gripper_offset = gripper_xyz - plug_xyz``를
    base_link 기준 벡터로 저장한다. 따라서 plug 위치는
    ``gripper_xyz - plug_tip_gripper_offset``이다.
    """

    orientation = (
        gripper_pose_in_baselink.orientation
        if plug_orientation_in_baselink is None
        else plug_orientation_in_baselink
    )
    return Pose3D.from_xyz_quat(
        gripper_pose_in_baselink.position - _vec3(plug_tip_gripper_offset_in_baselink),
        orientation,
    )


def port_in_baselink(
    parent_pose_in_baselink: Pose3D,
    port_position_in_parent: VectorLike,
    port_orientation_in_parent: VectorLike = (0.0, 0.0, 0.0, 1.0),
) -> Pose3D:
    """mount/module 기준 port pose를 ``base_link`` 기준으로 변환."""

    return pose_in_baselink(
        parent_pose_in_baselink,
        port_position_in_parent,
        port_orientation_in_parent,
    )


def camera_in_baselink(
    parent_pose_in_baselink: Pose3D,
    camera_position_in_parent: VectorLike,
    camera_orientation_in_parent: VectorLike = (0.0, 0.0, 0.0, 1.0),
) -> Pose3D:
    """parent 기준 camera pose를 ``base_link`` 기준으로 변환."""

    return pose_in_baselink(
        parent_pose_in_baselink,
        camera_position_in_parent,
        camera_orientation_in_parent,
    )


def port_in_baselink_from_top_center(
    port_top_center_position_in_baselink: VectorLike,
    port_orientation_in_baselink: VectorLike,
    top_center_to_port_position_in_port: VectorLike,
) -> Pose3D:
    """port 상단 중앙점에서 실제 port 원점을 ``base_link`` 기준으로 계산.

    YOLO bbox 3D position이 port 상단 중앙점을 의미한다고 보고,
    그 점에서 실제 port 기준점까지의 offset을 port frame 기준으로 넣는다.
    """

    port_quat = _quat_xyzw(port_orientation_in_baselink)
    port_pos = _vec3(port_top_center_position_in_baselink) + (
        _rotation_matrix_xyzw(port_quat) @ _vec3(top_center_to_port_position_in_port)
    )
    return Pose3D.from_xyz_quat(port_pos, port_quat)


def plug_to_port_baselink(plug_pose_in_baselink: Pose3D, port_pose_in_baselink: Pose3D) -> PlugToPort:
    """plug가 port에 맞기 위해 이동해야 할 ``base_link`` 기준 벡터 계산."""

    return PlugToPort(port_pose_in_baselink.position - plug_pose_in_baselink.position)


def plug_to_port_from_top_center_baselink(
    plug_pose_in_baselink: Pose3D,
    port_top_center_position_in_baselink: VectorLike,
    port_orientation_in_baselink: VectorLike,
    top_center_to_port_position_in_port: VectorLike,
) -> tuple[Pose3D, PlugToPort]:
    """삼각측량 경로의 현재 검증 대상 계산.

    아직 camera->port 상단 중앙점 계산은 없으므로, 이미 base_link 기준으로
    얻은 port 상단 중앙점에서 실제 port pose를 복원하고 plug->port를 계산한다.
    """

    port_pose = port_in_baselink_from_top_center(
        port_top_center_position_in_baselink,
        port_orientation_in_baselink,
        top_center_to_port_position_in_port,
    )
    return port_pose, plug_to_port_baselink(plug_pose_in_baselink, port_pose)


__all__ = [
    "camera_in_baselink",
    "plug_in_baselink_from_gripper",
    "plug_to_port_baselink",
    "plug_to_port_from_top_center_baselink",
    "port_in_baselink",
    "port_in_baselink_from_top_center",
    "pose_in_baselink",
]
