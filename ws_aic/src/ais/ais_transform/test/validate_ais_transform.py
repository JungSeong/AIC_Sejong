#!/usr/bin/env python3
"""ais_transform 유효성 검사.

검증 항목:
1. plug, port, camera를 base_link 기준으로 변환할 수 있는지 확인
2. base_link 기준 plug->port 이동량을 계산하는지 확인
3. 삼각측량 경로는 아직 camera->port 상단 중앙점이 없으므로,
   port 상단 중앙점에서 실제 port 원점으로 보정되는지만 확인
"""

from __future__ import annotations

import math

import numpy as np

from ais_transform import (
    Pose3D,
    camera_in_baselink,
    plug_in_baselink_from_gripper,
    plug_to_port_baselink,
    plug_to_port_from_top_center_baselink,
    port_in_baselink,
    port_in_baselink_from_top_center,
)


def _assert_close(name: str, actual, expected, atol: float = 1e-9) -> None:
    np.testing.assert_allclose(actual, expected, atol=atol)
    print(f"[OK] {name}: {np.asarray(actual)}")


def _quat_z(rad: float) -> np.ndarray:
    half = rad / 2.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=float)


def _rot_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def main() -> None:
    # 1-1. plug: cheatcode 관례(gripper_xyz - plug_xyz)를 base_link 기준으로 확인한다.
    gripper_base = Pose3D.from_xyz_quat([0.45, -0.12, 0.31], _quat_z(math.radians(15)))
    gripper_minus_plug = np.array([0.0, 0.015385, 0.04245])
    plug_base = plug_in_baselink_from_gripper(gripper_base, gripper_minus_plug)
    _assert_close("plug in base_link", plug_base.position, [0.45, -0.135385, 0.26755])

    # 1-2. port: parent module/mount pose와 local port offset을 base_link 기준으로 올린다.
    mount_base = Pose3D.from_xyz_quat([0.2, -0.1, 0.4], _quat_z(math.radians(90)))
    port_local = np.array([0.01295, -0.031572, 0.00501])
    port_base = port_in_baselink(mount_base, port_local)
    expected_port = mount_base.position + _rot_xyzw(mount_base.orientation) @ port_local
    _assert_close("port in base_link", port_base.position, expected_port)

    # 1-3. camera: camera도 같은 parent->child 규칙으로 base_link 기준 pose가 된다.
    camera_local = np.array([-0.03, 0.04, 0.12])
    camera_base = camera_in_baselink(mount_base, camera_local)
    expected_camera = mount_base.position + _rot_xyzw(mount_base.orientation) @ camera_local
    _assert_close("camera in base_link", camera_base.position, expected_camera)

    # 2. base_link 기반 plug->port: port_xyz - plug_xyz가 곧 이동 벡터다.
    delta_base = plug_to_port_baselink(plug_base, port_base)
    _assert_close("plug to port in base_link", delta_base.vector, port_base.position - plug_base.position)

    # 3. 삼각측량 검증 대체:
    #    camera->port 상단 중앙은 아직 없으므로, 이미 base_link 기준인 상단 중앙점에서
    #    실제 port 원점이 정확히 복원되는지만 확인한다.
    top_center_to_port = np.array([0.003, -0.004, -0.012])
    true_port = Pose3D.from_xyz_quat([0.31, -0.08, 0.22], _quat_z(math.radians(-30)))
    top_center_base = true_port.position - _rot_xyzw(true_port.orientation) @ top_center_to_port
    recovered_port = port_in_baselink_from_top_center(
        top_center_base,
        true_port.orientation,
        top_center_to_port,
    )
    _assert_close("port from top center in base_link", recovered_port.position, true_port.position)

    # 4. 삼각측량 경로의 plug->port도 최종적으로 base_link 기준 벡터로 계산한다.
    recovered_port, triangulated_delta = plug_to_port_from_top_center_baselink(
        plug_base,
        top_center_base,
        true_port.orientation,
        top_center_to_port,
    )
    _assert_close("triangulated port in base_link", recovered_port.position, true_port.position)
    _assert_close(
        "plug to triangulated port in base_link",
        triangulated_delta.vector,
        true_port.position - plug_base.position,
    )

    print("ais_transform validation passed.")


if __name__ == "__main__":
    main()
