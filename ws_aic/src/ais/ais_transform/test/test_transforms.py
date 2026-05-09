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


def _quat_z_90():
    half = math.pi / 4.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def test_plug_in_baselink_from_gripper_matches_cheatcode_offset():
    gripper = Pose3D.from_xyz_quat([1.0, 2.0, 3.0], _quat_z_90())

    plug = plug_in_baselink_from_gripper(gripper, [0.1, -0.2, 0.3])

    np.testing.assert_allclose(plug.position, [0.9, 2.2, 2.7], atol=1e-9)


def test_port_and_camera_can_be_converted_to_baselink():
    parent = Pose3D.from_xyz_quat([0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0])

    port = port_in_baselink(parent, [0.01, -0.02, 0.03])
    camera = camera_in_baselink(parent, [-0.1, 0.2, 0.3])

    np.testing.assert_allclose(port.position, [0.21, 0.28, 0.43], atol=1e-9)
    np.testing.assert_allclose(camera.position, [0.1, 0.5, 0.7], atol=1e-9)


def test_pose_has_transform_matrix():
    pose = Pose3D.from_xyz_quat([1.0, 2.0, 3.0], _quat_z_90())

    np.testing.assert_allclose(pose.position, [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(
        pose.transform_matrix,
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        atol=1e-9,
    )


def test_plug_to_port_baselink_is_port_minus_plug():
    plug = Pose3D.from_xyz_quat([0.9, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    port = Pose3D.from_xyz_quat([1.0, 2.2, 2.7], _quat_z_90())

    delta = plug_to_port_baselink(plug, port)

    np.testing.assert_allclose(delta.vector, [0.1, 0.2, -0.3], atol=1e-9)


def test_port_in_baselink_from_top_center_applies_port_frame_offset():
    top_center = [1.0, 2.0, 3.0]
    top_center_to_port = [0.02, 0.0, -0.01]

    port = port_in_baselink_from_top_center(top_center, _quat_z_90(), top_center_to_port)

    np.testing.assert_allclose(port.position, [1.0, 2.02, 2.99], atol=1e-9)


def test_plug_to_port_from_top_center_baselink_returns_recovered_port_and_delta():
    plug = Pose3D.from_xyz_quat([0.9, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])

    port, delta = plug_to_port_from_top_center_baselink(
        plug,
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(port.position, [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(delta.vector, [0.1, 0.0, 0.0], atol=1e-9)
