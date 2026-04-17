import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class CheatCodePlanner:
    def __init__(self, i_gain: float = 0.15, max_integrator_windup: float = 0.05):
        self.i_gain = float(i_gain)
        self.max_integrator_windup = float(max_integrator_windup)
        self.reset()

    def reset(self) -> None:
        self.tip_x_error_integrator = 0.0
        self.tip_y_error_integrator = 0.0

    def build_pose(
        self,
        port_transform: Transform,
        plug_transform: Transform,
        gripper_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> tuple[Pose, dict[str, float]]:
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        q_plug = (
            plug_transform.rotation.w,
            plug_transform.rotation.x,
            plug_transform.rotation.y,
            plug_transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        q_gripper = (
            gripper_transform.rotation.w,
            gripper_transform.rotation.x,
            gripper_transform.rotation.y,
            gripper_transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_transform.translation.x,
            gripper_transform.translation.y,
            gripper_transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_transform.translation.x,
            plug_transform.translation.y,
            plug_transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self.reset()
        else:
            self.tip_x_error_integrator = float(
                np.clip(
                    self.tip_x_error_integrator + tip_x_error,
                    -self.max_integrator_windup,
                    self.max_integrator_windup,
                )
            )
            self.tip_y_error_integrator = float(
                np.clip(
                    self.tip_y_error_integrator + tip_y_error,
                    -self.max_integrator_windup,
                    self.max_integrator_windup,
                )
            )

        target_x = port_xy[0] + self.i_gain * self.tip_x_error_integrator
        target_y = port_xy[1] + self.i_gain * self.tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        pose = Pose(
            position=Point(
                x=float(blend_xyz[0]),
                y=float(blend_xyz[1]),
                z=float(blend_xyz[2]),
            ),
            orientation=Quaternion(
                w=float(q_gripper_slerp[0]),
                x=float(q_gripper_slerp[1]),
                y=float(q_gripper_slerp[2]),
                z=float(q_gripper_slerp[3]),
            ),
        )
        extras = {
            "tip_x_error": float(tip_x_error),
            "tip_y_error": float(tip_y_error),
            "tip_x_error_integrator": float(self.tip_x_error_integrator),
            "tip_y_error_integrator": float(self.tip_y_error_integrator),
            "target_x": float(target_x),
            "target_y": float(target_y),
            "target_z": float(target_z),
            "z_offset": float(z_offset),
            "slerp_fraction": float(slerp_fraction),
            "position_fraction": float(position_fraction),
        }
        return pose, extras
