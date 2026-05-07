"""Stage 2 alignment and Stage 3 insertion controller."""

from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Transform, Vector3

from motion_planning_node.core.geometry import (
    quat_to_tuple,
    tuple_to_quat,
)


class Stage23Controller:
    """Run the temporary rule-based alignment and insertion stages."""

    def __init__(self, policy):
        self._policy = policy
        self._x_integrator = 0.0
        self._y_integrator = 0.0
        self._max_windup = 0.05
        self._i_gain = 0.15

    @property
    def _task(self):
        return self._policy._task

    def get_logger(self):
        return self._policy.get_logger()

    def sleep_for(self, duration_sec: float) -> None:
        self._policy.sleep_for(duration_sec)

    def set_pose_target(self, *args, **kwargs):
        return self._policy.set_pose_target(*args, **kwargs)

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return self._policy._transform_to_pose(tf)

    @staticmethod
    def _tcp_transform_from_obs(obs) -> Optional[Transform]:
        if obs is None:
            return None
        pose = getattr(getattr(obs, "controller_state", None), "tcp_pose", None)
        if pose is None:
            return None
        return Transform(
            translation=Vector3(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z,
            ),
            rotation=pose.orientation,
        )

    # ─────────────────────────────────────────────────────
    #  Stage 2/3: 임시 (기존과 동일)
    # ─────────────────────────────────────────────────────

    def _reset_integrator(self):
        self._x_integrator = 0.0
        self._y_integrator = 0.0

    def _compute_stage23_pose(self, z_offset, use_integrator=False,
                              obs=None,
                              port_pose_vision: Optional[Pose] = None):
        """Stage 2/3용 목표 pose.

        포트 좌표는 Stage 1의 Vision 결과만 사용한다.
        """
        gripper_tf = self._tcp_transform_from_obs(obs)
        if gripper_tf is None or port_pose_vision is None:
            return None

        q_gripper = quat_to_tuple(
            self._transform_to_pose(gripper_tf).orientation
        )
        # 포트 위치 + z_offset + 그리퍼-플러그 대략 오프셋
        tx = port_pose_vision.position.x
        ty = port_pose_vision.position.y + 0.015
        tz = port_pose_vision.position.z + z_offset + 0.045

        return Pose(
            position=Point(x=float(tx), y=float(ty), z=float(tz)),
            orientation=tuple_to_quat(q_gripper),
        )

    def align(self, move_robot, send_feedback,
              port_pose_vision=None, get_observation=None):
        self.get_logger().info("━━━ Stage 2: Vision offset 정렬 시작 ━━━")
        send_feedback("Stage 2: aligning to port with vision offset")
        self._reset_integrator()

        Z_START, Z_END, N = 0.10, 0.005, 100
        for i in range(N):
            t = (i + 1) / N
            z = Z_START + t * (Z_END - Z_START)
            pose = self._compute_stage23_pose(
                z_offset=z, use_integrator=True,
                obs=get_observation() if get_observation is not None else None,
                port_pose_vision=port_pose_vision,
            )
            if pose is not None:
                pred_offset = None
                if get_observation is not None and hasattr(self._policy, "predict_distance_offset"):
                    pred_offset = self._policy.predict_distance_offset(get_observation())
                if pred_offset is not None:
                    dx = float(np.clip(-pred_offset[0], -0.004, 0.004))
                    dy = float(np.clip(-pred_offset[1], -0.004, 0.004))
                    pose.position.x = float(pose.position.x + dx)
                    pose.position.y = float(pose.position.y + dy)
                    self.get_logger().info(
                        f"Stage 2 vision offset xyz="
                        f"{pred_offset[0]:+.4f},{pred_offset[1]:+.4f},{pred_offset[2]:+.4f}m "
                        f"cmd_xy={dx:+.4f},{dy:+.4f}"
                    )
                    if float(np.linalg.norm(pred_offset[:2])) < 0.002:
                        self.set_pose_target(move_robot=move_robot, pose=pose)
                        break
                self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.05)

        self.get_logger().info("━━━ Stage 2: 정렬 완료 ━━━")

    def insert(self, get_observation, move_robot, send_feedback,
               port_pose_vision=None):
        self.get_logger().info("━━━ Stage 3: 삽입 시작 ━━━")
        send_feedback("Stage 3: inserting cable")

        FORCE_LIMIT = 18.0
        z = 0.005
        stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

        while z > -0.015:
            z -= 0.0005
            obs = get_observation()
            if obs is not None:
                f = obs.wrist_wrench.wrench.force
                if np.sqrt(f.x**2 + f.y**2 + f.z**2) > FORCE_LIMIT:
                    self.sleep_for(0.2)
                    continue

            pose = self._compute_stage23_pose(
                z_offset=z, use_integrator=True,
                obs=obs,
                port_pose_vision=port_pose_vision,
            )
            if pose is not None:
                self.set_pose_target(
                    move_robot=move_robot, pose=pose,
                    stiffness=stiffness, damping=damping,
                )
            self.sleep_for(0.05)

        self.sleep_for(3.0)
        self.get_logger().info("━━━ Stage 3: 삽입 완료 ━━━")
