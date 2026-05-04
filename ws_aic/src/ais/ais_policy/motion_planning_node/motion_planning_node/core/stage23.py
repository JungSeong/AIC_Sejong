"""Stage 2 alignment and Stage 3 insertion controller."""

from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Transform
from transforms3d._gohlketransforms import quaternion_multiply

from motion_planning_node.core.geometry import (
    quat_inverse,
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

    def _port_frame(self) -> str:
        return self._policy._port_frame()

    def _plug_frame(self) -> str:
        return self._policy._plug_frame()

    def _lookup_tf(self, frame: str) -> Optional[Transform]:
        return self._policy._lookup_tf(frame)

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return self._policy._transform_to_pose(tf)

    # ─────────────────────────────────────────────────────
    #  Stage 2/3: 임시 (기존과 동일)
    # ─────────────────────────────────────────────────────

    def _reset_integrator(self):
        self._x_integrator = 0.0
        self._y_integrator = 0.0

    def _compute_stage23_pose(self, z_offset, use_integrator=False,
                              port_pose_vision: Optional[Pose] = None):
        """Stage 2/3용 목표 pose.

        port_pose_vision이 제공되면 Vision 결과 사용 (ground_truth=false 환경).
        아니면 TF 기반 CheatCode 방식 (ground_truth=true 환경).
        """
        gripper_tf = self._lookup_tf("gripper/tcp")
        if gripper_tf is None:
            return None

        # Vision 모드라면 TF lookup 건너뛰기 (무한 재시도 방지)
        if port_pose_vision is not None:
            port_tf = None
            plug_tf = None
        else:
            port_tf = self._lookup_tf(self._port_frame())
            plug_tf = self._lookup_tf(self._plug_frame())

        if port_tf is not None and plug_tf is not None:
            # ── TF 경로 (ground_truth=true) ──
            q_port = quat_to_tuple(self._transform_to_pose(port_tf).orientation)
            q_plug = quat_to_tuple(self._transform_to_pose(plug_tf).orientation)
            q_gripper = quat_to_tuple(
                self._transform_to_pose(gripper_tf).orientation
            )
            q_diff = quaternion_multiply(q_port, quat_inverse(q_plug))
            q_target = quaternion_multiply(q_diff, q_gripper)

            offset_z = gripper_tf.translation.z - plug_tf.translation.z
            tip_x_err = port_tf.translation.x - plug_tf.translation.x
            tip_y_err = port_tf.translation.y - plug_tf.translation.y

            if use_integrator:
                self._x_integrator = np.clip(
                    self._x_integrator + tip_x_err,
                    -self._max_windup, self._max_windup,
                )
                self._y_integrator = np.clip(
                    self._y_integrator + tip_y_err,
                    -self._max_windup, self._max_windup,
                )
                tx = port_tf.translation.x + self._i_gain * self._x_integrator
                ty = port_tf.translation.y + self._i_gain * self._y_integrator
            else:
                tx = port_tf.translation.x
                ty = port_tf.translation.y
            tz = port_tf.translation.z + z_offset + offset_z

            return Pose(
                position=Point(x=float(tx), y=float(ty), z=float(tz)),
                orientation=tuple_to_quat(q_target),
            )

        elif port_pose_vision is not None:
            # ── Vision 경로 (ground_truth=false) ──
            # 플러그 TF 없음 → 대략적 오프셋으로 그리퍼 목표 생성
            q_gripper = quat_to_tuple(
                self._transform_to_pose(gripper_tf).orientation
            )
            # 포트 위치 + z_offset + 그리퍼-플러그 대략 오프셋
            tx = port_pose_vision.position.x
            ty = port_pose_vision.position.y + 0.015  # 플러그-그리퍼 y 오프셋
            tz = port_pose_vision.position.z + z_offset + 0.045  # z 오프셋

            return Pose(
                position=Point(x=float(tx), y=float(ty), z=float(tz)),
                orientation=tuple_to_quat(q_gripper),
            )

        return None

    def align(self, move_robot, send_feedback,
              port_pose_vision=None):
        self.get_logger().info("━━━ Stage 2: 정렬 시작 ━━━")
        send_feedback("Stage 2: aligning to port")
        self._reset_integrator()

        Z_START, Z_END, N = 0.10, 0.005, 100
        for i in range(N):
            t = (i + 1) / N
            z = Z_START + t * (Z_END - Z_START)
            pose = self._compute_stage23_pose(
                z_offset=z, use_integrator=True,
                port_pose_vision=port_pose_vision,
            )
            if pose is not None:
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
