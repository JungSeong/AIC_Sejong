"""Stage 1 approach controller for the staged policy."""

from typing import Optional

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
from transforms3d._gohlketransforms import quaternion_slerp

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.geometry import (
    interp_profile,
    quat_to_tuple,
    tuple_to_quat,
)
from motion_planning_node.core.stage1_types import Stage1Result


class Stage1Approach:
    """Find the target port and move the TCP to the pre-insertion pose."""

    def __init__(self, policy, vision):
        self._policy = policy
        self._vision = vision

    @property
    def _task(self):
        return self._policy._task

    def get_logger(self):
        return self._policy.get_logger()

    def time_now(self):
        return self._policy.time_now()

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
    #  포트 pose 획득: Vision only
    # ─────────────────────────────────────────────────────

    def _get_port_pose(self, get_observation) -> tuple:
        """(port_pose, port_source) 반환.

        port_source:
          "vision": YOLO + 스테레오
          None: 실패
        """
        obs = get_observation()
        if obs is None:
            return None, None

        # 타겟 클래스 결정 (task.plug_name으로 판단)
        if "sc" in self._task.plug_name.lower():
            target_class_id = 1  # sc_port
        else:
            target_class_id = 0  # sfp_port

        # task.port_name으로 올바른 후보 선택 (SFP 포트 0/1 구분)
        # 예: "sfp_port_0", "sfp_port_1", "sc_port_base"
        port_name_hint = self._task.port_name or ""
        port_3d = self._vision.estimate(
            obs, target_class_id,
            port_hint=port_name_hint,
        )
        if port_3d is None:
            return None, None

        self.get_logger().info(
            f"Vision 선택 결과 (port_hint='{port_name_hint}'): "
            f"({port_3d[0]:+.3f}, {port_3d[1]:+.3f}, {port_3d[2]:+.3f})"
        )

        # Vision은 위치만 주고, 방향은 추정 안 함 → 단위 쿼터니언 사용
        # (월드 +z 접근 방식을 쓰므로 방향 정보가 덜 중요)
        pose = Pose(
            position=Point(
                x=float(port_3d[0]), y=float(port_3d[1]), z=float(port_3d[2]),
            ),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return pose, "vision"

    def _wait_for_port_pose(self, get_observation, target_class_id: int, port_name_hint: str) -> tuple:
        """DataCollect2처럼 YOLO 결과가 준비될 때까지 observation을 계속 공급한다."""
        start = self.time_now()
        timeout = Stage1Config.VISION_ACQUIRE_TIMEOUT_S
        attempt = 0
        while (self.time_now() - start).nanoseconds / 1e9 < timeout:
            obs = get_observation()
            port_3d = self._estimate_point(
                obs,
                target_class_id,
                port_hint=port_name_hint,
            )
            if port_3d is not None:
                self.get_logger().info(
                    f"Vision 선택 결과 (port_hint='{port_name_hint}'): "
                    f"({port_3d[0]:+.3f}, {port_3d[1]:+.3f}, {port_3d[2]:+.3f})"
                )
                return self._pose_from_point(port_3d), "vision"
            if attempt % 5 == 0:
                self.get_logger().warn(
                    f"Vision: 포트 검출 대기 중 "
                    f"(target_class_id={target_class_id}, attempt={attempt})"
                )
            attempt += 1
            self.sleep_for(Stage1Config.VISION_ACQUIRE_DT)
        return None, None

    def _target_class_ids(self) -> tuple[int, int, bool]:
        plug_name = (self._task.plug_name or "").lower()
        port_type = (getattr(self._task, "port_type", "") or "").lower()
        is_sc = "sc" in plug_name or "sc" in port_type
        if is_sc:
            return 1, 3, False
        return 0, 2, True

    @staticmethod
    def _pose_from_point(point_3d: np.ndarray) -> Pose:
        return Pose(
            position=Point(
                x=float(point_3d[0]),
                y=float(point_3d[1]),
                z=float(point_3d[2]),
            ),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def _estimate_point(self, obs, target_class_id: int, port_hint: str = "") -> Optional[np.ndarray]:
        if obs is None:
            return None
        return self._vision.estimate(
            obs,
            target_class_id,
            port_hint=port_hint if port_hint else None,
        )

    def _measure_tip_to_port_offsets(
        self,
        tip_3d: Optional[np.ndarray],
        port_3d: Optional[np.ndarray],
        port_axis: np.ndarray,
    ) -> tuple[Optional[dict], dict]:
        extras = {
            "triangulated_tip_to_port_offsets_valid": False,
            "triangulated_z_offset_valid": False,
            "triangulated_xy_offset_valid": False,
            "triangulated_z_stop_threshold": float(Stage1Config.TRIANGULATION_STOP_Z_OFFSET),
            "triangulated_x_stop_threshold": float(Stage1Config.TRIANGULATION_STOP_X_OFFSET),
            "triangulated_y_stop_threshold": float(Stage1Config.TRIANGULATION_STOP_Y_OFFSET),
        }
        if tip_3d is None or port_3d is None:
            return None, extras

        axis = np.asarray(port_axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return None, extras
        axis /= axis_norm
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        for basis in (x_axis, y_axis):
            basis -= axis * float(np.dot(basis, axis))
            basis_norm = float(np.linalg.norm(basis))
            if basis_norm > 1e-9:
                basis /= basis_norm

        delta = np.asarray(tip_3d, dtype=float) - np.asarray(port_3d, dtype=float)
        x_offset = float(np.dot(delta, x_axis))
        y_offset = float(np.dot(delta, y_axis))
        z_offset = float(np.dot(delta, axis))
        xy_offset = float(np.linalg.norm([x_offset, y_offset]))
        within_threshold = (
            abs(x_offset) <= Stage1Config.TRIANGULATION_STOP_X_OFFSET
            and abs(y_offset) <= Stage1Config.TRIANGULATION_STOP_Y_OFFSET
            and abs(z_offset) <= Stage1Config.TRIANGULATION_STOP_Z_OFFSET
        )
        offsets = {
            "x": x_offset,
            "y": y_offset,
            "z": z_offset,
            "xy": xy_offset,
            "within_threshold": within_threshold,
        }
        extras.update({
            "triangulated_tip_to_port_offsets_valid": True,
            "triangulated_z_offset_valid": abs(z_offset) <= Stage1Config.TRIANGULATION_STOP_Z_OFFSET,
            "triangulated_xy_offset_valid": (
                abs(x_offset) <= Stage1Config.TRIANGULATION_STOP_X_OFFSET
                and abs(y_offset) <= Stage1Config.TRIANGULATION_STOP_Y_OFFSET
            ),
            "triangulated_x_offset": x_offset,
            "triangulated_y_offset": y_offset,
            "triangulated_z_offset": z_offset,
            "triangulated_xy_offset": xy_offset,
            "triangulated_within_threshold": within_threshold,
        })
        return offsets, extras

    # ─────────────────────────────────────────────────────
    #  Stage 1: 이동 (Vision 통합)
    # ─────────────────────────────────────────────────────

    def _compute_approach_pose(
        self, port_pose: Pose, gripper_tf: Transform,
        z_offset: Optional[float] = None,
    ) -> tuple[Pose, np.ndarray]:
        """접근점 pose 계산. (approach_pose, port_axis_world)

        Vision-only 기본 플러그-그리퍼 오프셋을 사용한다.
        """
        # 접근 축 (월드 +z 방향 기본)
        port_axis_world = np.array([0.0, 0.0, 1.0])

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        target_z_offset = (
            Stage1Config.APPROACH_Z_OFFSET_SC if z_offset is None else z_offset
        )
        approach_pos = port_pos + port_axis_world * target_z_offset

        gripper_target_pos = approach_pos + np.array([0.0, 0.015, 0.045])

        # 방향
        # Vision 모드: 현재 그리퍼 방향 유지 (안전)
        q_target = quat_to_tuple(
            self._transform_to_pose(gripper_tf).orientation
        )

        approach_pose = Pose(
            position=Point(
                x=float(gripper_target_pos[0]),
                y=float(gripper_target_pos[1]),
                z=float(gripper_target_pos[2]),
            ),
            orientation=tuple_to_quat(q_target),
        )
        return approach_pose, port_axis_world

    def _check_stage1_termination(
        self,
        gripper_tf: Transform,
        port_axis: np.ndarray,
        port_pose: Pose,
        target_z: Optional[float] = None,
    ) -> tuple[bool, str]:
        """target_z 기준으로 Stage 1 접근 완료 여부를 검증한다."""
        if target_z is None:
            target_z = Stage1Config.APPROACH_Z_OFFSET_SC

        ref = np.array([
            gripper_tf.translation.x,
            gripper_tf.translation.y,
            gripper_tf.translation.z,
        ])
        ref = ref - np.array([0.0, 0.015, 0.045])
        ref_name = "gripper(offset)"

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        to_port = ref - port_pos
        axial = float(np.dot(to_port, port_axis))
        axial_err = abs(axial - target_z)
        radial = float(np.linalg.norm(to_port - axial * port_axis))

        info = (
            f"{ref_name} axial={axial*100:.1f}cm "
            f"(target {target_z*100:.0f}, err {axial_err*100:.1f}), "
            f"radial={radial*100:.1f}cm"
        )

        if axial_err > Stage1Config.Z_OFFSET_TOLERANCE:
            return False, f"axial_err too large: {info}"
        if radial > Stage1Config.XY_TOLERANCE:
            return False, f"radial_err too large: {info}"
        return True, info

    def run(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> Stage1Result:
        self.get_logger().info("━━━ Stage 1: 이동 시작 ━━━")
        send_feedback("Stage 1: approaching port axis")

        # 0. F/T baseline (포트 획득 전에 측정)
        baseline_force_mag = 0.0
        init_obs = get_observation()
        if init_obs is not None:
            f = init_obs.wrist_wrench.wrench.force
            baseline_force_mag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
            self.get_logger().info(
                f"  F/T baseline: {baseline_force_mag:.2f}N"
            )

        port_class_id, tip_class_id, is_sfp = self._target_class_ids()
        port_name_hint = self._task.port_name or ""

        # 1. 포트 pose 획득 (Vision only) — DataCollect2처럼 준비될 때까지 대기
        port_pose, port_source = self._wait_for_port_pose(
            get_observation,
            port_class_id,
            port_name_hint,
        )
        if port_pose is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=None, port_axis=None,
                elapsed_time=0.0,
                failure_reason="포트 좌표 획득 실패 (Vision)",
                port_source="none",
            )

        self.get_logger().info(
            f"  포트 좌표 소스: {port_source}\n"
            f"  포트 위치: ({port_pose.position.x:+.3f}, "
            f"{port_pose.position.y:+.3f}, {port_pose.position.z:+.3f})"
        )

        # ★ 시간 측정은 실제 이동 시작 전에 개시
        t0 = self.time_now()

        # 2. 그리퍼 pose는 observation의 controller_state.tcp_pose 사용.
        obs_for_tcp = get_observation()
        gripper_tf = self._tcp_transform_from_obs(obs_for_tcp)
        if gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=None, elapsed_time=0.0,
                failure_reason="gripper tcp pose 관측 실패",
                port_source=port_source,
            )

        # 3. 접근점 계산. DataCollect2와 동일하게 포트 타입별 단일 z_offset에서 시작한다.
        initial_approach_z = (
            Stage1Config.APPROACH_Z_OFFSET_SFP
            if is_sfp
            else Stage1Config.APPROACH_Z_OFFSET_SC
        )
        approach_pose, port_axis = self._compute_approach_pose(
            port_pose, gripper_tf, z_offset=initial_approach_z
        )
        self.get_logger().info(
            f"  접근점: ({approach_pose.position.x:+.3f}, "
            f"{approach_pose.position.y:+.3f}, {approach_pose.position.z:+.3f}) "
            f"z_offset={initial_approach_z:.3f}m"
        )

        # 4. S-curve 직선 보간
        start_pose = self._transform_to_pose(gripper_tf)
        q_start = quat_to_tuple(start_pose.orientation)
        q_end = quat_to_tuple(approach_pose.orientation)
        p_start = np.array([
            start_pose.position.x, start_pose.position.y, start_pose.position.z
        ])
        reached_triangulation_stop = False
        last_measured_offsets = None
        yolo_tracking_started = True

        for i in range(Stage1Config.APPROACH_STEPS):
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed > Stage1Config.MAX_DURATION_S:
                return Stage1Result(
                    success=False, final_pose=None, port_pose=port_pose,
                    port_axis=port_axis, elapsed_time=elapsed,
                    failure_reason="timeout", port_source=port_source,
                )

            # 충돌 체크
            obs = get_observation()
            if obs is not None:
                f = obs.wrist_wrench.wrench.force
                fmag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
                fdelta = fmag - baseline_force_mag
                if fdelta > Stage1Config.FORCE_DELTA_LIMIT_N:
                    self.get_logger().warn(
                        f"충돌 감지: {fmag:.1f}N (baseline+{fdelta:.1f}N)"
                    )
                    return Stage1Result(
                        success=False, final_pose=None, port_pose=port_pose,
                        port_axis=port_axis, elapsed_time=elapsed,
                        failure_reason=f"collision (+{fdelta:.1f}N)",
                        port_source=port_source,
                    )

            fresh_port_3d = self._estimate_point(
                obs,
                port_class_id,
                port_hint=port_name_hint,
            )
            if fresh_port_3d is not None:
                port_pose = self._pose_from_point(fresh_port_3d)
            else:
                fresh_port_3d = np.array([
                    port_pose.position.x,
                    port_pose.position.y,
                    port_pose.position.z,
                ], dtype=float)

            tip_3d = self._estimate_point(obs, tip_class_id)
            measured_offsets, offset_extras = self._measure_tip_to_port_offsets(
                tip_3d,
                fresh_port_3d,
                port_axis,
            )
            approach_z_offset = (
                Stage1Config.TRIANGULATION_STOP_Z_OFFSET
                if yolo_tracking_started
                else initial_approach_z
            )
            dynamic_approach_pose, _ = self._compute_approach_pose(
                port_pose,
                gripper_tf,
                z_offset=approach_z_offset,
            )
            p_end = np.array([
                dynamic_approach_pose.position.x,
                dynamic_approach_pose.position.y,
                dynamic_approach_pose.position.z,
            ])

            # 5차 Hermite (옵션: 3차). 끝 가속도 0 → 관성 떨림 감소
            t_smooth = interp_profile(
                (i + 1) / float(Stage1Config.APPROACH_STEPS),
                quintic=Stage1Config.USE_QUINTIC_HERMITE,
            )
            pos = p_start * (1.0 - t_smooth) + p_end * t_smooth
            q = quaternion_slerp(q_start, q_end, t_smooth)

            waypoint = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=tuple_to_quat(q),
            )
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=waypoint,
                    stiffness=list(Stage1Config.STIFFNESS),
                    damping=list(Stage1Config.DAMPING),
                )
            except Exception as ex:
                self.get_logger().warn(f"Stage 1 pose command 오류: {ex}")

            measured_offset_text = (
                f" tri_offsets_xyz: {measured_offsets['x']:+0.4f} "
                f"{measured_offsets['y']:+0.4f} "
                f"{measured_offsets['z']:0.4f} "
                f"xy={measured_offsets['xy']:0.4f}"
                if measured_offsets is not None
                else " tri_offsets_xyz: n/a"
            )
            self.get_logger().info(
                f"pfrac: {t_smooth:.3} guide: yolo "
                f"z_target: {approach_z_offset:0.4} "
                f"tip_valid: {offset_extras['triangulated_tip_to_port_offsets_valid']} "
                f"{measured_offset_text}"
            )
            if measured_offsets is not None and measured_offsets["within_threshold"]:
                reached_triangulation_stop = True
                last_measured_offsets = measured_offsets
                self.get_logger().info(
                    f"Approach reached triangulated x/y/z thresholds: "
                    f"x<={Stage1Config.TRIANGULATION_STOP_X_OFFSET:.4f}m "
                    f"y<={Stage1Config.TRIANGULATION_STOP_Y_OFFSET:.4f}m "
                    f"z<={Stage1Config.TRIANGULATION_STOP_Z_OFFSET:.4f}m"
                )
                break

            self.sleep_for(Stage1Config.DT)

        # 5. 종료 검증
        final_gripper_tf = self._tcp_transform_from_obs(get_observation())
        if final_gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=port_axis,
                elapsed_time=(self.time_now() - t0).nanoseconds / 1e9,
                failure_reason="final gripper tcp pose observation failed",
                port_source=port_source,
            )

        if reached_triangulation_stop:
            ok = True
            info = (
                "triangulated tip-to-port threshold met: "
                f"x={last_measured_offsets['x']:+.4f}m "
                f"y={last_measured_offsets['y']:+.4f}m "
                f"z={last_measured_offsets['z']:+.4f}m "
                f"xy={last_measured_offsets['xy']:.4f}m"
            )
        else:
            ok, info = self._check_stage1_termination(
                final_gripper_tf, port_axis, port_pose,
                target_z=(
                    Stage1Config.TRIANGULATION_STOP_Z_OFFSET
                    if yolo_tracking_started
                    else initial_approach_z
                ),
            )
        elapsed = (self.time_now() - t0).nanoseconds / 1e9

        self.get_logger().info(
            f"━━━ Stage 1: 완료 ━━━ "
            f"(source={port_source}, {info}, elapsed {elapsed:.2f}s, ok={ok})"
        )

        return Stage1Result(
            success=ok,
            final_pose=self._transform_to_pose(final_gripper_tf),
            port_pose=port_pose,
            port_axis=port_axis,
            elapsed_time=elapsed,
            failure_reason=None if ok else f"spec not met: {info}",
            port_source=port_source,
        )
