"""Stage 1 approach controller for the staged policy."""

import time
from typing import Optional

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.geometry import (
    interp_profile,
    quat_inverse,
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
    def _parent_node(self):
        return self._policy._parent_node

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

    def _port_frame(self) -> str:
        return self._policy._port_frame()

    def _plug_frame(self) -> str:
        return self._policy._plug_frame()

    def _lookup_tf(self, frame: str) -> Optional[Transform]:
        return self._policy._lookup_tf(frame)

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return self._policy._transform_to_pose(tf)

    # ─────────────────────────────────────────────────────
    #  포트 pose 획득: TF → Vision fallback
    # ─────────────────────────────────────────────────────

    def _get_port_pose(self, get_observation) -> tuple:
        """(port_pose, port_source) 반환.

        port_source:
          "tf": Ground truth TF 사용 (훈련 환경)
          "vision": YOLO + 스테레오 (평가 환경)
          None: 실패
        """
        # 1순위: TF
        port_tf = self._lookup_tf(self._port_frame())
        if port_tf is not None:
            return self._transform_to_pose(port_tf), "tf"

        # 2순위: Vision
        self.get_logger().warn(
            f"TF로 포트 좌표 못 얻음 → Vision 시도"
        )
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
            obs, self._parent_node._tf_buffer, target_class_id,
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

    # ─────────────────────────────────────────────────────
    #  Stage 1: 이동 (Vision 통합)
    # ─────────────────────────────────────────────────────

    def _compute_approach_pose(
        self, port_pose: Pose, plug_tf: Optional[Transform],
        gripper_tf: Transform, port_source: str
    ) -> tuple[Pose, np.ndarray]:
        """접근점 pose 계산. (approach_pose, port_axis_world)

        port_source='vision'이면 plug_tf가 None일 수 있으므로 처리.
        """
        # 접근 축 (월드 +z 방향 기본)
        port_axis_world = np.array([0.0, 0.0, 1.0])

        port_pos = np.array([
            port_pose.position.x,
            port_pose.position.y,
            port_pose.position.z,
        ])
        approach_pos = port_pos + port_axis_world * Stage1Config.Z_OFFSET

        # 플러그-그리퍼 오프셋 (TF 가능한 경우만)
        if plug_tf is not None:
            plug_pos = np.array([
                plug_tf.translation.x,
                plug_tf.translation.y,
                plug_tf.translation.z,
            ])
            gripper_pos = np.array([
                gripper_tf.translation.x,
                gripper_tf.translation.y,
                gripper_tf.translation.z,
            ])
            offset = gripper_pos - plug_pos
            gripper_target_pos = approach_pos + offset
        else:
            # Vision 모드: 플러그 TF 없음 → 대략적 오프셋 (SFP 기준 ~5cm)
            gripper_target_pos = approach_pos + np.array([0.0, 0.015, 0.045])

        # 방향
        if port_source == "tf" and plug_tf is not None:
            # TF 방식: 포트 방향에 맞춤
            q_port = quat_to_tuple(port_pose.orientation)
            q_plug = quat_to_tuple(self._transform_to_pose(plug_tf).orientation)
            q_gripper = quat_to_tuple(self._transform_to_pose(gripper_tf).orientation)
            q_diff = quaternion_multiply(q_port, quat_inverse(q_plug))
            q_target = quaternion_multiply(q_diff, q_gripper)
        else:
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
        plug_tf: Optional[Transform],
        gripper_tf: Transform,
        port_axis: np.ndarray,
        port_pose: Pose,
        target_z: Optional[float] = None,
    ) -> tuple[bool, str]:
        """target_z가 주어지면 그 값을 목표로 검증 (기본은 Stage 1-A 목표).

        Stage 1-B까지 완료된 경우엔 target_z=Z_OFFSET_MID (3cm) 사용.
        """
        if target_z is None:
            target_z = Stage1Config.Z_OFFSET

        # 플러그 TF 있으면 플러그 기준, 없으면 그리퍼 기준
        if plug_tf is not None:
            ref = np.array([
                plug_tf.translation.x,
                plug_tf.translation.y,
                plug_tf.translation.z,
            ])
            ref_name = "plug"
        else:
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

        # 1. 포트 pose 획득 (TF → Vision fallback) — 시간 측정 전
        port_pose, port_source = self._get_port_pose(get_observation)
        if port_pose is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=None, port_axis=None,
                elapsed_time=0.0,
                failure_reason="포트 좌표 획득 실패 (TF/Vision 모두 실패)",
                port_source="none",
            )

        self.get_logger().info(
            f"  포트 좌표 소스: {port_source}\n"
            f"  포트 위치: ({port_pose.position.x:+.3f}, "
            f"{port_pose.position.y:+.3f}, {port_pose.position.z:+.3f})"
        )

        # ★ 시간 측정은 실제 이동 시작 전에 개시
        t0 = self.time_now()

        # 2. 플러그 / 그리퍼 TF (Vision 모드에서도 플러그 TF는 시도)
        plug_tf = self._lookup_tf(self._plug_frame())  # None 가능
        gripper_tf = self._lookup_tf("gripper/tcp")
        if gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=None, elapsed_time=0.0,
                failure_reason="gripper TF 조회 실패",
                port_source=port_source,
            )

        # 3. 접근점 계산
        approach_pose, port_axis = self._compute_approach_pose(
            port_pose, plug_tf, gripper_tf, port_source
        )
        self.get_logger().info(
            f"  접근점: ({approach_pose.position.x:+.3f}, "
            f"{approach_pose.position.y:+.3f}, {approach_pose.position.z:+.3f})"
        )

        # 4. S-curve 직선 보간
        start_pose = self._transform_to_pose(gripper_tf)
        q_start = quat_to_tuple(start_pose.orientation)
        q_end = quat_to_tuple(approach_pose.orientation)
        p_start = np.array([
            start_pose.position.x, start_pose.position.y, start_pose.position.z
        ])
        p_end = np.array([
            approach_pose.position.x,
            approach_pose.position.y,
            approach_pose.position.z,
        ])

        for i in range(Stage1Config.N_STEPS):
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

            # 5차 Hermite (옵션: 3차). 끝 가속도 0 → 관성 떨림 감소
            t_smooth = interp_profile(
                (i + 1) / Stage1Config.N_STEPS,
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
            except TransformException as ex:
                self.get_logger().warn(f"Stage 1 TF 오류: {ex}")

            self.sleep_for(Stage1Config.DT)

        # 4.5. Stage 1-A 후 안정화 대기 (관성 떨림 완화)
        if Stage1Config.SETTLE_AFTER_STAGE1A > 0:
            self.get_logger().info(
                f"  Stage 1-A 안정화 대기 {Stage1Config.SETTLE_AFTER_STAGE1A:.1f}s"
            )
            # 마지막 waypoint 재전송으로 holding (떨림 감소)
            last_pose = Pose(
                position=Point(x=float(p_end[0]), y=float(p_end[1]), z=float(p_end[2])),
                orientation=tuple_to_quat(q_end),
            )
            settle_end = time.time() + Stage1Config.SETTLE_AFTER_STAGE1A
            while time.time() < settle_end:
                try:
                    self.set_pose_target(
                        move_robot=move_robot, pose=last_pose,
                        stiffness=list(Stage1Config.STIFFNESS),
                        damping=list(Stage1Config.DAMPING),
                    )
                except TransformException:
                    pass
                self.sleep_for(0.05)

        # ═════════════════════════════════════════════════════════
        # 4-B. Stage 1-B: Mid Approach (7cm → 3cm 하강 + 정렬)
        # ═════════════════════════════════════════════════════════
        if Stage1Config.ENABLE_STAGE1B:
            self.get_logger().info("━━━ Stage 1-B: 중간 접근 시작 (7→3cm) ━━━")
            send_feedback("Stage 1-B: mid approach")

            # Stage 1-B 시작점 = 현재 그리퍼 (= 접근점 근처)
            mid_start_tf = self._lookup_tf("gripper/tcp")
            if mid_start_tf is None:
                self.get_logger().warn("Stage 1-B: gripper TF 조회 실패, 건너뜀")
            else:
                mid_start_pose = self._transform_to_pose(mid_start_tf)
                p_start_mid = np.array([
                    mid_start_pose.position.x,
                    mid_start_pose.position.y,
                    mid_start_pose.position.z,
                ])
                q_start_mid = quat_to_tuple(mid_start_pose.orientation)

                for i in range(Stage1Config.N_STEPS_MID):
                    elapsed = (self.time_now() - t0).nanoseconds / 1e9
                    if elapsed > Stage1Config.MAX_DURATION_S:
                        return Stage1Result(
                            success=False, final_pose=None,
                            port_pose=port_pose, port_axis=port_axis,
                            elapsed_time=elapsed,
                            failure_reason="timeout (stage 1-B)",
                            port_source=port_source,
                        )

                    # 충돌 체크
                    obs = get_observation()
                    if obs is not None:
                        f = obs.wrist_wrench.wrench.force
                        fmag = float(np.sqrt(f.x*f.x + f.y*f.y + f.z*f.z))
                        fdelta = fmag - baseline_force_mag
                        if fdelta > Stage1Config.FORCE_DELTA_LIMIT_N:
                            self.get_logger().warn(
                                f"Stage 1-B 충돌 감지: {fmag:.1f}N "
                                f"(baseline+{fdelta:.1f}N)"
                            )
                            return Stage1Result(
                                success=False, final_pose=None,
                                port_pose=port_pose, port_axis=port_axis,
                                elapsed_time=elapsed,
                                failure_reason=f"stage1b collision (+{fdelta:.1f}N)",
                                port_source=port_source,
                            )

                    # FEEDBACK: 매 스텝 포트 TF 재조회 (있으면) — 드리프트 보정
                    if Stage1Config.FEEDBACK_MID and port_source == "tf":
                        fresh_port_tf = self._lookup_tf(self._port_frame())
                        fresh_gripper_tf = self._lookup_tf("gripper/tcp")
                        fresh_plug_tf = self._lookup_tf(self._plug_frame())
                        if fresh_port_tf is not None and fresh_gripper_tf is not None:
                            # 현재 포트 기준 접근점 재계산 (더 낮은 z_offset)
                            fresh_port_pose = self._transform_to_pose(fresh_port_tf)
                            # 임시로 Z_OFFSET을 Z_OFFSET_MID로 바꿔 계산
                            saved_z = Stage1Config.Z_OFFSET
                            Stage1Config.Z_OFFSET = Stage1Config.Z_OFFSET_MID
                            mid_target_pose, _ = self._compute_approach_pose(
                                fresh_port_pose, fresh_plug_tf,
                                fresh_gripper_tf, port_source,
                            )
                            Stage1Config.Z_OFFSET = saved_z
                            p_end_mid = np.array([
                                mid_target_pose.position.x,
                                mid_target_pose.position.y,
                                mid_target_pose.position.z,
                            ])
                            q_end_mid = quat_to_tuple(
                                mid_target_pose.orientation
                            )
                        else:
                            # fallback: 처음 p_end에서 z만 내림
                            p_end_mid = p_end.copy()
                            p_end_mid[2] -= (
                                Stage1Config.Z_OFFSET - Stage1Config.Z_OFFSET_MID
                            )
                            q_end_mid = q_end
                    else:
                        # Vision 모드: 처음 p_end에서 z만 내림
                        p_end_mid = p_end.copy()
                        p_end_mid[2] -= (
                            Stage1Config.Z_OFFSET - Stage1Config.Z_OFFSET_MID
                        )
                        q_end_mid = q_end

                    # [Cable tension compensation] SFP 만 14mm 더 내림
                    # 근거: 측정된 13.6mm steady-state err ≈ 2N 장력 / 150 N/m
                    plug_name = (self._task.plug_name or "").lower()
                    if "sfp" in plug_name:
                        z_before = p_end_mid[2]
                        p_end_mid[2] -= Stage1Config.SFP_CABLE_TENSION_COMPENSATION
                        if i == 0:
                            # 진단: 실제로 얼마나 낮게 명령되는지 확인
                            self.get_logger().info(
                                f"  [SFP] cable tension compensation applied: "
                                f"gripper z {z_before:.4f}m → {p_end_mid[2]:.4f}m "
                                f"(Δ=-{Stage1Config.SFP_CABLE_TENSION_COMPENSATION*1000:.0f}mm)"
                            )

                    t_norm = (i + 1) / Stage1Config.N_STEPS_MID
                    t_smooth = interp_profile(
                        t_norm, quintic=Stage1Config.USE_QUINTIC_HERMITE
                    )
                    pos_mid = p_start_mid * (1.0 - t_smooth) + p_end_mid * t_smooth
                    q_mid = quaternion_slerp(q_start_mid, q_end_mid, t_smooth)

                    waypoint_mid = Pose(
                        position=Point(
                            x=float(pos_mid[0]),
                            y=float(pos_mid[1]),
                            z=float(pos_mid[2]),
                        ),
                        orientation=tuple_to_quat(q_mid),
                    )
                    try:
                        self.set_pose_target(
                            move_robot=move_robot,
                            pose=waypoint_mid,
                            stiffness=list(Stage1Config.STIFFNESS_MID),
                            damping=list(Stage1Config.DAMPING_MID),
                        )
                    except TransformException as ex:
                        self.get_logger().warn(f"Stage 1-B TF 오류: {ex}")

                    self.sleep_for(Stage1Config.DT_MID)

                # ─ Stage 1-B 수렴 대기 (위치 + 안정성 기반) ─
                # 근거:
                #   1) compensation 으로 정적 offset 제거됨 → target 정확 도달 기대
                #   2) 그러나 cable 떨림(oscillation) 가능 → "err < tol 이
                #      연속 N회" 로 안정 상태 확인
                # 단순 "순간 err < tol" 이면 진동 중 우연히 낮은 순간에 통과할
                # 수 있음 → 연속 체크로 진짜 수렴만 인정.
                # 진단 로그: 명령 vs 목표
                expected_plug_z = None
                if port_pose is not None:
                    expected_plug_z = port_pose.position.z + Stage1Config.Z_OFFSET_MID
                self.get_logger().info(
                    f"  Stage 1-B 수렴 대기 진단: "
                    f"commanded gripper z = {p_end_mid[2]:.4f}m, "
                    f"desired plug z = {expected_plug_z:.4f}m "
                    f"(port_z={port_pose.position.z:.4f} + {Stage1Config.Z_OFFSET_MID:.3f})"
                )
                self.get_logger().info(
                    f"  수렴 기준: err ≤ {Stage1Config.STAGE1B_CONVERGENCE_TOL_M*1000:.0f}mm "
                    f"× {Stage1Config.STAGE1B_STABLE_CONSECUTIVE}회 연속 or "
                    f"{Stage1Config.STAGE1B_CONVERGENCE_MAX_WAIT_S:.1f}s"
                )
                hold_pose = Pose(
                    position=Point(
                        x=float(p_end_mid[0]),
                        y=float(p_end_mid[1]),
                        z=float(p_end_mid[2]),
                    ),
                    orientation=tuple_to_quat(q_end_mid),
                )
                convergence_tol = Stage1Config.STAGE1B_CONVERGENCE_TOL_M
                max_wait = Stage1Config.STAGE1B_CONVERGENCE_MAX_WAIT_S
                stable_needed = Stage1Config.STAGE1B_STABLE_CONSECUTIVE
                wait_end = time.time() + max_wait
                stable_count = 0
                last_err = None
                converged = False
                wait_start = time.time()
                last_log_time = 0.0
                # [옵션 A] 수렴 대기 구간에서 Z-stiffness 부스트 (150→500 N/m)
                # 케이블 평형점 극복 목적 — S-curve 본체는 낮은 K 그대로 유지.
                plug_is_sfp = "sfp" in (self._task.plug_name or "").lower()
                hold_stiffness = (
                    Stage1Config.STIFFNESS_MID_BOOST if plug_is_sfp
                    else Stage1Config.STIFFNESS_MID
                )
                hold_damping = (
                    Stage1Config.DAMPING_MID_BOOST if plug_is_sfp
                    else Stage1Config.DAMPING_MID
                )
                if plug_is_sfp:
                    self.get_logger().info(
                        f"  [SFP] 수렴 대기 Z-stiffness 부스트: "
                        f"{Stage1Config.STIFFNESS_MID[2]:.0f} → "
                        f"{Stage1Config.STIFFNESS_MID_BOOST[2]:.0f} N/m"
                    )
                while time.time() < wait_end:
                    try:
                        self.set_pose_target(
                            move_robot=move_robot, pose=hold_pose,
                            stiffness=list(hold_stiffness),
                            damping=list(hold_damping),
                        )
                    except TransformException:
                        pass
                    self.sleep_for(0.05)

                    # 수렴 체크: 실제 plug z vs TARGET z
                    cur_plug_tf = self._lookup_tf(self._plug_frame())
                    cur_gripper_tf = self._lookup_tf("gripper/tcp")
                    if cur_plug_tf is not None and port_pose is not None:
                        desired_plug_z = (
                            port_pose.position.z + Stage1Config.Z_OFFSET_MID
                        )
                        cur_axial = abs(
                            cur_plug_tf.translation.z - desired_plug_z
                        )
                        last_err = cur_axial

                        # [진단] 0.3초마다 실시간 상태 출력
                        elapsed = time.time() - wait_start
                        if elapsed - last_log_time >= 0.3:
                            gripper_z_str = (
                                f"{cur_gripper_tf.translation.z:.4f}"
                                if cur_gripper_tf else "N/A"
                            )
                            self.get_logger().info(
                                f"    [t={elapsed:.1f}s] "
                                f"plug_z={cur_plug_tf.translation.z:.4f}m, "
                                f"gripper_z={gripper_z_str}, "
                                f"err={cur_axial*1000:.1f}mm, "
                                f"stable={stable_count}"
                            )
                            last_log_time = elapsed

                        if cur_axial < convergence_tol:
                            stable_count += 1
                            if stable_count >= stable_needed:
                                converged = True
                                self.get_logger().info(
                                    f"  수렴 완료: axial err "
                                    f"{cur_axial*1000:.1f}mm × {stable_count}회 연속"
                                )
                                break
                        else:
                            stable_count = 0  # 떨림 중 — 카운터 reset
                if not converged and last_err is not None:
                    self.get_logger().warn(
                        f"  수렴 대기 타임아웃 ({max_wait:.1f}s): "
                        f"최종 err {last_err*1000:.1f}mm, "
                        f"stable_count={stable_count} (필요 {stable_needed})"
                    )

                self.get_logger().info("━━━ Stage 1-B: 중간 접근 완료 ━━━")

        # 5. 종료 검증
        final_plug_tf = self._lookup_tf(self._plug_frame())
        final_gripper_tf = self._lookup_tf("gripper/tcp")
        if final_gripper_tf is None:
            return Stage1Result(
                success=False, final_pose=None, port_pose=port_pose,
                port_axis=port_axis,
                elapsed_time=(self.time_now() - t0).nanoseconds / 1e9,
                failure_reason="final TF lookup failed",
                port_source=port_source,
            )

        # Stage 1-B가 실행됐으면 최종 목표는 Z_OFFSET_MID (3cm)
        final_target_z = (
            Stage1Config.Z_OFFSET_MID if Stage1Config.ENABLE_STAGE1B
            else Stage1Config.Z_OFFSET
        )
        ok, info = self._check_stage1_termination(
            final_plug_tf, final_gripper_tf, port_axis, port_pose,
            target_z=final_target_z,
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
