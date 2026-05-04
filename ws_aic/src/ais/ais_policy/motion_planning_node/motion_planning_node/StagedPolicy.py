"""
StagedPolicy v3: Vision 통합 3단계 State Machine 정책

Stage 1 (이동) — 모션 플래닝 + Vision (신찬희 담당)
  - 포트 좌표 획득: Ground truth TF 우선, 실패 시 YOLO+스테레오
  - 목표: 그리퍼를 포트 축선 위 10cm 지점까지 이동
  - 방식: S-curve 직선 보간

Stage 2/3: 임시 (ground_truth 기반, 추후 AI 교체)

환경별 동작:
  - ground_truth=true:  TF로 포트 좌표 직접 읽음 (오차 0)
  - ground_truth=false: YOLO 검출 + 스테레오 삼각측량 (오차 ~17mm)

실행:
  pixi reinstall ros-kilted-motion-planning-node
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=motion_planning_node.StagedPolicy
"""

from typing import Optional

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.stage1 import Stage1Approach
from motion_planning_node.core.stage23 import Stage23Controller
from motion_planning_node.core.vision import VisionPortEstimator


# ═══════════════════════════════════════════════════════════
#  StagedPolicy
# ═══════════════════════════════════════════════════════════

class StagedPolicy(Policy):
    """3단계 State Machine 정책 (Vision 통합)."""

    PORT_AXIS_LOCAL: np.ndarray = np.array([0.0, 0.0, 1.0])

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task: Optional[Task] = None

        # Vision 모듈
        self._vision = VisionPortEstimator(
            model_path=Stage1Config.YOLO_MODEL_PATH,
            conf_thresh=Stage1Config.YOLO_CONF_THRESH,
            logger=self.get_logger(),
        )
        # ★ YOLO 모델 즉시 로드 (Stage 1 지연 방지)
        # __init__ 단계는 60초 여유 있음 (lifecycle configured 단계)
        self.get_logger().info("YOLO 모델 사전 로드 중...")
        self.get_logger().info(f"  YOLO model path: {Stage1Config.YOLO_MODEL_PATH}")
        self.get_logger().info(f"  YOLO conf threshold: {Stage1Config.YOLO_CONF_THRESH}")
        self._vision._ensure_loaded()
        self.get_logger().info(f"  YOLO loaded: {self._vision._loaded}")
        if not self._vision._loaded:
            self.get_logger().error(
                "  ✗ YOLO 모델 로드 실패! Vision fallback 불가.\n"
                "  해결: AIC_YOLO_MODEL_PATH 환경변수 설정 또는 모델 파일 경로 확인"
            )
        else:
            self.get_logger().info("  ✓ YOLO 사전 로드 완료")

        self._stage1 = Stage1Approach(self, self._vision)
        self._stage23 = Stage23Controller(self)

    # ─────────────────────────────────────────────────────
    #  프레임 이름 / TF 조회
    # ─────────────────────────────────────────────────────

    def _port_frame(self) -> str:
        return (
            f"task_board/{self._task.target_module_name}"
            f"/{self._task.port_name}_link"
        )

    def _plug_frame(self) -> str:
        return f"{self._task.cable_name}/{self._task.plug_name}_link"

    def _wait_for_tf(self, frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    "base_link", frame, Time()
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"TF 대기 중: {frame} "
                        "(ground_truth:=true 환경이면 TF 제공됨)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().warn(f"TF 대기 시간 초과: {frame}")
        return False

    def _lookup_tf(self, frame: str) -> Optional[Transform]:
        for _ in range(Stage1Config.TF_RETRY):
            try:
                return self._parent_node._tf_buffer.lookup_transform(
                    "base_link", frame, Time()
                ).transform
            except TransformException:
                self.sleep_for(Stage1Config.TF_RETRY_DT)
        return None

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return Pose(
            position=Point(
                x=tf.translation.x, y=tf.translation.y, z=tf.translation.z,
            ),
            orientation=Quaternion(
                x=tf.rotation.x, y=tf.rotation.y,
                z=tf.rotation.z, w=tf.rotation.w,
            ),
        )

    # ─────────────────────────────────────────────────────
    #  메인
    # ─────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            f"StagedPolicy (Vision 통합) 시작\n"
            f"  cable={task.cable_name}, plug={task.plug_name}\n"
            f"  port={task.port_name}, target={task.target_module_name}"
        )
        self._task = task

        # [신규] Pre-Stage settle — Trial 1 초기 cable/physics 불안정 완화
        # 근거:
        #   Trial 1 은 cable 이 막 gripper 에 attach 된 직후 시작되어
        #   flexible cable 이 아직 흔들리는 상태. 관측: Trial 1 이 다른
        #   trial 대비 Stage 1 axial err 일관되게 크고 rim 걸림 많음.
        # 효과 가설:
        #   0.8 초 정지 대기 → cable 이 중력으로 안정화 → 첫 이동 시 tracking 향상.
        # 비용: 시간 점수 영향 미미 (max 점수의 ~3%).
        self.get_logger().info("Pre-Stage settle (0.8s) — cable 안정화 대기")
        self.sleep_for(0.8)

        # TF 대기 (training 모드에서만; 평가 모드에선 실패해도 Vision으로 진행)
        # 짧게 대기 (최대 1초) — 없으면 바로 Vision으로
        self._wait_for_tf(self._port_frame(), timeout_sec=1.0)

        # Stage 1 (Vision 자동 fallback)
        result = self._stage1.run(get_observation, move_robot, send_feedback)
        self.get_logger().info(
            f"Stage 1 result: success={result.success}, "
            f"source={result.port_source}, "
            f"elapsed={result.elapsed_time:.2f}s, "
            f"reason={result.failure_reason}"
        )

        # TF/Vision 둘 다 실패한 경우 → Stage 2/3는 항상 None pose만 반환하므로
        # 시간만 낭비. 즉시 종료하여 평가 시스템이 다음 태스크로 넘어갈 수 있도록 함.
        if result.port_source == "none":
            self.get_logger().error(
                "포트 좌표 획득 완전 실패 (TF/Vision 모두 실패) → 조기 종료"
            )
            send_feedback("failed: port not detected (skipping stage 2/3)")
            return False

        # Stage 2/3 — Vision 모드면 Stage1 result의 port_pose를 전달
        # (TF 없어서 _compute_stage23_pose 내부에서 무한 대기 방지)
        port_pose_for_23 = None
        if result.port_source == "vision":
            port_pose_for_23 = result.port_pose

        try:
            self._stage23.align(move_robot, send_feedback,
                                port_pose_vision=port_pose_for_23)
            self._stage23.insert(get_observation, move_robot, send_feedback,
                                 port_pose_vision=port_pose_for_23)
        except Exception as ex:
            self.get_logger().warn(f"Stage 2/3 실행 중 예외: {ex}")

        self.get_logger().info("StagedPolicy 완료")
        return True
