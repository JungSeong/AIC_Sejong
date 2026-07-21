from __future__ import annotations

import math
import os
import threading
import numpy as np

from typing import TYPE_CHECKING, Optional
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from final_policy.config import FinalPolicyConfig
from final_policy.debug import FinalPolicyDebugMixin
from final_policy.geometry import interp_profile, quat_to_tuple, tuple_to_quat
from final_policy.model_store import (
    SC_YOLO_MODEL,
    SC_VISION_OFFSET_MODEL,
    SFP_YOLO_MODEL,
    SFP_VISION_OFFSET_MODEL,
    format_model_log,
    resolve_model_path,
)
from final_policy.vision import VisionPortEstimator

if TYPE_CHECKING:
    from final_policy.vision_offset import VisionOffsetPredictor


class FinalPolicy(FinalPolicyDebugMixin, Policy):
    """
    YOLO 포트 검출, 접근, 정렬, 삽입을 순서대로 수행하는 최종 정책.
    """

    TARGET_CLASS_ID_SFP = 0
    TARGET_CLASS_ID_SC = 0

    def __init__(self, parent_node):
        """정책 실행 중 공유할 모델 경로, 비전 추정기, 캐시 상태를 초기화한다."""
        Policy.__init__(self, parent_node)
        self._task: Optional[Task] = None
        self._sfp_yolo_model_path: Optional[str] = None
        self._sc_yolo_model_path: Optional[str] = None
        self._sfp_vision_offset_model_path: Optional[str] = None
        self._sc_vision_offset_model_path: Optional[str] = None
        self._cached_port_base: Optional[np.ndarray] = None
        self._target_orientation = None
        self._fixed_target_orientation = None
        self._sfp_yolo_conf_thresh = float(
            os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.8")
        )
        self._sc_yolo_conf_thresh = float(
            os.environ.get(
                "AIC_DEBUG_SC_YOLO_CONF_THRESH",
                os.environ.get("AIC_DEBUG_SFP_YOLO_CONF_THRESH", "0.8"),
            )
        )
        self._vision_by_port_type = {}
        self._vision_debug_save_enabled = False
        self._align_debug_call_count = 0
        self._triangulation_debug_call_count = 0
        self._yolo_download_threads: dict[str, threading.Thread] = {}
        self._yolo_download_lock = threading.Lock()
        self._vision_offset_predictor_by_port_type: dict[str, VisionOffsetPredictor] = {}
        self._vision_offset_download_threads: dict[str, threading.Thread] = {}
        self._vision_offset_download_lock = threading.Lock()
        self._triangulated_port_xyz_pub = self._create_triangulated_port_xyz_publisher()
        self._send_feedback: Optional[SendFeedbackCallback] = None
        self.get_logger().info(
            "FinalPolicy ready: "
            "yolo_model=initial_load, "
            "vision_offset_model=initial_load, "
            "background_download=enabled"
        )

    def _create_triangulated_port_xyz_publisher(self):
        """옵션이 켜진 경우 triangulation 포트 XYZ 평가 토픽 publisher를 만든다."""
        if not FinalPolicyConfig.PUBLISH_TRIANGULATED_PORT_XYZ:
            return None
        topic = str(FinalPolicyConfig.TRIANGULATED_PORT_XYZ_TOPIC or "").strip()
        if not topic:
            self.get_logger().warn(
                "AIC_PUBLISH_TRIANGULATED_PORT_XYZ is true, but topic is empty"
            )
            return None
        publisher = self._parent_node.create_publisher(PointStamped, topic, 10)
        self.get_logger().info(
            "Triangulated port XYZ publisher enabled: "
            f"topic={topic}, frame={FinalPolicyConfig.TRIANGULATED_PORT_XYZ_FRAME_ID}"
        )
        return publisher

    def _publish_triangulated_port_xyz(self, port: np.ndarray, label: str) -> None:
        """캐시된 triangulation 결과를 PointStamped로 발행한다."""
        if self._triangulated_port_xyz_pub is None:
            return
        port = np.asarray(port, dtype=np.float64).reshape(-1)
        if port.size < 3 or not np.isfinite(port[:3]).all():
            self.get_logger().warn(
                f"Triangulated port XYZ publish skipped: invalid port={port}"
            )
            return
        msg = PointStamped()
        msg.header.stamp = self.time_now().to_msg()
        msg.header.frame_id = str(FinalPolicyConfig.TRIANGULATED_PORT_XYZ_FRAME_ID)
        msg.point.x = float(port[0])
        msg.point.y = float(port[1])
        msg.point.z = float(port[2])
        self._triangulated_port_xyz_pub.publish(msg)
        self.get_logger().info(
            "Triangulated port XYZ published: "
            f"label={label}, topic={FinalPolicyConfig.TRIANGULATED_PORT_XYZ_TOPIC}, "
            f"base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
        )

    @staticmethod
    def _copy_pose(pose: Pose) -> Pose:
        """ROS Pose 메시지를 값 복사해서 이후 수정이 원본에 영향을 주지 않게 한다."""
        return Pose(
            position=Point(
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
            ),
            orientation=Quaternion(
                x=float(pose.orientation.x),
                y=float(pose.orientation.y),
                z=float(pose.orientation.z),
                w=float(pose.orientation.w),
            ),
        )

    @staticmethod
    def _copy_quaternion(quat: Quaternion) -> Quaternion:
        """ROS Quaternion 메시지를 값 복사한다."""
        return Quaternion(
            x=float(quat.x),
            y=float(quat.y),
            z=float(quat.z),
            w=float(quat.w),
        )

    @staticmethod
    def _normalize_quat(q):
        """쿼터니언 튜플을 단위 길이로 정규화한다."""
        values = np.asarray(q, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        values /= norm
        return tuple(float(v) for v in values)

    @staticmethod
    def _tcp_pose(observation) -> Optional[Pose]:
        """현재 controller_state의 TCP pose를 안전하게 복사해서 반환한다."""
        if observation is None:
            return None
        return FinalPolicy._copy_pose(observation.controller_state.tcp_pose)

    @staticmethod
    def _axis_angle_quat(axis: np.ndarray, angle_rad: float):
        """주어진 축과 회전각을 (w, x, y, z) 쿼터니언으로 변환한다."""
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        axis /= norm
        half = 0.5 * float(angle_rad)
        sin_half = float(math.sin(half))
        return FinalPolicy._normalize_quat(
            (
                float(math.cos(half)),
                float(axis[0] * sin_half),
                float(axis[1] * sin_half),
                float(axis[2] * sin_half),
            )
        )

    @staticmethod
    def _rpy_delta_quat_base(rpy_rad: np.ndarray):
        """base_link 축 기준 roll/pitch/yaw 보정량을 쿼터니언 증분으로 변환한다."""
        roll, pitch, yaw = [float(value) for value in np.asarray(rpy_rad, dtype=np.float64)]
        q_roll = FinalPolicy._axis_angle_quat(np.array([1.0, 0.0, 0.0]), roll)
        q_pitch = FinalPolicy._axis_angle_quat(np.array([0.0, 1.0, 0.0]), pitch)
        q_yaw = FinalPolicy._axis_angle_quat(np.array([0.0, 0.0, 1.0]), yaw)
        return FinalPolicy._normalize_quat(
            quaternion_multiply(q_yaw, quaternion_multiply(q_pitch, q_roll))
        )

    def _port_type(self) -> str:
        """task 문자열들에서 sc 여부를 찾아 sc/sfp 포트 타입을 판별한다."""
        tokens = " ".join(
            str(value or "").lower()
            for value in (
                getattr(self._task, "plug_name", ""),
                getattr(self._task, "port_name", ""),
                getattr(self._task, "port_type", ""),
                getattr(self._task, "task_type", ""),
            )
        )
        return "sc" if "sc" in tokens else "sfp"

    def _target_class_id(self, port_type: str) -> int:
        """포트 타입별 YOLO target class id를 환경변수 또는 기본값에서 읽는다."""
        if port_type == "sc":
            return int(
                os.environ.get("AIC_DEBUG_SC_TARGET_CLASS_ID", self.TARGET_CLASS_ID_SC)
            )
        return int(
            os.environ.get("AIC_DEBUG_SFP_TARGET_CLASS_ID", self.TARGET_CLASS_ID_SFP)
        )

    def _ensure_yolo_model_ready_for_port_type(
        self,
        port_type: str,
        send_feedback: Optional[SendFeedbackCallback] = None,
    ) -> bool:
        """현재 task 포트 타입에 필요한 YOLO 모델 하나만 준비한다."""
        port_type = "sc" if port_type == "sc" else "sfp"
        with self._yolo_download_lock:
            background_thread = self._yolo_download_threads.get(port_type)
        if (
            background_thread is not None
            and background_thread.is_alive()
            and background_thread is not threading.current_thread()
        ):
            self.get_logger().info(
                format_model_log(
                    f"Waiting for background {port_type.upper()} YOLO download"
                )
            )
            background_thread.join()

        if port_type == "sc":
            if self._sc_yolo_model_path is None:
                if send_feedback is not None:
                    send_feedback("Final Policy: preparing SC YOLO model")
                self._sc_yolo_model_path = resolve_model_path(
                    SC_YOLO_MODEL,
                    logger=self.get_logger(),
                )
            return True

        if self._sfp_yolo_model_path is None:
            if send_feedback is not None:
                send_feedback("Final Policy: preparing SFP YOLO model")
            self._sfp_yolo_model_path = resolve_model_path(
                SFP_YOLO_MODEL,
                logger=self.get_logger(),
            )
        return True

    def _start_background_yolo_model_download(self, port_type: str) -> None:
        """지금 당장 쓰지 않는 YOLO checkpoint를 stage 진행과 병렬로 받아둔다."""
        port_type = "sc" if port_type == "sc" else "sfp"
        model_path = (
            self._sc_yolo_model_path
            if port_type == "sc"
            else self._sfp_yolo_model_path
        )
        if model_path is not None:
            return

        with self._yolo_download_lock:
            existing = self._yolo_download_threads.get(port_type)
            if existing is not None and existing.is_alive():
                return

            def download_model() -> None:
                try:
                    self.get_logger().info(
                        format_model_log(
                            f"Background {port_type.upper()} YOLO download start"
                        )
                    )
                    self._ensure_yolo_model_ready_for_port_type(
                        port_type,
                        send_feedback=None,
                    )
                    self.get_logger().info(
                        format_model_log(
                            f"Background {port_type.upper()} YOLO download done"
                        )
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        format_model_log(
                            f"Background {port_type.upper()} YOLO download failed: {exc}"
                        )
                    )

            thread = threading.Thread(
                target=download_model,
                name=f"final-policy-{port_type}-yolo-download",
                daemon=True,
            )
            self._yolo_download_threads[port_type] = thread
            thread.start()

    def _start_background_yolo_model_downloads(self) -> None:
        """현재 포트 타입 외 YOLO checkpoint도 미리 받아 다음 task를 준비한다."""
        active_port_type = self._port_type()
        active_port_type = "sc" if active_port_type == "sc" else "sfp"
        for port_type in ("sfp", "sc"):
            if port_type != active_port_type:
                self._start_background_yolo_model_download(port_type)

    def _ensure_vision_offset_model_ready_for_port_type(
        self,
        port_type: str,
        send_feedback: Optional[SendFeedbackCallback] = None,
    ) -> bool:
        """현재 task 포트 타입에 필요한 vision-offset 모델 경로 하나만 준비한다."""
        port_type = "sc" if port_type == "sc" else "sfp"
        with self._vision_offset_download_lock:
            background_thread = self._vision_offset_download_threads.get(port_type)
        if (
            background_thread is not None
            and background_thread.is_alive()
            and background_thread is not threading.current_thread()
        ):
            self.get_logger().info(
                format_model_log(
                    f"Waiting for background {port_type.upper()} vision-offset download"
                )
            )
            background_thread.join()

        if port_type == "sc":
            if self._sc_vision_offset_model_path is None:
                if send_feedback is not None:
                    send_feedback("Final Policy: preparing SC vision-offset model")
                self._sc_vision_offset_model_path = resolve_model_path(
                    SC_VISION_OFFSET_MODEL,
                    logger=self.get_logger(),
                )
            return True

        if self._sfp_vision_offset_model_path is None:
            if send_feedback is not None:
                send_feedback("Final Policy: preparing SFP vision-offset model")
            self._sfp_vision_offset_model_path = resolve_model_path(
                SFP_VISION_OFFSET_MODEL,
                logger=self.get_logger(),
            )
        return True

    def _start_background_vision_offset_model_download(self, port_type: str) -> None:
        """지금 당장 쓰지 않는 vision-offset checkpoint를 stage 진행과 병렬로 받아둔다."""
        port_type = "sc" if port_type == "sc" else "sfp"
        model_path = (
            self._sc_vision_offset_model_path
            if port_type == "sc"
            else self._sfp_vision_offset_model_path
        )
        if model_path is not None:
            return

        with self._vision_offset_download_lock:
            existing = self._vision_offset_download_threads.get(port_type)
            if existing is not None and existing.is_alive():
                return

            def download_model() -> None:
                try:
                    self.get_logger().info(
                        format_model_log(
                            f"Background {port_type.upper()} vision-offset download start"
                        )
                    )
                    self._ensure_vision_offset_model_ready_for_port_type(
                        port_type,
                        send_feedback=None,
                    )
                    self.get_logger().info(
                        format_model_log(
                            f"Background {port_type.upper()} vision-offset download done"
                        )
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        format_model_log(
                            f"Background {port_type.upper()} vision-offset download failed: {exc}"
                        )
                    )

            thread = threading.Thread(
                target=download_model,
                name=f"final-policy-{port_type}-vision-offset-download",
                daemon=True,
            )
            self._vision_offset_download_threads[port_type] = thread
            thread.start()

    def _start_background_vision_offset_model_downloads(self) -> None:
        """현재 포트 타입 외 vision-offset checkpoint도 미리 받아 다음 task를 준비한다."""
        active_port_type = self._port_type()
        active_port_type = "sc" if active_port_type == "sc" else "sfp"
        for port_type in ("sfp", "sc"):
            if port_type != active_port_type:
                self._start_background_vision_offset_model_download(port_type)

    def _vision_for_port_type(self, port_type: str) -> VisionPortEstimator:
        """포트 타입에 맞는 VisionPortEstimator를 lazy 생성하고 재사용한다."""
        port_type = "sc" if port_type == "sc" else "sfp"
        self._ensure_yolo_model_ready_for_port_type(port_type, self._send_feedback)
        if port_type not in self._vision_by_port_type:
            model_path = (
                self._sc_yolo_model_path
                if port_type == "sc"
                else self._sfp_yolo_model_path
            )
            conf_thresh = (
                self._sc_yolo_conf_thresh
                if port_type == "sc"
                else self._sfp_yolo_conf_thresh
            )
            self.get_logger().info(
                format_model_log(f"Loading {port_type.upper()} YOLO model: {model_path}")
            )
            vision = VisionPortEstimator(
                model_path=model_path,
                conf_thresh=conf_thresh,
                logger=self.get_logger(),
                debug_save_enabled=self._vision_debug_save_enabled,
                auto_start=False,
            )
            self._vision_by_port_type[port_type] = vision
        return self._vision_by_port_type[port_type]

    def _preload_detection_model_for_current_task(self) -> None:
        """현재 task 포트 타입에 맞는 YOLO detector를 정책 시작 시점에 동기 로드한다."""
        port_type = self._port_type()
        vision = self._vision_for_port_type(port_type)
        if not vision.load_model():
            raise RuntimeError(f"{port_type.upper()} YOLO model load failed")

    def _vision_offset_predictor_for_align(self):
        """현재 task의 SFP/SC 타입에 맞는 vision-offset predictor를 lazy 로드한다."""
        from final_policy.vision_offset import VisionOffsetPredictor

        port_type = self._port_type()
        port_type = "sc" if port_type == "sc" else "sfp"
        self._ensure_vision_offset_model_ready_for_port_type(
            port_type,
            self._send_feedback,
        )
        if port_type not in self._vision_offset_predictor_by_port_type:
            checkpoint_path = (
                self._sc_vision_offset_model_path
                if port_type == "sc"
                else self._sfp_vision_offset_model_path
            )
            if self._send_feedback is not None:
                self._send_feedback(
                    f"Final Policy: loading {port_type.upper()} vision-offset model"
                )
            self.get_logger().info(
                format_model_log(
                    f"Loading {port_type.upper()} vision-offset model: {checkpoint_path}"
                )
            )
            self._vision_offset_predictor_by_port_type[port_type] = VisionOffsetPredictor(
                checkpoint_path=checkpoint_path,
                logger=self.get_logger(),
            )
        return self._vision_offset_predictor_by_port_type[port_type]

    def _manual_rotation_deg(self) -> float:
        """포트 타입별 수동 wrist 회전 보정각을 도 단위로 반환한다."""
        if self._port_type() == "sc":
            return float(FinalPolicyConfig.APPROACH_SC_MANUAL_ROTATION_DEG)
        return float(FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_DEG)

    def _insertion_stiffness(self) -> tuple:
        """포트 타입별 삽입 단계 stiffness를 반환한다."""
        if self._port_type() == "sc":
            return FinalPolicyConfig.SC_INSERTION_STIFFNESS
        return FinalPolicyConfig.SFP_INSERTION_STIFFNESS

    def _insertion_damping(self) -> tuple:
        """포트 타입별 삽입 단계 damping을 반환한다."""
        if self._port_type() == "sc":
            return FinalPolicyConfig.SC_INSERTION_DAMPING
        return FinalPolicyConfig.SFP_INSERTION_DAMPING

    def _axis(self, pose: Pose) -> np.ndarray:
        """수동 wrist 회전에 사용할 축을 base 또는 TCP 좌표계 기준으로 계산한다."""
        axis_name = str(FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS)
        base_axes = {
            "base_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "base_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "base_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if axis_name in base_axes:
            return base_axes[axis_name]

        local_axes = {
            "tcp_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "tcp_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "tcp_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        local_axis = local_axes.get(axis_name, local_axes["tcp_z"])
        q = quat_to_tuple(pose.orientation)
        rotated = quaternion_multiply(
            quaternion_multiply(q, (0.0, *local_axis)),
            (q[0], -q[1], -q[2], -q[3]),
        )
        return np.array([rotated[1], rotated[2], rotated[3]], dtype=np.float64)

    def _follow_pose(
        self,
        *,
        move_robot,
        start_pose: Pose,
        target_pose: Pose,
        steps: int,
        stiffness: tuple,
        damping: tuple,
        dt: float,
        label: str,
    ) -> None:
        """현재 pose에서 목표 pose까지 위치/자세를 S-curve로 보간해 순차 명령한다."""
        start = np.array(
            [start_pose.position.x, start_pose.position.y, start_pose.position.z],
            dtype=np.float64,
        )
        target = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z],
            dtype=np.float64,
        )
        q_start = quat_to_tuple(start_pose.orientation)
        q_target = quat_to_tuple(target_pose.orientation)

        step_count = max(1, int(steps))
        for index in range(step_count):
            t = interp_profile((index + 1) / step_count, quintic=True)
            pos = start * (1.0 - t) + target * t
            quat = quaternion_slerp(q_start, q_target, t)
            pose = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=tuple_to_quat(quat),
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=list(stiffness),
                damping=list(damping),
            )
            if index == 0 or index == step_count - 1:
                self.get_logger().info(
                    f"{label}: waypoint {index + 1}/{step_count} "
                    f"tcp=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
                )
            self.sleep_for(dt)

    def _target_wrist_orientation(self, start_pose: Pose) -> Quaternion:
        """접근 단계에서 유지할 목표 wrist orientation을 계산하고 캐시한다."""
        if self._fixed_target_orientation is not None:
            return self._copy_quaternion(self._fixed_target_orientation)

        angle_deg = self._manual_rotation_deg()
        if abs(angle_deg) < 1e-9:
            self._fixed_target_orientation = self._copy_quaternion(start_pose.orientation)
            return self._copy_quaternion(self._fixed_target_orientation)

        q_delta = self._axis_angle_quat(self._axis(start_pose), math.radians(angle_deg))
        q_target = self._normalize_quat(
            quaternion_multiply(q_delta, quat_to_tuple(start_pose.orientation))
        )
        self._fixed_target_orientation = tuple_to_quat(q_target)
        return self._copy_quaternion(self._fixed_target_orientation)

    def _configure_detection_context(self, vision: VisionPortEstimator) -> None:
        """현재 task 정보를 디버그 이미지 파일명에 들어갈 detection context로 설정한다."""
        vision.set_debug_task_context(
            target_module_name=str(getattr(self._task, "target_module_name", "") or ""),
            port_name=str(getattr(self._task, "port_name", "") or ""),
            plug_name=str(getattr(self._task, "plug_name", "") or ""),
            cable_name=str(getattr(self._task, "cable_name", "") or ""),
            port_type=self._port_type(),
        )

    def _cache_detected_port(
        self,
        port: np.ndarray,
        tcp_pose: Pose,
        label: str,
        *,
        obs=None,
        vision: Optional[VisionPortEstimator] = None,
    ) -> None:
        """검출된 포트 base 좌표와 접근에 사용할 wrist orientation을 캐시에 저장한다."""
        self._cached_port_base = np.asarray(port, dtype=np.float64)
        self._target_orientation = self._target_wrist_orientation(tcp_pose)
        self._publish_triangulated_port_xyz(self._cached_port_base, label)
        if obs is not None and vision is not None:
            self._save_triangulation_debug_images(
                obs=obs,
                vision=vision,
                predicted_port=self._cached_port_base,
                label=label,
            )
        self.get_logger().info(
            f"{label}: detection cached, "
            f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
            f"axis={FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
            f"angle={self._manual_rotation_deg():+.2f}deg"
        )

    def _settle_after_lift_detect(self) -> None:
        """lift_up_detect 성공 후 approach로 넘어가기 전에 잠시 안정화한다."""
        settle_s = float(FinalPolicyConfig.LIFT_DETECT_TO_APPROACH_SETTLE_S)
        if settle_s <= 0.0:
            return
        self.get_logger().info(
            f"lift_up_detect settle before approach: {settle_s:.2f}s"
        )
        self.sleep_for(settle_s)

    def _cached_port_estimate(
        self,
        vision: VisionPortEstimator,
        target_class_id: int,
    ) -> Optional[np.ndarray]:
        """비동기 YOLO 워커가 이미 만든 포트 추정값을 기다리지 않고 확인한다."""
        return vision.cached_estimate(
            target_class_id,
            port_hint=str(getattr(self._task, "port_name", "") or ""),
            target_module_name=str(getattr(self._task, "target_module_name", "") or ""),
        )

    def _estimate_port(self, get_observation) -> Optional[np.ndarray]:
        """현재 task hint와 YOLO 비전 추정기로 목표 포트의 base 좌표를 반복 추정한다."""
        port_hint = str(getattr(self._task, "port_name", "") or "")
        target_module_name = str(getattr(self._task, "target_module_name", "") or "")
        port_type = self._port_type()
        target_class_id = self._target_class_id(port_type)
        vision = self._vision_for_port_type(port_type)
        for attempt in range(FinalPolicyConfig.APPROACH_VISION_RETRIES):
            obs = get_observation()
            port = vision.estimate(
                obs,
                target_class_id,
                port_hint=port_hint,
                target_module_name=target_module_name,
            )
            if port is not None:
                self.get_logger().info(
                    "YOLO port estimate: "
                    f"attempt={attempt + 1}, "
                    f"type={port_type}, "
                    f"target={target_module_name}, "
                    f"port={port_hint}, "
                    f"class_id={target_class_id}, "
                    f"base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f})"
                )
                return port
            self.sleep_for(FinalPolicyConfig.APPROACH_RETRY_DT)
        return None

    def _stage_lift_up_detect(self, get_observation, move_robot) -> bool:
        """lift_up 및 YOLO Triangulation을 동시에 수행, 검출 즉시 approach로 넘긴다."""
        port_type = self._port_type()
        target_class_id = self._target_class_id(port_type)
        port_hint = str(getattr(self._task, "port_name", "") or "")
        lift_m = float(FinalPolicyConfig.INITIAL_LIFT_M)
        self.get_logger().info(
            f"[lift_up_detect] Start: "
            f"type={port_type}, class_id={target_class_id}, dz={lift_m * 1000.0:.1f}mm"
        )

        self._vision_debug_save_enabled = True
        vision = self._vision_for_port_type(port_type)
        self._configure_detection_context(vision)

        if not vision.load_model():
            self.get_logger().error(
                f"lift_up_detect failed: {port_type.upper()} YOLO model load failed"
            )
            return False

        vision.start_detection(
            enable_debug_save=True,
            reset_counts=True,
            reset_cache=True,
        )

        try:
            obs = get_observation()
            latest_obs = obs
            start_pose = self._tcp_pose(obs)
            if start_pose is None:
                self.get_logger().error("lift_up_detect failed: missing TCP pose")
                return False

            if obs is not None:
                vision.request_estimate(
                    obs,
                    target_class_id,
                    port_hint=port_hint,
                )

            def finish_if_detected(current_pose: Pose, label: str) -> bool:
                port = self._cached_port_estimate(vision, target_class_id)
                if port is None:
                    return False
                self._cache_detected_port(
                    port,
                    current_pose,
                    label,
                    obs=latest_obs,
                    vision=vision,
                )
                self._settle_after_lift_detect()
                self.get_logger().info("[lift_up_detect] Done")
                return True

            if finish_if_detected(start_pose, "lift_up_detect initial"):
                return True

            target_pose = self._copy_pose(start_pose)
            target_pose.position.z += lift_m
            start = np.array(
                [start_pose.position.x, start_pose.position.y, start_pose.position.z],
                dtype=np.float64,
            )
            target = np.array(
                [target_pose.position.x, target_pose.position.y, target_pose.position.z],
                dtype=np.float64,
            )
            q_start = quat_to_tuple(start_pose.orientation)
            q_target = quat_to_tuple(target_pose.orientation)
            step_count = max(1, int(FinalPolicyConfig.INITIAL_LIFT_STEPS))
            current_pose = start_pose

            for index in range(step_count):
                if finish_if_detected(
                    current_pose,
                    f"lift_up_detect before_step_{index + 1}",
                ):
                    return True

                t = interp_profile((index + 1) / step_count, quintic=True)
                pos = start * (1.0 - t) + target * t
                quat = quaternion_slerp(q_start, q_target, t)
                pose = Pose(
                    position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                    orientation=tuple_to_quat(quat),
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=list(FinalPolicyConfig.APPROACH_NEAR_STIFFNESS),
                    damping=list(FinalPolicyConfig.APPROACH_NEAR_DAMPING),
                )
                if index == 0 or index == step_count - 1:
                    self.get_logger().info(
                        f"lift_up_detect: waypoint {index + 1}/{step_count} "
                        f"tcp=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
                    )
                self.sleep_for(FinalPolicyConfig.INITIAL_LIFT_DT)

                obs = get_observation()
                latest_obs = obs
                current_pose = self._tcp_pose(obs) or pose
                if obs is not None:
                    vision.request_estimate(
                        obs,
                        target_class_id,
                        port_hint=port_hint,
                    )
                if finish_if_detected(current_pose, f"lift_up_detect step_{index + 1}"):
                    return True

            self.get_logger().info(
                "lift_up_detect: lift completed before detection; waiting at lifted pose"
            )
            if FinalPolicyConfig.INITIAL_LIFT_SETTLE_S > 0:
                self.sleep_for(FinalPolicyConfig.INITIAL_LIFT_SETTLE_S)

            port = self._estimate_port(get_observation)
            obs = get_observation()
            current_pose = self._tcp_pose(obs) or target_pose
            if port is None:
                self.get_logger().error(
                    "lift_up_detect failed: YOLO port estimate unavailable"
                )
                return False
            self._cache_detected_port(
                port,
                current_pose,
                "lift_up_detect fallback",
                obs=obs,
                vision=vision,
            )
            self._settle_after_lift_detect()
            self.get_logger().info("[lift_up_detect] Done")
            return True
        finally:
            for estimator in self._vision_by_port_type.values():
                estimator.stop_detection()
                estimator.set_debug_save_enabled(False)
            self._vision_debug_save_enabled = False

    def _stage_approach(self, get_observation, move_robot) -> bool:
        """
            검출된 포트 앞의 목표 TCP 위치까지 단일 접근 경로로 이동한다.
        """
        self.get_logger().info("[approach] Start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("Approach failed: missing TCP pose")
            return False

        port = self._cached_port_base
        if port is None:
            self.get_logger().error("Approach failed: missing cached YOLO port estimate")
            return False

        target_orientation = self._target_orientation
        if target_orientation is None:
            target_orientation = self._target_wrist_orientation(start_pose)
            self._target_orientation = target_orientation

        tcp_offset = np.array(
            [
                FinalPolicyConfig.TCP_OFFSET_X,
                FinalPolicyConfig.TCP_OFFSET_Y,
                FinalPolicyConfig.TCP_OFFSET_Z,
            ],
            dtype=np.float64,
        )
        target_z_offset = float(FinalPolicyConfig.APPROACH_NEAR_Z_OFFSET_M)

        def make_approach_pose(z_offset: float) -> tuple[Pose, np.ndarray]:
            target = port + np.array([0.0, 0.0, z_offset], dtype=np.float64)
            target = target + tcp_offset
            return (
                Pose(
                    position=Point(
                        x=float(target[0]),
                        y=float(target[1]),
                        z=float(target[2]),
                    ),
                    orientation=self._copy_quaternion(target_orientation),
                ),
                target,
            )

        approach_pose, approach_target = make_approach_pose(target_z_offset)
        self.get_logger().info(
            "approach target: "
            f"z_plus={target_z_offset*1000:.1f}mm, "
            f"tcp_offset=({tcp_offset[0]*1000:+.1f}, "
            f"{tcp_offset[1]*1000:+.1f}, {tcp_offset[2]*1000:+.1f})mm, "
            f"target_tcp=({approach_target[0]:+.4f}, "
            f"{approach_target[1]:+.4f}, {approach_target[2]:+.4f})"
        )
        self._follow_pose(
            move_robot=move_robot,
            start_pose=start_pose,
            target_pose=approach_pose,
            steps=FinalPolicyConfig.APPROACH_STEPS,
            stiffness=FinalPolicyConfig.APPROACH_STIFFNESS,
            damping=FinalPolicyConfig.APPROACH_DAMPING,
            dt=FinalPolicyConfig.APPROACH_DT,
            label="approach",
        )
        if FinalPolicyConfig.APPROACH_SETTLE_S > 0:
            self.get_logger().info(
                f"approach settle: {FinalPolicyConfig.APPROACH_SETTLE_S:.2f}s"
            )
            self.sleep_for(FinalPolicyConfig.APPROACH_SETTLE_S)
        self.get_logger().info("[approach] Done")
        return True

    def _tip_target_class_id(self) -> int:
        """현재 connector의 케이블 tip을 검출할 YOLO class id를 반환한다."""
        if self._port_type() == "sc":
            return int(FinalPolicyConfig.SC_TIP_TARGET_CLASS_ID)
        return int(FinalPolicyConfig.SFP_TIP_TARGET_CLASS_ID)

    def _triangulation_target_xy(self) -> np.ndarray:
        """정렬 완료 시 기대하는 port-tip XY 잔차를 connector별로 반환한다."""
        if self._port_type() == "sc":
            return np.array(
                [
                    FinalPolicyConfig.SC_TRIANGULATION_TARGET_DX_M,
                    FinalPolicyConfig.SC_TRIANGULATION_TARGET_DY_M,
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                FinalPolicyConfig.SFP_TRIANGULATION_TARGET_DX_M,
                FinalPolicyConfig.SFP_TRIANGULATION_TARGET_DY_M,
            ],
            dtype=np.float64,
        )

    def _estimate_align_triangulation_xy(
        self,
        get_observation,
        vision: VisionPortEstimator,
    ) -> Optional[np.ndarray]:
        """fresh port triangulation과 visual/kinematic tip으로 base_link XY를 검증한다."""
        port_class_id = self._target_class_id(self._port_type())
        tip_class_id = self._tip_target_class_id()
        port_hint = str(getattr(self._task, "port_name", "") or "")
        module_name = str(getattr(self._task, "target_module_name", "") or "")
        retries = max(1, int(FinalPolicyConfig.ALIGN_TRIANGULATION_RETRIES))
        visual_tip_supported = vision.supports_class_id(tip_class_id)

        for attempt in range(retries):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if obs is None or tcp_pose is None:
                self.sleep_for(FinalPolicyConfig.ALIGN_TRIANGULATION_RETRY_DT)
                continue

            port = vision.estimate(
                obs,
                port_class_id,
                port_hint=port_hint,
                target_module_name=module_name,
            )
            if port is None:
                self.get_logger().warn(
                    "align triangulation unavailable: "
                    f"attempt={attempt + 1}/{retries}, port=false"
                )
                self.sleep_for(FinalPolicyConfig.ALIGN_TRIANGULATION_RETRY_DT)
                continue

            port = np.asarray(port, dtype=np.float64).reshape(3)
            if visual_tip_supported:
                tip_candidates = vision.estimate_all(obs, tip_class_id)
                if not tip_candidates:
                    self.get_logger().warn(
                        "align triangulation unavailable: "
                        f"attempt={attempt + 1}/{retries}, visual_tip=false"
                    )
                    self.sleep_for(FinalPolicyConfig.ALIGN_TRIANGULATION_RETRY_DT)
                    continue
                tip = np.asarray(
                    min(
                        tip_candidates,
                        key=lambda candidate: float(
                            np.linalg.norm(np.asarray(candidate["pos"]) - port)
                        ),
                    )["pos"],
                    dtype=np.float64,
                ).reshape(3)
                correction_xy = (
                    port[:2] - tip[:2] - self._triangulation_target_xy()
                )
                source = "visual_port_tip"
            elif FinalPolicyConfig.ALIGN_TRIANGULATION_ALLOW_TCP_TIP_FALLBACK:
                tcp_xy = np.array(
                    [tcp_pose.position.x, tcp_pose.position.y],
                    dtype=np.float64,
                )
                tcp_from_tip_xy = np.array(
                    [
                        FinalPolicyConfig.ALIGN_TCP_FROM_TIP_X_M,
                        FinalPolicyConfig.ALIGN_TCP_FROM_TIP_Y_M,
                    ],
                    dtype=np.float64,
                )
                tip = np.array(
                    [
                        tcp_xy[0] - tcp_from_tip_xy[0],
                        tcp_xy[1] - tcp_from_tip_xy[1],
                        tcp_pose.position.z,
                    ],
                    dtype=np.float64,
                )
                correction_xy = port[:2] - tip[:2]
                source = "triangulated_port+tcp_tip"
            else:
                self.get_logger().error(
                    "align triangulation requires a visual tip class, but the current "
                    f"YOLO checkpoint does not contain class_id={tip_class_id}"
                )
                return None

            if not np.isfinite(correction_xy).all():
                continue
            self.get_logger().info(
                "align triangulation: "
                f"source={source}, "
                f"port=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
                f"tip=({tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f}), "
                f"correction_xy=({correction_xy[0]*1000:+.2f}, "
                f"{correction_xy[1]*1000:+.2f})mm"
            )
            return correction_xy

        return None

    @staticmethod
    def _scaled_clamped_vector(
        vector: np.ndarray,
        *,
        gain: float,
        max_norm: float,
    ) -> np.ndarray:
        """보정 벡터에 gain을 적용한 뒤 전체 norm을 한 번의 최대 이동량으로 제한한다."""
        step = np.asarray(vector, dtype=np.float64).copy()
        step *= max(0.0, float(gain))
        step_norm = float(np.linalg.norm(step))
        norm_limit = max(0.0, float(max_norm))
        if step_norm > norm_limit and step_norm > 1e-12:
            step *= norm_limit / step_norm
        return step

    @staticmethod
    def _prediction_window_spread(
        predictions: list[np.ndarray],
    ) -> tuple[float, float]:
        """정지 상태 예측 창에서 XY/RPY가 중앙값 주변으로 퍼진 최대 거리를 계산한다."""
        values = np.asarray(predictions, dtype=np.float64)
        center = np.median(values, axis=0)
        xy_spread = float(np.max(np.linalg.norm(values[:, :2] - center[:2], axis=1)))
        rpy_spread = float(np.max(np.linalg.norm(values[:, 2:] - center[2:], axis=1)))
        return xy_spread, rpy_spread

    def _stage_align(self, get_observation, move_robot) -> bool:
        """5D/6D vision-offset과 port-tip 삼각측량으로 XY/RPY를 정렬한다."""
        self.get_logger().info(
            f"[vision_offset_align] Start ({self._port_type().upper()})"
        )
        vision_offset_predictor = self._vision_offset_predictor_for_align()
        vision = self._vision_for_port_type(self._port_type())
        self._configure_detection_context(vision)
        validation_steps = max(
            1,
            int(FinalPolicyConfig.STABLE_STEPS),
            int(FinalPolicyConfig.VISION_OFFSET_VALIDATION_STEPS),
        )
        stable_predictions: list[np.ndarray] = []
        last_xy_error = None
        last_z_error = None
        last_rpy_error = None
        align_z = None

        for step in range(FinalPolicyConfig.ALIGN_MAX_STEPS):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if tcp_pose is None:
                self.sleep_for(FinalPolicyConfig.DT)
                continue
            if align_z is None:
                align_z = float(tcp_pose.position.z)
                self.get_logger().info(
                    f"vision_offset_align: fixed_z={align_z:+.4f}m"
                )

            correction = vision_offset_predictor.predict(obs)
            if correction is None:
                self.sleep_for(FinalPolicyConfig.DT)
                continue

            correction = np.asarray(correction, dtype=np.float64).reshape(-1)
            if correction.size != 6 or not np.isfinite(correction).all():
                self.get_logger().warn(
                    f"vision_offset_align[{step:03d}]: invalid correction={correction}"
                )
                self.sleep_for(FinalPolicyConfig.DT)
                continue

            position_correction = correction[:3]
            rpy_correction = correction[3:]
            if (
                float(np.max(np.abs(position_correction)))
                > float(FinalPolicyConfig.VISION_OFFSET_MAX_ABS_POSITION_M)
                or float(np.max(np.abs(rpy_correction)))
                > float(FinalPolicyConfig.VISION_OFFSET_MAX_ABS_RPY_RAD)
            ):
                stable_predictions.clear()
                self.get_logger().warn(
                    f"vision_offset_align[{step:03d}]: prediction rejected, "
                    f"xyz=({position_correction[0]*1000:+.2f}, "
                    f"{position_correction[1]*1000:+.2f}, "
                    f"{position_correction[2]*1000:+.2f})mm, "
                    f"rpy=({math.degrees(rpy_correction[0]):+.2f}, "
                    f"{math.degrees(rpy_correction[1]):+.2f}, "
                    f"{math.degrees(rpy_correction[2]):+.2f})deg"
                )
                self.sleep_for(FinalPolicyConfig.DT)
                continue

            position_xy_error = float(np.linalg.norm(position_correction[:2]))
            position_z_error = abs(float(position_correction[2]))
            rpy_error = float(np.linalg.norm(rpy_correction))
            last_xy_error = position_xy_error
            last_z_error = position_z_error
            last_rpy_error = rpy_error
            model_within_threshold = (
                position_xy_error < FinalPolicyConfig.VISION_OFFSET_XY_TOL_M
                and rpy_error < FinalPolicyConfig.VISION_OFFSET_RPY_TOL_RAD
            )

            if model_within_threshold:
                stable_predictions.append(
                    np.concatenate((position_correction[:2], rpy_correction))
                )
                stable_predictions = stable_predictions[-validation_steps:]
            else:
                stable_predictions.clear()

            stable_count = len(stable_predictions)
            xy_spread = float("inf")
            rpy_spread = float("inf")
            model_window_stable = False
            if stable_count >= validation_steps:
                xy_spread, rpy_spread = self._prediction_window_spread(
                    stable_predictions
                )
                model_window_stable = (
                    xy_spread <= FinalPolicyConfig.VISION_OFFSET_XY_SPREAD_TOL_M
                    and rpy_spread
                    <= FinalPolicyConfig.VISION_OFFSET_RPY_SPREAD_TOL_RAD
                )

            if model_within_threshold:
                hold_pose = self._copy_pose(tcp_pose)
                hold_pose.position.z = align_z
                self._save_align_debug_images(
                    obs=obs,
                    vision=vision,
                    tcp_pose=tcp_pose,
                    target_pose=hold_pose,
                    step_index=step,
                    position_correction=position_correction,
                    rpy_correction=rpy_correction,
                    step_xyz=np.zeros(3, dtype=np.float64),
                    step_rpy=np.zeros(3, dtype=np.float64),
                    position_xy_error=position_xy_error,
                    position_z_error=position_z_error,
                    rpy_error=rpy_error,
                    stable_count=stable_count,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=hold_pose,
                    stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                    damping=list(FinalPolicyConfig.ALIGN_DAMPING),
                )

                if model_window_stable:
                    if not FinalPolicyConfig.ALIGN_TRIANGULATION_ENABLED:
                        self.get_logger().info(
                            "vision_offset_align stable: triangulation disabled, "
                            f"xy_norm={position_xy_error*1000:.2f}mm, "
                            f"rpy_norm={math.degrees(rpy_error):.2f}deg"
                        )
                        return True

                    triangulation_xy = self._estimate_align_triangulation_xy(
                        get_observation,
                        vision,
                    )
                    if triangulation_xy is None:
                        if FinalPolicyConfig.ALIGN_TRIANGULATION_REQUIRED:
                            stable_predictions.clear()
                            self.get_logger().warn(
                                "vision_offset_align: model stable but triangulation "
                                "is required and unavailable"
                            )
                            self.sleep_for(
                                FinalPolicyConfig.VISION_OFFSET_VALIDATION_DT
                            )
                            continue
                        self.get_logger().warn(
                            "vision_offset_align stable: triangulation unavailable "
                            "(<2 visible cameras); accepting stationary model window"
                        )
                        return True

                    triangulation_norm = float(np.linalg.norm(triangulation_xy))
                    if (
                        triangulation_norm
                        <= FinalPolicyConfig.ALIGN_TRIANGULATION_XY_TOL_M
                    ):
                        self.get_logger().info(
                            "vision_offset_align stable: "
                            f"model_xy={position_xy_error*1000:.2f}mm, "
                            f"triangulation_xy={triangulation_norm*1000:.2f}mm, "
                            f"rpy_norm={math.degrees(rpy_error):.2f}deg, "
                            f"spread_xy={xy_spread*1000:.2f}mm, "
                            f"spread_rpy={math.degrees(rpy_spread):.2f}deg"
                        )
                        return True

                    triangulation_step = self._scaled_clamped_vector(
                        triangulation_xy,
                        gain=FinalPolicyConfig.ALIGN_TRIANGULATION_MOVE_GAIN,
                        max_norm=FinalPolicyConfig.ALIGN_TRIANGULATION_MAX_STEP_M,
                    )
                    target_pose = self._copy_pose(tcp_pose)
                    target_pose.position.x += float(triangulation_step[0])
                    target_pose.position.y += float(triangulation_step[1])
                    target_pose.position.z = align_z
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=target_pose,
                        stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                        damping=list(FinalPolicyConfig.ALIGN_DAMPING),
                    )
                    self.get_logger().warn(
                        "vision_offset_align: model/triangulation disagreement, "
                        f"triangulation_xy={triangulation_norm*1000:.2f}mm, "
                        f"move_xy=({triangulation_step[0]*1000:+.2f}, "
                        f"{triangulation_step[1]*1000:+.2f})mm"
                    )
                    stable_predictions.clear()
                    self.sleep_for(FinalPolicyConfig.COMMAND_SETTLE_S)
                    continue

                spread_text = (
                    "pending"
                    if not np.isfinite(xy_spread)
                    else f"xy={xy_spread*1000:.2f}mm, "
                    f"rpy={math.degrees(rpy_spread):.2f}deg"
                )
                self.get_logger().info(
                    f"vision_offset_align[{step:03d}]: validating stationary "
                    f"predictions {stable_count}/{validation_steps}, "
                    f"spread={spread_text}"
                )
                self.sleep_for(FinalPolicyConfig.VISION_OFFSET_VALIDATION_DT)
                continue

            # 5D는 z=0으로 정규화되고, 기존 6D의 z도 ALIGN 이동에는 사용하지 않는다.
            step_xy = self._scaled_clamped_vector(
                position_correction[:2],
                gain=FinalPolicyConfig.VISION_OFFSET_XY_MOVE_GAIN,
                max_norm=FinalPolicyConfig.VISION_OFFSET_MAX_XY_STEP_M,
            )
            step_xyz = np.array(
                [step_xy[0], step_xy[1], 0.0],
                dtype=np.float64,
            )
            step_rpy = self._scaled_clamped_vector(
                rpy_correction,
                gain=FinalPolicyConfig.VISION_OFFSET_RPY_MOVE_GAIN,
                max_norm=FinalPolicyConfig.VISION_OFFSET_MAX_RPY_STEP_RAD,
            )
            target_pose = self._copy_pose(tcp_pose)
            target_pose.position.x += float(step_xyz[0])
            target_pose.position.y += float(step_xyz[1])
            target_pose.position.z = align_z
            q_delta = self._rpy_delta_quat_base(step_rpy)
            q_target = self._normalize_quat(
                quaternion_multiply(q_delta, quat_to_tuple(tcp_pose.orientation))
            )
            target_pose.orientation = tuple_to_quat(q_target)

            self._save_align_debug_images(
                obs=obs,
                vision=vision,
                tcp_pose=tcp_pose,
                target_pose=target_pose,
                step_index=step,
                position_correction=position_correction,
                rpy_correction=rpy_correction,
                step_xyz=step_xyz,
                step_rpy=step_rpy,
                position_xy_error=position_xy_error,
                position_z_error=position_z_error,
                rpy_error=rpy_error,
                stable_count=0,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                damping=list(FinalPolicyConfig.ALIGN_DAMPING),
            )
            self.get_logger().info(
                f"vision_offset_align[{step:03d}]: "
                f"pred_xy=({position_correction[0]*1000:+.2f}, "
                f"{position_correction[1]*1000:+.2f})mm, "
                f"move_xy=({step_xyz[0]*1000:+.2f}, "
                f"{step_xyz[1]*1000:+.2f})mm, "
                f"z_ignored={position_correction[2]*1000:+.2f}mm, "
                f"pred_rpy=({math.degrees(rpy_correction[0]):+.2f}, "
                f"{math.degrees(rpy_correction[1]):+.2f}, "
                f"{math.degrees(rpy_correction[2]):+.2f})deg, "
                f"move_rpy=({math.degrees(step_rpy[0]):+.2f}, "
                f"{math.degrees(step_rpy[1]):+.2f}, "
                f"{math.degrees(step_rpy[2]):+.2f})deg"
            )
            self.sleep_for(FinalPolicyConfig.COMMAND_SETTLE_S)

        if last_xy_error is None or last_rpy_error is None:
            self.get_logger().error("vision_offset_align failed: no model predictions")
            return False
        self.get_logger().error(
            "vision_offset_align failed: validation never completed, "
            f"last_xy_norm={last_xy_error*1000:.2f}mm, "
            f"last_z_ignored={(last_z_error or 0.0)*1000:.2f}mm, "
            f"last_rpy_norm={math.degrees(last_rpy_error):.2f}deg"
        )
        return False

    def _stage_insert(self, get_observation, move_robot) -> bool:
        """align 성공 pose에서 x/y와 자세를 고정하고 z 방향으로만 바로 내려 삽입한다."""
        self.get_logger().info("[insert] start")
        obs = get_observation()
        start_pose = self._tcp_pose(obs)
        if start_pose is None:
            self.get_logger().error("insert failed: missing TCP pose")
            return False

        max_depth = float(FinalPolicyConfig.MAX_INSERT_DEPTH_M)
        step_m = min(
            float(FinalPolicyConfig.INSERT_STEP_M),
            float(FinalPolicyConfig.MAX_DOWN_STEP_M),
        )
        max_steps = min(
            int(math.ceil(max_depth / max(step_m, 1e-6))),
            int(FinalPolicyConfig.INSERT_MAX_STEPS),
        )
        fixed_x = float(start_pose.position.x)
        fixed_y = float(start_pose.position.y)
        start_z = float(start_pose.position.z)
        fixed_orientation = self._copy_quaternion(start_pose.orientation)
        self.get_logger().info(
            "insert vertical descent: "
            f"depth={max_depth * 1000.0:.1f}mm, "
            f"step={step_m * 1000.0:.2f}mm, "
            f"max_steps={max_steps}, "
            f"fixed_xy=({fixed_x:+.4f}, {fixed_y:+.4f})"
        )

        for inserted_steps in range(max_steps):
            target_pose = self._copy_pose(start_pose)
            target_pose.position.x = fixed_x
            target_pose.position.y = fixed_y
            target_pose.position.z = float(start_z - (inserted_steps + 1) * step_m)
            target_pose.orientation = self._copy_quaternion(fixed_orientation)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(self._insertion_stiffness()),
                damping=list(self._insertion_damping()),
            )
            if inserted_steps == 0 or inserted_steps % 10 == 0:
                self.get_logger().info(
                    f"insert[{inserted_steps:03d}]: "
                    f"dz={-(inserted_steps + 1) * step_m * 1000.0:.1f}mm"
                )
            self.sleep_for(FinalPolicyConfig.INSERT_DT)

        if FinalPolicyConfig.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(FinalPolicyConfig.SETTLE_AFTER_INSERT_S)
        self.get_logger().info("[insert] done")
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._task = task
        self._send_feedback = send_feedback
        self._cached_port_base = None
        self._target_orientation = None
        self._fixed_target_orientation = None
        self._align_debug_call_count = 0
        self.get_logger().info(
            "FinalPolicy Start: "
            f"target={task.target_module_name}, port={task.port_name}, "
            f"cable={task.cable_name}, plug={task.plug_name}"
        )
        triangulation_eval_only = bool(FinalPolicyConfig.TRIANGULATION_EVAL_ONLY)
        if triangulation_eval_only:
            self.get_logger().info(
                "FinalPolicy triangulation eval-only mode: "
                "will stop after lift_up_detect"
            )
            if not FinalPolicyConfig.PUBLISH_TRIANGULATED_PORT_XYZ:
                self.get_logger().warn(
                    "AIC_TRIANGULATION_EVAL_ONLY=1 but "
                    "AIC_PUBLISH_TRIANGULATED_PORT_XYZ is disabled"
                )
        try:
            self._preload_detection_model_for_current_task()
        except Exception as exc:
            self.get_logger().error(
                format_model_log(f"FinalPolicy initial YOLO model load failed: {exc}")
            )
            send_feedback("failed: load YOLO model")
            return False
        self._start_background_yolo_model_downloads()

        if not triangulation_eval_only:
            try:
                self._vision_offset_predictor_for_align()
            except Exception as exc:
                self.get_logger().error(
                    format_model_log(
                        f"FinalPolicy initial vision-offset model load failed: {exc}"
                    )
                )
                send_feedback("failed: load vision_offset model")
                return False
            self._start_background_vision_offset_model_downloads()

        stages = [
            ("lift_up_detect", lambda: self._stage_lift_up_detect(get_observation, move_robot)),
        ]
        if not triangulation_eval_only:
            stages.extend(
                [
                    ("approach", lambda: self._stage_approach(get_observation, move_robot)),
                    ("vision_offset_align", lambda: self._stage_align(get_observation, move_robot)),
                    ("insert", lambda: self._stage_insert(get_observation, move_robot)),
                ]
            )
        for name, stage in stages:
            send_feedback(f"Final Policy: {name}")
            try:
                if not stage():
                    self.get_logger().error(f"FinalPolicy failed at stage: {name}")
                    send_feedback(f"failed: {name}")
                    return False
            except Exception as exc:
                self.get_logger().error(f"FinalPolicy exception at {name}: {exc}")
                send_feedback(f"failed: {name} exception")
                return False
        if triangulation_eval_only:
            send_feedback("Final Policy: triangulation eval done")
            self.get_logger().info(
                "FinalPolicy triangulation eval done; waiting for next task"
            )
            return True
        send_feedback("Final Policy: done")
        self.get_logger().info("FinalPolicy done")
        return True
