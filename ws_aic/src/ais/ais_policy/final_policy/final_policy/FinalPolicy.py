from __future__ import annotations

import math
import os
import threading
import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from final_policy.config import FinalPolicyConfig
from final_policy.geometry import (
    interp_profile,
    project_3d_to_pixel,
    quat_to_tuple,
    tuple_to_quat,
)
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


class FinalPolicy(Policy):
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
        self._yolo_download_threads: dict[str, threading.Thread] = {}
        self._yolo_download_lock = threading.Lock()
        self._vision_offset_predictor_by_port_type: dict[str, VisionOffsetPredictor] = {}
        self._vision_offset_download_threads: dict[str, threading.Thread] = {}
        self._vision_offset_download_lock = threading.Lock()
        self._send_feedback: Optional[SendFeedbackCallback] = None
        self.get_logger().info(
            "FinalPolicy ready: "
            "yolo_model=initial_load, "
            "vision_offset_model=initial_load, "
            "background_download=enabled"
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

    def _align_debug_save_dir(self) -> Optional[Path]:
        """align 디버그 이미지를 저장할 디렉토리를 반환한다."""
        debug_root = getattr(VisionPortEstimator, "DEBUG_SAVE_DIR", None)
        if not debug_root:
            return None
        return Path(debug_root) / "align"

    @staticmethod
    def _pose_position_array(pose: Pose) -> np.ndarray:
        """Pose의 position을 base_link 3D numpy 벡터로 변환한다."""
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )

    @staticmethod
    def _clip_pixel(pixel: np.ndarray, width: int, height: int) -> tuple[int, int]:
        """이미지 밖으로 나간 픽셀 좌표를 이미지 경계 안으로 제한한다."""
        return (
            int(np.clip(pixel[0], 0, max(0, width - 1))),
            int(np.clip(pixel[1], 0, max(0, height - 1))),
        )

    @staticmethod
    def _draw_align_arrow(
        image: np.ndarray,
        start_px: np.ndarray,
        end_px: np.ndarray,
        label: str,
    ) -> bool:
        """align 추론 이동 방향을 이미지 위 화살표와 X 목표점으로 그린다."""
        import cv2

        height, width = image.shape[:2]
        start_px = np.asarray(start_px, dtype=np.float64)
        end_px = np.asarray(end_px, dtype=np.float64)
        if not (np.isfinite(start_px).all() and np.isfinite(end_px).all()):
            return False

        delta = end_px - start_px
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            delta = np.array([1.0, 0.0], dtype=np.float64)
            delta_norm = 1.0

        start_in_image = 0 <= start_px[0] < width and 0 <= start_px[1] < height
        end_in_image = 0 <= end_px[0] < width and 0 <= end_px[1] < height
        if start_in_image and end_in_image:
            arrow_start = start_px
            arrow_end = end_px
        else:
            arrow_start = np.array([width * 0.5, height * 0.5], dtype=np.float64)
            arrow_len = min(max(delta_norm, 36.0), 120.0)
            arrow_end = arrow_start + delta / delta_norm * arrow_len
            label = f"{label} screen-scaled"

        p0 = FinalPolicy._clip_pixel(arrow_start, width, height)
        p1 = FinalPolicy._clip_pixel(arrow_end, width, height)
        for thickness, color in ((9, (0, 0, 0)), (5, (0, 255, 255))):
            cv2.arrowedLine(
                image,
                p0,
                p1,
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        cross_size = 14
        for thickness, color in ((7, (0, 0, 0)), (3, (0, 255, 255))):
            cv2.line(
                image,
                (p1[0] - cross_size, p1[1] - cross_size),
                (p1[0] + cross_size, p1[1] + cross_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                image,
                (p1[0] - cross_size, p1[1] + cross_size),
                (p1[0] + cross_size, p1[1] - cross_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.circle(image, p0, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p0, 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (max(8, p1[0] + 8), max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (max(8, p1[0] + 8), max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return True

    @staticmethod
    def _draw_axis_arrow(
        image: np.ndarray,
        start_px: np.ndarray,
        end_px: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> bool:
        """카메라 이미지 위에 base 축 방향 참조 화살표를 그린다."""
        import cv2

        height, width = image.shape[:2]
        start_px = np.asarray(start_px, dtype=np.float64)
        end_px = np.asarray(end_px, dtype=np.float64)
        if not (np.isfinite(start_px).all() and np.isfinite(end_px).all()):
            return False
        if np.any(start_px < 0.0) or np.any(end_px < 0.0):
            return False

        start_in_image = 0 <= start_px[0] < width and 0 <= start_px[1] < height
        end_in_image = 0 <= end_px[0] < width and 0 <= end_px[1] < height
        if not (start_in_image or end_in_image):
            return False

        p0 = FinalPolicy._clip_pixel(start_px, width, height)
        p1 = FinalPolicy._clip_pixel(end_px, width, height)
        for thickness, line_color in ((7, (0, 0, 0)), (3, color)):
            cv2.arrowedLine(
                image,
                p0,
                p1,
                line_color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        text_origin = (max(8, p1[0] + 6), max(18, p1[1] - 6))
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        return True

    def _draw_align_axes(
        self,
        image: np.ndarray,
        anchor_base: np.ndarray,
        k_matrix: np.ndarray,
        t_camera_base: np.ndarray,
    ) -> None:
        """align debug 이미지에 base_link +X/+Y 방향을 카메라 투영으로 표시한다."""
        axis_length = float(os.environ.get("AIC_ALIGN_DEBUG_AXIS_LENGTH_M", "0.03"))
        anchor_base = np.asarray(anchor_base, dtype=np.float64)
        anchor_px = np.array(
            project_3d_to_pixel(anchor_base, k_matrix, t_camera_base),
            dtype=np.float64,
        )
        axes = (
            ("+X base", np.array([axis_length, 0.0, 0.0], dtype=np.float64), (0, 0, 255)),
            ("+Y base", np.array([0.0, axis_length, 0.0], dtype=np.float64), (0, 255, 0)),
        )
        for label, axis, color in axes:
            axis_px = np.array(
                project_3d_to_pixel(anchor_base + axis, k_matrix, t_camera_base),
                dtype=np.float64,
            )
            self._draw_axis_arrow(image, anchor_px, axis_px, label, color)

    def _save_align_debug_images(
        self,
        *,
        obs,
        vision: VisionPortEstimator,
        tcp_pose: Pose,
        target_pose: Pose,
        step_index: int,
        position_correction: np.ndarray,
        rpy_correction: np.ndarray,
        step_xyz: np.ndarray,
        position_xy_error: float,
        position_z_error: float,
        rpy_error: float,
        stable_count: int,
    ) -> None:
        """align 추론 보정량과 실제 이동 목표를 카메라 이미지 위에 저장한다."""
        save_dir = self._align_debug_save_dir()
        if save_dir is None or obs is None:
            return

        try:
            import cv2

            frame_id = self._align_debug_call_count
            self._align_debug_call_count += 1
            if frame_id == 0:
                self.get_logger().info(f"[Align Debug] dir: {save_dir}")
            task_label = VisionPortEstimator._sanitize_debug_token(
                getattr(vision, "debug_task_label", "task_unknown")
            ) or "task_unknown"
            tcp_base = self._pose_position_array(tcp_pose)
            default_target_base = self._pose_position_array(target_pose)
            if self._cached_port_base is not None:
                move_anchor_base = np.asarray(self._cached_port_base, dtype=np.float64)
                move_target_base = move_anchor_base + np.asarray(
                    step_xyz,
                    dtype=np.float64,
                )
            else:
                move_anchor_base = tcp_base
                move_target_base = default_target_base

            for cam_name, _ in VisionPortEstimator.CAMERAS:
                img_msg = getattr(obs, f"{cam_name}_image", None)
                camera_info = getattr(obs, f"{cam_name}_camera_info", None)
                if img_msg is None or camera_info is None:
                    continue

                debug_img = VisionPortEstimator._image_from_msg(img_msg).copy()
                k_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
                t_camera_base = vision._base_to_camera_optical_matrix(obs, cam_name)
                arrow_drawn = False
                if t_camera_base is not None:
                    self._draw_align_axes(
                        debug_img,
                        move_anchor_base,
                        k_matrix,
                        t_camera_base,
                    )
                    start_px = np.array(
                        project_3d_to_pixel(move_anchor_base, k_matrix, t_camera_base),
                        dtype=np.float64,
                    )
                    target_px = np.array(
                        project_3d_to_pixel(move_target_base, k_matrix, t_camera_base),
                        dtype=np.float64,
                    )
                    if np.all(start_px >= 0.0) and np.all(target_px >= 0.0):
                        arrow_drawn = self._draw_align_arrow(
                            debug_img,
                            start_px,
                            target_px,
                            "OFFSET base-vector",
                        )

                if not arrow_drawn:
                    height, width = debug_img.shape[:2]
                    center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
                    base_xy = np.array(
                        [float(step_xyz[0]), -float(step_xyz[1])],
                        dtype=np.float64,
                    )
                    base_norm = float(np.linalg.norm(base_xy))
                    if base_norm < 1e-9:
                        base_xy = np.array([1.0, 0.0], dtype=np.float64)
                        base_norm = 1.0
                    length = min(max(base_norm * 4000.0, 36.0), 120.0)
                    self._draw_align_arrow(
                        debug_img,
                        center,
                        center + base_xy / base_norm * length,
                        "OFFSET base-xy",
                    )

                VisionPortEstimator._put_text_lines(
                    debug_img,
                    [
                        f"task={task_label} align_step={step_index:03d} cam={cam_name}",
                        (
                            "offset_xyz="
                            f"({position_correction[0] * 1000.0:+.1f}, "
                            f"{position_correction[1] * 1000.0:+.1f}, "
                            f"{position_correction[2] * 1000.0:+.1f})mm"
                        ),
                        (
                            "offset_rpy="
                            f"({math.degrees(rpy_correction[0]):+.1f}, "
                            f"{math.degrees(rpy_correction[1]):+.1f}, "
                            f"{math.degrees(rpy_correction[2]):+.1f})deg"
                        ),
                        (
                            f"xy_norm={position_xy_error * 1000.0:.1f}mm "
                            f"z_offset={position_z_error * 1000.0:.1f}mm "
                            f"rpy_norm={math.degrees(rpy_error):.1f}deg "
                            f"stable={stable_count}/{FinalPolicyConfig.STABLE_STEPS}"
                        ),
                    ],
                    10,
                    24,
                )

                fname = (
                    save_dir
                    / VisionPortEstimator._sanitize_debug_token(cam_name or "camera")
                    / f"{task_label}__align_{frame_id:04d}.jpg"
                )
                os.makedirs(fname.parent, exist_ok=True)
                if not cv2.imwrite(str(fname), debug_img):
                    self.get_logger().warn(f"[Align Debug] save failed: {fname}")
        except Exception as exc:
            self.get_logger().warn(f"[Align Debug] save failed: {exc}")

    def _cache_detected_port(self, port: np.ndarray, tcp_pose: Pose, label: str) -> None:
        """검출된 포트 base 좌표와 접근에 사용할 wrist orientation을 캐시에 저장한다."""
        self._cached_port_base = np.asarray(port, dtype=np.float64)
        self._target_orientation = self._target_wrist_orientation(tcp_pose)
        self.get_logger().info(
            f"{label}: detection cached, "
            f"port_base=({port[0]:+.4f}, {port[1]:+.4f}, {port[2]:+.4f}), "
            f"axis={FinalPolicyConfig.APPROACH_SFP_MANUAL_ROTATION_AXIS}, "
            f"angle={self._manual_rotation_deg():+.2f}deg"
        )

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
                self._cache_detected_port(port, current_pose, label)
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
            self._cache_detected_port(port, current_pose, "lift_up_detect fallback")
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

    def _stage_align(self, get_observation, move_robot) -> bool:
        """vision-offset 모델의 6D base_link 보정값으로 포트와 plug tip을 정렬한다."""
        self.get_logger().info(
            f"[vision_offset_align] Start ({self._port_type().upper()})"
        )
        vision_offset_predictor = self._vision_offset_predictor_for_align()
        vision = self._vision_for_port_type(self._port_type())
        self._configure_detection_context(vision)
        stable_count = 0
        last_xy_error = None
        last_z_error = None
        last_rpy_error = None

        for step in range(FinalPolicyConfig.ALIGN_MAX_STEPS):
            obs = get_observation()
            tcp_pose = self._tcp_pose(obs)
            if tcp_pose is None:
                self.sleep_for(FinalPolicyConfig.DT)
                continue

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

            if (
                position_xy_error < FinalPolicyConfig.VISION_OFFSET_XY_TOL_M
                and rpy_error < FinalPolicyConfig.VISION_OFFSET_RPY_TOL_RAD
            ):
                stable_count += 1
            else:
                stable_count = 0

            step_xyz = position_correction
            step_rpy = rpy_correction

            target_pose = self._copy_pose(tcp_pose)
            target_pose.position.x += float(step_xyz[0])
            target_pose.position.y += float(step_xyz[1])
            target_pose.position.z += float(step_xyz[2])
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
                position_xy_error=position_xy_error,
                position_z_error=position_z_error,
                rpy_error=rpy_error,
                stable_count=stable_count,
            )
            if stable_count >= FinalPolicyConfig.STABLE_STEPS:
                self.get_logger().info(
                    "vision_offset_align stable: "
                    f"xy_norm={position_xy_error * 1000.0:.2f}mm, "
                    f"z_offset={position_z_error * 1000.0:.2f}mm, "
                    f"rpy_norm={math.degrees(rpy_error):.2f}deg"
                )
                return True

            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(FinalPolicyConfig.ALIGN_STIFFNESS),
                damping=list(FinalPolicyConfig.ALIGN_DAMPING),
            )
            self.get_logger().info(
                f"vision_offset_align[{step:03d}]: "
                f"offset_xyz=({position_correction[0]*1000:+.2f}, "
                f"{position_correction[1]*1000:+.2f}, "
                f"{position_correction[2]*1000:+.2f})mm, "
                f"offset_rpy=({math.degrees(rpy_correction[0]):+.2f}, "
                f"{math.degrees(rpy_correction[1]):+.2f}, "
                f"{math.degrees(rpy_correction[2]):+.2f})deg, "
                f"xy_norm={position_xy_error * 1000.0:.2f}mm, "
                f"z_offset={position_z_error * 1000.0:.2f}mm, "
                f"stable={stable_count}/{FinalPolicyConfig.STABLE_STEPS}"
            )
            self.sleep_for(FinalPolicyConfig.COMMAND_SETTLE_S)

        if last_xy_error is None or last_rpy_error is None:
            self.get_logger().error("vision_offset_align failed: no model predictions")
            return False
        success = (
            last_xy_error < FinalPolicyConfig.VISION_OFFSET_XY_TOL_M * 1.5
            and last_rpy_error < FinalPolicyConfig.VISION_OFFSET_RPY_TOL_RAD * 1.5
        )
        self.get_logger().info(
            f"[vision_offset_align] done: success={success}, "
            f"last_xy_norm={last_xy_error * 1000.0:.2f}mm, "
            f"last_z_offset={(last_z_error or 0.0) * 1000.0:.2f}mm, "
            f"last_rpy_norm={math.degrees(last_rpy_error):.2f}deg"
        )
        return success

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
        fixed_orientation = (
            self._copy_quaternion(self._target_orientation)
            if self._target_orientation is not None
            else self._copy_quaternion(start_pose.orientation)
        )
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
        try:
            self._preload_detection_model_for_current_task()
        except Exception as exc:
            self.get_logger().error(
                format_model_log(f"FinalPolicy initial YOLO model load failed: {exc}")
            )
            send_feedback("failed: load YOLO model")
            return False
        self._start_background_yolo_model_downloads()

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

        stages = (
            ("lift_up_detect", lambda: self._stage_lift_up_detect(get_observation, move_robot)),
            ("approach", lambda: self._stage_approach(get_observation, move_robot)),
            ("vision_offset_align", lambda: self._stage_align(get_observation, move_robot)),
            ("insert", lambda: self._stage_insert(get_observation, move_robot)),
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
        send_feedback("Final Policy: done")
        self.get_logger().info("FinalPolicy done")
        return True
