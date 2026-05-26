from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Vector3, Wrench
from std_msgs.msg import Header

from distance_prediction_policy.model_feedback import VisionOffsetPredictor

from .config import ModelEvalConfig
from .geometry import (
    pose_dict,
    pose_from_state,
    quat_xyzw_to_matrix,
    rotation_angle_rad,
    rpy_to_matrix,
    state_from_matrix,
    transform_inverse,
    transform_state_from_pose,
)
from .ground_truth import (
    GroundTruthReader,
    GroundTruthState,
    port_id_from_task,
    sample_initial_offset_and_rpy,
)
from .predictors import OrientationDeltaPredictor

try:
    import cv2
except ImportError:
    cv2 = None


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _clip_vector(values: np.ndarray, max_norm: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if max_norm <= 0.0:
        return values
    norm = float(np.linalg.norm(values))
    if norm <= max_norm or norm < 1e-12:
        return values
    return values * (max_norm / norm)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _initial_rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64)
    return rpy_to_matrix([roll, pitch, -yaw])


_TOOL0_TO_OPTICAL = {
    "left": (
        [-0.100516584, -0.058032593, -0.008935891],
        [-0.113039947, 0.065265728, -0.495722390, 0.858616135],
    ),
    "center": (
        [-0.000000001, -0.116079183, -0.008937891],
        [-0.130528330, 0.000001827, -0.000000288, 0.991444580],
    ),
    "right": (
        [0.100516583, -0.058032595, -0.008935891],
        [-0.113041775, -0.065262563, 0.495721890, 0.858616424],
    ),
}


def _matrix_from_translation_quat(translation, quat_xyzw) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _pose_quat_array(pose: Pose) -> np.ndarray:
    return np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )


def _align_pose_quaternion_to_reference(pose: Pose, reference_pose: Pose) -> bool:
    if float(np.dot(_pose_quat_array(pose), _pose_quat_array(reference_pose))) >= 0.0:
        return False
    pose.orientation.x *= -1.0
    pose.orientation.y *= -1.0
    pose.orientation.z *= -1.0
    pose.orientation.w *= -1.0
    return True


class _BaseModelEvalPolicy(Policy):
    model_kind = "base"

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task: Task | None = None
        self._gt = GroundTruthReader(parent_node, logger=self.get_logger())
        self._run_id = ModelEvalConfig.RUN_ID
        self._trial_index = ModelEvalConfig.TRIAL_INDEX
        self._t_tool0_tcp = np.eye(4, dtype=np.float64)
        self._t_tool0_tcp[2, 3] = ModelEvalConfig.TOOL0_TO_TCP_Z
        self._t_tool0_to_optical = {
            name: _matrix_from_translation_quat(translation, quat)
            for name, (translation, quat) in _TOOL0_TO_OPTICAL.items()
        }
        self._video_writer = None
        self._video_path: Path | None = None
        self._video_warned = False
        ModelEvalConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"{self.__class__.__name__} ready: run_id={self._run_id}, "
            f"trial_index={self._trial_index}, output={ModelEvalConfig.OUTPUT_DIR}"
        )

    def _tcp_pose(self, observation) -> Pose | None:
        if observation is None:
            return None
        return observation.controller_state.tcp_pose

    def _image_msg_for_camera(self, observation, camera_name: str):
        return getattr(observation, f"{camera_name}_image", None)

    def _camera_info_for_camera(self, observation, camera_name: str):
        return getattr(observation, f"{camera_name}_camera_info", None)

    def _camera_intrinsic_matrix(self, camera_info) -> np.ndarray | None:
        if camera_info is None or len(camera_info.k) < 9:
            return None
        k = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        if abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9:
            return None
        return k

    def _image_msg_to_bgr(self, image_msg) -> np.ndarray | None:
        if image_msg is None or image_msg.width == 0 or image_msg.height == 0:
            return None
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()
        if encoding in {"rgba8", "bgra8"}:
            channels = 4
        elif encoding in {"rgb8", "bgr8"}:
            channels = 3
        else:
            channels = 4 if len(image_msg.data) >= height * width * 4 else 3
        flat = np.frombuffer(image_msg.data, dtype=np.uint8)
        step = int(getattr(image_msg, "step", 0))
        try:
            if step > 0 and flat.size >= height * step:
                rows = flat[: height * step].reshape(height, step)
                image = rows[:, : width * channels].reshape(height, width, channels)
            else:
                image = flat[: height * width * channels].reshape(height, width, channels)
        except ValueError:
            return None
        if channels == 4:
            image = image[:, :, :3]
        if encoding in {"rgb8", "rgba8"}:
            if cv2 is None:
                return image[:, :, ::-1].copy()
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    def _video_frame_from_observation(
        self,
        observation,
        phase: str,
        task: Task | None = None,
    ) -> np.ndarray | None:
        if cv2 is None or observation is None:
            return None
        frames = []
        for camera in ModelEvalConfig.VIDEO_CAMERAS:
            bgr = self._image_msg_to_bgr(self._image_msg_for_camera(observation, camera))
            if bgr is None:
                continue
            max_height = int(ModelEvalConfig.VIDEO_MAX_HEIGHT)
            if max_height > 0 and bgr.shape[0] > max_height:
                scale = max_height / float(bgr.shape[0])
                bgr = cv2.resize(
                    bgr,
                    (max(1, int(round(bgr.shape[1] * scale))), max_height),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.putText(
                bgr,
                camera,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(bgr)
        if not frames:
            return None
        height = max(frame.shape[0] for frame in frames)
        padded = []
        for frame in frames:
            if frame.shape[0] < height:
                pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                frame = np.vstack([frame, pad])
            padded.append(frame)
        composed = np.hstack(padded)
        cv2.putText(
            composed,
            self._display_phase_label(phase),
            (12, composed.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if task is not None:
            state = self._gt.read(task)
            if state is not None:
                orientation = state.orientation
                distance = state.distance
                lines = [
                    (
                        "rpy "
                        f"r={orientation['roll_deg']:+.2f} "
                        f"p={orientation['pitch_deg']:+.2f} "
                        f"y={orientation['yaw_deg']:+.2f} deg"
                    ),
                    (
                        f"ang={orientation['angular_deg']:.2f} "
                        f"geo={orientation['geodesic_deg']:.2f} deg"
                    ),
                    (
                        "dist "
                        f"x={distance['x_mm']:+.1f} "
                        f"y={distance['y_mm']:+.1f} "
                        f"z={distance['z_mm']:+.1f} mm"
                    ),
                ]
                y = 62
                for line in lines:
                    cv2.putText(
                        composed,
                        line,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 24
        return composed

    def _display_phase_label(self, phase: str) -> str:
        labels = {
            "initial_settle": "setup_initial_settle",
            "initial_wait": "setup_initial_wait",
            "before_distance_action": "eval_before_distance_action",
            "distance_action_settle": "eval_distance_action_settle",
            "distance_action_wait": "eval_distance_action_wait",
            "before_orientation_action": "eval_before_orientation_action",
            "orientation_action_settle": "eval_orientation_action_settle",
            "orientation_action_wait": "eval_orientation_action_wait",
        }
        return labels.get(phase, phase)

    def _safe_task_id(self, task: Task) -> str:
        text = str(getattr(task, "id", "") or f"trial_{self._trial_index:04d}")
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)

    def _ensure_video_writer(self, task: Task, frame: np.ndarray) -> bool:
        if not ModelEvalConfig.RECORD_VIDEO:
            return False
        if cv2 is None:
            if not self._video_warned:
                self.get_logger().warn("model-eval video recording requested, but cv2 is unavailable")
                self._video_warned = True
            return False
        if self._video_writer is not None:
            return True
        ModelEvalConfig.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self._video_path = (
            ModelEvalConfig.VIDEO_DIR
            / f"{self.model_kind}_trial_{self._trial_index:04d}_{self._safe_task_id(task)}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = max(1.0, float(ModelEvalConfig.VIDEO_FPS))
        self._video_writer = cv2.VideoWriter(
            str(self._video_path),
            fourcc,
            fps,
            (int(frame.shape[1]), int(frame.shape[0])),
        )
        if not self._video_writer.isOpened():
            self.get_logger().warn(f"Failed to open video writer: {self._video_path}")
            self._video_writer = None
            self._video_path = None
            return False
        return True

    def _record_video_frame(self, get_observation, task: Task, phase: str) -> None:
        if not ModelEvalConfig.RECORD_VIDEO:
            return
        frame = self._video_frame_from_observation(get_observation(), phase, task)
        if frame is None:
            return
        if self._ensure_video_writer(task, frame):
            self._video_writer.write(frame)

    def _record_video_for_duration(
        self,
        get_observation,
        task: Task,
        phase: str,
        duration_s: float,
    ) -> None:
        if not ModelEvalConfig.RECORD_VIDEO:
            self.sleep_for(duration_s)
            return
        end = time.monotonic() + max(0.0, duration_s)
        period = 1.0 / max(1.0, float(ModelEvalConfig.VIDEO_FPS))
        while time.monotonic() < end:
            self._record_video_frame(get_observation, task, phase)
            self.sleep_for(min(period, max(0.0, end - time.monotonic())))

    def _close_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _base_to_camera_optical_matrix(self, observation, camera_name: str) -> np.ndarray:
        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            raise RuntimeError("missing_tcp_pose")
        t_base_tcp = transform_state_from_pose(tcp_pose).matrix
        t_base_tool0 = t_base_tcp @ np.linalg.inv(self._t_tool0_tcp)
        t_base_optical = t_base_tool0 @ self._t_tool0_to_optical[camera_name]
        return np.linalg.inv(t_base_optical)

    def _port_projection_for_camera(
        self,
        observation,
        camera_name: str,
        state: GroundTruthState,
    ) -> dict[str, Any]:
        img_msg = self._image_msg_for_camera(observation, camera_name)
        k = self._camera_intrinsic_matrix(self._camera_info_for_camera(observation, camera_name))
        if img_msg is None or img_msg.width == 0 or img_msg.height == 0 or k is None:
            return {"visible": False, "reason": "missing_image_or_intrinsics"}
        try:
            t_cam_base = self._base_to_camera_optical_matrix(observation, camera_name)
        except Exception as exc:
            return {"visible": False, "reason": f"camera_transform_error: {exc}"}

        point_base = np.append(state.port.translation, 1.0)
        point_cam = t_cam_base @ point_base
        depth = float(point_cam[2])
        if depth <= 1e-6:
            return {"visible": False, "reason": "behind_camera", "depth_m": depth}

        u = float(k[0, 0] * point_cam[0] / depth + k[0, 2])
        v = float(k[1, 1] * point_cam[1] / depth + k[1, 2])
        margin = float(ModelEvalConfig.VISIBILITY_MARGIN_PX)
        visible = (
            margin <= u < float(img_msg.width) - margin
            and margin <= v < float(img_msg.height) - margin
        )
        return {
            "visible": bool(visible),
            "u_px": u,
            "v_px": v,
            "depth_m": depth,
            "width": int(img_msg.width),
            "height": int(img_msg.height),
            "margin_px": margin,
        }

    def _port_visibility(self, observation, state: GroundTruthState) -> dict[str, Any]:
        cameras = tuple(
            camera for camera in ModelEvalConfig.CAMERAS if camera in self._t_tool0_to_optical
        )
        required = int(ModelEvalConfig.MIN_VISIBLE_CAMERAS)
        if required <= 0:
            required = len(cameras)
        required = min(max(1, required), len(cameras))
        projections = {
            camera: self._port_projection_for_camera(observation, camera, state)
            for camera in cameras
        }
        visible_cameras = [
            camera
            for camera, projection in projections.items()
            if projection.get("visible", False)
        ]
        return {
            "target": "target_port_center",
            "required_visible_cameras": int(required),
            "visible_cameras": visible_cameras,
            "visible_count": int(len(visible_cameras)),
            "visible_enough": bool(len(visible_cameras) >= required),
            "projections": projections,
        }

    def set_pose_target(self, move_robot, pose: Pose, stiffness=None, damping=None) -> None:
        s = stiffness if stiffness is not None else ModelEvalConfig.STIFFNESS
        d = damping if damping is not None else ModelEvalConfig.DAMPING
        motion_update = MotionUpdate(
            header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
            pose=pose,
            target_stiffness=np.diag(s).flatten(),
            target_damping=np.diag(d).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        move_robot(motion_update=motion_update)

    def _state_record(self, state: GroundTruthState | None) -> dict[str, Any] | None:
        if state is None:
            return None
        return {
            "port_frame": state.port_frame,
            "plug_frame": state.plug_frame,
            "distance": state.distance,
            "orientation": state.orientation,
        }

    def _append_metric(self, record: dict[str, Any]) -> None:
        ModelEvalConfig.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ModelEvalConfig.METRICS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._write_summary()

    def _write_summary(self) -> None:
        path = ModelEvalConfig.METRICS_PATH
        if not path.exists():
            return
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mine = [row for row in rows if row.get("model_kind") == self.model_kind]
        if not mine:
            return
        summary = {
            "run_id": self._run_id,
            "model_kind": self.model_kind,
            "count": len(mine),
            "success_rate": sum(1 for row in mine if row.get("success")) / max(len(mine), 1),
        }
        improved = [
            row
            for row in mine
            if (improvement := _finite_float(row.get("improvement"))) is not None
            and improvement > 0.0
        ]
        summary["improvement_rate"] = len(improved) / max(len(mine), 1)
        summary["success_definition"] = (
            f"actual_after_norm <= {ModelEvalConfig.DISTANCE_SUCCESS_M * 1000.0:.3f} mm"
            if self.model_kind == "distance"
            else f"actual_after_norm <= {math.degrees(ModelEvalConfig.ORIENTATION_SUCCESS_RAD):.3f} deg"
        )
        for key in (
            "model_error_norm_before",
            "actual_before_norm",
            "actual_after_norm",
            "improvement",
        ):
            values = [
                value
                for row in mine
                if (value := _finite_float(row.get(key))) is not None
            ]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "max": float(np.max(values)),
                }
                if key in {"actual_before_norm", "actual_after_norm", "improvement"}:
                    scale = 1000.0 if self.model_kind == "distance" else 180.0 / math.pi
                    unit = "mm" if self.model_kind == "distance" else "deg"
                    summary[f"{key}_{unit}"] = {
                        "mean": float(np.mean(values) * scale),
                        "median": float(np.median(values) * scale),
                        "p90": float(np.percentile(values, 90) * scale),
                        "max": float(np.max(values) * scale),
                    }
        ModelEvalConfig.SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _make_initial_pose(
        self,
        *,
        observation,
        before: GroundTruthState,
        attempt_index: int,
    ) -> tuple[Pose, dict[str, Any]] | None:
        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            return None

        offset_m, rpy_rad = sample_initial_offset_and_rpy(
            ModelEvalConfig.SEED,
            self._trial_index,
            attempt_index,
        )
        tcp_state = transform_state_from_pose(tcp_pose)
        tcp_to_plug = transform_inverse(tcp_state.matrix) @ before.plug_reference.matrix

        port_rotation = before.port.rotation_matrix
        approach_axis = -port_rotation[:, 2]
        tcp_to_plug_rotation = transform_inverse(tcp_state.matrix)[:3, :3] @ before.plug_reference.rotation_matrix
        tcp_to_plug_translation_base = tcp_state.translation - before.plug_reference.translation

        desired_plug = np.eye(4, dtype=np.float64)
        desired_plug[:3, :3] = port_rotation @ _initial_rpy_to_matrix(rpy_rad)
        desired_plug[:3, 3] = (
            before.port.translation
            + offset_m[0] * port_rotation[:, 0]
            + offset_m[1] * port_rotation[:, 1]
            + offset_m[2] * approach_axis
        )
        desired_tcp = np.eye(4, dtype=np.float64)
        desired_tcp[:3, :3] = desired_plug[:3, :3] @ tcp_to_plug_rotation.T
        desired_tcp[:3, 3] = desired_plug[:3, 3] + tcp_to_plug_translation_base
        pose = pose_from_state(state_from_matrix(desired_tcp))
        target_quaternion_flipped = _align_pose_quaternion_to_reference(pose, tcp_pose)
        metadata = {
            "seed": ModelEvalConfig.SEED,
            "trial_index": self._trial_index,
            "attempt_index": int(attempt_index),
            "target_quaternion_flipped_to_match_current": bool(target_quaternion_flipped),
            "position_reference": "port_entrance_outward_axis",
            "offset_convention": "dx/dy are port local x/y; dz is outward approach distance from port entrance",
            "tcp_position_mode": "preserve_current_base_tcp_to_plug_reference_vector",
            "offset_m": [float(v) for v in offset_m],
            "offset_mm": [float(v * 1000.0) for v in offset_m],
            "expected_distance_label_m": [
                float(offset_m[0]),
                float(offset_m[1]),
                float(-offset_m[2]),
            ],
            "expected_distance_label_mm": [
                float(offset_m[0] * 1000.0),
                float(offset_m[1] * 1000.0),
                float(-offset_m[2] * 1000.0),
            ],
            "rpy_rad": [float(v) for v in rpy_rad],
            "rpy_deg": [float(math.degrees(v)) for v in rpy_rad],
            "desired_plug_position_m": [float(v) for v in desired_plug[:3, 3]],
            "desired_plug_rotation": desired_plug[:3, :3].tolist(),
        }
        return pose, metadata

    def _initial_pose_error(
        self,
        state: GroundTruthState,
        metadata: dict[str, Any],
    ) -> tuple[float, float]:
        desired_position = np.asarray(metadata["desired_plug_position_m"], dtype=np.float64)
        desired_rotation = np.asarray(metadata["desired_plug_rotation"], dtype=np.float64)
        position_error = float(np.linalg.norm(state.plug_reference.translation - desired_position))
        rotation_error = rotation_angle_rad(state.plug_reference.rotation_matrix.T @ desired_rotation)
        return position_error, rotation_error

    def _tcp_target_error(self, observation, target_pose: Pose) -> tuple[float, float] | None:
        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            return None
        actual = transform_state_from_pose(tcp_pose)
        target = transform_state_from_pose(target_pose)
        position_error = float(np.linalg.norm(actual.translation - target.translation))
        rotation_error = rotation_angle_rad(actual.rotation_matrix.T @ target.rotation_matrix)
        return position_error, rotation_error

    def _wait_for_action_target(
        self,
        *,
        get_observation,
        task: Task,
        target_pose: Pose,
        phase: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + ModelEvalConfig.ACTION_WAIT_TIMEOUT_S
        best_position_error = float("inf")
        best_rotation_error = float("inf")
        reached = False
        while True:
            observation = get_observation()
            errors = self._tcp_target_error(observation, target_pose)
            if errors is not None:
                position_error, rotation_error = errors
                if rotation_error < best_rotation_error:
                    best_rotation_error = rotation_error
                    best_position_error = position_error
                if (
                    position_error <= ModelEvalConfig.ACTION_POSITION_TOLERANCE_M
                    and rotation_error <= ModelEvalConfig.ACTION_ORIENTATION_TOLERANCE_RAD
                ):
                    reached = True
                    best_position_error = position_error
                    best_rotation_error = rotation_error
                    break
            self._record_video_frame(lambda: observation, task, phase)
            if time.monotonic() >= deadline:
                break
            self.sleep_for(1.0 / max(1.0, float(ModelEvalConfig.VIDEO_FPS)))
        return {
            "target_reached": bool(reached),
            "position_error_mm": float(best_position_error * 1000.0),
            "orientation_error_deg": float(math.degrees(best_rotation_error)),
            "timeout_s": float(ModelEvalConfig.ACTION_WAIT_TIMEOUT_S),
            "position_tolerance_mm": float(ModelEvalConfig.ACTION_POSITION_TOLERANCE_M * 1000.0),
            "orientation_tolerance_deg": float(math.degrees(ModelEvalConfig.ACTION_ORIENTATION_TOLERANCE_RAD)),
        }

    def _wait_for_initial_pose(
        self,
        *,
        task: Task,
        metadata: dict[str, Any],
        get_observation=None,
    ) -> GroundTruthState | None:
        deadline = time.monotonic() + ModelEvalConfig.INITIAL_WAIT_TIMEOUT_S
        best_state = None
        best_position_error = float("inf")
        best_rotation_error = float("inf")
        while True:
            state = self._gt.read(task)
            if state is not None:
                position_error, rotation_error = self._initial_pose_error(state, metadata)
                if position_error < best_position_error:
                    best_state = state
                    best_position_error = position_error
                    best_rotation_error = rotation_error
                if (
                    position_error <= ModelEvalConfig.INITIAL_POSITION_TOLERANCE_M
                    and rotation_error <= ModelEvalConfig.INITIAL_ORIENTATION_TOLERANCE_RAD
                ):
                    metadata["initial_reached"] = True
                    metadata["initial_position_error_mm"] = float(position_error * 1000.0)
                    metadata["initial_orientation_error_deg"] = float(math.degrees(rotation_error))
                    return state
            if time.monotonic() >= deadline:
                metadata["initial_reached"] = False
                metadata["initial_position_error_mm"] = float(best_position_error * 1000.0)
                metadata["initial_orientation_error_deg"] = float(math.degrees(best_rotation_error))
                self.get_logger().warn(
                    "model-eval initial pose did not reach tolerance: "
                    f"position_error={metadata['initial_position_error_mm']:.2f}mm, "
                    f"orientation_error={metadata['initial_orientation_error_deg']:.2f}deg"
                )
                return best_state or state
            if get_observation is not None:
                self._record_video_frame(get_observation, task, "initial_wait")
            self.sleep_for(0.2)

    def _randomize_initial_pose(
        self,
        *,
        get_observation,
        move_robot,
        task: Task,
    ) -> tuple[GroundTruthState | None, dict[str, Any]]:
        before = self._gt.read(task)
        if before is None:
            return None, {}

        if not ModelEvalConfig.RANDOMIZE_INITIAL_POSE:
            metadata = {"randomize_initial_pose": False}
            visibility = self._port_visibility(get_observation(), before)
            metadata["visibility"] = visibility
            if not visibility["visible_enough"]:
                self.get_logger().error(
                    "model-eval target port is not visible enough before action: "
                    f"visible={visibility['visible_cameras']}, "
                    f"required={visibility['required_visible_cameras']}"
                )
                return None, metadata
            return before, metadata

        last_metadata: dict[str, Any] = {}
        max_attempts = max(1, int(ModelEvalConfig.MAX_VISIBILITY_ATTEMPTS))
        for attempt_index in range(max_attempts):
            current = self._gt.read(task) or before
            observation = get_observation()
            result = self._make_initial_pose(
                observation=observation,
                before=current,
                attempt_index=attempt_index,
            )
            if result is None:
                last_metadata = {
                    "attempt_index": int(attempt_index),
                    "visibility": {"visible_enough": False, "reason": "missing_tcp_pose"},
                }
                continue
            pose, metadata = result
            metadata["visibility_attempt"] = int(attempt_index + 1)
            metadata["max_visibility_attempts"] = int(max_attempts)
            self.get_logger().info(
                "model-eval initial pose: "
                f"attempt={attempt_index + 1}/{max_attempts}, "
                f"offset_mm={metadata['offset_mm']}, rpy_deg={metadata['rpy_deg']}"
            )
            self.set_pose_target(move_robot, pose)
            self._record_video_for_duration(
                get_observation,
                task,
                "initial_settle",
                ModelEvalConfig.INITIAL_SETTLE_S,
            )
            before_after_move = self._wait_for_initial_pose(
                task=task,
                metadata=metadata,
                get_observation=get_observation,
            )
            if before_after_move is None:
                last_metadata = metadata
                continue

            visibility = self._port_visibility(get_observation(), before_after_move)
            metadata["visibility"] = visibility
            if visibility["visible_enough"]:
                return before_after_move, metadata

            last_metadata = metadata
            self.get_logger().warn(
                "model-eval target port visibility rejected: "
                f"attempt={attempt_index + 1}/{max_attempts}, "
                f"visible={visibility['visible_cameras']}, "
                f"required={visibility['required_visible_cameras']}"
            )

        self.get_logger().error(
            "model-eval failed to find an initial pose with target port visible in "
            f"{max_attempts} attempt(s)."
        )
        return None, last_metadata

    def _base_record(
        self,
        *,
        task: Task,
        initial_randomization: dict[str, Any],
        before: GroundTruthState | None,
        after: GroundTruthState | None,
    ) -> dict[str, Any]:
        return {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self._run_id,
            "trial_index": self._trial_index,
            "model_kind": self.model_kind,
            "task": {
                "id": str(getattr(task, "id", "")),
                "cable_name": str(getattr(task, "cable_name", "")),
                "plug_name": str(getattr(task, "plug_name", "")),
                "port_name": str(getattr(task, "port_name", "")),
                "target_module_name": str(getattr(task, "target_module_name", "")),
            },
            "initial_randomization": initial_randomization,
            "ground_truth_before": self._state_record(before),
            "ground_truth_after": self._state_record(after),
            "video_path": (
                str(self._video_path.relative_to(ModelEvalConfig.OUTPUT_DIR))
                if self._video_path is not None
                else None
            ),
        }


class DistanceModelEvalPolicy(_BaseModelEvalPolicy):
    model_kind = "distance"

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._predictor = VisionOffsetPredictor(
            device=ModelEvalConfig.DEVICE,
            logger=self.get_logger(),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        send_feedback("model-eval distance: randomize initial pose")
        before, init_meta = self._randomize_initial_pose(
            get_observation=get_observation,
            move_robot=move_robot,
            task=task,
        )
        if before is None:
            send_feedback("failed: missing ground truth before")
            self._close_video()
            return False

        observation = get_observation()
        self._record_video_frame(get_observation, task, "before_distance_action")
        pred_m = self._predictor.predict_offset_m(observation, port_id_from_task(task))
        if pred_m is None:
            send_feedback("failed: distance prediction unavailable")
            self._close_video()
            return False

        actual_before = np.array(
            [before.distance["x_m"], before.distance["y_m"], before.distance["z_m"]],
            dtype=np.float64,
        )
        model_error = pred_m - actual_before
        correction_base = before.port.rotation_matrix @ (-pred_m)
        correction_base = _clip_vector(correction_base, ModelEvalConfig.MAX_DISTANCE_ACTION_M)

        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            send_feedback("failed: missing TCP pose")
            self._close_video()
            return False
        target_pose = Pose()
        target_pose.position.x = float(tcp_pose.position.x + correction_base[0])
        target_pose.position.y = float(tcp_pose.position.y + correction_base[1])
        target_pose.position.z = float(tcp_pose.position.z + correction_base[2])
        target_pose.orientation = tcp_pose.orientation

        send_feedback("model-eval distance: apply one action")
        self.set_pose_target(move_robot, target_pose)
        self._record_video_for_duration(
            get_observation,
            task,
            "distance_action_settle",
            ModelEvalConfig.ACTION_SETTLE_S,
        )
        action_tracking = self._wait_for_action_target(
            get_observation=get_observation,
            task=task,
            target_pose=target_pose,
            phase="distance_action_wait",
        )
        after = self._gt.read(task)

        actual_after_norm = after.distance["norm_m"] if after is not None else None
        record = self._base_record(task=task, initial_randomization=init_meta, before=before, after=after)
        record.update(
            {
                "prediction": {
                    "offset_m": [float(v) for v in pred_m],
                    "offset_mm": [float(v * 1000.0) for v in pred_m],
                },
                "action": {
                    "translation_base_m": [float(v) for v in correction_base],
                    "translation_base_mm": [float(v * 1000.0) for v in correction_base],
                    "target_pose": pose_dict(target_pose),
                    "tracking": action_tracking,
                },
                "model_error": {
                    "xyz_m": [float(v) for v in model_error],
                    "xyz_mm": [float(v * 1000.0) for v in model_error],
                    "norm_m": _norm(model_error),
                    "norm_mm": _norm(model_error) * 1000.0,
                },
                "model_error_norm_before": _norm(model_error),
                "actual_before_norm": float(before.distance["norm_m"]),
                "actual_after_norm": actual_after_norm,
                "improvement": None
                if actual_after_norm is None
                else float(before.distance["norm_m"] - actual_after_norm),
                "improved": bool(
                    actual_after_norm is not None
                    and before.distance["norm_m"] - actual_after_norm > 0.0
                ),
                "success_definition": (
                    f"actual_after_norm <= {ModelEvalConfig.DISTANCE_SUCCESS_M * 1000.0:.3f} mm"
                ),
                "success": bool(actual_after_norm is not None and actual_after_norm <= ModelEvalConfig.DISTANCE_SUCCESS_M),
            }
        )
        self._append_metric(record)
        self.get_logger().info(
            "distance eval: "
            f"pred={pred_m * 1000.0}mm, "
            f"before={before.distance['norm_mm']:.2f}mm, "
            f"after={float('nan') if actual_after_norm is None else actual_after_norm * 1000.0:.2f}mm"
        )
        self._close_video()
        return True


class OrientationModelEvalPolicy(_BaseModelEvalPolicy):
    model_kind = "orientation"

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._predictor = OrientationDeltaPredictor(logger=self.get_logger())

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        send_feedback("model-eval orientation: randomize initial pose")
        before, init_meta = self._randomize_initial_pose(
            get_observation=get_observation,
            move_robot=move_robot,
            task=task,
        )
        if before is None:
            send_feedback("failed: missing ground truth before")
            self._close_video()
            return False

        observation = get_observation()
        self._record_video_frame(get_observation, task, "before_orientation_action")
        pred_rpy = self._predictor.predict_rpy_rad(observation, port_id_from_task(task))
        if pred_rpy is None:
            send_feedback("failed: orientation prediction unavailable")
            self._close_video()
            return False
        pred_rpy = _clip_vector(pred_rpy, ModelEvalConfig.MAX_ORIENTATION_ACTION_RAD)

        actual_before = np.array(
            [
                before.orientation["roll_rad"],
                before.orientation["pitch_rad"],
                before.orientation["yaw_rad"],
            ],
            dtype=np.float64,
        )
        model_error = pred_rpy - actual_before

        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            send_feedback("failed: missing TCP pose")
            self._close_video()
            return False
        tcp_state = transform_state_from_pose(tcp_pose)
        tcp_to_plug_rotation = transform_inverse(tcp_state.matrix)[:3, :3] @ before.plug_reference.rotation_matrix
        desired_tcp = np.eye(4, dtype=np.float64)
        desired_tcp[:3, :3] = before.plug_reference.rotation_matrix @ rpy_to_matrix(pred_rpy) @ tcp_to_plug_rotation.T
        desired_tcp[:3, 3] = tcp_state.translation
        target_pose = pose_from_state(state_from_matrix(desired_tcp))
        target_quaternion_flipped = _align_pose_quaternion_to_reference(target_pose, tcp_pose)

        send_feedback("model-eval orientation: apply one action")
        self.set_pose_target(move_robot, target_pose)
        self._record_video_for_duration(
            get_observation,
            task,
            "orientation_action_settle",
            ModelEvalConfig.ACTION_SETTLE_S,
        )
        action_tracking = self._wait_for_action_target(
            get_observation=get_observation,
            task=task,
            target_pose=target_pose,
            phase="orientation_action_wait",
        )
        after = self._gt.read(task)

        actual_after_norm = after.orientation["angular_rad"] if after is not None else None
        record = self._base_record(task=task, initial_randomization=init_meta, before=before, after=after)
        record.update(
            {
                "prediction": {
                    "rpy_rad": [float(v) for v in pred_rpy],
                    "rpy_deg": [float(math.degrees(v)) for v in pred_rpy],
                    "angular_rad": _norm(pred_rpy),
                    "angular_deg": math.degrees(_norm(pred_rpy)),
                },
                "action": {
                    "target_pose": pose_dict(target_pose),
                    "rotation_about": "tcp_position_fixed",
                    "target_quaternion_flipped_to_match_current": bool(target_quaternion_flipped),
                    "tracking": action_tracking,
                },
                "model_error": {
                    "rpy_rad": [float(v) for v in model_error],
                    "rpy_deg": [float(math.degrees(v)) for v in model_error],
                    "angular_rad": _norm(model_error),
                    "angular_deg": math.degrees(_norm(model_error)),
                },
                "model_error_norm_before": _norm(model_error),
                "actual_before_norm": float(before.orientation["angular_rad"]),
                "actual_after_norm": actual_after_norm,
                "improvement": None
                if actual_after_norm is None
                else float(before.orientation["angular_rad"] - actual_after_norm),
                "improved": bool(
                    actual_after_norm is not None
                    and before.orientation["angular_rad"] - actual_after_norm > 0.0
                ),
                "success_definition": (
                    "actual_after_norm <= "
                    f"{math.degrees(ModelEvalConfig.ORIENTATION_SUCCESS_RAD):.3f} deg"
                ),
                "success": bool(actual_after_norm is not None and actual_after_norm <= ModelEvalConfig.ORIENTATION_SUCCESS_RAD),
            }
        )
        self._append_metric(record)
        self.get_logger().info(
            "orientation eval: "
            f"pred_deg={[math.degrees(v) for v in pred_rpy]}, "
            f"before={before.orientation['angular_deg']:.2f}deg, "
            f"after={float('nan') if actual_after_norm is None else math.degrees(actual_after_norm):.2f}deg"
        )
        self._close_video()
        return True
