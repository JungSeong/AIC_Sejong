"""Model loading and closed-loop align/insert control."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException

from motion_planning_node.core.geometry import quat_to_tuple, rotate_vector_by_quat

from distance_prediction_policy.config import (
    DistancePredictionConfig,
    SRC_ROOT,
)


def _load_position_model_module():
    module_name = "_aic_distance_position_model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = (
        SRC_ROOT
        / "ais"
        / "ais_distance_prediction"
        / "model"
        / "position_model.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing distance model code: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class VisionOffsetPredictor:
    """Load the trained ResNet offset model and run image inference."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DistancePredictionConfig.CHECKPOINT_PATH,
        device: str = DistancePredictionConfig.DEVICE,
        logger=None,
    ) -> None:
        self._logger = logger
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = self._resolve_device(device)
        self.model = None
        self.image_size: int | tuple[int, int] | None = 224
        self.cameras = DistancePredictionConfig.CAMERAS
        self.checkpoint_cameras = ()
        self.aggregation = "mean"
        self.num_views = 1
        self.metrics = {}
        self.target_mean = torch.zeros(3, dtype=torch.float32)
        self.target_std = torch.ones(3, dtype=torch.float32)
        self._load()

    def _log_info(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message)

    def _log_warn(self, message: str) -> None:
        if self._logger is not None:
            self._logger.warn(message)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Distance prediction checkpoint not found: {self.checkpoint_path}"
            )

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Invalid checkpoint format: {self.checkpoint_path}")

        config = payload.get("config", {})
        state_dict = payload["model_state_dict"]
        image_size = config.get("image_size", 224)
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, (list, tuple)):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = int(image_size)
        self.checkpoint_cameras = tuple(config.get("cameras", ()))
        self.cameras = DistancePredictionConfig.CAMERAS
        self.metrics = dict(payload.get("metrics", {}))
        self.target_mean = torch.as_tensor(
            config.get("target_mean", [0.0, 0.0, 0.0]), dtype=torch.float32
        )
        self.target_std = torch.as_tensor(
            config.get("target_std", [1.0, 1.0, 1.0]), dtype=torch.float32
        )

        position_model = _load_position_model_module()
        self.aggregation = config.get("aggregation", "mean")
        self.num_views = int(config.get("num_views", len(config.get("cameras", self.cameras))))
        if self.aggregation == "concat" and len(self.cameras) != self.num_views:
            raise ValueError(
                "Concat distance checkpoint expects "
                f"{self.num_views} views, but runtime cameras are {self.cameras}."
            )
        if "head.0.weight" in state_dict:
            hidden_dim = int(state_dict["head.0.weight"].shape[0])
            num_port_heads = int(config.get("num_port_heads", 1))
        else:
            head_indices = {
                int(match.group(1))
                for key in state_dict
                if (match := re.match(r"heads\.(\d+)\.0\.weight$", key)) is not None
            }
            if not head_indices:
                raise ValueError("Checkpoint has no recognizable regression head weights.")
            hidden_dim = int(state_dict[f"heads.{min(head_indices)}.0.weight"].shape[0])
            num_port_heads = int(config.get("num_port_heads", max(head_indices) + 1))
        output_dim = int(self.target_mean.numel())
        self.model = position_model.build_resnet_position_model(
            backbone_name=config.get("backbone", "resnet50"),
            pretrained=False,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=0.1,
            aggregation=self.aggregation,
            num_port_heads=num_port_heads,
            num_views=self.num_views,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._log_info(
            "Distance model loaded: "
            f"path={self.checkpoint_path}, device={self.device}, "
            f"cameras={self.cameras}, checkpoint_cameras={self.checkpoint_cameras}, "
            f"image_size={self.image_size}, aggregation={self.aggregation}, "
            f"num_views={self.num_views}, metrics={self.metrics}"
        )
        if self.checkpoint_cameras and self.checkpoint_cameras != self.cameras:
            self._log_warn(
                "Distance model camera config differs from checkpoint metadata: "
                f"runtime={self.cameras}, checkpoint={self.checkpoint_cameras}"
            )

    def _image_msg_to_tensor(self, image_msg: Image) -> torch.Tensor:
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()

        if encoding in {"rgba8", "bgra8"}:
            channels = 4
        elif encoding in {"rgb8", "bgr8"}:
            channels = 3
        else:
            pixel_count = height * width
            data_len = len(image_msg.data)
            channels = 4 if data_len >= pixel_count * 4 else 3

        try:
            flat = np.frombuffer(image_msg.data, dtype=np.uint8)
        except TypeError:
            flat = np.asarray(image_msg.data, dtype=np.uint8)

        step = int(getattr(image_msg, "step", 0))
        if step > 0 and flat.size >= height * step:
            rows = flat[: height * step].reshape(height, step)
            image = rows[:, : width * channels].reshape(height, width, channels)
        else:
            expected = height * width * channels
            if flat.size < expected:
                raise ValueError(
                    f"Image buffer too small: got {flat.size}, expected {expected}"
                )
            image = flat[:expected].reshape(height, width, channels)

        if channels == 4:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image).copy()

        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        if self.image_size is not None:
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def _camera_image(self, observation, camera: str) -> Optional[Image]:
        return getattr(observation, f"{camera}_image", None)

    @torch.inference_mode()
    def predict_offset_m(self, observation, port_id: int | None = None) -> Optional[np.ndarray]:
        if observation is None:
            return None

        images = []
        for camera in self.cameras:
            image_msg = self._camera_image(observation, camera)
            if image_msg is None:
                self._log_warn(f"Observation missing {camera}_image")
                return None
            images.append(self._image_msg_to_tensor(image_msg))

        if len(images) == 1:
            model_input = images[0].unsqueeze(0)
        else:
            model_input = torch.stack(images, dim=0).unsqueeze(0)

        port_tensor = None
        if port_id is not None:
            port_tensor = torch.tensor([int(port_id)], dtype=torch.long, device=self.device)
        pred = self.model(model_input.to(self.device), port_tensor).cpu()[0]
        pred_mm = pred * self.target_std + self.target_mean
        offset_m = pred_mm.numpy().astype(np.float64) / 1000.0
        if not np.isfinite(offset_m).all():
            self._log_warn(f"Non-finite distance prediction: {offset_m}")
            return None
        return offset_m


class ModelFeedbackController:
    """Use predicted residual offsets in separate align and insert stages."""

    def __init__(
        self,
        policy,
        predictor: VisionOffsetPredictor,
        config=DistancePredictionConfig,
    ) -> None:
        self._policy = policy
        self._predictor = predictor
        self._config = config
        self._filtered_offset_m: Optional[np.ndarray] = None

    def get_logger(self):
        return self._policy.get_logger()

    def sleep_for(self, duration_sec: float) -> None:
        self._policy.sleep_for(duration_sec)

    def set_pose_target(self, *args, **kwargs):
        return self._policy.set_pose_target(*args, **kwargs)

    def _copy_pose(self, pose: Pose) -> Pose:
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

    def _tcp_pose(self, observation) -> Optional[Pose]:
        if observation is None:
            return None
        return self._copy_pose(observation.controller_state.tcp_pose)

    def _force_mag(self, observation) -> Optional[float]:
        if observation is None:
            return None
        force = observation.wrist_wrench.wrench.force
        return float(np.sqrt(force.x * force.x + force.y * force.y + force.z * force.z))

    def _port_id(self) -> int:
        task = self._policy._task
        for attr in ("port_name", "target_module_name"):
            value = str(getattr(task, attr, "") or "")
            match = re.search(r"_(\d+)$", value)
            if match is not None:
                return int(match.group(1))
        return 0

    def _is_sfp_task(self) -> bool:
        task = self._policy._task
        tokens = " ".join(
            str(value or "").lower()
            for value in (
                getattr(task, "plug_name", ""),
                getattr(task, "port_name", ""),
                getattr(task, "port_type", ""),
                getattr(task, "task_type", ""),
            )
        )
        return "sfp" in tokens

    def _port_frame(self) -> str:
        task = self._policy._task
        base_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        mode = str(getattr(self._config, "LABEL_PORT_FRAME_MODE", "entrance")).lower()
        if self._is_sfp_task() and mode in {"entrance", "auto"}:
            return f"{base_frame}_entrance"
        return base_frame

    def _lookup_port_rotation(self) -> Optional[Quaternion]:
        frame = self._port_frame()
        try:
            transform = self._policy._parent_node._tf_buffer.lookup_transform(
                "base_link",
                frame,
                Time(),
            ).transform
        except TransformException as exc:
            if (
                self._is_sfp_task()
                and str(getattr(self._config, "LABEL_PORT_FRAME_MODE", "entrance")).lower() == "auto"
            ):
                fallback_frame = f"task_board/{self._policy._task.target_module_name}/{self._policy._task.port_name}_link"
                try:
                    transform = self._policy._parent_node._tf_buffer.lookup_transform(
                        "base_link",
                        fallback_frame,
                        Time(),
                    ).transform
                except TransformException:
                    pass
                else:
                    return Quaternion(
                        x=float(transform.rotation.x),
                        y=float(transform.rotation.y),
                        z=float(transform.rotation.z),
                        w=float(transform.rotation.w),
                    )
            self.get_logger().warn(
                f"Port TF unavailable for distance correction ({frame}): {exc}"
            )
            return None
        return Quaternion(
            x=float(transform.rotation.x),
            y=float(transform.rotation.y),
            z=float(transform.rotation.z),
            w=float(transform.rotation.w),
        )

    def _correction_in_base(self, offset_m: np.ndarray) -> np.ndarray:
        correction_port = -np.asarray(offset_m, dtype=np.float64)
        port_rotation = self._lookup_port_rotation()
        if port_rotation is None:
            return correction_port
        return rotate_vector_by_quat(correction_port, quat_to_tuple(port_rotation))

    def _smooth_offset(self, offset_m: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self._config.SMOOTHING_ALPHA, 0.0, 1.0))
        if self._filtered_offset_m is None:
            self._filtered_offset_m = offset_m
        else:
            self._filtered_offset_m = (
                alpha * offset_m + (1.0 - alpha) * self._filtered_offset_m
            )
        return self._filtered_offset_m

    def _bounded_step(
        self,
        offset_m: np.ndarray,
        force_limited: bool,
        *,
        allow_xy: bool,
        allow_z: bool,
    ) -> np.ndarray:
        step = np.zeros(3, dtype=np.float64)

        correction_m = self._correction_in_base(offset_m)

        if allow_xy:
            xy = correction_m[:2] * self._config.XY_GAIN
            xy[np.abs(xy) < self._config.XY_DEADBAND_M] = 0.0
            step[:2] = np.clip(
                xy,
                -self._config.MAX_XY_STEP_M,
                self._config.MAX_XY_STEP_M,
            )

        if allow_z:
            z = float(correction_m[2] * self._config.Z_GAIN)
            if abs(z) < self._config.Z_DEADBAND_M:
                z = 0.0

            if force_limited:
                z = np.clip(z, 0.0, self._config.MAX_UP_STEP_M)
            else:
                z = np.clip(
                    z,
                    -self._config.MAX_DOWN_STEP_M,
                    self._config.MAX_UP_STEP_M,
                )
            step[2] = z
        return step

    def _pose_from_tcp_and_step(
        self,
        tcp_pose: Pose,
        step_m: np.ndarray,
        min_z: float,
    ) -> Pose:
        target_z = max(float(tcp_pose.position.z + step_m[2]), min_z)
        return Pose(
            position=Point(
                x=float(tcp_pose.position.x + step_m[0]),
                y=float(tcp_pose.position.y + step_m[1]),
                z=target_z,
            ),
            orientation=self._copy_pose(tcp_pose).orientation,
        )

    def _initial_min_z(self, get_observation) -> Optional[float]:
        initial_observation = get_observation()
        start_pose = self._tcp_pose(initial_observation)
        if start_pose is None:
            self.get_logger().error("Model feedback failed: missing TCP pose")
            return None
        return float(start_pose.position.z - self._config.MAX_INSERT_DEPTH_M)

    def _predict_smoothed_offset(self, observation) -> Optional[np.ndarray]:
        offset_m = self._predictor.predict_offset_m(observation, self._port_id())
        if offset_m is None:
            return None
        return self._smooth_offset(offset_m)

    def _move_from_offset(
        self,
        *,
        offset_m: np.ndarray,
        observation,
        move_robot,
        min_z: float,
        allow_xy: bool,
        allow_z: bool,
    ) -> tuple[np.ndarray, Optional[float]]:
        force_mag = self._force_mag(observation)
        force_limited = force_mag is not None and force_mag > self._config.FORCE_LIMIT_N
        tcp_pose = self._tcp_pose(observation)
        if tcp_pose is None:
            self.get_logger().warn("Model feedback skipped: missing TCP pose")
            return np.zeros(3, dtype=np.float64), force_mag

        step_m = self._bounded_step(
            offset_m,
            force_limited,
            allow_xy=allow_xy,
            allow_z=allow_z,
        )
        pose = self._pose_from_tcp_and_step(tcp_pose, step_m, min_z)
        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=list(self._config.STIFFNESS),
            damping=list(self._config.DAMPING),
        )
        return step_m, force_mag

    def _run_align(self, get_observation, move_robot, send_feedback, min_z: float) -> bool:
        self.get_logger().info("Model align stage start")
        send_feedback("Model feedback: aligning XY")
        self._filtered_offset_m = None

        stable_count = 0
        last_xy_m = None

        for step_idx in range(self._config.ALIGN_MAX_STEPS):
            observation = get_observation()
            offset_m = self._predict_smoothed_offset(observation)
            if offset_m is None:
                self.sleep_for(self._config.DT)
                continue

            xy_m = float(np.linalg.norm(offset_m[:2]))
            last_xy_m = xy_m

            if xy_m < self._config.ALIGN_FINISH_XY_M:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= self._config.ALIGN_STABLE_STEPS:
                self.get_logger().info(
                    "Model align stable: "
                    f"xy={xy_m * 1000.0:.1f}mm x {stable_count}"
                )
                return True

            step_m, force_mag = self._move_from_offset(
                offset_m=offset_m,
                observation=observation,
                move_robot=move_robot,
                min_z=min_z,
                allow_xy=True,
                allow_z=False,
            )

            if step_idx % 10 == 0:
                force_text = "N/A" if force_mag is None else f"{force_mag:.1f}N"
                self.get_logger().info(
                    f"[align {step_idx:03d}] "
                    f"offset=({offset_m[0]*1000:+.1f}, "
                    f"{offset_m[1]*1000:+.1f}, {offset_m[2]*1000:+.1f})mm, "
                    f"xy={xy_m*1000:.1f}mm, "
                    f"step=({step_m[0]*1000:+.1f}, {step_m[1]*1000:+.1f}, "
                    f"{step_m[2]*1000:+.1f})mm, force={force_text}"
                )

            self.sleep_for(self._config.DT)

        if last_xy_m is None:
            self.get_logger().warn("Model align finished without any prediction")
            return False

        success = last_xy_m < self._config.ALIGN_FINISH_XY_M * 1.5
        self.get_logger().info(
            "Model align stage done: "
            f"success={success}, last_xy={last_xy_m*1000:.1f}mm"
        )
        return success

    def _run_insert(self, get_observation, move_robot, send_feedback, min_z: float) -> bool:
        self.get_logger().info("Model insert stage start")
        send_feedback("Model feedback: inserting")
        self._filtered_offset_m = None

        stable_count = 0
        last_distance_m = None

        for step_idx in range(self._config.INSERT_MAX_STEPS):
            observation = get_observation()
            offset_m = self._predict_smoothed_offset(observation)
            if offset_m is None:
                self.sleep_for(self._config.DT)
                continue

            residual_m = float(np.linalg.norm(offset_m))
            last_distance_m = residual_m

            if residual_m < self._config.FINISH_DISTANCE_M:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= self._config.FINISH_STABLE_STEPS:
                self.get_logger().info(
                    "Model insert residual stable: "
                    f"{residual_m * 1000.0:.1f}mm x {stable_count}"
                )
                break

            step_m, force_mag = self._move_from_offset(
                offset_m=offset_m,
                observation=observation,
                move_robot=move_robot,
                min_z=min_z,
                allow_xy=False,
                allow_z=True,
            )

            if step_idx % 10 == 0:
                force_text = "N/A" if force_mag is None else f"{force_mag:.1f}N"
                self.get_logger().info(
                    f"[insert {step_idx:03d}] "
                    f"offset=({offset_m[0]*1000:+.1f}, "
                    f"{offset_m[1]*1000:+.1f}, {offset_m[2]*1000:+.1f})mm, "
                    f"step=({step_m[0]*1000:+.1f}, "
                    f"{step_m[1]*1000:+.1f}, {step_m[2]*1000:+.1f})mm, "
                    f"residual={residual_m*1000:.1f}mm, force={force_text}"
                )

            self.sleep_for(self._config.DT)

        if last_distance_m is None:
            self.get_logger().warn("Model insert finished without any prediction")
            return False

        success = (
            stable_count >= self._config.FINISH_STABLE_STEPS
            or last_distance_m < self._config.FINISH_DISTANCE_M * 1.5
        )
        self.get_logger().info(
            "Model insert stage done: "
            f"success={success}, last_residual={last_distance_m*1000:.1f}mm"
        )
        return success

    def run(self, get_observation, move_robot, send_feedback) -> bool:
        self.get_logger().info("Model feedback stages start: align -> insert")
        min_z = self._initial_min_z(get_observation)
        if min_z is None:
            return False

        aligned = self._run_align(get_observation, move_robot, send_feedback, min_z)
        if not aligned:
            self.get_logger().warn("Model feedback aborted: align stage failed")
            send_feedback("failed: model align failed")
            return False

        success = self._run_insert(get_observation, move_robot, send_feedback, min_z)
        if self._config.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(self._config.SETTLE_AFTER_INSERT_S)
        return success
