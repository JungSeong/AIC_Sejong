from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np
import rclpy
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from distance_prediction_policy.model_feedback import VisionOffsetPredictor

from .core.config import SfpGrvsConfig
from .core.geometry import plug_tip_to_port_label
from .core.task_frames import sfp_plug_tip_frame_candidates, sfp_port_frame_candidates
from .data.distance_dataset import SfpDistanceSampleRecorder
from .data.rotation_dataset import SfpRotationSampleRecorder


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class SfpGrvsCaptureNode(Node):
    """Capture GRVS datasets from observations and TF without commanding motion."""

    def __init__(self) -> None:
        super().__init__("sfp_grvs_capture")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=True,
        )
        self._latest_observation: Observation | None = None
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._sample_index = 0
        self._capture_index = 0
        self._episode_name = self._param_str("episode_name", "") or (
            f"capture_{self._run_id}"
        )
        self._task = self._task_from_params()
        self._record_distance = self._param_bool("record_distance", True)
        self._record_rotation = self._param_bool("record_rotation", True)
        self._include_backbone = self._param_bool("include_backbone_offset", True)
        self._max_samples = self._param_int("max_samples", 0)
        interval_s = self._param_float("sample_interval_s", 0.10)
        self._distance_recorder = SfpDistanceSampleRecorder(
            self._param_str("distance_dir", str(SfpGrvsConfig.DISTANCE_DATASET_DIR))
        )
        self._rotation_recorder = SfpRotationSampleRecorder(
            self._param_str("rotation_dir", str(SfpGrvsConfig.ROTATION_DATASET_DIR))
        )
        self._distance = (
            VisionOffsetPredictor(logger=self.get_logger())
            if self._record_distance and self._include_backbone
            else None
        )
        self.create_subscription(Observation, "observations", self._on_observation, 10)
        self.create_timer(max(0.01, interval_s), self._capture_once)
        self.get_logger().info(
            "SfpGrvsCaptureNode ready: "
            f"episode={self._episode_name}, "
            f"target={self._task.target_module_name}, port={self._task.port_name}, "
            f"distance={self._record_distance}, rotation={self._record_rotation}, "
            f"interval={interval_s:.3f}s, max_captures={self._max_samples or 'unlimited'}"
        )

    def _param_str(self, name: str, default: str) -> str:
        env_name = f"AIC_GRVS_CAPTURE_{name.upper()}"
        self.declare_parameter(name, os.environ.get(env_name, default))
        return str(self.get_parameter(name).value)

    def _param_int(self, name: str, default: int) -> int:
        env_name = f"AIC_GRVS_CAPTURE_{name.upper()}"
        env_value = os.environ.get(env_name)
        self.declare_parameter(name, int(env_value) if env_value else int(default))
        return int(self.get_parameter(name).value)

    def _param_float(self, name: str, default: float) -> float:
        env_name = f"AIC_GRVS_CAPTURE_{name.upper()}"
        env_value = os.environ.get(env_name)
        self.declare_parameter(name, float(env_value) if env_value else float(default))
        return float(self.get_parameter(name).value)

    def _param_bool(self, name: str, default: bool) -> bool:
        env_name = f"AIC_GRVS_CAPTURE_{name.upper()}"
        self.declare_parameter(name, _env_bool(env_name, default))
        return bool(self.get_parameter(name).value)

    def _task_from_params(self) -> Task:
        task = Task()
        task.id = self._param_str("task_id", "capture_sfp")
        task.cable_type = self._param_str("cable_type", "sfp_sc")
        task.cable_name = self._param_str("cable_name", "cable_0")
        task.plug_type = self._param_str("plug_type", "sfp")
        task.plug_name = self._param_str("plug_name", "sfp_tip")
        task.port_type = self._param_str("port_type", "sfp")
        task.port_name = self._param_str("port_name", "sfp_port_0")
        task.target_module_name = self._param_str(
            "target_module_name",
            "nic_card_mount_0",
        )
        task.time_limit = self._param_int("time_limit", 0)
        return task

    def _on_observation(self, msg: Observation) -> None:
        self._latest_observation = msg

    def _lookup_first_transform(self, frames: tuple[str, ...]):
        for frame in frames:
            try:
                return frame, self._tf_buffer.lookup_transform(
                    "base_link",
                    frame,
                    Time(),
                ).transform
            except Exception:
                continue
        return None

    def _next_sample_id(self, prefix: str) -> str:
        sample_id = f"{self._run_id}_{prefix}_{self._sample_index:06d}"
        self._sample_index += 1
        return sample_id

    def _port_id(self) -> int:
        match = re.search(r"_(\d+)$", str(self._task.port_name))
        return int(match.group(1)) if match is not None else 0

    def _distance_extras(self, observation, port_tf, plug_tf) -> dict[str, Any]:
        backbone_offset = None
        if self._distance is not None:
            backbone_offset = self._distance.predict_offset_m(observation, self._port_id())
        gt_label = plug_tip_to_port_label(port_tf, plug_tf)
        gt_offset = np.array(
            [gt_label["x_m"], gt_label["y_m"], gt_label["z_m"]],
            dtype=np.float64,
        )
        residual = None
        if backbone_offset is not None:
            residual = gt_offset - np.asarray(backbone_offset, dtype=np.float64)
        return {
            "capture_node": True,
            "backbone_offset_m": backbone_offset,
            "residual_target_m": residual,
            "action_m": None,
            "success_gt": None,
        }

    def _capture_once(self) -> None:
        if self._max_samples > 0 and self._capture_index >= self._max_samples:
            return
        observation = self._latest_observation
        if observation is None:
            return
        port = self._lookup_first_transform(sfp_port_frame_candidates(self._task))
        plug = self._lookup_first_transform(sfp_plug_tip_frame_candidates(self._task))
        if port is None or plug is None:
            self.get_logger().warn("capture skipped: missing port or plug TF")
            return

        saved = 0
        step_index = self._capture_index
        if self._record_distance:
            sample_id = self._next_sample_id("dist")
            if self._distance_recorder.record(
                observation=observation,
                task=self._task,
                sample_id=sample_id,
                episode_name=self._episode_name,
                step_index=step_index,
                port_tf=port[1],
                plug_tf=plug[1],
                port_frame=port[0],
                plug_frame=plug[0],
                extras=self._distance_extras(observation, port[1], plug[1]),
            ):
                saved += 1
        if self._record_rotation:
            sample_id = self._next_sample_id("rot")
            if self._rotation_recorder.record(
                observation=observation,
                task=self._task,
                sample_id=sample_id,
                episode_name=self._episode_name,
                stage="capture",
                port_tf=port[1],
                plug_tf=plug[1],
                port_frame=port[0],
                plug_frame=plug[0],
                extras={"capture_node": True},
            ):
                saved += 1
        if saved:
            self._capture_index += 1
            self.get_logger().info(
                f"capture saved: records={saved}, capture_index={self._capture_index}"
            )


def main() -> None:
    rclpy.init()
    node = SfpGrvsCaptureNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
