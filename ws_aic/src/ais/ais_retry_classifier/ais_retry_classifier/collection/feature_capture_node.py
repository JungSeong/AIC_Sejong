from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from aic_model_interfaces.msg import Observation
from rclpy.node import Node
from std_msgs.msg import String

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is optional; .npy fallback is used.
    cv2 = None

try:
    from ros_gz_interfaces.msg import Contacts
except Exception:  # pragma: no cover - keeps the node importable without Gazebo msgs.
    Contacts = None

from ..core.features import (
    InstantFeature,
    label_from_features,
    make_instant_feature,
    summarize_episode,
)
from ..core.io import append_csv_row, append_jsonl
from ..core.schema import FEATURE_COLUMNS, LabelThresholds, SUCCESS_CLASS


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class RetryFeatureCaptureNode(Node):
    """Capture one episode-level retry classifier feature row.

    The node uses only deployable runtime signals:
    center image, vision-predicted XY offset, force, and commanded insertion
    progress inferred from TCP z motion. TF ground-truth depth/offsets are not
    collected as model features.
    """

    def __init__(self) -> None:
        super().__init__("retry_feature_capture")
        self._latest_observation: Observation | None = None
        self._feature_observation: Observation | None = None
        self._samples: list[InstantFeature] = []
        self._success_event_observed = False
        self._latest_insertion_event = ""
        self._contact_count = 0
        self._done = False
        self._baseline_fz_values: list[float] = []
        self._baseline_fz: float | None = None
        self._insert_start_z_m: float | None = None
        self._prediction_fail_count = 0

        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._episode_id = self._param_str("episode_id", f"retry_{self._run_id}")
        self._intended_class = self._param_str("intended_class", "")
        self._force_class = self._param_str("force_class", "")
        self._target_module_name = self._param_str("target_module_name", "nic_card_mount_0")
        self._port_name = self._param_str("port_name", "sfp_port_0")
        self._cable_name = self._param_str("cable_name", "cable_0")
        self._plug_name = self._param_str("plug_name", "sfp_tip")
        self._observation_topic = self._param_str("observation_topic", "observations")
        self._distance_model_path = self._param_str("distance_model_path", "")
        self._distance_device = self._param_str("distance_device", "auto")
        self._output_dir = Path(
            os.path.expanduser(
                self._param_str(
                    "output_dir",
                    str(Path.cwd() / "data" / "retry_classifier"),
                )
            )
        ).resolve()
        self._duration_s = self._param_float("duration_s", 8.0)
        self._sample_interval_s = max(0.01, self._param_float("sample_interval_s", 0.05))
        self._baseline_window_s = max(0.0, self._param_float("baseline_window_s", 0.5))
        self._stop_on_success = self._param_bool("stop_on_success", False)
        self._post_success_s = max(0.0, self._param_float("post_success_s", 1.0))
        self._success_time_s: float | None = None
        self._start_time = self.get_clock().now()

        self._thresholds = LabelThresholds(
            centered_xy_mm=self._param_float("centered_xy_mm", 2.0),
            wall_xy_mm=self._param_float("wall_xy_mm", 4.0),
            fxy_contact_n=self._param_float("fxy_contact_n", 4.0),
            fz_contact_n=self._param_float("fz_contact_n", 4.0),
            fz_stuck_n=self._param_float("fz_stuck_n", 8.0),
            min_cmd_insert_depth_mm=self._param_float("min_cmd_insert_depth_mm", 5.0),
        )

        self._distance = self._make_distance_predictor()
        self.create_subscription(
            Observation,
            self._observation_topic,
            self._on_observation,
            10,
        )
        self.create_subscription(String, "/scoring/insertion_event", self._on_insertion_event, 10)
        if Contacts is not None:
            self.create_subscription(
                Contacts,
                "/aic/gazebo/contacts/off_limit",
                self._on_contacts,
                10,
            )
        self.create_timer(self._sample_interval_s, self._capture_once)
        self.create_timer(0.25, self._check_finished)
        self.get_logger().info(
            "RetryFeatureCaptureNode ready: "
            f"episode={self._episode_id}, intended={self._intended_class or 'auto'}, "
            f"target={self._target_module_name}/{self._port_name}, "
            f"observation_topic={self._observation_topic}, "
            f"duration={self._duration_s:.2f}s, baseline={self._baseline_window_s:.2f}s, "
            f"output={self._output_dir}"
        )

    @property
    def done(self) -> bool:
        return self._done

    def _param_str(self, name: str, default: str) -> str:
        env_name = f"AIC_RETRY_{name.upper()}"
        self.declare_parameter(name, os.environ.get(env_name, default))
        return str(self.get_parameter(name).value)

    def _param_float(self, name: str, default: float) -> float:
        env_name = f"AIC_RETRY_{name.upper()}"
        env_value = os.environ.get(env_name)
        self.declare_parameter(name, float(env_value) if env_value else default)
        return float(self.get_parameter(name).value)

    def _param_bool(self, name: str, default: bool) -> bool:
        env_name = f"AIC_RETRY_{name.upper()}"
        self.declare_parameter(name, _env_bool(env_name, default))
        return bool(self.get_parameter(name).value)

    def _make_distance_predictor(self):
        try:
            from distance_prediction_policy.model_feedback import VisionOffsetPredictor
        except Exception as exc:
            raise RuntimeError(
                "distance_prediction_policy is required to compute pred_xy_offset_mm"
            ) from exc

        kwargs = {"device": self._distance_device, "logger": self.get_logger()}
        if self._distance_model_path:
            kwargs["checkpoint_path"] = self._distance_model_path
        return VisionOffsetPredictor(**kwargs)

    def _port_id(self) -> int:
        match = re.search(r"_(\d+)$", self._port_name)
        return int(match.group(1)) if match is not None else 0

    def _on_observation(self, msg: Observation) -> None:
        self._latest_observation = msg

    def _on_insertion_event(self, msg: String) -> None:
        value = msg.data.strip().strip("/")
        if not value:
            return
        self._latest_insertion_event = value
        self._success_event_observed = True
        if self._success_time_s is None:
            self._success_time_s = self._elapsed_s()

    def _on_contacts(self, msg) -> None:
        contacts = getattr(msg, "contacts", None)
        self._contact_count += len(contacts) if contacts is not None else 1

    def _elapsed_s(self) -> float:
        return (self.get_clock().now() - self._start_time).nanoseconds * 1e-9

    def _update_baselines(self, obs: Observation, elapsed_s: float) -> None:
        force = obs.wrist_wrench.wrench.force
        tcp_pose = obs.controller_state.tcp_pose
        fz = float(force.z)
        if elapsed_s <= self._baseline_window_s or self._baseline_fz is None:
            self._baseline_fz_values.append(fz)
            self._baseline_fz = float(np.mean(self._baseline_fz_values))
            self._insert_start_z_m = float(tcp_pose.position.z)

    def _predict_offset_m(self, obs: Observation) -> Optional[np.ndarray]:
        try:
            return self._distance.predict_offset_m(obs, self._port_id())
        except Exception as exc:
            self._prediction_fail_count += 1
            if self._prediction_fail_count == 1 or self._prediction_fail_count % 20 == 0:
                self.get_logger().warn(f"distance prediction failed: {exc}")
            return None

    def _image_msg_to_rgb(self, image_msg) -> np.ndarray:
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()
        if height <= 0 or width <= 0:
            raise ValueError("empty image")

        if encoding in {"rgba8", "bgra8"}:
            channels = 4
        elif encoding in {"rgb8", "bgr8"}:
            channels = 3
        elif encoding in {"mono8", "8uc1"}:
            channels = 1
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
            image = flat[: height * width * channels].reshape(height, width, channels)

        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 4:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    def _save_center_image(self) -> str:
        obs = self._feature_observation
        if obs is None:
            return ""
        try:
            rgb = self._image_msg_to_rgb(obs.center_image)
        except Exception as exc:
            self.get_logger().warn(f"center image save skipped: {exc}")
            return ""

        image_dir = self._output_dir / "images" / "center"
        image_dir.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            path = image_dir / f"{self._episode_id}_center.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                self.get_logger().warn(f"center image save failed: {path}")
                return ""
        else:
            path = image_dir / f"{self._episode_id}_center.npy"
            np.save(path, rgb)
        return str(path)

    def _capture_once(self) -> None:
        if self._done or self._latest_observation is None:
            return
        obs = self._latest_observation
        elapsed = self._elapsed_s()
        self._update_baselines(obs, elapsed)
        pred_offset_m = self._predict_offset_m(obs)
        if pred_offset_m is None:
            return
        tcp_pose = obs.controller_state.tcp_pose
        self._samples.append(
            make_instant_feature(
                t_s=elapsed,
                pred_offset_m=pred_offset_m,
                wrench=obs.wrist_wrench.wrench,
                baseline_fz=float(self._baseline_fz or 0.0),
                insert_start_z_m=float(self._insert_start_z_m or tcp_pose.position.z),
                current_tcp_z_m=float(tcp_pose.position.z),
            )
        )
        self._feature_observation = obs

    def _check_finished(self) -> None:
        if self._done:
            return
        elapsed = self._elapsed_s()
        if self._stop_on_success and self._success_time_s is not None:
            if elapsed >= self._success_time_s + self._post_success_s:
                self._finish()
                return
        if elapsed >= self._duration_s:
            self._finish()

    def _finish(self) -> None:
        summary = summarize_episode(
            self._samples,
            success_event_observed=self._success_event_observed,
            contact_count=self._contact_count,
        )
        for column in FEATURE_COLUMNS:
            summary.setdefault(column, 0)
        label = label_from_features(summary, self._thresholds)
        if self._force_class:
            binary = 1 if self._force_class == SUCCESS_CLASS else 0
            label_dict = {
                "class_name": self._force_class,
                "binary_success": binary,
                "label_reason": "forced_by_parameter",
            }
        else:
            label_dict = label.as_dict()

        center_image_path = self._save_center_image()
        row = {
            "episode_id": self._episode_id,
            "run_id": self._run_id,
            "intended_class": self._intended_class,
            "target_module_name": self._target_module_name,
            "port_name": self._port_name,
            "cable_name": self._cable_name,
            "plug_name": self._plug_name,
            "observation_topic": self._observation_topic,
            "duration_s": round(self._elapsed_s(), 6),
            "baseline_window_s": self._baseline_window_s,
            "baseline_fz": self._baseline_fz if self._baseline_fz is not None else 0.0,
            "center_image_path": center_image_path,
            "latest_insertion_event": self._latest_insertion_event,
            **label_dict,
            **summary,
        }
        append_csv_row(self._output_dir / "features.csv", row)
        append_jsonl(self._output_dir / "episodes.jsonl", row)
        self._done = True
        self.get_logger().info(
            "retry feature episode saved: "
            f"episode={self._episode_id}, class={row['class_name']}, "
            f"success={row['binary_success']}, samples={row['sample_count']}, "
            f"csv={self._output_dir / 'features.csv'}"
        )


def main() -> None:
    rclpy.init()
    node = RetryFeatureCaptureNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
