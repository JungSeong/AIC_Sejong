from __future__ import annotations

"""Dataset and label writers for PortOffsetCollect."""

import json
import time
from typing import Any, Optional

import cv2
import numpy as np
from geometry_msgs.msg import Pose, Transform


def _connector_dir_for_task(task) -> str:
    """저장 경로에 사용할 커넥터 타입 디렉터리 이름을 반환한다."""
    for value in (
        getattr(task, "port_type", ""),
        getattr(task, "port_name", ""),
        getattr(task, "plug_type", ""),
        getattr(task, "plug_name", ""),
    ):
        text = str(value).lower()
        if "sfp" in text:
            return "SFP"
        if "sc" in text:
            return "SC"
    return "UNKNOWN"


def _split_for_sample(self, sample_index: int) -> str:
    if self._rpy_val_ratio <= 0.0:
        return "train"
    period = max(1, round(1.0 / self._rpy_val_ratio))
    return "val" if sample_index % period == 0 else "train"

def _write_rpy_data_yaml(self) -> None:
    self._rpy_dataset_dir.mkdir(parents=True, exist_ok=True)
    (self._rpy_dataset_dir / "data.yaml").write_text(
        "\n".join(
                [
                    f"path: {self._rpy_dataset_dir.resolve()}",
                    "train: images/train",
                    "val: images/val",
                    "image_layout: images/<split>/<connector>/<camera>/*.jpg",
                    "metadata: metadata/<split>/<connector>/<camera>/*.json",
                    "label_format: json_sidecar",
                "task: ais_rpy_randomization",
                f"version: {self._rpy_dataset_version or 'default'}",
                "collect_range_mm:",
                f"  dx: [{self.port_collect_x_min_m * 1000.0:.6f}, {self.port_collect_x_max_m * 1000.0:.6f}]",
                f"  dy: [{self.port_collect_y_min_m * 1000.0:.6f}, {self.port_collect_y_max_m * 1000.0:.6f}]",
                f"  dz: [{self.port_collect_z_min_m * 1000.0:.6f}, {self.port_collect_z_max_m * 1000.0:.6f}]",
                f"base_z_offset_mm: {self._triangulation_stop_z_offset * 1000.0:.6f}",
                f"capture_settle_s: {getattr(self, 'collect_capture_settle_sec', 0.0):.6f}",
                "collect_rpy_range_deg:",
                f"  roll: [{np.rad2deg(self.port_collect_roll_min_rad):.6f}, {np.rad2deg(self.port_collect_roll_max_rad):.6f}]",
                f"  pitch: [{np.rad2deg(self.port_collect_pitch_min_rad):.6f}, {np.rad2deg(self.port_collect_pitch_max_rad):.6f}]",
                f"  yaw: [{np.rad2deg(self.port_collect_yaw_min_rad):.6f}, {np.rad2deg(self.port_collect_yaw_max_rad):.6f}]",
                f"  norm_max_rad: {self.port_collect_rpy_norm_max_rad:.9f}",
                "fields:",
                "  command: base_link target pose xyz + quaternion xyzw",
                "  location: measured plug reference offset from port entrance in base_link after settle",
                "  label: base_link correction from plug reference to port entrance alignment",
                "  collect: commanded port-local dx/dy/dz + roll/pitch/yaw sample",
                "  image: camera image captured after command settle",
                "",
            ]
        ),
        encoding="utf-8",
    )

def _port_projection_for_camera(
    self,
    obs,
    camera_name: str,
    port_tf: Transform,
) -> dict[str, Any]:
    img_msg = self._image_msg_for_camera(obs, camera_name)
    k = self._camera_intrinsic_matrix(self._camera_info_for_camera(obs, camera_name))
    if img_msg is None or img_msg.width == 0 or img_msg.height == 0 or k is None:
        return {"visible": False, "reason": "missing_image_or_intrinsics"}

    try:
        t_cam_base = self._base_to_camera_optical_matrix(obs, camera_name)
    except Exception as exc:
        return {"visible": False, "reason": f"camera_transform_error: {exc}"}

    point_base = np.array(
        [
            port_tf.translation.x,
            port_tf.translation.y,
            port_tf.translation.z,
            1.0,
        ],
        dtype=float,
    )
    point_cam = t_cam_base @ point_base
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return {"visible": False, "reason": "behind_camera", "depth_m": depth}

    u = float(k[0, 0] * point_cam[0] / depth + k[0, 2])
    v = float(k[1, 1] * point_cam[1] / depth + k[1, 2])
    margin = self._rpy_visibility_margin_px
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

def _scenario_metadata(self, task) -> dict[str, Any]:
    try:
        params = json.loads(
            self._scenario_params_file.read_text(encoding="utf-8")
        )
        return params.get(task.id, {}) or {}
    except Exception:
        return {}

def _save_xyz_rpy_sample(
    self,
    episode_name: str,
    task,
    phase: str,
    step_idx: int,
    obs,
    port_tf: Transform,
    plug_tf: Transform,
    pose: Pose,
    extras: dict[str, Any],
    detections_by_camera: Optional[dict[str, Optional[dict[str, Any]]]],
) -> None:
    if obs is None:
        return

    projections = {
        camera_name: self._port_projection_for_camera(obs, camera_name, port_tf)
        for camera_name in ("left", "center", "right")
    }
    visible_cameras = [
        camera_name
        for camera_name, projection in projections.items()
        if projection.get("visible", False)
    ]
    if len(visible_cameras) < self._rpy_min_visible_cameras:
        self.get_logger().warn(
            "[PortOffsetCollect] Skipping sample: port not visible enough "
            f"visible={visible_cameras}, min={self._rpy_min_visible_cameras}"
        )
        return

    sample_index = self._rpy_sample_count
    self._rpy_sample_count += 1
    split = self._split_for_sample(sample_index)
    connector_dir = _connector_dir_for_task(task)
    sample_id = f"{episode_name}_{phase}_{step_idx:06d}"
    image_records = {}

    for camera_name in visible_cameras:
        bgr = self._image_msg_to_bgr(
            self._image_msg_for_camera(obs, camera_name),
            camera_name,
        )
        if bgr is None:
            image_records[camera_name] = ""
            continue

        stem = f"{sample_id}_{camera_name}"
        image_path = (
            self._rpy_dataset_dir
            / "images"
            / split
            / connector_dir
            / camera_name
            / f"{stem}.jpg"
        )
        metadata_path = (
            self._rpy_dataset_dir
            / "metadata"
            / split
            / connector_dir
            / camera_name
            / f"{stem}.json"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(image_path), bgr)

        label_record = {
            "sample_id": sample_id,
            "camera": camera_name,
            "connector": connector_dir,
            "image": str(image_path.relative_to(self._rpy_dataset_dir)),
            "task": {
                "id": task.id,
                "port_type": task.port_type,
                "port_name": task.port_name,
                "target_module_name": task.target_module_name,
                "cable_name": task.cable_name,
                "plug_name": task.plug_name,
            },
            "scenario": self._scenario_metadata(task),
            "command": {
                "position": {
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "z": float(pose.position.z),
                },
                "orientation_xyzw": {
                    "x": float(pose.orientation.x),
                    "y": float(pose.orientation.y),
                    "z": float(pose.orientation.z),
                    "w": float(pose.orientation.w),
                },
            },
            "location": {
                "x_m": float(extras.get("location", {}).get("x_m", 0.0)),
                "y_m": float(extras.get("location", {}).get("y_m", 0.0)),
                "z_m": float(extras.get("location", {}).get("z_m", 0.0)),
                "roll_rad": float(extras.get("location", {}).get("roll_rad", 0.0)),
                "pitch_rad": float(extras.get("location", {}).get("pitch_rad", 0.0)),
                "yaw_rad": float(extras.get("location", {}).get("yaw_rad", 0.0)),
            },
            "label": {
                "x_m": float(extras.get("label", {}).get("x_m", 0.0)),
                "y_m": float(extras.get("label", {}).get("y_m", 0.0)),
                "z_m": float(extras.get("label", {}).get("z_m", 0.0)),
                "roll_rad": float(extras.get("label", {}).get("roll_rad", 0.0)),
                "pitch_rad": float(extras.get("label", {}).get("pitch_rad", 0.0)),
                "yaw_rad": float(extras.get("label", {}).get("yaw_rad", 0.0)),
            },
            "collect": {
                "pattern": str(extras.get("collect_pattern", self.collect_pattern)),
                "local_x_m": float(extras.get("collect_local_x", 0.0)),
                "local_y_m": float(extras.get("collect_local_y", 0.0)),
                "local_z_m": float(extras.get("collect_local_z", 0.0)),
                "local_roll_rad": float(extras.get("collect_local_roll", 0.0)),
                "local_pitch_rad": float(extras.get("collect_local_pitch", 0.0)),
                "local_yaw_rad": float(extras.get("collect_local_yaw", 0.0)),
                "local_roll_deg": float(extras.get("collect_local_roll_deg", 0.0)),
                "local_pitch_deg": float(extras.get("collect_local_pitch_deg", 0.0)),
                "local_yaw_deg": float(extras.get("collect_local_yaw_deg", 0.0)),
                "distance_m": float(
                    extras.get("collect_distance", extras.get("collect_radius", 0.0))
                ),
            },
            "visibility": {
                "camera": projections[camera_name],
                "visible_cameras": visible_cameras,
            },
        }
        metadata_path.write_text(
            json.dumps(self._json_safe(label_record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_records[camera_name] = str(image_path.relative_to(self._rpy_dataset_dir))

    if not image_records:
        self.get_logger().warn(
            "[PortOffsetCollect] Skipping sample: no visible camera images"
        )
        return

    metadata = {
            "sample_id": sample_id,
            "split": split,
            "connector": connector_dir,
            "phase": phase,
            "step_index": int(step_idx),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "images": image_records,
            "metadata_dir": f"metadata/{split}/{connector_dir}",
            "image_layout": "images/<split>/<connector>/<camera>",
            "metadata_layout": "metadata/<split>/<connector>/<camera>",
            "visible_cameras": visible_cameras,
        }
    with self._rpy_metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(self._json_safe(metadata), ensure_ascii=False) + "\n")

def _save_vision_offset_sample(
    self,
    episode_name: str,
    task,
    phase: str,
    step_idx: int,
    obs,
    port_tf: Transform,
    plug_tf: Transform,
    pose: Pose,
    extras: dict[str, Any],
    detections_by_camera: Optional[dict[str, Optional[dict[str, Any]]]] = None,
) -> None:
    self._save_xyz_rpy_sample(
        episode_name,
        task,
        phase,
        step_idx,
        obs,
        port_tf,
        plug_tf,
        pose,
        extras,
        detections_by_camera,
    )
