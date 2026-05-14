"""Port-relative offset collection policy.

This policy reuses DataCollect2's dataset recording path, but replaces the
COLLECT motion sampler with wide port-local XYZ offsets and diverse RPY.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from geometry_msgs.msg import Pose, Transform

from .DataCollect2 import (
    DataCollect2,
    _matrix_from_translation_quat,
    _quat_from_axis_angle_xyzw,
    _quat_multiply_xyzw,
)

_SFP_PORT_SIZE_M = (0.014, 0.010)


def _env_mm(name: str, default_mm: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default_mm / 1000.0
    try:
        return float(value) / 1000.0
    except ValueError:
        return default_mm / 1000.0


def _env_optional_mm(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except ValueError:
        return None


def _env_mm_range(
    min_name: str,
    max_name: str,
    default_min_m: float,
    default_max_m: float,
) -> tuple[float, float]:
    low = _env_optional_mm(min_name)
    high = _env_optional_mm(max_name)
    if low is None:
        low = default_min_m
    if high is None:
        high = default_max_m
    if low > high:
        low, high = high, low
    return low, high


def _env_deg(name: str, default_deg: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return np.deg2rad(default_deg)
    try:
        return np.deg2rad(float(value))
    except ValueError:
        return np.deg2rad(default_deg)


def _env_optional_deg(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return np.deg2rad(float(value))
    except ValueError:
        return None


def _env_deg_range(
    min_name: str,
    max_name: str,
    default_min_rad: float,
    default_max_rad: float,
) -> tuple[float, float]:
    low = _env_optional_deg(min_name)
    high = _env_optional_deg(max_name)
    if low is None:
        low = default_min_rad
    if high is None:
        high = default_max_rad
    if low > high:
        low, high = high, low
    return low, high


def _default_dataset_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[5] / "data" / "ais_rpy_randomization"
    version = os.environ.get("AIC_RPY_DATASET_VERSION", "").strip()
    return base_dir / version if version else base_dir


def _port_corners_in_frame(port_size_m: tuple[float, float]) -> np.ndarray:
    width, height = port_size_m
    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=float,
    )


def _order_image_corners(points: np.ndarray) -> np.ndarray:
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmax(diffs)],
            points[np.argmax(sums)],
            points[np.argmin(diffs)],
        ],
        dtype=float,
    )


def _make_bbox_from_points(
    points: np.ndarray,
    image_w: int,
    image_h: int,
    margin: float,
) -> tuple[float, float, float, float] | None:
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    pad_x = (x_max - x_min) * margin
    pad_y = (y_max - y_min) * margin

    x_min = float(np.clip(x_min - pad_x, 0, image_w - 1))
    y_min = float(np.clip(y_min - pad_y, 0, image_h - 1))
    x_max = float(np.clip(x_max + pad_x, 0, image_w - 1))
    y_max = float(np.clip(y_max + pad_y, 0, image_h - 1))
    if x_max <= x_min or y_max <= y_min:
        return None

    return (
        ((x_min + x_max) / 2.0) / image_w,
        ((y_min + y_max) / 2.0) / image_h,
        (x_max - x_min) / image_w,
        (y_max - y_min) / image_h,
    )


class PortOffsetCollect(DataCollect2):
    """Collect data by moving around the target port in port-local XYZ/RPY."""

    def __init__(self, parent_node):
        os.environ.setdefault("AIC_COLLECT_STEPS", "24")
        dataset_dir = Path(
            os.environ.setdefault(
                "AIC_VISION_OFFSET_DATASET_DIR", str(_default_dataset_dir())
            )
        ).expanduser()
        super().__init__(parent_node)
        self._rpy_dataset_dir = dataset_dir
        self._rpy_dataset_version = os.environ.get("AIC_RPY_DATASET_VERSION", "").strip()
        self._rpy_metadata_path = self._rpy_dataset_dir / "metadata.jsonl"
        self._rpy_sample_count = 0
        self._rpy_val_ratio = float(
            os.environ.get("AIC_RPY_RANDOMIZATION_VAL_RATIO", "0.1")
        )
        self._rpy_visibility_margin_px = float(
            os.environ.get("AIC_RPY_VISIBILITY_MARGIN_PX", "8.0")
        )
        self._rpy_min_visible_cameras = max(
            1, int(os.environ.get("AIC_RPY_MIN_VISIBLE_CAMERAS", "1"))
        )
        self._yolo_bbox_margin = float(os.environ.get("AIC_RPY_YOLO_BBOX_MARGIN", "0.08"))
        self.collect_pattern = "port_offset_xyz_rpy"
        xy_limit_m = _env_mm("AIC_PORT_COLLECT_XY_LIMIT_MM", 50.0)
        z_limit_m = _env_mm("AIC_PORT_COLLECT_Z_LIMIT_MM", 150.0)
        self.port_collect_x_min_m, self.port_collect_x_max_m = _env_mm_range(
            "AIC_PORT_COLLECT_DX_MIN_MM",
            "AIC_PORT_COLLECT_DX_MAX_MM",
            -xy_limit_m,
            xy_limit_m,
        )
        self.port_collect_y_min_m, self.port_collect_y_max_m = _env_mm_range(
            "AIC_PORT_COLLECT_DY_MIN_MM",
            "AIC_PORT_COLLECT_DY_MAX_MM",
            -xy_limit_m,
            xy_limit_m,
        )
        self.port_collect_z_min_m, self.port_collect_z_max_m = _env_mm_range(
            "AIC_PORT_COLLECT_DZ_MIN_MM",
            "AIC_PORT_COLLECT_DZ_MAX_MM",
            -z_limit_m,
            z_limit_m,
        )
        roll_limit_rad = _env_deg(
            "AIC_PORT_COLLECT_ROLL_LIMIT_DEG", 25.0
        )
        pitch_limit_rad = _env_deg(
            "AIC_PORT_COLLECT_PITCH_LIMIT_DEG", 25.0
        )
        yaw_limit_rad = _env_deg(
            "AIC_PORT_COLLECT_YAW_LIMIT_DEG", 35.0
        )
        self.port_collect_roll_min_rad, self.port_collect_roll_max_rad = _env_deg_range(
            "AIC_PORT_COLLECT_ROLL_MIN_DEG",
            "AIC_PORT_COLLECT_ROLL_MAX_DEG",
            -roll_limit_rad,
            roll_limit_rad,
        )
        self.port_collect_pitch_min_rad, self.port_collect_pitch_max_rad = _env_deg_range(
            "AIC_PORT_COLLECT_PITCH_MIN_DEG",
            "AIC_PORT_COLLECT_PITCH_MAX_DEG",
            -pitch_limit_rad,
            pitch_limit_rad,
        )
        self.port_collect_yaw_min_rad, self.port_collect_yaw_max_rad = _env_deg_range(
            "AIC_PORT_COLLECT_YAW_MIN_DEG",
            "AIC_PORT_COLLECT_YAW_MAX_DEG",
            -yaw_limit_rad,
            yaw_limit_rad,
        )
        self._write_rpy_data_yaml()
        self._port_collect_samples = self._build_port_collect_samples(
            max(1, self.collect_steps)
        )
        self.get_logger().info(
            "[PortOffsetCollect] Ready: "
            f"dx=[{self.port_collect_x_min_m * 1000.0:.1f}, "
            f"{self.port_collect_x_max_m * 1000.0:.1f}]mm, "
            f"dy=[{self.port_collect_y_min_m * 1000.0:.1f}, "
            f"{self.port_collect_y_max_m * 1000.0:.1f}]mm, "
            f"dz=[{self.port_collect_z_min_m * 1000.0:.1f}, "
            f"{self.port_collect_z_max_m * 1000.0:.1f}]mm, "
            f"roll=[{np.rad2deg(self.port_collect_roll_min_rad):.1f}, "
            f"{np.rad2deg(self.port_collect_roll_max_rad):.1f}]deg, "
            f"pitch=[{np.rad2deg(self.port_collect_pitch_min_rad):.1f}, "
            f"{np.rad2deg(self.port_collect_pitch_max_rad):.1f}]deg, "
            f"yaw=[{np.rad2deg(self.port_collect_yaw_min_rad):.1f}, "
            f"{np.rad2deg(self.port_collect_yaw_max_rad):.1f}]deg, "
            f"dataset={self._rpy_dataset_dir}, "
            f"version={self._rpy_dataset_version or 'default'}, "
            f"min_visible_cameras={self._rpy_min_visible_cameras}, "
            "first COLLECT sample uses zero offset/rpy"
        )

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
                    "metadata: metadata",
                    "label_format: json_sidecar",
                    "task: ais_rpy_randomization",
                    f"version: {self._rpy_dataset_version or 'default'}",
                    "collect_range_mm:",
                    f"  dx: [{self.port_collect_x_min_m * 1000.0:.6f}, {self.port_collect_x_max_m * 1000.0:.6f}]",
                    f"  dy: [{self.port_collect_y_min_m * 1000.0:.6f}, {self.port_collect_y_max_m * 1000.0:.6f}]",
                    f"  dz: [{self.port_collect_z_min_m * 1000.0:.6f}, {self.port_collect_z_max_m * 1000.0:.6f}]",
                    "collect_rpy_range_deg:",
                    f"  roll: [{np.rad2deg(self.port_collect_roll_min_rad):.6f}, {np.rad2deg(self.port_collect_roll_max_rad):.6f}]",
                    f"  pitch: [{np.rad2deg(self.port_collect_pitch_min_rad):.6f}, {np.rad2deg(self.port_collect_pitch_max_rad):.6f}]",
                    f"  yaw: [{np.rad2deg(self.port_collect_yaw_min_rad):.6f}, {np.rad2deg(self.port_collect_yaw_max_rad):.6f}]",
                    "fields:",
                    "  command: base_link target pose xyz + quaternion xyzw",
                    "  collect: port-local dx/dy/dz + roll/pitch/yaw",
                    "  wrench: raw observation wrist force/torque; no LPF",
                    "  yolo_keypoints: see yolo/data.yaml",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        yolo_dir = self._rpy_dataset_dir / "yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        (yolo_dir / "data.yaml").write_text(
            "\n".join(
                [
                    f"path: {yolo_dir.resolve()}",
                    "train: images/train",
                    "val: images/val",
                    "names:",
                    "  0: port_pair",
                    "kpt_shape: [8, 2]",
                    "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]",
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

    def _lookup_optional_transform_matrix(self, frame: str) -> np.ndarray | None:
        for candidate in (f"{frame}_entrance", frame):
            try:
                tf = self._lookup_transform("base_link", candidate)
                return _matrix_from_translation_quat(
                    [tf.translation.x, tf.translation.y, tf.translation.z],
                    [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w],
                )
            except Exception:
                continue
        return None

    def _project_point(
        self,
        point_base: np.ndarray,
        camera_k: np.ndarray,
        base_to_camera: np.ndarray,
    ) -> tuple[float, float, float] | None:
        point_cam = base_to_camera @ np.append(point_base, 1.0)
        depth = float(point_cam[2])
        if depth < 1e-6:
            return None
        u = float(camera_k[0, 0] * point_cam[0] / depth + camera_k[0, 2])
        v = float(camera_k[1, 1] * point_cam[1] / depth + camera_k[1, 2])
        return u, v, depth

    def _sfp_yolo_keypoint_label(self, obs, camera_name: str, task) -> dict[str, Any]:
        img_msg = self._image_msg_for_camera(obs, camera_name)
        camera_k = self._camera_intrinsic_matrix(self._camera_info_for_camera(obs, camera_name))
        if img_msg is None or img_msg.width == 0 or img_msg.height == 0 or camera_k is None:
            return {"available": False, "reason": "missing_image_or_intrinsics"}

        try:
            base_to_camera = self._base_to_camera_optical_matrix(obs, camera_name)
        except Exception as exc:
            return {"available": False, "reason": f"camera_transform_error: {exc}"}

        local_corners = _port_corners_in_frame(_SFP_PORT_SIZE_M)
        projected_ports = []
        for port_name in ("sfp_port_0", "sfp_port_1"):
            port_frame = f"task_board/{task.target_module_name}/{port_name}_link"
            port_to_base = self._lookup_optional_transform_matrix(port_frame)
            if port_to_base is None:
                return {"available": False, "reason": f"missing_tf:{port_frame}"}

            projected = []
            for corner in local_corners:
                point_base = (port_to_base @ np.append(corner, 1.0))[:3]
                uvz = self._project_point(point_base, camera_k, base_to_camera)
                if uvz is None:
                    return {"available": False, "reason": "point_behind_camera"}
                u, v, depth = uvz
                if (
                    u < 0.0
                    or u >= float(img_msg.width)
                    or v < 0.0
                    or v >= float(img_msg.height)
                ):
                    return {"available": False, "reason": "keypoint_out_of_frame"}
                projected.append([u, v, depth])

            ordered = _order_image_corners(np.asarray(projected, dtype=float)[:, :2])
            projected_ports.append(
                {
                    "port_name": port_name,
                    "center_x": float(np.mean(ordered[:, 0])),
                    "points_px": ordered,
                    "depth_m": float(np.mean(np.asarray(projected, dtype=float)[:, 2])),
                }
            )

        projected_ports.sort(key=lambda item: item["center_x"])
        points_px = np.vstack([item["points_px"] for item in projected_ports])
        bbox = _make_bbox_from_points(
            points_px,
            int(img_msg.width),
            int(img_msg.height),
            self._yolo_bbox_margin,
        )
        if bbox is None:
            return {"available": False, "reason": "invalid_bbox"}

        normalized_points = points_px.copy()
        normalized_points[:, 0] /= float(img_msg.width)
        normalized_points[:, 1] /= float(img_msg.height)
        values = [0, *bbox, *normalized_points.reshape(-1).tolist()]
        label_line = " ".join(
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in values
        )
        return {
            "available": True,
            "target": "SFP",
            "class_id": 0,
            "class_name": "port_pair",
            "format": "class x_center y_center width height kpt0_x kpt0_y ... kpt7_x kpt7_y",
            "label": label_line,
            "bbox_xywh_norm": [float(v) for v in bbox],
            "keypoints_norm": normalized_points.tolist(),
            "keypoints_px": points_px.tolist(),
            "port_order_left_to_right": [item["port_name"] for item in projected_ports],
        }

    def _scenario_metadata(self, task) -> dict[str, Any]:
        try:
            params = json.loads(
                self._scenario_params_file.read_text(encoding="utf-8")
            )
            return params.get(task.id, {}) or {}
        except Exception:
            return {}

    def _wrist_wrench_metadata(self, obs) -> dict[str, Any]:
        wrist_wrench = getattr(obs, "wrist_wrench", None)
        wrench = getattr(wrist_wrench, "wrench", None)
        if wrench is None:
            return {"available": False, "reason": "missing_wrist_wrench"}

        force = wrench.force
        torque = wrench.torque
        force_xyz = np.array([force.x, force.y, force.z], dtype=float)
        torque_xyz = np.array([torque.x, torque.y, torque.z], dtype=float)
        return {
            "available": True,
            "source": "observation.wrist_wrench.wrench",
            "filter": "raw_no_lpf",
            "force": {
                "x": float(force_xyz[0]),
                "y": float(force_xyz[1]),
                "z": float(force_xyz[2]),
                "norm": float(np.linalg.norm(force_xyz)),
            },
            "torque": {
                "x": float(torque_xyz[0]),
                "y": float(torque_xyz[1]),
                "z": float(torque_xyz[2]),
                "norm": float(np.linalg.norm(torque_xyz)),
            },
        }

    def _save_yolo_style_rpy_sample(
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
            image_path = self._rpy_dataset_dir / "images" / split / f"{stem}.jpg"
            metadata_path = self._rpy_dataset_dir / "metadata" / split / f"{stem}.json"
            yolo_image_path = self._rpy_dataset_dir / "yolo" / "images" / split / f"{stem}.jpg"
            yolo_label_path = self._rpy_dataset_dir / "yolo" / "labels" / split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            yolo_image_path.parent.mkdir(parents=True, exist_ok=True)
            yolo_label_path.parent.mkdir(parents=True, exist_ok=True)

            yolo_keypoints = self._sfp_yolo_keypoint_label(obs, camera_name, task)
            if not yolo_keypoints.get("available", False):
                continue

            cv2.imwrite(str(image_path), bgr)
            cv2.imwrite(str(yolo_image_path), bgr)
            yolo_label_path.write_text(
                str(yolo_keypoints["label"]) + "\n",
                encoding="utf-8",
            )

            label_record = {
                "sample_id": sample_id,
                "camera": camera_name,
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
                "wrench": self._wrist_wrench_metadata(obs),
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
                "label": {
                    "plug_tip_to_port": self._plug_tip_to_port_label(port_tf, plug_tf),
                    "ports": self._all_ports_relative_label(task, plug_tf),
                    "insertion_wrist": self._insertion_wrist_label(obs, plug_tf),
                },
                "triangulation": {
                    "valid": bool(extras.get("triangulated_tip_to_port_offsets_valid", False)),
                    "x_m": float(extras.get("triangulated_x_offset", 0.0)),
                    "y_m": float(extras.get("triangulated_y_offset", 0.0)),
                    "z_m": float(extras.get("triangulated_z_offset", 0.0)),
                    "xy_m": float(extras.get("triangulated_xy_offset", 0.0)),
                },
                "yolo": {
                    "keypoints": yolo_keypoints,
                    "label_path": (
                        str(yolo_label_path.relative_to(self._rpy_dataset_dir))
                        if yolo_keypoints.get("available", False)
                        else ""
                    ),
                    "image_path": str(yolo_image_path.relative_to(self._rpy_dataset_dir)),
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
                "[PortOffsetCollect] Skipping sample: no camera had valid YOLO keypoints"
            )
            return

        metadata = {
            "sample_id": sample_id,
            "split": split,
            "phase": phase,
            "step_index": int(step_idx),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "images": image_records,
            "metadata_dir": f"metadata/{split}",
            "yolo_images_dir": f"yolo/images/{split}",
            "yolo_labels_dir": f"yolo/labels/{split}",
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
        self._save_yolo_style_rpy_sample(
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

    def _stratified_axis(self, low: float, high: float, steps: int) -> np.ndarray:
        if steps <= 1:
            return np.array([(low + high) * 0.5], dtype=float)
        edges = np.linspace(low, high, steps + 1, dtype=float)
        values = self._collect_rng.uniform(edges[:-1], edges[1:])
        self._collect_rng.shuffle(values)
        return values

    def _build_port_collect_samples(self, steps: int) -> list[dict[str, float]]:
        if steps <= 1:
            return [
                {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                }
            ]

        random_steps = steps - 1
        x_values = self._stratified_axis(
            self.port_collect_x_min_m,
            self.port_collect_x_max_m,
            random_steps,
        )
        y_values = self._stratified_axis(
            self.port_collect_y_min_m,
            self.port_collect_y_max_m,
            random_steps,
        )
        z_values = self._stratified_axis(
            self.port_collect_z_min_m,
            self.port_collect_z_max_m,
            random_steps,
        )
        roll_values = self._stratified_axis(
            self.port_collect_roll_min_rad,
            self.port_collect_roll_max_rad,
            random_steps,
        )
        pitch_values = self._stratified_axis(
            self.port_collect_pitch_min_rad,
            self.port_collect_pitch_max_rad,
            random_steps,
        )
        yaw_values = self._stratified_axis(
            self.port_collect_yaw_min_rad,
            self.port_collect_yaw_max_rad,
            random_steps,
        )

        samples = [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }
        ]
        for idx in range(random_steps):
            samples.append(
                {
                    "x": float(x_values[idx]),
                    "y": float(y_values[idx]),
                    "z": float(z_values[idx]),
                    "roll": float(roll_values[idx]),
                    "pitch": float(pitch_values[idx]),
                    "yaw": float(yaw_values[idx]),
                }
            )
        samples[1:] = sorted(
            samples[1:],
            key=lambda sample: float(
                np.linalg.norm([sample["x"], sample["y"], sample["z"]])
            ),
        )
        return samples

    def _sample_port_collect_offset(self, step_idx: int) -> dict[str, float]:
        if len(self._port_collect_samples) != max(1, self.collect_steps):
            self._port_collect_samples = self._build_port_collect_samples(
                max(1, self.collect_steps)
            )
        return self._port_collect_samples[step_idx % len(self._port_collect_samples)]

    def _apply_collect_offset(
        self,
        pose: Pose,
        port_transform: Transform,
        port_axis: Optional[dict[str, float]],
        step_idx: int,
    ) -> tuple[Pose, dict[str, Any]]:
        """Apply one stratified port-local XYZ offset and port-local RPY."""
        denom = float(max(1, self.collect_steps - 1))
        progress = float(np.clip(step_idx / denom, 0.0, 1.0))
        sample = self._sample_port_collect_offset(step_idx)
        x_axis, y_axis, z_axis = self._port_local_xy_axes(port_transform, port_axis)
        offset = sample["x"] * x_axis + sample["y"] * y_axis + sample["z"] * z_axis

        pose.position.x += float(offset[0])
        pose.position.y += float(offset[1])
        pose.position.z += float(offset[2])

        roll_quat = _quat_from_axis_angle_xyzw(x_axis, sample["roll"])
        pitch_quat = _quat_from_axis_angle_xyzw(y_axis, sample["pitch"])
        yaw_quat = _quat_from_axis_angle_xyzw(z_axis, sample["yaw"])
        delta_quat = _quat_multiply_xyzw(
            yaw_quat, _quat_multiply_xyzw(pitch_quat, roll_quat)
        )
        base_quat = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        qx, qy, qz, qw = _quat_multiply_xyzw(delta_quat, base_quat)
        q_norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
        if q_norm > 1e-9:
            pose.orientation.x = float(qx / q_norm)
            pose.orientation.y = float(qy / q_norm)
            pose.orientation.z = float(qz / q_norm)
            pose.orientation.w = float(qw / q_norm)

        distance = float(np.linalg.norm([sample["x"], sample["y"], sample["z"]]))
        return pose, {
            "collect_pattern": self.collect_pattern,
            "collect_progress": progress,
            "collect_theta": float(np.arctan2(sample["y"], sample["x"])),
            "collect_radius": float(np.hypot(sample["x"], sample["y"])),
            "collect_distance": distance,
            "collect_local_x": sample["x"],
            "collect_local_y": sample["y"],
            "collect_local_z": sample["z"],
            "collect_local_roll": sample["roll"],
            "collect_local_pitch": sample["pitch"],
            "collect_local_yaw": sample["yaw"],
            "collect_local_roll_deg": float(np.rad2deg(sample["roll"])),
            "collect_local_pitch_deg": float(np.rad2deg(sample["pitch"])),
            "collect_local_yaw_deg": float(np.rad2deg(sample["yaw"])),
            "collect_offset_x": float(offset[0]),
            "collect_offset_y": float(offset[1]),
            "collect_offset_z": float(offset[2]),
            "collect_spin_angle": sample["yaw"],
        }
