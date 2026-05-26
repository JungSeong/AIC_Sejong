from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .dataset_format import get_target_config, normalize_target


CAMERA_NAMES = ("left", "center", "right")
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_WIDTH = 1152
IMAGE_HEIGHT = 1024
HORIZONTAL_FOV_RAD = 0.8718

CAMERA_LINK_POSES = {
    "center": ((0.0, -0.1077, -0.00719), (0.0, -1.30899630, 1.57079623)),
    "left": ((-0.09326, -0.053843, -0.007188), (0.0, -1.30899630, 0.523599027)),
    "right": ((0.09326, -0.053843, -0.007188), (0.0, -1.30899630, 2.61799343)),
}
SENSOR_IN_CAMERA_LINK = ((0.02174, 0.0, 0.0145), (0.0, 0.0, 0.0))
OPTICAL_IN_SENSOR = ((0.0, 0.0, 0.0), (-1.5708, 0.0, -1.5708))


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rpy(*rpy)
    transform[:3, 3] = np.array(xyz, dtype=np.float64)
    return transform


def camera_to_rig_transforms() -> dict[str, np.ndarray]:
    sensor_to_camera = transform_from_xyz_rpy(*SENSOR_IN_CAMERA_LINK)
    optical_to_sensor = transform_from_xyz_rpy(*OPTICAL_IN_SENSOR)
    transforms = {}
    for name, pose in CAMERA_LINK_POSES.items():
        camera_link_to_rig = transform_from_xyz_rpy(*pose)
        transforms[name] = camera_link_to_rig @ sensor_to_camera @ optical_to_sensor
    return transforms


def intrinsics_from_fov(
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    horizontal_fov_rad: float = HORIZONTAL_FOV_RAD,
) -> np.ndarray:
    focal = width / (2.0 * np.tan(horizontal_fov_rad / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def projection_matrices(
    camera_transforms: Mapping[str, np.ndarray],
    camera_k: Mapping[str, np.ndarray] | np.ndarray,
) -> dict[str, np.ndarray]:
    projections = {}
    for name, camera_to_rig in camera_transforms.items():
        k = camera_k[name] if isinstance(camera_k, Mapping) else camera_k
        rig_to_camera = np.linalg.inv(camera_to_rig)
        projections[name] = k @ rig_to_camera[:3, :]
    return projections


def triangulate_multiview(points_by_camera: Mapping[str, np.ndarray], projections: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = []
    for camera, point in points_by_camera.items():
        u, v = np.asarray(point, dtype=np.float64)
        p = projections[camera]
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    if len(rows) < 4:
        raise ValueError("Need observations from at least two cameras.")
    _, _, vh = np.linalg.svd(np.vstack(rows))
    point_h = vh[-1]
    return point_h[:3] / point_h[3]


def project_to_camera(point_3d_rig: np.ndarray, camera: str, camera_to_rig: np.ndarray, camera_k: np.ndarray) -> np.ndarray | None:
    point_camera = np.linalg.inv(camera_to_rig) @ np.append(point_3d_rig, 1.0)
    x, y, z = point_camera[:3]
    if z <= 1e-9:
        return None
    u = camera_k[0, 0] * x / z + camera_k[0, 2]
    v = camera_k[1, 1] * y / z + camera_k[1, 2]
    return np.array([u, v], dtype=np.float64)


def split_episode_camera(stem: str) -> tuple[str, str] | None:
    for camera in CAMERA_NAMES:
        suffix = f"_{camera}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], camera
    return None


def label_path_for_image(image_path: Path, dataset_dir: Path, split: str) -> Path:
    return dataset_dir / "labels" / split / f"{image_path.stem}.txt"


def group_split_images(dataset_dir: Path, split: str) -> dict[str, dict[str, Path]]:
    image_dir = dataset_dir / "images" / split
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        episode_camera = split_episode_camera(image_path.stem)
        if episode_camera is None:
            continue
        episode, camera = episode_camera
        grouped[episode][camera] = image_path
    return dict(grouped)


def load_gt_points(label_path: Path, image_shape: tuple[int, int, int], target: str) -> dict[str, np.ndarray]:
    if not label_path.exists():
        return {}
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}

    values = np.array([float(value) for value in re.split(r"\s+", lines[0])], dtype=np.float64)
    height, width = image_shape[:2]
    target = normalize_target(target)
    keypoint_names = get_target_config(target)["keypoint_names"]

    if keypoint_names:
        expected_cols = 5 + len(keypoint_names) * 2
        if len(values) != expected_cols:
            raise ValueError(
                f"{label_path}: expected {expected_cols} values for {target} pose labels, "
                f"got {len(values)}. Recollect {target} data with the current keypoint format."
            )
        keypoints = values[5:].reshape(-1, 2)
        points = {}
        for name, xy in zip(keypoint_names, keypoints):
            points[name] = np.array([xy[0] * width, xy[1] * height], dtype=np.float64)
        return points

    x_center, y_center = values[1], values[2]
    return {"sc_center": np.array([x_center * width, y_center * height], dtype=np.float64)}


def predict_points(model, image_bgr: np.ndarray, target: str, conf: float, imgsz: int, device: str | None) -> dict[str, np.ndarray]:
    target = normalize_target(target)
    keypoint_names = get_target_config(target)["keypoint_names"]
    kwargs = {"verbose": False, "conf": conf, "imgsz": imgsz}
    if device is not None:
        kwargs["device"] = device
    result = model(image_bgr, **kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return {}

    confidences = result.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confidences))

    if keypoint_names:
        if result.keypoints is None or result.keypoints.xy is None:
            return {}
        keypoints = result.keypoints.xy.detach().cpu().numpy()[best_idx]
        return {
            name: np.array(keypoints[idx], dtype=np.float64)
            for idx, name in enumerate(keypoint_names)
        }

    xyxy = result.boxes.xyxy.detach().cpu().numpy()[best_idx]
    center = np.array([(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0], dtype=np.float64)
    return {"sc_center": center}


def target_point_names(target: str) -> list[str]:
    config = get_target_config(target)
    if config["task"] == "pose":
        return list(config["keypoint_names"])
    return ["sc_center"]


def evaluate_validation_triangulation(
    dataset_dir: Path,
    model_path: Path,
    target: str,
    split: str = "val",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    max_episodes: int | None = None,
) -> list[dict]:
    from ultralytics import YOLO

    target = normalize_target(target)
    dataset_dir = Path(dataset_dir)
    model_path = Path(model_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    camera_k = intrinsics_from_fov()
    camera_transforms = camera_to_rig_transforms()
    projections = projection_matrices(camera_transforms, camera_k)
    grouped_images = group_split_images(dataset_dir, split)
    model = YOLO(str(model_path))

    records = []
    for episode_idx, (episode, images_by_camera) in enumerate(sorted(grouped_images.items())):
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        gt_by_camera = {}
        pred_by_camera = {}
        image_shapes = {}
        for camera, image_path in sorted(images_by_camera.items()):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image_shapes[camera] = image.shape
            label_path = label_path_for_image(image_path, dataset_dir, split)
            gt_by_camera[camera] = load_gt_points(label_path, image.shape, target)
            pred_by_camera[camera] = predict_points(model, image, target, conf, imgsz, device)

        for point_name in target_point_names(target):
            gt_views = {
                camera: points[point_name]
                for camera, points in gt_by_camera.items()
                if point_name in points and camera in projections
            }
            pred_views = {
                camera: points[point_name]
                for camera, points in pred_by_camera.items()
                if point_name in points and camera in projections
            }
            if len(gt_views) < 2 or len(pred_views) < 2:
                continue

            gt_3d = triangulate_multiview(gt_views, projections)
            pred_3d = triangulate_multiview(pred_views, projections)
            reprojection_errors = []
            for camera, gt_uv in gt_views.items():
                pred_uv = project_to_camera(pred_3d, camera, camera_transforms[camera], camera_k)
                if pred_uv is not None:
                    reprojection_errors.append(float(np.linalg.norm(pred_uv - gt_uv)))

            if not reprojection_errors:
                continue

            delta_3d = pred_3d - gt_3d
            error_3d = float(np.linalg.norm(delta_3d))
            records.append(
                {
                    "episode": episode,
                    "point": point_name,
                    "target": target,
                    "gt_cameras": "+".join(sorted(gt_views)),
                    "pred_cameras": "+".join(sorted(pred_views)),
                    "n_gt_cameras": len(gt_views),
                    "n_pred_cameras": len(pred_views),
                    "dx_m": float(delta_3d[0]),
                    "dy_m": float(delta_3d[1]),
                    "dz_m": float(delta_3d[2]),
                    "dx_mm": float(delta_3d[0] * 1000.0),
                    "dy_mm": float(delta_3d[1] * 1000.0),
                    "dz_mm": float(delta_3d[2] * 1000.0),
                    "error_3d_m": error_3d,
                    "error_3d_mm": float(error_3d * 1000.0),
                    "reprojection_error_px_mean": float(np.mean(reprojection_errors)),
                    "reprojection_error_px_max": float(np.max(reprojection_errors)),
                }
            )
    return records


def empirical_coverage_thresholds(values: np.ndarray, coverages: tuple[float, ...] = (0.90, 0.95, 0.99)) -> dict[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {coverage: float("nan") for coverage in coverages}
    return {coverage: float(np.quantile(values, coverage)) for coverage in coverages}


def central_coverage_intervals(values: np.ndarray, coverages: tuple[float, ...] = (0.90, 0.95, 0.99)) -> dict[float, tuple[float, float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {coverage: (float("nan"), float("nan")) for coverage in coverages}

    intervals = {}
    for coverage in coverages:
        tail = (1.0 - coverage) / 2.0
        intervals[coverage] = (
            float(np.quantile(values, tail)),
            float(np.quantile(values, 1.0 - tail)),
        )
    return intervals
