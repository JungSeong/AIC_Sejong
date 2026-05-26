from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dataset_format import TASK_BOARD_KEYPOINT_NAMES, TASK_BOARD_KEYPOINT_POINTS_M
from .triangulation_eval import (
    camera_to_rig_transforms,
    group_split_images,
    intrinsics_from_fov,
    label_path_for_image,
)


DIST_COEFFS_ZERO = np.zeros((5, 1), dtype=np.float64)


def task_board_object_points() -> np.ndarray:
    return np.array(TASK_BOARD_KEYPOINT_POINTS_M, dtype=np.float64)


def _make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def _planar_points_for_ippe(object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return z=0 object points for IPPE and the offset back to board frame."""
    pnp_points = np.asarray(object_points, dtype=np.float64).copy()
    if not np.allclose(pnp_points[:, 2], pnp_points[0, 2], atol=1e-9):
        return pnp_points, np.zeros(3, dtype=np.float64)

    offset = np.array([0.0, 0.0, float(pnp_points[0, 2])], dtype=np.float64)
    pnp_points[:, 2] -= offset[2]
    return pnp_points, offset


def _to_board_frame_transform(t_camera_plane: np.ndarray, plane_offset_in_board: np.ndarray) -> np.ndarray:
    if np.allclose(plane_offset_in_board, 0.0):
        return t_camera_plane
    t_camera_board = t_camera_plane.copy()
    t_camera_board[:3, 3] -= t_camera_board[:3, :3] @ plane_offset_in_board
    return t_camera_board


def project_object_points(
    object_points: np.ndarray,
    t_camera_board: np.ndarray,
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    dist = DIST_COEFFS_ZERO if dist_coeffs is None else dist_coeffs
    rvec, _ = cv2.Rodrigues(t_camera_board[:3, :3])
    tvec = t_camera_board[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_k, dist)
    return projected.reshape(-1, 2)


def solve_task_board_pnp(
    image_points_px: np.ndarray,
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    object_points: np.ndarray | None = None,
) -> dict[str, Any] | None:
    object_points = task_board_object_points() if object_points is None else object_points
    image_points = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    if len(image_points) != len(object_points):
        raise ValueError(
            f"Expected {len(object_points)} TaskBoard keypoints, got {len(image_points)}"
        )

    dist = DIST_COEFFS_ZERO if dist_coeffs is None else dist_coeffs
    pnp_points, plane_offset = _planar_points_for_ippe(object_points)
    flags = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE

    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            pnp_points,
            image_points,
            camera_k,
            dist,
            flags=flags,
        )
        if retval:
            for rvec, tvec in zip(rvecs, tvecs):
                transform = _to_board_frame_transform(
                    _make_transform(rvec, tvec),
                    plane_offset,
                )
                projected = project_object_points(object_points, transform, camera_k, dist)
                errors = np.linalg.norm(projected - image_points, axis=1)
                depths = (transform[:3, :3] @ object_points.T + transform[:3, 3:4]).T[:, 2]
                if np.all(depths > 0.0):
                    candidates.append((float(np.mean(errors)), transform, projected, errors))
    except cv2.error:
        candidates = []

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(
            pnp_points,
            image_points,
            camera_k,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        transform = _to_board_frame_transform(
            _make_transform(rvec, tvec),
            plane_offset,
        )
        projected = project_object_points(object_points, transform, camera_k, dist)
        errors = np.linalg.norm(projected - image_points, axis=1)
        candidates.append((float(np.mean(errors)), transform, projected, errors))

    _, transform, projected, errors = min(candidates, key=lambda item: item[0])
    return {
        "t_camera_board": transform,
        "projected_points_px": projected,
        "reprojection_errors_px": errors,
        "reprojection_error_px_mean": float(np.mean(errors)),
        "reprojection_error_px_max": float(np.max(errors)),
    }


def load_task_board_gt_keypoints(label_path: Path, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not label_path.exists():
        return None
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    values = np.array([float(value) for value in lines[0].split()], dtype=np.float64)
    expected_cols = 5 + len(TASK_BOARD_KEYPOINT_NAMES) * 2
    if len(values) != expected_cols:
        raise ValueError(f"{label_path}: expected {expected_cols} values, got {len(values)}")

    height, width = image_shape[:2]
    keypoints = values[5:].reshape(-1, 2)
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def predict_task_board_keypoints(
    model,
    image_bgr: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | int | None = None,
) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {"verbose": False, "conf": conf, "imgsz": imgsz}
    if device is not None:
        kwargs["device"] = device
    result = model(image_bgr, **kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    scores = result.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(scores))
    keypoints = result.keypoints.xy.detach().cpu().numpy()[best_idx]
    if len(keypoints) < len(TASK_BOARD_KEYPOINT_NAMES):
        return None

    return {
        "keypoints_px": np.asarray(keypoints[: len(TASK_BOARD_KEYPOINT_NAMES)], dtype=np.float64),
        "confidence": float(scores[best_idx]),
    }


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a @ rotation_b.T
    cos_angle = (float(np.trace(delta)) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def _pose_spread(poses: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    if len(poses) < 2:
        return {
            "n": len(poses),
            "cameras": "+".join(camera for camera, _ in poses),
            "origin_std_mm": float("nan"),
            "origin_pairwise_max_mm": float("nan"),
            "rotation_pairwise_max_deg": float("nan"),
        }

    positions = np.array([pose[:3, 3] for _, pose in poses], dtype=np.float64)
    pairwise_distances = []
    pairwise_rotations = []
    for idx_a in range(len(poses)):
        for idx_b in range(idx_a + 1, len(poses)):
            _, pose_a = poses[idx_a]
            _, pose_b = poses[idx_b]
            pairwise_distances.append(float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]) * 1000.0))
            pairwise_rotations.append(rotation_error_deg(pose_a[:3, :3], pose_b[:3, :3]))

    return {
        "n": len(poses),
        "cameras": "+".join(camera for camera, _ in poses),
        "origin_std_mm": float(np.linalg.norm(np.std(positions, axis=0)) * 1000.0),
        "origin_pairwise_max_mm": float(max(pairwise_distances)),
        "rotation_pairwise_max_deg": float(max(pairwise_rotations)),
    }


def evaluate_task_board_pose_dataset(
    dataset_dir: Path,
    model_path: Path,
    split: str = "val",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | int | None = None,
    max_episodes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    model_path = Path(model_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    grouped_images = group_split_images(dataset_dir, split)
    camera_to_rig = camera_to_rig_transforms()

    records: list[dict[str, Any]] = []
    poses_by_episode: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = defaultdict(
        lambda: {"gt": [], "pred": []}
    )

    for episode_idx, (episode, images_by_camera) in enumerate(sorted(grouped_images.items())):
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        for camera, image_path in sorted(images_by_camera.items()):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                records.append(
                    {
                        "episode": episode,
                        "camera": camera,
                        "ok": False,
                        "reason": "image_read_failed",
                        "image": str(image_path),
                    }
                )
                continue

            height, width = image.shape[:2]
            camera_k = intrinsics_from_fov(width=width, height=height)
            label_path = label_path_for_image(image_path, dataset_dir, split)
            gt_keypoints = load_task_board_gt_keypoints(label_path, image.shape)
            pred = predict_task_board_keypoints(model, image, conf=conf, imgsz=imgsz, device=device)

            if gt_keypoints is None:
                records.append(
                    {
                        "episode": episode,
                        "camera": camera,
                        "ok": False,
                        "reason": "missing_gt_label",
                        "image": str(image_path),
                    }
                )
                continue
            if pred is None:
                records.append(
                    {
                        "episode": episode,
                        "camera": camera,
                        "ok": False,
                        "reason": "missing_prediction",
                        "image": str(image_path),
                    }
                )
                continue

            gt_pnp = solve_task_board_pnp(gt_keypoints, camera_k)
            pred_pnp = solve_task_board_pnp(pred["keypoints_px"], camera_k)
            if gt_pnp is None or pred_pnp is None:
                records.append(
                    {
                        "episode": episode,
                        "camera": camera,
                        "ok": False,
                        "reason": "pnp_failed",
                        "image": str(image_path),
                        "confidence": pred["confidence"],
                    }
                )
                continue

            t_camera_board_gt = gt_pnp["t_camera_board"]
            t_camera_board_pred = pred_pnp["t_camera_board"]
            pred_to_gt_projected = project_object_points(
                task_board_object_points(),
                t_camera_board_pred,
                camera_k,
            )
            pred_to_gt_reprojection = np.linalg.norm(pred_to_gt_projected - gt_keypoints, axis=1)
            keypoint_errors = np.linalg.norm(pred["keypoints_px"] - gt_keypoints, axis=1)

            if camera in camera_to_rig:
                poses_by_episode[episode]["gt"].append(
                    (camera, camera_to_rig[camera] @ t_camera_board_gt)
                )
                poses_by_episode[episode]["pred"].append(
                    (camera, camera_to_rig[camera] @ t_camera_board_pred)
                )

            translation_error_mm = float(
                np.linalg.norm(t_camera_board_pred[:3, 3] - t_camera_board_gt[:3, 3]) * 1000.0
            )
            records.append(
                {
                    "episode": episode,
                    "camera": camera,
                    "ok": True,
                    "reason": "",
                    "image": str(image_path),
                    "confidence": pred["confidence"],
                    "keypoint_error_px_mean": float(np.mean(keypoint_errors)),
                    "keypoint_error_px_max": float(np.max(keypoint_errors)),
                    "gt_pnp_residual_px_mean": gt_pnp["reprojection_error_px_mean"],
                    "gt_pnp_residual_px_max": gt_pnp["reprojection_error_px_max"],
                    "pred_pnp_residual_px_mean": pred_pnp["reprojection_error_px_mean"],
                    "pred_pnp_residual_px_max": pred_pnp["reprojection_error_px_max"],
                    "pred_to_gt_reprojection_px_mean": float(np.mean(pred_to_gt_reprojection)),
                    "pred_to_gt_reprojection_px_max": float(np.max(pred_to_gt_reprojection)),
                    "translation_error_mm": translation_error_mm,
                    "rotation_error_deg": rotation_error_deg(
                        t_camera_board_pred[:3, :3],
                        t_camera_board_gt[:3, :3],
                    ),
                    "gt_depth_m": float(t_camera_board_gt[2, 3]),
                    "pred_depth_m": float(t_camera_board_pred[2, 3]),
                }
            )

    consistency_records: list[dict[str, Any]] = []
    for episode, pose_groups in sorted(poses_by_episode.items()):
        gt_spread = _pose_spread(pose_groups["gt"])
        pred_spread = _pose_spread(pose_groups["pred"])
        consistency_records.append(
            {
                "episode": episode,
                "gt_n": gt_spread["n"],
                "gt_cameras": gt_spread["cameras"],
                "gt_origin_std_mm": gt_spread["origin_std_mm"],
                "gt_origin_pairwise_max_mm": gt_spread["origin_pairwise_max_mm"],
                "gt_rotation_pairwise_max_deg": gt_spread["rotation_pairwise_max_deg"],
                "pred_n": pred_spread["n"],
                "pred_cameras": pred_spread["cameras"],
                "pred_origin_std_mm": pred_spread["origin_std_mm"],
                "pred_origin_pairwise_max_mm": pred_spread["origin_pairwise_max_mm"],
                "pred_rotation_pairwise_max_deg": pred_spread["rotation_pairwise_max_deg"],
            }
        )

    return records, consistency_records


def percentile_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def summarize_pose_eval(
    records: list[dict[str, Any]],
    consistency_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    ok_records = [row for row in records if row.get("ok")]
    summary = {
        "counts": {
            "records": float(len(records)),
            "ok_records": float(len(ok_records)),
            "failed_records": float(len(records) - len(ok_records)),
            "consistency_records": float(len(consistency_records)),
        }
    }
    for key in [
        "keypoint_error_px_mean",
        "keypoint_error_px_max",
        "pred_to_gt_reprojection_px_mean",
        "pred_to_gt_reprojection_px_max",
        "translation_error_mm",
        "rotation_error_deg",
    ]:
        summary[key] = percentile_summary([float(row[key]) for row in ok_records if key in row])

    for key in [
        "pred_origin_std_mm",
        "pred_origin_pairwise_max_mm",
        "pred_rotation_pairwise_max_deg",
        "gt_origin_std_mm",
        "gt_origin_pairwise_max_mm",
        "gt_rotation_pairwise_max_deg",
    ]:
        summary[key] = percentile_summary(
            [float(row[key]) for row in consistency_records if key in row]
        )
    return summary
