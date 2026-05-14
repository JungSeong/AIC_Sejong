from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import LabelThresholds, RetryLabel, SUCCESS_CLASS


@dataclass
class InstantFeature:
    t_s: float
    pred_xy_offset_mm: float
    fz: float
    delta_fz: float
    fxy_norm: float
    cmd_insert_depth_mm: float


def make_instant_feature(
    *,
    t_s: float,
    pred_offset_m,
    wrench,
    baseline_fz: float,
    insert_start_z_m: float,
    current_tcp_z_m: float,
) -> InstantFeature:
    """Build deployable retry-classifier features from runtime observations."""

    pred_offset = np.asarray(pred_offset_m, dtype=np.float64)
    force = wrench.force
    fx = float(force.x)
    fy = float(force.y)
    fz = float(force.z)
    cmd_depth = max(0.0, float(insert_start_z_m - current_tcp_z_m) * 1000.0)
    return InstantFeature(
        t_s=float(t_s),
        pred_xy_offset_mm=float(np.linalg.norm(pred_offset[:2]) * 1000.0),
        fz=fz,
        delta_fz=float(fz - baseline_fz),
        fxy_norm=float(math.hypot(fx, fy)),
        cmd_insert_depth_mm=cmd_depth,
    )


def summarize_episode(
    samples: Iterable[InstantFeature],
    success_event_observed: bool,
    contact_count: int,
) -> dict[str, float | int]:
    values = list(samples)
    if not values:
        return {
            "pred_xy_offset_mm": 0.0,
            "fz": 0.0,
            "delta_fz": 0.0,
            "fxy_norm": 0.0,
            "cmd_insert_depth_mm": 0.0,
            "sample_count": 0,
            "contact_count": int(contact_count),
            "success_event_observed": int(success_event_observed),
        }

    final = values[-1]
    return {
        "pred_xy_offset_mm": final.pred_xy_offset_mm,
        "fz": final.fz,
        "delta_fz": final.delta_fz,
        "fxy_norm": final.fxy_norm,
        "cmd_insert_depth_mm": final.cmd_insert_depth_mm,
        "sample_count": len(values),
        "contact_count": int(contact_count),
        "success_event_observed": int(success_event_observed),
    }


def label_from_features(
    row: dict[str, float | int | str],
    thresholds: LabelThresholds,
) -> RetryLabel:
    if int(row.get("success_event_observed", 0)) == 1:
        return RetryLabel(SUCCESS_CLASS, 1, "insertion_event_observed")

    pred_xy = float(row.get("pred_xy_offset_mm", 1e9))
    fxy_norm = float(row.get("fxy_norm", 0.0))
    delta_fz = float(row.get("delta_fz", 0.0))
    cmd_depth = float(row.get("cmd_insert_depth_mm", 0.0))

    if pred_xy <= thresholds.centered_xy_mm and (
        delta_fz >= thresholds.fz_stuck_n
        or (
            cmd_depth >= thresholds.min_cmd_insert_depth_mm
            and delta_fz >= thresholds.fz_contact_n
        )
    ):
        return RetryLabel("partial_insert", 0, "near_center_with_force_stuck")
    if pred_xy >= thresholds.wall_xy_mm and fxy_norm >= thresholds.fxy_contact_n:
        return RetryLabel("side_wall_contact", 0, "far_from_center_with_lateral_force")
    if pred_xy >= thresholds.wall_xy_mm and delta_fz >= thresholds.fz_contact_n:
        return RetryLabel("top_surface_contact", 0, "far_from_center_with_vertical_force")
    return RetryLabel("timeout_or_unknown", 0, "no_success_event_observed")
