from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rclpy.time import Time

from .config import ModelEvalConfig
from .geometry import (
    SFP_TIP_TOP_CENTER_OFFSET_M,
    TransformState,
    local_distance_label,
    orientation_correction_label,
    shift_transform_origin,
    transform_state_from_tf,
)


def port_id_from_task(task: Any) -> int:
    text = str(getattr(task, "port_name", "") or "")
    try:
        return int(text.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def sfp_port_frame_candidates(task: Any) -> tuple[str, ...]:
    module = str(getattr(task, "target_module_name", "") or "")
    port = str(getattr(task, "port_name", "") or "")
    if port.endswith("_link"):
        names = (f"{port}_entrance", port)
    else:
        names = (f"{port}_link_entrance", f"{port}_link", port)
    return tuple(f"task_board/{module}/{name}" for name in names)


def sfp_plug_frame_candidates(task: Any) -> tuple[str, ...]:
    cable = str(getattr(task, "cable_name", "") or "")
    plug = str(getattr(task, "plug_name", "") or "")
    names: list[str] = []
    if plug:
        names.extend(
            [
                f"{cable}/{plug}_tip_link",
                f"{cable}/{plug}_link",
                f"{cable}/{plug}",
                f"{plug}_tip_link",
                f"{plug}_link",
                plug,
            ]
        )
    names.extend([f"{cable}/sfp_tip_link", f"{cable}/sfp_tip_tip_link", f"{cable}/sfp_link"])
    deduped = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)


@dataclass(frozen=True)
class GroundTruthState:
    port_frame: str
    plug_frame: str
    port: TransformState
    plug_raw: TransformState
    plug_reference: TransformState
    distance: dict[str, float]
    orientation: dict[str, Any]


class GroundTruthReader:
    def __init__(self, parent_node, *, logger=None) -> None:
        self.parent_node = parent_node
        self.logger = logger

    @property
    def tf_buffer(self):
        return getattr(self.parent_node, "_tf_buffer", None)

    def lookup_first(self, frames: tuple[str, ...]) -> tuple[str, TransformState] | None:
        if self.tf_buffer is None:
            return None
        deadline = time.monotonic() + ModelEvalConfig.TF_WAIT_S
        while True:
            for frame in frames:
                try:
                    tf = self.tf_buffer.lookup_transform("base_link", frame, Time()).transform
                except Exception:
                    continue
                return frame, transform_state_from_tf(tf)
            if time.monotonic() >= deadline:
                return None
            time.sleep(ModelEvalConfig.TF_POLL_S)

    def read(self, task: Any) -> GroundTruthState | None:
        port = self.lookup_first(sfp_port_frame_candidates(task))
        plug = self.lookup_first(sfp_plug_frame_candidates(task))
        if port is None or plug is None:
            if self.logger is not None:
                self.logger.warn(
                    "Ground-truth TF unavailable: "
                    f"port={sfp_port_frame_candidates(task)}, plug={sfp_plug_frame_candidates(task)}"
                )
            return None
        plug_ref = shift_transform_origin(plug[1], SFP_TIP_TOP_CENTER_OFFSET_M)
        return GroundTruthState(
            port_frame=port[0],
            plug_frame=plug[0],
            port=port[1],
            plug_raw=plug[1],
            plug_reference=plug_ref,
            distance=local_distance_label(port[1], plug_ref),
            orientation=orientation_correction_label(port[1], plug_ref),
        )


def sample_initial_offset_and_rpy(
    seed: int,
    trial_index: int,
    attempt_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        int(seed) + 1009 * int(trial_index) + 104729 * int(attempt_index)
    )
    dz_min = min(ModelEvalConfig.DZ_MIN_M, ModelEvalConfig.DZ_MAX_M)
    dz_max = max(ModelEvalConfig.DZ_MIN_M, ModelEvalConfig.DZ_MAX_M)
    offset = np.array(
        [
            rng.uniform(ModelEvalConfig.DX_MIN_M, ModelEvalConfig.DX_MAX_M),
            rng.uniform(ModelEvalConfig.DY_MIN_M, ModelEvalConfig.DY_MAX_M),
            rng.uniform(dz_min, dz_max),
        ],
        dtype=np.float64,
    )
    rpy_norm_max = float(ModelEvalConfig.RPY_NORM_MAX_RAD)
    rpy = None
    for _ in range(256):
        candidate = np.array(
            [
                rng.uniform(ModelEvalConfig.ROLL_MIN_RAD, ModelEvalConfig.ROLL_MAX_RAD),
                rng.uniform(ModelEvalConfig.PITCH_MIN_RAD, ModelEvalConfig.PITCH_MAX_RAD),
                rng.uniform(ModelEvalConfig.YAW_MIN_RAD, ModelEvalConfig.YAW_MAX_RAD),
            ],
            dtype=np.float64,
        )
        if rpy_norm_max <= 0.0 or float(np.linalg.norm(candidate)) <= rpy_norm_max:
            rpy = candidate
            break
    if rpy is None:
        rpy = candidate
        norm = float(np.linalg.norm(rpy))
        if norm > 1e-12 and rpy_norm_max > 0.0:
            rpy = rpy * (rpy_norm_max / norm)
    return offset, rpy
