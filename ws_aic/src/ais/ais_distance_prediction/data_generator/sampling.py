from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# Original YOLO triangulation CI99 in the camera rig/tool0 frame:
#   x=[-1.4305, +1.0263], y=[-2.1049, +0.9690], z=[-4.4473, +5.5253] mm.
# Converted through the current SFP pre-align transform into the plug-tip frame:
#   raw tip-frame error x=[-1.5938, +1.2268],
#   raw tip-frame error y=[-3.6157, +2.9126],
#   raw tip-frame error z=[-4.5043, +5.9138] mm.
#
# The generator stores z_error_mm as outward-distance error. Since local -Z is
# the outward approach direction, outward_error_mm = -raw_tip_z_error_mm.
DEFAULT_CI99_X_MM = (-1.5938, 1.2268)
DEFAULT_CI99_Y_MM = (-3.6157, 2.9126)
DEFAULT_CI99_Z_ERROR_MM = (-5.9138, 4.5043)
DEFAULT_APPROACH_OFFSET_MM = 20.0


@dataclass(frozen=True)
class OffsetRangeMM:
    low: float
    high: float

    def values(self, count: int) -> list[float]:
        if count < 1:
            raise ValueError("count must be >= 1")
        if count == 1:
            return [0.5 * (self.low + self.high)]
        return np.linspace(self.low, self.high, count).astype(float).tolist()

    def values_by_step(self, step: float) -> list[float]:
        if step <= 0.0:
            raise ValueError("step must be > 0")
        values = np.arange(self.low, self.high + step * 0.5, step, dtype=np.float64)
        values = values[values <= self.high]
        result = values.astype(float).tolist()
        if not result or abs(result[-1] - self.high) > 1e-9:
            result.append(float(self.high))
        return result

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LocalOffset:
    x_m: float
    y_m: float
    z_m: float
    x_error_mm: float
    y_error_mm: float
    z_error_mm: float
    approach_offset_mm: float

    def vector_m(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.z_m], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def make_local_offset(
    x_error_mm: float,
    y_error_mm: float,
    z_error_mm: float,
    approach_offset_mm: float,
) -> LocalOffset:
    """Convert tip-frame triangulation error in mm into a plug-tip offset.

    x_error_mm and y_error_mm are local lateral errors. z_error_mm is the
    outward-distance error, not raw local z. The existing controller treats
    local -Z as the outward approach direction, so a perfect 20 mm pre-align
    pose is labelled z_m=-0.020.
    """
    z_distance_mm = approach_offset_mm + z_error_mm
    if z_distance_mm <= 0.0:
        raise ValueError(
            "approach_offset_mm + z_error_mm must stay positive; "
            f"got {z_distance_mm:.4f} mm"
        )
    return LocalOffset(
        x_m=float(x_error_mm / 1000.0),
        y_m=float(y_error_mm / 1000.0),
        z_m=float(-z_distance_mm / 1000.0),
        x_error_mm=float(x_error_mm),
        y_error_mm=float(y_error_mm),
        z_error_mm=float(z_error_mm),
        approach_offset_mm=float(approach_offset_mm),
    )


def make_offset_grid(
    *,
    x_range: OffsetRangeMM,
    y_range: OffsetRangeMM,
    z_error_range: OffsetRangeMM,
    approach_offset_mm: float = DEFAULT_APPROACH_OFFSET_MM,
    points_per_axis: int = 5,
) -> list[LocalOffset]:
    offsets: list[LocalOffset] = []
    for x_mm in x_range.values(points_per_axis):
        for y_mm in y_range.values(points_per_axis):
            for z_mm in z_error_range.values(points_per_axis):
                offsets.append(
                    make_local_offset(
                        x_error_mm=x_mm,
                        y_error_mm=y_mm,
                        z_error_mm=z_mm,
                        approach_offset_mm=approach_offset_mm,
                    )
                )
    return offsets


def make_offset_step_grid(
    *,
    x_range: OffsetRangeMM,
    y_range: OffsetRangeMM,
    z_error_range: OffsetRangeMM,
    approach_offset_mm: float = DEFAULT_APPROACH_OFFSET_MM,
    step_mm: float = 1.0,
) -> list[LocalOffset]:
    offsets: list[LocalOffset] = []
    for x_mm in x_range.values_by_step(step_mm):
        for y_mm in y_range.values_by_step(step_mm):
            for z_mm in z_error_range.values_by_step(step_mm):
                offsets.append(
                    make_local_offset(
                        x_error_mm=x_mm,
                        y_error_mm=y_mm,
                        z_error_mm=z_mm,
                        approach_offset_mm=approach_offset_mm,
                    )
                )
    return offsets


def make_random_offsets(
    *,
    x_range: OffsetRangeMM,
    y_range: OffsetRangeMM,
    z_error_range: OffsetRangeMM,
    approach_offset_mm: float = DEFAULT_APPROACH_OFFSET_MM,
    count: int,
    seed: int = 42,
) -> list[LocalOffset]:
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = np.random.default_rng(seed)
    offsets: list[LocalOffset] = []
    for _ in range(count):
        offsets.append(
            make_local_offset(
                x_error_mm=float(rng.uniform(x_range.low, x_range.high)),
                y_error_mm=float(rng.uniform(y_range.low, y_range.high)),
                z_error_mm=float(rng.uniform(z_error_range.low, z_error_range.high)),
                approach_offset_mm=approach_offset_mm,
            )
        )
    return offsets


def make_uniform_offsets(
    *,
    x_range: OffsetRangeMM,
    y_range: OffsetRangeMM,
    z_error_range: OffsetRangeMM,
    approach_offset_mm: float = DEFAULT_APPROACH_OFFSET_MM,
    count: int,
    seed: int = 42,
) -> list[LocalOffset]:
    return make_random_offsets(
        x_range=x_range,
        y_range=y_range,
        z_error_range=z_error_range,
        approach_offset_mm=approach_offset_mm,
        count=count,
        seed=seed,
    )
