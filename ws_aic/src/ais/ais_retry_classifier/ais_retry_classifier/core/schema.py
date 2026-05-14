from __future__ import annotations

from dataclasses import dataclass


SUCCESS_CLASS = "complete_insert"
FAILURE_CLASSES = (
    "partial_insert",
    "side_wall_contact",
    "top_surface_contact",
    "timeout_or_unknown",
)
CLASS_NAMES = (SUCCESS_CLASS, *FAILURE_CLASSES)

FEATURE_COLUMNS = (
    "pred_xy_offset_mm",
    "fz",
    "delta_fz",
    "fxy_norm",
    "cmd_insert_depth_mm",
)

IMAGE_COLUMNS = (
    "center_image_path",
)

LABEL_ONLY_COLUMNS = (
    "success_event_observed",
)


@dataclass(frozen=True)
class LabelThresholds:
    centered_xy_mm: float = 2.0
    wall_xy_mm: float = 4.0
    fxy_contact_n: float = 4.0
    fz_contact_n: float = 4.0
    fz_stuck_n: float = 8.0
    min_cmd_insert_depth_mm: float = 5.0


@dataclass(frozen=True)
class RetryLabel:
    class_name: str
    binary_success: int
    reason: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "class_name": self.class_name,
            "binary_success": self.binary_success,
            "label_reason": self.reason,
        }
