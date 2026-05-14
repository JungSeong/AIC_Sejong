"""Debug distance policy using a single YOLO detector for all port types."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from aic_model.policy import Policy
from geometry_msgs.msg import Quaternion

from distance_prediction_policy.DebugSfpDistancePolicy import DebugSfpDistancePolicy
from distance_prediction_policy.config import DistancePredictionConfig
from distance_prediction_policy.model_feedback import VisionOffsetPredictor
from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.vision import VisionPortEstimator


SRC_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_YOLO_MODEL_PATHS = (
    SRC_ROOT / "model" / "yolo-port-keypoint-detection" / "weights" / "best.pt",
    SRC_ROOT
    / "model"
    / "yolo-port-keypoint-detection"
    / "approach"
    / "SFP"
    / "weights"
    / "best.pt",
)


def _first_existing_model_path(paths: tuple[Path, ...]) -> Optional[str]:
    for path in paths:
        if path.is_file():
            return str(path)
    return None


def _resolve_yolo_model_path() -> str:
    env_path = os.environ.get("AIC_YOLO_MODEL_PATH")
    if env_path:
        return env_path
    default_path = _first_existing_model_path(DEFAULT_YOLO_MODEL_PATHS)
    if default_path is not None:
        return default_path
    return Stage1Config.DETECTION_MODEL_PATH


class DebugDistancePolicy(DebugSfpDistancePolicy):
    """Debug policy variant that reuses one YOLO model for SFP and SC tasks."""

    TARGET_CLASS_ID = 0

    def __init__(self, parent_node):
        Policy.__init__(self, parent_node)
        self._task = None
        self._yolo_model_path = _resolve_yolo_model_path()
        self._cached_port_base: Optional[np.ndarray] = None
        self._target_orientation: Optional[Quaternion] = None
        self._fixed_target_orientation: Optional[Quaternion] = None
        self._yolo_conf_thresh = float(os.environ.get("AIC_DEBUG_YOLO_CONF_THRESH", "0.5"))
        self._vision_by_port_type = {}

        self._vision = self._vision_for_port_type("default")
        self._distance = VisionOffsetPredictor(logger=self.get_logger())

        self.get_logger().info(
            "DebugDistancePolicy ready: "
            f"yolo={self._yolo_model_path}, "
            f"distance={DistancePredictionConfig.CHECKPOINT_PATH}, "
            f"yolo_conf_thresh={self._yolo_conf_thresh}"
        )

    def _target_class_id(self, port_type: str) -> int:
        return int(os.environ.get("AIC_DEBUG_TARGET_CLASS_ID", self.TARGET_CLASS_ID))

    def _vision_for_port_type(self, port_type: str) -> VisionPortEstimator:
        cache_key = "unified"
        if cache_key not in self._vision_by_port_type:
            self.get_logger().info(f"Loading unified YOLO model: {self._yolo_model_path}")
            vision = VisionPortEstimator(
                model_path=self._yolo_model_path,
                conf_thresh=self._yolo_conf_thresh,
                logger=self.get_logger(),
            )
            vision._ensure_loaded()
            self._vision_by_port_type[cache_key] = vision
        return self._vision_by_port_type[cache_key]
