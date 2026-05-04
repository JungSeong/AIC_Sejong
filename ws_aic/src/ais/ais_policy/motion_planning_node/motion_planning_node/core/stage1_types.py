"""Shared result types for Stage 1 motion planning."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose


@dataclass
class Stage1Result:
    success: bool
    final_pose: Optional[Pose]
    port_pose: Optional[Pose]
    port_axis: Optional[np.ndarray]
    elapsed_time: float
    failure_reason: Optional[str] = None
    port_source: str = "unknown"  # "tf" | "vision" | "fallback"


