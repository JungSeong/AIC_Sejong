#!/usr/bin/env python3

import os
import time
from pathlib import Path
from typing import Optional

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


class SpawnHold(Policy):
    """Passive policy that keeps an aic_engine-spawned trial scene alive."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._stop_file = Path(
            os.environ.get("AIC_SPAWN_HOLD_STOP_FILE", "/tmp/aic_spawn_hold_stop")
        )
        self._max_hold_sec = float(os.environ.get("AIC_SPAWN_HOLD_MAX_SEC", "3600"))
        self._sleep_sec = float(os.environ.get("AIC_SPAWN_HOLD_SLEEP_SEC", "0.5"))
        self.get_logger().info(
            f"[SpawnHold] Passive scene-hold policy ready "
            f"(max_hold_sec={self._max_hold_sec:.1f})."
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> Optional[bool]:
        self.get_logger().info(
            f"[SpawnHold] Holding spawned scene for task '{task.id}'."
        )
        send_feedback("spawn hold running")

        start = time.monotonic()
        while time.monotonic() - start < self._max_hold_sec:
            if self._stop_file.exists():
                self.get_logger().info("[SpawnHold] Stop file detected; ending hold.")
                try:
                    self._stop_file.unlink()
                except OSError:
                    pass
                return False
            self.sleep_for(self._sleep_sec)

        self.get_logger().warn("[SpawnHold] Max hold time reached; ending hold.")
        return False
