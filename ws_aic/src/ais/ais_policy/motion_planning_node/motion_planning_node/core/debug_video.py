"""Shared debug video recorder for staged policy runs."""

from datetime import datetime
from pathlib import Path
import re
import threading
from typing import Optional

import cv2
import numpy as np


class DebugVideoRecorder:
    """Append annotated BGR frames from multiple stages into one mp4."""

    def __init__(self, enabled: bool, output_dir: str, fps: float, logger=None):
        self._enabled = enabled
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._logger = logger
        self._lock = threading.RLock()
        self._writer = None
        self._frame_size: Optional[tuple[int, int]] = None
        self._path: Optional[Path] = None
        self._active = False

        if self._enabled:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as ex:
                self._enabled = False
                if self._logger:
                    self._logger.warn(f"[Staged Debug] video dir create failed: {ex}")

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def start(self, task=None) -> None:
        """Start a new output file for one staged-policy task."""
        self.close()
        if not self._enabled:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_label = self._task_label(task)
        self._path = self._output_dir / f"{timestamp}_{task_label}_stage0_to_stage3.mp4"
        self._active = True
        if self._logger:
            self._logger.info(f"[Staged Debug] final video path: {self._path}")

    def write(self, stage_label: str, image_bgr: np.ndarray, lines: Optional[list[str]] = None) -> None:
        """Write one frame to the final video."""
        if not self._enabled or not self._active or image_bgr is None:
            return
        with self._lock:
            try:
                frame = image_bgr.copy()
                if frame.ndim != 3 or frame.shape[2] != 3:
                    return
                self._draw_header(frame, stage_label, lines or [])

                h, w = frame.shape[:2]
                if self._writer is None:
                    if self._path is None:
                        self.start()
                    self._frame_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, self._frame_size)
                    if not self._writer.isOpened():
                        if self._logger:
                            self._logger.warn(f"[Staged Debug] video open failed: {self._path}")
                        self._writer = None
                        return
                elif self._frame_size != (w, h):
                    frame = cv2.resize(frame, self._frame_size, interpolation=cv2.INTER_AREA)

                self._writer.write(frame)
            except Exception as ex:
                if self._logger:
                    self._logger.warn(f"[Staged Debug] frame write failed: {ex}")

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                    if self._logger and self._path is not None:
                        self._logger.info(f"[Staged Debug] final video saved: {self._path}")
                except Exception as ex:
                    if self._logger:
                        self._logger.warn(f"[Staged Debug] video close failed: {ex}")
            self._writer = None
            self._frame_size = None
            self._path = None
            self._active = False

    @staticmethod
    def _task_label(task=None) -> str:
        if task is None:
            return "task"
        label = "_".join(
            part for part in (
                getattr(task, "id", ""),
                getattr(task, "plug_name", ""),
                getattr(task, "port_name", ""),
            )
            if part
        ) or "task"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:80]

    @staticmethod
    def _draw_header(frame: np.ndarray, stage_label: str, lines: list[str]) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28 + 22 * len(lines)), (0, 0, 0), -1)
        cv2.putText(
            frame,
            stage_label,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (12, 46 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
