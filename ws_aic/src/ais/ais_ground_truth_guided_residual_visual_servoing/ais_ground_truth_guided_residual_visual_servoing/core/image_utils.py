from __future__ import annotations

import cv2
import numpy as np


def image_msg_to_bgr(image_msg) -> np.ndarray | None:
    if image_msg is None or int(getattr(image_msg, "width", 0)) == 0:
        return None
    if int(getattr(image_msg, "height", 0)) == 0:
        return None

    height = int(image_msg.height)
    width = int(image_msg.width)
    encoding = str(getattr(image_msg, "encoding", "")).lower()
    if encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        channels = 3

    flat = np.frombuffer(image_msg.data, dtype=np.uint8)
    step = int(getattr(image_msg, "step", 0))
    if step > 0 and flat.size >= height * step:
        rows = flat[: height * step].reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)
    else:
        expected = height * width * channels
        if flat.size < expected:
            return None
        image = flat[:expected].reshape(height, width, channels)

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image).copy()
