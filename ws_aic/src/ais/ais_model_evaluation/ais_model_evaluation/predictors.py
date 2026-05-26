from __future__ import annotations

import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image

from .config import ModelEvalConfig, SRC_ROOT


def _load_orientation_model_module():
    module_name = "_ais_orientation_prediction_model"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = (
        SRC_ROOT
        / "ais"
        / "ais_orientation_prediction"
        / "model"
        / "orientation_model.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing orientation model code: {module_path}")
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def image_msg_to_tensor(image_msg: Image, image_size: int | tuple[int, int] | None) -> torch.Tensor:
    height = int(image_msg.height)
    width = int(image_msg.width)
    encoding = getattr(image_msg, "encoding", "").lower()
    if encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"rgb8", "bgr8"}:
        channels = 3
    else:
        pixel_count = height * width
        channels = 4 if len(image_msg.data) >= pixel_count * 4 else 3

    flat = np.frombuffer(image_msg.data, dtype=np.uint8)
    step = int(getattr(image_msg, "step", 0))
    if step > 0 and flat.size >= height * step:
        rows = flat[: height * step].reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)
    else:
        expected = height * width * channels
        if flat.size < expected:
            raise ValueError(f"Image buffer too small: got {flat.size}, expected {expected}")
        image = flat[:expected].reshape(height, width, channels)

    if channels == 4:
        image = image[:, :, :3]
    if encoding in {"bgr8", "bgra8"}:
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image).copy()

    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    if image_size is not None:
        size = (image_size, image_size) if isinstance(image_size, int) else image_size
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class OrientationDeltaPredictor:
    """Load an orientation checkpoint and predict corrective RPY in radians."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = ModelEvalConfig.ORIENTATION_CHECKPOINT_PATH,
        device: str = ModelEvalConfig.DEVICE,
        cameras: tuple[str, ...] = ModelEvalConfig.CAMERAS,
        logger=None,
    ) -> None:
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = resolve_device(device)
        self.cameras = tuple(cameras)
        self.image_size: int | tuple[int, int] | None = 224
        self.target_mean = torch.zeros(3, dtype=torch.float32)
        self.target_std = torch.ones(3, dtype=torch.float32)
        self.aggregation = "mean"
        self.num_views = len(self.cameras)
        self.model = None
        self._load()

    def _info(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Orientation checkpoint not found: {self.checkpoint_path}")
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Invalid checkpoint format: {self.checkpoint_path}")

        config = payload.get("config", {})
        state_dict = payload["model_state_dict"]
        image_size = config.get("image_size", 224)
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, (list, tuple)):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = int(image_size)

        self.target_mean = torch.as_tensor(config.get("target_mean", [0.0, 0.0, 0.0]), dtype=torch.float32)
        self.target_std = torch.as_tensor(config.get("target_std", [1.0, 1.0, 1.0]), dtype=torch.float32)
        self.aggregation = str(config.get("aggregation", "mean"))
        self.num_views = int(config.get("num_views", len(config.get("cameras", self.cameras))))

        if self.aggregation == "concat" and len(self.cameras) != self.num_views:
            raise ValueError(
                f"Orientation checkpoint expects {self.num_views} views, runtime cameras={self.cameras}"
            )

        if "head.0.weight" in state_dict:
            hidden_dim = int(state_dict["head.0.weight"].shape[0])
            num_port_heads = int(config.get("num_port_heads", 1))
        else:
            head_indices = {
                int(match.group(1))
                for key in state_dict
                if (match := re.match(r"heads\.(\d+)\.0\.weight$", key)) is not None
            }
            if not head_indices:
                raise ValueError("Checkpoint has no recognizable regression head weights.")
            hidden_dim = int(state_dict[f"heads.{min(head_indices)}.0.weight"].shape[0])
            num_port_heads = int(config.get("num_port_heads", max(head_indices) + 1))

        orientation_model = _load_orientation_model_module()
        self.model = orientation_model.build_resnet_orientation_model(
            backbone_name=config.get("backbone", "resnet50"),
            pretrained=False,
            output_dim=int(self.target_mean.numel()),
            hidden_dim=hidden_dim,
            dropout=0.1,
            aggregation=self.aggregation,
            num_port_heads=num_port_heads,
            num_views=self.num_views,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._info(
            "Orientation model loaded: "
            f"path={self.checkpoint_path}, device={self.device}, "
            f"cameras={self.cameras}, image_size={self.image_size}, "
            f"aggregation={self.aggregation}, num_views={self.num_views}"
        )

    def _camera_image(self, observation, camera: str) -> Optional[Image]:
        return getattr(observation, f"{camera}_image", None)

    @torch.inference_mode()
    def predict_rpy_rad(self, observation, port_id: int | None = None) -> Optional[np.ndarray]:
        if observation is None:
            return None
        images = []
        for camera in self.cameras:
            image_msg = self._camera_image(observation, camera)
            if image_msg is None:
                if self.logger is not None:
                    self.logger.warn(f"Observation missing {camera}_image")
                return None
            images.append(image_msg_to_tensor(image_msg, self.image_size))

        model_input = images[0].unsqueeze(0) if len(images) == 1 else torch.stack(images, dim=0).unsqueeze(0)
        port_tensor = None
        if port_id is not None:
            port_tensor = torch.tensor([int(port_id)], dtype=torch.long, device=self.device)
        pred = self.model(model_input.to(self.device), port_tensor).cpu()[0]
        pred_rpy = pred * self.target_std + self.target_mean
        values = pred_rpy.numpy().astype(np.float64)
        if not np.isfinite(values).all():
            if self.logger is not None:
                self.logger.warn(f"Non-finite orientation prediction: {values}")
            return None
        return values
