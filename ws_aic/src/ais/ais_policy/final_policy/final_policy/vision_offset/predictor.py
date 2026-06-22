from __future__ import annotations

"""Vision-offset checkpoint loader and ROS observation predictor."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image

from final_policy.config import FinalPolicyConfig
from final_policy.model_store import format_model_log
from final_policy.vision_offset.model import build_vision_offset_model


class VisionOffsetPredictor:
    """Predict 6D base_link correction from left/center/right camera images."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = FinalPolicyConfig.DEVICE,
        logger=None,
    ) -> None:
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.cameras = FinalPolicyConfig.CAMERAS
        self.image_size: int | tuple[int, int] | None = 224
        self.model = None
        self._load()

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def _infer_model_name_from_path(self) -> str:
        for parent in self.checkpoint_path.parents:
            name = parent.name.strip().lower()
            if name:
                return name
        return "cross_attention_bilinear"

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Vision-offset checkpoint not found: {self.checkpoint_path}"
            )
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload: {self.checkpoint_path}")

        config = dict(payload.get("config", {}))
        model_name = (
            payload.get("model_name")
            or config.get("model_name")
            or self._infer_model_name_from_path()
        )
        backbone_name = (
            payload.get("backbone_name")
            or config.get("backbone_name")
            or config.get("backbone")
            or "efficientnetv2_rw_s"
        )
        feature_dim = int(payload.get("feature_dim", config.get("feature_dim", 128)))
        self.image_size = payload.get("image_size", config.get("image_size", 224))
        self.cameras = tuple(config.get("cameras", self.cameras))
        share_backbone_weights = bool(
            payload.get(
                "share_backbone_weights",
                config.get("share_backbone_weights", True),
            )
        )
        attention_heads = int(
            payload.get("attention_heads", config.get("attention_heads", 8))
        )
        attention_layers = int(
            payload.get("attention_layers", config.get("attention_layers", 2))
        )
        attention_dropout = float(
            payload.get("attention_dropout", config.get("attention_dropout", 0.1))
        )
        attention_pos_grid = int(
            payload.get("attention_pos_grid", config.get("attention_pos_grid", 7))
        )
        state_dict = payload.get("model") or payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                f"Vision-offset checkpoint missing model state_dict: {self.checkpoint_path}"
            )

        self.model = build_vision_offset_model(
            model_name,
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            num_views=len(self.cameras),
            share_backbone_weights=share_backbone_weights,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            attention_dropout=attention_dropout,
            attention_pos_grid=attention_pos_grid,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._log(
            "info",
            format_model_log(
                "Vision-offset model loaded: "
                f"path={self.checkpoint_path}, model={model_name}, "
                f"backbone={backbone_name}, feature_dim={feature_dim}, "
                f"image_size={self.image_size}, cameras={self.cameras}"
            ),
        )

    def _camera_image(self, observation, camera: str) -> Optional[Image]:
        return getattr(observation, f"{camera}_image", None)

    def _image_msg_to_tensor(self, image_msg: Image) -> torch.Tensor:
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        flat = np.frombuffer(image_msg.data, dtype=np.uint8)
        step = int(getattr(image_msg, "step", 0))
        if step > 0 and flat.size >= height * step:
            rows = flat[: height * step].reshape(height, step)
            image = rows[:, : width * channels].reshape(height, width, channels)
        else:
            image = flat[: height * width * channels].reshape(height, width, channels)
        if channels == 4:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        if self.image_size is not None:
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    @torch.inference_mode()
    def predict(self, observation) -> Optional[np.ndarray]:
        if observation is None:
            return None
        images = []
        for camera in self.cameras:
            image_msg = self._camera_image(observation, camera)
            if image_msg is None:
                self._log("warn", f"Observation missing {camera}_image")
                return None
            images.append(self._image_msg_to_tensor(image_msg))
        batch = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
        output = self.model(batch)[0].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(output).all():
            self._log("warn", "Non-finite vision-offset prediction")
            return None
        return output
