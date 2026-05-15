from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


Aggregation = Literal["mean", "max", "concat"]


def _resolve_weights(backbone_name: str, pretrained: bool | str | None):
    if pretrained is None or pretrained is False:
        return None
    if isinstance(pretrained, str) and pretrained.lower() not in {"true", "default"}:
        weights_enum = models.get_model_weights(backbone_name)
        return weights_enum[pretrained]
    weights_enum = models.get_model_weights(backbone_name)
    return weights_enum.DEFAULT


def _build_backbone(backbone_name: str, pretrained: bool | str | None) -> tuple[nn.Module, int]:
    weights = _resolve_weights(backbone_name, pretrained)
    backbone = models.get_model(backbone_name, weights=weights)

    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if hasattr(backbone, "classifier"):
        classifier = backbone.classifier
        feature_dim = None
        if isinstance(classifier, nn.Linear):
            feature_dim = classifier.in_features
        elif isinstance(classifier, nn.Sequential):
            for module in reversed(classifier):
                if isinstance(module, nn.Linear):
                    feature_dim = module.in_features
                    break
        if feature_dim is not None:
            backbone.classifier = nn.Identity()
            return backbone, feature_dim

    if hasattr(backbone, "heads"):
        heads = backbone.heads
        feature_dim = None
        if isinstance(heads, nn.Linear):
            feature_dim = heads.in_features
        elif isinstance(heads, nn.Sequential):
            for module in reversed(heads):
                if isinstance(module, nn.Linear):
                    feature_dim = module.in_features
                    break
        if feature_dim is not None:
            backbone.heads = nn.Identity()
            return backbone, feature_dim

    raise ValueError(
        f"Could not infer classifier head for torchvision backbone {backbone_name!r}."
    )


def _make_regression_head(feature_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, max(hidden_dim // 2, output_dim)),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(max(hidden_dim // 2, output_dim), output_dim),
    )


class ResNetPositionRegressor(nn.Module):
    """ResNet encoder with a regression head for 3D offset prediction.

    Input can be ``[B, C, H, W]`` for a single view or ``[B, V, C, H, W]``
    for multi-view training. Multi-view features are aggregated before the
    MLP head.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "resnet50",
        output_dim: int = 3,
        pretrained: bool | str | None = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: Aggregation = "mean",
        freeze_backbone: bool = False,
        num_port_heads: int = 2,
        num_views: int = 1,
        rpy_dim: int = 0,
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "max", "concat"}:
            raise ValueError("aggregation must be 'mean', 'max', or 'concat'.")
        if num_views < 1:
            raise ValueError("num_views must be >= 1.")
        if not backbone_name.startswith("resnet"):
            raise ValueError(
                "ResNetPositionRegressor expects a torchvision ResNet backbone, "
                f"got {backbone_name!r}."
            )

        self.encoder, feature_dim = _build_backbone(backbone_name, pretrained)
        self.aggregation = aggregation
        self.num_views = int(num_views)
        self.rpy_dim = int(rpy_dim)
        if self.rpy_dim < 0:
            raise ValueError("rpy_dim must be >= 0.")
        head_feature_dim = feature_dim * self.num_views if aggregation == "concat" else feature_dim
        head_feature_dim += self.rpy_dim

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        if num_port_heads < 1:
            raise ValueError("num_port_heads must be >= 1.")
        self.num_port_heads = int(num_port_heads)
        if self.num_port_heads == 1:
            self.head = _make_regression_head(head_feature_dim, hidden_dim, output_dim, dropout)
        else:
            self.heads = nn.ModuleList(
                _make_regression_head(head_feature_dim, hidden_dim, output_dim, dropout)
                for _ in range(self.num_port_heads)
            )

    def _append_rpy(self, features: torch.Tensor, rpy: torch.Tensor | None) -> torch.Tensor:
        if self.rpy_dim == 0:
            return features
        if rpy is None:
            rpy = features.new_zeros((features.shape[0], self.rpy_dim))
        else:
            rpy = rpy.to(device=features.device, dtype=features.dtype).view(features.shape[0], -1)
            if rpy.shape[1] != self.rpy_dim:
                raise ValueError(f"Expected {self.rpy_dim} RPY values, got {rpy.shape[1]}.")
        return torch.cat([features, rpy], dim=1)

    def _predict_from_features(
        self,
        features: torch.Tensor,
        port_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.num_port_heads == 1:
            return self.head(features)

        if port_id is None:
            port_id = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
        port_id = port_id.to(device=features.device, dtype=torch.long).view(-1)
        if port_id.shape[0] != features.shape[0]:
            raise ValueError(
                f"Expected {features.shape[0]} port ids, got {port_id.shape[0]}."
            )
        port_id = port_id.clamp(0, self.num_port_heads - 1)
        all_outputs = torch.stack([head(features) for head in self.heads], dim=1)
        return all_outputs[torch.arange(features.shape[0], device=features.device), port_id]

    def forward(
        self,
        image: torch.Tensor,
        port_id: torch.Tensor | None = None,
        rpy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim == 4:
            if self.aggregation == "concat" and self.num_views != 1:
                raise ValueError(
                    "Concat aggregation expects a multi-view image tensor with shape "
                    f"[B, {self.num_views}, C, H, W]."
                )
            features = self.encoder(image)
            features = self._append_rpy(features, rpy)
            return self._predict_from_features(features, port_id)

        if image.ndim != 5:
            raise ValueError(
                f"Expected image tensor with 4 or 5 dimensions, got shape {tuple(image.shape)}."
            )

        batch_size, num_views, channels, height, width = image.shape
        flat_images = image.reshape(batch_size * num_views, channels, height, width)
        flat_features = self.encoder(flat_images)
        features = flat_features.reshape(batch_size, num_views, -1)

        if self.aggregation == "concat":
            if num_views != self.num_views:
                raise ValueError(f"Expected {self.num_views} views, got {num_views}.")
            features = features.reshape(batch_size, num_views * features.shape[-1])
        elif self.aggregation == "mean":
            features = features.mean(dim=1)
        else:
            features = features.max(dim=1).values
        features = self._append_rpy(features, rpy)
        return self._predict_from_features(features, port_id)


ImagePositionRegressor = ResNetPositionRegressor


def build_resnet_position_model(**kwargs) -> ResNetPositionRegressor:
    return ResNetPositionRegressor(**kwargs)


def build_position_model(**kwargs) -> ResNetPositionRegressor:
    return build_resnet_position_model(**kwargs)
