from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


Aggregation = Literal["mean", "max"]


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
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'.")
        if not backbone_name.startswith("resnet"):
            raise ValueError(
                "ResNetPositionRegressor expects a torchvision ResNet backbone, "
                f"got {backbone_name!r}."
            )

        self.encoder, feature_dim = _build_backbone(backbone_name, pretrained)
        self.aggregation = aggregation

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, output_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, output_dim), output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            features = self.encoder(image)
            return self.head(features)

        if image.ndim != 5:
            raise ValueError(
                f"Expected image tensor with 4 or 5 dimensions, got shape {tuple(image.shape)}."
            )

        batch_size, num_views, channels, height, width = image.shape
        flat_images = image.reshape(batch_size * num_views, channels, height, width)
        flat_features = self.encoder(flat_images)
        features = flat_features.reshape(batch_size, num_views, -1)

        if self.aggregation == "mean":
            features = features.mean(dim=1)
        else:
            features = features.max(dim=1).values
        return self.head(features)


ImagePositionRegressor = ResNetPositionRegressor


def build_resnet_position_model(**kwargs) -> ResNetPositionRegressor:
    return ResNetPositionRegressor(**kwargs)


def build_position_model(**kwargs) -> ResNetPositionRegressor:
    return build_resnet_position_model(**kwargs)
