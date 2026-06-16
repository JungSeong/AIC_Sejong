"""이미지, force/torque, joint 상태를 함께 쓰는 pose prediction 모델 정의."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


Aggregation = Literal["mean", "max", "concat"]


def _resolve_weights(backbone_name: str, pretrained: bool | str | None):
    """torchvision backbone의 pretrained weights 설정을 이름 또는 기본값으로 해석한다."""
    if pretrained is None or pretrained is False:
        return None
    if isinstance(pretrained, str) and pretrained.lower() not in {"true", "default"}:
        weights_enum = models.get_model_weights(backbone_name)
        return weights_enum[pretrained]
    return models.get_model_weights(backbone_name).DEFAULT


def _build_backbone(
    backbone_name: str,
    pretrained: bool | str | None,
) -> tuple[nn.Module, int]:
    """torchvision backbone을 만들고 마지막 FC를 제거해 feature encoder로 사용한다."""
    weights = _resolve_weights(backbone_name, pretrained)
    backbone = models.get_model(backbone_name, weights=weights)
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone for pose prediction: {backbone_name!r}")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """회귀 head에 쓰는 작은 MLP 블록을 생성한다."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, max(hidden_dim // 2, output_dim)),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(max(hidden_dim // 2, output_dim), output_dim),
    )


class MultimodalPoseRegressor(nn.Module):
    """멀티뷰 이미지와 로봇 상태를 fusion해 포트 offset과 yaw 오차를 예측한다."""

    def __init__(
        self,
        *,
        backbone_name: str = "resnet50",
        pretrained: bool | str | None = True,
        hidden_dim: int = 256,
        force_torque_dim: int = 6,
        force_torque_hidden_dim: int = 64,
        force_torque_feature_dim: int = 64,
        joint_dim: int = 12,
        joint_hidden_dim: int = 64,
        joint_feature_dim: int = 64,
        dropout: float = 0.1,
        aggregation: Aggregation = "concat",
        freeze_backbone: bool = False,
        num_views: int = 3,
    ) -> None:
        """backbone, force/torque encoder, joint encoder, 회귀 head들을 구성한다."""
        super().__init__()
        if aggregation not in {"mean", "max", "concat"}:
            raise ValueError("aggregation must be 'mean', 'max', or 'concat'.")
        if num_views < 1:
            raise ValueError("num_views must be >= 1.")

        self.encoder, feature_dim = _build_backbone(backbone_name, pretrained)
        self.aggregation = aggregation
        self.num_views = int(num_views)
        image_feature_dim = (
            feature_dim * self.num_views if aggregation == "concat" else feature_dim
        )

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.force_torque_encoder = nn.Sequential(
            nn.Linear(force_torque_dim, force_torque_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(force_torque_hidden_dim, force_torque_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, joint_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_dim, joint_feature_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = image_feature_dim + force_torque_feature_dim + joint_feature_dim
        self.port0_position_head = _mlp(fused_dim, hidden_dim, 2, dropout)
        self.port1_position_head = _mlp(fused_dim, hidden_dim, 2, dropout)
        self.dyaw_head = _mlp(fused_dim, hidden_dim, 1, dropout)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """단일/멀티뷰 이미지를 backbone feature로 인코딩하고 view별 feature를 합친다."""
        if image.ndim == 4:
            if self.aggregation == "concat" and self.num_views != 1:
                raise ValueError(
                    "Concat aggregation expects [B, V, C, H, W] when num_views > 1."
                )
            return self.encoder(image)
        if image.ndim != 5:
            raise ValueError(
                f"Expected image tensor with 4 or 5 dims, got {tuple(image.shape)}"
            )

        batch_size, num_views, channels, height, width = image.shape
        flat_images = image.reshape(batch_size * num_views, channels, height, width)
        flat_features = self.encoder(flat_images)
        features = flat_features.reshape(batch_size, num_views, -1)
        if self.aggregation == "concat":
            if num_views != self.num_views:
                raise ValueError(f"Expected {self.num_views} views, got {num_views}.")
            return features.reshape(batch_size, num_views * features.shape[-1])
        if self.aggregation == "mean":
            return features.mean(dim=1)
        return features.max(dim=1).values

    def forward(
        self,
        image: torch.Tensor,
        force_torque: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """이미지/힘/관절 feature를 결합해 port0, port1 offset과 dyaw를 예측한다."""
        image_features = self._encode_image(image)
        force_torque_features = self.force_torque_encoder(force_torque)
        joint_features = self.joint_encoder(joint_positions)
        features = torch.cat([image_features, force_torque_features, joint_features], dim=1)
        return {
            "port0_position": self.port0_position_head(features),
            "port1_position": self.port1_position_head(features),
            "dyaw": self.dyaw_head(features).squeeze(1),
        }


def build_pose_model(**kwargs) -> MultimodalPoseRegressor:
    """checkpoint config에서 받은 인자로 pose prediction 모델을 생성한다."""
    return MultimodalPoseRegressor(**kwargs)
