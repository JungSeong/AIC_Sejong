from __future__ import annotations

"""Vision-offset regression models used by FinalPolicy ALIGN."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


class TimmFeatureBackbone(nn.Module):
    """timm feature-map backbone with a 1x1 projection head."""

    def __init__(
        self,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = False,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for vision-offset ALIGN models. "
                "Install timm in the active pixi environment."
            ) from _TIMM_IMPORT_ERROR
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.raw_feature_dim = int(self.model.num_features)
        self.feature_dim = int(feature_dim)
        if self.raw_feature_dim == self.feature_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Sequential(
                nn.Conv2d(self.raw_feature_dim, self.feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True),
            )

    def _to_nchw_feature_map(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if features.shape[1] == self.raw_feature_dim:
                return features
            if features.shape[-1] == self.raw_feature_dim:
                return features.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"Unsupported 4D timm feature shape: {tuple(features.shape)}")
        if features.ndim == 3:
            batch, tokens, channels = features.shape
            if channels != self.raw_feature_dim and tokens == self.raw_feature_dim:
                features = features.transpose(1, 2).contiguous()
                batch, tokens, channels = features.shape
            if channels != self.raw_feature_dim:
                raise ValueError(f"Unsupported 3D timm feature shape: {tuple(features.shape)}")
            side = int(tokens**0.5)
            if side * side == tokens:
                return features.transpose(1, 2).reshape(batch, channels, side, side)
            return features.transpose(1, 2).unsqueeze(-1)
        if features.ndim == 2:
            if features.shape[1] != self.raw_feature_dim:
                raise ValueError(f"Unsupported 2D timm feature shape: {tuple(features.shape)}")
            return features[:, :, None, None]
        raise ValueError(f"Unsupported timm feature shape: {tuple(features.shape)}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(image)
        else:
            features = self.model(image)
        return self.project(self._to_nchw_feature_map(features))


def bilinear_pool(feature_map: torch.Tensor) -> torch.Tensor:
    """Compute normalized bilinear outer-product descriptor."""
    batch, channels, height, width = feature_map.shape
    descriptors = feature_map.reshape(batch, channels, height * width)
    pooled = torch.bmm(descriptors, descriptors.transpose(1, 2))
    pooled = pooled / float(height * width)
    pooled = pooled.reshape(batch, channels * channels)
    pooled = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-5)
    return F.normalize(pooled, dim=1)


def bilinear_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Compute normalized bilinear descriptor from [B, tokens, channels]."""
    batch, token_count, channels = tokens.shape
    descriptors = tokens.transpose(1, 2).contiguous()
    pooled = torch.bmm(descriptors, descriptors.transpose(1, 2))
    pooled = pooled / float(token_count)
    pooled = pooled.reshape(batch, channels * channels)
    pooled = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-5)
    return F.normalize(pooled, dim=1)


class CrossAttentionBlock(nn.Module):
    """Residual cross-attention block between one view and the other views."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(query_tokens)
        key_value = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        return query_tokens + attn_out


class MultiViewBilinearCNNRegressor(nn.Module):
    """View-aware bilinear CNN for 6D base_link correction regression."""

    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 6,
        num_views: int = 3,
        share_backbone_weights: bool = True,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.share_backbone_weights = bool(share_backbone_weights)
        if self.share_backbone_weights:
            self.shared_backbone = TimmFeatureBackbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                feature_dim=feature_dim,
            )
            self.view_backbones = None
        else:
            self.shared_backbone = None
            self.view_backbones = nn.ModuleList(
                TimmFeatureBackbone(
                    backbone_name=backbone_name,
                    pretrained=pretrained,
                    feature_dim=feature_dim,
                )
                for _ in range(self.num_views)
            )
        descriptor_dim = self.num_views * feature_dim * feature_dim
        self.head = nn.Sequential(
            nn.Linear(descriptor_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 5:
            raise ValueError("Expected [B, V, 3, H, W] images")
        batch, views, channels, height, width = images.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {views}")
        descriptors = []
        for view_index in range(self.num_views):
            view_image = images[:, view_index, :, :, :]
            if self.share_backbone_weights:
                features = self.shared_backbone(view_image)
            else:
                features = self.view_backbones[view_index](view_image)
            descriptors.append(bilinear_pool(features))
        return self.head(torch.cat(descriptors, dim=1))


class MultiViewBidirectionalCrossAttentionBilinearRegressor(nn.Module):
    """Cross-attended multiview tokens followed by per-view bilinear pooling."""

    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 6,
        num_views: int = 3,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        pos_grid: int = 7,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.num_views = int(num_views)
        self.feature_dim = int(feature_dim)
        self.backbone = TimmFeatureBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, feature_dim, pos_grid, pos_grid) * 0.02
        )
        self.cross_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(feature_dim, num_heads, dropout)
                        for _ in range(self.num_views)
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.token_norms = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(self.num_views)]
        )
        descriptor_dim = self.num_views * feature_dim * feature_dim
        self.head = nn.Sequential(
            nn.Linear(descriptor_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def _to_tokens(self, feature_map: torch.Tensor) -> torch.Tensor:
        pos = F.interpolate(
            self.pos_embed,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (feature_map + pos).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 5:
            raise ValueError("Expected [B, V, 3, H, W] images")
        batch, views, channels, height, width = images.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {views}")
        flat = images.reshape(batch * views, channels, height, width)
        features = self.backbone(flat)
        _, feature_channels, feature_h, feature_w = features.shape
        features = features.reshape(
            batch,
            views,
            feature_channels,
            feature_h,
            feature_w,
        )
        tokens = [self._to_tokens(features[:, view_index]) for view_index in range(views)]

        for layer in self.cross_layers:
            previous_tokens = tokens
            next_tokens = []
            for view_index, block in enumerate(layer):
                other_tokens = torch.cat(
                    [
                        previous_tokens[other_index]
                        for other_index in range(views)
                        if other_index != view_index
                    ],
                    dim=1,
                )
                next_tokens.append(block(previous_tokens[view_index], other_tokens))
            tokens = next_tokens

        descriptors = [
            bilinear_pool_tokens(self.token_norms[view_index](view_tokens))
            for view_index, view_tokens in enumerate(tokens)
        ]
        return self.head(torch.cat(descriptors, dim=1))


def build_vision_offset_model(
    name: str,
    *,
    feature_dim: int = 128,
    backbone_name: str = "efficientnetv2_rw_s",
    num_views: int = 3,
    share_backbone_weights: bool = True,
    attention_heads: int = 8,
    attention_layers: int = 2,
    attention_dropout: float = 0.1,
    attention_pos_grid: int = 7,
) -> nn.Module:
    """Build a 6D vision-offset model from checkpoint metadata."""
    normalized = str(name or "").strip().lower()
    if normalized in {
        "cross_attention_bilinear",
        "cross-attention-bilinear",
        "bidirectional_cross_attention_bilinear",
        "mv_bca_bilinear",
    }:
        return MultiViewBidirectionalCrossAttentionBilinearRegressor(
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            num_views=num_views,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=attention_dropout,
            pos_grid=attention_pos_grid,
        )
    if normalized in {"multiview_bilinear", "mv_bilinear", "bilinear"}:
        return MultiViewBilinearCNNRegressor(
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            num_views=num_views,
            share_backbone_weights=share_backbone_weights,
        )
    raise ValueError(
        "Unsupported vision-offset model name: "
        f"{name}. Expected cross_attention_bilinear or multiview_bilinear."
    )
