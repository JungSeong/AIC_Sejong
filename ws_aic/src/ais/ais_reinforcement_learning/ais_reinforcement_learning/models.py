from __future__ import annotations

import torch
from torch import nn


class ActionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
