"""Example of registering a project-specific model."""

from __future__ import annotations

import torch
from torch import nn

from basicstfm.registry import MODELS


@MODELS.register()
class ResidualMLPForecaster(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_len * output_dim),
        )

    def forward(self, x: torch.Tensor, graph=None, mode: str = "forecast"):
        del graph, mode
        batch, _, nodes, _ = x.shape
        nodewise = x.permute(0, 2, 1, 3).reshape(batch * nodes, -1)
        out = self.net(nodewise)
        out = out.reshape(batch, nodes, self.output_len, self.output_dim)
        return {"forecast": out.permute(0, 2, 1, 3).contiguous()}
