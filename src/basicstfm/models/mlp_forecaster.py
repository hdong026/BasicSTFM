"""A compact baseline forecaster."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicstfm.registry import MODELS


@MODELS.register()
class MLPForecaster(nn.Module):
    """Node-wise MLP forecaster for [B, T, N, C] inputs."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.output_dim = output_dim
        layers = []
        in_dim = input_len * input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_len * output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mode: str = "forecast",
        **_: object,
    ):
        del graph, mode
        batch, _, nodes, _ = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {nodes}")
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, -1)
        forecast = self.net(x)
        forecast = forecast.reshape(batch, nodes, self.output_len, self.output_dim)
        forecast = forecast.permute(0, 2, 1, 3).contiguous()
        return {"forecast": forecast}
