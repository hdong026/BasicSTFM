"""Fusion head that combines stable and residual forecasts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FusionPredictor(nn.Module):
    """Support additive, gate-based, and confidence-based fusion."""

    def __init__(
        self,
        output_dim: int,
        fusion_mode: str = "additive",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"additive", "gate", "confidence"}:
            raise ValueError("fusion_mode must be one of: additive, gate, confidence")
        self.output_dim = int(output_dim)
        self.fusion_mode = fusion_mode
        self.gate_net = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.stable_confidence = nn.Linear(self.output_dim, self.output_dim)
        self.residual_confidence = nn.Linear(self.output_dim, self.output_dim)

    def forward(
        self,
        stable_forecast: torch.Tensor,
        residual_forecast: torch.Tensor,
        mode: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        if stable_forecast.shape != residual_forecast.shape:
            raise ValueError(
                "stable_forecast and residual_forecast must match: "
                f"{tuple(stable_forecast.shape)} vs {tuple(residual_forecast.shape)}"
            )
        fusion_mode = mode or self.fusion_mode

        if fusion_mode == "additive":
            forecast = stable_forecast + residual_forecast
            fusion_weight = torch.ones_like(stable_forecast[..., :1])
        elif fusion_mode == "gate":
            gate = torch.sigmoid(self.gate_net(torch.cat([stable_forecast, residual_forecast], dim=-1)))
            forecast = stable_forecast + gate * residual_forecast
            fusion_weight = gate.mean(dim=-1, keepdim=True)
        elif fusion_mode == "confidence":
            stable_conf = torch.softmax(self.stable_confidence(stable_forecast), dim=-1)
            residual_conf = torch.softmax(self.residual_confidence(residual_forecast), dim=-1)
            total_conf = (stable_conf + residual_conf).clamp_min(1e-6)
            residual_weight = residual_conf / total_conf
            forecast = stable_forecast + residual_weight * residual_forecast
            fusion_weight = residual_weight.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        return {
            "forecast": forecast,
            "fusion_weight": fusion_weight,
            "fusion_mode": forecast.new_tensor(0.0),
        }
