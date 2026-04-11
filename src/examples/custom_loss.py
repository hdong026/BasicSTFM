"""Example of registering a custom loss."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicstfm.registry import LOSSES


@LOSSES.register("smooth_l1_mae_mix")
class SmoothL1MAEMix(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = self.alpha * self.smooth_l1(pred, target) + (1.0 - self.alpha) * (pred - target).abs()
        if mask is None:
            return value.mean()
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=value.dtype, device=value.device).expand_as(value)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)
