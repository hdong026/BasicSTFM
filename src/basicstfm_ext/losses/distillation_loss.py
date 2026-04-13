"""Regression-style teacher distillation loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.registry import LOSSES


@LOSSES.register("protocol_distillation")
class DistillationLoss(nn.Module):
    """Distill a student forecast toward a teacher forecast."""

    def __init__(self, mode: str = "mse") -> None:
        super().__init__()
        self.mode = str(mode).lower()

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "smooth_l1":
            diff = F.smooth_l1_loss(student, teacher, reduction="none")
        else:
            diff = (student - teacher).pow(2)
        if mask is not None:
            diff = diff * mask.to(dtype=diff.dtype)
            denom = mask.to(dtype=diff.dtype).sum().clamp_min(1.0)
            return diff.sum() / denom
        return diff.mean()
