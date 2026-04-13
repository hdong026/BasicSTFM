"""Identity-style regularization for matched protocol settings."""

from __future__ import annotations

import torch
from torch import nn

from basicstfm.registry import LOSSES


@LOSSES.register("protocol_identity")
class IdentityLoss(nn.Module):
    """Penalize deviation from a protocol-aligned identity target."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = str(reduction)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = (pred - target).pow(2)
        if mask is not None:
            diff = diff * mask.to(dtype=diff.dtype)
            denom = mask.to(dtype=diff.dtype).sum().clamp_min(1.0)
            return diff.sum() / denom
        if self.reduction == "sum":
            return diff.sum()
        return diff.mean()
