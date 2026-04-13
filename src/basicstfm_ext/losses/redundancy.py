"""Redundancy-reduction penalties for interface/shared features."""

from __future__ import annotations

import torch
from torch import nn

from basicstfm.registry import LOSSES


@LOSSES.register("feature_redundancy")
class RedundancyPenalty(nn.Module):
    """Barlow-Twins-style off-diagonal covariance penalty."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        features = features - features.mean(dim=0, keepdim=True)
        features = features / features.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
        cov = features.t().matmul(features) / max(features.shape[0], 1)
        diag = torch.diag_embed(torch.diagonal(cov))
        off_diag = cov - diag
        return off_diag.pow(2).mean()
