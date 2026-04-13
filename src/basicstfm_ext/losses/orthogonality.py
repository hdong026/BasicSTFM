"""Orthogonality penalties for shared/private feature separation."""

from __future__ import annotations

import torch
from torch import nn

from basicstfm.registry import LOSSES


@LOSSES.register("feature_orthogonality")
class OrthogonalityLoss(nn.Module):
    """Penalize correlation between shared and private features."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
        if shared.ndim > 2:
            shared = shared.reshape(shared.shape[0], -1)
        if private.ndim > 2:
            private = private.reshape(private.shape[0], -1)
        if shared.shape[0] != private.shape[0]:
            raise ValueError("shared and private features must share batch dimension")
        dim = min(shared.shape[-1], private.shape[-1])
        shared = shared[..., :dim]
        private = private[..., :dim]

        shared = shared - shared.mean(dim=0, keepdim=True)
        private = private - private.mean(dim=0, keepdim=True)
        shared = shared / shared.std(dim=0, keepdim=True).clamp_min(self.eps)
        private = private / private.std(dim=0, keepdim=True).clamp_min(self.eps)
        cross_cov = shared.t().matmul(private) / max(shared.shape[0], 1)
        return cross_cov.pow(2).mean()
