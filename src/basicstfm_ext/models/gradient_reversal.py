"""Gradient-reversal helpers for domain-adversarial training."""

from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


class GradientReversal(nn.Module):
    """Identity in the forward pass, sign-flipped gradient in the backward pass."""

    def __init__(self, coeff: float = 1.0) -> None:
        super().__init__()
        self.coeff = float(coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.coeff)
