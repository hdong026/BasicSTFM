"""Residual-signal helpers used by stable-residual training tasks."""

from __future__ import annotations

import torch

from basicstfm.utils.spectral_ops import ensure_4d


def residual_signal(observed: torch.Tensor, stable_component: torch.Tensor) -> torch.Tensor:
    """Compute residual ``R = observed - stable_component``."""

    observed = ensure_4d(observed)
    stable_component = ensure_4d(stable_component)
    if observed.shape != stable_component.shape:
        raise ValueError(
            f"Observed and stable tensors must match, got {tuple(observed.shape)} "
            f"vs {tuple(stable_component.shape)}"
        )
    return observed - stable_component


def residual_energy(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-node residual energy as ``sqrt(mean(r^2))`` over time/channels."""

    value = ensure_4d(value)
    energy = value.pow(2).mean(dim=(1, 3), keepdim=True).sqrt().clamp_min(eps)
    return energy


def masked_mean_abs(
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean absolute value with optional broadcastable mask."""

    value = ensure_4d(value).abs()
    if mask is None:
        return value.mean()
    mask = ensure_4d(mask).to(dtype=value.dtype, device=value.device)
    if mask.shape != value.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match value shape {tuple(value.shape)}")
    denom = mask.sum().clamp_min(eps)
    return (value * mask).sum() / denom


def normalize_residual(
    value: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize residuals by per-sample energy for stable training scales."""

    value = ensure_4d(value)
    denom = value.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(eps)
    return value / denom
