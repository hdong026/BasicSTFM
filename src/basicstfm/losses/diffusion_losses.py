"""Loss terms for residual event diffusion learning."""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import ensure_4d


def residual_forecast_loss(
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Main supervision for residual branch forecasting."""

    residual_pred = ensure_4d(residual_pred)
    residual_target = ensure_4d(residual_target)
    value = (residual_pred - residual_target).abs()
    if mask is None:
        return value.mean()
    mask = ensure_4d(mask).to(dtype=value.dtype, device=value.device)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def propagation_consistency_loss(
    residual_pred: torch.Tensor,
    propagation_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage residual magnitude to match propagated message energy."""

    residual_pred = ensure_4d(residual_pred)
    if propagation_map.ndim != 4:
        raise ValueError(
            f"propagation_map must be [B, H, N, N], got {tuple(propagation_map.shape)}"
        )
    propagation_energy = propagation_map.abs().sum(dim=-1, keepdim=True)
    residual_energy = residual_pred.abs().mean(dim=-1, keepdim=True)
    propagation_energy = propagation_energy / propagation_energy.mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    residual_energy = residual_energy / residual_energy.mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return F.mse_loss(propagation_energy, residual_energy)


def spillover_reconstruction_loss(
    spillover_gate: torch.Tensor,
    residual_target: torch.Tensor,
) -> torch.Tensor:
    """Match spillover intensity with spatially averaged residual activity."""

    if spillover_gate.ndim != 4:
        raise ValueError(f"spillover_gate must be [B, H, N, 1], got {tuple(spillover_gate.shape)}")
    residual_target = ensure_4d(residual_target)
    global_energy = residual_target.abs().mean(dim=2, keepdim=True)
    global_energy = global_energy.mean(dim=-1, keepdim=True)
    global_energy = global_energy.expand_as(spillover_gate)
    return F.l1_loss(spillover_gate, global_energy)


def propagation_sparsity_loss(
    propagation_map: torch.Tensor,
) -> torch.Tensor:
    """Promote sparse edge-level propagation."""

    if propagation_map.ndim != 4:
        raise ValueError(
            f"propagation_map must be [B, H, N, N], got {tuple(propagation_map.shape)}"
        )
    return propagation_map.abs().mean()


def attenuation_regularization(
    attenuation_gate: torch.Tensor,
    target_decay: float = 0.6,
) -> torch.Tensor:
    """Prevent no-stop or all-stop collapse in diffusion rollout."""

    if attenuation_gate.ndim != 4:
        raise ValueError(
            f"attenuation_gate must be [B, H, N, 1], got {tuple(attenuation_gate.shape)}"
        )
    target = attenuation_gate.new_full(attenuation_gate.shape, float(target_decay))
    return F.mse_loss(attenuation_gate, target)


def event_locality_regularization(
    event_score: torch.Tensor,
    event_locality: torch.Tensor,
) -> torch.Tensor:
    """Discourage globally-active event masks with low locality."""

    if event_score.shape != event_locality.shape:
        raise ValueError(
            f"event_score and event_locality must match, got {tuple(event_score.shape)} "
            f"vs {tuple(event_locality.shape)}"
        )
    return (event_score * (1.0 - event_locality)).mean()
