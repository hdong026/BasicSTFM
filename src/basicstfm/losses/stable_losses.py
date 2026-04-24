"""Loss terms for stable trunk discovery."""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import ensure_4d, split_low_high_frequency


def masked_reduction(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reduce value with an optional broadcastable mask."""

    if mask is None:
        return value.mean()
    mask = ensure_4d(mask)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=value.dtype, device=value.device).expand_as(value)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def stable_forecast_loss(
    stable_pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Direct regression loss for stable branch forecasts."""

    stable_pred = ensure_4d(stable_pred)
    target = ensure_4d(target)
    return masked_reduction((stable_pred - target).abs(), mask=mask)


def stable_forecast_mae_per_batch_item(
    stable_pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-batch-item mean absolute error in [B] for group-wise (graph/domain) risk splitting."""

    stable_pred = ensure_4d(stable_pred)
    target = ensure_4d(target)
    err = (stable_pred - target).abs()
    if mask is None:
        return err.mean(dim=(1, 2, 3))
    m = ensure_4d(mask)
    while m.ndim < err.ndim:
        m = m.unsqueeze(-1)
    m = m.to(dtype=err.dtype, device=err.device).expand_as(err)
    denom = m.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return (err * m).sum(dim=(1, 2, 3)) / denom


def trend_consistency_loss(
    stable_pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align first-order temporal trend between stable forecast and target."""

    stable_pred = ensure_4d(stable_pred)
    target = ensure_4d(target)
    if stable_pred.shape[1] < 2:
        return stable_pred.new_tensor(0.0)
    pred_delta = stable_pred[:, 1:] - stable_pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    trend_mask = None
    if mask is not None:
        trend_mask = ensure_4d(mask)
        trend_mask = trend_mask[:, 1:] & trend_mask[:, :-1]
    return masked_reduction((pred_delta - target_delta).abs(), mask=trend_mask)


def low_frequency_consistency_loss(
    stable_pred: torch.Tensor,
    target: torch.Tensor,
    low_ratio: float = 0.3,
) -> torch.Tensor:
    """Keep stable branch aligned with low-frequency dynamics."""

    pred_low, _ = split_low_high_frequency(stable_pred, low_ratio=low_ratio)
    target_low, _ = split_low_high_frequency(target, low_ratio=low_ratio)
    return F.mse_loss(pred_low, target_low)


def temporal_smoothness_loss(
    stable_component: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize high total variation for the stable branch."""

    stable_component = ensure_4d(stable_component)
    if stable_component.shape[1] < 2:
        return stable_component.new_tensor(0.0)
    diff = (stable_component[:, 1:] - stable_component[:, :-1]).abs()
    smooth_mask = None
    if mask is not None:
        smooth_mask = ensure_4d(mask)
        smooth_mask = smooth_mask[:, 1:] & smooth_mask[:, :-1]
    return masked_reduction(diff, mask=smooth_mask)
