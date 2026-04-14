"""Spectral anti-drift regularization for long-horizon forecasting."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import (
    cosine_spectrum_alignment,
    ensure_4d,
    multi_scale_spectral_distance,
    split_low_high_frequency,
)


def stable_low_frequency_alignment(
    stable_pred: torch.Tensor,
    target: torch.Tensor,
    low_ratio: float = 0.25,
) -> torch.Tensor:
    """Stable branch should preserve low-frequency structure."""

    stable_pred = ensure_4d(stable_pred)
    target = ensure_4d(target)
    stable_low, _ = split_low_high_frequency(stable_pred, low_ratio=low_ratio)
    target_low, _ = split_low_high_frequency(target, low_ratio=low_ratio)
    return F.mse_loss(stable_low, target_low)


def residual_high_frequency_alignment(
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    low_ratio: float = 0.25,
) -> torch.Tensor:
    """Residual branch should retain mid/high-frequency event content."""

    residual_pred = ensure_4d(residual_pred)
    residual_target = ensure_4d(residual_target)
    _, pred_high = split_low_high_frequency(residual_pred, low_ratio=low_ratio)
    _, target_high = split_low_high_frequency(residual_target, low_ratio=low_ratio)
    return F.l1_loss(pred_high, target_high)


def anti_spectral_drift_loss(
    final_pred: torch.Tensor,
    target: torch.Tensor,
    scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Penalize long-horizon spectrum collapse and drift."""

    final_pred = ensure_4d(final_pred)
    target = ensure_4d(target)
    mse_term = multi_scale_spectral_distance(final_pred, target, scales=scales, log_amplitude=True)
    cosine_term = cosine_spectrum_alignment(final_pred, target, log_amplitude=True)
    return mse_term + 0.5 * cosine_term


def anti_oversmoothing_loss(final_pred: torch.Tensor) -> torch.Tensor:
    """Discourage collapse into temporally-flat predictions."""

    final_pred = ensure_4d(final_pred)
    if final_pred.shape[1] < 2:
        return final_pred.new_tensor(0.0)
    temporal_delta = (final_pred[:, 1:] - final_pred[:, :-1]).abs().mean()
    return 1.0 / temporal_delta.clamp_min(1e-4)
