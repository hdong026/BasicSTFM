"""Reusable masking and spectral helpers for STFM tasks."""

from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Sequence

import torch
from torch.nn import functional as F


def sample_spatiotemporal_mask(
    x: torch.Tensor,
    mask_ratio: float,
    strategy: str = "random",
    strategies: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Sample a broadcastable mask for tensors shaped ``[B, T, N, C]``."""

    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if x.ndim != 4:
        raise ValueError(f"Expected [B, T, N, C], got {tuple(x.shape)}")

    if strategy == "mixed":
        candidates = list(strategies or ("random", "temporal", "tube", "block"))
        if not candidates:
            raise ValueError("mask_strategies must be non-empty when mask_strategy='mixed'")
        strategy = random.choice(candidates)

    batch, steps, nodes, channels = x.shape
    mask = torch.zeros((batch, steps, nodes, 1), device=x.device, dtype=torch.bool)

    if strategy == "random":
        token_mask = torch.rand((batch, steps, nodes, 1), device=x.device) < mask_ratio
        mask = token_mask
    elif strategy == "temporal":
        masked_steps = max(1, int(math.ceil(steps * mask_ratio)))
        mask[:, steps - masked_steps :, :, :] = True
    elif strategy == "tube":
        masked_nodes = max(1, int(math.ceil(nodes * mask_ratio)))
        for batch_index in range(batch):
            node_index = torch.randperm(nodes, device=x.device)[:masked_nodes]
            mask[batch_index, :, node_index, :] = True
    elif strategy == "block":
        block_steps = max(1, int(round(steps * math.sqrt(mask_ratio))))
        block_nodes = max(1, int(round(nodes * math.sqrt(mask_ratio))))
        for batch_index in range(batch):
            start_t = random.randint(0, max(steps - block_steps, 0))
            start_n = random.randint(0, max(nodes - block_nodes, 0))
            mask[
                batch_index,
                start_t : start_t + block_steps,
                start_n : start_n + block_nodes,
                :,
            ] = True
    else:
        raise ValueError(f"Unsupported mask strategy: {strategy!r}")

    return mask.expand(batch, steps, nodes, channels)


def temporal_tail_mask(
    x: torch.Tensor,
    future_steps: int,
) -> torch.Tensor:
    """Mask the suffix of length ``future_steps`` on ``[B, T, N, C]`` tensors."""

    if x.ndim != 4:
        raise ValueError(f"Expected [B, T, N, C], got {tuple(x.shape)}")
    if future_steps <= 0 or future_steps >= x.shape[1]:
        raise ValueError("future_steps must be in (0, total_steps)")
    mask = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 1), device=x.device, dtype=torch.bool)
    mask[:, -future_steps:, :, :] = True
    return mask.expand_as(x)


def multi_scale_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scales: Optional[Iterable[int]] = None,
    log_amplitude: bool = True,
) -> torch.Tensor:
    """Compare temporal spectra across one or more pooled resolutions."""

    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    scales = list(scales or (1,))
    losses = []
    for scale in scales:
        pooled_pred = _downsample_time(pred, int(scale))
        pooled_target = _downsample_time(target, int(scale))
        pred_fft = torch.fft.rfft(pooled_pred, dim=1).abs()
        target_fft = torch.fft.rfft(pooled_target, dim=1).abs()
        if log_amplitude:
            pred_fft = torch.log1p(pred_fft)
            target_fft = torch.log1p(target_fft)
        losses.append(F.mse_loss(pred_fft, target_fft))
    return torch.stack(losses).mean()


def _downsample_time(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return x
    batch, steps, nodes, channels = x.shape
    if steps < scale:
        return x
    usable = (steps // scale) * scale
    x = x[:, :usable]
    x = x.permute(0, 2, 3, 1).reshape(batch * nodes * channels, 1, usable)
    x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
    reduced = x.shape[-1]
    x = x.reshape(batch, nodes, channels, reduced).permute(0, 3, 1, 2).contiguous()
    return x


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(-1)
    if x.ndim != 4:
        raise ValueError(f"Expected [B, T, N, C] or [B, T, N], got {tuple(x.shape)}")
    return x
