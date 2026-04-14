"""Frequency-domain helpers for stable-residual spatio-temporal modeling."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


EPS = 1e-6


def ensure_4d(value: torch.Tensor) -> torch.Tensor:
    """Ensure tensors are ``[B, T, N, C]``."""

    if value.ndim == 3:
        return value.unsqueeze(-1)
    if value.ndim != 4:
        raise ValueError(f"Expected [B, T, N, C] or [B, T, N], got {tuple(value.shape)}")
    return value


def temporal_amplitude(
    value: torch.Tensor,
    *,
    log_amplitude: bool = True,
) -> torch.Tensor:
    """Return rFFT amplitudes over the temporal axis."""

    value = ensure_4d(value)
    amplitude = torch.fft.rfft(value, dim=1).abs()
    if log_amplitude:
        amplitude = torch.log1p(amplitude)
    return amplitude


def split_low_high_frequency(
    value: torch.Tensor,
    low_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split temporal spectrum into low/high components."""

    value = ensure_4d(value)
    if not 0.0 < float(low_ratio) < 1.0:
        raise ValueError("low_ratio must be in (0, 1)")

    fft = torch.fft.rfft(value, dim=1)
    freq_bins = fft.shape[1]
    low_bins = max(1, int(round(freq_bins * float(low_ratio))))

    low_fft = torch.zeros_like(fft)
    low_fft[:, :low_bins] = fft[:, :low_bins]
    high_fft = fft - low_fft

    low = torch.fft.irfft(low_fft, n=value.shape[1], dim=1)
    high = torch.fft.irfft(high_fft, n=value.shape[1], dim=1)
    return low, high


def downsample_time(value: torch.Tensor, scale: int) -> torch.Tensor:
    """Average-pool the temporal axis by ``scale``."""

    value = ensure_4d(value)
    if scale <= 1:
        return value

    batch, steps, nodes, channels = value.shape
    if steps < scale:
        return value

    usable = (steps // scale) * scale
    value = value[:, :usable]
    value = value.permute(0, 2, 3, 1).reshape(batch * nodes * channels, 1, usable)
    value = F.avg_pool1d(value, kernel_size=scale, stride=scale)
    reduced_steps = value.shape[-1]
    value = value.reshape(batch, nodes, channels, reduced_steps).permute(0, 3, 1, 2).contiguous()
    return value


def multi_scale_spectral_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scales: Sequence[int] | Iterable[int] = (1, 2, 4),
    log_amplitude: bool = True,
) -> torch.Tensor:
    """Compute mean squared spectral distance over pooled scales."""

    pred = ensure_4d(pred)
    target = ensure_4d(target)
    if pred.shape != target.shape:
        raise ValueError(f"Expected equal shapes, got {tuple(pred.shape)} vs {tuple(target.shape)}")

    losses = []
    for scale in scales:
        pooled_pred = downsample_time(pred, int(scale))
        pooled_target = downsample_time(target, int(scale))
        pred_amp = temporal_amplitude(pooled_pred, log_amplitude=log_amplitude)
        target_amp = temporal_amplitude(pooled_target, log_amplitude=log_amplitude)
        losses.append((pred_amp - target_amp).pow(2).mean())
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def cosine_spectrum_alignment(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    log_amplitude: bool = True,
) -> torch.Tensor:
    """1 - cosine similarity between flattened temporal spectra."""

    pred_amp = temporal_amplitude(pred, log_amplitude=log_amplitude).reshape(pred.shape[0], -1)
    target_amp = temporal_amplitude(target, log_amplitude=log_amplitude).reshape(target.shape[0], -1)
    pred_norm = pred_amp / pred_amp.norm(dim=-1, keepdim=True).clamp_min(EPS)
    target_norm = target_amp / target_amp.norm(dim=-1, keepdim=True).clamp_min(EPS)
    cos = (pred_norm * target_norm).sum(dim=-1)
    return (1.0 - cos).mean()
