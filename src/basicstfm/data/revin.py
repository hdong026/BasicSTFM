"""FactoST-style instance normalization / RevIN over the value channels only."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import torch


def _normalize_value_channels_tuple(value_channels: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(value_channels, int):
        return (int(value_channels),)
    return tuple(int(c) for c in value_channels)


def factost_value_revin_normalize(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: Dict[str, Any],
    *,
    value_channels: Union[int, Sequence[int]] = 0,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize only selected channels using mean/std along time *of the input window x* only.

    Shapes:
        x: [B, T, N, C]
        y: [B, H, N, C]
    Stored in ``batch`` (overwritten each step):
        revin_mean, revin_std: [B, 1, N, C_v]
        revin_value_channels: tuple[int, ...]

    Statistics are computed per batch item, per node, per value channel.
    """

    vc = _normalize_value_channels_tuple(value_channels)
    for c in vc:
        if c < 0 or c >= x.shape[-1]:
            raise IndexError(f"value channel index {c} invalid for input with C={x.shape[-1]}")
        if c >= y.shape[-1]:
            raise IndexError(f"value channel index {c} invalid for target with C={y.shape[-1]}")

    slices_m = tuple(x[..., c] for c in vc)
    xv = torch.stack(slices_m, dim=-1)
    mean = xv.mean(dim=1, keepdim=True)
    std = xv.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(eps))

    batch["revin_mean"] = mean
    batch["revin_std"] = std
    batch["revin_value_channels"] = vc

    x_out = x.clone()
    y_out = y.clone()
    for i, c in enumerate(vc):
        m = mean[..., i : i + 1]
        s = std[..., i : i + 1]
        x_out[..., c : c + 1] = (x[..., c : c + 1] - m) / s
        y_out[..., c : c + 1] = (y[..., c : c + 1] - m) / s
    return x_out, y_out


def factost_value_revin_inverse(tensor: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
    """Invert RevIN on the same value channels recorded in ``batch``; leave other channels untouched."""
    if "revin_mean" not in batch or "revin_std" not in batch:
        return tensor
    mean = batch["revin_mean"]
    std = batch["revin_std"]
    vc_tuple = batch.get("revin_value_channels")
    if not isinstance(vc_tuple, (tuple, list)):
        return tensor
    vc = tuple(int(c) for c in vc_tuple)
    out = tensor.clone()
    c_pred = tensor.shape[-1]
    for i, c in enumerate(vc):
        if c >= c_pred:
            continue
        m = mean[..., i : i + 1]
        s = std[..., i : i + 1]
        out[..., c : c + 1] = tensor[..., c : c + 1] * s + m
    return out
