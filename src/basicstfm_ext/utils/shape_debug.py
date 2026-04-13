"""Small shape-debug helpers for wrapper models."""

from __future__ import annotations

from typing import Iterable

import torch


def ensure_shape(name: str, value: torch.Tensor, ndim: int) -> None:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {tuple(value.shape)}")


def ensure_last_dim(name: str, value: torch.Tensor, allowed: Iterable[int]) -> None:
    allowed = tuple(int(item) for item in allowed)
    if value.shape[-1] not in allowed:
        raise ValueError(
            f"{name} last dimension must be in {allowed}, got {tuple(value.shape)}"
        )
