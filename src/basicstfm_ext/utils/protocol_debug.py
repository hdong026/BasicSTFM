"""Shape assertions and optional logging for protocol adapters."""

from __future__ import annotations

import logging

import torch


def ensure_protocol_shape(name: str, value: torch.Tensor, ndim: int) -> None:
    if value.ndim != int(ndim):
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(value.shape)}")


def debug_tensor_shape(
    logger: logging.Logger,
    name: str,
    value: torch.Tensor,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    logger.debug("%s shape=%s dtype=%s", name, tuple(value.shape), value.dtype)
