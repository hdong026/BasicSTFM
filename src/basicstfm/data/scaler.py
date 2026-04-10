"""Scalers for spatio-temporal arrays.

The default tensors in BasicSTFM use one of these shapes:
  - raw arrays: [T, N, C]
  - mini-batches: [B, T, N, C]

Scalers are fitted on the training split only. Built-in tasks apply scaling
after dataloader collation and inverse scaling before loss/metric computation,
following the same high-level convention used by BasicTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class StandardScaler:
    """Channel-wise standardization over time and nodes."""

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1e-6

    def fit(self, array: np.ndarray) -> "StandardScaler":
        self.mean = array.mean(axis=(0, 1), keepdims=True)
        self.std = array.std(axis=(0, 1), keepdims=True)
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, array: Any) -> Any:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        if _is_torch_tensor(array):
            mean = _to_torch_stat(self.mean, array)
            std = _to_torch_stat(self.std, array)
            return (array - mean) / std
        return (array - self.mean) / self.std

    def inverse_transform(self, array: Any) -> Any:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        if _is_torch_tensor(array):
            mean = _to_torch_stat(self.mean, array)
            std = _to_torch_stat(self.std, array)
            return array * std + mean
        return array * self.std + self.mean


class IdentityScaler:
    """No-op scaler with the same API as StandardScaler."""

    def fit(self, array: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, array: Any) -> Any:
        return array

    def inverse_transform(self, array: Any) -> Any:
        return array


def build_scaler(cfg: Optional[dict]) -> object:
    if cfg is None:
        return IdentityScaler()
    scaler_type = str(cfg.get("type", "standard")).lower()
    params = dict(cfg.get("params", {}))
    if scaler_type in {"standard", "zscore", "standard_scaler"}:
        return StandardScaler(**params)
    if scaler_type in {"identity", "none"}:
        return IdentityScaler()
    raise KeyError(f"Unknown scaler type: {scaler_type}")


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _to_torch_stat(stat: np.ndarray, like: Any) -> Any:
    import torch

    tensor = torch.as_tensor(stat, dtype=like.dtype, device=like.device)
    while tensor.ndim < like.ndim:
        tensor = tensor.unsqueeze(0)
    return tensor
