"""Scalers for spatio-temporal arrays shaped as [time, nodes, channels]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (array - self.mean) / self.std

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        return array * self.std + self.mean


class IdentityScaler:
    """No-op scaler with the same API as StandardScaler."""

    def fit(self, array: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
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
