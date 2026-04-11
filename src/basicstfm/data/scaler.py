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
from typing import Any, Optional, Sequence

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


class DatasetAwareScaler:
    """Per-dataset scaler wrapper for multi-dataset batches.

    The wrapper keeps one scaler per dataset and dispatches transform /
    inverse_transform by reading ``batch["dataset_index"]``. It currently
    supports the built-in channel-wise StandardScaler and IdentityScaler.
    """

    def __init__(
        self,
        names: Sequence[str],
        means: Optional[np.ndarray],
        stds: Optional[np.ndarray],
        identity: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> None:
        self.names = list(names)
        self.means = means
        self.stds = stds
        self.identity = np.asarray(identity, dtype=np.bool_) if identity is not None else None
        self.eps = float(eps)

    @classmethod
    def from_scalers(
        cls,
        names: Sequence[str],
        scalers: Sequence[object],
        max_channels: int,
    ) -> "DatasetAwareScaler":
        means = []
        stds = []
        identity = []
        eps = 1e-6
        for scaler in scalers:
            if isinstance(scaler, StandardScaler):
                mean = scaler.mean
                std = scaler.std
                if mean is None or std is None:
                    raise RuntimeError("StandardScaler must be fitted before DatasetAwareScaler")
                padded_mean = np.zeros((1, 1, max_channels), dtype=np.float32)
                padded_std = np.ones((1, 1, max_channels), dtype=np.float32)
                padded_mean[..., : mean.shape[-1]] = mean.astype(np.float32)
                padded_std[..., : std.shape[-1]] = np.maximum(std.astype(np.float32), scaler.eps)
                means.append(padded_mean)
                stds.append(padded_std)
                identity.append(False)
                eps = max(eps, float(scaler.eps))
            elif isinstance(scaler, IdentityScaler):
                means.append(np.zeros((1, 1, max_channels), dtype=np.float32))
                stds.append(np.ones((1, 1, max_channels), dtype=np.float32))
                identity.append(True)
            else:
                raise TypeError(
                    "DatasetAwareScaler currently supports only StandardScaler and IdentityScaler"
                )
        return cls(
            names=names,
            means=np.stack(means, axis=0),
            stds=np.stack(stds, axis=0),
            identity=np.asarray(identity, dtype=np.bool_),
            eps=eps,
        )

    def transform(self, array: Any, batch: Optional[dict] = None) -> Any:
        return self._apply(array, batch=batch, inverse=False)

    def inverse_transform(self, array: Any, batch: Optional[dict] = None) -> Any:
        return self._apply(array, batch=batch, inverse=True)

    def _apply(self, array: Any, batch: Optional[dict], inverse: bool) -> Any:
        if self.means is None or self.stds is None:
            return array
        if batch is None:
            if len(self.names) != 1:
                raise ValueError("DatasetAwareScaler requires batch context for multi-dataset transforms")
            dataset_index = [0]
        else:
            dataset_index = batch.get("dataset_index")
            if dataset_index is None:
                raise KeyError("Batch is missing dataset_index required by DatasetAwareScaler")
        if _is_torch_tensor(array):
            return self._apply_torch(array, dataset_index, inverse=inverse)
        return self._apply_numpy(array, dataset_index, inverse=inverse)

    def _apply_numpy(self, array: Any, dataset_index: Any, inverse: bool) -> Any:
        indices = np.asarray(dataset_index, dtype=np.int64).reshape(-1)
        out = np.array(array, copy=True)
        for idx in np.unique(indices):
            sample_mask = indices == idx
            mean = self.means[idx]
            std = np.maximum(self.stds[idx], self.eps)
            if inverse:
                out[sample_mask] = out[sample_mask] * std + mean
            else:
                out[sample_mask] = (out[sample_mask] - mean) / std
        return out

    def _apply_torch(self, array: Any, dataset_index: Any, inverse: bool) -> Any:
        import torch

        if not torch.is_tensor(dataset_index):
            dataset_index = torch.as_tensor(dataset_index, dtype=torch.long, device=array.device)
        dataset_index = dataset_index.to(device=array.device, dtype=torch.long).reshape(-1)
        means = torch.as_tensor(self.means, dtype=array.dtype, device=array.device)
        stds = torch.as_tensor(self.stds, dtype=array.dtype, device=array.device).clamp_min(self.eps)
        while means.ndim < array.ndim:
            means = means.unsqueeze(1)
            stds = stds.unsqueeze(1)
        batch_mean = means.index_select(0, dataset_index)
        batch_std = stds.index_select(0, dataset_index)
        if inverse:
            return array * batch_std + batch_mean
        return (array - batch_mean) / batch_std


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
