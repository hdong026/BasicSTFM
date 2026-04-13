"""Dataset-descriptor utilities for wrapper-based transfer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from basicstfm.models.foundation.common import ensure_4d, normalize_adjacency
from basicstfm.utils.checkpoint import torch_load


@dataclass
class DescriptorConfig:
    use_graph_stats: bool = True
    use_spectral_stats: bool = True
    max_time_steps: int = 96
    max_nodes: int = 128
    max_channels: int = 8


class DescriptorCache:
    """Simple in-memory cache with momentum updates."""

    def __init__(self, momentum: float = 0.9, cache_path: Optional[str] = None) -> None:
        self.momentum = float(momentum)
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._cache: Dict[str, Tensor] = {}
        self._load()

    def get(self, key: str) -> Optional[Tensor]:
        value = self._cache.get(str(key))
        return None if value is None else value.clone()

    def update(self, key: str, value: Tensor) -> Tensor:
        key = str(key)
        value = value.detach()
        cached = self._cache.get(key)
        if cached is None:
            self._cache[key] = value.clone()
        else:
            self._cache[key] = self.momentum * cached + (1.0 - self.momentum) * value
        self._persist()
        return self._cache[key].clone()

    def _load(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        payload = torch_load(str(self.cache_path), map_location="cpu")
        if not isinstance(payload, dict):
            return
        for key, value in payload.items():
            if isinstance(value, Tensor):
                self._cache[str(key)] = value.detach().cpu()

    def _persist(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in self._cache.items()}, self.cache_path)


def compute_dataset_descriptor(
    x: Tensor,
    *,
    graph: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[DescriptorConfig] = None,
) -> Tensor:
    """Build a bounded descriptor from unlabeled batch statistics.

    The descriptor intentionally uses only cheap, dataset-level statistics so it
    can initialize a target interface without target labels.
    """

    cfg = config or DescriptorConfig()
    x = ensure_4d(x).detach()
    if mask is not None:
        mask = ensure_4d(mask).detach().bool()
    x = x[:, : cfg.max_time_steps, : cfg.max_nodes, : cfg.max_channels]
    if mask is not None:
        mask = mask[:, : x.shape[1], : x.shape[2], : x.shape[3]]

    observed = _resolve_observed(x, mask)
    missing_ratio = 0.0 if mask is None else float((~mask).float().mean().item())
    lag1 = _average_autocorrelation(observed, lag=1)
    lag2 = _average_autocorrelation(observed, lag=2)
    spectral_ratio, spectral_centroid = (
        _spectral_features(observed) if cfg.use_spectral_stats else (0.0, 0.0)
    )
    mean = float(observed.mean().item()) if observed.numel() else 0.0
    std = float(observed.std(unbiased=False).item()) if observed.numel() else 0.0
    min_value = float(observed.min().item()) if observed.numel() else 0.0
    max_value = float(observed.max().item()) if observed.numel() else 0.0
    num_nodes = float(x.shape[2])
    num_channels = float(x.shape[3])
    sampling_interval = float(_read_numeric(metadata, "sampling_interval", default=0.0))

    if cfg.use_graph_stats:
        density, avg_degree, spectral_radius = _graph_features(graph, x.shape[2], x.device, x.dtype)
    else:
        density, avg_degree, spectral_radius = 0.0, 0.0, 0.0

    descriptor = x.new_tensor(
        [
            num_nodes,
            num_channels,
            sampling_interval,
            mean,
            std,
            min_value,
            max_value,
            missing_ratio,
            lag1,
            lag2,
            spectral_ratio,
            spectral_centroid,
            density,
            avg_degree,
            spectral_radius,
        ]
    )
    return descriptor


def _resolve_observed(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x.reshape(-1)
    observed = x.masked_select(mask)
    return observed if observed.numel() else x.reshape(-1)


def _average_autocorrelation(values: Tensor, lag: int) -> float:
    if values.numel() <= lag + 1:
        return 0.0
    series = values.reshape(-1)
    left = series[:-lag]
    right = series[lag:]
    left = left - left.mean()
    right = right - right.mean()
    denom = left.norm() * right.norm()
    if float(denom.item()) <= 1e-6:
        return 0.0
    return float((left @ right / denom).item())


def _spectral_features(values: Tensor) -> tuple[float, float]:
    if values.numel() < 8:
        return 0.0, 0.0
    flat = values.reshape(-1)
    flat = flat[: max(8, min(flat.numel(), 4096))]
    spectrum = torch.fft.rfft(flat.float())
    power = spectrum.abs().pow(2)
    if power.numel() <= 1:
        return 0.0, 0.0
    power = power[1:]
    total = power.sum().clamp_min(1e-6)
    cutoff = max(1, power.numel() // 4)
    low_ratio = float((power[:cutoff].sum() / total).item())
    freqs = torch.linspace(0.0, 1.0, power.numel(), device=power.device)
    centroid = float((freqs * power).sum().div(total).item())
    return low_ratio, centroid


def _graph_features(
    graph: Optional[Tensor],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    if graph is None:
        return 0.0, 0.0, 0.0
    adj = normalize_adjacency(graph, num_nodes, device, dtype, add_self_loops=False)
    density = float((adj > 0).float().mean().item())
    avg_degree = float(adj.sum(dim=-1).mean().item())
    spectral_radius = _power_iteration_spectral_radius(adj)
    return density, avg_degree, spectral_radius


def _power_iteration_spectral_radius(matrix: Tensor, num_iters: int = 8) -> float:
    if matrix.numel() == 0:
        return 0.0
    vector = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    vector = vector / vector.norm().clamp_min(1e-6)
    for _ in range(num_iters):
        vector = matrix @ vector
        vector = vector / vector.norm().clamp_min(1e-6)
    estimate = vector @ (matrix @ vector)
    return float(estimate.abs().item())


def _read_numeric(metadata: Optional[Mapping[str, Any]], key: str, default: float = 0.0) -> float:
    if metadata is None:
        return default
    value = metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
