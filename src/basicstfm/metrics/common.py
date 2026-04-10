"""Common metrics used by forecasting tasks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from basicstfm.registry import METRICS


def _masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=value.dtype, device=value.device).expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


@METRICS.register("mae")
class MAEMetric(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _masked_mean((pred - target).abs(), mask)


@METRICS.register("rmse")
class RMSEMetric(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.sqrt(_masked_mean((pred - target).pow(2), mask).clamp_min(1e-12))


@METRICS.register("mape")
class MAPEMetric(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        denom = target.abs().clamp_min(self.eps)
        return _masked_mean((pred - target).abs() / denom, mask)


class MetricCollection:
    """Build and evaluate metrics from config dicts."""

    def __init__(self, metric_cfgs: Optional[Iterable[dict]] = None) -> None:
        self.items: List[tuple[str, nn.Module]] = []
        for index, cfg in enumerate(metric_cfgs or []):
            cfg = dict(cfg)
            name = str(cfg.pop("name", cfg.get("type", f"metric_{index}")))
            metric = METRICS.build(cfg)
            if not isinstance(metric, nn.Module):
                raise TypeError(f"Metric {name!r} must be a torch.nn.Module")
            self.items.append((name, metric))

    def to(self, device: torch.device) -> "MetricCollection":
        for _, metric in self.items:
            metric.to(device)
        return self

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return {
            f"metric/{name}": metric(pred, target, mask=mask).detach()
            for name, metric in self.items
        }
