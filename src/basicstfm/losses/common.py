"""Common losses and weighted loss composition."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from basicstfm.registry import LOSSES


def _apply_mask(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=value.dtype, device=value.device).expand_as(value)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


@LOSSES.register("mae")
class MAELoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _apply_mask((pred - target).abs(), mask)


@LOSSES.register("mse")
class MSELoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _apply_mask((pred - target).pow(2), mask)


@LOSSES.register("huber")
class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (pred - target).abs()
        quadratic = torch.minimum(diff, torch.tensor(self.delta, device=diff.device, dtype=diff.dtype))
        linear = diff - quadratic
        value = 0.5 * quadratic.pow(2) + self.delta * linear
        return _apply_mask(value, mask)


class LossCollection:
    """Build and combine a list of weighted losses.

    Config example:
        [
          {"type": "mae", "weight": 1.0, "name": "forecast_mae"},
          {"type": "mse", "weight": 0.1}
        ]
    """

    def __init__(self, loss_cfgs: Optional[Iterable[dict]] = None) -> None:
        self.items: List[tuple[str, float, nn.Module]] = []
        for index, cfg in enumerate(loss_cfgs or [{"type": "mae"}]):
            cfg = dict(cfg)
            weight = float(cfg.pop("weight", 1.0))
            name = str(cfg.pop("name", cfg.get("type", f"loss_{index}")))
            loss = LOSSES.build(cfg)
            if not isinstance(loss, nn.Module):
                raise TypeError(f"Loss {name!r} must be a torch.nn.Module")
            self.items.append((name, weight, loss))

    def to(self, device: torch.device) -> "LossCollection":
        for _, _, loss in self.items:
            loss.to(device)
        return self

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        total = pred.new_tensor(0.0)
        logs: Dict[str, torch.Tensor] = {}
        for name, weight, loss in self.items:
            value = loss(pred, target, mask=mask)
            total = total + value * weight
            logs[f"loss/{name}"] = value.detach()
        logs["loss/total"] = total.detach()
        return {"loss": total, "logs": logs}
