"""Factories for PyTorch optimizers and schedulers."""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def build_optimizer(cfg: Optional[dict], params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    cfg = dict(cfg or {})
    optim_type = str(cfg.pop("type", "AdamW"))
    optim_params = dict(cfg.pop("params", {}))
    optim_params.update(cfg)
    optim_cls = getattr(torch.optim, optim_type, None)
    if optim_cls is None:
        raise KeyError(f"Unknown torch optimizer: {optim_type}")
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters are available for the optimizer")
    return optim_cls(trainable, **optim_params)


def build_scheduler(
    cfg: Optional[dict],
    optimizer: torch.optim.Optimizer,
):
    if not cfg:
        return None
    cfg = dict(cfg)
    scheduler_type = str(cfg.pop("type"))
    params = dict(cfg.pop("params", {}))
    params.update(cfg)
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type, None)
    if scheduler_cls is None:
        raise KeyError(f"Unknown torch scheduler: {scheduler_type}")
    return scheduler_cls(optimizer, **params)
