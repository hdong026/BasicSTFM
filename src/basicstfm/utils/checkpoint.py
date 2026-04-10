"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "extra": ckpt.get("extra", {}),
    }
