"""Checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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
        "rng_state": collect_rng_state(),
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
    restore_rng: bool = False,
) -> Dict[str, Any]:
    ckpt = torch_load(path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "extra": ckpt.get("extra", {}),
    }


def read_checkpoint_metadata(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Read checkpoint metadata without requiring model construction."""

    ckpt = torch_load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        return dict(ckpt.get("extra", {}))
    return {}


def collect_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_to_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_to_byte_tensor(item) for item in state["torch_cuda"]])


def _to_byte_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


def torch_load(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
