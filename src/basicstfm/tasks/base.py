"""Base helpers for trainable task flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


class Task(ABC):
    """A task owns the per-batch training logic for a stage."""

    @abstractmethod
    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        raise NotImplementedError
