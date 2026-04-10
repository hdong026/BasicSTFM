"""Example of registering a custom training task."""

from __future__ import annotations

from typing import Any, Dict

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register()
class ForecastWithAuxEmbeddingPenalty(Task):
    """Forecasting with a small embedding norm penalty."""

    def __init__(self, penalty_weight: float = 1e-4) -> None:
        self.penalty_weight = penalty_weight

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        outputs = model(batch["x"], graph=batch.get("graph"), mode="both")
        loss_out = losses(outputs["forecast"], batch["y"])
        penalty = outputs["embedding"].pow(2).mean() * self.penalty_weight
        total = loss_out["loss"] + penalty
        logs = dict(loss_out["logs"])
        logs["loss/embedding_penalty"] = penalty.detach()
        logs["loss/total"] = total.detach()
        return {
            "loss": total,
            "logs": logs,
            "pred": outputs["forecast"].detach(),
            "target": batch["y"].detach(),
        }
