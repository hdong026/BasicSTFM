"""Forecasting task flow."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register()
class ForecastingTask(Task):
    """Predict future windows from context windows."""

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        output_key: str = "forecast",
        model_mode: str = "forecast",
        mask_key: Optional[str] = None,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.mask_key = mask_key

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        x = self.transform(batch[self.input_key])
        target = batch[self.target_key]
        outputs = model(x, graph=batch.get("graph"), mode=self.model_mode)
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred = self.inverse_transform(pred_scaled)
        mask = batch.get(self.mask_key) if self.mask_key else None
        loss_out = losses(pred, target, mask=mask)
        return {
            "loss": loss_out["loss"],
            "logs": loss_out["logs"],
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": None if mask is None else mask.detach(),
        }
