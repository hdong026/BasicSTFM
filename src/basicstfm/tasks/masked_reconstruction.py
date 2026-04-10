"""Masked spatio-temporal reconstruction task flow."""

from __future__ import annotations

from typing import Any, Dict

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register()
class MaskedReconstructionTask(Task):
    """Randomly mask observations and reconstruct the original signal."""

    def __init__(
        self,
        input_key: str = "x",
        output_key: str = "reconstruction",
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
        model_mode: str = "reconstruct",
    ) -> None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        self.input_key = input_key
        self.output_key = output_key
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.model_mode = model_mode

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        x = self.transform(raw_x)
        mask = torch.rand_like(x[..., :1]) < self.mask_ratio
        mask = mask.expand_as(x)
        masked_x = x.masked_fill(mask, self.mask_value)
        outputs = model(masked_x, graph=batch.get("graph"), mask=mask, mode=self.model_mode)
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred = self.inverse_transform(pred_scaled)
        loss_out = losses(pred, raw_x, mask=mask)
        return {
            "loss": loss_out["loss"],
            "logs": loss_out["logs"],
            "pred": pred.detach(),
            "target": raw_x.detach(),
            "mask": mask.detach(),
        }
