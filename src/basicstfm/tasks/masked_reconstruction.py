"""Masked spatio-temporal reconstruction task flow."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.masking import sample_spatiotemporal_mask
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
        mask_strategy: str = "random",
        mask_strategies: Optional[Sequence[str]] = None,
        model_mode: str = "reconstruct",
    ) -> None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        self.input_key = input_key
        self.output_key = output_key
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_strategy = mask_strategy
        self.mask_strategies = mask_strategies
        self.model_mode = model_mode

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        x = self.transform(raw_x, batch=batch)
        sampled_mask = sample_spatiotemporal_mask(
            x,
            self.mask_ratio,
            strategy=self.mask_strategy,
            strategies=self.mask_strategies,
        )
        valid_mask = batch.get("x_mask")
        mask = self.merge_masks(sampled_mask, valid_mask)
        masked_x = x.masked_fill(mask, self.mask_value)
        outputs = model(masked_x, graph=batch.get("graph"), mask=mask, mode=self.model_mode)
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred_scaled = self.slice_prediction_to_data_channels(pred_scaled, x)
        pred = self.inverse_transform(pred_scaled, batch=batch)
        target, mask = self.align_prediction_target(pred, raw_x, mask)
        loss_out = losses(pred, target, mask=mask)
        return {
            "loss": loss_out["loss"],
            "logs": loss_out["logs"],
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": mask.detach(),
        }
