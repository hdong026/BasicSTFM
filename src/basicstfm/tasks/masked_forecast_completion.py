"""Masked future completion for prompt-style spatio-temporal transfer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device
from basicstfm.tasks.masking import temporal_tail_mask


@TASKS.register()
class MaskedForecastCompletionTask(Task):
    """Predict future slices by masking the target suffix of a full sequence.

    This task is designed for prompt-based stage-2 transfer regimes such as
    UniST, where the model sees the observed history, receives masked future
    slots, and learns to complete the missing suffix.
    """

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        output_key: str = "reconstruction",
        mask_value: float = 0.0,
        model_mode: str = "reconstruct",
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.mask_value = float(mask_value)
        self.model_mode = model_mode

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        history = batch[self.input_key]
        future = batch[self.target_key]
        full_sequence = torch.cat([history, future], dim=1)
        scaled_sequence = self.transform(full_sequence)

        mask = temporal_tail_mask(scaled_sequence, future_steps=future.shape[1])
        masked_sequence = scaled_sequence.masked_fill(mask, self.mask_value)
        outputs = model(
            masked_sequence,
            graph=batch.get("graph"),
            mask=mask,
            mode=self.model_mode,
        )
        pred_full = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred_full = self.inverse_transform(pred_full)
        pred = pred_full[:, -future.shape[1] :]
        future_mask = mask[:, -future.shape[1] :]
        loss_out = losses(pred, future, mask=future_mask)
        return {
            "loss": loss_out["loss"],
            "logs": loss_out["logs"],
            "pred": pred.detach(),
            "target": future.detach(),
            "mask": future_mask.detach(),
        }
