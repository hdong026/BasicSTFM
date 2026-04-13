"""Forecasting task that passes dataset context and auxiliary losses to wrapper models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register("InterfaceForecastingTask")
class InterfaceForecastingTask(Task):
    """Wrapper-aware forecasting task with optional auxiliary regularization."""

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        output_key: str = "forecast",
        model_mode: str = "forecast",
        mask_key: Optional[str] = None,
        dataset_name: Optional[str] = None,
        include_aux_in_eval: bool = False,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.mask_key = mask_key
        self.dataset_name = dataset_name
        self.include_aux_in_eval = bool(include_aux_in_eval)

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        x = self.transform(batch[self.input_key], batch=batch)
        target = batch[self.target_key]

        dataset_context = {
            "dataset_name": batch.get("dataset_name", self.dataset_name),
            "dataset_index": batch.get("dataset_index"),
            "x_mask": batch.get("x_mask"),
            "y_mask": batch.get("y_mask"),
            "metadata": {
                "num_nodes": x.shape[2],
                "num_channels": x.shape[3],
                "input_len": x.shape[1],
                "target_len": target.shape[1],
            },
        }
        outputs = model(
            x,
            graph=batch.get("graph"),
            mask=None,
            mode=self.model_mode,
            dataset_context=dataset_context,
        )
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred = self.inverse_transform(pred_scaled, batch=batch)
        mask = self.merge_masks(
            batch.get("y_mask"),
            batch.get(self.mask_key) if self.mask_key else None,
        )
        target, mask = self.align_prediction_target(pred, target, mask)
        loss_out = losses(pred, target, mask=mask)
        total = loss_out["loss"]
        logs = dict(loss_out["logs"])

        if isinstance(outputs, dict) and outputs.get("aux_losses") and (model.training or self.include_aux_in_eval):
            for name, value in outputs["aux_losses"].items():
                if value is None:
                    continue
                total = total + value
                logs[f"aux/{name}"] = value.detach()

        if isinstance(outputs, dict) and "prototype_weights" in outputs:
            weights = outputs["prototype_weights"]
            for index, value in enumerate(weights):
                logs[f"proto/alpha_{index}"] = value.detach()

        logs["loss/total"] = total.detach()
        return {
            "loss": total,
            "logs": logs,
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": None if mask is None else mask.detach(),
        }
