"""Protocol-aware forecasting task with optional length curriculum."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register("ProtocolForecastingTask")
class ProtocolForecastingTask(Task):
    """Forecasting task that passes protocol metadata and auxiliary losses."""

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        output_key: str = "forecast",
        model_mode: str = "forecast",
        mask_key: Optional[str] = None,
        dataset_name: Optional[str] = None,
        include_aux_in_eval: bool = False,
        input_length_choices: Optional[Sequence[int]] = None,
        output_length_choices: Optional[Sequence[int]] = None,
        curriculum_in_eval: bool = False,
        curriculum_seed: int = 42,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.mask_key = mask_key
        self.dataset_name = dataset_name
        self.include_aux_in_eval = bool(include_aux_in_eval)
        self.input_length_choices = _normalize_lengths(input_length_choices)
        self.output_length_choices = _normalize_lengths(output_length_choices)
        self.curriculum_in_eval = bool(curriculum_in_eval)
        self.rng = random.Random(int(curriculum_seed))

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        x = self.transform(batch[self.input_key], batch=batch)
        target = batch[self.target_key]
        x_mask = batch.get("x_mask")
        y_mask = batch.get("y_mask")

        input_len = self._pick_length(
            self.input_length_choices,
            available=x.shape[1],
            training=bool(model.training),
        )
        target_len = self._pick_length(
            self.output_length_choices,
            available=target.shape[1],
            training=bool(model.training),
        )

        x = x[:, -input_len:]
        target = target[:, :target_len]
        if x_mask is not None:
            x_mask = x_mask[:, -input_len:]
        if y_mask is not None:
            y_mask = y_mask[:, :target_len]

        dataset_context = {
            "dataset_name": batch.get("dataset_name", self.dataset_name),
            "dataset_index": batch.get("dataset_index"),
            "x_mask": x_mask,
            "y_mask": y_mask,
            "metadata": {
                "num_nodes": x.shape[2],
                "num_channels": x.shape[3],
                "input_len": input_len,
                "target_len": target_len,
                "sampling_interval": _read_sampling_interval(batch),
            },
        }
        outputs = model(
            x,
            graph=batch.get("graph"),
            mask=x_mask,
            mode=self.model_mode,
            dataset_context=dataset_context,
        )
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred = self.inverse_transform(pred_scaled, batch=batch)
        mask = self.merge_masks(
            y_mask,
            batch.get(self.mask_key) if self.mask_key else None,
        )
        target, mask = self.align_prediction_target(pred, target, mask)
        loss_out = losses(pred, target, mask=mask)
        total = loss_out["loss"]
        logs = dict(loss_out["logs"])
        logs["protocol/input_len"] = float(input_len)
        logs["protocol/output_len"] = float(target_len)

        if isinstance(outputs, dict) and outputs.get("aux_losses") and (model.training or self.include_aux_in_eval):
            for name, value in outputs["aux_losses"].items():
                if value is None:
                    continue
                total = total + value
                logs[f"aux/{name}"] = value.detach()

        logs["loss/total"] = total.detach()
        return {
            "loss": total,
            "logs": logs,
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": None if mask is None else mask.detach(),
        }

    def _pick_length(
        self,
        candidates: Sequence[int],
        *,
        available: int,
        training: bool,
    ) -> int:
        if not candidates:
            return int(available)
        valid = [int(item) for item in candidates if int(item) <= int(available)]
        if not valid:
            return int(available)
        if training or self.curriculum_in_eval:
            return int(self.rng.choice(valid))
        return int(max(valid))


def _normalize_lengths(values: Optional[Sequence[int]]) -> list[int]:
    if not values:
        return []
    normalized = sorted({int(item) for item in values if int(item) > 0})
    return normalized


def _read_sampling_interval(batch: Dict[str, Any]) -> float:
    value = batch.get("sampling_interval")
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
