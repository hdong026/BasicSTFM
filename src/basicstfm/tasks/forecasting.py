"""Forecasting task flow."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from basicstfm.data.revin import factost_value_revin_inverse, factost_value_revin_normalize
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
        primary_supervision_space: str = "denormalized",
        basicts_scale_logging: bool = False,
        use_revin: bool = False,
        revin_value_channel: int = 0,
        revin_eps: float = 1e-5,
        revin_scaled_std_floor: float = 0.05,
        revin_loss_space: str = "normalized",
        factost_original_scale: bool = True,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.mask_key = mask_key
        pss = str(primary_supervision_space).lower().replace("-", "_")
        if pss not in {"denormalized", "normalized"}:
            raise ValueError("primary_supervision_space must be 'denormalized' or 'normalized'")
        self.primary_supervision_space = pss
        self.basicts_scale_logging = bool(basicts_scale_logging)
        self.use_revin = bool(use_revin)
        self.revin_value_channel = int(revin_value_channel)
        self.revin_eps = float(revin_eps)
        self.revin_scaled_std_floor = float(revin_scaled_std_floor)
        rls = str(revin_loss_space).lower().replace("-", "_")
        if rls not in {"normalized", "raw"}:
            raise ValueError("revin_loss_space must be 'normalized' or 'raw'")
        self.revin_loss_space = rls
        self.factost_original_scale = bool(factost_original_scale)

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        raw_target = batch[self.target_key]

        if self.use_revin:
            x_scaled = self.transform(raw_x, batch=batch)
            y_scaled = self.transform(raw_target, batch=batch)
            x_scaled, y_scaled = factost_value_revin_normalize(
                x_scaled,
                y_scaled,
                batch,
                value_channels=(self.revin_value_channel,),
                eps=self.revin_eps,
                scaled_std_floor=self.revin_scaled_std_floor,
            )
            outputs = model(x_scaled, graph=batch.get("graph"), mode=self.model_mode)
            pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
            pred_scaled = self.slice_prediction_to_data_channels(pred_scaled, x_scaled)
            mask = self.merge_masks(
                batch.get("y_mask"),
                batch.get(self.mask_key) if self.mask_key else None,
            )
            pred_scaled, y_scaled, mask = self.align_prediction_target(pred_scaled, y_scaled, mask)

            pred_after_revin = factost_value_revin_inverse(pred_scaled, batch)
            pred = self.inverse_transform(pred_after_revin, batch=batch)
            target_denorm = raw_target
            pred, target_denorm, mask = self.align_prediction_target(pred, target_denorm, mask)

            if self.revin_loss_space == "normalized":
                loss_out = losses(pred_scaled, y_scaled, mask=mask)
            else:
                loss_out = losses(pred, target_denorm, mask=mask)

            logs = dict(loss_out["logs"])
            if self.factost_original_scale:
                with torch.no_grad():

                    def _mm(t: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
                        if m is None:
                            return t.mean()
                        mf = m.to(dtype=t.dtype)
                        return (t * mf).sum() / mf.sum().clamp_min(1.0)

                    err = (pred - target_denorm).abs()
                    sq = (pred - target_denorm).pow(2)
                    logs["metric/mae_original"] = _mm(err, mask)
                    logs["metric/rmse_original"] = torch.sqrt(_mm(sq, mask).clamp_min(1e-12))
            return {
                "loss": loss_out["loss"],
                "logs": logs,
                "pred": pred.detach(),
                "target": target_denorm.detach(),
                "mask": None if mask is None else mask.detach(),
            }

        x = self.transform(raw_x, batch=batch)
        target = raw_target
        outputs = model(x, graph=batch.get("graph"), mode=self.model_mode)
        pred_scaled = outputs[self.output_key] if isinstance(outputs, dict) else outputs
        pred_scaled = self.slice_prediction_to_data_channels(pred_scaled, x)
        pred = self.inverse_transform(pred_scaled, batch=batch)
        mask = self.merge_masks(
            batch.get("y_mask"),
            batch.get(self.mask_key) if self.mask_key else None,
        )
        target, mask = self.align_prediction_target(pred, target, mask)
        loss_out = losses(pred, target, mask=mask)
        return {
            "loss": loss_out["loss"],
            "logs": loss_out["logs"],
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": None if mask is None else mask.detach(),
        }
