"""Joint pretraining tasks used by stage recipes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device
from basicstfm.tasks.masking import multi_scale_spectral_loss, sample_spatiotemporal_mask


@TASKS.register()
class JointReconstructionForecastTask(Task):
    """Optimize reconstruction and forecasting in one stage.

    This task is useful for foundation-model recipes such as FactoST's UTP,
    where the backbone is trained with complementary self-supervised and
    predictive objectives before downstream spatio-temporal adaptation.
    """

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        forecast_key: str = "forecast",
        reconstruction_key: str = "reconstruction",
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
        mask_strategy: str = "random",
        mask_strategies: Optional[Sequence[str]] = None,
        reconstruction_weight: float = 1.0,
        forecast_weight: float = 1.0,
        reconstruction_spectral_weight: float = 0.0,
        forecast_spectral_weight: float = 0.0,
        spectral_scales: Optional[Sequence[int]] = None,
        spectral_log_amplitude: bool = True,
        forecast_mask_key: Optional[str] = None,
        model_mode: str = "both",
    ) -> None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        self.input_key = input_key
        self.target_key = target_key
        self.forecast_key = forecast_key
        self.reconstruction_key = reconstruction_key
        self.mask_ratio = float(mask_ratio)
        self.mask_value = float(mask_value)
        self.mask_strategy = mask_strategy
        self.mask_strategies = list(mask_strategies) if mask_strategies is not None else None
        self.reconstruction_weight = float(reconstruction_weight)
        self.forecast_weight = float(forecast_weight)
        self.reconstruction_spectral_weight = float(reconstruction_spectral_weight)
        self.forecast_spectral_weight = float(forecast_spectral_weight)
        self.spectral_scales = list(spectral_scales) if spectral_scales is not None else [1]
        self.spectral_log_amplitude = bool(spectral_log_amplitude)
        self.forecast_mask_key = forecast_mask_key
        self.model_mode = model_mode

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        x = self.transform(raw_x, batch=batch)
        target = batch[self.target_key]

        sampled_mask = sample_spatiotemporal_mask(
            x,
            self.mask_ratio,
            strategy=self.mask_strategy,
            strategies=self.mask_strategies,
        )
        recon_mask = self.merge_masks(sampled_mask, batch.get("x_mask"))
        masked_x = x.masked_fill(recon_mask, self.mask_value)

        outputs = model(masked_x, graph=batch.get("graph"), mask=recon_mask, mode=self.model_mode)
        if not isinstance(outputs, dict):
            raise TypeError("JointReconstructionForecastTask expects model outputs to be a dict")

        recon = self.inverse_transform(outputs[self.reconstruction_key], batch=batch)
        forecast = self.inverse_transform(outputs[self.forecast_key], batch=batch)
        forecast_mask = self.merge_masks(
            batch.get("y_mask"),
            batch.get(self.forecast_mask_key) if self.forecast_mask_key else None,
        )
        raw_x, recon_mask = self.align_prediction_target(recon, raw_x, recon_mask)
        target, forecast_mask = self.align_prediction_target(forecast, target, forecast_mask)

        recon_loss_out = losses(recon, raw_x, mask=recon_mask)
        forecast_loss_out = losses(forecast, target, mask=forecast_mask)
        total = (
            self.reconstruction_weight * recon_loss_out["loss"]
            + self.forecast_weight * forecast_loss_out["loss"]
        )

        if self.reconstruction_spectral_weight > 0.0:
            recon_spectral = multi_scale_spectral_loss(
                recon,
                raw_x,
                scales=self.spectral_scales,
                log_amplitude=self.spectral_log_amplitude,
            )
            total = total + self.reconstruction_spectral_weight * recon_spectral
        else:
            recon_spectral = None

        if self.forecast_spectral_weight > 0.0:
            forecast_spectral = multi_scale_spectral_loss(
                forecast,
                target,
                scales=self.spectral_scales,
                log_amplitude=self.spectral_log_amplitude,
            )
            total = total + self.forecast_spectral_weight * forecast_spectral
        else:
            forecast_spectral = None

        logs: Dict[str, torch.Tensor] = {}
        for key, value in recon_loss_out["logs"].items():
            logs[f"reconstruction/{key}"] = value.detach()
        for key, value in forecast_loss_out["logs"].items():
            logs[f"forecast/{key}"] = value.detach()
        if recon_spectral is not None:
            logs["reconstruction/loss/spectral"] = recon_spectral.detach()
        if forecast_spectral is not None:
            logs["forecast/loss/spectral"] = forecast_spectral.detach()
        logs["loss/total"] = total.detach()

        return {
            "loss": total,
            "logs": logs,
            "pred": forecast.detach(),
            "target": target.detach(),
            "mask": None if forecast_mask is None else forecast_mask.detach(),
        }
