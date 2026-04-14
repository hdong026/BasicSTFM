"""Stage task for stable-trunk + residual-diffusion forecasting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from basicstfm.losses.diffusion_losses import (
    attenuation_regularization,
    event_locality_regularization,
    propagation_consistency_loss,
    propagation_sparsity_loss,
    residual_forecast_loss,
    spillover_reconstruction_loss,
)
from basicstfm.losses.disentangle_losses import (
    cross_covariance_penalty,
    energy_allocation_regularizer,
    mutual_exclusion_regularizer,
    orthogonality_loss,
)
from basicstfm.losses.stable_losses import (
    low_frequency_consistency_loss,
    stable_forecast_loss,
    temporal_smoothness_loss,
    trend_consistency_loss,
)
from basicstfm.models.spectral_regularizer import (
    anti_oversmoothing_loss,
    anti_spectral_drift_loss,
    residual_high_frequency_alignment,
    stable_low_frequency_alignment,
)
from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register("StableResidualForecastingTask")
class StableResidualForecastingTask(Task):
    """Task that enforces stable-first, residual-diffusion-second learning."""

    def __init__(
        self,
        input_key: str = "x",
        target_key: str = "y",
        output_key: str = "forecast",
        model_mode: str = "forecast",
        phase: str = "joint",
        mask_key: Optional[str] = None,
        final_weight: float = 1.0,
        stable_forecast_weight: float = 0.5,
        stable_reconstruction_weight: float = 0.2,
        trend_weight: float = 0.1,
        low_freq_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        residual_forecast_weight: float = 0.6,
        propagation_consistency_weight: float = 0.1,
        spillover_weight: float = 0.05,
        propagation_sparsity_weight: float = 0.01,
        attenuation_weight: float = 0.01,
        locality_weight: float = 0.05,
        spectral_drift_weight: float = 0.15,
        anti_oversmoothing_weight: float = 0.01,
        orthogonality_weight: float = 0.05,
        cross_cov_weight: float = 0.05,
        energy_allocation_weight: float = 0.02,
        mutual_exclusion_weight: float = 0.02,
        low_ratio: float = 0.25,
        spectral_scales: Optional[Sequence[int]] = None,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.phase = phase
        self.mask_key = mask_key

        self.final_weight = float(final_weight)
        self.stable_forecast_weight = float(stable_forecast_weight)
        self.stable_reconstruction_weight = float(stable_reconstruction_weight)
        self.trend_weight = float(trend_weight)
        self.low_freq_weight = float(low_freq_weight)
        self.smoothness_weight = float(smoothness_weight)

        self.residual_forecast_weight = float(residual_forecast_weight)
        self.propagation_consistency_weight = float(propagation_consistency_weight)
        self.spillover_weight = float(spillover_weight)
        self.propagation_sparsity_weight = float(propagation_sparsity_weight)
        self.attenuation_weight = float(attenuation_weight)
        self.locality_weight = float(locality_weight)

        self.spectral_drift_weight = float(spectral_drift_weight)
        self.anti_oversmoothing_weight = float(anti_oversmoothing_weight)

        self.orthogonality_weight = float(orthogonality_weight)
        self.cross_cov_weight = float(cross_cov_weight)
        self.energy_allocation_weight = float(energy_allocation_weight)
        self.mutual_exclusion_weight = float(mutual_exclusion_weight)

        self.low_ratio = float(low_ratio)
        self.spectral_scales = tuple(int(item) for item in (spectral_scales or (1, 2, 4)))

    def _phase_factor(self, category: str) -> float:
        table = {
            "stable": {
                "final": 0.4,
                "stable": 1.0,
                "diffusion": 0.0,
                "spectral": 0.6,
                "disentangle": 0.2,
            },
            "diffusion": {
                "final": 0.6,
                "stable": 0.1,
                "diffusion": 1.0,
                "spectral": 0.8,
                "disentangle": 0.6,
            },
            "stable_only": {
                "final": 1.0,
                "stable": 1.0,
                "diffusion": 0.0,
                "spectral": 0.6,
                "disentangle": 0.0,
            },
            "joint": {
                "final": 1.0,
                "stable": 1.0,
                "diffusion": 1.0,
                "spectral": 1.0,
                "disentangle": 1.0,
            },
            "scratch": {
                "final": 1.0,
                "stable": 1.0,
                "diffusion": 1.0,
                "spectral": 1.0,
                "disentangle": 1.0,
            },
        }
        phase = self.phase if self.phase in table else "joint"
        return table[phase][category]

    def _add_weighted(
        self,
        total: torch.Tensor,
        logs: Dict[str, torch.Tensor],
        *,
        key: str,
        value: torch.Tensor,
        weight: float,
        factor: float,
    ) -> torch.Tensor:
        logs[key] = value.detach()
        weighted = float(weight) * float(factor)
        if weighted <= 0.0:
            return total
        total = total + value * weighted
        logs[f"{key}/weighted"] = (value * weighted).detach()
        return total

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        raw_target = batch[self.target_key]
        x = self.transform(raw_x, batch=batch)
        target_scaled = self.transform(raw_target, batch=batch)

        outputs = model(
            x,
            graph=batch.get("graph"),
            mode=self.model_mode,
            dataset_index=batch.get("dataset_index"),
            target=target_scaled,
        )
        if not isinstance(outputs, dict):
            raise TypeError("StableResidualForecastingTask expects model outputs to be a dict")

        pred_scaled = outputs[self.output_key]
        pred = self.inverse_transform(pred_scaled, batch=batch)
        mask = self.merge_masks(
            batch.get("y_mask"),
            batch.get(self.mask_key) if self.mask_key else None,
        )
        target, mask = self.align_prediction_target(pred, raw_target, mask)

        loss_out = losses(pred, target, mask=mask)
        logs: Dict[str, torch.Tensor] = dict(loss_out["logs"])
        total = pred.new_tensor(0.0)
        total = self._add_weighted(
            total,
            logs,
            key="loss/final",
            value=loss_out["loss"],
            weight=self.final_weight,
            factor=self._phase_factor("final"),
        )

        stable_pred = self.inverse_transform(outputs["stable_forecast"], batch=batch)
        stable_target, stable_mask = self.align_prediction_target(stable_pred, target, mask)
        stable_loss = stable_forecast_loss(stable_pred, stable_target, mask=stable_mask)
        trend_loss = trend_consistency_loss(stable_pred, stable_target, mask=stable_mask)
        low_freq_loss = low_frequency_consistency_loss(stable_pred, stable_target, low_ratio=self.low_ratio)
        smooth_loss = temporal_smoothness_loss(stable_pred, mask=stable_mask)

        stable_reconstruction = self.inverse_transform(outputs["stable_reconstruction"], batch=batch)
        recon_target, recon_mask = self.align_prediction_target(
            stable_reconstruction,
            raw_x,
            self.merge_masks(batch.get("x_mask")),
        )
        reconstruction_loss = stable_forecast_loss(
            stable_reconstruction,
            recon_target,
            mask=recon_mask,
        )

        stable_factor = self._phase_factor("stable")
        total = self._add_weighted(
            total,
            logs,
            key="loss/stable_forecast",
            value=stable_loss,
            weight=self.stable_forecast_weight,
            factor=stable_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/stable_reconstruction",
            value=reconstruction_loss,
            weight=self.stable_reconstruction_weight,
            factor=stable_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/trend_consistency",
            value=trend_loss,
            weight=self.trend_weight,
            factor=stable_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/low_frequency",
            value=low_freq_loss,
            weight=self.low_freq_weight,
            factor=stable_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/temporal_smoothness",
            value=smooth_loss,
            weight=self.smoothness_weight,
            factor=stable_factor,
        )

        residual_pred = self.inverse_transform(outputs["residual_forecast"], batch=batch)
        residual_target = target - stable_pred.detach()
        residual_target, residual_mask = self.align_prediction_target(residual_pred, residual_target, mask)

        diffusion_factor = self._phase_factor("diffusion")
        residual_loss = residual_forecast_loss(residual_pred, residual_target, mask=residual_mask)
        propagation_loss = propagation_consistency_loss(residual_pred, outputs["propagation_map"])
        spillover_loss = spillover_reconstruction_loss(outputs["spillover_gate"], residual_target)
        sparsity_loss = propagation_sparsity_loss(outputs["propagation_map"])
        attenuation_loss = attenuation_regularization(outputs["attenuation_gate"], target_decay=0.6)
        locality_loss = event_locality_regularization(outputs["event_score"], outputs["event_locality"])

        total = self._add_weighted(
            total,
            logs,
            key="loss/residual_forecast",
            value=residual_loss,
            weight=self.residual_forecast_weight,
            factor=diffusion_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/propagation_consistency",
            value=propagation_loss,
            weight=self.propagation_consistency_weight,
            factor=diffusion_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/spillover",
            value=spillover_loss,
            weight=self.spillover_weight,
            factor=diffusion_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/propagation_sparsity",
            value=sparsity_loss,
            weight=self.propagation_sparsity_weight,
            factor=diffusion_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/attenuation_regularization",
            value=attenuation_loss,
            weight=self.attenuation_weight,
            factor=diffusion_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/event_locality",
            value=locality_loss,
            weight=self.locality_weight,
            factor=diffusion_factor,
        )

        spectral_factor = self._phase_factor("spectral")
        stable_low = stable_low_frequency_alignment(stable_pred, target, low_ratio=self.low_ratio)
        residual_high = residual_high_frequency_alignment(
            residual_pred,
            residual_target,
            low_ratio=self.low_ratio,
        )
        spectral_drift = anti_spectral_drift_loss(pred, target, scales=self.spectral_scales)
        anti_smooth = anti_oversmoothing_loss(pred)

        total = self._add_weighted(
            total,
            logs,
            key="loss/stable_low_frequency_alignment",
            value=stable_low,
            weight=self.low_freq_weight,
            factor=spectral_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/residual_high_frequency_alignment",
            value=residual_high,
            weight=self.residual_forecast_weight,
            factor=spectral_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/anti_spectral_drift",
            value=spectral_drift,
            weight=self.spectral_drift_weight,
            factor=spectral_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/anti_oversmoothing",
            value=anti_smooth,
            weight=self.anti_oversmoothing_weight,
            factor=spectral_factor,
        )

        disentangle_factor = self._phase_factor("disentangle")
        orth_loss = orthogonality_loss(stable_pred, residual_pred)
        cov_loss = cross_covariance_penalty(stable_pred, residual_pred)
        energy_loss = energy_allocation_regularizer(stable_pred, residual_pred, target_stable_ratio=0.7)
        exclusion_loss = mutual_exclusion_regularizer(stable_pred, residual_pred)

        total = self._add_weighted(
            total,
            logs,
            key="loss/orthogonality",
            value=orth_loss,
            weight=self.orthogonality_weight,
            factor=disentangle_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/cross_covariance",
            value=cov_loss,
            weight=self.cross_cov_weight,
            factor=disentangle_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/energy_allocation",
            value=energy_loss,
            weight=self.energy_allocation_weight,
            factor=disentangle_factor,
        )
        total = self._add_weighted(
            total,
            logs,
            key="loss/mutual_exclusion",
            value=exclusion_loss,
            weight=self.mutual_exclusion_weight,
            factor=disentangle_factor,
        )

        debug_keys = [
            "residual_energy",
            "event_activation",
            "diffusion_gate",
            "inertia_gate",
            "attenuation_gate",
            "propagation_map",
            "fusion_weight",
        ]
        for key in debug_keys:
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                logs[f"debug/{key}"] = value.detach().mean()

        logs["loss/total"] = total.detach()
        return {
            "loss": total,
            "logs": logs,
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": None if mask is None else mask.detach(),
        }
