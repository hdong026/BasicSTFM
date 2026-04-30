"""Stage task for stable-trunk + residual-diffusion forecasting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from basicstfm.data.revin import factost_value_revin_inverse, factost_value_revin_normalize
from basicstfm.losses.disentangle_losses import cross_covariance_penalty
from basicstfm.losses.stable_losses import stable_forecast_loss
from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device
from basicstfm.utils.spectral_ops import (
    extract_temporal_trend,
    multi_scale_spectral_distance,
    split_low_high_frequency,
)


def _scalar_masked_mean(abs_err: torch.Tensor, mask: Any) -> torch.Tensor:
    if mask is None:
        return abs_err.mean()
    m = mask
    while m.ndim < abs_err.ndim:
        m = m.unsqueeze(-1)
    m = m.to(dtype=abs_err.dtype, device=abs_err.device).expand_as(abs_err)
    return (abs_err * m).sum() / m.sum().clamp_min(1.0)


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
        stable_target: str = "lowfreq",
        stable_low_ratio: float = 0.3,
        stable_num_low_bins: Optional[int] = None,
        trend_scale: int = 4,
        final_weight: float = 1.0,
        stable_weight: float = 0.2,
        residual_weight: float = 0.2,
        spectral_weight: float = 0.02,
        spectral_scales: Optional[Sequence[int]] = None,
        log_interval: int = 200,
        log_aux_losses: bool = False,
        log_debug_tensors: bool = False,
        print_debug: bool = False,
        # Disabled by default in the current repair phase.
        cross_cov_weight: float = 0.0,
        primary_supervision_space: str = "denormalized",
        use_revin: bool = False,
        revin_value_channel: Union[int, Sequence[int]] = 0,
        revin_eps: float = 1e-5,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.model_mode = model_mode
        self.phase = str(phase)
        self.mask_key = mask_key

        pss = str(primary_supervision_space).lower().replace("-", "_")
        if pss not in {"denormalized", "normalized"}:
            raise ValueError("primary_supervision_space must be 'denormalized' or 'normalized'")
        self.primary_supervision_space = pss

        if stable_target not in {"lowfreq", "trend", "target"}:
            raise ValueError("stable_target must be one of: lowfreq, trend, target")
        self.stable_target = stable_target
        self.stable_low_ratio = float(stable_low_ratio)
        self.stable_num_low_bins = (
            None if stable_num_low_bins is None else int(stable_num_low_bins)
        )
        self.trend_scale = int(trend_scale)

        self.final_weight = float(final_weight)
        self.stable_weight = float(stable_weight)
        self.residual_weight = float(residual_weight)
        self.spectral_weight = float(spectral_weight)
        self.spectral_scales = tuple(int(item) for item in (spectral_scales or (1,)))

        self.cross_cov_weight = float(cross_cov_weight)
        self.log_interval = max(1, int(log_interval))
        self.log_aux_losses = bool(log_aux_losses)
        self.log_debug_tensors = bool(log_debug_tensors)
        self.print_debug = bool(print_debug)
        self._step_counter = 0

        self.use_revin = bool(use_revin)
        if isinstance(revin_value_channel, int):
            self.revin_value_channels: Tuple[int, ...] = (int(revin_value_channel),)
        else:
            self.revin_value_channels = tuple(int(c) for c in revin_value_channel)
        self.revin_eps = float(revin_eps)
        #: Logged once per stage build in ``MultiStageTrainer`` when RevIN targets are enabled.
        self.scaling_protocol: Optional[str] = "factost_revin_value" if self.use_revin else None

    def _phase_weights(self) -> tuple[float, float, float, float]:
        if self.phase == "stable":
            # Stage 1: train stable branch only.
            return 0.0, 1.0, 0.0, 0.0
        if self.phase == "diffusion":
            # Stage 2: frozen stable, train residual diffusion with oracle residual target.
            return 0.0, 0.0, 1.0, 0.0
        if self.phase in {"joint", "scratch"}:
            # Stage 3 / control: minimal trainable objective.
            return self.final_weight, self.stable_weight, self.residual_weight, self.spectral_weight
        if self.phase == "stable_only":
            return 1.0, 0.2, 0.0, 0.02
        return self.final_weight, self.stable_weight, self.residual_weight, self.spectral_weight

    def _align_pred_target(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        aligned_target, aligned_mask = self.align_prediction_target(pred, target, mask)
        aligned_pred = pred
        if pred.shape[-1] > aligned_target.shape[-1]:
            aligned_pred = pred[..., : aligned_target.shape[-1]]
        elif pred.shape[-1] < aligned_target.shape[-1]:
            pad_channels = aligned_target.shape[-1] - pred.shape[-1]
            aligned_pred = torch.cat(
                [pred, pred.new_zeros(*pred.shape[:-1], pad_channels)],
                dim=-1,
            )
        return aligned_pred, aligned_target, aligned_mask

    @staticmethod
    def _stats(value: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = value.detach()
        return {
            "mean": value.mean(),
            "std": value.std(unbiased=False),
            "min": value.min(),
            "max": value.max(),
        }

    def _maybe_log_tensor_stats(self, logs: Dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
        stats = self._stats(value)
        for stat_name, stat_value in stats.items():
            logs[f"debug/{name}/{stat_name}"] = stat_value

    def _maybe_print_key_stats(
        self,
        *,
        y: torch.Tensor,
        y_stable: torch.Tensor,
        y_residual: torch.Tensor,
        y_hat: torch.Tensor,
        residual_target: torch.Tensor,
    ) -> None:
        if self._step_counter % self.log_interval != 0:
            return

        def fmt(name: str, tensor: torch.Tensor) -> str:
            s = self._stats(tensor)
            return (
                f"{name}(mean={float(s['mean']):.4f}, std={float(s['std']):.4f}, "
                f"min={float(s['min']):.4f}, max={float(s['max']):.4f})"
            )

        print(
            "[SRD-DEBUG] "
            f"step={self._step_counter} phase={self.phase} "
            f"{fmt('y', y)} {fmt('y_stable', y_stable)} {fmt('y_residual', y_residual)} "
            f"{fmt('y_hat', y_hat)} {fmt('residual_target', residual_target)}"
        )

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        self._step_counter += 1

        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        raw_target = batch[self.target_key]

        x_scaled = self.transform(raw_x, batch=batch)
        y_scaled = self.transform(raw_target, batch=batch)
        if self.use_revin:
            x_scaled, y_scaled = factost_value_revin_normalize(
                x_scaled,
                y_scaled,
                batch,
                value_channels=self.revin_value_channels,
                eps=self.revin_eps,
            )

        outputs = model(
            x_scaled,
            graph=batch.get("graph"),
            mode=self.model_mode,
            dataset_index=batch.get("dataset_index"),
            target=y_scaled,
        )
        if not isinstance(outputs, dict):
            raise TypeError("StableResidualForecastingTask expects model outputs to be a dict")

        y_hat_scaled = outputs[self.output_key]
        y_stable_scaled = outputs["stable_forecast"]
        y_residual_scaled = outputs["residual_forecast"]

        raw_mask = self.merge_masks(
            batch.get("y_mask"),
            batch.get(self.mask_key) if self.mask_key else None,
        )

        y_hat_scaled, y_scaled, loss_mask = self._align_pred_target(y_hat_scaled, y_scaled, raw_mask)
        y_stable_scaled, _, _ = self._align_pred_target(y_stable_scaled, y_scaled, loss_mask)
        y_residual_scaled, _, _ = self._align_pred_target(y_residual_scaled, y_scaled, loss_mask)

        if self.stable_target == "lowfreq":
            stable_target_scaled, _ = split_low_high_frequency(
                y_scaled,
                low_ratio=self.stable_low_ratio,
                num_low_bins=self.stable_num_low_bins,
            )
        elif self.stable_target == "trend":
            stable_target_scaled = extract_temporal_trend(y_scaled, scale=self.trend_scale)
        else:
            stable_target_scaled = y_scaled

        residual_target_scaled = y_scaled - y_stable_scaled.detach()

        y_hat_rawspace = (
            factost_value_revin_inverse(y_hat_scaled, batch) if self.use_revin else y_hat_scaled
        )
        pred_denorm = self.inverse_transform(y_hat_rawspace, batch=batch)
        pred_denorm, target_denorm, metric_mask = self._align_pred_target(
            pred_denorm, raw_target, raw_mask
        )

        use_norm_main = self.use_revin or self.primary_supervision_space == "normalized"
        if use_norm_main:
            main_loss_out = losses(y_hat_scaled, y_scaled, mask=loss_mask)
        else:
            main_loss_out = losses(pred_denorm, target_denorm, mask=metric_mask)

        l_final = main_loss_out["loss"]
        l_stable = stable_forecast_loss(y_stable_scaled, stable_target_scaled, mask=loss_mask)
        l_residual = stable_forecast_loss(y_residual_scaled, residual_target_scaled, mask=loss_mask)
        l_spec = multi_scale_spectral_distance(
            y_hat_scaled,
            y_scaled,
            scales=self.spectral_scales,
            log_amplitude=True,
        )

        w_final, w_stable, w_residual, w_spec = self._phase_weights()
        total = (
            w_final * l_final
            + w_stable * l_stable
            + w_residual * l_residual
            + w_spec * l_spec
        )

        logs: Dict[str, torch.Tensor] = dict(main_loss_out["logs"])
        if self.use_revin:
            for lk, lv in list(logs.items()):
                if lk.startswith("loss/") and lk != "loss/total":
                    logs[f"{lk}_norm"] = lv

        with torch.no_grad():
            if self.primary_supervision_space == "normalized" or self.use_revin:
                logs["metric/raw_mae"] = _scalar_masked_mean(
                    (pred_denorm - target_denorm).abs(),
                    metric_mask,
                )

        if self.log_aux_losses:
            logs.update(
                {
                    "aux/stable": l_stable.detach(),
                    "aux/residual": l_residual.detach(),
                    "aux/spec": l_spec.detach(),
                    "aux/stable/weighted": (w_stable * l_stable).detach(),
                    "aux/residual/weighted": (w_residual * l_residual).detach(),
                    "aux/spec/weighted": (w_spec * l_spec).detach(),
                }
            )

        # TODO: keep disabled in current repair phase to avoid scale pollution.
        # If re-enabled, only standardized implementation should be used.
        if self.cross_cov_weight > 0.0:
            l_cross_cov = cross_covariance_penalty(y_stable_scaled, y_residual_scaled)
            total = total + self.cross_cov_weight * l_cross_cov
            if self.log_aux_losses:
                logs["aux/cross_covariance"] = l_cross_cov.detach()
                logs["aux/cross_covariance/weighted"] = (
                    self.cross_cov_weight * l_cross_cov
                ).detach()

        if self.log_debug_tensors:
            self._maybe_log_tensor_stats(logs, "y", y_scaled)
            self._maybe_log_tensor_stats(logs, "y_stable", y_stable_scaled)
            self._maybe_log_tensor_stats(logs, "y_residual", y_residual_scaled)
            self._maybe_log_tensor_stats(logs, "y_hat", y_hat_scaled)
            self._maybe_log_tensor_stats(logs, "residual_target", residual_target_scaled)

            gate_keys = [
                "event_activation",
                "diffusion_gate",
                "inertia_gate",
                "attenuation_gate",
                "propagation_map",
                "fusion_weight",
            ]
            for key in gate_keys:
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    logs[f"debug/{key}/mean"] = value.detach().mean()

        if self.print_debug:
            self._maybe_print_key_stats(
                y=y_scaled,
                y_stable=y_stable_scaled,
                y_residual=y_residual_scaled,
                y_hat=y_hat_scaled,
                residual_target=residual_target_scaled,
            )

        logs["loss/total"] = total.detach()

        if self.use_revin:
            metric_pred = pred_denorm.detach()
            metric_target = target_denorm.detach()
        elif self.primary_supervision_space == "normalized":
            metric_pred = y_hat_scaled.detach()
            metric_target = y_scaled.detach()
        else:
            metric_pred = pred_denorm.detach()
            metric_target = target_denorm.detach()

        return {
            "loss": total,
            "logs": logs,
            "pred": metric_pred,
            "target": metric_target,
            "mask": None if metric_mask is None else metric_mask.detach(),
        }
