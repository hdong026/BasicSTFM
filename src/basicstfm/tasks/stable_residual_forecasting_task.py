"""Stage task for stable-trunk + residual-diffusion forecasting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from basicstfm.data.revin import factost_value_revin_inverse, factost_value_revin_normalize
from basicstfm.losses.disentangle_losses import cross_covariance_penalty
from basicstfm.losses.stable_losses import stable_forecast_loss
from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device
from basicstfm.utils.persistence_anchor import persistence_forecast_from_input
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
        basicts_scale_logging: bool = False,
        use_revin: bool = False,
        revin_value_channel: Union[int, Sequence[int]] = 0,
        revin_eps: float = 1e-5,
        revin_scaled_std_floor: float = 0.05,
        revin_loss_space: str = "normalized",
        revin_metric_space: str = "raw",
        factost_original_scale: bool = False,
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

        self.basicts_scale_logging = bool(basicts_scale_logging)

        self.use_revin = bool(use_revin)
        if isinstance(revin_value_channel, int):
            self.revin_value_channels: Tuple[int, ...] = (int(revin_value_channel),)
        else:
            self.revin_value_channels = tuple(int(c) for c in revin_value_channel)
        self.revin_eps = float(revin_eps)
        self.revin_scaled_std_floor = float(revin_scaled_std_floor)

        rls = str(revin_loss_space).lower().replace("-", "_")
        if rls not in {"normalized", "raw"}:
            raise ValueError("revin_loss_space must be 'normalized' or 'raw'")
        self.revin_loss_space = rls

        rms = str(revin_metric_space).lower().replace("-", "_")
        if rms not in {"normalized", "raw"}:
            raise ValueError("revin_metric_space must be 'normalized' or 'raw'")
        self.revin_metric_space = rms

        self.factost_original_scale = bool(factost_original_scale)

        #: Logged once per stage start when RevIN / BasicTS banners apply.
        if self.use_revin:
            self.scale_protocol = (
                "factost_revin_value|factost_original_scale="
                f"{self.factost_original_scale}|primary_test_metric="
                f"{'original' if self.factost_original_scale else 'revin_raw'}"
            )
        elif (
            self.basicts_scale_logging
            and not self.use_revin
            and self.primary_supervision_space == "denormalized"
        ):
            self.scale_protocol = "basicts_standard_original"
        else:
            self.scale_protocol = None

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
                scaled_std_floor=self.revin_scaled_std_floor,
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

        use_persistence = bool(getattr(model, "use_persistence_anchor", False))
        y_per_scaled: Optional[torch.Tensor] = None
        if use_persistence:
            y_per_scaled = persistence_forecast_from_input(x_scaled, y_scaled.shape[1])

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
        if use_persistence and y_per_scaled is not None:
            y_per_scaled, _, _ = self._align_pred_target(y_per_scaled, y_scaled, loss_mask)

        if self.stable_target == "lowfreq":
            stable_target_scaled, _ = split_low_high_frequency(
                y_scaled,
                low_ratio=self.stable_low_ratio,
                num_low_bins=self.stable_num_low_bins,
            )
            if use_persistence and y_per_scaled is not None:
                stable_target_scaled = stable_target_scaled - y_per_scaled
        elif self.stable_target == "trend":
            stable_target_scaled = extract_temporal_trend(y_scaled, scale=self.trend_scale)
            if use_persistence and y_per_scaled is not None:
                stable_target_scaled = stable_target_scaled - y_per_scaled
        else:
            stable_target_scaled = y_scaled
            if use_persistence and y_per_scaled is not None:
                stable_target_scaled = stable_target_scaled - y_per_scaled

        if use_persistence and y_per_scaled is not None:
            residual_target_scaled = y_scaled - y_per_scaled - y_stable_scaled.detach()
        else:
            residual_target_scaled = y_scaled - y_stable_scaled.detach()

        y_hat_rawspace = (
            factost_value_revin_inverse(y_hat_scaled, batch) if self.use_revin else y_hat_scaled
        )
        pred_denorm = self.inverse_transform(y_hat_rawspace, batch=batch)
        pred_denorm, target_denorm, metric_mask = self._align_pred_target(
            pred_denorm, raw_target, raw_mask
        )

        revin_rr_pred: Optional[torch.Tensor] = None
        revin_rr_tgt: Optional[torch.Tensor] = None
        revin_rr_mask: Any = None
        if self.use_revin:
            y_hat_rr_u = factost_value_revin_inverse(y_hat_scaled.detach(), batch)
            y_tgt_rr_u = factost_value_revin_inverse(y_scaled.detach(), batch)
            revin_rr_pred, revin_rr_tgt, revin_rr_mask = self._align_pred_target(
                y_hat_rr_u, y_tgt_rr_u, loss_mask
            )

        if self.use_revin:
            if self.revin_loss_space == "normalized":
                main_loss_out = losses(y_hat_scaled, y_scaled, mask=loss_mask)
            else:
                main_loss_out = losses(pred_denorm, target_denorm, mask=metric_mask)
        elif self.primary_supervision_space == "normalized":
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
        for _pk in (
            "propagator_alpha_spatial",
            "propagator_alpha_temporal",
            "propagator_alpha_event",
            "propagator_gate_entropy",
            "propagator_event_intensity_mean",
        ):
            _v = outputs.get(_pk)
            if isinstance(_v, torch.Tensor) and _v.numel() > 0:
                logs[f"metric/{_pk}"] = _v.detach().reshape(())
        if self.use_revin:
            rm = batch["revin_mean"]
            rs = batch["revin_std"]
            logs["revin/mean_abs"] = rm.abs().mean()
            logs["revin/std_mean"] = rs.mean()
            if self.revin_loss_space == "normalized":
                if "loss/mae" in logs:
                    logs["loss/mae_norm"] = logs["loss/mae"]
                if "loss/mse" in logs:
                    logs["loss/mse_norm"] = logs["loss/mse"]
            with torch.no_grad():
                err_norm = (y_hat_scaled - y_scaled).abs()
                sq_norm = (y_hat_scaled - y_scaled).pow(2)
                logs["metric/mae_norm"] = _scalar_masked_mean(err_norm, loss_mask)
                logs["metric/rmse_norm"] = torch.sqrt(
                    _scalar_masked_mean(sq_norm, loss_mask).clamp_min(1e-12)
                )
                assert revin_rr_pred is not None and revin_rr_tgt is not None
                err_rr = (revin_rr_pred - revin_rr_tgt).abs()
                sq_rr = (revin_rr_pred - revin_rr_tgt).pow(2)
                logs["metric/mae_revin_raw"] = _scalar_masked_mean(err_rr, revin_rr_mask)
                logs["metric/rmse_revin_raw"] = torch.sqrt(
                    _scalar_masked_mean(sq_rr, revin_rr_mask).clamp_min(1e-12)
                )
                logs["metric/mae_original"] = _scalar_masked_mean(
                    (pred_denorm - target_denorm).abs(),
                    metric_mask,
                )
                sq_o = (pred_denorm - target_denorm).pow(2)
                logs["metric/rmse_original"] = torch.sqrt(
                    _scalar_masked_mean(sq_o, metric_mask).clamp_min(1e-12)
                )
        elif self.primary_supervision_space == "normalized":
            with torch.no_grad():
                logs["metric/mae_raw"] = _scalar_masked_mean(
                    (pred_denorm - target_denorm).abs(),
                    metric_mask,
                )
                sq = (pred_denorm - target_denorm).pow(2)
                logs["metric/rmse_raw"] = torch.sqrt(
                    _scalar_masked_mean(sq, metric_mask).clamp_min(1e-12)
                )
        elif not self.use_revin and self.primary_supervision_space == "denormalized":
            with torch.no_grad():
                err_o = (pred_denorm - target_denorm).abs()
                sq_o = (pred_denorm - target_denorm).pow(2)
                logs["metric/mae_original_after_inverse_standard_scaler"] = _scalar_masked_mean(
                    err_o, metric_mask
                )
                logs["metric/rmse_original_after_inverse_standard_scaler"] = torch.sqrt(
                    _scalar_masked_mean(sq_o, metric_mask).clamp_min(1e-12)
                )
                if use_persistence and y_per_scaled is not None:
                    yper_dn = self.inverse_transform(y_per_scaled, batch=batch)
                    yper_dn, tgt_d, pm = self._align_pred_target(yper_dn, raw_target, raw_mask)
                    logs["metric/persistence_mae"] = _scalar_masked_mean((yper_dn - tgt_d).abs(), pm)
                    logs["metric/stable_delta_mae"] = _scalar_masked_mean(
                        (y_stable_scaled - stable_target_scaled).abs(), loss_mask
                    )
                    logs["metric/final_mae"] = _scalar_masked_mean(err_o, metric_mask)

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
            assert revin_rr_pred is not None and revin_rr_tgt is not None
            if self.factost_original_scale:
                metric_pred = pred_denorm.detach()
                metric_target = target_denorm.detach()
                out_mask = metric_mask
            else:
                metric_pred = revin_rr_pred
                metric_target = revin_rr_tgt
                out_mask = revin_rr_mask
        elif self.primary_supervision_space == "normalized":
            metric_pred = y_hat_scaled.detach()
            metric_target = y_scaled.detach()
            out_mask = loss_mask
        else:
            metric_pred = pred_denorm.detach()
            metric_target = target_denorm.detach()
            out_mask = metric_mask

        return {
            "loss": total,
            "logs": logs,
            "pred": metric_pred,
            "target": metric_target,
            "mask": None if out_mask is None else out_mask.detach(),
        }
