"""Stage-I robust stable-law pretraining for DPM v3 (mini-batch EMA, group-wise standardized risk)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from basicstfm.losses.disentangle_losses import cross_covariance_penalty
from basicstfm.losses.stable_losses import (
    stable_forecast_loss,
    stable_forecast_mae_per_batch_item,
)
from basicstfm.registry import TASKS
from basicstfm.tasks.base import move_to_device
from basicstfm.tasks.stable_residual_forecasting_task import StableResidualForecastingTask
from basicstfm.utils.spectral_ops import (
    extract_temporal_trend,
    multi_scale_spectral_distance,
    split_low_high_frequency,
)


def _group_keys_from_batch(
    batch: Dict[str, Any],
    *,
    group_key: str,
    device: torch.device,
) -> List[str]:
    """Return one string group id per batch item (length B)."""

    x = batch["x"]
    if not isinstance(x, Tensor):
        raise TypeError("batch['x'] must be a tensor")
    b = int(x.shape[0])
    gk = str(group_key).lower().strip()

    if gk in {"dataset", "domain"}:
        if "dataset_index" in batch and isinstance(batch["dataset_index"], Tensor):
            di = batch["dataset_index"].to(device=device)
            if di.numel() == b:
                return [f"ds{int(di[i].item())}" for i in range(b)]
        name = batch.get("dataset_name")
        if isinstance(name, str):
            return [f"name:{name}"] * b
        return ["ds0"] * b

    if gk == "graph":
        out: List[str] = []
        g = batch.get("graph")
        n_nodes = int(x.shape[2])
        di = 0
        if "dataset_index" in batch and isinstance(batch["dataset_index"], Tensor):
            di = int(batch["dataset_index"].reshape(-1)[0].item())
        if isinstance(g, Tensor) and g.ndim == 2:
            gn, gm = int(g.shape[0]), int(g.shape[1])
            # Structure id + dataset (same graph per dataset in typical ST benchmarks)
            out = [f"g{gn}x{gm}_ds{di}" for _ in range(b)]
        else:
            out = [f"nodes{n_nodes}_ds{di}" for _ in range(b)]
        return out

    return ["ds0"] * b


@TASKS.register("StableResidualForecastingTaskV3")
class StableResidualForecastingTaskV3(StableResidualForecastingTask):
    """Same as StableResidualForecastingTask, but optional Stage-I (phase=stable) robust reweighting."""

    def __init__(
        self,
        *args,
        robust_stage1: bool = False,
        robust_lambda: float = 0.3,
        robust_temperature: float = 1.0,
        robust_ema_momentum: float = 0.95,
        robust_use_standardized_risk: bool = True,
        robust_topk_groups: int = 0,
        robust_group_key: str = "dataset",
        robust_relu_cap: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.robust_stage1 = bool(robust_stage1)
        self.robust_lambda = float(robust_lambda)
        self.robust_temperature = max(float(robust_temperature), 1e-6)
        self.robust_ema_momentum = float(robust_ema_momentum)
        self.robust_use_standardized_risk = bool(robust_use_standardized_risk)
        self.robust_topk_groups = max(0, int(robust_topk_groups))
        gk = str(robust_group_key).lower().strip()
        if gk not in {"dataset", "graph", "domain"}:
            raise ValueError("robust_group_key must be one of: dataset, graph, domain")
        self.robust_group_key = gk
        self.robust_relu_cap = float(robust_relu_cap)
        # Per-group EMA of mean loss and second moment (for std)
        self._ema_mean: Dict[str, float] = {}
        self._ema_m2: Dict[str, float] = {}

    def _ema_std(self, g: str) -> float:
        m = self._ema_mean.get(g, 0.0)
        m2 = self._ema_m2.get(g, m * m + 1.0)
        v = max(m2 - m * m, 1e-8)
        return float(v**0.5)

    def _update_ema(self, g: str, loss_val: float) -> None:
        mom = self.robust_ema_momentum
        m = self._ema_mean.get(g, loss_val)
        m2 = self._ema_m2.get(g, loss_val * loss_val)
        self._ema_mean[g] = mom * m + (1.0 - mom) * loss_val
        self._ema_m2[g] = mom * m2 + (1.0 - mom) * (loss_val * loss_val)

    def _per_group_m_g(
        self,
        l_g: Tensor,
        g: str,
    ) -> Tensor:
        """Per-group extra multiplier m_g = 1 + lambda * relu( tau * r_g ) (r_g uses detached EMA)."""
        if g not in self._ema_mean:
            r = l_g.new_zeros(())
        else:
            mu = self._ema_mean[g]
            sig = self._ema_std(g)
            if self.robust_use_standardized_risk:
                r = (l_g.detach() - float(mu)) / (float(sig) + 1e-6)
            else:
                r = l_g.detach() / (float(sig) + 1e-6)
            r = torch.clamp(r, -self.robust_relu_cap * 2.0, self.robust_relu_cap * 2.0)
        w = F.relu(self.robust_temperature * r)
        w = torch.clamp(w, min=0.0, max=self.robust_relu_cap)
        if self.robust_topk_groups > 0:
            # placeholder for future: could mask low-r groups
            pass
        return 1.0 + self.robust_lambda * w

    def _robust_effective_stable_loss(
        self,
        l_stable: Tensor,
        per_item: Tensor,
        group_ids: List[str],
    ) -> Tensor:
        """L_eff = sum_g (n_g / N) * L_g * m_g with per-group m_g; reduces to m * l_stable for one group."""
        n = int(per_item.shape[0])
        if n == 0:
            return l_stable
        unique: List[str] = []
        for gid in group_ids:
            if gid not in unique:
                unique.append(gid)
        if len(unique) == 1 and n >= 1:
            g0 = group_ids[0]
            l_g = per_item.mean()
            m = self._per_group_m_g(l_g, g0)
            return l_stable * m

        terms: List[Tensor] = []
        for g in unique:
            mask = per_item.new_tensor(
                [1.0 if group_ids[i] == g else 0.0 for i in range(n)],
                dtype=per_item.dtype,
            )
            n_g = mask.sum().clamp_min(1.0)
            l_g = (per_item * mask).sum() / n_g
            m = self._per_group_m_g(l_g, g)
            terms.append((n_g / float(n)) * l_g * m)
        if not terms:
            return l_stable
        return torch.stack(terms, dim=0).sum()

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        if not (self.robust_stage1 and self.phase == "stable" and self.model_mode == "stable_pretrain"):
            return super().step(model, batch, losses, device)

        self._step_counter += 1
        batch = move_to_device(batch, device)
        raw_x = batch[self.input_key]
        raw_target = batch[self.target_key]
        x_scaled = self.transform(raw_x, batch=batch)
        y_scaled = self.transform(raw_target, batch=batch)

        outputs = model(
            x_scaled,
            graph=batch.get("graph"),
            mode=self.model_mode,
            dataset_index=batch.get("dataset_index"),
            target=y_scaled,
        )
        if not isinstance(outputs, dict):
            raise TypeError("StableResidualForecastingTaskV3 expects model outputs to be a dict")

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

        pred = self.inverse_transform(y_hat_scaled, batch=batch)
        pred, target_denorm, metric_mask = self._align_pred_target(pred, raw_target, raw_mask)
        main_loss_out = losses(pred, target_denorm, mask=metric_mask)
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
        per_item = stable_forecast_mae_per_batch_item(
            y_stable_scaled, stable_target_scaled, mask=loss_mask
        )
        gids = _group_keys_from_batch(batch, group_key=self.robust_group_key, device=device)
        l_stable_effective = self._robust_effective_stable_loss(l_stable, per_item, gids)

        unique_g: List[str] = []
        for g in gids:
            if g not in unique_g:
                unique_g.append(g)
        for g in unique_g:
            msk = per_item.new_tensor(
                [1.0 if gids[i] == g else 0.0 for i in range(len(gids))],
                dtype=per_item.dtype,
                device=per_item.device,
            )
            l_g = (per_item * msk).sum() / msk.sum().clamp_min(1.0)
            self._update_ema(g, float(l_g.item()))

        total = (
            w_final * l_final
            + w_stable * l_stable_effective
            + w_residual * l_residual
            + w_spec * l_spec
        )

        logs: Dict[str, Tensor] = dict(main_loss_out["logs"])
        with torch.no_grad():
            l_base = l_stable.detach().clamp_min(1e-9)
            logs["aux/robust/stable_ratio"] = (l_stable_effective.detach() / l_base)
        logs["aux/robust/l_stable_effective"] = l_stable_effective.detach()
        if self.log_aux_losses:
            logs["aux/stable"] = l_stable.detach()
            logs["aux/robust/l_stable_baseline"] = l_stable.detach()

        if self.cross_cov_weight > 0.0:
            l_cross_cov = cross_covariance_penalty(y_stable_scaled, y_residual_scaled)
            total = total + self.cross_cov_weight * l_cross_cov
            if self.log_aux_losses:
                logs["aux/cross_covariance"] = l_cross_cov.detach()

        if self.log_debug_tensors:
            self._maybe_log_tensor_stats(logs, "y", y_scaled)
        logs["loss/total"] = total.detach()
        return {
            "loss": total,
            "logs": logs,
            "pred": pred.detach(),
            "target": target_denorm.detach(),
            "mask": None if metric_mask is None else metric_mask.detach(),
        }
