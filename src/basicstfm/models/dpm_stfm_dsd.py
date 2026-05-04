"""Dataset-Specific Diffusion / Dynamics (DSD) variant of SRD-STFM / DPM-SR++.

Stage-I stable trunk is unchanged. When ``stage2.dataset_specific`` is true, residual diffusion
and optional per-domain fusion gates are routed by ``routing`` (dataset / domain / graph id).

This does **not** change the default ``SRDSTFMBackbone`` forward behavior; use type
``SRDSTFMBackboneDSD`` and DSD configs only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from basicstfm.models.dpm_stfm import SRDSTFMBackbone
from basicstfm.models.diffusion_mechanism_learner import DiffusionMechanismLearner
from basicstfm.registry import MODELS
from basicstfm.utils.diffusion_debug import build_diffusion_debug_payload
from basicstfm.utils.domain_routing import (
    DEFAULT_SHARED_KEY,
    build_routing_from_batch,
    group_indices_by_key,
    sanitize_domain_key,
)
from basicstfm.utils.persistence_anchor import persistence_forecast_from_input


class EventLatentAdapter(nn.Module):
    """Bottleneck residual adapter on event latent [B, T, N, D]."""

    def __init__(self, hidden_dim: int, bottleneck: int = 32) -> None:
        super().__init__()
        b = max(4, min(int(bottleneck), int(hidden_dim)))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, b),
            nn.GELU(),
            nn.Linear(b, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _inv_sigmoid(p: float) -> float:
    p = float(min(max(p, 1e-4), 1.0 - 1e-4))
    return math.log(p / (1.0 - p))


@MODELS.register("SRDSTFMBackboneDSD")
class SRDSTFMBackboneDSD(SRDSTFMBackbone):
    """SRD backbone with optional dataset-specific residual diffusion and gates."""

    def __init__(self, *args: Any, stage2: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        uig = bool(kwargs.get("use_inertia_gate", True))
        uat = bool(kwargs.get("use_attenuation_gate", True))
        super().__init__(*args, **kwargs)
        self._stage2: Dict[str, Any] = dict(stage2 or {})
        self._dsd_active = bool(self._stage2.get("dataset_specific", False))
        self._registered_domains: Tuple[str, ...] = ()
        self.dsd_adapters: Optional[nn.ModuleDict] = None
        self.dsd_diffusions: Optional[nn.ModuleDict] = None
        self.dsd_fusion_gates: Optional[nn.ParameterDict] = None
        self._adapter_scale: float = 0.25
        self._per_domain_gate: bool = False

        if not self._dsd_active:
            return

        raw_list = self._stage2.get("known_domains") or []
        if not raw_list:
            raise ValueError("stage2.known_domains is required when dataset_specific is true")
        domains = tuple(sanitize_domain_key(str(x)) for x in raw_list)
        fb = str(self._stage2.get("fallback_head", "shared")).lower()
        if fb == "shared":
            domain_list = (DEFAULT_SHARED_KEY,) + domains
        elif fb == "raise":
            domain_list = domains
        else:
            raise ValueError("fallback_head must be 'shared' or 'raise'")

        self._registered_domains = tuple(sorted(set(domain_list)))
        self._domain_set = set(self._registered_domains)
        self._adapter_scale = float(self._stage2.get("adapter_scale", 0.25))

        head_mode = str(self._stage2.get("head_mode", "adapter")).lower()
        if head_mode == "lora":
            head_mode = "adapter"
        if head_mode not in {"adapter", "full"}:
            raise ValueError("stage2.head_mode must be 'adapter' or 'full'")

        share_evt = bool(self._stage2.get("share_event_abstraction", True))
        if not share_evt:
            raise NotImplementedError("share_event_abstraction=false is not implemented yet (DSD-lite).")

        bottleneck = int(self._stage2.get("adapter_bottleneck", 32))
        if head_mode == "adapter":
            if not bool(self._stage2.get("share_diffusion_base", True)):
                raise ValueError("adapter mode requires share_diffusion_base=true")
            self.dsd_adapters = nn.ModuleDict(
                {d: EventLatentAdapter(self.hidden_dim, bottleneck) for d in self._registered_domains}
            )
        else:
            self.dsd_diffusions = nn.ModuleDict(
                {
                    d: DiffusionMechanismLearner(
                        hidden_dim=self.hidden_dim,
                        output_dim=self.output_dim,
                        num_datasets=1,
                        use_inertia_gate=uig,
                        use_attenuation_gate=uat,
                    )
                    for d in self._registered_domains
                }
            )

        self._per_domain_gate = bool(self._stage2.get("per_dataset_fusion_gate", True))
        gate_init = float(self._stage2.get("fusion_gate_init", 0.1))
        if self._per_domain_gate:
            g0 = _inv_sigmoid(gate_init)
            self.dsd_fusion_gates = nn.ParameterDict(
                {d: nn.Parameter(torch.tensor(g0, dtype=torch.float32)) for d in self._registered_domains}
            )

    def prepare_routing_from_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self._dsd_active:
            return {}
        bsz = int(batch["x"].shape[0])
        return {
            "routing": build_routing_from_batch(
                batch,
                routing_key=str(self._stage2.get("routing_key", "dataset_name")),
                batch_size=bsz,
            )
        }

    def _resolve_domain_key(self, k: str) -> str:
        if k in self._domain_set:
            return k
        fb = str(self._stage2.get("fallback_head", "shared")).lower()
        if fb == "shared" and DEFAULT_SHARED_KEY in self._domain_set:
            return DEFAULT_SHARED_KEY
        raise KeyError(f"DSD unknown domain key {k!r} (fallback_head={fb!r})")

    def _run_diffusion_domain(
        self,
        event_latent: torch.Tensor,
        event_score: torch.Tensor,
        graph: Optional[torch.Tensor],
        domain_key: str,
        dataset_index: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        dk = self._resolve_domain_key(domain_key)
        head_mode = str(self._stage2.get("head_mode", "adapter")).lower()
        if head_mode == "adapter":
            assert self.dsd_adapters is not None
            z = event_latent + self._adapter_scale * self.dsd_adapters[dk](event_latent)
            return self.diffusion_mechanism_learner(
                event_latent=z,
                event_score=event_score,
                graph=graph,
                output_len=self.output_len,
                dataset_index=None,
            )
        assert self.dsd_diffusions is not None
        return self.dsd_diffusions[dk](
            event_latent=event_latent,
            event_score=event_score,
            graph=graph,
            output_len=self.output_len,
            dataset_index=dataset_index,
        )

    def _forward_chunk(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        mode: str,
        target: Optional[torch.Tensor],
        dataset_index: Optional[torch.Tensor],
        domain_key: str,
        original_channels: int,
    ) -> Dict[str, torch.Tensor]:
        padded_mask = mask
        persistence = None
        if self.use_persistence_anchor:
            persistence = persistence_forecast_from_input(x, self.output_len)

        stable = self.stable_trunk(x, mask=padded_mask)
        stable_reconstruction = stable["stable_reconstruction"]
        stable_forecast = stable["stable_forecast"]

        residual_pack = self.residual_constructor(
            x=x,
            stable_reconstruction=stable_reconstruction,
            stable_forecast=stable_forecast,
            target=target,
        )
        if self.use_persistence_anchor and persistence is not None and target is not None:
            sg = (
                stable_forecast.detach()
                if self.residual_constructor.detach_stable
                else stable_forecast
            )
            residual_pack["residual_target"] = target - persistence - sg
        residual_input = residual_pack["residual_input"]

        event = self.residual_event_encoder(residual_input, graph=graph)

        diffusion_active = self.diffusion_enabled and mode not in {"stable_pretrain", "stable_only"}
        if diffusion_active:
            diffusion = self._run_diffusion_domain(
                event["event_latent"],
                event["event_score"],
                graph,
                domain_key,
                dataset_index,
            )
            residual_forecast = diffusion["residual_forecast"]
        else:
            residual_forecast = stable_forecast.new_zeros(stable_forecast.shape)
            zeros_gate = stable_forecast.new_zeros(
                stable_forecast.shape[0],
                self.output_len,
                stable_forecast.shape[2],
                1,
            )
            zeros_map = stable_forecast.new_zeros(
                stable_forecast.shape[0],
                self.output_len,
                stable_forecast.shape[2],
                stable_forecast.shape[2],
            )
            diffusion = {
                "residual_forecast": residual_forecast,
                "event_activation": zeros_gate,
                "diffusion_gate": zeros_gate,
                "inertia_gate": zeros_gate,
                "attenuation_gate": zeros_gate,
                "spillover_gate": zeros_gate,
                "propagation_map": zeros_map,
                "dataset_gamma": stable_forecast.new_zeros(stable_forecast.shape[0], self.hidden_dim),
                "dataset_beta": stable_forecast.new_zeros(stable_forecast.shape[0], self.hidden_dim),
                "propagator_alpha_spatial": stable_forecast.new_tensor(0.0),
                "propagator_alpha_temporal": stable_forecast.new_tensor(0.0),
                "propagator_alpha_event": stable_forecast.new_tensor(0.0),
                "propagator_gate_entropy": stable_forecast.new_tensor(0.0),
                "propagator_event_intensity_mean": stable_forecast.new_tensor(0.0),
            }

        fusion_mode = self.fusion_mode
        if mode == "gate_fusion":
            fusion_mode = "gate"
        elif mode == "confidence_fusion":
            fusion_mode = "confidence"

        dk = self._resolve_domain_key(domain_key)
        if self._per_domain_gate and diffusion_active and self.dsd_fusion_gates is not None:
            gate = torch.sigmoid(self.dsd_fusion_gates[dk])
            combined_delta = stable_forecast + gate * residual_forecast
            fusion_weight = stable_forecast[..., :1] * 0.0 + gate
            fused = {
                "forecast": combined_delta,
                "fusion_weight": fusion_weight,
                "fusion_mode": stable_forecast.new_tensor(0.0),
            }
            dsd_gate_vec = gate.expand(x.shape[0]).contiguous()
        else:
            fused = self.fusion_predictor(
                stable_forecast=stable_forecast,
                residual_forecast=residual_forecast,
                mode=fusion_mode,
            )
            dsd_gate_vec = fused["fusion_weight"].mean(dim=(1, 2, 3)).detach()

        combined_delta = fused["forecast"]
        if self.use_persistence_anchor and persistence is not None:
            forecast = persistence + self.calibration_head(combined_delta)
        else:
            forecast = self.calibration_head(combined_delta)

        if original_channels < self.output_dim:
            forecast = forecast[..., :original_channels]
            stable_forecast = stable_forecast[..., :original_channels]
            residual_forecast = residual_forecast[..., :original_channels]
            stable_reconstruction = stable_reconstruction[..., :original_channels]
            residual_input = residual_input[..., :original_channels]
            combined_delta = combined_delta[..., :original_channels]
            if persistence is not None:
                persistence = persistence[..., :original_channels]

        out: Dict[str, torch.Tensor] = {
            "forecast": forecast,
            "stable_forecast": stable_forecast,
            "stable_reconstruction": stable_reconstruction,
            "stable_latent": stable["stable_latent"],
            "stable_summary": stable["stable_summary"],
            "stable_temporal_weight": stable["stable_temporal_weight"],
            "stable_component": stable_forecast,
            "stable_local_branch": stable["stable_local_branch"],
            "stable_coarse_branch": stable["stable_coarse_branch"],
            "stable_frequency_branch": stable["stable_frequency_branch"],
            "stable_branch_weight": stable["stable_branch_weight"],
            "residual_input": residual_input,
            "residual_target": residual_pack["residual_target"],
            "residual_forecast": residual_forecast,
            "residual_energy": residual_input.pow(2).mean(dim=(1, 2, 3), keepdim=True),
            "event_score": event["event_score"],
            "event_intensity": event["event_intensity"],
            "event_locality": event["event_locality"],
            "event_activation": diffusion["event_activation"],
            "diffusion_gate": diffusion["diffusion_gate"],
            "inertia_gate": diffusion["inertia_gate"],
            "attenuation_gate": diffusion["attenuation_gate"],
            "spillover_gate": diffusion["spillover_gate"],
            "propagation_map": diffusion["propagation_map"],
            "dataset_gamma": diffusion["dataset_gamma"],
            "dataset_beta": diffusion["dataset_beta"],
            "propagator_alpha_spatial": diffusion["propagator_alpha_spatial"],
            "propagator_alpha_temporal": diffusion["propagator_alpha_temporal"],
            "propagator_alpha_event": diffusion["propagator_alpha_event"],
            "propagator_gate_entropy": diffusion["propagator_gate_entropy"],
            "propagator_event_intensity_mean": diffusion["propagator_event_intensity_mean"],
            "fusion_weight": fused["fusion_weight"],
            "combined_delta": combined_delta,
            "dsd_gate_value": dsd_gate_vec,
        }
        if persistence is not None:
            out["persistence_forecast"] = persistence
        else:
            out["persistence_forecast"] = forecast.new_zeros(forecast.shape)
        return out

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
        dataset_index: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._dsd_active:
            return super().forward(
                x,
                graph=graph,
                mask=mask,
                mode=mode,
                dataset_index=dataset_index,
                target=target,
                routing=routing,
            )
        if routing is None:
            raise RuntimeError(
                "SRDSTFMBackboneDSD(dataset_specific=true) requires routing= from "
                "prepare_routing_from_batch(batch); check StableResidualForecastingTask hook."
            )

        x, padded_mask, original_channels = self._pad_input(x, mask)
        target = self._pad_target(target)
        keys: List[str] = list(routing["keys"])
        bsz = x.shape[0]
        if len(keys) != bsz:
            raise ValueError("routing keys length mismatch")

        buckets = group_indices_by_key(keys)
        merged: Dict[str, torch.Tensor] = {}

        for dom, positions in buckets.items():
            idx = torch.tensor(positions, device=x.device, dtype=torch.long)
            x_sub = x.index_select(0, idx)
            tgt_sub = target.index_select(0, idx) if target is not None else None
            ds_sub = dataset_index.index_select(0, idx) if dataset_index is not None else None
            sub = self._forward_chunk(
                x_sub,
                graph,
                None if padded_mask is None else padded_mask.index_select(0, idx),
                mode,
                tgt_sub,
                ds_sub,
                dom,
                original_channels,
            )
            for k, t in sub.items():
                if not isinstance(t, torch.Tensor):
                    continue
                if t.ndim == 0:
                    if k not in merged:
                        merged[k] = t.clone()
                    continue
                if k not in merged:
                    merged[k] = t.new_zeros(bsz, *t.shape[1:])
                merged[k].index_copy_(0, idx, t)
        debug_payload = build_diffusion_debug_payload(merged)
        for key, stats in debug_payload.items():
            for stat_name, stat_value in stats.items():
                merged[f"debug/{key}/{stat_name}"] = merged["forecast"].new_tensor(stat_value)
        return merged
