"""ZED: Zero-shot Expert Diffusion — source-only expert bank + inference-time router (MoE)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from basicstfm.models.dpm_stfm import SRDSTFMBackbone
from basicstfm.models.diffusion_mechanism_learner import DiffusionMechanismLearner
from basicstfm.registry import MODELS
from basicstfm.utils.diffusion_debug import build_diffusion_debug_payload
from basicstfm.utils.persistence_anchor import persistence_forecast_from_input
from basicstfm.utils.zed_routing import build_route_features
from basicstfm.utils.domain_routing import sanitize_domain_key


class _SoftmaxRouterMLP(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden: int = 128, top_k: int = 3) -> None:
        super().__init__()
        self.top_k = int(top_k)
        h = max(8, int(hidden))
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), h),
            nn.GELU(),
            nn.Linear(h, int(num_experts)),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(feat)
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            topv, _ = logits.topk(self.top_k, dim=-1)
            thresh = topv[..., -1, None]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        w = F.softmax(logits, dim=-1)
        return w, logits


class _RouterFusionGate(nn.Module):
    def __init__(self, in_dim: int, init_prob: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(int(in_dim), 1)
        p = float(min(max(init_prob, 1e-4), 1.0 - 1e-4))
        bias = math.log(p / (1.0 - p))
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(feat).squeeze(-1))


class _EventAdapter(nn.Module):
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


def _blend_weighted_dicts(
    dicts: List[Dict[str, torch.Tensor]],
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Blend per-expert diffusion dicts with weights ``[B, K]``."""

    if not dicts:
        raise ValueError("empty dict list")
    bsz = int(weights.shape[0])
    k = int(weights.shape[1])
    if len(dicts) != k:
        raise ValueError("expert count mismatch")

    out: Dict[str, torch.Tensor] = {}
    keys = [key for key in dicts[0] if isinstance(dicts[0][key], torch.Tensor)]
    for key in keys:
        ref = dicts[0][key]
        if ref.ndim == 0:
            acc = dicts[0][key].new_zeros(())
            for i in range(k):
                acc = acc + weights[:, i].mean() * dicts[i][key]
            out[key] = acc
            continue
        stk = torch.stack([dicts[i][key] for i in range(k)], dim=1)
        w = weights.view(bsz, k, *([1] * (stk.ndim - 2)))
        out[key] = (stk * w).sum(dim=1)
    return out


@MODELS.register("SRDSTFMBackboneZED")
class SRDSTFMBackboneZED(SRDSTFMBackbone):
    """MoE residual diffusion: experts trained on source domains; router mixes at inference."""

    def __init__(self, *args: Any, stage2: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        kwargs.pop("stage1", None)  # optional YAML documentation; loading uses trainer stages
        uig = bool(kwargs.get("use_inertia_gate", True))
        uat = bool(kwargs.get("use_attenuation_gate", True))
        super().__init__(*args, **kwargs)
        self._stage2: Dict[str, Any] = dict(stage2 or {})
        if not bool(self._stage2.get("expert_bank", False)):
            raise ValueError("SRDSTFMBackboneZED requires stage2.expert_bank: true")

        raw_experts = self._stage2.get("expert_domains") or self._stage2.get("expert_names") or []
        if not raw_experts:
            raise ValueError("stage2.expert_domains (or expert_names) is required for ZED")
        self._expert_order: Tuple[str, ...] = tuple(
            sanitize_domain_key(str(x)) for x in raw_experts
        )
        self.num_experts = len(self._expert_order)

        head_mode = str(self._stage2.get("head_mode", "adapter")).lower()
        if head_mode == "lora":
            head_mode = "adapter"
        if head_mode not in {"adapter", "full"}:
            raise ValueError("ZED stage2.head_mode must be 'adapter' or 'full'")

        if not bool(self._stage2.get("share_event_abstraction", True)):
            raise NotImplementedError("ZED requires share_event_abstraction=true")

        share_base = bool(self._stage2.get("share_diffusion_base", True))
        self._adapter_scale = float(self._stage2.get("adapter_scale", 0.25))
        bottleneck = int(self._stage2.get("adapter_bottleneck", 32))

        if head_mode == "adapter":
            if not share_base:
                raise ValueError("ZED adapter mode requires share_diffusion_base=true")
            self.zed_expert_adapters = nn.ModuleDict(
                {name: _EventAdapter(self.hidden_dim, bottleneck) for name in self._expert_order}
            )
            self.zed_expert_diffusions = None
        else:
            self.zed_expert_adapters = None
            self.zed_expert_diffusions = nn.ModuleDict(
                {
                    name: DiffusionMechanismLearner(
                        hidden_dim=self.hidden_dim,
                        output_dim=self.output_dim,
                        num_datasets=1,
                        use_inertia_gate=uig,
                        use_attenuation_gate=uat,
                    )
                    for name in self._expert_order
                }
            )

        r_inputs = self._stage2.get("router_inputs") or [
            "graph_signature",
            "temporal_statistics",
            "residual_event_signature",
        ]
        self._router_inputs: List[str] = [str(x) for x in r_inputs]

        # Router input dim: infer by probe
        probe_b = 2
        probe_t = int(max(4, min(self.input_len, 64)))
        probe_n = int(max(2, min(getattr(self, "num_nodes", 8), 128)))
        probe_c = int(self.input_dim)
        probe_d = int(self.hidden_dim)
        px = torch.zeros(probe_b, probe_t, probe_n, probe_c)
        pz = torch.zeros(probe_b, probe_t, probe_n, probe_d)
        pg = torch.ones(probe_n, probe_n) * 0.5
        feat_dim = int(
            build_route_features(
                px,
                pg,
                pz,
                self._router_inputs,
                optional_dataset_id=(
                    torch.zeros(probe_b, dtype=torch.long)
                    if "dataset_id_embed" in {s.lower().strip() for s in self._router_inputs}
                    else None
                ),
                max_dataset_embed=int(self._stage2.get("max_dataset_embed", 64)),
            ).shape[-1]
        )

        rtype = str(self._stage2.get("router_type", "softmax_mlp")).lower()
        if rtype not in {"softmax_mlp", "softmax"}:
            raise ValueError(f"Unsupported router_type {rtype!r}")
        top_k = int(self._stage2.get("top_k_experts", 3))
        router_hidden = int(self._stage2.get("router_hidden", 128))
        self.zed_router = _SoftmaxRouterMLP(
            feat_dim,
            self.num_experts,
            hidden=router_hidden,
            top_k=top_k,
        )

        self._use_router_fusion_gate = bool(self._stage2.get("router_fusion_gate", True))
        gate_init = float(self._stage2.get("fusion_gate_init", 0.1))
        self.zed_route_gate: Optional[_RouterFusionGate] = None
        if self._use_router_fusion_gate:
            self.zed_route_gate = _RouterFusionGate(feat_dim, init_prob=gate_init)

        self._routing_key_cfg = str(self._stage2.get("routing_key", "auto")).lower()

        self.zed_fewshot_adapter: Optional[_EventAdapter] = None
        self._fewshot_scale = float(self._stage2.get("fewshot_adapter_scale", 0.25))
        if bool(self._stage2.get("enable_fewshot_adapter", False)):
            self.zed_fewshot_adapter = _EventAdapter(self.hidden_dim, bottleneck)

        self._max_dataset_embed = int(self._stage2.get("max_dataset_embed", 64))

    def prepare_zed_router_aux(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "dataset_id_embed" not in {s.lower().strip() for s in self._router_inputs}:
            return {}
        idx = batch.get("dataset_index")
        if isinstance(idx, torch.Tensor) and idx.ndim == 1:
            return {"router_dataset_index": idx}
        return {}

    def _run_single_expert(
        self,
        event_latent: torch.Tensor,
        event_score: torch.Tensor,
        graph: Optional[torch.Tensor],
        expert_key: str,
        dataset_index: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.zed_expert_adapters is not None:
            z = event_latent + self._adapter_scale * self.zed_expert_adapters[expert_key](event_latent)
            return self.diffusion_mechanism_learner(
                event_latent=z,
                event_score=event_score,
                graph=graph,
                output_len=self.output_len,
                dataset_index=dataset_index,
            )
        assert self.zed_expert_diffusions is not None
        return self.zed_expert_diffusions[expert_key](
            event_latent=event_latent,
            event_score=event_score,
            graph=graph,
            output_len=self.output_len,
            dataset_index=dataset_index,
        )

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
        dataset_index: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        routing: Optional[Dict[str, Any]] = None,
        router_dataset_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = routing  # unused
        x, padded_mask, original_channels = self._pad_input(x, mask)
        target = self._pad_target(target)

        persistence = None
        if self.use_persistence_anchor:
            persistence = persistence_forecast_from_input(x, self.output_len)

        stable = self.stable_trunk(x, mask=padded_mask)
        stable_reconstruction = stable["stable_reconstruction"]
        stable_forecast = stable["stable_forecast"]

        if mode in {"encode", "embedding"}:
            return {"embedding": stable["stable_latent"]}

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
        event_latent = event["event_latent"]

        if self.zed_fewshot_adapter is not None:
            event_latent = event_latent + self._fewshot_scale * self.zed_fewshot_adapter(event_latent)

        opt_ds = router_dataset_index
        route_feat = build_route_features(
            x,
            graph,
            event_latent,
            self._router_inputs,
            optional_dataset_id=opt_ds,
            max_dataset_embed=self._max_dataset_embed,
        )

        expert_w, expert_logits = self.zed_router(route_feat)
        entropy = -(expert_w * (expert_w.clamp_min(1e-9).log())).sum(dim=-1)
        top1 = expert_w.argmax(dim=-1)

        if self.zed_route_gate is not None:
            gate = self.zed_route_gate(route_feat)
        else:
            gate = x.new_ones(x.shape[0])

        diffusion_active = self.diffusion_enabled and mode not in {"stable_pretrain", "stable_only"}
        if diffusion_active:
            expert_diffs: List[Dict[str, torch.Tensor]] = []
            for expert_key in self._expert_order:
                expert_diffs.append(
                    self._run_single_expert(
                        event_latent,
                        event["event_score"],
                        graph,
                        expert_key,
                        dataset_index,
                    )
                )
            diffusion = _blend_weighted_dicts(expert_diffs, expert_w)
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

        if diffusion_active and self._use_router_fusion_gate:
            g4 = gate.view(-1, 1, 1, 1)
            combined_delta = stable_forecast + g4 * residual_forecast
            fusion_weight = stable_forecast[..., :1] * 0.0 + g4
            fused_mode = stable_forecast.new_tensor(0.0)
        elif not diffusion_active:
            fusion_mode = self.fusion_mode
            if mode == "gate_fusion":
                fusion_mode = "gate"
            elif mode == "confidence_fusion":
                fusion_mode = "confidence"
            fused = self.fusion_predictor(
                stable_forecast=stable_forecast,
                residual_forecast=residual_forecast,
                mode=fusion_mode,
            )
            combined_delta = fused["forecast"]
            fusion_weight = fused["fusion_weight"]
            fused_mode = fused["fusion_mode"]
        else:
            fusion_mode = self.fusion_mode
            if mode == "gate_fusion":
                fusion_mode = "gate"
            elif mode == "confidence_fusion":
                fusion_mode = "confidence"
            fused = self.fusion_predictor(
                stable_forecast=stable_forecast,
                residual_forecast=residual_forecast,
                mode=fusion_mode,
            )
            combined_delta = fused["forecast"]
            fusion_weight = fused["fusion_weight"]
            fused_mode = fused["fusion_mode"]

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

        topv, _ = torch.topk(expert_w, k=min(3, expert_w.shape[-1]), dim=-1)

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
            "fusion_weight": fusion_weight,
            "combined_delta": combined_delta,
            "zed_router_entropy": entropy,
            "zed_top1_expert": top1.to(dtype=torch.float32),
            "zed_expert_weights": expert_w,
            "zed_expert_logits": expert_logits,
            "zed_fusion_gate": gate,
            "zed_topk_weight_sum": topv.sum(dim=-1),
            "zed_fusion_mode_scalar": fused_mode,
        }
        if persistence is not None:
            out["persistence_forecast"] = persistence
        else:
            out["persistence_forecast"] = forecast.new_zeros(forecast.shape)

        debug_payload = build_diffusion_debug_payload(out)
        for key, stats in debug_payload.items():
            for stat_name, stat_value in stats.items():
                out[f"debug/{key}/{stat_name}"] = out["forecast"].new_tensor(stat_value)
        return out
