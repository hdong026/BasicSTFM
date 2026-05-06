"""ZED: Zero-shot Expert Diffusion — source-only expert bank + inference-time router (MoE)."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# Parameters loaded by ``load_zed_zero_shot_backbone`` (ZED-FA few-shot pretrain snapshot).
ZED_ZERO_SHOT_BACKBONE_PREFIXES: Tuple[str, ...] = (
    "stable_trunk.",
    "residual_event_encoder.",
    "residual_constructor.",
    "diffusion_mechanism_learner.",
    "zed_expert_adapters.",
    "zed_expert_diffusions.",
    "zed_router.",
    "zed_route_gate.",
    "zed_router_feat_adapter.",
    "fusion_predictor.",
    "calibration_head.",
)


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
        return self._logits_to_weights(logits)

    def _logits_to_weights(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class _ZEDSpatialDeltaAdapter(nn.Module):
    """Bottleneck residual adapter on fused delta ``[B, T, N, C]`` (FactoST-style spatial/temporal calibration)."""

    def __init__(self, channels: int, bottleneck_ratio: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = max(2, int(bottleneck_ratio))
        b = max(4, c // r)
        self.norm = nn.LayerNorm(c)
        self.down = nn.Linear(c, b)
        self.up = nn.Linear(b, c)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = h.shape
        x = h.reshape(-1, shape[-1])
        x = self.norm(x)
        x = self.up(F.gelu(self.down(x)))
        return x.view(*shape)


class _ZEDAffineCalib(nn.Module):
    """Affine calibration ``y <- alpha * y + beta`` on final forecast (lightweight target adaptation)."""

    def __init__(
        self,
        mode: str,
        num_nodes: int,
        channels: int,
        init_alpha: float = 1.0,
        init_beta: float = 0.0,
        node_channel_param_cap: int = 4096,
    ) -> None:
        super().__init__()
        mode = str(mode).lower().strip()
        c = int(channels)
        n = int(num_nodes)
        if mode not in {"scalar", "channel", "node_channel"}:
            raise ValueError("calibration.mode must be scalar|channel|node_channel")
        if mode == "node_channel" and n * c > int(node_channel_param_cap):
            mode = "channel"
        self.mode = mode
        if mode == "scalar":
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
            self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        elif mode == "channel":
            self.alpha = nn.Parameter(torch.full((c,), float(init_alpha)))
            self.beta = nn.Parameter(torch.full((c,), float(init_beta)))
        else:
            self.alpha = nn.Parameter(torch.full((n, c), float(init_alpha)))
            self.beta = nn.Parameter(torch.full((n, c), float(init_beta)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "scalar":
            return y * self.alpha + self.beta
        if self.mode == "channel":
            a = self.alpha.view(1, 1, 1, -1)
            b = self.beta.view(1, 1, 1, -1)
            return y * a + b
        a = self.alpha.view(1, 1, *self.alpha.shape)
        b = self.beta.view(1, 1, *self.beta.shape)
        return y * a + b


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

    def __init__(
        self,
        *args: Any,
        stage2: Optional[Dict[str, Any]] = None,
        few_shot: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("stage1", None)  # optional YAML documentation; loading uses trainer stages
        self._few_shot_cfg: Dict[str, Any] = dict(few_shot or {})
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

        adapt_mode = str(self._few_shot_cfg.get("adaptation_mode", "router_gate_adapter")).lower()
        if adapt_mode in {"factorized_adapter", "zed_fa"}:
            use_rfa = bool(self._few_shot_cfg.get("train_router_feat_adapter", False))
        elif adapt_mode != "router_gate_adapter":
            use_rfa = bool(self._few_shot_cfg.get("train_router_feat_adapter", False))
        else:
            use_rfa = bool(self._few_shot_cfg.get("train_router_feat_adapter", True))
        rfa_bottleneck = int(self._few_shot_cfg.get("router_feat_bottleneck", bottleneck))
        self.zed_router_feat_adapter: Optional[_EventAdapter] = None
        self._router_feat_adapter_scale = float(self._few_shot_cfg.get("router_feat_adapter_scale", 0.25))
        if self._few_shot_cfg and use_rfa:
            self.zed_router_feat_adapter = _EventAdapter(feat_dim, rfa_bottleneck)

        self.zed_fewshot_adapter: Optional[_EventAdapter] = None
        self._fewshot_scale = float(self._stage2.get("fewshot_adapter_scale", 0.25))
        if bool(self._stage2.get("enable_fewshot_adapter", False)):
            train_tgt_ad = (
                bool(self._few_shot_cfg.get("train_target_adapter", True))
                if self._few_shot_cfg
                else True
            )
            if train_tgt_ad:
                self.zed_fewshot_adapter = _EventAdapter(self.hidden_dim, bottleneck)

        self._max_dataset_embed = int(self._stage2.get("max_dataset_embed", 64))

        # --- Factorized adaptation (ZED-FA): lightweight target-only modules (few-shot). ---
        self._zed_fa_enabled = False
        self.zed_fa_prompt_embed: Optional[nn.Embedding] = None
        self.zed_fa_prompt_proj: Optional[nn.Linear] = None
        self.zed_fa_router_bias: Optional[nn.Parameter] = None
        self.zed_fa_spatial_adapter: Optional[_ZEDSpatialDeltaAdapter] = None
        self._zed_fa_spatial_scale = float(self._few_shot_cfg.get("spatial_adapter_scale", 0.25))
        self.zed_fa_affine: Optional[_ZEDAffineCalib] = None

        fa_on = bool(self._few_shot_cfg.get("enable_factorized_adaptation", False))
        if fa_on and self._few_shot_cfg:
            self._zed_fa_enabled = True
            if bool(self._few_shot_cfg.get("train_target_prompt", True)):
                p_dim = max(8, int(self._few_shot_cfg.get("target_prompt_dim", 64)))
                n_ds = max(
                    int(self._few_shot_cfg.get("max_target_prompt_datasets", self._max_dataset_embed)),
                    1,
                )
                self.zed_fa_prompt_embed = nn.Embedding(n_ds, p_dim)
                self.zed_fa_prompt_proj = nn.Linear(p_dim, feat_dim)
                nn.init.normal_(self.zed_fa_prompt_embed.weight, std=0.02)
                nn.init.zeros_(self.zed_fa_prompt_proj.bias)

            if bool(self._few_shot_cfg.get("router_offset", True)):
                self.zed_fa_router_bias = nn.Parameter(torch.zeros(self.num_experts))

            if bool(self._few_shot_cfg.get("target_spatial_adapter", True)):
                ratio = int(self._few_shot_cfg.get("spatial_adapter_bottleneck_ratio", 8))
                self.zed_fa_spatial_adapter = _ZEDSpatialDeltaAdapter(self.output_dim, bottleneck_ratio=ratio)

            if bool(self._few_shot_cfg.get("affine_calibration", True)):
                cal = self._few_shot_cfg.get("calibration") or {}
                mode = str(cal.get("mode", "node_channel"))
                cap = int(cal.get("node_channel_param_cap", 4096))
                self.zed_fa_affine = _ZEDAffineCalib(
                    mode,
                    int(getattr(self, "num_nodes", 1) or 1),
                    int(self.output_dim),
                    init_alpha=float(cal.get("init_alpha", 1.0)),
                    init_beta=float(cal.get("init_beta", 0.0)),
                    node_channel_param_cap=cap,
                )

    def prepare_zed_router_aux(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "dataset_id_embed" not in {s.lower().strip() for s in self._router_inputs}:
            return {}
        idx = batch.get("dataset_index")
        if isinstance(idx, torch.Tensor) and idx.ndim == 1:
            return {"router_dataset_index": idx}
        return {}

    def load_zed_zero_shot_backbone(self, path: str, strict: bool = False) -> Dict[str, Any]:
        """Load pretrained ZED (joint / zero-shot) weights; leave FA-only modules randomly initialized."""

        from basicstfm.models.foundation.common import load_filtered_weights

        miss, unexp = load_filtered_weights(
            self,
            path,
            strict=strict,
            include_prefixes=ZED_ZERO_SHOT_BACKBONE_PREFIXES,
            map_location="cpu",
        )
        logger.info(
            "load_zed_zero_shot_backbone: path=%r strict=%s included_prefixes=%s",
            path,
            strict,
            list(ZED_ZERO_SHOT_BACKBONE_PREFIXES),
        )
        return {
            "missing_keys": miss,
            "unexpected_keys": unexp,
            "zed_load_included_prefixes": list(ZED_ZERO_SHOT_BACKBONE_PREFIXES),
            "zed_load_excluded_note": (
                "Parameters outside included prefixes (notably zed_fa_* and zed_fewshot_adapter.*) "
                "keep current initialization."
            ),
        }

    def zed_fa_enabled(self) -> bool:
        return bool(getattr(self, "_zed_fa_enabled", False))

    def zed_fa_adapter_l2_penalty(self) -> torch.Tensor:
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        reg = torch.zeros((), device=dev, dtype=dt)
        modules: List[Optional[nn.Module]] = [
            self.zed_fa_spatial_adapter,
            self.zed_fa_prompt_proj,
        ]
        if self.zed_fa_prompt_embed is not None:
            reg = reg + self.zed_fa_prompt_embed.weight.pow(2).sum()
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                reg = reg + p.pow(2).sum()
        if self.zed_fa_router_bias is not None:
            reg = reg + self.zed_fa_router_bias.pow(2).sum()
        if self.zed_fa_affine is not None:
            reg = reg + self.zed_fa_affine.alpha.pow(2).sum() + self.zed_fa_affine.beta.pow(2).sum()
        return reg

    def zed_fa_logging_scalars(self, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.zed_fa_prompt_embed is not None:
            out["zed_fa/prompt_weight_norm"] = self.zed_fa_prompt_embed.weight.detach().norm()
        if self.zed_fa_spatial_adapter is not None:
            s = ref.new_tensor(0.0)
            for p in self.zed_fa_spatial_adapter.parameters():
                s = s + p.detach().pow(2).sum()
            out["zed_fa/spatial_adapter_norm"] = s.sqrt()
        if self.zed_fa_affine is not None:
            a = self.zed_fa_affine.alpha.detach()
            b = self.zed_fa_affine.beta.detach()
            out["zed_fa/calib_alpha_mean"] = a.mean()
            out["zed_fa/calib_alpha_std"] = a.std(unbiased=False)
            out["zed_fa/calib_beta_mean"] = b.mean()
            out["zed_fa/calib_beta_std"] = b.std(unbiased=False)
        return out

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
        zed_disable_fa: bool = False,
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

        if (
            self.zed_fa_prompt_embed is not None
            and self.zed_fa_prompt_proj is not None
            and not zed_disable_fa
        ):
            prm_idx = router_dataset_index if router_dataset_index is not None else dataset_index
            if isinstance(prm_idx, torch.Tensor) and prm_idx.ndim == 1:
                pid = prm_idx.long().clamp(0, self.zed_fa_prompt_embed.num_embeddings - 1)
                route_feat = route_feat + self.zed_fa_prompt_proj(self.zed_fa_prompt_embed(pid))

        route_to_router = route_feat
        if self.zed_router_feat_adapter is not None:
            route_to_router = route_feat + self._router_feat_adapter_scale * self.zed_router_feat_adapter(
                route_feat
            )

        logits_r = self.zed_router.net(route_to_router)
        if self.zed_fa_router_bias is not None and not zed_disable_fa:
            logits_r = logits_r + self.zed_fa_router_bias
        expert_w, expert_logits = self.zed_router._logits_to_weights(logits_r)
        entropy = -(expert_w * (expert_w.clamp_min(1e-9).log())).sum(dim=-1)
        top1 = expert_w.argmax(dim=-1)

        if self.zed_route_gate is not None:
            gate = self.zed_route_gate(route_to_router)
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

        if self.zed_fa_spatial_adapter is not None and not zed_disable_fa:
            combined_delta = combined_delta + self._zed_fa_spatial_scale * self.zed_fa_spatial_adapter(
                combined_delta
            )

        if self.use_persistence_anchor and persistence is not None:
            forecast = persistence + self.calibration_head(combined_delta)
        else:
            forecast = self.calibration_head(combined_delta)

        if self.zed_fa_affine is not None and not zed_disable_fa:
            forecast = self.zed_fa_affine(forecast)

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
