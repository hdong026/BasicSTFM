"""DPM v2: SRD-STFM with graph-aware Stage-I stable trunk (DPMV2Backbone)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.diffusion_mechanism_learner import DiffusionMechanismLearner
from basicstfm.models.dpm_v2_stable_trunk_encoder import StableTrunkEncoderV2
from basicstfm.models.foundation.common import ensure_4d, load_filtered_weights, load_weights
from basicstfm.models.fusion_predictor import FusionPredictor
from basicstfm.models.residual_constructor import ResidualConstructor
from basicstfm.models.residual_event_encoder import ResidualEventEncoder
from basicstfm.utils.diffusion_debug import build_diffusion_debug_payload
from basicstfm.registry import MODELS


@MODELS.register("DPMV2Backbone")
class DPMV2Backbone(nn.Module):
    """Same three-stage DPM forward as SRDSTFMBackbone, but Stage-I uses ``StableTrunkEncoderV2``."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        residual_mode: str = "forecast",
        fusion_mode: str = "additive",
        use_frequency_branch: bool = True,
        stable_summary_mode: str = "attention",
        stable_mixer_layers: int = 2,
        stable_mixer_kernel_size: int = 3,
        stable_coarse_scale: int = 4,
        stable_frequency_low_ratio: float = 0.3,
        stable_frequency_num_low_bins: Optional[int] = None,
        use_stable_graph_context: bool = True,
        stable_graph_num_layers: int = 1,
        stable_graph_share_weights: bool = False,
        num_datasets: int = 1,
        diffusion_enabled: bool = True,
        use_inertia_gate: bool = True,
        use_attenuation_gate: bool = True,
        use_calibration_head: bool = True,
        detach_stable_for_residual: bool = True,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.hidden_dim = int(hidden_dim)
        self.residual_mode = str(residual_mode)
        self.fusion_mode = str(fusion_mode)
        self.diffusion_enabled = bool(diffusion_enabled)
        self.use_calibration_head = bool(use_calibration_head)
        self.use_stable_graph_context = bool(use_stable_graph_context)
        self.stable_graph_num_layers = int(stable_graph_num_layers)
        self.stable_graph_share_weights = bool(stable_graph_share_weights)

        self.stable_trunk = StableTrunkEncoderV2(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_len=self.output_len,
            output_dim=self.output_dim,
            coarse_scale=int(stable_coarse_scale),
            use_frequency_branch=bool(use_frequency_branch),
            summary_mode=str(stable_summary_mode),
            stable_mixer_layers=int(stable_mixer_layers),
            stable_mixer_kernel_size=int(stable_mixer_kernel_size),
            frequency_low_ratio=float(stable_frequency_low_ratio),
            frequency_num_low_bins=stable_frequency_num_low_bins,
            use_stable_graph_context=self.use_stable_graph_context,
            stable_graph_num_layers=self.stable_graph_num_layers,
            stable_graph_share_weights=self.stable_graph_share_weights,
        )
        self.residual_constructor = ResidualConstructor(
            mode=self.residual_mode,
            detach_stable=bool(detach_stable_for_residual),
        )
        self.residual_event_encoder = ResidualEventEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )
        self.diffusion_mechanism_learner = DiffusionMechanismLearner(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_datasets=int(num_datasets),
            use_inertia_gate=bool(use_inertia_gate),
            use_attenuation_gate=bool(use_attenuation_gate),
        )
        self.fusion_predictor = FusionPredictor(
            output_dim=self.output_dim,
            fusion_mode=self.fusion_mode,
        )
        if self.use_calibration_head:
            self.calibration_head = nn.Linear(self.output_dim, self.output_dim)
            self._init_calibration_identity()
        else:
            self.calibration_head = nn.Identity()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def _init_calibration_identity(self) -> None:
        nn.init.zeros_(self.calibration_head.weight)
        nn.init.zeros_(self.calibration_head.bias)
        eye = torch.eye(self.output_dim)
        with torch.no_grad():
            self.calibration_head.weight.copy_(eye)

    def _pad_input(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        x = ensure_4d(x)
        # num_nodes 仅作配置/元数据占位，不限制前向的 N（支持跨图迁移到更大/更小目标图）。
        if x.shape[-1] > self.input_dim:
            raise ValueError(f"Expected at most {self.input_dim} channels, got {x.shape[-1]}")

        channels = x.shape[-1]
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
            if mask is not None:
                mask = F.pad(mask, (0, self.input_dim - channels))
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.zeros_like(x))
        return x, mask, channels

    def _pad_target(self, target: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if target is None:
            return None
        target = ensure_4d(target)
        if target.shape[-1] > self.output_dim:
            target = target[..., : self.output_dim]
        if target.shape[-1] < self.output_dim:
            target = F.pad(target, (0, self.output_dim - target.shape[-1]))
        return target

    def load_backbone_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        return load_filtered_weights(
            self,
            path,
            strict=strict,
            exclude_prefixes=(
                "fusion_predictor.",
                "calibration_head.",
            ),
        )

    def load_stable_trunk_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        return load_filtered_weights(
            self,
            path,
            strict=strict,
            include_prefixes=("stable_trunk.",),
        )

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
        dataset_index: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x, padded_mask, original_channels = self._pad_input(x, mask)
        target = self._pad_target(target)

        stable = self.stable_trunk(x, mask=padded_mask, graph=graph)
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
        residual_input = residual_pack["residual_input"]

        event = self.residual_event_encoder(
            residual_input,
            graph=graph,
        )

        diffusion_active = self.diffusion_enabled and mode not in {"stable_pretrain", "stable_only"}
        if diffusion_active:
            diffusion = self.diffusion_mechanism_learner(
                event_latent=event["event_latent"],
                event_score=event["event_score"],
                graph=graph,
                output_len=self.output_len,
                dataset_index=dataset_index,
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
            }

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
        forecast = self.calibration_head(fused["forecast"])

        if original_channels < self.output_dim:
            forecast = forecast[..., :original_channels]
            stable_forecast = stable_forecast[..., :original_channels]
            residual_forecast = residual_forecast[..., :original_channels]
            stable_reconstruction = stable_reconstruction[..., :original_channels]
            residual_input = residual_input[..., :original_channels]

        outputs: dict[str, torch.Tensor] = {
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
            "fusion_weight": fused["fusion_weight"],
        }

        debug_payload = build_diffusion_debug_payload(outputs)
        for key, stats in debug_payload.items():
            for stat_name, stat_value in stats.items():
                outputs[f"debug/{key}/{stat_name}"] = forecast.new_tensor(stat_value)

        return outputs
