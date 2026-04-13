"""Protocol-adapter wrapper around the fixed-window OpenCity backbone."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import ensure_4d
from basicstfm.models.foundation.opencity import OpenCityFoundationModel
from basicstfm.registry import MODELS
from basicstfm.utils.checkpoint import torch_load
from basicstfm_ext.losses.distillation_loss import DistillationLoss
from basicstfm_ext.losses.identity_loss import IdentityLoss
from basicstfm_ext.losses.orthogonality import OrthogonalityLoss
from basicstfm_ext.losses.redundancy import RedundancyPenalty
from basicstfm_ext.models.dataset_calibration import CalibrationCondition, ProtocolDatasetCalibration
from basicstfm_ext.models.domain_regularizers import GRLDomainClassifier
from basicstfm_ext.models.horizon_query_decoder import build_horizon_decoder
from basicstfm_ext.models.latent_resampler import build_latent_resampler
from basicstfm_ext.models.teacher_distillation import OpenCityTeacher
from basicstfm_ext.utils.protocol_debug import debug_tensor_shape, ensure_protocol_shape

LOGGER = logging.getLogger(__name__)


@dataclass
class ProtocolAdapterState:
    adapted: torch.Tensor
    base: torch.Tensor
    residual: torch.Tensor
    adapter_feat: torch.Tensor


class InputProtocolAdapter(nn.Module):
    """Residual protocol adapter from variable input windows to fixed backbone slots."""

    def __init__(
        self,
        *,
        runtime_input_dim: int,
        backbone_input_dim: int,
        backbone_input_len: int,
        hidden_dim: int,
        bottleneck_dim: int,
        backend_type: str = "cross_attention",
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-3,
        calibration_enabled: bool = True,
        dataset_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.runtime_input_dim = int(runtime_input_dim)
        self.backbone_input_dim = int(backbone_input_dim)
        self.backbone_input_len = int(backbone_input_len)
        self.hidden_dim = int(hidden_dim)
        self.calibration_enabled = bool(calibration_enabled)
        self.resampler = build_latent_resampler(
            backend_type=backend_type,
            input_dim=self.runtime_input_dim,
            hidden_dim=self.hidden_dim,
            num_slots=self.backbone_input_len,
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
            dataset_embedding_dim=int(dataset_embedding_dim),
        )
        self.residual_proj = nn.Linear(self.hidden_dim, self.backbone_input_dim)
        self.adapter_proj = nn.Linear(self.hidden_dim, int(bottleneck_dim))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        conditioning: CalibrationCondition,
    ) -> ProtocolAdapterState:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if channels > self.runtime_input_dim:
            raise ValueError(
                f"Expected at most runtime_input_dim={self.runtime_input_dim}, got {channels}"
            )
        if channels < self.runtime_input_dim:
            x = F.pad(x, (0, self.runtime_input_dim - channels))

        base = align_protocol_input(
            x,
            target_len=self.backbone_input_len,
            target_dim=self.backbone_input_dim,
        )
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.runtime_input_dim)
        embedding = _expand_dataset_embedding(conditioning.dataset_embedding, batch * nodes)
        hidden = self.resampler(seq, dataset_embedding=embedding)
        residual = self.residual_proj(hidden)
        if self.calibration_enabled:
            residual = _apply_affine(residual, conditioning.input_scale, conditioning.input_shift)
            residual = _apply_gate(residual, conditioning.input_gate)
        adapted = base.permute(0, 2, 1, 3).reshape(
            batch * nodes,
            self.backbone_input_len,
            self.backbone_input_dim,
        )
        adapted = adapted + self.residual_scale.to(dtype=adapted.dtype) * residual
        adapted = adapted.reshape(batch, nodes, self.backbone_input_len, self.backbone_input_dim)
        adapted = adapted.permute(0, 2, 1, 3).contiguous()
        adapter_feat = self.adapter_proj(hidden.mean(dim=1)).reshape(batch, nodes, -1).mean(dim=1)
        return ProtocolAdapterState(
            adapted=adapted,
            base=base,
            residual=residual.reshape(batch, nodes, self.backbone_input_len, self.backbone_input_dim).permute(
                0,
                2,
                1,
                3,
            ).contiguous(),
            adapter_feat=adapter_feat,
        )


class OutputQueryDecoder(nn.Module):
    """Residual query decoder from fixed backbone forecast slots to arbitrary horizons."""

    def __init__(
        self,
        *,
        backbone_output_dim: int,
        runtime_output_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        backend_type: str = "query_attention",
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-3,
        calibration_enabled: bool = True,
        dataset_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone_output_dim = int(backbone_output_dim)
        self.runtime_output_dim = int(runtime_output_dim)
        self.calibration_enabled = bool(calibration_enabled)
        self.decoder = build_horizon_decoder(
            backend_type=backend_type,
            memory_dim=self.backbone_output_dim,
            hidden_dim=int(hidden_dim),
            output_dim=self.runtime_output_dim,
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
            dataset_embedding_dim=int(dataset_embedding_dim),
        )
        self.adapter_proj = nn.Linear(int(hidden_dim), int(bottleneck_dim))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        fixed_forecast: torch.Tensor,
        *,
        target_len: int,
        conditioning: CalibrationCondition,
    ) -> ProtocolAdapterState:
        fixed_forecast = ensure_4d(fixed_forecast)
        batch, steps, nodes, channels = fixed_forecast.shape
        base = align_protocol_output(
            fixed_forecast,
            target_len=int(target_len),
            target_dim=self.runtime_output_dim,
        )
        memory = fixed_forecast.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        embedding = _expand_dataset_embedding(conditioning.dataset_embedding, batch * nodes)
        residual, hidden_feat = self.decoder(
            memory,
            target_len=int(target_len),
            dataset_embedding=embedding,
        )
        if self.calibration_enabled:
            residual = _apply_affine(residual, conditioning.output_scale, conditioning.output_shift)
            residual = _apply_gate(residual, conditioning.output_gate)
        forecast = base.permute(0, 2, 1, 3).reshape(batch * nodes, int(target_len), self.runtime_output_dim)
        forecast = forecast + self.residual_scale.to(dtype=forecast.dtype) * residual
        forecast = forecast.reshape(batch, nodes, int(target_len), self.runtime_output_dim)
        forecast = forecast.permute(0, 2, 1, 3).contiguous()
        adapter_feat = self.adapter_proj(hidden_feat).reshape(batch, nodes, -1).mean(dim=1)
        return ProtocolAdapterState(
            adapted=forecast,
            base=base,
            residual=residual.reshape(batch, nodes, int(target_len), self.runtime_output_dim).permute(
                0,
                2,
                1,
                3,
            ).contiguous(),
            adapter_feat=adapter_feat,
        )


@MODELS.register("OpenCityProtocolAdapterWrapper")
class OpenCityProtocolAdapterWrapper(nn.Module):
    """Length-generalized protocol adapter around the vendored OpenCity backbone."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        backbone_cfg: Optional[Dict[str, Any]] = None,
        adapter_cfg: Optional[Dict[str, Any]] = None,
        calibration_cfg: Optional[Dict[str, Any]] = None,
        distill_cfg: Optional[Dict[str, Any]] = None,
        regularization_cfg: Optional[Dict[str, Any]] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        self.runtime_num_nodes = int(num_nodes)
        self.runtime_input_dim = int(input_dim)
        self.runtime_output_dim = int(output_dim)
        self.runtime_input_len = int(input_len)
        self.runtime_output_len = int(output_len)

        backbone_cfg = dict(backbone_cfg or {})
        adapter_cfg = dict(adapter_cfg or {})
        calibration_cfg = dict(calibration_cfg or {})
        distill_cfg = dict(distill_cfg or {})
        regularization_cfg = dict(regularization_cfg or {})

        self.backbone_input_len = int(backbone_cfg.pop("input_len", 12))
        self.backbone_output_len = int(backbone_cfg.pop("output_len", 12))
        self.backbone_input_dim = int(backbone_cfg.pop("input_dim", self.runtime_input_dim))
        self.backbone_output_dim = int(backbone_cfg.pop("output_dim", self.runtime_output_dim))
        self.backbone_hidden_dim = int(backbone_cfg.get("hidden_dim", 128))

        backbone_kwargs = {
            "num_nodes": int(backbone_cfg.pop("num_nodes", self.runtime_num_nodes)),
            "input_dim": self.backbone_input_dim,
            "output_dim": self.backbone_output_dim,
            "input_len": self.backbone_input_len,
            "output_len": self.backbone_output_len,
            **backbone_cfg,
        }
        self.backbone = OpenCityFoundationModel(**backbone_kwargs)

        adapter_hidden = int(adapter_cfg.get("hidden_dim", 64))
        adapter_bottleneck = int(adapter_cfg.get("bottleneck_dim", 16))
        adapter_heads = int(adapter_cfg.get("num_heads", 4))
        adapter_layers = int(adapter_cfg.get("num_layers", 1))
        adapter_dropout = float(adapter_cfg.get("dropout", 0.0))
        residual_scale_init = float(adapter_cfg.get("residual_scale_init", 1e-3))
        enable_conditioning = bool(adapter_cfg.get("enable_conditioning", True))
        self.debug_shapes = bool(adapter_cfg.get("debug_shapes", False))

        calibration_embedding_dim = int(calibration_cfg.get("embedding_dim", 32))
        self.calibration = ProtocolDatasetCalibration(
            input_feature_dim=self.backbone_input_dim,
            output_feature_dim=self.runtime_output_dim,
            embedding_dim=calibration_embedding_dim,
            stats_hidden_dim=int(calibration_cfg.get("stats_hidden_dim", 64)),
            descriptor_momentum=float(calibration_cfg.get("descriptor_momentum", 0.9)),
            descriptor_cache_path=calibration_cfg.get("descriptor_cache_path"),
            use_graph_stats=bool(calibration_cfg.get("use_graph_stats", True)),
            use_spectral_stats=bool(calibration_cfg.get("use_spectral_stats", True)),
            enable_conditioning=bool(calibration_cfg.get("enable_conditioning", True)),
            input_conditioning=bool(calibration_cfg.get("input_conditioning", True)),
            output_conditioning=bool(calibration_cfg.get("output_conditioning", True)),
            input_strength=float(calibration_cfg.get("input_strength", 1.0)),
            output_strength=float(calibration_cfg.get("output_strength", 0.5)),
        )
        self.input_protocol_adapter = InputProtocolAdapter(
            runtime_input_dim=self.runtime_input_dim,
            backbone_input_dim=self.backbone_input_dim,
            backbone_input_len=self.backbone_input_len,
            hidden_dim=adapter_hidden,
            bottleneck_dim=adapter_bottleneck,
            backend_type=str(adapter_cfg.get("input_backend", "cross_attention")),
            num_heads=adapter_heads,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
            residual_scale_init=residual_scale_init,
            calibration_enabled=enable_conditioning,
            dataset_embedding_dim=calibration_embedding_dim,
        )
        self.output_query_decoder = OutputQueryDecoder(
            backbone_output_dim=self.backbone_output_dim,
            runtime_output_dim=self.runtime_output_dim,
            hidden_dim=adapter_hidden,
            bottleneck_dim=adapter_bottleneck,
            backend_type=str(adapter_cfg.get("output_backend", "query_attention")),
            num_heads=adapter_heads,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
            residual_scale_init=residual_scale_init,
            calibration_enabled=enable_conditioning,
            dataset_embedding_dim=calibration_embedding_dim,
        )

        teacher_source = str(distill_cfg.get("teacher_source", "backbone"))
        teacher_pretrained_path = distill_cfg.get("teacher_pretrained_path")
        teacher_strict = bool(distill_cfg.get("teacher_strict_load", False))
        self.teacher = OpenCityTeacher(
            source=teacher_source,
            pretrained_path=teacher_pretrained_path,
            strict_load=teacher_strict,
            backbone_kwargs=backbone_kwargs,
            backbone=self.backbone,
        )
        self.lambda_distill_matched = float(distill_cfg.get("lambda_distill_matched", 0.0))
        self.lambda_identity_in = float(distill_cfg.get("lambda_identity_in", 0.0))
        self.lambda_identity_out = float(distill_cfg.get("lambda_identity_out", 0.0))
        self.distill_only_when_matched = bool(distill_cfg.get("distill_only_when_matched", True))
        self.identity_only_when_matched = bool(distill_cfg.get("identity_only_when_matched", True))
        self.identity_loss = IdentityLoss()
        self.distillation_loss = DistillationLoss(mode=str(distill_cfg.get("distill_mode", "mse")))

        self.source_datasets = tuple(
            str(item) for item in calibration_cfg.get("source_datasets", regularization_cfg.get("source_datasets", []))
        )
        self.source_name_to_index = {name: idx for idx, name in enumerate(self.source_datasets)}
        self.lambda_adv = float(regularization_cfg.get("lambda_adv", 0.0))
        self.lambda_ortho = float(regularization_cfg.get("lambda_ortho", 0.0))
        self.lambda_red = float(regularization_cfg.get("lambda_red", 0.0))
        self.domain_classifier = (
            GRLDomainClassifier(
                input_dim=self.backbone_hidden_dim,
                num_domains=max(1, len(self.source_datasets)),
                hidden_dim=int(regularization_cfg.get("domain_hidden_dim", 64)),
                grl_coeff=float(regularization_cfg.get("grl_coeff", 1.0)),
            )
            if self.lambda_adv > 0.0 and self.source_datasets
            else None
        )
        self.orthogonality = OrthogonalityLoss()
        self.redundancy = RedundancyPenalty()

        if pretrained_path:
            self.load_full_weights(pretrained_path, strict=strict_load)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
        dataset_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        del mask
        x = ensure_4d(x)
        ensure_protocol_shape("x", x, 4)
        runtime_target_len = _resolve_target_len(dataset_context, self.runtime_output_len)
        matched_input = int(x.shape[1]) == self.backbone_input_len
        matched_output = int(runtime_target_len) == self.backbone_output_len

        conditioning = self.calibration(x, graph=graph, dataset_context=dataset_context)
        input_state = self.input_protocol_adapter(x, conditioning=conditioning)
        debug_tensor_shape(LOGGER, "protocol_input", input_state.adapted, enabled=self.debug_shapes)

        shared_encoded = self.backbone.encode(input_state.adapted, graph=graph, mask=None)
        shared_feat = shared_encoded.mean(dim=(1, 2))
        summary = shared_encoded[:, -1]
        fixed_backbone_forecast = self.backbone.forecast_head(summary)
        batch, nodes, _ = fixed_backbone_forecast.shape
        fixed_backbone_forecast = fixed_backbone_forecast.reshape(
            batch,
            nodes,
            self.backbone_output_len,
            self.backbone_output_dim,
        ).permute(0, 2, 1, 3).contiguous()
        output_state = self.output_query_decoder(
            fixed_backbone_forecast,
            target_len=runtime_target_len,
            conditioning=conditioning,
        )
        debug_tensor_shape(LOGGER, "protocol_output", output_state.adapted, enabled=self.debug_shapes)

        teacher_forecast = None
        if matched_input and matched_output and (self.lambda_distill_matched > 0.0 or mode == "teacher"):
            teacher_input = align_protocol_input(
                x,
                target_len=self.backbone_input_len,
                target_dim=self.backbone_input_dim,
            )
            teacher_forecast = align_protocol_output(
                self.teacher.forecast(backbone=self.backbone, x=teacher_input, graph=graph),
                target_len=runtime_target_len,
                target_dim=self.runtime_output_dim,
            )

        adapter_feat = 0.5 * (input_state.adapter_feat + output_state.adapter_feat)
        aux_losses = self._compute_aux_losses(
            shared_feat=shared_feat,
            adapter_feat=adapter_feat,
            input_state=input_state,
            output_state=output_state,
            teacher_forecast=teacher_forecast,
            matched_input=matched_input,
            matched_output=matched_output,
            dataset_context=dataset_context,
        )

        out: Dict[str, torch.Tensor] = {
            "forecast": output_state.adapted,
            "backbone_forecast": output_state.base,
            "fixed_backbone_forecast": fixed_backbone_forecast,
            "shared_feat": shared_feat,
            "adapter_feat": adapter_feat,
            "dataset_embedding": conditioning.dataset_embedding,
            "aux_losses": aux_losses,
        }
        if teacher_forecast is not None:
            out["teacher_forecast"] = teacher_forecast
        if mode in {"encode", "embedding"}:
            out["embedding"] = shared_encoded
        return out

    def load_full_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        state = self._read_state_dict(path)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        return list(missing), list(unexpected)

    def load_backbone_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        state = self._read_state_dict(path)
        backbone_state = {}
        if any(key.startswith("backbone.") for key in state):
            for key, value in state.items():
                if key.startswith("backbone."):
                    backbone_state[key[len("backbone.") :]] = value
        else:
            for key, value in state.items():
                if key in self.backbone.state_dict():
                    backbone_state[key] = value
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=strict)
        self.teacher.sync_from_backbone(self.backbone)
        return list(missing), list(unexpected)

    def _compute_aux_losses(
        self,
        *,
        shared_feat: torch.Tensor,
        adapter_feat: torch.Tensor,
        input_state: ProtocolAdapterState,
        output_state: ProtocolAdapterState,
        teacher_forecast: Optional[torch.Tensor],
        matched_input: bool,
        matched_output: bool,
        dataset_context: Optional[Mapping[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        aux: Dict[str, torch.Tensor] = {}
        if self.lambda_identity_in > 0.0 and (matched_input or not self.identity_only_when_matched):
            aux["identity_in"] = self.lambda_identity_in * self.identity_loss(
                input_state.adapted,
                input_state.base,
            )
        if self.lambda_identity_out > 0.0 and (matched_output or not self.identity_only_when_matched):
            aux["identity_out"] = self.lambda_identity_out * self.identity_loss(
                output_state.adapted,
                output_state.base,
            )
        if (
            self.lambda_distill_matched > 0.0
            and teacher_forecast is not None
            and ((matched_input and matched_output) or not self.distill_only_when_matched)
        ):
            aux["distill_matched"] = self.lambda_distill_matched * self.distillation_loss(
                output_state.adapted,
                teacher_forecast,
            )
        if self.lambda_ortho > 0.0:
            aux["orthogonality"] = self.lambda_ortho * self.orthogonality(shared_feat, adapter_feat)
        if self.lambda_red > 0.0:
            aux["redundancy"] = self.lambda_red * self.redundancy(
                torch.cat([shared_feat, adapter_feat], dim=-1)
            )
        domain_index = self._resolve_domain_index(
            dataset_context,
            device=shared_feat.device,
            batch_size=int(shared_feat.shape[0]),
        )
        if self.domain_classifier is not None and domain_index is not None:
            aux["domain_adv"] = self.lambda_adv * self.domain_classifier(shared_feat, domain_index)
        return aux

    def _resolve_domain_index(
        self,
        dataset_context: Optional[Mapping[str, Any]],
        *,
        device: torch.device,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if not self.source_datasets or dataset_context is None:
            return None
        value = dataset_context.get("dataset_index")
        if isinstance(value, torch.Tensor):
            labels = value.reshape(-1).to(device=device, dtype=torch.long)
            if labels.numel() == 1 and batch_size > 1:
                labels = labels.expand(batch_size)
            elif labels.numel() != batch_size:
                raise ValueError(
                    f"dataset_index has {labels.numel()} labels for batch_size={batch_size}"
                )
            return labels
        dataset_name = dataset_context.get("dataset_name")
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if dataset_name else None
        if dataset_name is None:
            return None
        index = self.source_name_to_index.get(str(dataset_name))
        if index is None:
            return None
        return torch.full((batch_size,), index, device=device, dtype=torch.long)

    @staticmethod
    def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
        state = torch_load(path, map_location="cpu")
        if isinstance(state, dict):
            for candidate in ("model_state", "model", "state_dict"):
                if candidate in state and isinstance(state[candidate], dict):
                    state = state[candidate]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint at {path!r} does not contain a state dict")
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = key[7:] if key.startswith("module.") else key
            normalized[clean_key] = value
        return normalized


def align_protocol_input(x: torch.Tensor, *, target_len: int, target_dim: int) -> torch.Tensor:
    x = ensure_4d(x)
    batch, steps, nodes, channels = x.shape
    if channels < int(target_dim):
        x = F.pad(x, (0, int(target_dim) - channels))
    elif channels > int(target_dim):
        x = x[..., : int(target_dim)]
    if steps == int(target_len):
        return x.contiguous()
    seq = x.permute(0, 2, 3, 1).reshape(batch * nodes, int(target_dim), steps)
    seq = F.interpolate(seq, size=int(target_len), mode="linear", align_corners=False)
    return seq.reshape(batch, nodes, int(target_dim), int(target_len)).permute(0, 3, 1, 2).contiguous()


def align_protocol_output(x: torch.Tensor, *, target_len: int, target_dim: int) -> torch.Tensor:
    x = ensure_4d(x)
    batch, steps, nodes, channels = x.shape
    if channels < int(target_dim):
        x = F.pad(x, (0, int(target_dim) - channels))
    elif channels > int(target_dim):
        x = x[..., : int(target_dim)]
    if steps == int(target_len):
        return x.contiguous()
    seq = x.permute(0, 2, 3, 1).reshape(batch * nodes, int(target_dim), steps)
    seq = F.interpolate(seq, size=int(target_len), mode="linear", align_corners=False)
    return seq.reshape(batch, nodes, int(target_dim), int(target_len)).permute(0, 3, 1, 2).contiguous()


def _apply_affine(
    value: torch.Tensor,
    scale: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
) -> torch.Tensor:
    if scale is not None:
        value = value * (1.0 + scale.to(device=value.device, dtype=value.dtype).view(1, 1, -1))
    if shift is not None:
        value = value + shift.to(device=value.device, dtype=value.dtype).view(1, 1, -1)
    return value


def _apply_gate(value: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return value
    return value * gate.to(device=value.device, dtype=value.dtype).view(1, 1, 1)


def _expand_dataset_embedding(
    embedding: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.shape[0] == 1 and int(batch_size) > 1:
        embedding = embedding.expand(int(batch_size), -1)
    return embedding


def _resolve_target_len(dataset_context: Optional[Mapping[str, Any]], default: int) -> int:
    if dataset_context is None:
        return int(default)
    metadata = dataset_context.get("metadata")
    if not isinstance(metadata, Mapping):
        return int(default)
    value = metadata.get("target_len", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
