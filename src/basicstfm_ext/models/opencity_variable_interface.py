"""Wrapper-based variable-interface extension around the OpenCity backbone."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from basicstfm.models.foundation.common import ensure_4d
from basicstfm.models.foundation.opencity import OpenCityFoundationModel
from basicstfm.registry import MODELS
from basicstfm.utils.checkpoint import torch_load
from basicstfm_ext.losses.orthogonality import OrthogonalityLoss
from basicstfm_ext.losses.redundancy import RedundancyPenalty
from basicstfm_ext.models.dataset_conditioning import build_conditioner
from basicstfm_ext.models.domain_regularizers import GRLDomainClassifier
from basicstfm_ext.models.sequence_heads import VariableInputInterfaceHead, VariableOutputInterfaceHead
from basicstfm_ext.utils.shape_debug import ensure_shape

LOGGER = logging.getLogger(__name__)


@MODELS.register("OpenCityVariableInterfaceWrapper")
class OpenCityVariableInterfaceWrapper(nn.Module):
    """Wrap OpenCity with dataset-conditioned variable-length interface heads.

    The wrapper preserves a fixed-length OpenCity backbone while learning
    lightweight dataset-aware heads on both sides:

        arbitrary runtime window
            -> input interface head
            -> fixed OpenCity backbone
            -> output interface head
            -> arbitrary forecast horizon
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        backbone_cfg: Optional[Dict[str, Any]] = None,
        interface_cfg: Optional[Dict[str, Any]] = None,
        conditioning_cfg: Optional[Dict[str, Any]] = None,
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
        interface_cfg = dict(interface_cfg or {})
        conditioning_cfg = dict(conditioning_cfg or {})
        regularization_cfg = dict(regularization_cfg or {})

        self.backbone_input_len = int(backbone_cfg.pop("input_len", 12))
        self.backbone_output_len = int(backbone_cfg.pop("output_len", 12))
        self.backbone_input_dim = int(backbone_cfg.pop("input_dim", self.runtime_input_dim))
        self.backbone_output_dim = int(backbone_cfg.pop("output_dim", self.runtime_output_dim))

        self.backbone = OpenCityFoundationModel(
            num_nodes=int(backbone_cfg.pop("num_nodes", self.runtime_num_nodes)),
            input_dim=self.backbone_input_dim,
            output_dim=self.backbone_output_dim,
            input_len=self.backbone_input_len,
            output_len=self.backbone_output_len,
            **backbone_cfg,
        )

        head_type = str(interface_cfg.get("head_type", "gru"))
        head_hidden_dim = int(interface_cfg.get("hidden_dim", 32))
        bottleneck_dim = int(interface_cfg.get("bottleneck_dim", 16))
        head_layers = int(interface_cfg.get("num_layers", 1))
        head_dropout = float(interface_cfg.get("dropout", 0.0))
        stronger_input = bool(interface_cfg.get("stronger_input_conditioning_than_output", True))
        self.enable_private_branch = bool(interface_cfg.get("enable_private_branch", True))

        self.input_head = VariableInputInterfaceHead(
            runtime_input_dim=self.runtime_input_dim,
            backbone_input_dim=self.backbone_input_dim,
            backbone_input_len=self.backbone_input_len,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=head_layers,
            backend_type=head_type,
            dropout=head_dropout,
            stronger_conditioning=stronger_input,
        )
        self.output_head = VariableOutputInterfaceHead(
            backbone_output_dim=self.backbone_output_dim,
            runtime_output_dim=self.runtime_output_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=head_layers,
            backend_type=head_type,
            dropout=head_dropout,
            stronger_conditioning=not stronger_input,
        )

        variant = str(interface_cfg.get("variant", "C")).upper()
        condition_rank = int(conditioning_cfg.get("rank", max(2, bottleneck_dim // 2)))
        source_datasets = conditioning_cfg.get("source_datasets")
        self.conditioning = build_conditioner(
            variant=variant,
            input_feature_dim=head_hidden_dim,
            output_feature_dim=head_hidden_dim,
            backbone_input_dim=self.backbone_input_dim,
            target_output_dim=self.runtime_output_dim,
            rank=condition_rank,
            embedding_dim=int(conditioning_cfg.get("embedding_dim", 32)),
            stats_hidden_dim=int(conditioning_cfg.get("stats_hidden_dim", 64)),
            source_datasets=source_datasets,
            use_graph_stats=bool(conditioning_cfg.get("use_graph_stats", True)),
            use_spectral_stats=bool(conditioning_cfg.get("use_spectral_stats", True)),
            descriptor_momentum=float(conditioning_cfg.get("descriptor_momentum", 0.9)),
            descriptor_cache_path=conditioning_cfg.get("descriptor_cache_path"),
            instance_refinement=bool(conditioning_cfg.get("instance_refinement", False)),
            zero_shot_init_method=str(conditioning_cfg.get("zero_shot_init_method", "dataset_stats")),
            temperature=float(conditioning_cfg.get("temperature", 0.5)),
            hidden_dim=int(conditioning_cfg.get("hyper_hidden_dim", 96)),
        )
        self.source_datasets = tuple(str(item) for item in (source_datasets or ()))
        self.source_name_to_index = {name: idx for idx, name in enumerate(self.source_datasets)}

        self.lambda_adv = float(regularization_cfg.get("lambda_adv", 0.0))
        self.lambda_ortho = float(regularization_cfg.get("lambda_ortho", 0.0))
        self.lambda_red = float(regularization_cfg.get("lambda_red", 0.0))
        self.domain_classifier = (
            GRLDomainClassifier(
                input_dim=int(backbone_cfg.get("hidden_dim", 128)),
                num_domains=max(1, len(self.source_datasets)),
                hidden_dim=int(regularization_cfg.get("domain_hidden_dim", 64)),
                grl_coeff=float(regularization_cfg.get("grl_coeff", 1.0)),
            )
            if self.lambda_adv > 0.0 and self.source_datasets
            else None
        )
        self.orthogonality = OrthogonalityLoss()
        self.redundancy = RedundancyPenalty()
        self.debug_shapes = bool(interface_cfg.get("debug_shapes", False))

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
        x = ensure_4d(x)
        ensure_shape("x", x, 4)
        conditioning = self.conditioning(x, graph=graph, dataset_context=dataset_context)
        adapted_x, input_private = self.input_head(x, conditioning)
        if self.debug_shapes:
            LOGGER.debug("input -> adapted_x %s", tuple(adapted_x.shape))

        # The fixed OpenCity backbone expects masks aligned with its native
        # length/channel space. The wrapper keeps dataset masks for descriptor
        # estimation, but does not forward them directly unless they have been
        # explicitly adapted.
        shared_encoded = self.backbone.encode(adapted_x, graph=graph, mask=None)
        shared_feat = shared_encoded.mean(dim=(1, 2))
        summary = shared_encoded[:, -1]
        backbone_forecast = self.backbone.forecast_head(summary)
        batch, nodes, _ = backbone_forecast.shape
        backbone_forecast = backbone_forecast.reshape(
            batch,
            nodes,
            self.backbone_output_len,
            self.backbone_output_dim,
        ).permute(0, 2, 1, 3).contiguous()

        forecast, output_private = self.output_head(
            backbone_forecast,
            conditioning=conditioning,
            target_len=_resolve_target_len(dataset_context, self.runtime_output_len),
        )
        private_feat = 0.5 * (input_private + output_private)
        if not self.enable_private_branch:
            private_feat = torch.zeros_like(private_feat)
        aux_losses = self._compute_aux_losses(
            shared_feat=shared_feat,
            private_feat=private_feat,
            dataset_context=dataset_context,
        )
        out: Dict[str, torch.Tensor] = {
            "forecast": forecast,
            "shared_feat": shared_feat,
            "private_feat": private_feat,
            "dataset_embedding": conditioning.dataset_embedding,
            "aux_losses": aux_losses,
        }
        if conditioning.prototype_weights is not None:
            out["prototype_weights"] = conditioning.prototype_weights
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
        return list(missing), list(unexpected)

    def _compute_aux_losses(
        self,
        *,
        shared_feat: torch.Tensor,
        private_feat: torch.Tensor,
        dataset_context: Optional[Mapping[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        aux: Dict[str, torch.Tensor] = {}
        domain_index = self._resolve_domain_index(
            dataset_context,
            shared_feat.device,
            batch_size=int(shared_feat.shape[0]),
        )
        if self.domain_classifier is not None and domain_index is not None:
            domain_loss = self.domain_classifier(shared_feat, domain_index)
            if domain_loss is not None:
                aux["domain_adv"] = self.lambda_adv * domain_loss
        if self.lambda_ortho > 0.0:
            aux["orthogonality"] = self.lambda_ortho * self.orthogonality(shared_feat, private_feat)
        if self.lambda_red > 0.0:
            joint = torch.cat([shared_feat, private_feat], dim=-1)
            aux["redundancy"] = self.lambda_red * self.redundancy(joint)
        return aux

    def _resolve_domain_index(
        self,
        dataset_context: Optional[Mapping[str, Any]],
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


def _resolve_target_len(
    dataset_context: Optional[Mapping[str, Any]],
    default: int,
) -> int:
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
