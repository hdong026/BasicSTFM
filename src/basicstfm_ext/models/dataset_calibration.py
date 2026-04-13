"""Lightweight dataset-conditioned calibration for protocol adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from basicstfm_ext.utils.dataset_stats import DescriptorCache, DescriptorConfig, compute_dataset_descriptor


@dataclass
class CalibrationCondition:
    dataset_embedding: torch.Tensor
    raw_descriptor: torch.Tensor
    input_scale: Optional[torch.Tensor] = None
    input_shift: Optional[torch.Tensor] = None
    input_gate: Optional[torch.Tensor] = None
    output_scale: Optional[torch.Tensor] = None
    output_shift: Optional[torch.Tensor] = None
    output_gate: Optional[torch.Tensor] = None


class _DescriptorEncoder(nn.Module):
    def __init__(self, descriptor_dim: int = 15, hidden_dim: int = 64, embedding_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(descriptor_dim)),
            nn.Linear(int(descriptor_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim == 1:
            descriptor = descriptor.unsqueeze(0)
        return self.net(descriptor)


class ProtocolDatasetCalibration(nn.Module):
    """Dataset-stats-driven FiLM/gating for the protocol adapter."""

    def __init__(
        self,
        *,
        input_feature_dim: int,
        output_feature_dim: int,
        embedding_dim: int = 32,
        stats_hidden_dim: int = 64,
        descriptor_momentum: float = 0.9,
        descriptor_cache_path: Optional[str] = None,
        use_graph_stats: bool = True,
        use_spectral_stats: bool = True,
        enable_conditioning: bool = True,
        input_conditioning: bool = True,
        output_conditioning: bool = True,
        input_strength: float = 1.0,
        output_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.enable_conditioning = bool(enable_conditioning)
        self.input_conditioning = bool(input_conditioning)
        self.output_conditioning = bool(output_conditioning)
        self.input_strength = float(input_strength)
        self.output_strength = float(output_strength)
        self.descriptor_cfg = DescriptorConfig(
            use_graph_stats=bool(use_graph_stats),
            use_spectral_stats=bool(use_spectral_stats),
        )
        self.cache = DescriptorCache(
            momentum=descriptor_momentum,
            cache_path=descriptor_cache_path,
        )
        self.encoder = _DescriptorEncoder(
            descriptor_dim=15,
            hidden_dim=stats_hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.input_proj = (
            nn.Linear(int(embedding_dim), int(input_feature_dim) * 2 + 1)
            if self.enable_conditioning and self.input_conditioning
            else None
        )
        self.output_proj = (
            nn.Linear(int(embedding_dim), int(output_feature_dim) * 2 + 1)
            if self.enable_conditioning and self.output_conditioning
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        graph: Optional[torch.Tensor] = None,
        dataset_context: Optional[Mapping[str, Any]] = None,
    ) -> CalibrationCondition:
        descriptor = compute_dataset_descriptor(
            x,
            graph=graph,
            mask=_resolve_mask(dataset_context),
            metadata=_resolve_metadata(dataset_context),
            config=self.descriptor_cfg,
        )
        dataset_name = _resolve_dataset_name(dataset_context)
        if dataset_name:
            descriptor = self.cache.update(dataset_name, descriptor.detach().cpu()).to(
                device=descriptor.device,
                dtype=descriptor.dtype,
            )
        embedding = self.encoder(descriptor).squeeze(0)
        condition = CalibrationCondition(
            dataset_embedding=embedding,
            raw_descriptor=descriptor,
        )
        if self.input_proj is not None:
            condition.input_scale, condition.input_shift, condition.input_gate = self._split_affine(
                self.input_proj(embedding),
                strength=self.input_strength,
            )
        if self.output_proj is not None:
            condition.output_scale, condition.output_shift, condition.output_gate = self._split_affine(
                self.output_proj(embedding),
                strength=self.output_strength,
            )
        return condition

    @staticmethod
    def _split_affine(value: torch.Tensor, *, strength: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, shift, gate = torch.split(value, [((value.shape[0] - 1) // 2), ((value.shape[0] - 1) // 2), 1])
        return strength * scale, strength * shift, torch.sigmoid(gate).squeeze(-1)


def _resolve_dataset_name(dataset_context: Optional[Mapping[str, Any]]) -> Optional[str]:
    if dataset_context is None:
        return None
    value = dataset_context.get("dataset_name")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return None if value is None else str(value)


def _resolve_metadata(dataset_context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if dataset_context is None:
        return {}
    metadata = dataset_context.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _resolve_mask(dataset_context: Optional[Mapping[str, Any]]) -> Optional[torch.Tensor]:
    if dataset_context is None:
        return None
    if dataset_context.get("x_mask") is not None:
        return dataset_context.get("x_mask")  # type: ignore[return-value]
    if dataset_context.get("mask") is not None:
        return dataset_context.get("mask")  # type: ignore[return-value]
    return None
