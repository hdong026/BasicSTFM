"""Dataset-conditioned parameter generators for interface heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm_ext.utils.dataset_stats import DescriptorCache, DescriptorConfig, compute_dataset_descriptor


@dataclass
class HeadConditioning:
    dataset_embedding: torch.Tensor
    raw_descriptor: torch.Tensor
    input_scale: Optional[torch.Tensor] = None
    input_shift: Optional[torch.Tensor] = None
    input_gate: Optional[torch.Tensor] = None
    output_scale: Optional[torch.Tensor] = None
    output_shift: Optional[torch.Tensor] = None
    output_gate: Optional[torch.Tensor] = None
    input_down: Optional[torch.Tensor] = None
    input_up: Optional[torch.Tensor] = None
    output_down: Optional[torch.Tensor] = None
    output_up: Optional[torch.Tensor] = None
    prototype_weights: Optional[torch.Tensor] = None


class DatasetStatsEncoder(nn.Module):
    """Project deterministic dataset descriptors into a learned embedding space."""

    def __init__(
        self,
        descriptor_dim: int = 15,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim == 1:
            descriptor = descriptor.unsqueeze(0)
        return self.net(descriptor)


class BaseDatasetConditioner(nn.Module):
    """Shared descriptor/caching logic for the A/B/C variants."""

    def __init__(
        self,
        *,
        input_feature_dim: int,
        output_feature_dim: int,
        backbone_input_dim: int,
        target_output_dim: int,
        rank: int,
        embedding_dim: int = 32,
        stats_hidden_dim: int = 64,
        source_datasets: Optional[Sequence[str]] = None,
        use_graph_stats: bool = True,
        use_spectral_stats: bool = True,
        descriptor_momentum: float = 0.9,
        descriptor_cache_path: Optional[str] = None,
        instance_refinement: bool = False,
        zero_shot_init_method: str = "dataset_stats",
    ) -> None:
        super().__init__()
        self.input_feature_dim = int(input_feature_dim)
        self.output_feature_dim = int(output_feature_dim)
        self.backbone_input_dim = int(backbone_input_dim)
        self.target_output_dim = int(target_output_dim)
        self.rank = int(rank)
        self.embedding_dim = int(embedding_dim)
        self.source_datasets = tuple(str(item) for item in (source_datasets or ()))
        self.source_name_to_index = {name: idx for idx, name in enumerate(self.source_datasets)}
        self.descriptor_cfg = DescriptorConfig(
            use_graph_stats=bool(use_graph_stats),
            use_spectral_stats=bool(use_spectral_stats),
        )
        self.zero_shot_init_method = str(zero_shot_init_method)
        if self.zero_shot_init_method != "dataset_stats":
            raise ValueError(
                "Only zero_shot_init_method='dataset_stats' is currently supported"
            )
        self.descriptor_cache = DescriptorCache(
            momentum=descriptor_momentum,
            cache_path=descriptor_cache_path,
        )
        self.stats_encoder = DatasetStatsEncoder(
            descriptor_dim=15,
            hidden_dim=stats_hidden_dim,
            embedding_dim=self.embedding_dim,
        )
        self.instance_refinement = bool(instance_refinement)
        self.instance_refiner = (
            nn.Sequential(
                nn.Linear(15, self.embedding_dim),
                nn.Tanh(),
            )
            if self.instance_refinement
            else None
        )
        if self.source_datasets:
            self.register_buffer(
                "source_descriptor_bank",
                torch.zeros(len(self.source_datasets), 15),
            )
            self.register_buffer(
                "source_descriptor_seen",
                torch.zeros(len(self.source_datasets), dtype=torch.bool),
            )

    def describe(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor],
        dataset_context: Optional[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_name = _resolve_dataset_name(dataset_context)
        descriptor = compute_dataset_descriptor(
            x,
            graph=graph,
            mask=_resolve_mask(dataset_context),
            metadata=_resolve_metadata(dataset_context),
            config=self.descriptor_cfg,
        )
        if dataset_name:
            descriptor = self._cache_descriptor(dataset_name, descriptor)
        embedding = self.stats_encoder(descriptor).squeeze(0)
        if self.instance_refiner is not None:
            embedding = embedding + self.instance_refiner(descriptor.unsqueeze(0)).squeeze(0)
        return descriptor, embedding

    def _cache_descriptor(self, dataset_name: str, descriptor: torch.Tensor) -> torch.Tensor:
        cached = self.descriptor_cache.update(dataset_name, descriptor.detach().cpu())
        if dataset_name in self.source_name_to_index:
            index = self.source_name_to_index[dataset_name]
            with torch.no_grad():
                self.source_descriptor_bank[index].copy_(cached)
                self.source_descriptor_seen[index] = True
        return cached.to(device=descriptor.device, dtype=descriptor.dtype)


class PrototypeMixtureHead(BaseDatasetConditioner):
    """Variant A: build conditioning parameters as prototype mixtures."""

    def __init__(
        self,
        *,
        temperature: float = 0.5,
        num_prototypes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.temperature = float(temperature)
        prototype_count = int(num_prototypes or max(1, len(self.source_datasets) or 4))
        self.prototype_keys = nn.Parameter(torch.randn(prototype_count, self.embedding_dim) * 0.02)
        self.input_scale_bank = nn.Parameter(torch.zeros(prototype_count, self.input_feature_dim))
        self.input_shift_bank = nn.Parameter(torch.zeros(prototype_count, self.input_feature_dim))
        self.input_gate_bank = nn.Parameter(torch.zeros(prototype_count, 1))
        self.input_down_bank = nn.Parameter(
            torch.randn(prototype_count, self.input_feature_dim, self.rank) * 0.02
        )
        self.input_up_bank = nn.Parameter(
            torch.randn(prototype_count, self.rank, self.backbone_input_dim) * 0.02
        )
        self.output_scale_bank = nn.Parameter(torch.zeros(prototype_count, self.output_feature_dim))
        self.output_shift_bank = nn.Parameter(torch.zeros(prototype_count, self.output_feature_dim))
        self.output_gate_bank = nn.Parameter(torch.zeros(prototype_count, 1))
        self.output_down_bank = nn.Parameter(
            torch.randn(prototype_count, self.output_feature_dim, self.rank) * 0.02
        )
        self.output_up_bank = nn.Parameter(
            torch.randn(prototype_count, self.rank, self.target_output_dim) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        graph: Optional[torch.Tensor] = None,
        dataset_context: Optional[Mapping[str, Any]] = None,
    ) -> HeadConditioning:
        descriptor, embedding = self.describe(x, graph, dataset_context)
        key_bank = self.prototype_keys
        active = torch.arange(key_bank.shape[0], device=embedding.device)
        if self.source_datasets and bool(self.source_descriptor_seen.any()):
            active = self.source_descriptor_seen.nonzero(as_tuple=False).flatten().to(embedding.device)
            source_descriptors = self.source_descriptor_bank.index_select(
                0,
                active.to(self.source_descriptor_bank.device),
            ).to(
                device=embedding.device,
                dtype=embedding.dtype,
            )
            key_bank = self.stats_encoder(source_descriptors)
        logits = F.normalize(embedding, dim=0) @ F.normalize(key_bank, dim=-1).t()
        weights = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        input_scale_bank = self.input_scale_bank.index_select(0, active)
        input_shift_bank = self.input_shift_bank.index_select(0, active)
        input_gate_bank = self.input_gate_bank.index_select(0, active)
        input_down_bank = self.input_down_bank.index_select(0, active)
        input_up_bank = self.input_up_bank.index_select(0, active)
        output_scale_bank = self.output_scale_bank.index_select(0, active)
        output_shift_bank = self.output_shift_bank.index_select(0, active)
        output_gate_bank = self.output_gate_bank.index_select(0, active)
        output_down_bank = self.output_down_bank.index_select(0, active)
        output_up_bank = self.output_up_bank.index_select(0, active)
        return HeadConditioning(
            dataset_embedding=embedding,
            raw_descriptor=descriptor,
            input_scale=weights @ input_scale_bank,
            input_shift=weights @ input_shift_bank,
            input_gate=torch.sigmoid(weights @ input_gate_bank).squeeze(-1),
            output_scale=weights @ output_scale_bank,
            output_shift=weights @ output_shift_bank,
            output_gate=torch.sigmoid(weights @ output_gate_bank).squeeze(-1),
            input_down=torch.einsum("p,phr->hr", weights, input_down_bank),
            input_up=torch.einsum("p,pro->ro", weights, input_up_bank),
            output_down=torch.einsum("p,phr->hr", weights, output_down_bank),
            output_up=torch.einsum("p,pro->ro", weights, output_up_bank),
            prototype_weights=weights,
        )


class HyperHeadGenerator(BaseDatasetConditioner):
    """Variant B: generate lightweight low-rank head adapters from z_D."""

    def __init__(self, *, hidden_dim: int = 96, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        total = (
            self.input_feature_dim * 2
            + 1
            + self.output_feature_dim * 2
            + 1
            + self.input_feature_dim * self.rank
            + self.rank * self.backbone_input_dim
            + self.output_feature_dim * self.rank
            + self.rank * self.target_output_dim
        )
        self.hyper = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        graph: Optional[torch.Tensor] = None,
        dataset_context: Optional[Mapping[str, Any]] = None,
    ) -> HeadConditioning:
        descriptor, embedding = self.describe(x, graph, dataset_context)
        vector = self.hyper(embedding.unsqueeze(0)).squeeze(0)
        cursor = 0

        def take(size: int) -> torch.Tensor:
            nonlocal cursor
            chunk = vector[cursor : cursor + size]
            cursor += size
            return chunk

        input_scale = take(self.input_feature_dim)
        input_shift = take(self.input_feature_dim)
        input_gate = torch.sigmoid(take(1)).squeeze(0)
        output_scale = take(self.output_feature_dim)
        output_shift = take(self.output_feature_dim)
        output_gate = torch.sigmoid(take(1)).squeeze(0)
        input_down = take(self.input_feature_dim * self.rank).reshape(self.input_feature_dim, self.rank)
        input_up = take(self.rank * self.backbone_input_dim).reshape(self.rank, self.backbone_input_dim)
        output_down = take(self.output_feature_dim * self.rank).reshape(self.output_feature_dim, self.rank)
        output_up = take(self.rank * self.target_output_dim).reshape(self.rank, self.target_output_dim)

        return HeadConditioning(
            dataset_embedding=embedding,
            raw_descriptor=descriptor,
            input_scale=input_scale,
            input_shift=input_shift,
            input_gate=input_gate,
            output_scale=output_scale,
            output_shift=output_shift,
            output_gate=output_gate,
            input_down=input_down,
            input_up=input_up,
            output_down=output_down,
            output_up=output_up,
        )


class UniversalModulatedHead(BaseDatasetConditioner):
    """Variant C: shared heads with lightweight dataset-conditioned modulation."""

    def __init__(self, *, hidden_dim: int = 64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        total = self.input_feature_dim * 2 + 1 + self.output_feature_dim * 2 + 1
        self.modulator = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        graph: Optional[torch.Tensor] = None,
        dataset_context: Optional[Mapping[str, Any]] = None,
    ) -> HeadConditioning:
        descriptor, embedding = self.describe(x, graph, dataset_context)
        vector = self.modulator(embedding.unsqueeze(0)).squeeze(0)
        cursor = 0

        def take(size: int) -> torch.Tensor:
            nonlocal cursor
            chunk = vector[cursor : cursor + size]
            cursor += size
            return chunk

        return HeadConditioning(
            dataset_embedding=embedding,
            raw_descriptor=descriptor,
            input_scale=take(self.input_feature_dim),
            input_shift=take(self.input_feature_dim),
            input_gate=torch.sigmoid(take(1)).squeeze(0),
            output_scale=take(self.output_feature_dim),
            output_shift=take(self.output_feature_dim),
            output_gate=torch.sigmoid(take(1)).squeeze(0),
        )


def build_conditioner(
    variant: str,
    *,
    input_feature_dim: int,
    output_feature_dim: int,
    backbone_input_dim: int,
    target_output_dim: int,
    rank: int,
    embedding_dim: int,
    stats_hidden_dim: int,
    source_datasets: Optional[Sequence[str]],
    use_graph_stats: bool,
    use_spectral_stats: bool,
    descriptor_momentum: float,
    descriptor_cache_path: Optional[str],
    instance_refinement: bool,
    zero_shot_init_method: str,
    temperature: float = 0.5,
    hidden_dim: int = 96,
) -> BaseDatasetConditioner:
    variant = str(variant).upper()
    common = dict(
        input_feature_dim=input_feature_dim,
        output_feature_dim=output_feature_dim,
        backbone_input_dim=backbone_input_dim,
        target_output_dim=target_output_dim,
        rank=rank,
        embedding_dim=embedding_dim,
        stats_hidden_dim=stats_hidden_dim,
        source_datasets=source_datasets,
        use_graph_stats=use_graph_stats,
        use_spectral_stats=use_spectral_stats,
        descriptor_momentum=descriptor_momentum,
        descriptor_cache_path=descriptor_cache_path,
        instance_refinement=instance_refinement,
        zero_shot_init_method=zero_shot_init_method,
    )
    if variant == "A":
        return PrototypeMixtureHead(temperature=temperature, **common)
    if variant == "B":
        return HyperHeadGenerator(hidden_dim=hidden_dim, **common)
    if variant == "C":
        return UniversalModulatedHead(hidden_dim=hidden_dim, **common)
    raise ValueError(f"Unknown interface variant: {variant!r}")


def _resolve_dataset_name(dataset_context: Optional[Mapping[str, Any]]) -> Optional[str]:
    if dataset_context is None:
        return None
    value = dataset_context.get("dataset_name")
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return None if value is None else str(value)


def _resolve_mask(dataset_context: Optional[Mapping[str, Any]]) -> Optional[torch.Tensor]:
    if dataset_context is None:
        return None
    value = dataset_context.get("x_mask")
    if value is None:
        value = dataset_context.get("mask")
    return None if value is None else value


def _resolve_metadata(dataset_context: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if dataset_context is None:
        return None
    value = dataset_context.get("metadata")
    return value if isinstance(value, Mapping) else None
