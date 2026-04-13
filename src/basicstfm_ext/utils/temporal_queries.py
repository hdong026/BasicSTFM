"""Temporal query helpers for protocol-adapter models."""

from __future__ import annotations

import math

import torch
from torch import nn


def temporal_index_features(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return cheap continuous features for a temporal axis."""

    steps = torch.arange(int(length), device=device, dtype=dtype)
    if int(length) <= 1:
        normalized = torch.zeros_like(steps)
    else:
        normalized = steps / float(length - 1)
    angle = 2.0 * math.pi * normalized
    reciprocal = 1.0 / (steps + 1.0)
    return torch.stack(
        [normalized, torch.sin(angle), torch.cos(angle), reciprocal],
        dim=-1,
    )


class LearnedTemporalQueries(nn.Module):
    """Combine learned query slots with cheap temporal features."""

    def __init__(
        self,
        *,
        num_queries: int,
        dim: int,
        max_positions: int = 512,
        dataset_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.dim = int(dim)
        self.max_positions = int(max_positions)
        self.query_bank = nn.Parameter(torch.randn(self.num_queries, self.dim) * 0.02)
        self.feature_proj = nn.Linear(4, self.dim)
        self.position_emb = nn.Embedding(self.max_positions, self.dim)
        self.dataset_proj = (
            nn.Linear(int(dataset_embedding_dim), self.dim)
            if int(dataset_embedding_dim) > 0
            else None
        )

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        dataset_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = torch.arange(
            self.num_queries,
            device=device,
            dtype=torch.long,
        ).clamp_max(self.max_positions - 1)
        features = temporal_index_features(self.num_queries, device, dtype)
        queries = (
            self.query_bank.to(device=device, dtype=dtype)
            + self.position_emb(positions).to(dtype=dtype)
            + self.feature_proj(features)
        )
        if dataset_embedding is not None and self.dataset_proj is not None:
            if dataset_embedding.ndim == 1:
                dataset_embedding = dataset_embedding.unsqueeze(0)
            bias = self.dataset_proj(dataset_embedding.to(device=device, dtype=dtype))
            if bias.shape[0] == 1 and int(batch_size) > 1:
                bias = bias.expand(batch_size, -1)
            queries = queries.unsqueeze(0) + bias.unsqueeze(1)
        else:
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        return queries


class HorizonQueryEncoder(nn.Module):
    """Generate arbitrary-length forecast queries with harmonic features."""

    def __init__(
        self,
        *,
        dim: int,
        max_horizon: int = 512,
        dataset_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_horizon = int(max_horizon)
        self.position_emb = nn.Embedding(self.max_horizon, self.dim)
        self.feature_proj = nn.Linear(4, self.dim)
        self.dataset_proj = (
            nn.Linear(int(dataset_embedding_dim), self.dim)
            if int(dataset_embedding_dim) > 0
            else None
        )

    def forward(
        self,
        *,
        target_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        dataset_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = torch.arange(int(target_len), device=device, dtype=torch.long).clamp_max(
            self.max_horizon - 1
        )
        features = temporal_index_features(int(target_len), device, dtype)
        queries = self.position_emb(positions).to(dtype=dtype) + self.feature_proj(features)
        if dataset_embedding is not None and self.dataset_proj is not None:
            if dataset_embedding.ndim == 1:
                dataset_embedding = dataset_embedding.unsqueeze(0)
            bias = self.dataset_proj(dataset_embedding.to(device=device, dtype=dtype))
            if bias.shape[0] == 1 and int(batch_size) > 1:
                bias = bias.expand(batch_size, -1)
            queries = queries.unsqueeze(0) + bias.unsqueeze(1)
        else:
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        return queries
