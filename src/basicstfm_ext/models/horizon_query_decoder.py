"""Query-based decoders from fixed backbone slots to arbitrary horizons."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm_ext.models.sequence_heads import make_sequence_backend
from basicstfm_ext.utils.temporal_queries import HorizonQueryEncoder, temporal_index_features


class CrossAttentionHorizonDecoder(nn.Module):
    """Decode arbitrary forecast horizons by querying fixed backbone slots."""

    def __init__(
        self,
        *,
        memory_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        dataset_embedding_dim: int = 0,
        max_horizon: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.memory_proj = nn.Linear(int(memory_dim), self.hidden_dim)
        self.memory_pos = nn.Linear(4, self.hidden_dim)
        self.query_encoder = HorizonQueryEncoder(
            dim=self.hidden_dim,
            max_horizon=max_horizon,
            dataset_embedding_dim=dataset_embedding_dim,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        target_len: int,
        dataset_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = memory.shape
        features = temporal_index_features(steps, memory.device, memory.dtype)
        memory_h = self.memory_proj(memory) + self.memory_pos(features).unsqueeze(0)
        queries = self.query_encoder(
            target_len=int(target_len),
            batch_size=batch,
            device=memory.device,
            dtype=memory.dtype,
            dataset_embedding=dataset_embedding,
        )
        attended, _ = self.attn(queries, memory_h, memory_h, need_weights=False)
        hidden = self.norm1(queries + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return self.out_proj(hidden), self.feature_proj(hidden.mean(dim=1))


class SequenceHorizonDecoder(nn.Module):
    """Fallback horizon decoder using a temporal sequence backend plus interpolation."""

    def __init__(
        self,
        *,
        memory_dim: int,
        hidden_dim: int,
        output_dim: int,
        backend_type: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.backend = make_sequence_backend(
            backend_type,
            input_dim=int(memory_dim),
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        target_len: int,
        dataset_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dataset_embedding
        hidden = self.backend(memory)
        hidden = hidden.transpose(1, 2)
        hidden = F.interpolate(hidden, size=int(target_len), mode="linear", align_corners=False)
        hidden = self.norm(hidden.transpose(1, 2))
        return self.out_proj(hidden), self.feature_proj(hidden.mean(dim=1))


def build_horizon_decoder(
    *,
    backend_type: str,
    memory_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    dataset_embedding_dim: int = 0,
) -> nn.Module:
    backend = str(backend_type).lower()
    if backend in {"query_attention", "cross_attention", "attention"}:
        return CrossAttentionHorizonDecoder(
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            dataset_embedding_dim=dataset_embedding_dim,
        )
    if backend in {"gru", "mamba", "interp_mlp", "interp"}:
        actual_backend = "interp_mlp" if backend == "interp" else backend
        return SequenceHorizonDecoder(
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            backend_type=actual_backend,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown horizon decoder backend: {backend_type!r}")
