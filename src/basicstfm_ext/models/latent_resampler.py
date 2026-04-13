"""Latent temporal resamplers for fixed-slot protocol adaptation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm_ext.models.sequence_heads import make_sequence_backend
from basicstfm_ext.utils.temporal_queries import LearnedTemporalQueries, temporal_index_features


class CrossAttentionLatentResampler(nn.Module):
    """Resample arbitrary-length sequences into fixed latent slots."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_heads: int,
        dropout: float = 0.0,
        dataset_embedding_dim: int = 0,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.value_proj = nn.Linear(int(input_dim), self.hidden_dim)
        self.position_proj = nn.Linear(4, self.hidden_dim)
        self.position_emb = nn.Embedding(int(max_positions), self.hidden_dim)
        self.query_encoder = LearnedTemporalQueries(
            num_queries=self.num_slots,
            dim=self.hidden_dim,
            max_positions=max_positions,
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

    def forward(
        self,
        seq: torch.Tensor,
        *,
        dataset_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, steps, _ = seq.shape
        positions = torch.arange(steps, device=seq.device, dtype=torch.long).clamp_max(
            self.position_emb.num_embeddings - 1
        )
        features = temporal_index_features(steps, seq.device, seq.dtype)
        values = (
            self.value_proj(seq)
            + self.position_emb(positions).to(dtype=seq.dtype).unsqueeze(0)
            + self.position_proj(features).unsqueeze(0)
        )
        queries = self.query_encoder(
            batch_size=batch,
            device=seq.device,
            dtype=seq.dtype,
            dataset_embedding=dataset_embedding,
        )
        attended, _ = self.attn(queries, values, values, need_weights=False)
        hidden = self.norm1(queries + attended)
        return self.norm2(hidden + self.ffn(hidden))


class SequenceLatentResampler(nn.Module):
    """GRU/Mamba/interpolation-based latent resampler."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_slots: int,
        backend_type: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.backend = make_sequence_backend(
            backend_type,
            input_dim=int(input_dim),
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        seq: torch.Tensor,
        *,
        dataset_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del dataset_embedding
        hidden = self.backend(seq)
        hidden = hidden.transpose(1, 2)
        hidden = F.interpolate(hidden, size=self.num_slots, mode="linear", align_corners=False)
        hidden = hidden.transpose(1, 2)
        return self.norm(hidden)


def build_latent_resampler(
    *,
    backend_type: str,
    input_dim: int,
    hidden_dim: int,
    num_slots: int,
    num_heads: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    dataset_embedding_dim: int = 0,
) -> nn.Module:
    backend = str(backend_type).lower()
    if backend in {"cross_attention", "attention", "resampler"}:
        return CrossAttentionLatentResampler(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_heads=num_heads,
            dropout=dropout,
            dataset_embedding_dim=dataset_embedding_dim,
        )
    if backend in {"gru", "mamba", "interp_mlp", "interp"}:
        actual_backend = "interp_mlp" if backend == "interp" else backend
        return SequenceLatentResampler(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            backend_type=actual_backend,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown latent resampler backend: {backend_type!r}")
