"""Shared layers for built-in spatio-temporal foundation models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Return data as [B, T, N, C]."""

    if x.ndim == 3:
        return x.unsqueeze(-1)
    if x.ndim != 4:
        raise ValueError(f"Expected [B, T, N, C] or [B, T, N], got {tuple(x.shape)}")
    return x


def normalize_adjacency(
    graph: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build a row-normalized adjacency matrix."""

    if graph is None:
        graph = torch.eye(num_nodes, device=device, dtype=dtype)
    else:
        graph = graph.to(device=device, dtype=dtype)
        if graph.ndim != 2 or graph.shape[0] != num_nodes or graph.shape[1] != num_nodes:
            raise ValueError(
                f"Expected graph with shape [{num_nodes}, {num_nodes}], got {tuple(graph.shape)}"
            )
    if add_self_loops:
        graph = graph + torch.eye(num_nodes, device=device, dtype=dtype)
    degree = graph.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return graph / degree


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """Pad a tensor along one dimension so its length is divisible by ``multiple``."""

    length = x.shape[dim]
    pad_len = (multiple - length % multiple) % multiple
    if pad_len == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = x.new_zeros(pad_shape)
    return torch.cat([x, pad], dim=dim), pad_len


class DropPath(nn.Module):
    """Stochastic depth used by lightweight Transformer blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """A compact pre-norm Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = residual + self.drop_path(h)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class GraphMixingBlock(nn.Module):
    """Graph message passing block for [B, T, N, D] hidden states."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape
        adj = normalize_adjacency(graph, nodes, x.device, x.dtype)
        message = torch.einsum("ij,btjd->btid", adj, self.norm(x))
        return x + self.dropout(self.proj(message))


class MemoryPool(nn.Module):
    """Key-value memory used for prompt-based adaptation."""

    def __init__(self, num_slots: int, dim: int) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.keys = nn.Parameter(torch.empty(num_slots, dim))
        self.values = nn.Parameter(torch.empty(num_slots, dim))
        self.query_proj = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.keys, std=0.02)
        nn.init.trunc_normal_(self.values, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        original_shape = query.shape
        query = query.reshape(-1, original_shape[-1])
        query = torch.tanh(self.query_proj(query))
        attn = F.softmax(query @ self.keys.t() / math.sqrt(query.shape[-1]), dim=-1)
        out = attn @ self.values
        return out.reshape(original_shape)


def load_weights(
    module: nn.Module,
    path: str,
    strict: bool = False,
    key: Optional[str] = None,
    map_location: str = "cpu",
) -> Tuple[list[str], list[str]]:
    """Load weights from a common checkpoint format."""

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(str(ckpt_path), map_location=map_location)
    if key:
        state = state[key]
    elif isinstance(state, dict):
        for candidate in ("model_state", "model", "state_dict"):
            if candidate in state:
                state = state[candidate]
                break
    missing, unexpected = module.load_state_dict(state, strict=strict)
    return list(missing), list(unexpected)


def load_filtered_weights(
    module: nn.Module,
    path: str,
    strict: bool = False,
    key: Optional[str] = None,
    map_location: str = "cpu",
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[list[str], list[str]]:
    """Load a filtered subset of weights from a common checkpoint format."""

    state = _read_state_dict(path, key=key, map_location=map_location)
    filtered = {}
    for name, value in state.items():
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        filtered[name] = value
    missing, unexpected = module.load_state_dict(filtered, strict=strict)
    return list(missing), list(unexpected)


def _read_state_dict(
    path: str,
    key: Optional[str] = None,
    map_location: str = "cpu",
) -> dict[str, torch.Tensor]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(str(ckpt_path), map_location=map_location)
    if key:
        state = state[key]
    elif isinstance(state, dict):
        for candidate in ("model_state", "model", "state_dict"):
            if candidate in state:
                state = state[candidate]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {path!r} does not contain a state dict")
    return state
