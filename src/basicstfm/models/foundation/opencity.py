"""OpenCity-style spatio-temporal foundation model adapter."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import (
    GraphMixingBlock,
    TransformerBlock,
    ensure_4d,
    load_weights,
)
from basicstfm.registry import MODELS


class OpenCityBlock(nn.Module):
    """Temporal attention followed by graph-aware spatial mixing."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.temporal = TransformerBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.graph = GraphMixingBlock(hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        batch, steps, nodes, dim = x.shape
        h = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        h = self.temporal(h)
        h = h.reshape(batch, nodes, steps, dim).permute(0, 2, 1, 3).contiguous()
        h = self.graph(h, graph)
        return self.norm(h)


@MODELS.register("OpenCityFoundationModel")
class OpenCityFoundationModel(nn.Module):
    """Graph-temporal adapter aligned with OpenCity's encoder design.

    The original OpenCity implementation uses patch embeddings, temporal context
    attention, Laplacian positional encodings, graph convolution, and instance
    denormalization. This adapter keeps the same STFM contract while consuming
    BasicSTFM tensors directly: ``[B, T, N, C]`` in, forecast/reconstruction
    dictionaries out.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        max_seq_len: int = 4096,
        max_num_nodes: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        emb_n = int(max_num_nodes) if max_num_nodes is not None else int(num_nodes)
        self.num_nodes = emb_n
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.node_emb = nn.Embedding(emb_n, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))

        drops = torch.linspace(0.0, float(drop_path), int(num_layers)).tolist()
        self.blocks = nn.ModuleList(
            [
                OpenCityBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                    drop_path=float(drops[idx]),
                )
                for idx in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )
        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.reset_parameters()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if nodes > self.node_emb.num_embeddings:
            raise ValueError(
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows; "
                f"increase num_nodes / max_num_nodes in config."
            )
        if channels > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {channels}")
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
            if mask is not None:
                mask = F.pad(mask, (0, self.input_dim - channels))
            channels = self.input_dim

        h = self.value_proj(x)
        if mask is not None:
            token = self.mask_token.to(dtype=h.dtype, device=h.device)
            h = torch.where(mask[..., :1].bool(), token, h)
        node_ids = torch.arange(nodes, device=x.device)
        h = h + self.node_emb(node_ids)[None, None] + self.temporal_pos[:, :steps]

        for block in self.blocks:
            h = block(h, graph)
        return self.norm(h)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        encoded = self.encode(x, graph=graph, mask=mask)
        out = {"embedding": encoded}
        if mode in {"encode", "embedding"}:
            return out

        if mode in {"forecast", "zero_shot", "prompt_forecast", "both"}:
            summary = encoded[:, -1]
            forecast = self.forecast_head(summary)
            batch, nodes, _ = forecast.shape
            forecast = forecast.reshape(batch, nodes, self.output_len, self.output_dim)
            out["forecast"] = forecast.permute(0, 2, 1, 3).contiguous()

        if mode in {"reconstruct", "reconstruction", "both"}:
            out["reconstruction"] = self.reconstruction_head(encoded)
        return out
