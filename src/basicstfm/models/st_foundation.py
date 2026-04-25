"""Tiny spatio-temporal foundation model backbone."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicstfm.registry import MODELS


@MODELS.register("TinySTFoundationModel")
class TinySTFoundationModel(nn.Module):
    """A small Transformer backbone for pretrain-finetune experiments.

    Input shape is [B, T, N, C]. The model encodes each node's temporal context
    with shared Transformer weights, node embeddings, and temporal positions.
    It exposes two heads:
      - ``forecast``: future-window prediction.
      - ``reconstruction``: same-window masked reconstruction.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_num_nodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        emb_n = int(max_num_nodes) if max_num_nodes is not None else int(num_nodes)
        self.num_nodes = emb_n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_emb = nn.Embedding(emb_n, hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_len * output_dim),
        )
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape
        if nodes > self.node_emb.num_embeddings:
            raise ValueError(
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows; "
                f"increase num_nodes / max_num_nodes in config."
            )
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")

        h = self.input_proj(x)
        if mask is not None:
            token = self.mask_token.to(dtype=h.dtype, device=h.device)
            h = torch.where(mask[..., :1].bool(), token, h)

        node_ids = torch.arange(nodes, device=x.device)
        node_emb = self.node_emb(node_ids)[None, None, :, :]
        h = h + node_emb + self.temporal_pos[:, :steps]

        h = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.hidden_dim)
        h = self.encoder(h)
        h = self.norm(h)
        h = h.reshape(batch, nodes, steps, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
        return h

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        # The graph is accepted to keep the model signature compatible with
        # graph-aware custom backbones.
        del graph
        encoded = self.encode(x, mask=mask)
        out = {"embedding": encoded}

        if mode in {"forecast", "both"}:
            summary = encoded[:, -1]
            forecast = self.forecast_head(summary)
            batch, nodes, _ = forecast.shape
            forecast = forecast.reshape(batch, nodes, self.output_len, self.output_dim)
            out["forecast"] = forecast.permute(0, 2, 1, 3).contiguous()

        if mode in {"reconstruct", "reconstruction", "both"}:
            out["reconstruction"] = self.reconstruction_head(encoded)

        return out
