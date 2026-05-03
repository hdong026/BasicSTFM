"""STPFormer expert: multi-scale temporal pyramid + graph mixing (task-specific forecaster)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import GraphMixingBlock, TransformerBlock, ensure_4d, load_weights
from basicstfm.registry import MODELS


class TemporalPyramidMix(nn.Module):
    """Multi-branch average pooling along time + fusion (lite pyramid)."""

    def __init__(self, hidden_dim: int, branches: Tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.branches = tuple(int(b) for b in branches)
        self.fuse = nn.Linear(hidden_dim * len(self.branches), hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, D]
        b, t, n, d = h.shape
        outs = []
        for k in self.branches:
            if k <= 1:
                y = h
            else:
                hp = h.permute(0, 2, 3, 1).reshape(b * n, d, t)
                yp = F.avg_pool1d(hp, kernel_size=k, stride=k, ceil_mode=False)
                yp = F.interpolate(yp, size=t, mode="linear", align_corners=False)
                y = yp.reshape(b, n, d, t).permute(0, 3, 1, 2).contiguous()
            outs.append(y)
        stacked = torch.cat(outs, dim=-1)
        return self.fuse(stacked)


class STPFormerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        pyramid_branches: Tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.pyramid = TemporalPyramidMix(hidden_dim, branches=pyramid_branches)
        self.temporal = TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout=dropout)
        self.graph = GraphMixingBlock(hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        h = h + self.pyramid(h)
        batch, steps, nodes, dim = h.shape
        ht = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        ht = self.temporal(ht)
        h = ht.reshape(batch, nodes, steps, dim).permute(0, 2, 1, 3).contiguous()
        h = self.graph(h, graph)
        return self.norm(h)


@MODELS.register("STPFormerExpert")
class STPFormerExpert(nn.Module):
    """Spatial-temporal pyramid Transformer expert for [B, T, N, C] traffic tensors."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        pyramid_branches: Tuple[int, ...] = (1, 2, 4),
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
        branches = tuple(int(x) for x in pyramid_branches)
        self.layers = nn.ModuleList(
            [
                STPFormerLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                    pyramid_branches=branches,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )
        self.reset_parameters()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del mask
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if nodes > self.node_emb.num_embeddings:
            raise ValueError(f"Input has {nodes} nodes but embeddings have {self.node_emb.num_embeddings}")
        if channels > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {channels}")
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))

        h = self.value_proj(x)
        node_ids = torch.arange(nodes, device=x.device)
        h = h + self.node_emb(node_ids)[None, None] + self.temporal_pos[:, :steps]
        for layer in self.layers:
            h = layer(h, graph)
        return self.norm(h)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        del mask
        encoded = self.encode(x, graph=graph)
        out: dict = {"embedding": encoded}
        if mode in {"encode", "embedding"}:
            return out

        if mode in {"forecast", "zero_shot", "prompt_forecast", "both"}:
            summary = encoded[:, -1]
            forecast = self.forecast_head(summary)
            b, n, _ = forecast.shape
            forecast = forecast.reshape(b, n, self.output_len, self.output_dim)
            out["forecast"] = forecast.permute(0, 2, 1, 3).contiguous()
        return out
