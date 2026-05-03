"""ST-MambaSync-style baseline: bidirectional depthwise temporal conv + graph (no ``mamba_ssm`` CUDA kernel)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import GraphMixingBlock, ensure_4d, load_weights
from basicstfm.registry import MODELS


class BiSyncMambaLite(nn.Module):
    """Lightweight bidirectional gated depthwise conv along time (Mamba-inspired sync)."""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw_f = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.dw_b = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,N,D]."""
        b, t, n, d = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * n, d, t)
        yf = self.dw_f(y)
        yb = self.dw_b(y.flip(-1)).flip(-1)
        y = self.pw(yf + yb)
        y = y * torch.sigmoid(self.gate(y))
        y = self.dropout(y)
        y = y.reshape(b, n, d, t).permute(0, 3, 1, 2).contiguous()
        return x + y


class MambaSyncBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.temporal = BiSyncMambaLite(hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.graph = GraphMixingBlock(hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.temporal(x)
        x = self.graph(x, graph)
        return self.norm(x)


@MODELS.register("STMambaSyncFoundationModel")
class STMambaSyncFoundationModel(nn.Module):
    """Encoder advertised as **ST-MambaSync** in configs; implementation = BiSyncMambaLite + graph (see README)."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 96,
        num_layers: int = 4,
        mamba_kernel_size: int = 5,
        dropout: float = 0.1,
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
        self.blocks = nn.ModuleList(
            [
                MambaSyncBlock(
                    hidden_dim=self.hidden_dim,
                    kernel_size=int(mamba_kernel_size),
                    dropout=float(dropout),
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
            raise ValueError(
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows"
            )
        if channels > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {channels}")
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))

        h = self.value_proj(x)
        node_ids = torch.arange(nodes, device=x.device)
        h = h + self.node_emb(node_ids)[None, None] + self.temporal_pos[:, :steps]
        for blk in self.blocks:
            h = blk(h, graph)
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
