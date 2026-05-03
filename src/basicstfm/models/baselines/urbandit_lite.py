"""UrbanDiT-lite: patchified urban forecaster (DiT-style patching; deterministic, no diffusion loop)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import GraphMixingBlock, TransformerBlock, ensure_4d, load_weights
from basicstfm.registry import MODELS


@MODELS.register("UrbanDiTLiteFoundationModel")
class UrbanDiTLiteFoundationModel(nn.Module):
    """Patch tokens along time, small Transformer + graph mix (budget ``DiT-like`` STFM)."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 192,
        patch_size: int = 2,
        dropout: float = 0.1,
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
        self.patch_size = int(patch_size)

        ptokens = (self.input_len + self.patch_size - 1) // self.patch_size
        patch_in = self.patch_size * self.input_dim
        self.patch_embed = nn.Linear(patch_in, self.hidden_dim)
        self.node_emb = nn.Embedding(emb_n, self.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, ptokens, 1, self.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "temporal": TransformerBlock(
                            self.hidden_dim,
                            num_heads,
                            ffn_dim,
                            dropout=dropout,
                        ),
                        "graph": GraphMixingBlock(self.hidden_dim, dropout=dropout),
                        "norm": nn.LayerNorm(self.hidden_dim),
                    }
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,N,C] -> [B,N,P,H] tokens."""
        b, t, n, c = x.shape
        pad_len = (-t) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        t2 = x.shape[1]
        p = t2 // self.patch_size
        x = x[:, : p * self.patch_size]
        patches = x.reshape(b, p, self.patch_size, n, c).permute(0, 3, 1, 2, 4)
        patches = patches.reshape(b * n, p, self.patch_size * c)
        tok = self.patch_embed(patches)
        tok = tok.reshape(b, n, p, self.hidden_dim).permute(0, 2, 1, 3)
        tok = tok + self.pos[:, :p] + self.node_emb(torch.arange(n, device=x.device))[None, None, :, :]
        return tok

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del mask
        x = ensure_4d(x)
        _, _, nodes, channels = x.shape
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
        if nodes > self.node_emb.num_embeddings:
            raise ValueError(
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows"
            )
        return self._patchify(x)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        del mask
        h = self.encode(x, graph=graph)
        b, p, n, d = h.shape
        for blk in self.blocks:
            hp = h.permute(0, 2, 1, 3).reshape(b * n, p, d)
            hp = blk["temporal"](hp)
            h = hp.reshape(b, n, p, d).permute(0, 2, 1, 3).contiguous()
            h = blk["graph"](h, graph)
            h = blk["norm"](h)

        h = self.norm(h)
        summary = h.mean(dim=1)
        forecast = self.forecast_head(summary)
        forecast = forecast.reshape(b, n, self.output_len, self.output_dim)
        out = {"embedding": h, "forecast": forecast.permute(0, 2, 1, 3).contiguous()}
        if mode in {"encode", "embedding"}:
            return {"embedding": h}
        return out
