"""UniFlow-style STFM adapter: OpenCity-style graph-temporal stack + coupling residuals (lite)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import ensure_4d, load_weights
from basicstfm.models.foundation.opencity import OpenCityBlock
from basicstfm.registry import MODELS


class FlowCoupling(nn.Module):
    """Cheap additive coupling on channel halves (budget ``flow`` accent, not full NICE/NF)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("hidden_dim must be even for FlowCoupling")
        h = dim // 2
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([a, b + self.net(a)], dim=-1)


@MODELS.register("UniFlowFoundationModel")
class UniFlowFoundationModel(nn.Module):
    """Graph-temporal encoder with coupling stacks (UniFlow budget adapter for BasicSTFM)."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 192,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        flow_coupling_layers: int = 2,
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
        self.flow_couplings = nn.ModuleList(
            [FlowCoupling(self.hidden_dim) for _ in range(int(flow_coupling_layers))]
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
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows"
            )
        if channels > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {channels}")
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
            if mask is not None:
                mask = F.pad(mask, (0, self.input_dim - channels))

        h = self.value_proj(x)
        if mask is not None:
            token = self.mask_token.to(dtype=h.dtype, device=h.device)
            h = torch.where(mask[..., :1].bool(), token, h)
        node_ids = torch.arange(nodes, device=x.device)
        h = h + self.node_emb(node_ids)[None, None] + self.temporal_pos[:, :steps]

        for block in self.blocks:
            h = block(h, graph)
        for fc in self.flow_couplings:
            h = fc(h)
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
