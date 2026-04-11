"""UniST-style prompt-empowered spatio-temporal foundation model adapter."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import (
    MemoryPool,
    TransformerBlock,
    ensure_4d,
    load_filtered_weights,
    load_weights,
)
from basicstfm.registry import MODELS


class UniSTPrompt(nn.Module):
    """Spatial and temporal memory prompts inspired by UniST."""

    def __init__(self, hidden_dim: int, spatial_slots: int, temporal_slots: int) -> None:
        super().__init__()
        self.spatial_memory = MemoryPool(spatial_slots, hidden_dim)
        self.temporal_memory = MemoryPool(temporal_slots, hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, D]
        spatial_query = h.mean(dim=1)
        temporal_query = h.mean(dim=2)
        spatial_prompt = self.spatial_memory(spatial_query)[:, None]
        temporal_prompt = self.temporal_memory(temporal_query)[:, :, None]
        spatial_prompt = spatial_prompt.expand(-1, h.shape[1], -1, -1)
        temporal_prompt = temporal_prompt.expand(-1, -1, h.shape[2], -1)
        return self.fuse(torch.cat([spatial_prompt, temporal_prompt], dim=-1))


@MODELS.register("UniSTFoundationModel")
class UniSTFoundationModel(nn.Module):
    """Prompt-empowered masked autoencoding and forecasting adapter.

    UniST uses stage-1 masked spatio-temporal pretraining and stage-2
    knowledge-guided prompt tuning. The adapter exposes the same idea through a
    reusable encoder/decoder backbone plus trainable spatial and temporal memory
    prompts that can be unfrozen independently during prompt-tuning stages.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        use_prompt: bool = True,
        num_memory_spatial: int = 128,
        num_memory_temporal: int = 128,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_prompt = bool(use_prompt)

        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.node_emb = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(encoder_layers))
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.hidden_dim)

        self.prompt = UniSTPrompt(
            hidden_dim=self.hidden_dim,
            spatial_slots=int(num_memory_spatial),
            temporal_slots=int(num_memory_temporal),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(decoder_layers))
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )
        self.reset_parameters()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def load_backbone_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load the stage-1 backbone while leaving prompt memories stage-local."""

        return load_filtered_weights(
            self,
            path,
            strict=strict,
            exclude_prefixes=("prompt.",),
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

    def _tokenize(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if nodes > self.num_nodes:
            raise ValueError(f"Expected at most {self.num_nodes} nodes, got {nodes}")
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
        return h.reshape(batch, steps * nodes, self.hidden_dim), batch, steps, nodes

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del graph
        tokens, batch, steps, nodes = self._tokenize(x, mask=mask)
        h = tokens
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_norm(h)
        return h.reshape(batch, steps, nodes, self.hidden_dim)

    def _decode(self, encoded: torch.Tensor, use_prompt: bool) -> torch.Tensor:
        h = encoded
        if use_prompt:
            h = h + self.prompt(h)
        batch, steps, nodes, dim = h.shape
        tokens = h.reshape(batch, steps * nodes, dim)
        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)
        return tokens.reshape(batch, steps, nodes, dim)

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

        prompt_enabled = self.use_prompt or mode == "prompt_forecast"
        decoded = self._decode(encoded, use_prompt=prompt_enabled)

        if mode in {"forecast", "zero_shot", "prompt_forecast", "both"}:
            summary = decoded[:, -1]
            forecast = self.forecast_head(summary)
            batch, nodes, _ = forecast.shape
            forecast = forecast.reshape(batch, nodes, self.output_len, self.output_dim)
            out["forecast"] = forecast.permute(0, 2, 1, 3).contiguous()

        if mode in {"reconstruct", "reconstruction", "both"}:
            out["reconstruction"] = self.reconstruction_head(decoded)
        return out
