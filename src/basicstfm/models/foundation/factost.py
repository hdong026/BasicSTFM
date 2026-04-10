"""FactoST-style factorized spatio-temporal foundation model adapter."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from basicstfm.models.foundation.common import (
    TransformerBlock,
    ensure_4d,
    load_weights,
    pad_to_multiple,
)
from basicstfm.registry import MODELS


@MODELS.register("FactoSTFoundationModel")
class FactoSTFoundationModel(nn.Module):
    """Factorized temporal backbone with lightweight ST adaptation.

    FactoST decouples universal temporal pretraining from domain-specific
    spatio-temporal adaptation. This implementation keeps that decomposition:
    a channel-independent patch Transformer acts as the universal temporal
    backbone, while low-rank prompts and node/time metadata gates provide the
    factorized adapter used for zero-shot and few-shot transfer stages.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        patch_len: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_prompt_tokens: int = 3,
        max_patches: int = 4096,
        max_channels: int = 64,
        use_st_adapter: bool = True,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.patch_len = int(patch_len)
        self.hidden_dim = int(hidden_dim)
        self.max_patches = int(max_patches)
        self.max_channels = int(max_channels)
        self.use_st_adapter = bool(use_st_adapter)
        self.num_prompt_tokens = int(num_prompt_tokens)

        self.patch_proj = nn.Linear(self.patch_len, self.hidden_dim)
        self.patch_decoder = nn.Linear(self.hidden_dim, self.patch_len)
        self.forecast_head = nn.Linear(
            self.hidden_dim,
            math.ceil(self.output_len / self.patch_len) * self.patch_len,
        )
        self.patch_pos = nn.Parameter(torch.zeros(1, self.max_patches, self.hidden_dim))
        self.node_emb = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.channel_emb = nn.Embedding(self.max_channels, self.hidden_dim)

        self.prompt_u = nn.Parameter(torch.zeros(self.num_prompt_tokens, 1))
        self.prompt_v = nn.Parameter(torch.empty(1, self.hidden_dim))
        self.prompt_adapter = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.metadata_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.reset_parameters()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        nn.init.normal_(self.channel_emb.weight, std=0.02)
        nn.init.normal_(self.prompt_v, std=0.02)
        nn.init.xavier_uniform_(self.prompt_adapter.weight)
        nn.init.zeros_(self.prompt_adapter.bias)

    def _patch(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {nodes}")
        if channels != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {channels}")
        if channels > self.max_channels:
            raise ValueError(f"input_dim={channels} exceeds max_channels={self.max_channels}")
        x, pad_len = pad_to_multiple(x, self.patch_len, dim=1)
        padded_steps = x.shape[1]
        patches = x.permute(0, 2, 3, 1).unfold(-1, self.patch_len, self.patch_len)
        # [B, N, C, P, L]
        num_patches = patches.shape[-2]
        if num_patches > self.max_patches:
            raise ValueError(f"num_patches={num_patches} exceeds max_patches={self.max_patches}")
        return patches.contiguous(), padded_steps, pad_len

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del graph
        if mask is not None:
            x = torch.where(mask.bool(), torch.zeros_like(x), x)
        patches, _, _ = self._patch(x)
        batch, nodes, channels, num_patches, _ = patches.shape
        tokens = self.patch_proj(patches)

        node_ids = torch.arange(nodes, device=tokens.device)
        channel_ids = torch.arange(channels, device=tokens.device)
        metadata = (
            self.node_emb(node_ids)[None, :, None, None]
            + self.channel_emb(channel_ids)[None, None, :, None]
            + self.patch_pos[:, :num_patches].reshape(1, 1, 1, num_patches, self.hidden_dim)
        )
        tokens = tokens + metadata
        tokens = tokens.reshape(batch, nodes * channels, num_patches, self.hidden_dim)
        tokens = tokens.reshape(batch * nodes * channels, num_patches, self.hidden_dim)

        if self.num_prompt_tokens > 0:
            prompt = self.prompt_adapter(self.prompt_u @ self.prompt_v)
            prompt = prompt.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([prompt, tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        if self.num_prompt_tokens > 0:
            tokens = tokens[:, self.num_prompt_tokens :]

        tokens = self.norm(tokens)
        if self.use_st_adapter:
            gate = self.metadata_gate(tokens)
            tokens = tokens * (1.0 + gate)
        return tokens.reshape(batch, nodes, channels, num_patches, self.hidden_dim)

    def _embedding_grid(self, encoded: torch.Tensor) -> torch.Tensor:
        # [B, N, C, P, D] -> [B, P, N, D]
        return encoded.mean(dim=2).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        encoded = self.encode(x, graph=graph, mask=mask)
        out = {"embedding": self._embedding_grid(encoded)}
        if mode in {"encode", "embedding"}:
            return out

        batch, nodes, channels, num_patches, dim = encoded.shape
        if mode in {"forecast", "zero_shot", "prompt_forecast", "both"}:
            summary = encoded.mean(dim=3)
            forecast = self.forecast_head(summary)[..., : self.output_len]
            forecast = forecast.permute(0, 3, 1, 2).contiguous()
            if self.output_dim != channels:
                if self.output_dim == 1:
                    forecast = forecast.mean(dim=-1, keepdim=True)
                else:
                    raise ValueError(
                        "FactoSTFoundationModel currently requires output_dim=input_dim "
                        "unless output_dim=1"
                    )
            out["forecast"] = forecast

        if mode in {"reconstruct", "reconstruction", "both"}:
            recon = self.patch_decoder(encoded.reshape(batch * nodes * channels, num_patches, dim))
            recon = recon.reshape(batch, nodes, channels, num_patches * self.patch_len)
            recon = recon[..., : ensure_4d(x).shape[1]]
            out["reconstruction"] = recon.permute(0, 3, 1, 2).contiguous()
        return out
