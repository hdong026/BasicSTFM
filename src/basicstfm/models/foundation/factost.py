"""FactoST-style factorized spatio-temporal foundation model adapter."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import functional as F
from torch import nn

from basicstfm.models.foundation.common import (
    TransformerBlock,
    ensure_4d,
    load_filtered_weights,
    load_weights,
    normalize_adjacency,
    pad_to_multiple,
)
from basicstfm.registry import MODELS


@MODELS.register("FactoSTFoundationModel")
class FactoSTFoundationModel(nn.Module):
    """Factorized temporal backbone with framework-native ST adaptation.

    FactoST decouples universal temporal pretraining from domain-specific
    spatio-temporal adaptation. This implementation keeps that decomposition:
    a channel-independent patch Transformer acts as the universal temporal
    backbone, while a reusable ST adapter adds three ingredients inspired by
    the original design: metadata fusion, relation filtering, and prototype
    refinement. The implementation stays generic to the framework tensor API
    instead of depending on dataset-specific holiday or clock metadata.
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
        use_st_metadata: bool = True,
        use_st_filtering: bool = True,
        use_cpr: bool = True,
        filter_matrices: Optional[tuple[str, ...]] = None,
        max_delay_steps: int = 3,
        num_prototypes: int = 8,
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
        self.use_st_metadata = bool(use_st_metadata) and self.use_st_adapter
        self.use_st_filtering = bool(use_st_filtering) and self.use_st_metadata
        self.use_cpr = bool(use_cpr) and self.use_st_metadata
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.filter_matrices = tuple(filter_matrices or ("S_s", "S_t", "S_d"))
        self.max_delay_steps = int(max_delay_steps)
        self.num_prototypes = int(num_prototypes)

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
        self.st_node_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_channel_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_temporal_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_metadata_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.st_graph_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_cpr_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_delay_gates = nn.Parameter(torch.zeros(self.max_delay_steps))
        self.st_score_weights = nn.ParameterDict(
            {
                "S_s": nn.Parameter(torch.tensor(0.0)),
                "S_t": nn.Parameter(torch.tensor(0.0)),
                "S_d": nn.Parameter(torch.tensor(0.0)),
            }
        )
        self.st_temporal_queries = nn.Parameter(torch.empty(self.max_patches, self.hidden_dim))
        self.st_latent_prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.hidden_dim))

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

    def load_backbone_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load the shared UTP backbone while leaving STA-only parameters untouched."""

        return load_filtered_weights(
            self,
            path,
            strict=strict,
            exclude_prefixes=(
                "prompt_",
                "metadata_gate.",
                "st_",
            ),
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        nn.init.normal_(self.channel_emb.weight, std=0.02)
        nn.init.normal_(self.prompt_v, std=0.02)
        nn.init.xavier_uniform_(self.prompt_adapter.weight)
        nn.init.zeros_(self.prompt_adapter.bias)
        nn.init.xavier_uniform_(self.st_node_proj.weight)
        nn.init.zeros_(self.st_node_proj.bias)
        nn.init.xavier_uniform_(self.st_channel_proj.weight)
        nn.init.zeros_(self.st_channel_proj.bias)
        nn.init.xavier_uniform_(self.st_temporal_proj.weight)
        nn.init.zeros_(self.st_temporal_proj.bias)
        nn.init.xavier_uniform_(self.st_metadata_proj.weight)
        nn.init.zeros_(self.st_metadata_proj.bias)
        nn.init.xavier_uniform_(self.st_graph_proj.weight)
        nn.init.zeros_(self.st_graph_proj.bias)
        nn.init.xavier_uniform_(self.st_cpr_gate.weight)
        nn.init.zeros_(self.st_cpr_gate.bias)
        nn.init.trunc_normal_(self.st_temporal_queries, std=0.02)
        nn.init.trunc_normal_(self.st_latent_prototypes, std=0.02)

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

    def _st_metadata_fusion(
        self,
        tokens: torch.Tensor,
        graph: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, nodes, channels, num_patches, _ = tokens.shape
        node_ids = torch.arange(nodes, device=tokens.device)
        channel_ids = torch.arange(channels, device=tokens.device)

        node_context = self.st_node_proj(self.node_emb(node_ids)).view(1, nodes, 1, 1, self.hidden_dim)
        channel_context = self.st_channel_proj(self.channel_emb(channel_ids)).view(
            1, 1, channels, 1, self.hidden_dim
        )
        temporal_context = self.st_temporal_proj(self.patch_pos[:, :num_patches]).view(
            1, 1, 1, num_patches, self.hidden_dim
        )
        metadata = torch.cat(
            [
                node_context.expand(batch, -1, channels, num_patches, -1),
                channel_context.expand(batch, nodes, -1, num_patches, -1),
                temporal_context.expand(batch, nodes, channels, -1, -1),
            ],
            dim=-1,
        )
        fused = self.st_metadata_proj(metadata)

        graph_context = torch.zeros_like(tokens)
        if graph is not None:
            adj = normalize_adjacency(graph, nodes, tokens.device, tokens.dtype)
            graph_context = torch.einsum("ij,bjcpd->bicpd", adj, tokens)
            graph_context = self.st_graph_proj(graph_context)
            fused = fused + graph_context

        return fused, {
            "node": node_context,
            "channel": channel_context,
            "temporal": temporal_context,
            "graph": graph_context,
        }

    def _cyclic_prototype_refinement(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels, num_patches, _ = tokens.shape
        query = self.st_temporal_queries[:num_patches].view(1, 1, 1, num_patches, self.hidden_dim)
        proto_logits = torch.einsum(
            "bncpd,md->bncpm",
            tokens + query.expand(batch, nodes, channels, -1, -1),
            self.st_latent_prototypes,
        )
        proto_weights = F.softmax(proto_logits, dim=-1)
        prototype_update = torch.einsum("bncpm,md->bncpd", proto_weights, self.st_latent_prototypes)
        gate = torch.sigmoid(self.st_cpr_gate(tokens + query))
        return tokens + gate * prototype_update

    def _st_filter(
        self,
        tokens: torch.Tensor,
        contexts: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        scores = []
        weights = []

        if "S_s" in self.filter_matrices:
            spatial_reference = contexts["node"] + contexts["graph"]
            spatial_score = (tokens * spatial_reference).sum(dim=-1)
            scores.append(spatial_score)
            weights.append(self.st_score_weights["S_s"])

        if "S_t" in self.filter_matrices:
            temporal_reference = contexts["temporal"] + contexts["channel"]
            temporal_score = (tokens * temporal_reference).sum(dim=-1)
            scores.append(temporal_score)
            weights.append(self.st_score_weights["S_t"])

        if "S_d" in self.filter_matrices:
            delay_score = torch.zeros_like(tokens[..., 0])
            max_lag = min(self.max_delay_steps, tokens.shape[3])
            lag_weights = F.softplus(self.st_delay_gates[:max_lag])
            for lag in range(1, max_lag + 1):
                lagged = torch.zeros_like(tokens)
                lagged[:, :, :, lag:, :] = tokens[:, :, :, :-lag, :]
                proto_logits = torch.einsum("bncpd,md->bncpm", lagged, self.st_latent_prototypes)
                proto_weights = F.softmax(proto_logits, dim=-1)
                lagged_proto = torch.einsum("bncpm,md->bncpd", proto_weights, self.st_latent_prototypes)
                delay_score = delay_score + lag_weights[lag - 1] * (tokens * lagged_proto).sum(dim=-1)
            scores.append(delay_score)
            weights.append(self.st_score_weights["S_d"])

        if not scores:
            return tokens

        normalized_weights = F.softmax(torch.stack(weights), dim=0)
        fused_score = sum(weight * score for weight, score in zip(normalized_weights, scores))
        gate = torch.sigmoid(fused_score).unsqueeze(-1)
        return tokens * (1.0 + gate)

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        tokens = tokens.reshape(batch, nodes, channels, num_patches, self.hidden_dim)
        if self.use_st_metadata:
            fused_context, contexts = self._st_metadata_fusion(tokens, graph)
            tokens = tokens + fused_context
            if self.use_cpr:
                tokens = self._cyclic_prototype_refinement(tokens)
            if self.use_st_filtering:
                tokens = self._st_filter(tokens, contexts)
        if self.use_st_adapter:
            gate = self.metadata_gate(tokens)
            tokens = tokens * (1.0 + gate)
        return tokens

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
