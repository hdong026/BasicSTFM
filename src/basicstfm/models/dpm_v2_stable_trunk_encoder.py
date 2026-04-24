"""Stable Trunk Encoder v2: Stage-I multi-scale branches + static graph stable context (DPM_v2)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import normalize_adjacency
from basicstfm.utils.spectral_ops import (
    ensure_4d,
    extract_temporal_trend,
    split_low_high_frequency,
)
from basicstfm.models.stable_trunk_encoder import (
    StableMixerBlock,
    TemporalAttentionPooling,
    TwoLayerTemporalSummarizer,
)


class StableGraphContextLayer(nn.Module):
    """One static-graph aggregation step: row-normalized A_tilde, message passing, residual + LayerNorm.

    This is **not** a diffusion or dynamic event graph; it encodes a fixed structural
    prior for *stable* cross-node coordination at each time step.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor, adj_tilde: torch.Tensor) -> torch.Tensor:
        # z: [B, T, N, D], adj_tilde: [N, N]
        msg = torch.einsum("ij,btjd->btid", adj_tilde, z)
        return self.norm(z + self.msg_net(msg))


class StableGraphContextStack(nn.Module):
    """K light layers of stable graph context; optional weight sharing across layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("stable_graph_num_layers must be >= 1 when graph context is used")
        self.num_layers = int(num_layers)
        self.share_weights = bool(share_weights)
        if self.share_weights:
            self.shared = StableGraphContextLayer(hidden_dim, dropout=dropout)
            self.layers: Optional[nn.ModuleList] = None
        else:
            self.shared = None
            self.layers = nn.ModuleList(
                [StableGraphContextLayer(hidden_dim, dropout=dropout) for _ in range(self.num_layers)]
            )

    def forward(self, z: torch.Tensor, adj_tilde: torch.Tensor) -> torch.Tensor:
        if self.share_weights:
            assert self.shared is not None
            h = z
            for _ in range(self.num_layers):
                h = self.shared(h, adj_tilde)
            return h
        assert self.layers is not None
        h = z
        for layer in self.layers:
            h = layer(h, adj_tilde)
        return h


class StableTrunkEncoderV2(nn.Module):
    """Multi-branch stable encoder (same as v1) + optional static graph context before the mixer.

    Fused branch features Z^(0) are refined as Z' = G(Z^(0), A_tilde) when enabled, then passed
    to ``StableMixerBlock``s and the temporal summarizer. This keeps Stage II/III interfaces
    identical to the original backbone.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_len: int,
        output_dim: int,
        local_kernel_size: int = 3,
        coarse_scale: int = 4,
        use_frequency_branch: bool = True,
        dropout: float = 0.1,
        summary_mode: str = "attention",
        stable_mixer_layers: int = 2,
        stable_mixer_kernel_size: int = 3,
        frequency_low_ratio: float = 0.3,
        frequency_num_low_bins: Optional[int] = None,
        use_stable_graph_context: bool = True,
        stable_graph_num_layers: int = 1,
        stable_graph_share_weights: bool = False,
    ) -> None:
        super().__init__()
        if local_kernel_size <= 0 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be an odd positive integer")
        if coarse_scale <= 1:
            raise ValueError("coarse_scale must be > 1")
        if summary_mode not in {"attention", "summarizer"}:
            raise ValueError("summary_mode must be one of: attention, summarizer")
        if stable_mixer_layers < 1:
            raise ValueError("stable_mixer_layers must be >= 1")
        if stable_graph_num_layers < 1:
            raise ValueError("stable_graph_num_layers must be >= 1")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.coarse_scale = int(coarse_scale)
        self.use_frequency_branch = bool(use_frequency_branch)
        self.summary_mode = str(summary_mode)
        self.frequency_low_ratio = float(frequency_low_ratio)
        self.frequency_num_low_bins = (
            None if frequency_num_low_bins is None else int(frequency_num_low_bins)
        )
        self.use_stable_graph_context = bool(use_stable_graph_context)
        self.stable_graph_num_layers = int(stable_graph_num_layers)
        self.stable_graph_share_weights = bool(stable_graph_share_weights)

        padding = local_kernel_size // 2
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                self.input_dim,
                self.hidden_dim,
                kernel_size=(local_kernel_size, 1),
                padding=(padding, 0),
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coarse_branch = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frequency_branch = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        branch_count = 3 if self.use_frequency_branch else 2
        self.branch_logits = nn.Parameter(torch.zeros(branch_count))
        self.branch_norm = nn.LayerNorm(self.hidden_dim)

        if self.use_stable_graph_context:
            self.stable_graph = StableGraphContextStack(
                hidden_dim=self.hidden_dim,
                num_layers=self.stable_graph_num_layers,
                dropout=dropout,
                share_weights=self.stable_graph_share_weights,
            )
        else:
            self.stable_graph = None

        self.stable_mixer = nn.ModuleList(
            [
                StableMixerBlock(
                    hidden_dim=self.hidden_dim,
                    kernel_size=stable_mixer_kernel_size,
                    dropout=dropout,
                )
                for _ in range(stable_mixer_layers)
            ]
        )

        if self.summary_mode == "attention":
            self.temporal_pool = TemporalAttentionPooling(self.hidden_dim, dropout=dropout)
        else:
            self.temporal_pool = TwoLayerTemporalSummarizer(self.hidden_dim, dropout=dropout)

        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )

    def _coarse_temporal(self, x: torch.Tensor) -> torch.Tensor:
        return extract_temporal_trend(x, scale=self.coarse_scale)

    def _frequency_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        low, _ = split_low_high_frequency(
            x,
            low_ratio=self.frequency_low_ratio,
            num_low_bins=self.frequency_num_low_bins,
        )
        return low

    def _apply_stable_graph(
        self,
        stable_latent: torch.Tensor,
        graph: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_stable_graph_context or self.stable_graph is None:
            return stable_latent
        if graph is None:
            return stable_latent
        batch, steps, nodes, _ = stable_latent.shape
        if graph.shape[0] != nodes or graph.shape[1] != nodes:
            return stable_latent
        device = stable_latent.device
        dtype = stable_latent.dtype
        adj_tilde = normalize_adjacency(graph, nodes, device, dtype, add_self_loops=True)
        return self.stable_graph(stable_latent, adj_tilde)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        graph: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if channels > self.input_dim:
            raise ValueError(f"Expected at most {self.input_dim} channels, got {channels}")

        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
            if mask is not None:
                mask = F.pad(mask, (0, self.input_dim - channels))
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.zeros_like(x))

        local = self.local_branch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        coarse = self.coarse_branch(self._coarse_temporal(x))

        branches = [local, coarse]
        if self.use_frequency_branch:
            frequency = self.frequency_branch(self._frequency_skeleton(x))
            branches.append(frequency)
        else:
            frequency = local.new_zeros(local.shape)

        weights = F.softmax(self.branch_logits[: len(branches)], dim=0)
        stable_latent = sum(w * b for w, b in zip(weights, branches))
        stable_latent = self.branch_norm(stable_latent)

        stable_latent = self._apply_stable_graph(stable_latent, graph)

        for block in self.stable_mixer:
            stable_latent = block(stable_latent)

        stable_reconstruction = self.reconstruction_head(stable_latent)
        summary, temporal_weights = self.temporal_pool(stable_latent, mask=mask)
        stable_forecast = self.forecast_head(summary)
        stable_forecast = stable_forecast.reshape(batch, nodes, self.output_len, self.output_dim)
        stable_forecast = stable_forecast.permute(0, 2, 1, 3).contiguous()

        return {
            "stable_latent": stable_latent,
            "stable_reconstruction": stable_reconstruction,
            "stable_forecast": stable_forecast,
            "stable_summary": summary,
            "stable_temporal_weight": temporal_weights,
            "stable_local_branch": local,
            "stable_coarse_branch": coarse,
            "stable_branch_weight": weights,
            "stable_frequency_branch": frequency,
        }
