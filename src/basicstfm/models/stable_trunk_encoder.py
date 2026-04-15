"""Stable Trunk Encoder for cross-city stable dynamics discovery."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import (
    ensure_4d,
    extract_temporal_trend,
    split_low_high_frequency,
)


class TemporalAttentionPooling(nn.Module):
    """Temporal attention pooling over stable latent slots."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.scorer(value)
        if mask is not None:
            time_mask = ensure_4d(mask).any(dim=-1, keepdim=True)
            logits = logits.masked_fill(~time_mask, -1e4)
            weights = F.softmax(logits, dim=1) * time_mask.to(dtype=value.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            weights = F.softmax(logits, dim=1)
        pooled = (weights * value).sum(dim=1)
        return pooled, weights


class TwoLayerTemporalSummarizer(nn.Module):
    """Two-layer temporal encoder used as a learned stable summarizer."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, hidden_dim = value.shape
        seq = value.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden_dim)

        if mask is not None:
            time_mask = ensure_4d(mask).any(dim=-1)
            seq_mask = time_mask.permute(0, 2, 1).reshape(batch * nodes, steps)
            seq = seq * seq_mask.unsqueeze(-1).to(dtype=seq.dtype)
        else:
            seq_mask = None

        encoded, _ = self.encoder(seq)
        if seq_mask is None:
            summary = encoded[:, -1]
            weights = encoded.new_zeros(batch, steps, nodes, 1)
            weights[:, -1] = 1.0
        else:
            lengths = seq_mask.long().sum(dim=1).clamp_min(1)
            last_index = lengths - 1
            summary = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_index]
            weights = encoded.new_zeros(batch * nodes, steps, 1)
            weights[torch.arange(weights.shape[0], device=weights.device), last_index] = 1.0
            weights = weights.reshape(batch, nodes, steps, 1).permute(0, 2, 1, 3).contiguous()

        summary = self.norm(summary.reshape(batch, nodes, hidden_dim))
        return summary, weights


class StableMixerBlock(nn.Module):
    """Light temporal mixer that strengthens fused stable features."""

    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd positive integer")

        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm(value).permute(0, 3, 1, 2).contiguous()
        value = self.depthwise(value)
        value = F.gelu(value)
        value = self.pointwise(value)
        value = self.dropout(value)
        value = value.permute(0, 2, 3, 1).contiguous()
        return residual + value


class StableTrunkEncoder(nn.Module):
    """Multi-branch encoder emphasizing slow/stable spatio-temporal dynamics."""

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        stable_latent = sum(weight * branch for weight, branch in zip(weights, branches))
        stable_latent = self.branch_norm(stable_latent)
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
