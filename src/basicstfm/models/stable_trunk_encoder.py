"""Stable Trunk Encoder for cross-city stable dynamics discovery."""

from __future__ import annotations
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import ensure_4d, split_low_high_frequency


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
    ) -> None:
        super().__init__()
        if local_kernel_size <= 0 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be an odd positive integer")
        if coarse_scale <= 1:
            raise ValueError("coarse_scale must be > 1")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.coarse_scale = int(coarse_scale)
        self.use_frequency_branch = bool(use_frequency_branch)

        padding = local_kernel_size // 2
        self.local_branch = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=(local_kernel_size, 1), padding=(padding, 0)),
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
        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )

    def _coarse_temporal(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, channels = x.shape
        grid = x.permute(0, 2, 3, 1).reshape(batch * nodes, channels, steps)
        pooled = F.avg_pool1d(grid, kernel_size=self.coarse_scale, stride=self.coarse_scale, ceil_mode=True)
        upsampled = F.interpolate(
            pooled,
            size=steps,
            mode="linear",
            align_corners=False,
        )
        upsampled = upsampled.reshape(batch, nodes, channels, steps).permute(0, 3, 1, 2).contiguous()
        return upsampled

    def _frequency_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        low, _ = split_low_high_frequency(x, low_ratio=0.3)
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

        weights = F.softmax(self.branch_logits[: len(branches)], dim=0)
        stable_latent = sum(weight * branch for weight, branch in zip(weights, branches))
        stable_latent = self.branch_norm(stable_latent)

        stable_reconstruction = self.reconstruction_head(stable_latent)
        summary = stable_latent.mean(dim=1)
        stable_forecast = self.forecast_head(summary)
        stable_forecast = stable_forecast.reshape(batch, nodes, self.output_len, self.output_dim)
        stable_forecast = stable_forecast.permute(0, 2, 1, 3).contiguous()

        return {
            "stable_latent": stable_latent,
            "stable_reconstruction": stable_reconstruction,
            "stable_forecast": stable_forecast,
            "stable_local_branch": local,
            "stable_coarse_branch": coarse,
            "stable_branch_weight": weights,
            "stable_frequency_branch": branches[2] if self.use_frequency_branch else x.new_zeros(1),
        }
