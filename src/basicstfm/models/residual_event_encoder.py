"""Residual-to-event encoder that separates propagative events from local noise."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicstfm.models.foundation.common import normalize_adjacency
from basicstfm.utils.spectral_ops import ensure_4d


class ResidualEventEncoder(nn.Module):
    """Encode residual signals into event activations and event latents."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.temporal_filter = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=1,
        )
        self.event_score_head = nn.Linear(self.hidden_dim, 1)
        self.intensity_head = nn.Linear(self.hidden_dim, 1)
        self.locality_head = nn.Linear(self.hidden_dim, 1)
        self.activation_bias = nn.Parameter(torch.tensor(float(activation_bias)))
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _graph_locality(
        self,
        residual: torch.Tensor,
        graph: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, nodes, _ = residual.shape
        adj = normalize_adjacency(
            graph,
            num_nodes=nodes,
            device=residual.device,
            dtype=residual.dtype,
            add_self_loops=True,
        )
        neighbor = torch.einsum("ij,btjc->btic", adj, residual)
        locality = (residual - neighbor).abs().mean(dim=-1, keepdim=True)
        return locality

    def forward(
        self,
        residual: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        residual = ensure_4d(residual)
        _, _, _, channels = residual.shape
        if channels != self.input_dim:
            raise ValueError(f"Expected residual channels={self.input_dim}, got {channels}")

        value = self.value_proj(residual)
        temporal = self.temporal_filter(value.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        event_latent = self.norm(value + temporal)

        score = torch.sigmoid(self.event_score_head(event_latent) + self.activation_bias)
        intensity = torch.sigmoid(self.intensity_head(event_latent) + residual.abs().mean(dim=-1, keepdim=True))

        locality_raw = self._graph_locality(residual, graph)
        locality = torch.sigmoid(self.locality_head(event_latent) + locality_raw)

        propagation_worthiness = score * intensity * locality
        filtered_event = event_latent * propagation_worthiness

        return {
            "event_latent": filtered_event,
            "event_score": score,
            "event_intensity": intensity,
            "event_locality": locality,
            "event_activation": propagation_worthiness,
            "event_raw_latent": event_latent,
        }
