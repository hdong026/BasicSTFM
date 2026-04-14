"""Lightweight dataset-conditioned modulation for diffusion dynamics."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class DatasetConditionedFiLM(nn.Module):
    """Apply FiLM modulation with a shared dataset embedding table."""

    def __init__(
        self,
        hidden_dim: int,
        num_datasets: int = 1,
        modulation_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_datasets = max(1, int(num_datasets))
        self.modulation_scale = float(modulation_scale)
        self.embedding = nn.Embedding(self.num_datasets, self.hidden_dim)
        self.gamma_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.beta_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        value: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return modulated features and modulation tensors."""

        if value.ndim < 2:
            raise ValueError(f"Expected feature tensor with batch dimension, got {tuple(value.shape)}")
        if dataset_index is None or self.num_datasets == 1:
            zeros = value.new_zeros(value.shape[0], self.hidden_dim)
            return value, {"gamma": zeros, "beta": zeros}

        if not torch.is_tensor(dataset_index):
            dataset_index = torch.as_tensor(dataset_index, dtype=torch.long, device=value.device)
        dataset_index = dataset_index.to(device=value.device, dtype=torch.long).reshape(-1)
        if dataset_index.shape[0] != value.shape[0]:
            if dataset_index.numel() == 1:
                dataset_index = dataset_index.expand(value.shape[0])
            else:
                raise ValueError(
                    "dataset_index batch size mismatch: "
                    f"{dataset_index.shape[0]} vs {value.shape[0]}"
                )

        index = dataset_index.clamp(min=0, max=self.num_datasets - 1)
        cond = self.embedding(index)
        gamma = torch.tanh(self.gamma_proj(cond))
        beta = self.beta_proj(cond)

        expand_shape = [value.shape[0]] + [1] * (value.ndim - 2) + [value.shape[-1]]
        gamma_expanded = gamma.view(*expand_shape)
        beta_expanded = beta.view(*expand_shape)
        scale = self.modulation_scale
        modulated = value * (1.0 + scale * gamma_expanded) + scale * beta_expanded
        return modulated, {"gamma": gamma, "beta": beta}
