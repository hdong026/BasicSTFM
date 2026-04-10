"""Optional model base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class SpatioTemporalModel(nn.Module, ABC):
    """Base interface for models consuming [B, T, N, C] tensors."""

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError
