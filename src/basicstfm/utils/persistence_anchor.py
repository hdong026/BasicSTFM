"""Persistence (naive) forecast: repeat last observed timestep over the prediction horizon."""

from __future__ import annotations

import torch

from basicstfm.models.foundation.common import ensure_4d


def persistence_forecast_from_input(x: torch.Tensor, output_len: int) -> torch.Tensor:
    """Return ``Y_per`` with shape [B, output_len, N, C] matching multi-horizon targets."""

    x = ensure_4d(x)
    last = x[:, -1:, :, :]
    return last.expand(-1, int(output_len), -1, -1).contiguous()
