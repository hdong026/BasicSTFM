"""Residual definition module for stable-residual decomposition."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicstfm.utils.spectral_ops import ensure_4d


class ResidualConstructor(nn.Module):
    """Construct residual tensors in input-level or forecast-level mode."""

    def __init__(
        self,
        mode: str = "forecast",
        detach_stable: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"input", "forecast"}:
            raise ValueError("ResidualConstructor mode must be 'input' or 'forecast'")
        self.mode = mode
        self.detach_stable = bool(detach_stable)

    def forward(
        self,
        x: torch.Tensor,
        stable_reconstruction: torch.Tensor,
        stable_forecast: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Build residual components used by the event-diffusion branch."""

        x = ensure_4d(x)
        stable_reconstruction = ensure_4d(stable_reconstruction)
        if x.shape != stable_reconstruction.shape:
            raise ValueError(
                f"x and stable_reconstruction shape mismatch: {tuple(x.shape)} vs "
                f"{tuple(stable_reconstruction.shape)}"
            )

        stable_hist = stable_reconstruction.detach() if self.detach_stable else stable_reconstruction
        residual_input = x - stable_hist

        residual_target = None
        if self.mode == "forecast":
            if stable_forecast is None:
                raise ValueError("forecast mode requires stable_forecast")
            stable_forecast = ensure_4d(stable_forecast)
            if target is not None:
                target = ensure_4d(target)
                if target.shape != stable_forecast.shape:
                    raise ValueError(
                        "target and stable_forecast shape mismatch: "
                        f"{tuple(target.shape)} vs {tuple(stable_forecast.shape)}"
                    )
                stable_future = stable_forecast.detach() if self.detach_stable else stable_forecast
                residual_target = target - stable_future

        return {
            "residual_mode": self.mode,
            "residual_input": residual_input,
            "residual_target": residual_target,
        }
