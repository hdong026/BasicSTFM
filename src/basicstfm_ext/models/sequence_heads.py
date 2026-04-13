"""Variable-length interface heads used around a fixed-length backbone."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import ensure_4d
from basicstfm_ext.models.dataset_conditioning import HeadConditioning


def make_sequence_backend(
    backend_type: str,
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    backend_type = str(backend_type).lower()
    if backend_type == "gru":
        return GRUSequenceBackend(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if backend_type == "interp_mlp":
        return InterpMLPSequenceBackend(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if backend_type == "mamba":
        return MambaSequenceBackend(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown sequence backend: {backend_type!r}")


class GRUSequenceBackend(nn.Module):
    """Node-wise GRU sequence encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class InterpMLPSequenceBackend(nn.Module):
    """Cheap temporal baseline: per-step MLP with interpolation outside."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MambaSequenceBackend(nn.Module):
    """Optional Mamba backend with graceful GRU fallback."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.is_fallback = False
        try:
            from mamba_ssm import Mamba  # type: ignore

            blocks = []
            current_dim = input_dim
            for _ in range(int(num_layers)):
                blocks.append(nn.Linear(current_dim, hidden_dim))
                blocks.append(Mamba(d_model=hidden_dim))
                current_dim = hidden_dim
            self.net = nn.ModuleList(blocks)
        except Exception:
            self.is_fallback = True
            self.net = nn.ModuleList(
                [
                    GRUSequenceBackend(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fallback:
            return self.net[0](x)
        h = x
        for block in self.net:
            h = block(h)
        return h


class VariableInputInterfaceHead(nn.Module):
    """Map runtime context windows to the backbone's native input length."""

    def __init__(
        self,
        *,
        runtime_input_dim: int,
        backbone_input_dim: int,
        backbone_input_len: int,
        hidden_dim: int = 32,
        bottleneck_dim: int = 32,
        num_layers: int = 1,
        backend_type: str = "gru",
        dropout: float = 0.0,
        stronger_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.runtime_input_dim = int(runtime_input_dim)
        self.backbone_input_len = int(backbone_input_len)
        self.backbone_input_dim = int(backbone_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.stronger_conditioning = bool(stronger_conditioning)
        self.backend = make_sequence_backend(
            backend_type,
            input_dim=int(runtime_input_dim),
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.base_proj = nn.Linear(self.hidden_dim, self.backbone_input_dim)
        self.private_proj = nn.Linear(self.hidden_dim, int(bottleneck_dim))

    def forward(
        self,
        x: torch.Tensor,
        conditioning: HeadConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if channels > self.runtime_input_dim:
            raise ValueError(
                f"Expected at most runtime_input_dim={self.runtime_input_dim}, got {channels}"
            )
        if channels < self.runtime_input_dim:
            x = F.pad(x, (0, self.runtime_input_dim - channels))
            channels = self.runtime_input_dim
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        hidden = self.backend(seq)
        hidden = _resample_time(hidden, self.backbone_input_len)
        hidden = self.norm(hidden)
        hidden = _apply_affine(hidden, conditioning.input_scale, conditioning.input_shift)
        base = self.base_proj(hidden)
        residual = _apply_low_rank(
            hidden,
            conditioning.input_down,
            conditioning.input_up,
            output_dim=self.backbone_input_dim,
        )
        gate = _resolve_gate(hidden, conditioning.input_gate)
        gate = gate if self.stronger_conditioning else 0.5 * gate
        out = base + gate * residual
        private = self.private_proj(hidden.mean(dim=1)).reshape(batch, nodes, -1).mean(dim=1)
        out = out.reshape(batch, nodes, self.backbone_input_len, self.backbone_input_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out, private


class VariableOutputInterfaceHead(nn.Module):
    """Map fixed-horizon backbone forecasts to arbitrary runtime horizons."""

    def __init__(
        self,
        *,
        backbone_output_dim: int,
        runtime_output_dim: int,
        hidden_dim: int = 32,
        bottleneck_dim: int = 32,
        num_layers: int = 1,
        backend_type: str = "gru",
        dropout: float = 0.0,
        stronger_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.runtime_output_dim = int(runtime_output_dim)
        self.hidden_dim = int(hidden_dim)
        self.stronger_conditioning = bool(stronger_conditioning)
        self.backend = make_sequence_backend(
            backend_type,
            input_dim=int(backbone_output_dim),
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.base_proj = nn.Linear(self.hidden_dim, self.runtime_output_dim)
        self.private_proj = nn.Linear(self.hidden_dim, int(bottleneck_dim))

    def forward(
        self,
        backbone_forecast: torch.Tensor,
        *,
        conditioning: HeadConditioning,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_forecast = ensure_4d(backbone_forecast)
        batch, steps, nodes, channels = backbone_forecast.shape
        seq = backbone_forecast.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        hidden = self.backend(seq)
        hidden = _resample_time(hidden, int(target_len))
        hidden = self.norm(hidden)
        hidden = _apply_affine(hidden, conditioning.output_scale, conditioning.output_shift)
        base = self.base_proj(hidden)
        residual = _apply_low_rank(
            hidden,
            conditioning.output_down,
            conditioning.output_up,
            output_dim=self.runtime_output_dim,
        )
        gate = _resolve_gate(hidden, conditioning.output_gate)
        gate = 0.5 * gate if self.stronger_conditioning else gate
        out = base + gate * residual
        private = self.private_proj(hidden.mean(dim=1)).reshape(batch, nodes, -1).mean(dim=1)
        out = out.reshape(batch, nodes, int(target_len), self.runtime_output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out, private


def _resample_time(value: torch.Tensor, target_len: int) -> torch.Tensor:
    if value.shape[1] == int(target_len):
        return value
    value = value.transpose(1, 2)
    value = F.interpolate(value, size=int(target_len), mode="linear", align_corners=False)
    return value.transpose(1, 2)


def _apply_affine(
    value: torch.Tensor,
    scale: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
) -> torch.Tensor:
    if scale is not None:
        value = value * (1.0 + scale.view(1, 1, -1))
    if shift is not None:
        value = value + shift.view(1, 1, -1)
    return value


def _apply_low_rank(
    value: torch.Tensor,
    down: Optional[torch.Tensor],
    up: Optional[torch.Tensor],
    *,
    output_dim: int,
) -> torch.Tensor:
    if down is None or up is None:
        return value.new_zeros(value.shape[0], value.shape[1], int(output_dim))
    return torch.einsum("bth,hr,ro->bto", value, down, up)


def _resolve_gate(hidden: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return hidden.new_ones(1, 1, 1)
    if gate.ndim == 0:
        gate = gate.unsqueeze(0)
    return gate.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
