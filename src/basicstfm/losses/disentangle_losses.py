"""Losses encouraging stable/residual role separation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from basicstfm.utils.spectral_ops import ensure_4d


EPS = 1e-6


def _flatten(value: torch.Tensor) -> torch.Tensor:
    value = ensure_4d(value)
    return value.reshape(value.shape[0], -1)


def orthogonality_loss(stable_component: torch.Tensor, residual_component: torch.Tensor) -> torch.Tensor:
    """Penalize cosine similarity between stable and residual components."""

    stable_vec = _flatten(stable_component)
    residual_vec = _flatten(residual_component)
    stable_vec = stable_vec / stable_vec.norm(dim=-1, keepdim=True).clamp_min(EPS)
    residual_vec = residual_vec / residual_vec.norm(dim=-1, keepdim=True).clamp_min(EPS)
    cosine = (stable_vec * residual_vec).sum(dim=-1).abs()
    return cosine.mean()


def cross_covariance_penalty(stable_component: torch.Tensor, residual_component: torch.Tensor) -> torch.Tensor:
    """Penalize standardized cross-covariance off-diagonal terms.

    Implementation notes:
      1) center over batch;
      2) normalize by batch sample count;
      3) keep only off-diagonal terms.
    """

    stable = ensure_4d(stable_component).mean(dim=(1, 2))    # [B, C]
    residual = ensure_4d(residual_component).mean(dim=(1, 2))  # [B, C]
    if stable.shape != residual.shape:
        raise ValueError(f"Shape mismatch: {tuple(stable.shape)} vs {tuple(residual.shape)}")
    if stable.shape[0] < 2:
        return stable.new_tensor(0.0)

    stable = stable - stable.mean(dim=0, keepdim=True)
    residual = residual - residual.mean(dim=0, keepdim=True)

    # Standardize channel scale to keep loss magnitude stable.
    stable = stable / stable.std(dim=0, keepdim=True).clamp_min(EPS)
    residual = residual / residual.std(dim=0, keepdim=True).clamp_min(EPS)

    cov = stable.t().matmul(residual) / float(stable.shape[0] - 1)  # [C, C]
    diag = torch.diagonal(cov, dim1=0, dim2=1)
    off_diag = cov - torch.diag_embed(diag)
    return off_diag.pow(2).mean()


def energy_allocation_regularizer(
    stable_component: torch.Tensor,
    residual_component: torch.Tensor,
    target_stable_ratio: float = 0.7,
) -> torch.Tensor:
    """Keep energy split in a controllable range."""

    stable_energy = ensure_4d(stable_component).pow(2).mean(dim=(1, 2, 3))
    residual_energy = ensure_4d(residual_component).pow(2).mean(dim=(1, 2, 3))
    ratio = stable_energy / (stable_energy + residual_energy + EPS)
    target = ratio.new_full(ratio.shape, float(target_stable_ratio))
    return F.mse_loss(ratio, target)


def mutual_exclusion_regularizer(
    stable_component: torch.Tensor,
    residual_component: torch.Tensor,
) -> torch.Tensor:
    """Discourage simultaneous high activation in both components."""

    stable_mag = ensure_4d(stable_component).abs()
    residual_mag = ensure_4d(residual_component).abs()
    stable_norm = stable_mag / stable_mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
    residual_norm = residual_mag / residual_mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
    overlap = stable_norm * residual_norm
    return overlap.mean()
