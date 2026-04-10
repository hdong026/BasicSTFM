"""Representation pretraining tasks for foundation-model stages."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn import functional as F

from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task, move_to_device


@TASKS.register()
class ContrastiveRepresentationTask(Task):
    """SimCLR-style temporal representation pretraining.

    The task creates two stochastic views from the same spatio-temporal window,
    encodes them through a model's ``mode='encode'`` path, and optimizes a
    symmetric InfoNCE loss. It is intentionally model-agnostic and can be used
    before downstream forecasting fine-tuning for OpenCity, FactoST, UniST, or
    custom STFM backbones.
    """

    def __init__(
        self,
        input_key: str = "x",
        temperature: float = 0.2,
        noise_std: float = 0.02,
        drop_ratio: float = 0.1,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= drop_ratio < 1.0:
            raise ValueError("drop_ratio must be in [0, 1)")
        self.input_key = input_key
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.drop_ratio = float(drop_ratio)

    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        del losses
        batch = move_to_device(batch, device)
        x = self.transform(batch[self.input_key])
        view_a = self._augment(x)
        view_b = self._augment(x)
        out_a = model(view_a, graph=batch.get("graph"), mode="encode")
        out_b = model(view_b, graph=batch.get("graph"), mode="encode")
        z_a = self._pool_embedding(out_a["embedding"] if isinstance(out_a, dict) else out_a)
        z_b = self._pool_embedding(out_b["embedding"] if isinstance(out_b, dict) else out_b)
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = z_a @ z_b.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return {
            "loss": loss,
            "logs": {
                "loss/contrastive": loss.detach(),
                "loss/total": loss.detach(),
                "metric/contrastive_acc": acc.detach(),
            },
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        if self.drop_ratio > 0:
            keep = torch.rand_like(out[..., :1]) >= self.drop_ratio
            out = out * keep.to(dtype=out.dtype)
        return out

    @staticmethod
    def _pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim < 2:
            raise ValueError("embedding must have at least batch and feature dimensions")
        return embedding.reshape(embedding.shape[0], -1, embedding.shape[-1]).mean(dim=1)
