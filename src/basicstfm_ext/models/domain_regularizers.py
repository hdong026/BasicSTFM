"""Auxiliary regularizers that encourage domain-invariant shared features."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm_ext.models.gradient_reversal import GradientReversal


class GRLDomainClassifier(nn.Module):
    """Small domain classifier attached through gradient reversal."""

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int = 64,
        grl_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_domains = int(num_domains)
        self.grl = GradientReversal(coeff=grl_coeff)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_domains),
        )

    def forward(
        self,
        shared_feat: torch.Tensor,
        domain_labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.num_domains <= 1 or domain_labels is None:
            return None
        if shared_feat.ndim > 2:
            shared_feat = shared_feat.reshape(shared_feat.shape[0], -1)
        labels = domain_labels.reshape(-1).long()
        logits = self.classifier(self.grl(shared_feat))
        return F.cross_entropy(logits, labels)
