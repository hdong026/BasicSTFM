"""Extension auxiliary losses."""

from basicstfm_ext.losses.distillation_loss import DistillationLoss
from basicstfm_ext.losses.identity_loss import IdentityLoss
from basicstfm_ext.losses.orthogonality import OrthogonalityLoss
from basicstfm_ext.losses.redundancy import RedundancyPenalty

__all__ = [
    "DistillationLoss",
    "IdentityLoss",
    "OrthogonalityLoss",
    "RedundancyPenalty",
]
