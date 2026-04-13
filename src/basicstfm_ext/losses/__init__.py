"""Extension auxiliary losses."""

from basicstfm_ext.losses.orthogonality import OrthogonalityLoss
from basicstfm_ext.losses.redundancy import RedundancyPenalty

__all__ = ["OrthogonalityLoss", "RedundancyPenalty"]
