"""Optional paper-style baseline adapters (UniFlow, STPFormer, UrbanDiT-lite, Mamba-sync, Chronos-2 ZS)."""

from basicstfm.models.baselines import chronos2_zs  # noqa: F401
from basicstfm.models.baselines import st_mamba_sync  # noqa: F401
from basicstfm.models.baselines import stpformer  # noqa: F401
from basicstfm.models.baselines import uniflow  # noqa: F401
from basicstfm.models.baselines import urbandit_lite  # noqa: F401

__all__ = [
    "chronos2_zs",
    "st_mamba_sync",
    "stpformer",
    "uniflow",
    "urbandit_lite",
]
