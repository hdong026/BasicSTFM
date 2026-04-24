"""DPM v3 backbone: same as DPM v2; Stage-I robustness is implemented in the task (StableResidualForecastingTaskV3)."""

from __future__ import annotations

from basicstfm.models.dpm_v2_stfm import DPMV2Backbone
from basicstfm.registry import MODELS


@MODELS.register("DPMV3Backbone")
class DPMV3Backbone(DPMV2Backbone):
    """Token class so configs can name DPM v3; weights and forward match ``DPMV2Backbone``."""

    pass
