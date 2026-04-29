"""DPM-v5: DPM-SR (SRDSTFMBackbone) with DPM-v3-style robust Stage-I only."""

from __future__ import annotations

from basicstfm.registry import TASKS
from basicstfm.tasks.robust_stage1_mixin import RobustStage1TaskMixin
from basicstfm.tasks.stable_residual_forecasting_task import StableResidualForecastingTask


@TASKS.register("StableResidualForecastingTaskV5")
class StableResidualForecastingTaskV5(RobustStage1TaskMixin, StableResidualForecastingTask):
    """SRDSTFM + optional robust stable-law pretraining; Stage II/III match StableResidualForecastingTask."""
