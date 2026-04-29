"""Stage-I robust stable-law pretraining for DPM v3 (mini-batch EMA, group-wise standardized risk)."""

from __future__ import annotations

from basicstfm.registry import TASKS
from basicstfm.tasks.robust_stage1_mixin import RobustStage1TaskMixin
from basicstfm.tasks.stable_residual_forecasting_task import StableResidualForecastingTask


@TASKS.register("StableResidualForecastingTaskV3")
class StableResidualForecastingTaskV3(RobustStage1TaskMixin, StableResidualForecastingTask):
    """Same as StableResidualForecastingTask, but optional Stage-I (phase=stable) robust reweighting."""
