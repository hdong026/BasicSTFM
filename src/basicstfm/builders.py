"""Component import and build helpers."""

from __future__ import annotations

import importlib
from typing import Iterable

BUILTIN_MODULES = (
    "basicstfm.data.datamodule",
    "basicstfm.losses.disentangle_losses",
    "basicstfm.losses.diffusion_losses",
    "basicstfm.losses.common",
    "basicstfm.losses.stable_losses",
    "basicstfm.metrics.common",
    "basicstfm.models.dpm_stfm",
    "basicstfm.models.mlp_forecaster",
    "basicstfm.models.st_foundation",
    "basicstfm.models.foundation",
    "basicstfm.tasks.contrastive",
    "basicstfm.tasks.forecasting",
    "basicstfm.tasks.joint_pretraining",
    "basicstfm.tasks.masked_forecast_completion",
    "basicstfm.tasks.masked_reconstruction",
    "basicstfm.tasks.stable_residual_forecasting_task",
    "basicstfm.engines.trainer",
)


def import_builtin_components() -> None:
    """Import built-in modules so their registry decorators run."""

    for module_name in BUILTIN_MODULES:
        importlib.import_module(module_name)


def import_custom_modules(module_names: Iterable[str]) -> None:
    """Import user modules listed in ``custom_imports``."""

    for module_name in module_names:
        importlib.import_module(module_name)
