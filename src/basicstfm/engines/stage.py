"""Stage plan parsing for multi-stage training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StageSpec:
    name: str
    task: Dict[str, Any]
    epochs: int = 1
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [{"type": "mae"}])
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    optimizer: Dict[str, Any] = field(default_factory=lambda: {"type": "AdamW", "lr": 1e-3})
    scheduler: Optional[Dict[str, Any]] = None
    load_from: Optional[str] = None
    strict_load: bool = True
    freeze: List[str] = field(default_factory=list)
    unfreeze: List[str] = field(default_factory=list)
    validate_every: int = 1
    save_best_by: str = "val/loss/total"
    gradient_clip_val: Optional[float] = None
    save_every: int = 1
    save_best: bool = True
    save_last: bool = True
    save_epoch_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"Stage {self.name!r} must run for at least one epoch")
        if self.validate_every < 1:
            raise ValueError(f"Stage {self.name!r} validate_every must be >= 1")
        if self.save_every < 1:
            raise ValueError(f"Stage {self.name!r} save_every must be >= 1")
        if isinstance(self.freeze, str):
            self.freeze = [self.freeze]
        if isinstance(self.unfreeze, str):
            self.unfreeze = [self.unfreeze]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], index: int = 0) -> "StageSpec":
        cfg = dict(cfg)
        name = str(cfg.pop("name", f"stage_{index}"))
        task = cfg.pop("task", None)
        if not isinstance(task, dict):
            raise ValueError(f"Stage {name!r} must define a task mapping")
        epochs = int(cfg.pop("epochs", cfg.pop("max_epochs", 1)))
        losses = cfg.pop("losses", [{"type": "mae"}])
        if isinstance(losses, dict):
            losses = [losses]
        metrics = cfg.pop("metrics", [])
        if isinstance(metrics, dict):
            metrics = [metrics]
        cfg["metrics"] = metrics
        return cls(name=name, task=task, epochs=epochs, losses=losses, **cfg)


class StagePlan:
    """Ordered list of training stages."""

    def __init__(self, stages: Iterable[StageSpec]) -> None:
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("StagePlan requires at least one stage")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "StagePlan":
        pipeline = cfg.get("pipeline", {})
        stage_cfgs = pipeline.get("stages")
        if not stage_cfgs:
            raise ValueError("Config must define pipeline.stages")
        return cls(StageSpec.from_dict(stage_cfg, idx) for idx, stage_cfg in enumerate(stage_cfgs))

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": stage.name,
                "epochs": stage.epochs,
                "task": stage.task.get("type"),
                "losses": [loss.get("type") for loss in stage.losses],
                "metrics": [metric.get("type") for metric in stage.metrics],
                "load_from": stage.load_from,
                "freeze": stage.freeze,
                "unfreeze": stage.unfreeze,
            }
            for stage in self.stages
        ]
