"""Stage plan parsing for multi-stage training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StageSpec:
    name: str
    task: Dict[str, Any]
    model: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    reset_model: bool = False
    reset_data: bool = False
    epochs: int = 1
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [{"type": "mae"}])
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    optimizer: Dict[str, Any] = field(default_factory=lambda: {"type": "AdamW", "lr": 1e-3})
    scheduler: Optional[Dict[str, Any]] = None
    load_from: Optional[str] = None
    load_method: str = "checkpoint"
    strict_load: bool = True
    save_artifact: Optional[str] = None
    freeze: List[str] = field(default_factory=list)
    unfreeze: List[str] = field(default_factory=list)
    validate_every: int = 1
    save_best_by: str = "val/loss/total"
    gradient_clip_val: Optional[float] = None
    save_every: int = 1
    save_best: bool = True
    save_last: bool = True
    save_epoch_checkpoints: bool = True
    eval_only: bool = False
    train_fraction: Optional[float] = None
    train_windows: Optional[int] = None
    few_shot_ratio: Optional[float] = None
    few_shot_windows: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 and not self.eval_only:
            raise ValueError(f"Stage {self.name!r} must run for at least one epoch")
        if self.epochs < 0:
            raise ValueError(f"Stage {self.name!r} epochs must be >= 0")
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
                "model": None if stage.model is None else stage.model.get("type", "<inherit>"),
                "data": None if stage.data is None else stage.data.get("type", "<inherit>"),
                "reset_model": stage.reset_model,
                "reset_data": stage.reset_data,
                "losses": [loss.get("type") for loss in stage.losses],
                "metrics": [metric.get("type") for metric in stage.metrics],
                "load_from": stage.load_from,
                "load_method": stage.load_method,
                "save_artifact": stage.save_artifact,
                "freeze": stage.freeze,
                "unfreeze": stage.unfreeze,
                "eval_only": stage.eval_only,
                "train_fraction": (
                    stage.train_fraction
                    if stage.train_fraction is not None
                    else stage.few_shot_ratio
                ),
                "train_windows": (
                    stage.train_windows
                    if stage.train_windows is not None
                    else stage.few_shot_windows
                ),
            }
            for stage in self.stages
        ]
