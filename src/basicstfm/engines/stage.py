"""Stage plan parsing for multi-stage training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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
    allow_stable_trunk_channel_inflate: bool = False
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
    early_stop_patience: Optional[int] = None

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
                "allow_stable_trunk_channel_inflate": stage.allow_stable_trunk_channel_inflate,
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

    @staticmethod
    def describe_factost_protocol_audit(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge ``data`` like ``MultiStageTrainer`` and surface scaler / RevIN / split flags."""

        plan = StagePlan.from_config(cfg)
        base_data = dict(cfg.get("data") or {})
        resolved_recipes: List[Dict[str, Any]] = []
        audit_rows: List[Dict[str, Any]] = []
        current = deepcopy(base_data)
        for stage in plan.stages:
            override = stage.data
            reset_requested = stage.reset_data
            if reset_requested:
                override_type = None if not isinstance(override, dict) else override.get("type")
                base_type = base_data.get("type")
                if override_type is not None and override_type != base_type:
                    current = {"type": override_type}
                else:
                    current = deepcopy(base_data)
            elif resolved_recipes:
                current = deepcopy(resolved_recipes[-1])
            else:
                current = deepcopy(base_data)

            if override is not None:
                current = _merge_dicts(current, override)
            resolved_recipes.append(deepcopy(current))

            task = dict(stage.task)
            scaler = current.get("scaler")
            scaler_type = scaler.get("type") if isinstance(scaler, dict) else None
            data_path = current.get("data_path")
            dataset_key = current.get("dataset_key") or current.get("name")
            if dataset_key is None and data_path:
                dataset_key = Path(str(data_path)).parent.name
            audit_rows.append(
                {
                    "name": stage.name,
                    "dataset_key": dataset_key,
                    "data_path": data_path,
                    "input_len": current.get("input_len"),
                    "output_len": current.get("output_len"),
                    "data_scaler_type": scaler_type,
                    "factost_split": bool(current.get("factost_split", False)),
                    "split_mode": current.get("split_mode"),
                    "split": list(current["split"]) if current.get("split") is not None else None,
                    "task_type": task.get("type"),
                    "eval_only": bool(stage.eval_only),
                    "use_revin": bool(task.get("use_revin", False)),
                    "factost_original_scale": bool(task.get("factost_original_scale", False)),
                    "basicts_scale_logging": bool(task.get("basicts_scale_logging", False)),
                    "primary_supervision_space": task.get("primary_supervision_space"),
                    "revin_loss_space": task.get("revin_loss_space"),
                    "revin_metric_space": task.get("revin_metric_space"),
                }
            )
        return audit_rows

    @staticmethod
    def describe_few_shot_protocol(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Surface few-shot stages, freeze rules, and ``model.few_shot`` for dry-run / audits."""

        plan = StagePlan.from_config(cfg)
        model_fs = (cfg.get("model") or {}).get("few_shot")
        rows: List[Dict[str, Any]] = []
        for st in plan.stages:
            if st.few_shot_ratio is None:
                continue
            rows.append(
                {
                    "stage": st.name,
                    "few_shot_ratio": st.few_shot_ratio,
                    "load_from": st.load_from,
                    "anchor_to_zero_shot": (
                        None if not isinstance(model_fs, dict) else model_fs.get("anchor_to_zero_shot")
                    ),
                    "freeze": st.freeze,
                    "unfreeze": st.unfreeze,
                }
            )
        return {
            "experiment": cfg.get("experiment"),
            "model_few_shot": model_fs,
            "few_shot_stages": rows,
        }
