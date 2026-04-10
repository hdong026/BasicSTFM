"""Default multi-stage training engine."""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from basicstfm.engines.stage import StagePlan, StageSpec
from basicstfm.losses.common import LossCollection
from basicstfm.metrics.common import MetricCollection
from basicstfm.optim.factory import build_optimizer, build_scheduler
from basicstfm.registry import DATAMODULES, MODELS, TASKS, TRAINERS
from basicstfm.utils.checkpoint import load_checkpoint, save_checkpoint
from basicstfm.utils.logging import get_logger
from basicstfm.utils.seed import seed_everything


@TRAINERS.register("MultiStageTrainer")
class MultiStageTrainer:
    """A configurable runner for pretrain/finetune/evaluate pipelines."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        work_dir: Optional[str] = None,
        device: str = "auto",
        log_every: int = 20,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.log_every = int(log_every)
        self.dry_run = dry_run
        self.logger = get_logger()
        self.plan = StagePlan.from_config(cfg)
        self.work_dir = Path(work_dir or cfg.get("work_dir") or self._default_work_dir())
        self.device = self._resolve_device(device)
        self.datamodule = None
        self.model: Optional[torch.nn.Module] = None
        self.last_checkpoint: Optional[str] = None

    def run(self) -> None:
        seed_everything(int(self.cfg.get("seed", 42)))
        self.logger.info("Stage plan:\n%s", json.dumps(self.plan.describe(), indent=2))
        if self.dry_run:
            self.logger.info("Dry run complete. No model or data objects were built.")
            return

        self.setup()
        for stage in self.plan.stages:
            self.run_stage(stage)

    def setup(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Work directory: %s", self.work_dir)
        self.datamodule = DATAMODULES.build(self.cfg["data"])
        self.datamodule.setup()
        self.model = MODELS.build(self.cfg["model"]).to(self.device)
        self.logger.info("Model: %s", self.model.__class__.__name__)
        self.logger.info("Device: %s", self.device)

    def run_stage(self, stage: StageSpec) -> None:
        if self.model is None or self.datamodule is None:
            raise RuntimeError("Trainer.setup must run before run_stage")

        self.logger.info("Starting stage %s for %d epoch(s)", stage.name, stage.epochs)
        self._apply_trainability(stage)
        task = TASKS.build(stage.task)
        if hasattr(task, "set_scaler") and hasattr(self.datamodule, "get_scaler"):
            task.set_scaler(self.datamodule.get_scaler())
        losses = LossCollection(stage.losses).to(self.device)
        metrics = MetricCollection(stage.metrics).to(self.device)
        optimizer = build_optimizer(stage.optimizer, self.model.parameters())
        scheduler = build_scheduler(stage.scheduler, optimizer)

        if stage.load_from:
            ckpt_path = self._resolve_checkpoint_reference(stage.load_from)
            info = load_checkpoint(
                ckpt_path,
                self.model,
                strict=stage.strict_load,
                map_location=str(self.device),
            )
            self.logger.info(
                "Loaded checkpoint %s (missing=%s unexpected=%s)",
                ckpt_path,
                info["missing_keys"],
                info["unexpected_keys"],
            )

        best_score = float("inf")
        for epoch in range(1, stage.epochs + 1):
            score_for_scheduler: Optional[float] = None
            train_logs = self._run_loader(
                loader=self.datamodule.train_dataloader(),
                task=task,
                losses=losses,
                metrics=metrics,
                optimizer=optimizer,
                train=True,
                gradient_clip_val=stage.gradient_clip_val,
            )
            log_payload = {"epoch": epoch, **train_logs}

            if epoch % stage.validate_every == 0:
                val_logs = self._run_loader(
                    loader=self.datamodule.val_dataloader(),
                    task=task,
                    losses=losses,
                    metrics=metrics,
                    optimizer=None,
                    train=False,
                    gradient_clip_val=None,
                )
                log_payload.update(val_logs)
                score = float(val_logs.get(stage.save_best_by, val_logs.get("val/loss/total", best_score)))
                score_for_scheduler = score
                if score < best_score:
                    best_score = score
                    best_path = self.work_dir / "checkpoints" / f"{stage.name}_best.pt"
                    save_checkpoint(
                        str(best_path),
                        self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra={"stage": stage.name, "epoch": epoch, "score": best_score},
                    )
                    self.last_checkpoint = str(best_path)

            if scheduler is not None:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler_value = (
                        score_for_scheduler
                        if score_for_scheduler is not None
                        else log_payload["train/loss/total"]
                    )
                    scheduler.step(scheduler_value)
                else:
                    scheduler.step()

            self.logger.info(
                "Stage %s epoch %d: %s",
                stage.name,
                epoch,
                self._format_logs(log_payload),
            )

        last_path = self.work_dir / "checkpoints" / f"{stage.name}_last.pt"
        save_checkpoint(
            str(last_path),
            self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            extra={"stage": stage.name, "epoch": stage.epochs},
        )
        self.last_checkpoint = str(last_path)
        test_logs = self._run_loader(
            loader=self.datamodule.test_dataloader(),
            task=task,
            losses=losses,
            metrics=metrics,
            optimizer=None,
            train=False,
            gradient_clip_val=None,
            prefix="test",
        )
        self.logger.info("Finished stage %s: %s", stage.name, self._format_logs(test_logs))

    def _run_loader(
        self,
        loader,
        task,
        losses: LossCollection,
        metrics: MetricCollection,
        optimizer: Optional[torch.optim.Optimizer],
        train: bool,
        gradient_clip_val: Optional[float],
        prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        mode = "train" if train else "val"
        prefix = prefix or mode
        self.model.train(train)
        sums: Dict[str, float] = {}
        count = 0

        for batch_idx, batch in enumerate(loader, start=1):
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                output = task.step(self.model, batch, losses, self.device)
                loss = output["loss"]
                if train:
                    loss.backward()
                    if gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    optimizer.step()

            logs = dict(output.get("logs", {}))
            if metrics.items and "pred" in output and "target" in output:
                logs.update(metrics(output["pred"], output["target"], mask=output.get("mask")))
            for key, value in logs.items():
                scalar = float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value)
                sums[f"{prefix}/{key}"] = sums.get(f"{prefix}/{key}", 0.0) + scalar
            count += 1

            if train and self.log_every > 0 and batch_idx % self.log_every == 0:
                interim = {k: v / max(count, 1) for k, v in sums.items()}
                self.logger.info("%s batch %d: %s", prefix, batch_idx, self._format_logs(interim))

        if count == 0:
            return {}
        return {key: value / count for key, value in sums.items()}

    def _apply_trainability(self, stage: StageSpec) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if not stage.freeze and not stage.unfreeze:
            return
        for name, param in self.model.named_parameters():
            if _matches_any(name, stage.freeze):
                param.requires_grad = False
            if _matches_any(name, stage.unfreeze):
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Trainable parameters after stage rules: %d / %d", trainable, total)

    def _resolve_checkpoint_reference(self, reference: str) -> str:
        if reference == "previous":
            if not self.last_checkpoint:
                raise RuntimeError("stage.load_from='previous' requested before any checkpoint exists")
            return self.last_checkpoint
        return reference

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _default_work_dir(self) -> str:
        name = str(self.cfg.get("experiment_name", "basicstfm_experiment"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path("runs") / name / stamp)

    @staticmethod
    def _format_logs(logs: Dict[str, Any]) -> str:
        parts = []
        for key, value in sorted(logs.items()):
            if isinstance(value, float):
                parts.append(f"{key}={value:.5f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(pattern == "all" or fnmatch.fnmatch(name, pattern) for pattern in patterns)
