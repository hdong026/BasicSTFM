"""Default multi-stage training engine."""

from __future__ import annotations

from copy import deepcopy
import fnmatch
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from basicstfm.engines.stage import StagePlan, StageSpec
from basicstfm.losses.common import LossCollection
from basicstfm.metrics.common import MetricCollection
from basicstfm.optim.factory import build_optimizer, build_scheduler
from basicstfm.registry import DATAMODULES, MODELS, TASKS, TRAINERS
from basicstfm.utils.checkpoint import (
    build_resume_model_dim_baseline,
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from basicstfm.utils.distributed import (
    DistributedContext,
    barrier,
    broadcast_string,
    cleanup_distributed,
    init_distributed,
    unwrap_model,
)
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
        eval_log_every: int = 50,
        resume_from: Optional[str] = None,
        auto_resume: bool = False,
        resume_strict: bool = True,
        dry_run: bool = False,
        strategy: str = "auto",
        backend: Optional[str] = None,
        find_unused_parameters: bool = True,
        broadcast_buffers: bool = False,
    ) -> None:
        self.cfg = cfg
        self.log_every = int(log_every)
        self.eval_log_every = int(eval_log_every)
        self.resume_from = resume_from
        self.auto_resume = auto_resume
        self.resume_strict = resume_strict
        self.dry_run = dry_run
        self.distributed: DistributedContext = (
            DistributedContext(enabled=False, strategy="single")
            if dry_run
            else init_distributed(strategy=strategy, backend=backend)
        )
        self.find_unused_parameters = bool(find_unused_parameters)
        self.broadcast_buffers = bool(broadcast_buffers)
        self.logger = get_logger(
            rank=self.distributed.rank,
            is_main_process=self.distributed.is_main_process,
        )
        self.plan = StagePlan.from_config(cfg)
        self.work_dir = Path(self._resolve_work_dir(work_dir or cfg.get("work_dir")))
        self.device = self._resolve_device(device)
        self.datamodule = None
        self.model: Optional[torch.nn.Module] = None
        self.base_data_cfg = dict(cfg["data"])
        self.base_model_cfg = dict(cfg["model"])
        self._current_data_cfg: Optional[Dict[str, Any]] = None
        self._current_model_cfg: Optional[Dict[str, Any]] = None
        self._stage_data_recipes = self._compile_stage_recipes(
            base_cfg=self.base_data_cfg,
            attr_name="data",
            reset_attr="reset_data",
        )
        self._stage_model_recipes = self._compile_stage_recipes(
            base_cfg=self.base_model_cfg,
            attr_name="model",
            reset_attr="reset_model",
        )
        self.artifacts: Dict[str, str] = {}
        self.last_checkpoint: Optional[str] = None
        self._resume_checkpoint: Optional[str] = None
        self._resume_metadata: Dict[str, Any] = {}
        self._resume_consumed = False
        # resume 时若跳过前序 stage，_current_model_cfg 可能未更新；用断点里/权重推断的 I/O 维与 datamodule 取 max
        self._resume_model_baseline: Dict[str, int] = {}
        self.stage_results: list[Dict[str, Any]] = []

    def run(self) -> None:
        try:
            seed_everything(int(self.cfg.get("seed", 42)))
            self.logger.info("Stage plan:\n%s", json.dumps(self.plan.describe(), indent=2))
            if self.dry_run:
                self.logger.info("Dry run complete. No model or data objects were built.")
                return

            self.setup()
            self._prepare_resume()
            resume_stage_index = self._resume_stage_index()
            for stage_index, stage in enumerate(self.plan.stages):
                if resume_stage_index is not None and stage_index < resume_stage_index:
                    self.logger.info(
                        "Skipping completed stage %s due to resume checkpoint",
                        stage.name,
                    )
                    continue
                self.run_stage(stage, stage_index)
        finally:
            cleanup_distributed(self.distributed)

    def setup(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "results").mkdir(parents=True, exist_ok=True)
        self.logger.info("Work directory: %s", self.work_dir)
        self.logger.info("Device: %s", self.device)
        if self.distributed.enabled:
            self.logger.info(
                "Distributed strategy: %s (backend=%s, world_size=%d, rank=%d, local_rank=%d)",
                self.distributed.strategy,
                self.distributed.backend,
                self.distributed.world_size,
                self.distributed.rank,
                self.distributed.local_rank,
            )

    def run_stage(self, stage: StageSpec, stage_index: int = 0) -> None:
        self._prepare_stage_components(stage, stage_index)
        if self.model is None or self.datamodule is None:
            raise RuntimeError("Stage components are not initialized")

        suffix = " [eval-only]" if stage.eval_only else ""
        self.logger.info(
            "Starting stage %s for %d epoch(s)%s",
            stage.name,
            stage.epochs,
            suffix,
        )
        self._apply_trainability(stage)
        task = TASKS.build(stage.task)
        if hasattr(task, "set_scaler") and hasattr(self.datamodule, "get_scaler"):
            task.set_scaler(self.datamodule.get_scaler())
        losses = LossCollection(stage.losses).to(self.device)
        metrics = MetricCollection(stage.metrics).to(self.device)
        optimizer = (
            None
            if stage.eval_only
            else build_optimizer(stage.optimizer, self.model.parameters())
        )
        scheduler = None if optimizer is None else build_scheduler(stage.scheduler, optimizer)
        start_epoch = 1
        best_score = float("inf")
        resumed_this_stage = False
        train_logs: Optional[Dict[str, float]] = None
        val_logs: Optional[Dict[str, float]] = None

        if self._should_resume_stage(stage, stage_index):
            ckpt_path = str(self._resume_checkpoint)
            info = self._load_stage_weights(
                ckpt_path,
                stage=stage,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=self.resume_strict,
                restore_rng=True,
            )
            extra = info.get("extra", {})
            start_epoch = int(extra.get("epoch", 0)) + 1
            best_score = float(extra.get("best_score", extra.get("score", best_score)))
            self.last_checkpoint = ckpt_path
            self._resume_consumed = True
            resumed_this_stage = True
            self.logger.info(
                "Resumed stage %s from %s at epoch %d (missing=%s unexpected=%s)",
                stage.name,
                ckpt_path,
                start_epoch,
                info["missing_keys"],
                info["unexpected_keys"],
            )

        if stage.load_from and not resumed_this_stage:
            ckpt_path = self._resolve_checkpoint_reference(stage.load_from)
            info = self._load_stage_weights(
                ckpt_path,
                stage=stage,
                strict=stage.strict_load,
                restore_rng=False,
            )
            self.logger.info(
                "Loaded stage weights from %s with method=%s (missing=%s unexpected=%s)",
                ckpt_path,
                stage.load_method,
                info["missing_keys"],
                info["unexpected_keys"],
            )

        if stage.eval_only:
            val_logs = self._run_eval_loaders(
                loader=self.datamodule.val_dataloader(),
                task=task,
                losses=losses,
                metrics=metrics,
                optimizer=None,
                train=False,
                gradient_clip_val=None,
                prefix="val",
            )
            if val_logs:
                self.logger.info(
                    "Eval-only stage %s validation: %s",
                    stage.name,
                    self._format_logs(val_logs),
                )
            test_logs = self._run_eval_loaders(
                loader=self.datamodule.test_dataloader(),
                task=task,
                losses=losses,
                metrics=metrics,
                optimizer=None,
                train=False,
                gradient_clip_val=None,
                prefix="test",
            )
            self.logger.info(
                "Finished eval-only stage %s: %s",
                stage.name,
                self._format_logs(test_logs),
            )
            self._record_stage_result(
                stage=stage,
                stage_index=stage_index,
                train_logs=None,
                val_logs=val_logs,
                test_logs=test_logs,
                best_score=None,
            )
            return

        if start_epoch > stage.epochs:
            self.logger.info(
                "Stage %s already reached epoch %d; no remaining epochs to run",
                stage.name,
                stage.epochs,
            )
            return

        for epoch in range(start_epoch, stage.epochs + 1):
            if self.datamodule is not None and hasattr(self.datamodule, "set_epoch"):
                self.datamodule.set_epoch(epoch)
            self.logger.info("=" * 88)
            self.logger.info(
                "Stage %s | Epoch %d/%d",
                stage.name,
                epoch,
                stage.epochs,
            )
            self.logger.info("=" * 88)
            score_for_scheduler: Optional[float] = None
            train_logs = self._run_loader(
                loader=self._train_dataloader_for_stage(stage),
                task=task,
                losses=losses,
                metrics=metrics,
                optimizer=optimizer,
                train=True,
                gradient_clip_val=stage.gradient_clip_val,
            )
            log_payload = {"epoch": epoch, **train_logs}

            if epoch % stage.validate_every == 0:
                val_logs = self._run_eval_loaders(
                    loader=self.datamodule.val_dataloader(),
                    task=task,
                    losses=losses,
                    metrics=metrics,
                    optimizer=None,
                    train=False,
                    gradient_clip_val=None,
                )
                log_payload.update(val_logs)
                score = float(
                    val_logs.get(
                        stage.save_best_by,
                        val_logs.get("val/loss/total", best_score),
                    )
                )
                score_for_scheduler = score
                if score < best_score:
                    best_score = score
                    best_path = self.work_dir / "checkpoints" / f"{stage.name}_best.pt"
                    if stage.save_best:
                        self._save_checkpoint(
                            path=best_path,
                            stage=stage,
                            stage_index=stage_index,
                            epoch=epoch,
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            score=score,
                            best_score=best_score,
                            tag="best",
                        )

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
            self.logger.info("-" * 88)

            if epoch % stage.save_every == 0:
                if stage.save_epoch_checkpoints:
                    epoch_path = self.work_dir / "checkpoints" / f"{stage.name}_{epoch:03d}.pt"
                    self._save_checkpoint(
                        path=epoch_path,
                        stage=stage,
                        stage_index=stage_index,
                        epoch=epoch,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        score=score_for_scheduler,
                        best_score=best_score,
                        tag="epoch",
                    )
                if stage.save_last:
                    last_path = self.work_dir / "checkpoints" / f"{stage.name}_last.pt"
                    self._save_checkpoint(
                        path=last_path,
                        stage=stage,
                        stage_index=stage_index,
                        epoch=epoch,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        score=score_for_scheduler,
                        best_score=best_score,
                        tag="last",
                    )
                latest_path = self.work_dir / "checkpoints" / "latest.pt"
                self._save_checkpoint(
                    path=latest_path,
                    stage=stage,
                    stage_index=stage_index,
                    epoch=epoch,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    score=score_for_scheduler,
                    best_score=best_score,
                    tag="latest",
                )

        last_path = self.work_dir / "checkpoints" / f"{stage.name}_last.pt"
        if stage.save_last and stage.epochs % stage.save_every != 0:
            self._save_checkpoint(
                path=last_path,
                stage=stage,
                stage_index=stage_index,
                epoch=stage.epochs,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                score=None,
                best_score=best_score,
                tag="last",
            )
        test_logs = self._run_eval_loaders(
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
        self._record_stage_result(
            stage=stage,
            stage_index=stage_index,
            train_logs=train_logs,
            val_logs=val_logs,
            test_logs=test_logs,
            best_score=best_score,
        )

    @staticmethod
    def _loader_iter_length(loader) -> Optional[int]:
        try:
            n = len(loader)
        except TypeError:
            return None
        return int(n) if n > 0 else None

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
        loader_scope: Optional[str] = None,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        mode = "train" if train else "val"
        prefix = prefix or mode
        self.model.train(train)
        sums: Dict[str, float] = {}
        count = 0
        total_batches = self._loader_iter_length(loader) if not train else None

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
                scalar = float(
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                sums[f"{prefix}/{key}"] = sums.get(f"{prefix}/{key}", 0.0) + scalar
            count += 1

            if train and self.log_every > 0 and batch_idx % self.log_every == 0:
                interim = {k: v / max(count, 1) for k, v in sums.items()}
                self.logger.info(
                    "%s batch %d: %s",
                    prefix,
                    batch_idx,
                    self._format_logs(interim),
                )
            elif (
                not train
                and self.eval_log_every > 0
                and batch_idx % self.eval_log_every == 0
            ):
                interim = {k: v / max(count, 1) for k, v in sums.items()}
                scope = f"[{loader_scope}] " if loader_scope else ""
                if total_batches is not None:
                    pct = 100.0 * float(batch_idx) / float(total_batches)
                    self.logger.info(
                        "%s%s batch %d/%d (%.1f%%): %s",
                        scope,
                        prefix,
                        batch_idx,
                        total_batches,
                        pct,
                        self._format_logs(interim),
                    )
                else:
                    self.logger.info(
                        "%s%s batch %d: %s",
                        scope,
                        prefix,
                        batch_idx,
                        self._format_logs(interim),
                    )

        if count == 0:
            return {}
        if self.distributed.enabled:
            count_tensor = torch.tensor(float(count), device=self.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            reduced_count = max(float(count_tensor.item()), 1.0)
            reduced_logs = {}
            for key in sorted(sums):
                tensor = torch.tensor(float(sums[key]), device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_logs[key] = float(tensor.item()) / reduced_count
            return reduced_logs
        return {key: value / count for key, value in sums.items()}

    def _run_eval_loaders(
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
        if isinstance(loader, Mapping):
            named_logs: Dict[str, Dict[str, float]] = {}
            for dataset_name, subloader in loader.items():
                named_logs[dataset_name] = self._run_loader(
                    loader=subloader,
                    task=task,
                    losses=losses,
                    metrics=metrics,
                    optimizer=optimizer,
                    train=train,
                    gradient_clip_val=gradient_clip_val,
                    prefix=prefix,
                    loader_scope=dataset_name,
                )
            return self._aggregate_named_eval_logs(named_logs, prefix=prefix or ("train" if train else "val"))
        return self._run_loader(
            loader=loader,
            task=task,
            losses=losses,
            metrics=metrics,
            optimizer=optimizer,
            train=train,
            gradient_clip_val=gradient_clip_val,
            prefix=prefix,
        )

    @staticmethod
    def _aggregate_named_eval_logs(
        named_logs: Dict[str, Dict[str, float]],
        prefix: str,
    ) -> Dict[str, float]:
        aggregate: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for dataset_name, logs in named_logs.items():
            for key, value in logs.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
                suffix = key[len(prefix) + 1 :] if key.startswith(f"{prefix}/") else key
                aggregate[f"{prefix}/dataset/{dataset_name}/{suffix}"] = float(value)
        for key, total in list(aggregate.items()):
            if key.startswith(f"{prefix}/dataset/"):
                continue
            aggregate[key] = total / max(counts.get(key, 1), 1)
        return aggregate

    def _train_dataloader_for_stage(self, stage: StageSpec):
        kwargs = {
            "train_fraction": stage.train_fraction,
            "train_windows": stage.train_windows,
            "few_shot_ratio": stage.few_shot_ratio,
            "few_shot_windows": stage.few_shot_windows,
        }
        if all(value is None for value in kwargs.values()):
            return self.datamodule.train_dataloader()
        return self.datamodule.train_dataloader(**kwargs)

    def _prepare_stage_components(self, stage: StageSpec, stage_index: int) -> None:
        data_cfg = self._compose_stage_data_config(stage_index)
        data_cfg = self._inject_runtime_data_kwargs(data_cfg)
        if self.datamodule is None or self._current_data_cfg != data_cfg:
            self.datamodule = DATAMODULES.build(data_cfg)
            self.datamodule.setup()
            self._current_data_cfg = deepcopy(data_cfg)
            self.logger.info(
                "Stage %s data config: %s",
                stage.name,
                json.dumps(data_cfg, sort_keys=True),
            )

        model_cfg = self._compose_stage_model_config(stage_index)
        if self.model is None or self._current_model_cfg != model_cfg:
            model = MODELS.build(model_cfg).to(self.device)
            self.model = self._wrap_model(model)
            self._current_model_cfg = deepcopy(model_cfg)
            self.logger.info("Model: %s", unwrap_model(self.model).__class__.__name__)
            self.logger.info("Resolved model config: %s", json.dumps(model_cfg, sort_keys=True))

    def _compose_stage_data_config(self, stage_index: int) -> Dict[str, Any]:
        return deepcopy(self._stage_data_recipes[stage_index])

    def _compose_stage_model_config(self, stage_index: int) -> Dict[str, Any]:
        preserve_current = not self.plan.stages[stage_index].reset_model
        return self._resolve_model_config(
            deepcopy(self._stage_model_recipes[stage_index]),
            preserve_current=preserve_current,
        )

    def _compile_stage_recipes(
        self,
        base_cfg: Dict[str, Any],
        attr_name: str,
        reset_attr: str,
    ) -> list[Dict[str, Any]]:
        recipes: list[Dict[str, Any]] = []
        current = deepcopy(base_cfg)
        for stage in self.plan.stages:
            override = getattr(stage, attr_name)
            reset_requested = bool(getattr(stage, reset_attr))
            if reset_requested:
                override_type = None if not isinstance(override, dict) else override.get("type")
                base_type = base_cfg.get("type")
                if override_type is not None and override_type != base_type:
                    current = {"type": override_type}
                else:
                    current = deepcopy(base_cfg)
            elif recipes:
                current = deepcopy(recipes[-1])
            else:
                current = deepcopy(base_cfg)

            if (
                override is not None
                and "type" in override
                and current.get("type") not in {None, override["type"]}
                and not reset_requested
            ):
                raise ValueError(
                    f"Stage {stage.name!r} changes {attr_name} type from "
                    f"{current.get('type')!r} to {override['type']!r}. "
                    f"Set {reset_attr}: true to start from a fresh recipe."
                )

            if override is not None:
                current = _merge_dicts(current, override)
            recipes.append(deepcopy(current))
        return recipes

    def _load_stage_weights(
        self,
        path: str,
        stage: StageSpec,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
        restore_rng: bool = False,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        model = unwrap_model(self.model)
        if stage.load_method in {"checkpoint", "state_dict"}:
            return load_checkpoint(
                path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=strict,
                map_location=str(self.device),
                restore_rng=restore_rng,
            )

        loader = getattr(model, stage.load_method, None)
        if loader is None:
            raise AttributeError(
                f"Model {model.__class__.__name__} has no load method {stage.load_method!r}"
            )
        try:
            result = loader(path, strict=strict)
        except TypeError:
            result = loader(path)

        info = {
            "missing_keys": [],
            "unexpected_keys": [],
            "extra": read_checkpoint_metadata(path, map_location="cpu"),
        }
        if isinstance(result, tuple) and len(result) == 2:
            info["missing_keys"] = list(result[0])
            info["unexpected_keys"] = list(result[1])
        elif isinstance(result, dict):
            info["missing_keys"] = list(result.get("missing_keys", []))
            info["unexpected_keys"] = list(result.get("unexpected_keys", []))
            if "extra" in result:
                info["extra"] = dict(result["extra"])
        return info

    def _apply_trainability(self, stage: StageSpec) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        model = unwrap_model(self.model)
        for param in model.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if _matches_any(name, stage.freeze):
                param.requires_grad = False
            if _matches_any(name, stage.unfreeze):
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logger.info("Trainable parameters after stage rules: %d / %d", trainable, total)

    def _resolve_checkpoint_reference(self, reference: str) -> str:
        if reference == "previous":
            if not self.last_checkpoint:
                raise RuntimeError(
                    "stage.load_from='previous' requested before any checkpoint exists"
                )
            return self.last_checkpoint
        if reference in self.artifacts:
            return self.artifacts[reference]
        candidate = Path(reference)
        if candidate.exists():
            return str(candidate)
        checkpoint_dir = self.work_dir / "checkpoints"
        for suffix in ("_last.pt", "_best.pt", ".pt"):
            resolved = checkpoint_dir / f"{reference}{suffix}"
            if resolved.exists():
                return str(resolved)
        # Resume skips early stages, so ``self.artifacts[save_artifact]`` may be empty.
        # Checkpoints are named after ``stage.name``, while YAML often uses ``save_artifact``.
        for stage in self.plan.stages:
            if stage.save_artifact == reference:
                for suffix in ("_last.pt", "_best.pt"):
                    resolved = checkpoint_dir / f"{stage.name}{suffix}"
                    if resolved.exists():
                        return str(resolved)
        return reference

    def _resolve_model_config(
        self,
        model_cfg: Dict[str, Any],
        preserve_current: bool = True,
    ) -> Dict[str, Any]:
        if self.datamodule is None or not hasattr(self.datamodule, "get_metadata"):
            return model_cfg

        metadata = self.datamodule.get_metadata()
        inferred = {
            "num_nodes": metadata["num_nodes"],
            "input_dim": metadata["num_channels"],
            "output_dim": metadata["num_channels"],
            "input_len": metadata["input_len"],
            "output_len": metadata["target_len"],
        }
        for key, value in inferred.items():
            if key not in model_cfg or _is_auto(model_cfg[key]):
                if key in ("num_nodes", "input_dim", "output_dim"):
                    candidates: list[int] = [int(value)]
                    if self._current_model_cfg and not _is_auto(
                        self._current_model_cfg.get(key)
                    ):
                        try:
                            candidates.append(int(self._current_model_cfg[key]))
                        except (TypeError, ValueError):
                            pass
                    b = self._resume_model_baseline
                    if b and key in b:
                        try:
                            candidates.append(int(b[key]))
                        except (TypeError, ValueError):
                            pass
                    model_cfg[key] = max(candidates)
                elif preserve_current and self._current_model_cfg and not _is_auto(
                    self._current_model_cfg.get(key)
                ):
                    model_cfg[key] = self._current_model_cfg[key]
                else:
                    model_cfg[key] = value
        return model_cfg

    def _prepare_resume(self) -> None:
        checkpoint = self.resume_from
        if not checkpoint and self.auto_resume:
            latest = self.work_dir / "checkpoints" / "latest.pt"
            if latest.exists():
                checkpoint = str(latest)
        if not checkpoint:
            return
        path = Path(checkpoint)
        if path.is_dir():
            path = path / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        self._resume_checkpoint = str(path)
        self._load_stage_results()
        self._resume_metadata = read_checkpoint_metadata(str(path), map_location="cpu")
        self._resume_model_baseline = build_resume_model_dim_baseline(str(path))
        if self._resume_model_baseline:
            self.logger.info("Resume model dim baseline: %s", self._resume_model_baseline)
        stage_name = self._resume_metadata.get("stage")
        if stage_name:
            self.artifacts[str(stage_name)] = str(path)
        save_artifact = self._resume_metadata.get("save_artifact")
        if save_artifact:
            self.artifacts[str(save_artifact)] = str(path)
        self.logger.info("Resume checkpoint selected: %s", path)

    def _resume_stage_index(self) -> Optional[int]:
        if not self._resume_checkpoint:
            return None
        value = self._resume_metadata.get("stage_index")
        if value is not None:
            return int(value)
        stage_name = self._resume_metadata.get("stage")
        if stage_name is None:
            return None
        for index, stage in enumerate(self.plan.stages):
            if stage.name == stage_name:
                return index
        return None

    def _should_resume_stage(self, stage: StageSpec, stage_index: int) -> bool:
        if not self._resume_checkpoint or self._resume_consumed:
            return False
        resume_stage_index = self._resume_stage_index()
        if resume_stage_index is not None:
            return stage_index == resume_stage_index
        resume_stage_name = self._resume_metadata.get("stage")
        if resume_stage_name is not None:
            return stage.name == resume_stage_name
        return stage_index == 0

    def _save_checkpoint(
        self,
        path: Path,
        stage: StageSpec,
        stage_index: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        score: Optional[float],
        best_score: float,
        tag: str,
    ) -> None:
        extra: Dict[str, Any] = {
            "stage": stage.name,
            "stage_index": stage_index,
            "epoch": epoch,
            "score": score,
            "best_score": best_score,
            "tag": tag,
            "work_dir": str(self.work_dir),
            "save_artifact": stage.save_artifact,
        }
        if self._current_model_cfg:
            for k in ("num_nodes", "input_dim", "output_dim"):
                v = self._current_model_cfg.get(k)
                if v is not None and not _is_auto(v):
                    try:
                        extra[k] = int(v)
                    except (TypeError, ValueError):
                        pass
        if self.distributed.is_main_process:
            save_checkpoint(
                str(path),
                unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                extra=extra,
            )
        barrier(self.distributed)
        self.last_checkpoint = str(path)
        if tag == "last":
            self.artifacts[stage.name] = str(path)
            if stage.save_artifact:
                self.artifacts[stage.save_artifact] = str(path)
        self.logger.info("Checkpoint %s saved", path)

    def _record_stage_result(
        self,
        stage: StageSpec,
        stage_index: int,
        train_logs: Optional[Dict[str, float]],
        val_logs: Optional[Dict[str, float]],
        test_logs: Optional[Dict[str, float]],
        best_score: Optional[float],
    ) -> None:
        result = {
            "name": stage.name,
            "stage_index": stage_index,
            "task": stage.task.get("type"),
            "model_type": None if self._current_model_cfg is None else self._current_model_cfg.get("type"),
            "data_type": None if self._current_data_cfg is None else self._current_data_cfg.get("type"),
            "epochs": stage.epochs,
            "eval_only": stage.eval_only,
            "load_from": stage.load_from,
            "load_method": stage.load_method,
            "save_artifact": stage.save_artifact,
            "freeze": list(stage.freeze),
            "unfreeze": list(stage.unfreeze),
            "train_fraction": (
                stage.train_fraction if stage.train_fraction is not None else stage.few_shot_ratio
            ),
            "train_windows": (
                stage.train_windows if stage.train_windows is not None else stage.few_shot_windows
            ),
            "best_score": None if best_score is None or best_score == float("inf") else float(best_score),
            "checkpoint": self.last_checkpoint,
            "resolved_model": deepcopy(self._current_model_cfg) if self._current_model_cfg else None,
            "resolved_data": deepcopy(self._current_data_cfg) if self._current_data_cfg else None,
            "train": None if train_logs is None else {k: float(v) for k, v in train_logs.items()},
            "val": None if val_logs is None else {k: float(v) for k, v in val_logs.items()},
            "test": None if test_logs is None else {k: float(v) for k, v in test_logs.items()},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for index, item in enumerate(self.stage_results):
            if item.get("name") == stage.name:
                self.stage_results[index] = result
                break
        else:
            self.stage_results.append(result)
        if self.distributed.is_main_process:
            self._write_stage_results()

    def _stage_results_path(self) -> Path:
        return self.work_dir / "results" / "stage_results.json"

    def _load_stage_results(self) -> None:
        path = self._stage_results_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.logger.warning("Could not parse existing stage results file: %s", path)
            return
        if isinstance(payload, dict) and isinstance(payload.get("stages"), list):
            self.stage_results = list(payload["stages"])
        elif isinstance(payload, list):
            self.stage_results = list(payload)

    def _write_stage_results(self) -> None:
        payload = {
            "experiment_name": str(self.cfg.get("experiment_name", "basicstfm_experiment")),
            "work_dir": str(self.work_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "stages": self.stage_results,
        }
        self._stage_results_path().write_text(
            json.dumps(payload, indent=2, sort_keys=False),
            encoding="utf-8",
        )

    def _resolve_device(self, device: str) -> torch.device:
        if self.distributed.enabled and torch.cuda.is_available():
            if device in {"auto", "cuda"}:
                return torch.device(f"cuda:{self.distributed.local_rank}")
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_work_dir(self, work_dir: Optional[str]) -> str:
        if work_dir:
            return str(work_dir)
        if self.distributed.enabled:
            value = self._default_work_dir() if self.distributed.is_main_process else None
            resolved = broadcast_string(self.distributed, value)
            if resolved is None:
                raise RuntimeError("Failed to broadcast distributed work directory")
            return resolved
        return self._default_work_dir()

    def _default_work_dir(self) -> str:
        name = str(self.cfg.get("experiment_name", "basicstfm_experiment"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path("runs") / name / stamp)

    def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.distributed.enabled:
            return model
        kwargs = {
            "find_unused_parameters": self.find_unused_parameters,
            "broadcast_buffers": self.broadcast_buffers,
        }
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.distributed.local_rank],
                output_device=self.distributed.local_rank,
                **kwargs,
            )
        return DistributedDataParallel(model, **kwargs)

    def _inject_runtime_data_kwargs(self, data_cfg: Dict[str, Any]) -> Dict[str, Any]:
        data_type = data_cfg.get("type")
        if data_type is None or data_type not in DATAMODULES:
            return data_cfg
        constructor = DATAMODULES.get(str(data_type))
        try:
            parameters = inspect.signature(constructor).parameters
        except (TypeError, ValueError):
            return data_cfg

        runtime_values = {
            "distributed": self.distributed.enabled,
            "world_size": self.distributed.world_size,
            "rank": self.distributed.rank,
            "seed": int(self.cfg.get("seed", 42)),
        }
        merged = deepcopy(data_cfg)
        for key, value in runtime_values.items():
            if key in parameters:
                merged[key] = value
        return merged

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


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
