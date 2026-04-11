"""Helpers for single-node distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    strategy: str
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: Optional[str] = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(
    strategy: str = "auto",
    backend: Optional[str] = None,
) -> DistributedContext:
    strategy = str(strategy).lower()
    if strategy in {"single", "none", "off", "disabled"}:
        return DistributedContext(enabled=False, strategy="single")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if strategy == "auto" and world_size <= 1:
        return DistributedContext(enabled=False, strategy="single")
    if strategy == "ddp" and world_size <= 1:
        raise ValueError(
            "trainer.strategy='ddp' requires a distributed launch. "
            "Use torchrun --nproc_per_node=<NUM_GPUS> -m basicstfm ..."
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=selected_backend, init_method="env://")

    return DistributedContext(
        enabled=True,
        strategy="ddp",
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=selected_backend,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_string(context: DistributedContext, value: Optional[str]) -> Optional[str]:
    if not context.enabled or not dist.is_available() or not dist.is_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

