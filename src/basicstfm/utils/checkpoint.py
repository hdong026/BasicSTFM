"""Checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def adapt_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align a checkpoint to the current model, expanding small node/channel embedding tables.

    Used when transfer stages use a larger graph or ``max_num_nodes`` than pretraining (e.g. 256 → 307).
    Extra rows keep the model's current initialization. Other shape mismatches raise.
    """

    model_state = model.state_dict()
    out: Dict[str, Any] = {}
    for name, param in model_state.items():
        if name not in checkpoint_state:
            out[name] = param
            continue
        tensor = checkpoint_state[name]
        if not isinstance(tensor, torch.Tensor):
            out[name] = tensor
            continue
        if tensor.shape == param.shape:
            out[name] = tensor
            continue
        if (
            tensor.ndim == 2
            and param.ndim == 2
            and tensor.shape[1] == param.shape[1]
            and param.shape[0] > tensor.shape[0]
            and (
                "node_emb" in name
                or "channel_emb" in name
            )
        ):
            buf = param.clone()
            src = tensor.to(device=buf.device, dtype=buf.dtype)
            buf[: src.shape[0]].copy_(src)
            out[name] = buf
        else:
            raise RuntimeError(
                f"Cannot load {name!r}: checkpoint shape {tuple(tensor.shape)} "
                f"vs model {tuple(param.shape)}"
            )
    return out


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "extra": extra or {},
        "rng_state": collect_rng_state(),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: str = "cpu",
    restore_rng: bool = False,
) -> Dict[str, Any]:
    ckpt = torch_load(path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)
    try:
        state_dict = adapt_checkpoint_state_dict(model, state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint incompatible with current model: {exc}"
        ) from exc
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "extra": ckpt.get("extra", {}),
    }


def read_checkpoint_metadata(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Read checkpoint metadata without requiring model construction."""

    ckpt = torch_load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        return dict(ckpt.get("extra", {}))
    return {}


def build_resume_model_dim_baseline(path: str) -> Dict[str, int]:
    """Load ``input_dim`` / ``output_dim`` / ``num_nodes`` from checkpoint ``extra`` or state tensors.

    Used on resume when earlier stages are skipped and the trainer must align model I/O
    with the last saved weights (e.g. fixed 18-d pretrain vs 3-d target data).
    """

    ckpt = torch_load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {}
    out: Dict[str, int] = {}
    extra = ckpt.get("extra") or {}
    for key in ("num_nodes", "input_dim", "output_dim"):
        if key in extra:
            try:
                out[key] = int(extra[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
    if len(out) >= 3:
        return out

    state = ckpt.get("model")
    if not isinstance(state, dict):
        return out

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "stable_trunk" in name and "local_branch.0.weight" in name and tensor.ndim == 4:
            try:
                in_ch = int(tensor.shape[1])
                out["input_dim"] = max(out.get("input_dim", 0), in_ch)
            except (TypeError, ValueError):
                pass
        if "node_emb" in name and name.endswith("weight") and tensor.ndim == 2:
            try:
                n = int(tensor.shape[0])
                out["num_nodes"] = max(out.get("num_nodes", 0), n)
            except (TypeError, ValueError):
                pass
        if "channel_emb" in name and name.endswith("weight") and tensor.ndim == 2:
            try:
                c = int(tensor.shape[0])
                out["output_dim"] = max(out.get("output_dim", 0), c)
            except (TypeError, ValueError):
                pass
    if out.get("input_dim") and not out.get("output_dim"):
        out["output_dim"] = out["input_dim"]
    if out.get("output_dim") and not out.get("input_dim"):
        out["input_dim"] = out["output_dim"]
    return out


def collect_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_to_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_to_byte_tensor(item) for item in state["torch_cuda"]])


def _to_byte_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


def torch_load(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
