"""Checkpoint helpers."""

from __future__ import annotations

import fnmatch
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InflateOutcome:
    """Result of applying stable-trunk channel inflate for one parameter."""

    tensor: torch.Tensor
    used_checkpoint: bool


def _stable_trunk_prefix(name: str) -> bool:
    return name.startswith("stable_trunk.") or ".stable_trunk." in name


def _get_stable_trunk(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    return getattr(model, "stable_trunk", None)


def _inflate_local_branch_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (
        tensor.ndim == 4
        and param.ndim == 4
        and tensor.shape[0] == param.shape[0]
        and tensor.shape[1] == 1
        and param.shape[1] > 1
        and tensor.shape[2:] == param.shape[2:]
    ):
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    buf[:, 0:1, :, :].copy_(src)
    buf[:, 1:, :, :].copy_(param[:, 1:, :, :] * 0.01)
    return buf


def _inflate_linear_in_features_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (
        tensor.ndim == 2
        and param.ndim == 2
        and tensor.shape[0] == param.shape[0]
        and tensor.shape[1] == 1
        and param.shape[1] > 1
    ):
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    buf[:, 0:1].copy_(src)
    buf[:, 1:].copy_(param[:, 1:] * 0.01)
    return buf


def _inflate_reconstruction_head_weight_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (
        tensor.ndim == 2
        and param.ndim == 2
        and tensor.shape[0] == 1
        and param.shape[0] > 1
        and tensor.shape[1] == param.shape[1]
    ):
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    buf[0:1, :].copy_(src)
    buf[1:, :].copy_(param[1:, :] * 0.01)
    return buf


def _inflate_reconstruction_head_bias_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (tensor.ndim == 1 and param.ndim == 1 and tensor.shape[0] == 1 and param.shape[0] > 1):
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    buf[0:1].copy_(src)
    buf[1:].copy_(param[1:] * 0.01)
    return buf


def _align_dataset_modulation_embedding_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    """Truncate or pad (along num_datasets) for ``DatasetConditionedFiLM.embedding`` weights."""

    if tensor.ndim != 2 or param.ndim != 2 or tensor.shape[1] != param.shape[1]:
        return None
    ck_r, m_r = int(tensor.shape[0]), int(param.shape[0])
    if ck_r == m_r:
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    n = min(ck_r, m_r)
    buf[:n].copy_(src[:n])
    return buf


def _inflate_square_matrix_top_left_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    """Copy top-left ``min(r,c)`` block (e.g. ``calibration_head`` when ``output_dim`` grows)."""

    if tensor.ndim != 2 or param.ndim != 2:
        return None
    tr, tc = int(tensor.shape[0]), int(tensor.shape[1])
    pr, pc = int(param.shape[0]), int(param.shape[1])
    if tr != tc or pr != pc:
        return None
    if tr == pr:
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    n = min(tr, pr)
    buf[:n, :n].copy_(src[:n, :n])
    return buf


def _inflate_calibration_bias_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (tensor.ndim == 1 and param.ndim == 1 and tensor.shape[0] == 1 and param.shape[0] > 1):
        return None
    buf = param.clone()
    buf[0:1].copy_(tensor.to(device=buf.device, dtype=buf.dtype))
    return buf


def _inflate_fusion_additive_logit_ck(tensor: torch.Tensor, param: torch.Tensor) -> Optional[torch.Tensor]:
    if not (
        tensor.ndim == 4
        and param.ndim == 4
        and tensor.shape[:3] == param.shape[:3]
        and tensor.shape[3] == 1
        and param.shape[3] > 1
    ):
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    buf[..., 0:1].copy_(src)
    buf[..., 1:].copy_(param[..., 1:] * 0.01)
    return buf


def _try_forecast_head_last_inflate(
    name: str,
    tensor: torch.Tensor,
    param: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[_InflateOutcome]:
    if not (
        tensor.ndim == 2
        and param.ndim == 2
        and tensor.shape[1] == param.shape[1]
    ):
        return None
    ck_rows = int(tensor.shape[0])
    m_rows = int(param.shape[0])
    if m_rows % ck_rows != 0:
        return None
    c_new = m_rows // ck_rows
    if c_new <= 1:
        return None
    trunk = _get_stable_trunk(model)
    if trunk is not None:
        exp_len = int(getattr(trunk, "output_len", -1))
        exp_dim = int(getattr(trunk, "output_dim", 0))
        if exp_len >= 1 and ck_rows != exp_len:
            logger.warning(
                "stable_trunk forecast_head: checkpoint output rows %d != model output_len %d; "
                "skip inflate for %s (keep current initialization).",
                ck_rows,
                exp_len,
                name,
            )
            return _InflateOutcome(param.clone(), False)
        if exp_dim >= 1 and exp_len >= 1 and m_rows != exp_len * exp_dim:
            logger.warning(
                "stable_trunk forecast_head: model weight rows %d != output_len*output_dim (%d*%d); "
                "skip inflate for %s.",
                m_rows,
                exp_len,
                exp_dim,
                name,
            )
            return _InflateOutcome(param.clone(), False)

    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    for t in range(ck_rows):
        buf[t * c_new, :].copy_(src[t, :])
        for c in range(1, c_new):
            buf[t * c_new + c, :].copy_(param[t * c_new + c, :] * 0.01)
    return _InflateOutcome(buf, True)


def _try_forecast_head_last_bias_inflate(
    name: str,
    tensor: torch.Tensor,
    param: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[_InflateOutcome]:
    if not (tensor.ndim == 1 and param.ndim == 1):
        return None
    ck_len = int(tensor.shape[0])
    m_len = int(param.shape[0])
    if m_len % ck_len != 0:
        return None
    c_new = m_len // ck_len
    if c_new <= 1:
        return None
    trunk = _get_stable_trunk(model)
    if trunk is not None:
        exp_len = int(getattr(trunk, "output_len", -1))
        exp_dim = int(getattr(trunk, "output_dim", 0))
        if exp_len >= 1 and ck_len != exp_len:
            logger.warning(
                "stable_trunk forecast_head: checkpoint bias len %d != model output_len %d; "
                "skip inflate for %s (keep current initialization).",
                ck_len,
                exp_len,
                name,
            )
            return _InflateOutcome(param.clone(), False)
        if exp_dim >= 1 and exp_len >= 1 and m_len != exp_len * exp_dim:
            logger.warning(
                "stable_trunk forecast_head: model bias len %d != output_len*output_dim (%d*%d); "
                "skip inflate for %s.",
                m_len,
                exp_len,
                exp_dim,
                name,
            )
            return _InflateOutcome(param.clone(), False)

    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    for t in range(ck_len):
        buf[t * c_new].copy_(src[t])
        for c in range(1, c_new):
            buf[t * c_new + c].copy_(param[t * c_new + c] * 0.01)
    return _InflateOutcome(buf, True)


def _wrap_optional(
    maybe_tensor: Optional[torch.Tensor],
) -> Optional[_InflateOutcome]:
    if maybe_tensor is None:
        return None
    return _InflateOutcome(maybe_tensor, True)


def _forecast_head_last_linear_idx(model: torch.nn.Module) -> Optional[int]:
    trunk = _get_stable_trunk(model)
    if trunk is None or not hasattr(trunk, "forecast_head"):
        return None
    fh = trunk.forecast_head
    if not isinstance(fh, torch.nn.Sequential) or len(fh) < 1:
        return None
    return len(fh) - 1


def _is_forecast_head_last_weight(name: str, model: torch.nn.Module) -> bool:
    idx = _forecast_head_last_linear_idx(model)
    return idx is not None and name.endswith(f".forecast_head.{idx}.weight")


def _log_stable_trunk_inflate_line(
    name: str,
    ck_tensor: torch.Tensor,
    param: torch.Tensor,
    model: torch.nn.Module,
) -> None:
    if name.endswith(".local_branch.0.weight"):
        logger.info(
            "inflated stable_trunk.local_branch.0.weight: %d -> %d",
            int(ck_tensor.shape[1]),
            int(param.shape[1]),
        )
    elif name.endswith(".coarse_branch.0.weight"):
        logger.info(
            "inflated stable_trunk.coarse_branch.0.weight: %d -> %d",
            int(ck_tensor.shape[1]),
            int(param.shape[1]),
        )
    elif name.endswith(".frequency_branch.0.weight"):
        logger.info(
            "inflated stable_trunk.frequency_branch.0.weight: %d -> %d",
            int(ck_tensor.shape[1]),
            int(param.shape[1]),
        )
    elif name.endswith(".reconstruction_head.weight"):
        logger.info(
            "inflated stable_trunk.reconstruction_head: %d -> %d",
            int(ck_tensor.shape[0]),
            int(param.shape[0]),
        )
    elif name.endswith("residual_event_encoder.value_proj.weight"):
        logger.info(
            "inflated residual_event_encoder.value_proj.weight: %d -> %d",
            int(ck_tensor.shape[1]),
            int(param.shape[1]),
        )
    elif name.endswith(".dataset_modulation.embedding.weight"):
        logger.info(
            "aligned dataset_modulation.embedding.weight rows: %d -> %d",
            int(ck_tensor.shape[0]),
            int(param.shape[0]),
        )
    elif name.endswith("diffusion_mechanism_learner.output_proj.weight"):
        logger.info(
            "inflated diffusion_mechanism_learner.output_proj: %d -> %d",
            int(ck_tensor.shape[0]),
            int(param.shape[0]),
        )
    elif name.endswith("calibration_head.weight"):
        logger.info(
            "aligned calibration_head.weight size: %d -> %d",
            int(ck_tensor.shape[0]),
            int(param.shape[0]),
        )
    elif name.endswith("fusion_predictor.additive_logit"):
        logger.info(
            "inflated fusion_predictor.additive_logit channels: %d -> %d",
            int(ck_tensor.shape[3]),
            int(param.shape[3]),
        )
    elif _is_forecast_head_last_weight(name, model):
        trunk = _get_stable_trunk(model)
        if trunk is not None:
            c_old = int(ck_tensor.shape[0]) // int(trunk.output_len)
            c_new = int(param.shape[0]) // int(trunk.output_len)
        else:
            c_old = 1
            m = int(param.shape[0])
            ck = int(ck_tensor.shape[0])
            c_new = m // ck if ck > 0 else m
        logger.info("inflated stable_trunk.forecast_head: %d -> %d", c_old, c_new)


def _foundation_forecast_head_row_expand_weight(
    tensor: torch.Tensor,
    param: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Expand forecast head out-features (e.g. ``output_len * 1`` -> ``output_len * C``)."""

    if not (
        tensor.ndim == 2
        and param.ndim == 2
        and tensor.shape[1] == param.shape[1]
    ):
        return None
    ck_rows = int(tensor.shape[0])
    m_rows = int(param.shape[0])
    if m_rows <= ck_rows or m_rows % ck_rows != 0:
        return None
    factor = m_rows // ck_rows
    if factor <= 1:
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    for t in range(ck_rows):
        buf[t * factor, :].copy_(src[t, :])
        for c in range(1, factor):
            buf[t * factor + c, :].copy_(param[t * factor + c, :] * 0.01)
    return buf


def _foundation_forecast_head_row_expand_bias(
    tensor: torch.Tensor,
    param: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not (tensor.ndim == 1 and param.ndim == 1):
        return None
    ck_len = int(tensor.shape[0])
    m_len = int(param.shape[0])
    if m_len <= ck_len or m_len % ck_len != 0:
        return None
    factor = m_len // ck_len
    if factor <= 1:
        return None
    buf = param.clone()
    src = tensor.to(device=buf.device, dtype=buf.dtype)
    for t in range(ck_len):
        buf[t * factor].copy_(src[t])
        for c in range(1, factor):
            buf[t * factor + c].copy_(param[t * factor + c] * 0.01)
    return buf


def _foundation_is_reconstruction_head_weight(name: str, model: torch.nn.Module) -> bool:
    """True only for the last Linear's weight in ``reconstruction_head`` (avoid middle layers)."""

    rh = getattr(model, "reconstruction_head", None)
    if isinstance(rh, torch.nn.Sequential):
        for idx in range(len(rh) - 1, -1, -1):
            if isinstance(rh[idx], torch.nn.Linear):
                return name == f"reconstruction_head.{idx}.weight"
        return False
    return "reconstruction_head" in name and name.endswith(".weight")


def _foundation_is_reconstruction_head_bias(name: str, model: torch.nn.Module) -> bool:
    rh = getattr(model, "reconstruction_head", None)
    if isinstance(rh, torch.nn.Sequential):
        for idx in range(len(rh) - 1, -1, -1):
            if isinstance(rh[idx], torch.nn.Linear):
                return name == f"reconstruction_head.{idx}.bias"
        return False
    return "reconstruction_head" in name and name.endswith(".bias")


def _foundation_channel_inflate_pair(
    name: str,
    tensor: torch.Tensor,
    param: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[_InflateOutcome]:
    """Expand Monash (``C==1``) checkpoints into mixed-domain models for foundation adapters."""

    if name.endswith("value_proj.weight"):
        return _wrap_optional(_inflate_linear_in_features_ck(tensor, param))

    if _foundation_is_reconstruction_head_weight(name, model):
        maybe = _inflate_reconstruction_head_weight_ck(tensor, param)
        if maybe is not None:
            return _InflateOutcome(maybe, True)
    if _foundation_is_reconstruction_head_bias(name, model):
        maybe = _inflate_reconstruction_head_bias_ck(tensor, param)
        if maybe is not None:
            return _InflateOutcome(maybe, True)

    forecast_weight = False
    forecast_bias = False
    fh = getattr(model, "forecast_head", None)
    if isinstance(fh, torch.nn.Sequential):
        last_idx = None
        for idx in range(len(fh) - 1, -1, -1):
            if isinstance(fh[idx], torch.nn.Linear):
                last_idx = idx
                break
        if last_idx is not None:
            forecast_weight = name == f"forecast_head.{last_idx}.weight"
            forecast_bias = name == f"forecast_head.{last_idx}.bias"
    elif isinstance(fh, torch.nn.Linear):
        forecast_weight = name == "forecast_head.weight"
        forecast_bias = name == "forecast_head.bias"

    if forecast_weight:
        maybe = _foundation_forecast_head_row_expand_weight(tensor, param)
        if maybe is not None:
            logger.info(
                "inflated %s forecast rows: %d -> %d",
                name,
                int(tensor.shape[0]),
                int(param.shape[0]),
            )
            return _InflateOutcome(maybe, True)
    if forecast_bias:
        maybe = _foundation_forecast_head_row_expand_bias(tensor, param)
        if maybe is not None:
            logger.info(
                "inflated %s forecast bias len: %d -> %d",
                name,
                int(tensor.shape[0]),
                int(param.shape[0]),
            )
            return _InflateOutcome(maybe, True)

    # FactoST / single-module heads: never silent-mismatch patch/forecast linear layers.
    if name in {"patch_decoder.weight", "patch_decoder.bias"} and tensor.shape != param.shape:
        logger.warning(
            "foundation_channel_inflate: %s checkpoint shape %s vs model %s; keeping model initialization",
            name,
            tuple(tensor.shape),
            tuple(param.shape),
        )
        return _InflateOutcome(param.clone(), False)
    if name in {"forecast_head.weight", "forecast_head.bias"} and tensor.shape != param.shape:
        maybe_w = (
            _foundation_forecast_head_row_expand_weight(tensor, param)
            if name.endswith("weight")
            else None
        )
        maybe_b = (
            _foundation_forecast_head_row_expand_bias(tensor, param)
            if name.endswith("bias")
            else None
        )
        maybe = maybe_w or maybe_b
        if maybe is not None:
            logger.info(
                "inflated FactoST %s: %s -> %s",
                name,
                tuple(tensor.shape),
                tuple(param.shape),
            )
            return _InflateOutcome(maybe, True)
        logger.warning(
            "foundation_channel_inflate: %s checkpoint shape %s vs model %s; keeping model initialization",
            name,
            tuple(tensor.shape),
            tuple(param.shape),
        )
        return _InflateOutcome(param.clone(), False)

    return None


def _stable_trunk_channel_inflate_pair(
    name: str,
    tensor: torch.Tensor,
    param: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[_InflateOutcome]:
    # Monash stable pretrain uses input_dim==1; mixed-domain may differ num_datasets / output_dim.
    if name.endswith(".dataset_modulation.embedding.weight"):
        maybe_emb = _align_dataset_modulation_embedding_ck(tensor, param)
        if maybe_emb is not None:
            return _InflateOutcome(maybe_emb, True)
    if name.endswith("diffusion_mechanism_learner.output_proj.weight"):
        return _wrap_optional(_inflate_reconstruction_head_weight_ck(tensor, param))
    if name.endswith("diffusion_mechanism_learner.output_proj.bias"):
        return _wrap_optional(_inflate_reconstruction_head_bias_ck(tensor, param))
    if name.endswith("residual_event_encoder.value_proj.weight"):
        return _wrap_optional(_inflate_linear_in_features_ck(tensor, param))
    if name.endswith("calibration_head.weight"):
        return _wrap_optional(_inflate_square_matrix_top_left_ck(tensor, param))
    if name.endswith("calibration_head.bias"):
        return _wrap_optional(_inflate_calibration_bias_ck(tensor, param))
    if name.startswith("fusion_predictor."):
        if name.endswith(".additive_logit"):
            maybe_logit = _inflate_fusion_additive_logit_ck(tensor, param)
            if maybe_logit is not None:
                return _InflateOutcome(maybe_logit, True)
        if tensor.shape != param.shape:
            logger.warning(
                "stable_trunk_channel_inflate: %s checkpoint %s vs model %s; keeping model initialization",
                name,
                tuple(tensor.shape),
                tuple(param.shape),
            )
            return _InflateOutcome(param.clone(), False)
        return None
    if not _stable_trunk_prefix(name):
        return None
    if name.endswith(".local_branch.0.weight"):
        return _wrap_optional(_inflate_local_branch_ck(tensor, param))
    if name.endswith(".coarse_branch.0.weight"):
        return _wrap_optional(_inflate_linear_in_features_ck(tensor, param))
    if name.endswith(".frequency_branch.0.weight"):
        return _wrap_optional(_inflate_linear_in_features_ck(tensor, param))
    if name.endswith(".reconstruction_head.weight"):
        return _wrap_optional(_inflate_reconstruction_head_weight_ck(tensor, param))
    if name.endswith(".reconstruction_head.bias"):
        return _wrap_optional(_inflate_reconstruction_head_bias_ck(tensor, param))
    idxw = _forecast_head_last_linear_idx(model)
    if idxw is not None and name.endswith(f".forecast_head.{idxw}.weight"):
        return _try_forecast_head_last_inflate(name, tensor, param, model)
    if idxw is not None and name.endswith(f".forecast_head.{idxw}.bias"):
        return _try_forecast_head_last_bias_inflate(name, tensor, param, model)
    return None


def _migrate_zed_route_linear_weight(
    ck_tensor: torch.Tensor,
    model_tensor: torch.Tensor,
    *,
    hidden_dim: int,
) -> Optional[torch.Tensor]:
    """Align ZED router / route-gate first Linear when only input channels C differ.

    Feature layout matches ``build_route_features`` (no ``dataset_id_embed``):
    ``[temporal (2*C), graph_signature (8), residual_event_signature (H)]``.
    Graph and event columns are independent of C, so they copy verbatim. Overlapping
    per-channel temporal slots copy from the checkpoint; new channels keep ``model_tensor`` init.
    """

    if ck_tensor.shape == model_tensor.shape:
        return ck_tensor.to(device=model_tensor.device, dtype=model_tensor.dtype)
    if ck_tensor.ndim != 2 or model_tensor.ndim != 2:
        return None
    out_rows, in_old = ck_tensor.shape
    _, in_new = model_tensor.shape
    if out_rows != model_tensor.shape[0]:
        return None
    h = int(hidden_dim)
    if (in_old - 8 - h) < 0 or (in_new - 8 - h) < 0:
        return None
    if (in_old - 8 - h) % 2 != 0 or (in_new - 8 - h) % 2 != 0:
        return None
    c_old = (in_old - 8 - h) // 2
    c_new = (in_new - 8 - h) // 2
    if 2 * c_old + 8 + h != in_old or 2 * c_new + 8 + h != in_new:
        return None

    buf = model_tensor.clone()
    o_g, n_g = 2 * c_old, 2 * c_new
    buf[:, n_g : n_g + 8] = ck_tensor[:, o_g : o_g + 8].to(device=buf.device, dtype=buf.dtype)
    buf[:, n_g + 8 :] = ck_tensor[:, o_g + 8 :].to(device=buf.device, dtype=buf.dtype)
    for c in range(min(c_old, c_new)):
        buf[:, c] = ck_tensor[:, c].to(device=buf.device, dtype=buf.dtype)
        buf[:, c_new + c] = ck_tensor[:, c_old + c].to(device=buf.device, dtype=buf.dtype)
    return buf


def _normalize_checkpoint_patterns(patterns: Sequence[str]) -> List[str]:
    """Turn bare prefixes like ``zed_router`` into ``zed_router.*`` for fnmatch."""

    out: List[str] = []
    for raw in patterns:
        p = str(raw).strip()
        if not p:
            continue
        if any(ch in p for ch in "*?[]"):
            out.append(p)
        elif p.endswith(".*"):
            out.append(p)
        else:
            out.append(f"{p}.*")
    return out


def _checkpoint_key_matches_any(name: str, patterns: Sequence[str]) -> bool:
    for p in patterns:
        if fnmatch.fnmatch(name, p):
            return True
    return False


def filter_checkpoint_state_dict(
    checkpoint_state: Dict[str, Any],
    *,
    load_prefixes: Optional[Sequence[str]] = None,
    skip_prefixes: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Drop checkpoint keys per stage policy. Returns (filtered_state, list of keys not loaded from ckpt)."""

    skipped: List[str] = []
    work = {k: v for k, v in checkpoint_state.items() if isinstance(v, torch.Tensor)}
    if skip_prefixes:
        sp = _normalize_checkpoint_patterns(skip_prefixes)
        new: Dict[str, Any] = {}
        for k, v in work.items():
            if _checkpoint_key_matches_any(k, sp):
                skipped.append(k)
            else:
                new[k] = v
        work = new
    if load_prefixes:
        lp = _normalize_checkpoint_patterns(load_prefixes)
        new = {}
        for k, v in work.items():
            if _checkpoint_key_matches_any(k, lp):
                new[k] = v
            else:
                skipped.append(k)
        work = new
    return work, skipped


def _is_zed_route_feature_first_linear_weight(name: str) -> bool:
    if name in {
        "zed_router.net.0.weight",
        "zed_route_gate.fc.weight",
        "zed_router_feat_adapter.net.0.weight",
    }:
        return True
    for pat in ("target_router_adapter.net.0.weight", "router_offset.net.0.weight"):
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _migrate_zed_route_input_weight(
    name: str,
    ck_tensor: torch.Tensor,
    model_tensor: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """Migrate ZED route-input Linear weights ``[out_dim, route_dim]`` when ``route_dim`` changes."""

    if ck_tensor.ndim != 2 or model_tensor.ndim != 2:
        return None
    if ck_tensor.shape[0] != model_tensor.shape[0]:
        return None
    if ck_tensor.shape[1] == model_tensor.shape[1]:
        return ck_tensor.to(device=model_tensor.device, dtype=model_tensor.dtype)
    h_dim = getattr(model, "hidden_dim", None)
    if h_dim is not None:
        migrated = _migrate_zed_route_linear_weight(
            ck_tensor,
            model_tensor,
            hidden_dim=int(h_dim),
        )
        if migrated is not None:
            logger.info(
                "Migrated ZED routing weight %r (checkpoint in_features=%d -> model in_features=%d): "
                "copied graph + event blocks and overlapping temporal channels.",
                name,
                ck_tensor.shape[1],
                model_tensor.shape[1],
            )
            return migrated
    buf = model_tensor.clone()
    n = min(int(ck_tensor.shape[1]), int(model_tensor.shape[1]))
    buf[:, :n].copy_(ck_tensor[:, :n].to(device=buf.device, dtype=buf.dtype))
    logger.info(
        "Migrated ZED routing weight %r (fallback column overlap: in_features %d -> %d, copied %d).",
        name,
        ck_tensor.shape[1],
        model_tensor.shape[1],
        n,
    )
    return buf


def _is_zed_route_feature_row_linear_weight(name: str) -> bool:
    if name == "zed_router_feat_adapter.net.2.weight":
        return True
    if fnmatch.fnmatch(name, "zed_fa_prompt_proj.weight"):
        return True
    return False


def _migrate_zed_route_row_weight(
    name: str,
    ck_tensor: torch.Tensor,
    model_tensor: torch.Tensor,
    model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """Migrate weights shaped ``[route_feat_dim, other]`` when the leading dim uses route layout."""

    if ck_tensor.ndim != 2 or model_tensor.ndim != 2:
        return None
    if ck_tensor.shape[1] != model_tensor.shape[1]:
        return None
    if ck_tensor.shape[0] == model_tensor.shape[0]:
        return ck_tensor.to(device=model_tensor.device, dtype=model_tensor.dtype)
    h_dim = getattr(model, "hidden_dim", None)
    if h_dim is not None:
        migrated_t = _migrate_zed_route_linear_weight(
            ck_tensor.T,
            model_tensor.T,
            hidden_dim=int(h_dim),
        )
        if migrated_t is not None:
            logger.info(
                "Migrated ZED route-row weight %r (checkpoint dim0=%d -> model dim0=%d): "
                "copied graph + event blocks and overlapping temporal channels.",
                name,
                ck_tensor.shape[0],
                model_tensor.shape[0],
            )
            return migrated_t.T
    buf = model_tensor.clone()
    n = min(int(ck_tensor.shape[0]), int(model_tensor.shape[0]))
    buf[:n].copy_(ck_tensor[:n].to(device=buf.device, dtype=buf.dtype))
    logger.info(
        "Migrated ZED route-row weight %r (fallback row overlap: %d -> %d, copied %d).",
        name,
        ck_tensor.shape[0],
        model_tensor.shape[0],
        n,
    )
    return buf


def _is_zed_fa_adapter_only_key(name: str) -> bool:
    """Few-shot factorized-adaptation modules that must not be forced from unrelated checkpoints."""

    if name.startswith("zed_fa_"):
        return True
    if name.startswith("zed_fewshot_adapter"):
        return True
    return False


def adapt_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint_state: Dict[str, torch.Tensor],
    *,
    stable_trunk_channel_inflate: bool = False,
    foundation_channel_inflate: bool = False,
    load_prefixes: Optional[Sequence[str]] = None,
    skip_prefixes: Optional[Sequence[str]] = None,
    adapt_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Align a checkpoint to the current model, expanding small node/channel embedding tables.

    Used when transfer stages use a larger graph or ``max_num_nodes`` than pretraining (e.g. 256 → 307).
    Extra rows keep the model's current initialization. Other shape mismatches raise.

    With ``stable_trunk_channel_inflate=True``, Monash-style single-channel checkpoints
    (``input_dim==1``) are expanded for ``stable_trunk`` and ``residual_event_encoder.value_proj``;
    ``dataset_modulation.embedding`` rows are truncated or padded when ``num_datasets`` differs;
    ``diffusion_mechanism_learner.output_proj`` is inflated when ``output_dim`` grows (e.g. 1 → C);
    ``fusion_predictor.additive_logit`` last dim matches ``output_dim``; other ``fusion_predictor``
    tensors with incompatible shapes keep model initialization (with a warning);
    ``calibration_head`` square weights use top-left block copy when ``output_dim`` changes.

    With ``foundation_channel_inflate=True``, OpenCity / UniST / FactoST-style checkpoints trained
    with ``input_dim==1`` gain channel-expanded ``value_proj``, reconstruction head, and forecast head
    rows when loading into mixed-domain ``input_dim==C``. FactoST ``patch_decoder`` / ``forecast_head``
    mismatches that cannot be expanded emit an explicit warning and keep the current initialization.
    """

    working_ckpt: Dict[str, Any] = {k: v for k, v in checkpoint_state.items()}
    policy_skipped: list[str] = []
    if load_prefixes or skip_prefixes:
        filtered, policy_skipped = filter_checkpoint_state_dict(
            working_ckpt,
            load_prefixes=load_prefixes,
            skip_prefixes=skip_prefixes,
        )
        working_ckpt = filtered

    rep = adapt_report
    if rep is not None:
        rep["skipped_by_checkpoint_policy"] = list(policy_skipped)
        rep.setdefault("migrated_route_feature_keys", [])
        rep.setdefault("kept_init_keys", [])

    model_state = model.state_dict()
    out: Dict[str, Any] = {}
    inflated_keys: list[str] = []
    kept_init_keys: list[str] = []
    unchanged_keys: list[str] = []
    migrated_route_keys: list[str] = []
    ck_miss = [k for k in model_state if k not in working_ckpt]
    ck_unexp = [k for k in working_ckpt if k not in model_state]

    def _note_kept_init(key: str) -> None:
        kept_init_keys.append(key)
        if rep is not None:
            lst = rep.setdefault("kept_init_keys", [])
            if key not in lst:
                lst.append(key)

    def _note_route_migrated(key: str) -> None:
        migrated_route_keys.append(key)
        inflated_keys.append(key)
        if rep is not None:
            lst = rep.setdefault("migrated_route_feature_keys", [])
            if key not in lst:
                lst.append(key)

    for name, param in model_state.items():
        if name not in working_ckpt:
            out[name] = param
            continue
        tensor = working_ckpt[name]
        if not isinstance(tensor, torch.Tensor):
            out[name] = tensor
            continue

        if stable_trunk_channel_inflate:
            outcome = _stable_trunk_channel_inflate_pair(name, tensor, param, model)
            if outcome is not None:
                if outcome.tensor.shape != param.shape:
                    raise RuntimeError(
                        f"Internal inflate error for {name!r}: got {tuple(outcome.tensor.shape)} "
                        f"expected model {tuple(param.shape)}"
                    )
                out[name] = outcome.tensor
                if outcome.used_checkpoint:
                    inflated_keys.append(name)
                    _log_stable_trunk_inflate_line(name, tensor, param, model)
                else:
                    _note_kept_init(name)
                continue

        if foundation_channel_inflate:
            outcome = _foundation_channel_inflate_pair(name, tensor, param, model)
            if outcome is not None:
                if outcome.tensor.shape != param.shape:
                    raise RuntimeError(
                        f"Internal inflate error for {name!r}: got {tuple(outcome.tensor.shape)} "
                        f"expected model {tuple(param.shape)}"
                    )
                out[name] = outcome.tensor
                if outcome.used_checkpoint:
                    inflated_keys.append(name)
                else:
                    _note_kept_init(name)
                continue

        if tensor.shape == param.shape:
            out[name] = tensor.to(device=param.device, dtype=param.dtype)
            unchanged_keys.append(name)
            continue

        if name.endswith(".bias") and tensor.ndim == 1 and param.ndim == 1:
            logger.warning(
                "Checkpoint bias %r shape %s vs model %s; keeping model initialization.",
                name,
                tuple(tensor.shape),
                tuple(param.shape),
            )
            out[name] = param.clone()
            _note_kept_init(name)
            continue

        if _is_zed_fa_adapter_only_key(name):
            logger.warning(
                "Skipping FA-only key %r due to shape mismatch (checkpoint %s vs model %s); "
                "kept model initialization.",
                name,
                tuple(tensor.shape),
                tuple(param.shape),
            )
            out[name] = param.clone()
            _note_kept_init(name)
            continue

        if (
            not _is_zed_fa_adapter_only_key(name)
            and name.endswith(("norm.weight", "norm.bias"))
            and tensor.ndim == 1
            and param.ndim == 1
            and tensor.numel() == 1
            and param.numel() > 1
        ):
            buf = param.clone()
            val = tensor.to(dtype=buf.dtype, device=buf.device).reshape(())
            buf.fill_(val)
            out[name] = buf
            inflated_keys.append(name)
            logger.info(
                "Expanded LayerNorm-style param %r from scalar checkpoint (numel=%d -> %d).",
                name,
                int(tensor.numel()),
                int(param.numel()),
            )
            continue

        if _is_zed_route_feature_first_linear_weight(name) and name.endswith(".weight"):
            migrated = _migrate_zed_route_input_weight(name, tensor, param, model)
            if migrated is not None:
                out[name] = migrated
                _note_route_migrated(name)
                continue

        if _is_zed_route_feature_row_linear_weight(name) and name.endswith(".weight"):
            migrated_r = _migrate_zed_route_row_weight(name, tensor, param, model)
            if migrated_r is not None:
                out[name] = migrated_r
                _note_route_migrated(name)
                continue

        if name.startswith(("zed_router.", "zed_route_gate.", "zed_router_feat_adapter.")):
            raise RuntimeError(
                f"Cannot load {name!r}: checkpoint shape {tuple(tensor.shape)} vs model "
                f"{tuple(param.shape)} (ZED routing: try matching hidden_dim / router_inputs, "
                f"or remove dataset_id_embed from router for migration)"
            )

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

    if (
        stable_trunk_channel_inflate
        or foundation_channel_inflate
        or inflated_keys
        or kept_init_keys
        or policy_skipped
    ):
        logger.info("loaded unchanged keys count: %d", len(unchanged_keys))
        logger.info("inflated / migrated keys count: %d", len(inflated_keys))
        if migrated_route_keys:
            logger.info("ZED route-feature migrated keys: %s", migrated_route_keys)
        logger.info("kept model init (checkpoint skipped) count: %d", len(kept_init_keys))
        logger.info("missing keys (not in checkpoint): %d", len(ck_miss))
        logger.info("unexpected keys (not in model): %d", len(ck_unexp))
        if policy_skipped:
            logger.info("checkpoint policy skipped %d keys from file", len(policy_skipped))

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
    stable_trunk_channel_inflate: bool = False,
    foundation_channel_inflate: bool = False,
    load_prefixes: Optional[Sequence[str]] = None,
    skip_prefixes: Optional[Sequence[str]] = None,
    adapt_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ckpt = torch_load(path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)
    rep: Dict[str, Any] = adapt_report if adapt_report is not None else {}
    try:
        state_dict = adapt_checkpoint_state_dict(
            model,
            state_dict,
            stable_trunk_channel_inflate=stable_trunk_channel_inflate,
            foundation_channel_inflate=foundation_channel_inflate,
            load_prefixes=load_prefixes,
            skip_prefixes=skip_prefixes,
            adapt_report=rep,
        )
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
        "adapt": dict(rep),
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
