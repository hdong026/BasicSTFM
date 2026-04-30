"""Resolve train/val/test lengths from FactoST-style ``desc.json`` next to ``data.npz``."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

SplitTriple = Tuple[int, int, int]


def _as_int_lengths(total: int, spec: Sequence[Union[int, float]]) -> Optional[SplitTriple]:
    if len(spec) != 3:
        return None
    a, b, c = spec
    if all(isinstance(x, float) for x in spec) and max(spec) <= 1.0 + 1e-9:
        tr = int(total * float(a))
        va = int(total * float(b))
        te = total - tr - va
        if te < 0:
            return None
        return tr, va, te
    if all(isinstance(x, (int, float)) for x in spec):
        ia, ib, ic = int(a), int(b), int(c)
        if ia + ib + ic == total:
            return ia, ib, ic
        if ia + ib + ic < total and ia > 0 and ib >= 0 and ic >= 0:
            # Explicit lengths that do not fill series — extend test to EOF (compat shim).
            te = total - ia - ib
            return ia, ib, te
    return None


def _range_len(total: int, span: Any) -> Optional[Tuple[int, int]]:
    """Return (start, end) with end exclusive, clamped to [0, total]."""
    if isinstance(span, (list, tuple)) and len(span) == 2:
        s, e = int(span[0]), int(span[1])
        if s < 0 or e < s:
            return None
        s = max(0, min(s, total))
        e = max(s, min(e, total))
        return s, e
    return None


def _parse_datesplit_dict(total: int, ds: Mapping[str, Any]) -> Optional[SplitTriple]:
    # Canonical FactoST-like keys (half-open index ranges on the time axis).
    train_k = ds.get("train") or ds.get("train_range") or ds.get("train_idx")
    val_k = ds.get("val") or ds.get("valid") or ds.get("validation") or ds.get("val_range")
    test_k = ds.get("test") or ds.get("test_range")

    if train_k is None and {"train_begin", "train_end"}.issubset(ds.keys()):
        train_k = [int(ds["train_begin"]), int(ds["train_end"])]
    if val_k is None and {"val_begin", "val_end"}.issubset(ds.keys()):
        val_k = [int(ds["val_begin"]), int(ds["val_end"])]
    if test_k is None and {"test_begin", "test_end"}.issubset(ds.keys()):
        test_k = [int(ds["test_begin"]), int(ds["test_end"])]

    tr = _range_len(total, train_k)
    va = _range_len(total, val_k)
    te = _range_len(total, test_k)
    if tr and va and te:
        # Require contiguous coverage [0,total) without overlap (best-effort).
        t0, t1 = tr
        v0, v1 = va
        s0, s1 = te
        if t0 == 0 and t1 == v0 and v1 == s0 and s1 == total:
            return t1 - t0, v1 - v0, s1 - s0
        # Otherwise interpret as independent lengths from ranges.
        return t1 - t0, v1 - v0, s1 - s0

    # Single cumulative cut indices: train_end, val_end (optional explicit test_end).
    if "train_end" in ds:
        te_idx = int(ds["train_end"])
        ve_idx = int(ds.get("val_end", total))
        tst_idx = int(ds.get("test_end", total))
        if not (0 < te_idx <= ve_idx <= tst_idx <= total):
            return None
        return te_idx, ve_idx - te_idx, tst_idx - ve_idx

    return None


def parse_dataset_datesplit(total: int, dataset_datesplit: Any) -> Optional[SplitTriple]:
    """Parse ``dataset_datesplit`` field from ``desc.json`` into segment lengths."""
    if dataset_datesplit is None:
        return None
    if isinstance(dataset_datesplit, (list, tuple)):
        return _as_int_lengths(total, dataset_datesplit)
    if isinstance(dataset_datesplit, dict):
        return _parse_datesplit_dict(total, dataset_datesplit)
    return None


def load_factost_desc(data_path: str) -> Tuple[Optional[Dict[str, Any]], Path]:
    path = Path(data_path).expanduser().resolve()
    desc_path = path.parent / "desc.json"
    if not desc_path.is_file():
        return None, desc_path
    try:
        payload = json.loads(desc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, desc_path
    return payload if isinstance(payload, dict) else None, desc_path


def resolve_factost_split_lengths(
    data_path: str,
    total_timesteps: int,
    fallback_split: Sequence[float],
) -> Tuple[Optional[SplitTriple], Dict[str, Any]]:
    """Return ``(train_len, val_len, test_len)`` or ``None`` to keep YAML ``split`` ratios.

    Extra dict documents what was read (for audits / logging).
    """
    meta: Dict[str, Any] = {"data_path": str(Path(data_path).resolve()), "used_desc": False}
    desc, desc_path = load_factost_desc(data_path)
    meta["desc_path"] = str(desc_path)
    if desc is None:
        warnings.warn(
            f"factost_split enabled but no valid desc.json at {desc_path}; "
            f"falling back to ratio split {tuple(fallback_split)}",
            RuntimeWarning,
            stacklevel=3,
        )
        return None, meta

    meta["used_desc"] = True
    meta["split_type"] = desc.get("split_type")
    triple = parse_dataset_datesplit(total_timesteps, desc.get("dataset_datesplit"))
    if triple is None:
        warnings.warn(
            f"desc.json at {desc_path} missing usable dataset_datesplit for T={total_timesteps}; "
            f"falling back to ratio split {tuple(fallback_split)}",
            RuntimeWarning,
            stacklevel=3,
        )
        meta["parse_error"] = "dataset_datesplit_unparsed"
        return None, meta

    tr, va, te = triple
    if tr <= 0 or va < 0 or te <= 0 or tr + va + te > total_timesteps:
        warnings.warn(
            f"desc.json split lengths {(tr, va, te)} invalid for T={total_timesteps}; "
            f"falling back to ratio split {tuple(fallback_split)}",
            RuntimeWarning,
            stacklevel=3,
        )
        meta["parse_error"] = "invalid_lengths"
        return None, meta

    meta["dataset_datesplit_resolved"] = {"train_len": tr, "val_len": va, "test_len": te}
    return (tr, va, te), meta


def split_triple_to_split_field(triple: SplitTriple) -> List[int]:
    """Encode explicit lengths for :func:`basicstfm.data.datamodule._split_lengths`."""
    return [int(triple[0]), int(triple[1]), int(triple[2])]
