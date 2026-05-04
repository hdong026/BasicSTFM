"""Sanitize dataset / domain ids for ModuleDict keys and routing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional


def sanitize_domain_key(name: str) -> str:
    """Map registry names (e.g. ``METR-LA``) to safe Python identifiers."""

    s = str(name).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


DEFAULT_SHARED_KEY = "__shared__"


def build_routing_from_batch(
    batch: Mapping[str, Any],
    *,
    routing_key: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Return ``{"keys": List[str] len B}`` for DSD models."""

    rk = str(routing_key).lower().strip()
    if rk == "auto":
        if batch.get("dataset_name") is not None:
            rk = "dataset_name"
        elif batch.get("domain_id") is not None:
            rk = "domain_id"
        elif batch.get("graph_id") is not None:
            rk = "graph_id"
        else:
            return {
                "keys": [DEFAULT_SHARED_KEY] * int(batch_size),
                "batch_size": int(batch_size),
            }

    if rk == "dataset_name":
        raw = batch.get("dataset_name")
        if isinstance(raw, str):
            keys = [sanitize_domain_key(raw)] * int(batch_size)
        elif isinstance(raw, (list, tuple)):
            if len(raw) != int(batch_size):
                raise ValueError(
                    f"batch['dataset_name'] list length {len(raw)} != batch_size {batch_size}"
                )
            keys = [sanitize_domain_key(x) for x in raw]
        else:
            raise ValueError(
                "routing_key=dataset_name requires batch['dataset_name'] str or list; "
                "ensure WindowDataModule / MultiDataset collate sets it."
            )
    elif rk == "domain_id":
        ids = batch.get("domain_id")
        if ids is None:
            raise ValueError("routing_key=domain_id requires batch['domain_id']")
        if not hasattr(ids, "shape"):
            ids = [int(ids)] * int(batch_size)
            keys = [f"domain_{int(i)}" for i in ids]
        else:
            vec = ids.view(-1).tolist()
            if len(vec) != int(batch_size):
                raise ValueError("domain_id length mismatch")
            keys = [f"domain_{int(i)}" for i in vec]
    elif rk == "graph_id":
        gid = batch.get("graph_id")
        if gid is None:
            keys = ["graph_default"] * int(batch_size)
        else:
            vec = gid.view(-1).tolist()
            if len(vec) != int(batch_size):
                raise ValueError("graph_id length mismatch")
            keys = [f"graph_{int(i)}" for i in vec]
    else:
        raise ValueError(f"Unknown routing_key: {routing_key!r}")

    return {"keys": keys, "batch_size": int(batch_size)}


def group_indices_by_key(keys: List[str]) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    return buckets
