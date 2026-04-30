"""Train/val/test ratios aligned with GestaltCogTeam/BasicTS ``generate_training_data.py``.

Official ratios (see scripts under ``scripts/data_preparation/<DATASET>/``):

- ``METR-LA``, ``PEMS-BAY``: ``train_val_test_ratio`` = [0.7, 0.1, 0.2]
- ``PEMS03``, ``PEMS04``, ``PEMS07``, ``PEMS08``: [0.6, 0.2, 0.2]

Split lengths use ``int(T * train_ratio)``, ``int(T * val_ratio)``, remainder test —
matching BasicTS ``split_and_save_data``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

SplitTriple = Tuple[int, int, int]

BASICTS_TRAFFIC_TRAIN_VAL_TEST_RATIO: Dict[str, Tuple[float, float, float]] = {
    "METR-LA": (0.7, 0.1, 0.2),
    "PEMS-BAY": (0.7, 0.1, 0.2),
    "PEMS03": (0.6, 0.2, 0.2),
    "PEMS04": (0.6, 0.2, 0.2),
    "PEMS07": (0.6, 0.2, 0.2),
    "PEMS08": (0.6, 0.2, 0.2),
}


def canonical_basicts_traffic_key(dataset_key_or_folder: str) -> str:
    """Strip optional ``_BasicTS`` suffix from folder names like ``METR-LA_BasicTS``."""

    s = str(dataset_key_or_folder).strip()
    suf = "_BasicTS"
    if s.endswith(suf):
        return s[: -len(suf)]
    return s


def basicts_lengths_from_ratio(total_timesteps: int, ratio: Sequence[float]) -> SplitTriple:
    """Mirror BasicTS ``split_and_save_data`` indexing."""

    if len(ratio) != 3:
        raise ValueError("ratio must be length-3 train/val/test")
    train_ratio, val_ratio, _test_ratio = float(ratio[0]), float(ratio[1]), float(ratio[2])
    train_len = int(total_timesteps * train_ratio)
    val_len = int(total_timesteps * val_ratio)
    test_len = total_timesteps - train_len - val_len
    if train_len <= 0 or val_len < 0 or test_len <= 0:
        raise ValueError(
            f"Invalid BasicTS-style split for T={total_timesteps}: "
            f"train={train_len}, val={val_len}, test={test_len}"
        )
    return train_len, val_len, test_len


def resolve_basicts_split_lengths(
    *,
    total_timesteps: int,
    dataset_key: Optional[str],
    fallback_split: Sequence[float],
) -> Tuple[SplitTriple, Dict[str, Any]]:
    """Resolve split triple and metadata for ``split_mode: basicts``."""

    meta: Dict[str, Any] = {
        "split_mode": "basicts",
        "dataset_key_raw": dataset_key,
        "source": "GestaltCogTeam/BasicTS scripts/data_preparation/*/generate_training_data.py",
    }
    key = canonical_basicts_traffic_key(dataset_key or "")
    ratio = BASICTS_TRAFFIC_TRAIN_VAL_TEST_RATIO.get(key)
    if ratio is None:
        meta["resolution"] = "fallback_ratio"
        meta["fallback_split"] = list(float(x) for x in fallback_split)
        meta["warning"] = (
            f"Unknown traffic dataset_key={dataset_key!r} after canonicalization={key!r}; "
            f"using YAML fallback_split {meta['fallback_split']}. "
            "Document this in README if intentional."
        )
        triple = basicts_lengths_from_ratio(total_timesteps, fallback_split)
        meta["train_val_test_ratio"] = list(meta["fallback_split"])
    else:
        meta["resolution"] = "basic_ts_official"
        meta["canonical_dataset_key"] = key
        meta["train_val_test_ratio"] = list(ratio)
        triple = basicts_lengths_from_ratio(total_timesteps, ratio)
    meta["dataset_datesplit_resolved"] = {
        "train_len": triple[0],
        "val_len": triple[1],
        "test_len": triple[2],
    }
    return triple, meta


def infer_dataset_key_from_data_path(data_path: str) -> str:
    """Use registry ``name`` when available; otherwise parent folder name."""

    return Path(data_path).expanduser().resolve().parent.name
