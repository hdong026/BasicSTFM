#!/usr/bin/env python3
"""Emit BasicTS-style single-channel [T,N,1] traffic ``data.npz`` plus adj symlink.

Run from repo root::

    python scripts/data/build_basicts_traffic_targets.py

Reads ``data/<DATASET>/data.npz`` (expects key ``data``, shape ``[T,N,C]``).
If ``C>1``, keeps channel 0 only (traffic flow/speed primary channel).

Writes ``data/<DATASET>_BasicTS/data.npz`` and symlinks ``adj.npz`` to the
original dataset adjacency when missing.

See each ``*_BasicTS/README.md`` for shapes and protocol notes (no RevIN;
StandardScaler fit on train split during training — **not** pre-normalized here).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

TRAFFIC_DATASETS: Tuple[str, ...] = (
    "METR-LA",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "PEMS-BAY",
)


def _load_tnc(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "data" not in loaded.files:
            loaded.close()
            raise KeyError(f"{path}: npz missing 'data' key")
        arr = loaded["data"].copy()
        loaded.close()
    else:
        arr = np.asarray(loaded)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected [T,N,C], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _readme_block(dataset: str, src_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> str:
    return f"""# {dataset}_BasicTS

## Source dataset

- Raw NPZ: `../{dataset}/data.npz`
- Graph: `../{dataset}/adj.npz` (symlinked here as `adj.npz`)

## Shapes

| | Shape |
|--|--|
| Original `data` | `{src_shape}` |
| BasicTS-style `data` (this folder) | `{out_shape}` |

## Protocol

- **Selected channel**: index **0** only when `C>1` (matches BasicTS `target_channel = [0]`).
- **RevIN**: **disabled** — use `task.use_revin: false` with StandardScaler in the trainer.
- **Scaler**: StandardScaler **fitted on the training split inside the datamodule**, not applied in this script (arrays stay in **original scale** on disk).

This mirrors GestaltCogTeam/BasicTS forecasting conventions (inverse scaler for metrics).
"""


def ensure_basicts_dataset(repo_root: Path, dataset: str, *, overwrite: bool) -> None:
    src_dir = repo_root / "data" / dataset
    src_npz = src_dir / "data.npz"
    src_adj = src_dir / "adj.npz"
    if not src_npz.is_file():
        print(f"[skip] missing source {src_npz}")
        return
    out_dir = repo_root / "data" / f"{dataset}_BasicTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "data.npz"

    raw = _load_tnc(src_npz)
    t, n, c = raw.shape
    if c > 1:
        sliced = raw[:, :, 0:1].copy()
    else:
        sliced = raw.copy()
    assert sliced.shape == (t, n, 1)

    if out_npz.is_file() and not overwrite:
        print(f"[exists] {out_npz} (use --overwrite to replace)")
    else:
        np.savez_compressed(out_npz, data=sliced)
        print(f"[write] {out_npz} shape={sliced.shape}")

    readme = out_dir / "README.md"
    if overwrite or not readme.is_file():
        readme.write_text(
            _readme_block(dataset, (t, n, c), tuple(int(x) for x in sliced.shape)),
            encoding="utf-8",
        )
        print(f"[write] {readme}")

    out_adj = out_dir / "adj.npz"
    if out_adj.is_file() or out_adj.is_symlink():
        if not overwrite:
            print(f"[exists] {out_adj}")
            return
        out_adj.unlink()

    if src_adj.is_file():
        rel = os.path.relpath(src_adj, out_dir)
        os.symlink(rel, out_adj)
        print(f"[symlink] {out_adj} -> {rel}")
    else:
        print(f"[warn] no adjacency at {src_adj}; skipping symlink")


def parse_names(names: Iterable[str]) -> Tuple[str, ...]:
    out = tuple(str(x).strip() for x in names if str(x).strip())
    return out if out else TRAFFIC_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BasicTS-style single-channel traffic NPZs.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(TRAFFIC_DATASETS),
        help=f"Subset of datasets (default: all {len(TRAFFIC_DATASETS)} traffic sets)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs / symlinks.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root (parent of data/)",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    for ds in parse_names(args.datasets):
        ensure_basicts_dataset(root, ds, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
