#!/usr/bin/env python3
"""UCI Electricity (electricity.txt.gz) -> BasicSTFM data.npz + train-split Pearson top-k graph.

Raw layout (same as LSTNet / Informer / Autoformer releases): no header, comma-separated,
each row is one hour, each column is one customer's consumption (321 series).

Default input: ``data/raw_data/Electricity/electricity.txt.gz``
Default output: ``data/Electricity/`` (matches ``dataset_registry`` paths).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DATA = Path(__file__).resolve().parent
if str(_SCRIPTS_DATA) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA))

import prepare_ettm_for_basicstfm as ett  # noqa: E402


def load_electricity_txt(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return values [T, N] and synthetic node names."""
    df = pd.read_csv(path, header=None, compression="infer")
    if df.shape[1] < 1:
        raise ValueError(f"{path}: empty or invalid")
    values = df.to_numpy(dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path}: expected 2D, got {values.shape}")
    if not np.isfinite(values).any():
        raise ValueError(f"{path}: no finite values")
    n = values.shape[1]
    names = [f"client_{i}" for i in range(n)]
    return values, names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="electricity.txt.gz -> BasicSTFM data.npz + adj.npz")
    p.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="electricity.txt or .gz (comma, no header). Default: <repo>/data/raw_data/Electricity/electricity.txt.gz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/data/Electricity",
    )
    p.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.1, 0.2),
    )
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--topk", type=int, default=10, help="Top-k correlates per client (many nodes ⇒ use slightly larger k).")
    p.add_argument("--use-abs-corr", type=ett.str2bool, default=True)
    p.add_argument("--threshold", type=str, default="none")
    p.add_argument("--binary-graph", type=ett.str2bool, default=False)
    p.add_argument("--max-timesteps", type=int, default=None)
    p.add_argument("--time-start-row", type=int, default=0)
    return p


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    args = build_parser().parse_args()
    root = _repo_root()
    inp = (
        Path(args.input_path).resolve()
        if args.input_path is not None
        else root / "data" / "raw_data" / "Electricity" / "electricity.txt.gz"
    )
    out_dir = Path(args.output_dir).resolve() if args.output_dir is not None else root / "data" / "Electricity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Electricity raw not found: {inp}")

    values, node_names = load_electricity_txt(inp)

    ns = argparse.Namespace(
        split=tuple(float(x) for x in args.split),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    split = ett._resolve_split(ns)
    del ns

    t0 = int(args.time_start_row)
    if t0 < 0 or t0 >= len(values):
        raise ValueError("time_start_row out of range")
    values = values[t0:]
    if args.max_timesteps is not None:
        mt = int(args.max_timesteps)
        if mt < 2:
            raise ValueError("max_timesteps must be >= 2")
        values = values[:mt]

    data = values.astype(np.float32)[..., None]
    t, n, c = data.shape
    train_len, val_len, test_len = ett.split_lengths(t, split)

    train_x = data[:train_len, :, 0].astype(np.float64)
    corr = ett.pearson_corr_matrix(train_x)
    thr = ett.parse_threshold(args.threshold)
    dmask = ett.topk_directed_mask(corr, args.topk, threshold=thr)
    adj_w = ett.build_adjacency_pair(corr, dmask, weighted=True, use_abs_corr=args.use_abs_corr)
    adj_b = ett.build_adjacency_pair(corr, dmask, weighted=False, use_abs_corr=args.use_abs_corr)

    if args.binary_graph:
        adj_default = adj_b
        default_label = "binary (adj_binary_topk.npz)"
    else:
        adj_default = adj_w
        default_label = "weighted correlation (adj_corr_topk.npz)"

    np.savez_compressed(out_dir / "data.npz", data=data)
    np.savez_compressed(out_dir / "adj_corr_topk.npz", adj=adj_w)
    np.savez_compressed(out_dir / "adj_binary_topk.npz", adj=adj_b)
    np.savez_compressed(out_dir / "adj.npz", adj=adj_default)

    meta = {
        "time_column": None,
        "node_names": node_names,
        "num_nodes": n,
        "original_file": str(inp),
        "trim": {"time_start_row": t0, "max_timesteps": args.max_timesteps},
        "data_shape": [t, n, c],
        "split_ratios": list(split),
        "split_lengths": {"train": train_len, "val": val_len, "test": test_len},
        "graph": {
            "built_from": "train_split_only",
            "train_timesteps_used": train_len,
            "topk": args.topk,
            "use_abs_corr": args.use_abs_corr,
            "threshold": thr,
            "symmetrization": "A = max(A, A.T)",
            "diagonal": 1.0,
            "default_adj_file": default_label,
        },
    }
    (out_dir / "node_names.json").write_text(
        json.dumps(
            {
                "time_column": None,
                "feature_columns": node_names,
                "node_order_note": "node i ↔ client time series column i (no calendar column in raw UCI file)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    st_w = ett.adj_offdiag_stats(adj_w)
    st_b = ett.adj_offdiag_stats(adj_b)

    readme = f"""# Electricity (UCI) — BasicSTFM

Prepared by `scripts/data/prepare_electricity_for_basicstfm.py`.

## Source

- File: `{inp}`
- Layout: hourly rows × `{n}` client columns (no header, comma-separated).

## Arrays

- `data.npz` key `data`: shape `[T, N, C]` = `{tuple(data.shape)}`.

## Split (WindowDataModule)

train / val / test lengths: **{train_len} / {val_len} / {test_len}**.

## Graph

Train-only Pearson top-{args.topk}, same convention as `prepare_ettm_for_basicstfm.py`.

Weighted offdiag density: {st_w["density_offdiag"]:.6f}; binary: {st_b["density_offdiag"]:.6f}.

## Command

```bash
python scripts/data/prepare_electricity_for_basicstfm.py \\
  --input-path {inp} \\
  --output-dir {out_dir} \\
  --split {split[0]} {split[1]} {split[2]} \\
  --topk {args.topk}
```
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("=== Electricity -> BasicSTFM 完成 ===")
    print(f"输出: {out_dir}")
    print(f"data shape: {tuple(data.shape)}")
    print(f"train 段长度（构图用）: {train_len}")
    print(f"adj weighted / binary offdiag density: {st_w['density_offdiag']:.6f} / {st_b['density_offdiag']:.6f}")


if __name__ == "__main__":
    main()
