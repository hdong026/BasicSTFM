#!/usr/bin/env python3
"""Prepare ETTm1/ETTm2 (ETDataset CSV) for BasicSTFM: data.npz + correlation top-k graphs.

Graphs are built from the **training split timesteps only** (no val/test leakage),
using the same contiguous split convention as ``WindowDataModule`` /
``MultiDatasetWindowDataModule``: ``train=int(T*r0)``, ``val=int(T*r1)``,
``test`` = remainder (third float in ``split`` is informational only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def split_lengths(total: int, split: tuple[float, float, float]) -> tuple[int, int, int]:
    """Match ``basicstfm.data.datamodule._split_lengths`` for float ratios."""
    if len(split) != 3:
        raise ValueError("split must have 3 ratios [train, val, test]")
    if not all(isinstance(x, float) and x <= 1.0 for x in split):
        raise ValueError("split ratios must be floats in (0, 1]")
    train = int(total * split[0])
    val = int(total * split[1])
    test = total - train - val
    if test < 0:
        raise ValueError("split produces negative test length")
    return train, val, test


def load_ett_csv(path: Path) -> tuple[pd.DataFrame, str, list[str], np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: need at least a time column and one feature column")
    time_col = str(df.columns[0])
    feature_cols = [str(c) for c in df.columns[1:]]
    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path}: expected 2D feature block, got {values.shape}")
    return df, time_col, feature_cols, values


def pearson_corr_matrix(x: np.ndarray) -> np.ndarray:
    """x: [T, N] — variables are columns. Returns [N, N] with diag 1."""
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps to compute correlation")
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c.astype(np.float64)


def topk_directed_mask(
    corr: np.ndarray,
    k: int,
    *,
    threshold: Optional[float],
) -> np.ndarray:
    """Directed top-k by |corr|, excluding self. Returns binary mask [N, N]."""
    n = corr.shape[0]
    if k < 1:
        raise ValueError("topk must be >= 1")
    k_eff = min(k, n - 1)
    mask = np.zeros((n, n), dtype=np.float64)
    abs_c = np.abs(corr)
    for i in range(n):
        s = abs_c[i].astype(np.float64)
        s[i] = -np.inf
        if threshold is not None:
            s[abs_c[i] < threshold] = -np.inf
        if not np.any(np.isfinite(s)):
            continue
        pick = min(k_eff, int(np.isfinite(s).sum()))
        idx = np.argpartition(-s, pick - 1)[:pick]
        idx = idx[np.isfinite(s[idx])]
        mask[i, idx] = 1.0
    return mask


def build_adjacency_pair(
    corr: np.ndarray,
    directed_mask: np.ndarray,
    *,
    weighted: bool,
    use_abs_corr: bool,
) -> np.ndarray:
    """Symmetrize with max, set diag to 1."""
    n = corr.shape[0]
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if directed_mask[i, j] <= 0:
                continue
            if weighted:
                w[i, j] = abs(corr[i, j]) if use_abs_corr else corr[i, j]
            else:
                w[i, j] = 1.0
    sym = np.maximum(w, w.T)
    np.fill_diagonal(sym, 1.0)
    return sym.astype(np.float32)


def adj_offdiag_stats(adj: np.ndarray) -> dict[str, float]:
    n = adj.shape[0]
    off = adj.astype(np.float64).copy()
    np.fill_diagonal(off, 0.0)
    nnz = int(np.count_nonzero(off))
    denom = n * (n - 1)
    degree = off.sum(axis=1)
    return {
        "nnz_offdiag": float(nnz),
        "density_offdiag": float(nnz) / float(max(denom, 1)),
        "sparsity_offdiag": 1.0 - float(nnz) / float(max(denom, 1)),
        "mean_degree": float(degree.mean()),
    }


def str2bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got {s!r}")


def parse_threshold(s: str) -> Optional[float]:
    if s is None or str(s).lower() in ("none", "", "null"):
        return None
    return float(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETTm CSV -> BasicSTFM data.npz + correlation graphs")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Train/val/test ratios (third is informative); default 0.7 0.1 0.2 unless --train-ratio trio set.",
    )
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--topk", type=int, default=5, help="Top-k neighbors per node (by |corr|).")
    p.add_argument(
        "--use-abs-corr",
        type=str2bool,
        default=True,
        help="If true, edge weights use |corr| (weighted graph); else signed corr on kept edges.",
    )
    p.add_argument(
        "--threshold",
        type=str,
        default="none",
        help="Minimum |corr| for a candidate neighbor, or 'none'.",
    )
    p.add_argument(
        "--binary-graph",
        type=str2bool,
        default=False,
        help="If true, adj.npz copies the binary graph; default copies weighted.",
    )
    p.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Use only the first T rows (contiguous prefix) before split/graph.",
    )
    p.add_argument(
        "--time-start-row",
        type=int,
        default=0,
        help="0-based start row index before max_timesteps trim (advanced).",
    )
    return p


def _resolve_split(args: argparse.Namespace) -> tuple[float, float, float]:
    """Ratios align with WindowDataModule: only the first two scale train/val lengths."""
    trio = (args.train_ratio, args.val_ratio, args.test_ratio)
    if all(x is not None for x in trio):
        a, b, c = float(trio[0]), float(trio[1]), float(trio[2])
        if a < 0 or b < 0 or c < 0:
            raise ValueError("train/val/test ratios must be non-negative")
        if a + b > 1.0 + 1e-6:
            raise ValueError("train_ratio + val_ratio must be <= 1 (WindowDataModule convention)")
        if abs(a + b + c - 1.0) > 1e-2:
            raise ValueError(
                f"train+val+test should sum to ~1 for documentation (got {a+b+c}); "
                "note only train and val ratios determine split lengths."
            )
        split = (a, b, c)
    elif any(x is not None for x in trio):
        raise ValueError("Set all of --train-ratio, --val-ratio, --test-ratio, or use --split only")
    elif args.split is not None:
        split = tuple(float(x) for x in args.split)
    else:
        split = (0.7, 0.1, 0.2)
    return split


def main() -> None:
    args = build_parser().parse_args()
    inp = args.input_csv.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, time_col, node_names, values = load_ett_csv(inp)
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
    if c != 1:
        raise RuntimeError("internal: expected C=1")

    split = _resolve_split(args)
    train_len, val_len, test_len = split_lengths(t, split)

    train_x = data[:train_len, :, 0].astype(np.float64)
    corr = pearson_corr_matrix(train_x)
    thr = parse_threshold(args.threshold)
    dmask = topk_directed_mask(corr, args.topk, threshold=thr)
    adj_w = build_adjacency_pair(corr, dmask, weighted=True, use_abs_corr=args.use_abs_corr)
    adj_b = build_adjacency_pair(corr, dmask, weighted=False, use_abs_corr=args.use_abs_corr)

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
        "time_column": time_col,
        "node_names": node_names,
        "num_nodes": n,
        "original_csv": str(inp),
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
                "time_column": time_col,
                "feature_columns": node_names,
                "node_order_note": "node i corresponds to feature_columns[i]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    st_w = adj_offdiag_stats(adj_w)
    st_b = adj_offdiag_stats(adj_b)

    readme = f"""# ETTm — BasicSTFM

Prepared by `scripts/data/prepare_ettm_for_basicstfm.py`.

## Source

- CSV: `{inp}`
- 时间列: `{time_col}`（**不作为节点特征**）
- 节点列（顺序与 `data[..., i, 0]` 一致）: {", ".join(f"`{x}`" for x in node_names)}

## 数组形状

- 原始表: `{t}` 行 × `{n}` 个变量列（不含时间）
- `data.npz` key `data`: shape `[T, N, C]` = `{tuple(data.shape)}`（`C=1`）

## 训练 / 验证 / 测试划分

与 `WindowDataModule` 一致：`train = int(T×{split[0]})`，`val = int(T×{split[1]})`，`test` 为剩余。

- train / val / test 长度: **{train_len} / {val_len} / {test_len}**

## 图构造（**仅用 train 段，无验证/测试泄漏**）

在 train 段上计算变量两两 **Pearson** 相关矩阵 `corr`，按 **|corr|** 为每个节点取 **top-{args.topk}** 邻居（不含自环），再：

- 对称化：`A = max(A, A^T)`
- 对角线：`1`
- **use_abs_corr={args.use_abs_corr}**：加权版边权为 `{'|corr|' if args.use_abs_corr else 'corr（保留符号）'}`；二值版为 `0/1`
- **threshold**: `{thr}`（低于阈值的边不参与 top-k 候选）

输出文件：

- `adj_corr_topk.npz` — 加权相关图
- `adj_binary_topk.npz` — 二值 top-k 图
- `adj.npz` — 默认与 **{default_label}** 相同

## 统计（非对角）

| 版本 | 非零率 | 平均度 |
|------|--------|--------|
| weighted | {st_w["density_offdiag"]:.6f} | {st_w["mean_degree"]:.6f} |
| binary | {st_b["density_offdiag"]:.6f} | {st_b["mean_degree"]:.6f} |

## 复现命令

```bash
python scripts/data/prepare_ettm_for_basicstfm.py \\
  --input-csv {inp} \\
  --output-dir {out_dir} \\
  --split {split[0]} {split[1]} {split[2]} \\
  --topk {args.topk} \\
  --use-abs-corr {'true' if args.use_abs_corr else 'false'} \\
  --threshold {'none' if thr is None else thr} \\
  --binary-graph {'true' if args.binary_graph else 'false'}
```
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("=== ETTm -> BasicSTFM 完成 ===")
    print(f"输出: {out_dir}")
    print(f"data shape: {tuple(data.shape)}")
    print(f"train 段长度（构图用）: {train_len}")
    print(f"adj weighted / binary offdiag density: {st_w['density_offdiag']:.6f} / {st_b['density_offdiag']:.6f}")


if __name__ == "__main__":
    main()
