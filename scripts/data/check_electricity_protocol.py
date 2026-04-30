#!/usr/bin/env python3
"""Inspect Electricity npz layout, global z-score vs RevIN batch stats, and numeric health."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to Electricity data.npz (default: <repo>/data/Electricity/data.npz)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--output-len", type=int, default=96)
    args = parser.parse_args()
    root = _repo_root()
    data_path = args.data_path or (root / "data" / "Electricity" / "data.npz")
    if not data_path.exists():
        print(f"Missing file: {data_path}", file=sys.stderr)
        return 1

    z = np.load(data_path, allow_pickle=True)
    keys = list(z.files)
    print("npz keys:", keys)
    data = z["data"] if "data" in z.files else z[z.files[0]]
    print("array used:", "data" if "data" in z.files else z.files[0])
    print("shape:", data.shape, "dtype:", data.dtype)

    if data.ndim != 3:
        print("Expected [T, N, C] array; got ndim=", data.ndim)
        return 1
    t_max, n, c = data.shape
    is_t_n_1 = c == 1
    print("layout looks like [T,N,1]:", is_t_n_1, "(C=%d)" % c)

    # Global standard (train-style): channel-wise axis (0,1)
    mu_g = data.mean(axis=(0, 1), keepdims=True)
    sg_g = data.std(axis=(0, 1), keepdims=True) + 1e-5
    per_node_mu = data.mean(axis=0)
    per_node_std = data.std(axis=0) + 1e-5
    print("global scaler mean shape", mu_g.shape, "vals", mu_g.reshape(-1)[: min(8, mu_g.size)])
    print(
        "per-node mean: min/med/max",
        float(np.min(per_node_mu)),
        float(np.median(per_node_mu)),
        float(np.max(per_node_mu)),
    )
    print(
        "per-node std: min/med/max",
        float(np.min(per_node_std)),
        float(np.median(per_node_std)),
        float(np.max(per_node_std)),
    )

    # Synthetic batch windows (deterministic offsets)
    in_len = int(args.input_len)
    hi = t_max - in_len - int(args.output_len)
    bs = min(int(args.batch_size), max(1, hi))
    if hi < 1:
        print("Time series too short for windowing; T=", t_max)
        return 1
    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(0, hi), size=bs, replace=False)
    xb = np.stack([data[s : s + in_len] for s in starts], axis=0)
    yb = np.stack(
        [data[s + in_len : s + in_len + int(args.output_len)] for s in starts],
        axis=0,
    )
    # [B,T,N,C]
    xv = xb[..., 0]
    mean_b = xv.mean(axis=1, keepdims=True)
    std_b = xv.std(axis=1, keepdims=True) + 1e-5
    print("RevIN batch mean shape", mean_b.shape)
    print(
        "RevIN std (ch0) across batch: min/med/max",
        float(np.min(std_b)),
        float(np.median(std_b)),
        float(np.max(std_b)),
    )

    finite = np.isfinite(data).all()
    print("all finite:", finite)
    if not finite:
        print("non-finite count:", int(np.size(data) - np.isfinite(data).sum()))
    q = np.quantile(data, [0.0, 0.01, 0.5, 0.99, 1.0])
    print("raw quantiles [min,1%,50%,99%,max]:", q.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
