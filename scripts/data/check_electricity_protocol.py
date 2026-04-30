#!/usr/bin/env python3
"""Inspect Electricity npz layout, global z-score vs RevIN batch stats, and numeric sanity."""

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
    parser.add_argument("--value-channel", type=int, default=0)
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
    print("shape (expect [T, N, C]):", data.shape, "dtype:", data.dtype)

    if data.ndim != 3:
        print("Expected [T, N, C] array; got ndim=", data.ndim)
        return 1
    vc = int(args.value_channel)
    if vc < 0 or vc >= data.shape[-1]:
        print(f"value_channel {vc} out of range for C={data.shape[-1]}", file=sys.stderr)
        return 1

    t_max, n_nodes, c = data.shape
    finite = np.isfinite(data).all()
    print("all finite:", finite)
    if not finite:
        print("non-finite count:", int(np.size(data) - np.isfinite(data).sum()))

    # Global standard (dataset-level): channel-wise over time + nodes
    mu_g = np.nanmean(np.where(np.isfinite(data), data, np.nan), axis=(0, 1), keepdims=True)
    sg_g = np.nanstd(np.where(np.isfinite(data), data, np.nan), axis=(0, 1), keepdims=True)
    sg_g = np.maximum(sg_g, 1e-5)
    print("global scaler mean (first channels):", mu_g.reshape(-1)[: min(8, mu_g.size)])
    print("global scaler std (first channels):", sg_g.reshape(-1)[: min(8, sg_g.size)])

    per_node_mu = data[:, :, vc].mean(axis=0)
    per_node_std = data[:, :, vc].std(axis=0) + 1e-5
    print(
        "per-node mean (value ch=%d): min / p10 / median / p90 / max ="
        % vc,
        float(np.min(per_node_mu)),
        float(np.quantile(per_node_mu, 0.1)),
        float(np.median(per_node_mu)),
        float(np.quantile(per_node_mu, 0.9)),
        float(np.max(per_node_mu)),
    )
    print(
        "per-node std (value ch=%d): min / p10 / median / p90 / max ="
        % vc,
        float(np.min(per_node_std)),
        float(np.quantile(per_node_std, 0.1)),
        float(np.median(per_node_std)),
        float(np.quantile(per_node_std, 0.9)),
        float(np.max(per_node_std)),
    )

    g_std = float(sg_g.reshape(-1)[vc])
    ratio = per_node_std / g_std
    print(
        "global-standard vs per-node std (ch=%d): per_node_std / global_std —"
        " min / median / max =" % vc,
        float(np.min(ratio)),
        float(np.median(ratio)),
        float(np.max(ratio)),
    )

    q = np.quantile(data[:, :, vc], [0.0, 0.01, 0.5, 0.99, 1.0])
    print("raw value-channel quantiles [min,1%,50%,99%,max]:", q.tolist())

    in_len = int(args.input_len)
    out_len = int(args.output_len)
    hi = t_max - in_len - out_len
    bs = min(int(args.batch_size), max(1, hi))
    if hi < 1:
        print("Time series too short for windowing; T=", t_max)
        return 1
    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(0, hi), size=bs, replace=False)
    xb = np.stack([data[s : s + in_len] for s in starts], axis=0)
    # [B, T, N, C]
    xv = xb[..., vc]
    mean_b = xv.mean(axis=1, keepdims=True)
    std_b = xv.std(axis=1, keepdims=True) + 1e-5
    x_norm = (xv - mean_b) / std_b
    # After RevIN, stats along time per (batch, node)
    m_after = x_norm.mean(axis=1)
    s_after = x_norm.std(axis=1)
    print("RevIN-normalized input windows (ch=%d):" % vc)
    print(
        "  mean over T: abs max / median abs =",
        float(np.max(np.abs(m_after))),
        float(np.median(np.abs(m_after))),
    )
    print(
        "  std over T: min / median / max =",
        float(np.min(s_after)),
        float(np.median(s_after)),
        float(np.max(s_after)),
    )
    print("(expect mean ~0 and std ~1 per window after FactoST-style RevIN.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
