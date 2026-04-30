#!/usr/bin/env python3
"""THUML Weather benchmark (TSLib / Autoformer preprocessing) -> BasicSTFM.

Primary CSV: ``weather.csv`` from
`thuml/Time-Series-Library <https://huggingface.co/datasets/thuml/Time-Series-Library>`_
(path ``weather/weather.csv``), matching the Autoformer / TSLib preprocessed release.

Optional: train-split-only Pearson top-k graph (same convention as ``prepare_ettm_for_basicstfm.py``).

Missing / invalid entries in some TSLib releases are encoded as **±9999**; by default this script
masks those (and non-finite values), then applies **per-variable linear interpolation along time**
with **nearest-valid** edge fill before writing ``data.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse CSV + graph helpers from ETTm preparer (same multivariate layout).
_SCRIPTS_DATA = Path(__file__).resolve().parent
if str(_SCRIPTS_DATA) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA))

import prepare_ettm_for_basicstfm as ett  # noqa: E402
import timeseries_sentinel_clean as tsc  # noqa: E402

# Autoformer README: Google Drive folder with all six benchmarks (incl. Weather).
AUTOFORMER_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing"
)
# TSLib README: newer Drive + HuggingFace mirror.
TSLIB_DRIVE_URL = (
    "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing"
)
HF_WEATHER_CSV = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/"
    "resolve/main/weather/weather.csv"
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def infer_time_granularity_minutes(times: pd.Series) -> str:
    s = pd.to_datetime(times.iloc[: min(500, len(times))], errors="coerce")
    if s.isna().all():
        return "unknown"
    d = s.diff().dropna()
    if d.empty:
        return "unknown"
    sec = d.dt.total_seconds().median()
    if not np.isfinite(sec) or sec <= 0:
        return "unknown"
    mins = sec / 60.0
    if abs(mins - round(mins)) < 0.05:
        return f"{int(round(mins))}-minute"
    return f"~{mins:.2f}-minute (median spacing)"


def parse_threshold(s: str) -> float | None:
    if s is None or str(s).lower() in ("none", "", "null"):
        return None
    return float(s)


def str2bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weather.csv -> BasicSTFM data.npz (+ optional adj)")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Local weather.csv (TSLib layout). Default: <repo>/data/raw_data/Weather/weather.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/data/Weather",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help=f"Download CSV from Hugging Face mirror: {HF_WEATHER_CSV}",
    )
    p.add_argument(
        "--build-adj",
        action="store_true",
        help="Build train-only Pearson top-k adjacency (adj.npz + adj_corr_topk + adj_binary_topk).",
    )
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--use-abs-corr", type=str2bool, default=True)
    p.add_argument("--threshold", type=str, default="none")
    p.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=(0.7, 0.1, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="For graph construction / README; matches WindowDataModule.",
    )
    p.add_argument(
        "--missing-warn-frac",
        type=float,
        default=0.3,
        help="Warn per-variable when sentinel/missing fraction before interpolation exceeds this.",
    )
    p.add_argument(
        "--no-sentinel-clean",
        action="store_true",
        help="Disable ±9999 / non-finite masking and temporal interpolation (not recommended).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "data/raw_data/Weather/weather.csv"
    inp = Path(args.input_csv) if args.input_csv is not None else default_csv
    out_dir = Path(args.output_dir) if args.output_dir is not None else repo_root / "data/Weather"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download or not inp.exists():
        if not inp.exists() and not args.download:
            raise FileNotFoundError(
                f"{inp} not found. Run with --download or place TSLib weather.csv there.\n"
                f"  Autoformer (README): {AUTOFORMER_DRIVE_URL}\n"
                f"  TSLib Drive: {TSLIB_DRIVE_URL}\n"
                f"  TSLib Hugging Face: https://huggingface.co/datasets/thuml/Time-Series-Library"
            )
        print(f"Downloading -> {inp}")
        download_file(HF_WEATHER_CSV, inp)

    _, time_col, node_names, values = ett.load_ett_csv(inp)
    df_head = pd.read_csv(inp, nrows=min(5000, len(values) + 1))
    gran = infer_time_granularity_minutes(df_head[time_col])

    cleaning_meta: dict = {"applied": False}
    if args.no_sentinel_clean:
        cleaned = values.astype(np.float32)
    else:
        masked = tsc.mask_sentinels_and_nonfinite(values, sentinel_values=tsc.DEFAULT_SENTINELS)
        cleaned, interp_meta = tsc.interpolate_columns_linear_then_edge(
            masked,
            missing_warn_frac=float(args.missing_warn_frac),
            variable_names=node_names,
        )
        cleaning_meta = {
            "applied": True,
            "sentinel_values": list(tsc.DEFAULT_SENTINELS),
            "interpolation": interp_meta,
        }
        for w in interp_meta.get("warnings", []):
            print(f"[weather-clean] WARNING: {w}", file=sys.stderr)

    data = cleaned.astype(np.float32)[..., None]
    t, n, c = data.shape

    meta = {
        "source": "THUML Time-Series-Library / Autoformer preprocessed Weather",
        "download_urls_documented": {
            "autoformer_readme_google_drive": AUTOFORMER_DRIVE_URL,
            "tslib_readme_google_drive": TSLIB_DRIVE_URL,
            "tslib_huggingface_dataset": "https://huggingface.co/datasets/thuml/Time-Series-Library",
            "weather_csv_hf_resolve": HF_WEATHER_CSV,
        },
        "local_csv": str(inp.resolve()),
        "time_column": time_col,
        "inferred_granularity": gran,
        "num_timesteps": t,
        "num_variables": n,
        "variable_names": node_names,
        "data_shape_canonical": [t, n, c],
        "sentinel_cleaning": cleaning_meta,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    np.savez_compressed(out_dir / "data.npz", data=data)

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

    split = tuple(float(x) for x in args.split)
    train_len, val_len, test_len = ett.split_lengths(t, split)

    adj_md: dict = {"built": False}
    if args.build_adj:
        thr = parse_threshold(args.threshold)
        train_x = data[:train_len, :, 0].astype(np.float64)
        corr = ett.pearson_corr_matrix(train_x)
        dmask = ett.topk_directed_mask(corr, args.topk, threshold=thr)
        adj_w = ett.build_adjacency_pair(
            corr, dmask, weighted=True, use_abs_corr=args.use_abs_corr
        )
        adj_b = ett.build_adjacency_pair(
            corr, dmask, weighted=False, use_abs_corr=args.use_abs_corr
        )
        np.savez_compressed(out_dir / "adj_corr_topk.npz", adj=adj_w)
        np.savez_compressed(out_dir / "adj_binary_topk.npz", adj=adj_b)
        np.savez_compressed(out_dir / "adj.npz", adj=adj_w)
        st_w = ett.adj_offdiag_stats(adj_w)
        st_b = ett.adj_offdiag_stats(adj_b)
        adj_md = {
            "built": True,
            "from_train_timesteps_only": train_len,
            "topk": args.topk,
            "use_abs_corr": args.use_abs_corr,
            "threshold": thr,
            "weighted_offdiag_density": st_w["density_offdiag"],
            "binary_offdiag_density": st_b["density_offdiag"],
        }
        (out_dir / "graph_meta.json").write_text(json.dumps(adj_md, indent=2), encoding="utf-8")

    readme = f"""# Weather (TSLib / Autoformer) — BasicSTFM

## 来源

- **Autoformer** 官方 README 数据入口：[Google Drive 六大数据集]({AUTOFORMER_DRIVE_URL})
- **Time-Series-Library** 入口：[Google Drive]({TSLIB_DRIVE_URL}) · [Hugging Face 数据集](https://huggingface.co/datasets/thuml/Time-Series-Library)
- 本仓库使用的预处理文件：`weather.csv`（与 TSLib `dataset/weather/` 一致）
- 本地路径：`{inp.resolve()}`

## 原始 CSV 概要

- **行数（含表头）**：{t + 1}；**时间步 T**：{t}
- **时间列**：`{time_col}`（不作为节点）
- **推断时间粒度**：{gran}
- **变量数 N**：{n}（每变量一节点，multivariate 设定）
- **列名**：{", ".join(f"`{x}`" for x in node_names)}

## BasicSTFM 格式

- `data.npz` key `data`：shape **`[{t}, {n}, 1]`** = `[T, N, C]`

## 划分（与 WindowDataModule 一致）

- `split` = {list(split)} → train / val / test 长度 **{train_len} / {val_len} / {test_len}**

## 图（可选）

- **{"已生成" if args.build_adj else "未生成"}** train 段 Pearson top-k 图。
- 若已生成：见 `adj.npz`（默认加权）、`adj_corr_topk.npz`、`adj_binary_topk.npz`、`graph_meta.json`。

## 复现

```bash
python scripts/data/prepare_weather_for_basicstfm.py --download --build-adj --topk {args.topk}
```

Sentinel cleanup (±9999 / NaN / Inf), temporal interpolation, and edge fill run **by default**
(use ``--no-sentinel-clean`` only for debugging raw CSV dumps).
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("=== Weather -> BasicSTFM ===")
    print(f"T={t}, N={n}, C={c}; granularity ~ {gran}")
    print(f"data -> {out_dir / 'data.npz'}")
    if args.build_adj:
        print(f"adj (train_len={train_len}, topk={args.topk}) -> {out_dir / 'adj.npz'}")


if __name__ == "__main__":
    main()
