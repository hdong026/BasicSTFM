#!/usr/bin/env python3
"""Prepare Monash-15 subsets from HuggingFace ``Monash-University/monash_tsf``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np


DEFAULT_MONASH15: tuple[str, ...] = (
    "australian_electricity_demand",
    "london_smart_meters",
    "solar_10_minutes",
    "solar_4_seconds",
    "wind_farms_minutely",
    "wind_4_seconds",
    "weather",
    "kdd_cup_2018",
    "temperature_rain",
    "saugeenday",
    "sunspot",
    "pedestrian_counts",
    "kaggle_web_traffic",
    "bitcoin",
    "us_births",
)


def _flatten_target_to_series(target: Any) -> List[np.ndarray]:
    """Yield univariate series as float32 1D arrays (may contain NaNs)."""
    arr = np.asarray(target, dtype=np.float32)
    series: List[np.ndarray] = []
    if arr.size == 0:
        return series
    if arr.ndim <= 1:
        series.append(np.ascontiguousarray(arr.reshape(-1), dtype=np.float32))
        return series
    if arr.ndim == 2:
        for row in arr:
            series.append(np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(-1)))
        return series
    raise ValueError(f"Unsupported target ndim={arr.ndim}: shape={arr.shape}")


def _estimate_windows(n_points: int, input_len: int, output_len: int, stride: int) -> int:
    if input_len <= 0 or output_len <= 0 or n_points <= 0:
        return 0
    need = input_len + output_len
    if n_points < need:
        return 0
    stride = max(1, stride)
    return (n_points - need) // stride + 1


def _load_hf_split(
    ds_name: str,
    *,
    splits: Sequence[str],
    tqdm_fn: Callable[[Iterable], Iterable] | None,
) -> list[dict]:
    """Load and merge HF rows from one or more split names."""

    try:
        from datasets import concatenate_datasets, load_dataset  # noqa: PLC0415 — optional dep
    except ImportError as exc:
        raise ImportError(
            'Optional dependency missing: pip install datasets  (requires huggingface hub access for "hf" source)'
        ) from exc

    parts = []
    for split in splits:
        sub = load_dataset(
            "Monash-University/monash_tsf",
            ds_name,
            split=str(split),
            trust_remote_code=True,
        )
        parts.append(sub)
    merged = concatenate_datasets(parts)
    iterable = tqdm_fn(merged) if tqdm_fn else merged
    out: List[dict] = []
    for row in iterable:
        out.append(dict(row))
    return out


def extract_series_for_dataset(_ds_name: str, rows: Sequence[dict]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Unpack HF rows into a list of 1D float32 series."""

    all_series: list[np.ndarray] = []
    skipped_zero = 0

    for row in rows:
        target = row.get("target")
        for seq in _flatten_target_to_series(target):
            seq = seq.reshape(-1)
            if seq.size == 0:
                skipped_zero += 1
                continue
            all_series.append(seq)

    extra: dict[str, Any] = {
        "skipped_empty_series": skipped_zero,
    }
    return all_series, extra


def materialize_arrays(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.concatenate(series_list).astype(np.float32, copy=False)
    lengths = np.asarray([len(s) for s in series_list], dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if offsets.size:
        ofs = 0
        for i in range(len(lengths)):
            offsets[i] = ofs
            ofs += int(lengths[i])
        if ofs != len(values):
            raise RuntimeError("Internal concat offset mismatch")
    return values, offsets, lengths


def process_one_dataset(
    ds_name: str,
    output_root: Path,
    *,
    hf_splits: Sequence[str],
    skip_download: bool,
    tqdm_disable: bool,
) -> Path:
    from tqdm.auto import tqdm  # noqa: PLC0411

    tqdm_fn = (
        (lambda iterable: tqdm(iterable, desc=f"HF:{ds_name}", disable=tqdm_disable))
        if not skip_download
        else None
    )
    rows = _load_hf_split(ds_name, splits=hf_splits, tqdm_fn=tqdm_fn)
    series_list, _extra = extract_series_for_dataset(ds_name, rows)

    dst = output_root / ds_name
    dst.mkdir(parents=True, exist_ok=True)

    values, offsets, lengths = materialize_arrays(series_list)
    meta: dict[str, Any] = {
        "dataset_name": ds_name,
        "hf_dataset": "Monash-University/monash_tsf",
        "hf_splits": list(hf_splits),
        "n_series": int(len(lengths)),
        "total_timesteps": int(values.size),
    }

    np.save(dst / "values.npy", values)
    np.save(dst / "offsets.npy", offsets)
    np.save(dst / "lengths.npy", lengths)

    with (dst / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return dst


def summarize_dir(
    path: Path,
    *,
    input_len: int,
    output_len: int,
    stride: int,
    split: Sequence[float],
) -> dict[str, Any]:
    values = np.load(path / "values.npy", mmap_mode="r")
    offsets = np.load(path / "offsets.npy")
    lengths = np.load(path / "lengths.npy")

    st = max(1, stride)
    try:
        from basicstfm.data.monash_datamodule import enumerate_window_placements
    except Exception:
        wl = input_len + output_len
        windows_glob = 0
        for ln in lengths:
            ln = int(ln)
            if ln < wl:
                continue
            windows_glob += (ln - wl) // st + 1
        split_windows = {"train": windows_glob, "val": 0, "test": 0, "fallback": True}
    else:
        split_windows = {
            "train": len(
                enumerate_window_placements(offsets, lengths, split, input_len, output_len, stride, "train")
            ),
            "val": len(
                enumerate_window_placements(offsets, lengths, split, input_len, output_len, stride, "val")
            ),
            "test": len(
                enumerate_window_placements(offsets, lengths, split, input_len, output_len, stride, "test")
            ),
            "fallback": False,
        }

    payload = {
        "path": str(path),
        "n_series": int(len(lengths)),
        "total_timesteps": int(values.size),
        "estimated_windows_split": split_windows,
        "split": list(split),
        "input_len": input_len,
        "output_len": output_len,
        "stride": st,
    }
    meta_path = path / "meta.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            payload["meta_json"] = json.load(f)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("hf",), default="hf")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Monash15"),
        help="Directory to write datasets (each subfolder is one Monash HF config name)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_MONASH15),
        help="HF config names under monash_tsf (default: Monash15 list)",
    )
    p.add_argument(
        "--hf-splits",
        nargs="+",
        default=["train"],
        help="HF split names combined per dataset (default: train only)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="Alias for caching tests (same code path)")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only print counts for existing dirs under output-dir (--input/output/stride)",
    )
    p.add_argument("--input-len", type=int, default=96)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.1, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Per-series chronological split ratios (matches datamodule Monash loaders)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.output_dir)

    if args.summarize_only:
        roots = sorted(p for p in root.iterdir() if p.is_dir() and (p / "lengths.npy").exists())
        if not roots:
            print(json.dumps({"error": "no datasets found", "dir": str(root)}, indent=2))
            return
        all_payload = []
        for d in roots:
            info = summarize_dir(
                d,
                input_len=args.input_len,
                output_len=args.output_len,
                stride=args.stride,
                split=tuple(args.split),
            )
            all_payload.append(info)
            print(json.dumps(info, indent=2, sort_keys=False))
        return

    if args.source != "hf":
        raise ValueError("Only source=hf is implemented")

    for name in args.datasets:
        out_dir = root / name
        val_file = out_dir / "values.npy"
        if val_file.exists() and not args.overwrite:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        process_one_dataset(
            name,
            root,
            hf_splits=args.hf_splits,
            skip_download=args.skip_download,
            tqdm_disable=args.no_tqdm,
        )


if __name__ == "__main__":
    main()
