"""Batch-prepare datasets under a raw data directory.

The script scans directories such as:

  data/raw_data/
    LargeST/
      ca_his_raw_2017.h5
      ca_his_raw_2018.h5
      ca_rn_adj.npy
    METR-LA/
      METR-LA.h5
      adj_METR-LA.pkl
    PEMS04/
      PEMS04.npz
      adj_PEMS04.pkl

and writes:

  data/<DATASET_NAME>/data.npz
  data/<DATASET_NAME>/adj.npz
  data/<DATASET_NAME>/README.md

The heuristics are conservative but useful for BasicTS/DCRNN-style datasets:
  - data files prefer .npz, then .h5/.hdf5, then non-adjacency .csv/.npy.
  - adjacency files prefer names containing "adj", "graph", "distance", or "dist".
  - multiple data files are concatenated along the time axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from prepare_npz import load_array

DATA_SUFFIXES = {".npz", ".h5", ".hdf5", ".csv", ".txt", ".npy"}
ADJ_SUFFIXES = {".pkl", ".pickle", ".npy", ".npz", ".csv", ".txt"}
ADJ_TOKENS = ("adj", "adjacency", "graph", "distance", "dist")
META_TOKENS = ("meta", "location", "locations", "sensor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-prepare BasicSTFM datasets.")
    parser.add_argument("--raw-root", default="data/raw_data", help="Root directory of raw datasets.")
    parser.add_argument("--output-root", default="data", help="Prepared data output root.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to process. Defaults to every directory under raw-root.",
    )
    parser.add_argument("--data-key", default="data", help="Output key for data.npz.")
    parser.add_argument("--adj-key", default="adj", help="Output key for adj.npz.")
    parser.add_argument("--input-key", default=None, help="Optional key for all HDF5/NPZ data files.")
    parser.add_argument("--dtype", default="float32", help="Output dtype.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    parser.add_argument(
        "--allow-missing-adj",
        action="store_true",
        help="Prepare data.npz even when no adjacency file is found.",
    )
    parser.add_argument(
        "--no-add-channel",
        action="store_true",
        help="Do not convert [T, N] arrays to [T, N, 1].",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    dataset_dirs = discover_dataset_dirs(raw_root, args.datasets)

    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {raw_root}")

    for raw_dir in dataset_dirs:
        plan = build_plan(raw_dir)
        print_plan(plan)
        if args.dry_run:
            continue
        prepare_dataset(
            plan=plan,
            output_root=output_root,
            data_key=args.data_key,
            adj_key=args.adj_key,
            input_key=args.input_key,
            dtype=args.dtype,
            overwrite=args.overwrite,
            allow_missing_adj=args.allow_missing_adj,
            add_channel=not args.no_add_channel,
        )


def discover_dataset_dirs(raw_root: Path, names: Optional[list[str]]) -> list[Path]:
    if names:
        return [raw_root / name for name in names]
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


class DatasetPlan:
    def __init__(self, name: str, raw_dir: Path, data_files: list[Path], adj_file: Optional[Path]):
        self.name = name
        self.raw_dir = raw_dir
        self.data_files = data_files
        self.adj_file = adj_file


def build_plan(raw_dir: Path) -> DatasetPlan:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    files = sorted(path for path in raw_dir.iterdir() if path.is_file())
    data_files = select_data_files(files)
    adj_file = select_adj_file(files)
    if not data_files:
        raise FileNotFoundError(f"No data files found in {raw_dir}")
    return DatasetPlan(name=raw_dir.name, raw_dir=raw_dir, data_files=data_files, adj_file=adj_file)


def select_data_files(files: Iterable[Path]) -> list[Path]:
    candidates = [
        path
        for path in files
        if path.suffix.lower() in DATA_SUFFIXES
        and not is_adjacency_file(path)
        and not is_metadata_file(path)
    ]
    if not candidates:
        return []

    for suffix_group in ((".npz",), (".h5", ".hdf5"), (".csv", ".txt"), (".npy",)):
        group = [path for path in candidates if path.suffix.lower() in suffix_group]
        if group:
            return sorted(group)
    return sorted(candidates)


def select_adj_file(files: Iterable[Path]) -> Optional[Path]:
    candidates = [
        path
        for path in files
        if path.suffix.lower() in ADJ_SUFFIXES and is_adjacency_file(path)
    ]
    if not candidates:
        return None
    preferred_suffixes = (".pkl", ".pickle", ".npy", ".npz", ".csv", ".txt")
    for suffix in preferred_suffixes:
        group = [path for path in candidates if path.suffix.lower() == suffix]
        if group:
            return sorted(group, key=adj_priority)[0]
    return sorted(candidates, key=adj_priority)[0]


def is_adjacency_file(path: Path) -> bool:
    name = path.stem.lower()
    return any(token in name for token in ADJ_TOKENS)


def is_metadata_file(path: Path) -> bool:
    name = path.stem.lower()
    return any(token in name for token in META_TOKENS)


def adj_priority(path: Path) -> tuple[int, str]:
    name = path.stem.lower()
    if name.startswith("adj_") or name.startswith("adj"):
        return (0, path.name)
    if "distance" in name or "dist" in name:
        return (2, path.name)
    return (1, path.name)


def print_plan(plan: DatasetPlan) -> None:
    print(f"[{plan.name}]")
    print("  data:")
    for path in plan.data_files:
        print(f"    - {path}")
    print(f"  adj: {plan.adj_file if plan.adj_file else '<missing>'}")


def prepare_dataset(
    plan: DatasetPlan,
    output_root: Path,
    data_key: str,
    adj_key: str,
    input_key: Optional[str],
    dtype: str,
    overwrite: bool,
    allow_missing_adj: bool,
    add_channel: bool,
) -> None:
    output_dir = output_root / plan.name
    data_out = output_dir / "data.npz"
    adj_out = output_dir / "adj.npz"
    if data_out.exists() and not overwrite:
        print(f"  skip existing {data_out} (pass --overwrite to regenerate)")
        return
    if plan.adj_file is None and not allow_missing_adj:
        raise FileNotFoundError(
            f"No adjacency file found for {plan.name}. "
            "Pass --allow-missing-adj if this dataset has no graph."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_stack_data(plan.data_files, key=input_key, dtype=dtype)
    if add_channel and data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"{plan.name}: expected data shape [T, N, C], got {data.shape}")
    np.savez_compressed(data_out, **{data_key: data})
    print(f"  saved {data_key}: shape={data.shape} -> {data_out}")

    adj_shape = None
    if plan.adj_file is not None:
        adj = load_array(
            str(plan.adj_file),
            key=adj_key,
            output_key=adj_key,
            delimiter=",",
            pkl_index=None,
        ).astype(dtype, copy=False)
        if adj.ndim != 2:
            raise ValueError(f"{plan.name}: expected adjacency shape [N, N], got {adj.shape}")
        np.savez_compressed(adj_out, **{adj_key: adj})
        adj_shape = adj.shape
        print(f"  saved {adj_key}: shape={adj.shape} -> {adj_out}")

    write_readme(
        output_dir=output_dir,
        plan=plan,
        data_shape=data.shape,
        adj_shape=adj_shape,
        data_key=data_key,
        adj_key=adj_key,
    )


def load_and_stack_data(files: list[Path], key: Optional[str], dtype: str) -> np.ndarray:
    arrays = []
    for path in files:
        array = load_array(
            str(path),
            key=key,
            output_key="data",
            delimiter=",",
            pkl_index=None,
        )
        if array.ndim == 1:
            array = array[:, None]
        arrays.append(array.astype(dtype, copy=False))
    if len(arrays) == 1:
        return arrays[0]
    shapes = {array.shape[1:] for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Cannot concatenate files with different non-time shapes: {shapes}")
    return np.concatenate(arrays, axis=0)


def write_readme(
    output_dir: Path,
    plan: DatasetPlan,
    data_shape: tuple[int, ...],
    adj_shape: Optional[tuple[int, ...]],
    data_key: str,
    adj_key: str,
) -> None:
    raw_files = "\n".join(f"- `{path.name}`" for path in plan.data_files)
    adj_file = f"`{plan.adj_file.name}`" if plan.adj_file else "<missing>"
    adj_line = f"{adj_key}: shape={adj_shape}" if adj_shape is not None else "adj: <missing>"
    graph_path_line = (
        f"  data.graph_path=data/{plan.name}/adj.npz \\\n"
        if adj_shape is not None
        else ""
    )
    text = f"""# {plan.name}

Prepared for BasicSTFM by `scripts/data/prepare_all.py`.

## Raw Directory

`{plan.raw_dir}`

## Selected Time-Series Files

{raw_files}

## Selected Adjacency File

{adj_file}

## Prepared Files

```text
data.npz
  {data_key}: shape={data_shape}

adj.npz
  {adj_line}
```

## Run Example

```bash
basicstfm train configs/examples/file_forecasting.yaml \\
  --cfg-options \\
  data.data_path=data/{plan.name}/data.npz \\
{graph_path_line.rstrip()}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
