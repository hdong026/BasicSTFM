"""Stream LargeST yearly HDF5 files into one memory-mappable float32 array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LargeST yearly HDF5 files into one float32 .npy array."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_data/LargeST",
        help="Directory containing ca_his_raw_<year>.h5 and ca_rn_adj.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LargeST_full",
        help="Directory to write data.npy / adj.npy / metadata.json",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2017", "2018", "2019", "2020", "2021"],
        help="Year suffixes to concatenate in order.",
    )
    parser.add_argument(
        "--input-key",
        default="t",
        help="Pandas HDF key. Both `t` and `/t` are accepted.",
    )
    parser.add_argument(
        "--adj-name",
        default="ca_rn_adj.npy",
        help="Adjacency filename inside input-dir.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Output dtype for the concatenated data and adjacency.",
    )
    parser.add_argument(
        "--no-add-channel",
        action="store_true",
        help="Keep 2D output when source is [T, N]. Default adds a singleton channel dimension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npy"
    adj_path = output_dir / "adj.npy"
    meta_path = output_dir / "metadata.json"

    if (data_path.exists() or adj_path.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains output files. Pass --overwrite to regenerate."
        )

    year_files = [input_dir / f"ca_his_raw_{year}.h5" for year in args.years]
    for path in year_files:
        if not path.exists():
            raise FileNotFoundError(path)

    dtype = np.dtype(args.dtype)
    shapes = []
    for path in year_files:
        array = load_pandas_hdf(path, args.input_key)
        if array.ndim == 1:
            array = array[:, None]
        shapes.append(array.shape)
        print(f"Scanned {path.name}: shape={array.shape}, dtype={array.dtype}")

    feature_shapes = {shape[1:] for shape in shapes}
    if len(feature_shapes) != 1:
        raise ValueError(f"Cannot concatenate yearly files with different non-time shapes: {feature_shapes}")

    total_steps = sum(shape[0] for shape in shapes)
    trailing_shape = next(iter(feature_shapes))
    output_shape = (total_steps, *trailing_shape)
    if not args.no_add_channel and len(output_shape) == 2:
        output_shape = (*output_shape, 1)

    writer = np.lib.format.open_memmap(data_path, mode="w+", dtype=dtype, shape=output_shape)
    cursor = 0
    for path in year_files:
        array = load_pandas_hdf(path, args.input_key)
        if array.ndim == 1:
            array = array[:, None]
        if not args.no_add_channel and array.ndim == 2:
            array = array[..., None]
        if array.shape[1:] != output_shape[1:]:
            raise ValueError(
                f"{path.name}: expected trailing shape {output_shape[1:]}, got {array.shape[1:]}"
            )
        next_cursor = cursor + array.shape[0]
        writer[cursor:next_cursor] = array.astype(dtype, copy=False)
        writer.flush()
        print(f"Wrote {path.name}: steps={array.shape[0]} into [{cursor}:{next_cursor})")
        cursor = next_cursor

    adjacency = np.load(input_dir / args.adj_name, allow_pickle=False).astype(dtype, copy=False)
    np.save(adj_path, adjacency)

    metadata = {
        "source_dir": str(input_dir),
        "years": list(args.years),
        "input_key": str(args.input_key),
        "dtype": str(dtype),
        "data_path": str(data_path),
        "adj_path": str(adj_path),
        "data_shape": list(output_shape),
        "adj_shape": list(adjacency.shape),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved data -> {data_path}")
    print(f"Saved adj  -> {adj_path}")
    print(f"Saved meta -> {meta_path}")


def load_pandas_hdf(path: Path, key: Optional[str]) -> np.ndarray:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading pandas-style .h5 files requires pandas and tables."
        ) from exc

    h5_key = normalize_pandas_hdf_key(path, key)
    frame = pd.read_hdf(path, key=h5_key)
    if hasattr(frame, "to_frame"):
        frame = frame.to_frame()
    return np.asarray(frame.values if hasattr(frame, "values") else frame)


def normalize_pandas_hdf_key(path: Path, key: Optional[str]) -> str:
    import pandas as pd

    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
    if key:
        candidates = [key, f"/{key}" if not key.startswith("/") else key]
        for candidate in candidates:
            if candidate in keys:
                return candidate
        return candidates[-1]
    if len(keys) == 1:
        return keys[0]
    if "/df" in keys:
        return "/df"
    raise ValueError(f"Multiple HDF5 keys found in {path}: {keys}. Pass --input-key.")


if __name__ == "__main__":
    main()
