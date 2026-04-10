"""Convert simple raw arrays into BasicSTFM's canonical NPZ format.

Supported inputs:
  - .npy
  - .npz
  - .csv or .txt numeric matrix

Examples:
  python scripts/data/prepare_npz.py \
    --input data/METR-LA/raw/metr_la.csv \
    --output data/METR-LA/data.npz \
    --key data \
    --add-channel

  python scripts/data/prepare_npz.py \
    --input data/custom/raw/array.npy \
    --output data/custom/data.npz \
    --key data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a BasicSTFM .npz data file.")
    parser.add_argument("--input", required=True, help="Input .npy, .npz, .csv, or .txt file.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--key", default="data", help="Output key. Use 'data' or 'adj'.")
    parser.add_argument(
        "--input-key",
        default=None,
        help="Key to read when the input is .npz. Defaults to the output key, then first key.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV/TXT delimiter. Default: ','. Use '\\t' for tab-separated files.",
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=None,
        help="Optional target shape, e.g. --shape 34272 207 1.",
    )
    parser.add_argument(
        "--add-channel",
        action="store_true",
        help="Convert a [T, N] matrix into [T, N, 1].",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output dtype. Default: float32.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    array = load_array(args.input, key=args.input_key or args.key, delimiter=args.delimiter)

    if args.shape is not None:
        array = array.reshape(tuple(args.shape))
    if args.add_channel:
        if array.ndim != 2:
            raise ValueError("--add-channel expects a 2D array shaped [T, N]")
        array = array[..., None]

    array = array.astype(args.dtype, copy=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{args.key: array})
    print(f"saved {args.key} with shape {array.shape} to {output}")


def load_array(path: str, key: Optional[str], delimiter: str) -> np.ndarray:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        return np.load(input_path, allow_pickle=False)
    if suffix == ".npz":
        loaded = np.load(input_path, allow_pickle=False)
        selected_key = select_npz_key(loaded, key)
        array = loaded[selected_key].copy()
        loaded.close()
        return array
    if suffix in {".csv", ".txt"}:
        delimiter = "\t" if delimiter == "\\t" else delimiter
        return np.genfromtxt(input_path, delimiter=delimiter)
    raise ValueError(f"Unsupported input extension: {suffix}")


def select_npz_key(loaded: np.lib.npyio.NpzFile, key: Optional[str]) -> str:
    if key and key in loaded.files:
        return key
    if loaded.files:
        return loaded.files[0]
    raise ValueError("Input .npz file contains no arrays")


if __name__ == "__main__":
    main()
