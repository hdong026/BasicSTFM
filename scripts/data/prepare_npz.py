"""Convert raw spatio-temporal arrays into BasicSTFM's canonical NPZ format.

Supported inputs:
  - .npy
  - .npz
  - .h5 or .hdf5
  - .pkl or .pickle
  - .csv or .txt numeric matrix

Examples:
  python scripts/data/prepare_npz.py \
    --input data/raw_data/MY_DATASET/data.h5 \
    --output data/MY_DATASET/data.npz \
    --key data \
    --input-key df \
    --add-channel

  python scripts/data/prepare_npz.py \
    --input data/raw_data/MY_DATASET/adj.pkl \
    --output data/MY_DATASET/adj.npz \
    --key adj

  python scripts/data/prepare_npz.py \
    --input data/MY_DATASET/raw/data.csv \
    --output data/MY_DATASET/data.npz \
    --key data \
    --add-channel

  python scripts/data/prepare_npz.py \
    --input data/custom/raw/array.npy \
    --output data/custom/data.npz \
    --key data
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a BasicSTFM .npz data file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input .npy, .npz, .h5, .hdf5, .pkl, .pickle, .csv, or .txt file.",
    )
    parser.add_argument("--output", default=None, help="Output .npz file.")
    parser.add_argument("--key", default="data", help="Output key. Use 'data' or 'adj'.")
    parser.add_argument(
        "--input-key",
        default=None,
        help=(
            "Key to read from .npz/.h5 or dict-like .pkl. "
            "Defaults to the output key, then a format-specific fallback."
        ),
    )
    parser.add_argument(
        "--pkl-index",
        type=int,
        default=None,
        help=(
            "Index to read from tuple/list pickle files. "
            "For DCRNN/BasicTS-style adjacency tuples, key=adj defaults to index 2."
        ),
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
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List available keys/datasets for .npz, .h5, or dict-like .pkl and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_keys:
        list_available_keys(args.input)
        return
    if args.output is None:
        raise ValueError("--output is required unless --list-keys is used")

    array = load_array(
        args.input,
        key=args.input_key,
        output_key=args.key,
        delimiter=args.delimiter,
        pkl_index=args.pkl_index,
    )

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


def load_array(
    path: str,
    key: Optional[str],
    output_key: str,
    delimiter: str,
    pkl_index: Optional[int],
) -> np.ndarray:
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
    if suffix in {".h5", ".hdf5"}:
        return load_hdf5(input_path, key=key)
    if suffix in {".pkl", ".pickle"}:
        return load_pickle_array(input_path, key=key, output_key=output_key, pkl_index=pkl_index)
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


def load_hdf5(path: Path, key: Optional[str]) -> np.ndarray:
    """Load HDF5 from either pandas HDFStore or plain h5py datasets."""

    try:
        return load_pandas_hdf(path, key=key)
    except Exception as pandas_error:
        try:
            return load_h5py_dataset(path, key=key)
        except Exception as h5py_error:
            raise ValueError(
                f"Could not load HDF5 file {path}. "
                f"pandas error: {pandas_error}; h5py error: {h5py_error}"
            ) from h5py_error


def load_pandas_hdf(path: Path, key: Optional[str]) -> np.ndarray:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading pandas-style .h5 files requires pandas and tables. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    h5_key = normalize_pandas_hdf_key(path, key)
    frame = pd.read_hdf(path, key=h5_key)
    if hasattr(frame, "to_frame"):
        frame = frame.to_frame()
    return np.asarray(frame.values if hasattr(frame, "values") else frame)


def normalize_pandas_hdf_key(path: Path, key: Optional[str]) -> str:
    import pandas as pd

    if key:
        candidates = [key, f"/{key}" if not key.startswith("/") else key]
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
        for candidate in candidates:
            if candidate in keys:
                return candidate
        return candidates[-1]

    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
    if len(keys) == 1:
        return keys[0]
    if "/df" in keys:
        return "/df"
    raise ValueError(f"Multiple HDF5 keys found in {path}: {keys}. Pass --input-key.")


def load_h5py_dataset(path: Path, key: Optional[str]) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading plain HDF5 datasets requires h5py. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    with h5py.File(path, "r") as file:
        datasets = list_h5py_datasets(file)
        selected_key = select_named_key(datasets, key)
        return np.asarray(file[selected_key])


def list_h5py_datasets(group: Any, prefix: str = "") -> list[str]:
    datasets: list[str] = []
    for name, value in group.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "shape"):
            datasets.append(full_name)
        else:
            datasets.extend(list_h5py_datasets(value, full_name))
    return datasets


def load_pickle_array(
    path: Path,
    key: Optional[str],
    output_key: str,
    pkl_index: Optional[int],
) -> np.ndarray:
    obj = load_pickle(path)
    selected = select_pickle_object(obj, key=key, output_key=output_key, pkl_index=pkl_index)
    if hasattr(selected, "toarray"):
        selected = selected.toarray()
    return np.asarray(selected)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        try:
            return pickle.load(fp)
        except UnicodeDecodeError:
            fp.seek(0)
            return pickle.load(fp, encoding="latin1")


def select_pickle_object(
    obj: Any,
    key: Optional[str],
    output_key: str,
    pkl_index: Optional[int],
) -> Any:
    if isinstance(obj, dict):
        selected_key = select_named_key(obj.keys(), key)
        return obj[selected_key]

    if isinstance(obj, (tuple, list)):
        if pkl_index is not None:
            return obj[pkl_index]
        if output_key == "adj" and len(obj) >= 3:
            return obj[2]
        if len(obj) == 1:
            return obj[0]
        raise ValueError(
            "Tuple/list pickle has multiple values. Pass --pkl-index. "
            "For DCRNN/BasicTS-style adjacency pickles, use --key adj or --pkl-index 2."
        )

    return obj


def select_named_key(keys: Iterable[str], requested: Optional[str]) -> str:
    key_list = list(keys)
    if requested:
        candidates = [requested, requested.lstrip("/"), f"/{requested.lstrip('/')}"]
        for candidate in candidates:
            if candidate in key_list:
                return candidate
    if len(key_list) == 1:
        return key_list[0]
    for fallback in ("data", "df", "/df", "adj", "adj_mx"):
        if fallback in key_list:
            return fallback
    raise ValueError(f"Multiple keys/datasets found: {key_list}. Pass --input-key.")


def list_available_keys(path: str) -> None:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(input_path, allow_pickle=False)
        try:
            for key in loaded.files:
                print(key)
        finally:
            loaded.close()
        return
    if suffix in {".h5", ".hdf5"}:
        list_hdf5_keys(input_path)
        return
    if suffix in {".pkl", ".pickle"}:
        obj = load_pickle(input_path)
        if isinstance(obj, dict):
            for key in obj.keys():
                print(key)
        elif isinstance(obj, (tuple, list)):
            for index, value in enumerate(obj):
                shape = getattr(value, "shape", None)
                print(f"{index}: type={type(value).__name__}, shape={shape}")
        else:
            print(f"type={type(obj).__name__}, shape={getattr(obj, 'shape', None)}")
        return
    raise ValueError("--list-keys supports .npz, .h5, .hdf5, .pkl, and .pickle files")


def list_hdf5_keys(path: Path) -> None:
    try:
        import pandas as pd

        with pd.HDFStore(path, mode="r") as store:
            for key in store.keys():
                print(key)
        return
    except Exception:
        pass

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Listing plain HDF5 datasets requires h5py. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    with h5py.File(path, "r") as file:
        for key in list_h5py_datasets(file):
            print(key)


if __name__ == "__main__":
    main()
