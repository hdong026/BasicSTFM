#!/usr/bin/env python3
"""Convert ``raw_data/PEMS03/PEMS03.h5`` (pandas HDFStore) -> ``data/PEMS03/data.npz``.

Typical layout under ``PEMS03.h5`` (pytables): ``block0_values`` with shape ``(T, N)``.
This script writes ``data.npz`` with key ``data`` and shape ``[T, N, 1]`` (float32),
matching :func:`basicstfm.data.datamodule._load_numpy` expectations.

Also writes ``data/PEMS03/adj.npz`` with key ``adj`` (dense ``[N, N]``, float32).

Supports ``raw_data/PEMS03/adj.npz`` as either a dense square matrix or a **SciPy COO-packed**
sparse file (arrays ``row``, ``col``, ``data``, ``shape``), which often appears as 1D arrays of
length ``nnz`` (e.g. 870 entries for a sparse 358×358 graph).

Next step for BasicTS-style downstream::

    python scripts/data/build_basicts_traffic_targets.py --datasets PEMS03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_T = 26208
EXPECTED_N = 358


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_h5_matrix(h5_path: Path) -> np.ndarray:
    last_err: Exception | None = None
    for key in ("data", "/data"):
        try:
            df = pd.read_hdf(h5_path, key=key)
            break
        except (KeyError, ValueError, OSError) as exc:
            last_err = exc
            df = None
    if df is None:
        raise RuntimeError(f"Could not read {h5_path} with keys 'data' or '/data': {last_err}") from last_err

    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64, copy=True))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D DataFrame values, got shape {arr.shape}")

    # Heuristic: PEMS03 is [T=26208, N=358]. Some exports are transposed.
    if arr.shape == (EXPECTED_N, EXPECTED_T):
        arr = arr.T
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] in (EXPECTED_N,):
        arr = arr.T

    return arr.astype(np.float32, copy=False)


def _decode_npz_scalar_bytes(arr: np.ndarray) -> str:
    if arr.dtype.kind not in {"S", "O", "U"}:
        return ""
    flat = np.ravel(arr)
    if flat.size == 0:
        return ""
    val = flat[0]
    if isinstance(val, bytes):
        return val.decode("ascii", errors="ignore").strip().lower()
    return str(val).strip().lower()


def _load_dense_adj(path: Path, num_nodes: int) -> tuple[np.ndarray, str]:
    """Return ``(adj_dense_float32[N,N], provenance_note)``."""

    z = np.load(path, allow_pickle=False)
    keys = set(z.files)

    try:
        if {"row", "col", "data", "shape"}.issubset(keys):
            row = np.asarray(z["row"], dtype=np.int64)
            col = np.asarray(z["col"], dtype=np.int64)
            data = np.asarray(z["data"], dtype=np.float64)
            shape_arr = np.asarray(z["shape"]).ravel()
            if shape_arr.size != 2:
                raise ValueError(f"bad sparse shape metadata {shape_arr}")
            n_r, n_c = int(shape_arr[0]), int(shape_arr[1])
            if n_r != n_c:
                raise ValueError(f"sparse adjacency must be square, got {(n_r, n_c)}")
            if n_r != num_nodes:
                raise ValueError(f"sparse shape {n_r} does not match data nodes {num_nodes}")
            fmt = _decode_npz_scalar_bytes(z["format"]) if "format" in keys else "coo"
            if fmt and fmt != "coo":
                raise ValueError(
                    f"{path}: only COO-packed sparse npz is supported (got format={fmt!r}); "
                    "convert externally or extend this script."
                )
            dense = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            np.add.at(dense, (row, col), data)
            note = "sparse COO npz (row,col,data,shape) densified"
            return dense.astype(np.float32, copy=False), note

        candidate_keys = [k for k in ("adj", "graph", "adjacency") if k in keys]
        if candidate_keys:
            data_key = candidate_keys[0]
        elif len(keys) == 1:
            data_key = next(iter(keys))
        else:
            raise ValueError(
                f"{path}: expected dense adj key 'adj' or single array, "
                f"or sparse COO keys row/col/data/shape; got keys={sorted(keys)}"
            )

        adj = np.asarray(z[data_key], dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"{path}: expected square 2D adjacency, got {adj.shape}")
        if int(adj.shape[0]) != num_nodes:
            raise ValueError(
                f"{path}: adj size {adj.shape[0]} does not match data node count {num_nodes}"
            )
        note = f"dense npz key={data_key!r}"
        return adj, note
    finally:
        z.close()


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(description="PEMS03.h5 -> data/PEMS03/data.npz + adj.npz")
    p.add_argument(
        "--h5",
        type=Path,
        default=root / "data/raw_data/PEMS03/PEMS03.h5",
        help="Source pandas HDF5 file",
    )
    p.add_argument(
        "--raw-adj",
        type=Path,
        default=root / "data/raw_data/PEMS03/adj.npz",
        help="Adjacency npz next to the h5",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data/PEMS03",
        help="BasicSTFM canonical dataset directory",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    h5_path = args.h5.expanduser().resolve()
    raw_adj = args.raw_adj.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not h5_path.is_file():
        print(f"ERROR: missing H5 file: {h5_path}", file=sys.stderr)
        return 1
    if not raw_adj.is_file():
        print(f"ERROR: missing adjacency: {raw_adj}", file=sys.stderr)
        return 1

    matrix = _read_h5_matrix(h5_path)
    t, n = matrix.shape
    data_tnc = matrix[..., np.newaxis]

    print(f"H5 -> array shape [T,N] = {matrix.shape} -> writing [T,N,1] = {data_tnc.shape}")

    if (t, n) != (EXPECTED_T, EXPECTED_N):
        print(
            f"WARNING: expected (T,N)=({EXPECTED_T},{EXPECTED_N}) for canonical PEMS03; "
            f"got {(t, n)} — proceeding anyway.",
            file=sys.stderr,
        )

    adj_dense, adj_note = _load_dense_adj(raw_adj, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "data.npz"
    out_adj = out_dir / "adj.npz"

    if out_npz.exists() and not args.overwrite:
        print(f"Refusing to overwrite {out_npz} (pass --overwrite)", file=sys.stderr)
        return 1

    np.savez_compressed(out_npz, data=data_tnc)
    print(f"Wrote {out_npz}")
    np.savez_compressed(out_adj, adj=adj_dense)
    print(f"Wrote {out_adj} (dense adj [{adj_dense.shape[0]}, {adj_dense.shape[1]}], from {adj_note})")

    meta = out_dir / "prepare_pems03_meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"source_h5={h5_path}",
                f"source_adj={raw_adj}",
                f"adj_materialization={adj_note}",
                f"data_shape_TNC={list(data_tnc.shape)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {meta}")
    print("Next: python scripts/data/build_basicts_traffic_targets.py --datasets PEMS03")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
