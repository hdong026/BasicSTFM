#!/usr/bin/env python3
r"""
Build small LargeST *shards* for cross-domain pretraining (from ``data/LargeST`` only, not ``LargeST_full``).

Each shard:
  * connected-ish subgraph of ``target_nodes`` (default 256) via the same graph_greedy BFS/seed
    policy as :func:`basicstfm.data.datamodule._partition_node_ids`;
  * a single contiguous *time* slice, length ``min(max_timesteps, T)``;
  * saved as ``data/{out_name}/data.npz`` and ``adj.npz`` (keys ``data`` / ``adj``).

See ``--num-shards`` to change the number of output shards (default 2 → part0, part1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo imports (so we reuse the same graph partition policy as the trainer)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from basicstfm.data.datamodule import _load_numpy, _partition_node_ids  # noqa: E402


def _load_data_mmap(data_path: Path) -> np.ndarray:
    """[T, N, C] float32 array; use mmap for large .npz/.npy when possible."""
    p = str(data_path)
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=False, mmap_mode="r")
        if "data" not in z.files:
            z.close()
            raise KeyError(f"expected key 'data' in {p}, got {z.files}")
        return z["data"]
    return np.load(p, allow_pickle=False, mmap_mode="r")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        type=Path,
        default=_REPO,
        help="BasicSTFM repo root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--largest-dir",
        type=Path,
        default=Path("data/LargeST"),
        help="Relative to repo; must contain data.npz and adj.npz",
    )
    ap.add_argument("--num-shards", type=int, default=2, help="Number of output shards (e.g. 2 or 4)")
    ap.add_argument("--target-nodes", type=int, default=256, help="Nodes per partition (256–384 recommended)")
    ap.add_argument("--max-timesteps", type=int, default=32768, help="Max length of the time window per shard")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Partition RNG seed (graph_greedy); use the same for reproducibility",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="LargeST_xd_part",
        help="Output directory name prefix under data/ (e.g. LargeST_xd_part0)",
    )
    a = ap.parse_args()
    repo: Path = a.repo.resolve()
    base = (repo / a.largest_dir).resolve()
    data_path = base / "data.npz"
    adj_path = base / "adj.npz"
    if not data_path.is_file() or not adj_path.is_file():
        raise SystemExit(f"Need {data_path} and {adj_path}")

    print("Loading adj (full graph)...", flush=True)
    adj_full = _load_numpy(str(adj_path), "adj", mmap_mode=None)
    if adj_full.ndim != 2 or adj_full.shape[0] != adj_full.shape[1]:
        raise SystemExit(f"bad adj shape {adj_full.shape}")
    n_total = int(adj_full.shape[0])

    print("Memory-mapping data array...", flush=True)
    data_mm = _load_data_mmap(data_path)
    if data_mm.ndim != 3:
        raise SystemExit(f"expected [T,N,C] data, got {data_mm.shape}")
    t_all, n_data, c = data_mm.shape
    if n_data != n_total:
        raise SystemExit(f"data N={n_data} != adj N={n_total}")
    t_big = int(t_all)

    # Same local expansion policy as `partition_strategy: graph_greedy` in configs.
    # (METIS can be added later with an optional extra dependency; not required for correctness.)
    method_used = f"graph_greedy (target_nodes={a.target_nodes}, seed={a.seed}, max_partitions={a.num_shards})"
    partitions: list[np.ndarray] = _partition_node_ids(
        n_total,
        int(a.target_nodes),
        graph=np.asarray(adj_full, dtype=np.float32),
        strategy="graph_greedy",
        max_partitions=int(a.num_shards),
        seed=int(a.seed),
    )

    if len(partitions) < a.num_shards:
        raise SystemExit(
            f"only got {len(partitions)} partition(s); try lowering target_nodes or use smaller num_shards"
        )
    partitions = partitions[: int(a.num_shards)]
    t_span = int(min(a.max_timesteps, t_big))
    t_starts: list[tuple[int, int]] = []
    for k in range(len(partitions)):
        if k == 0:
            t0, t1 = 0, t_span
        elif k == 1 and len(partitions) == 2:
            t0, t1 = t_big - t_span, t_big
        else:
            # spread windows across the timeline
            if len(partitions) <= 1:
                t0, t1 = 0, t_span
            else:
                step = max(1, (t_big - t_span) // max(1, len(partitions) - 1))
                t0 = int(min(k * step, t_big - t_span))
                t1 = t0 + t_span
        t_starts.append((t0, t1))
        print(f"shard {k} time range [{t0}, {t1}) len={t1 - t0}", flush=True)

    out_meta: list[dict] = []
    for k, (nodes, (t0, t1)) in enumerate(zip(partitions, t_starts)):
        name = f"{a.out_prefix}{k}"
        out_dir = repo / "data" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        nodes = np.asarray(nodes, dtype=np.int64)
        n_h = int(nodes.shape[0])
        print(f"Building {name}: N'={n_h} nodes, T'={t1 - t0}", flush=True)
        # slice time + node subset (materialize float32)
        sub = np.asarray(data_mm[t0:t1, nodes, :], dtype=np.float32)
        sub_adj = np.asarray(adj_full[np.ix_(nodes, nodes)], dtype=np.float32)
        out_data = out_dir / "data.npz"
        out_adj = out_dir / "adj.npz"
        np.savez_compressed(out_data, data=sub)
        np.savez_compressed(out_adj, adj=sub_adj)
        meta = {
            "name": name,
            "source": str(base),
            "original_shape": [t_big, n_total, c],
            "partition_method": method_used,
            "shard_T": t1 - t0,
            "shard_N": n_h,
            "shard_C": c,
            "t_range": [t0, t1],
            "node_indices_min": int(nodes.min()) if n_h else 0,
            "node_indices_max": int(nodes.max()) if n_h else 0,
        }
        (out_dir / "shard_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        readme = f"""# {name} (LargeST cross-domain shard)

## Source
- **repo path**: `data/LargeST/` (not `LargeST_full`)
- **Original** `data` shape `[T, N, C]`: `{[t_big, n_total, c]}`
- **Adj** original: `(N, N) = ({n_total}, {n_total})`

## Shard
- **Partition**: {method_used}
- **This shard**:
  - `N\\_shard` = **{n_h}**
  - `T\\_shard` = **{t1 - t0}** (contiguous from timestep **{t0}** to **{t1}** exclusive)
  - `C` = **{c}**
- **Files**:
  - `data.npz` key `data`
  - `adj.npz` key `adj`, shape `({n_h}, {n_h})`

## Rebuild
Produced with `scripts/data/build_largest_cross_domain_shards.py` (see repo).
"""
        (out_dir / "README.md").write_text(readme, encoding="utf-8")
        out_meta.append(meta)
        del sub, sub_adj

    print("Done. Shards:", ", ".join(f"data/{a.out_prefix}{i}" for i in range(len(partitions))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
