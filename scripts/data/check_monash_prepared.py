#!/usr/bin/env python3
"""Sanity-check prepared Monash folders (values.npy / lengths.npy / meta.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_subset(root: Path) -> Dict[str, Any]:
    vals = np.load(root / "values.npy", mmap_mode="r")
    lens = np.load(root / "lengths.npy")
    meta = {}
    mp = root / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    fin = v[np.isfinite(v)]
    abs_max = float(np.max(np.abs(fin))) if fin.size else float("nan")
    out: Dict[str, Any] = {
        "path": str(root.resolve()),
        "n_series": int(lens.size),
        "total_points": int(vals.size),
        "finite_ratio": float(fin.size / max(v.size, 1)),
        "abs_max": abs_max,
        "min": float(fin.min()) if fin.size else float("nan"),
        "max": float(fin.max()) if fin.size else float("nan"),
        "mean": float(fin.mean()) if fin.size else float("nan"),
        "std": float(fin.std()) if fin.size else float("nan"),
        "p01": float(np.percentile(fin, 1)) if fin.size else float("nan"),
        "p50": float(np.percentile(fin, 50)) if fin.size else float("nan"),
        "p99": float(np.percentile(fin, 99)) if fin.size else float("nan"),
        "has_abs_gt_1e9": bool(abs_max > 1e9) if np.isfinite(abs_max) else False,
        "likely_ok_for_training": bool(
            fin.size > 0 and abs_max <= 1e9 and fin.size == vals.size
        ),
        "meta_json": meta,
    }
    return out


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Prepared dataset dir(s) or a parent containing subdirs with lengths.npy",
    )
    args = p.parse_args(argv)

    roots: List[Path] = []
    for r in args.roots:
        r = Path(r)
        if (r / "lengths.npy").is_file():
            roots.append(r)
        elif r.is_dir():
            for sub in sorted(p for p in r.iterdir() if p.is_dir() and (p / "lengths.npy").is_file()):
                roots.append(sub)
        else:
            print(f"skip missing: {r}", file=sys.stderr)

    if not roots:
        print("no prepared datasets found", file=sys.stderr)
        sys.exit(2)

    for root in roots:
        row = _load_subset(root)
        name = root.name
        print(
            f"{name}\tseries={row['n_series']}\tpoints={row['total_points']}\t"
            f"finite%={100 * row['finite_ratio']:.2f}\tabs_max={row['abs_max']:.6g}\t"
            f"p01/p50/p99={row['p01']:.6g}/{row['p50']:.6g}/{row['p99']:.6g}\t"
            f"ok={row['likely_ok_for_training']}"
        )
        if row["has_abs_gt_1e9"]:
            print(f"  WARNING: |max|>1e9 — avoid for pretraining without further scaling.", file=sys.stderr)


if __name__ == "__main__":
    main()
