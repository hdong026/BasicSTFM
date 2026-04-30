#!/usr/bin/env python3
"""
Read prepared dataset tensors (data/*.npz or data.npy) and report [T, N, C] scale,
valid sliding windows for a forecasting config, and rough volume = windows * N * C.

Large arrays are not fully loaded: only small .npy headers are parsed (strips null padding
before ast.literal_eval for NumPy2 compatibility on huge zip members).
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DataReport:
    name: str
    root: Path
    data_path: str | None = None
    data_key: str | None = None
    shape: tuple[int, ...] | None = None
    adj_path: str | None = None
    adj_key: str | None = None
    adj_shape: tuple[int, ...] | None = None
    notes: list[str] = field(default_factory=list)
    readme_excerpt: str | None = None


def _parse_npy_header_block(block: bytes) -> dict[str, Any]:
    """Block may contain trailing \\0 padding; numpy sometimes embeds nulls in header."""
    text = block.split(b"\x00", 1)[0].decode("latin-1", errors="strict")
    text = text.strip()
    d = ast.literal_eval(text)
    if not isinstance(d, dict) or "shape" not in d:
        raise ValueError("not a valid numpy array header dict")
    return d


def read_npy_header_from_fileobj(f) -> dict[str, Any]:
    """File position 0. Returns header dict (includes shape, descr, fortran_order)."""
    m = f.read(6)
    if m != b"\x93NUMPY":
        raise ValueError("not a .npy (bad magic)")
    v_major, v_minor = f.read(1)[0], f.read(1)[0]
    if (v_major, v_minor) == (1, 0):
        hlen = struct.unpack("<H", f.read(2))[0]
    else:
        hlen = struct.unpack("<I", f.read(4))[0]
    block = f.read(hlen)
    return _parse_npy_header_block(block)


def tuple_shape(d: dict[str, Any]) -> tuple[int, ...]:
    s = d["shape"]
    if s is None:
        return tuple()
    if np.isscalar(s) or (isinstance(s, (int, float)) and not isinstance(s, bool)):
        return (int(s),)
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    raise TypeError("bad shape in header")


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        d = read_npy_header_from_fileobj(f)
    return tuple_shape(d)


def read_npz_array_key_shape(
    path: Path, preferred_keys: list[str] | None = None
) -> tuple[str, tuple[int, ...]]:
    """Load first array matching preferred name stem (e.g. data) or fall back to largest .npy."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".npy")]
        if not members:
            raise ValueError("no .npy in .npz")
        want = [f"{k}.npy" for k in (preferred_keys or ("data",))]
        chosen: str | None = None
        for w in want:
            if w in members:
                chosen = w
                break
        if chosen is None:
            chosen = max(members, key=lambda m: zf.getinfo(m).file_size)
        with zf.open(chosen) as f:
            d = read_npy_header_from_fileobj(f)
        key = chosen[:-4]
        return key, tuple_shape(d)


def load_readme_excerpt(path: Path, max_chars: int = 4000) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n\n[… truncated …]\n"
    return text


def analyze_dir(repo: Path, rel: str) -> DataReport:
    root = repo / "data" / rel
    r = DataReport(name=rel, root=root)
    if not root.is_dir():
        r.notes.append("MISSING directory")
        return r
    r.readme_excerpt = load_readme_excerpt(root / "README.md")
    if r.readme_excerpt is None and (root / "metadata.json").is_file():
        try:
            meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            r.readme_excerpt = "metadata.json:\n" + json.dumps(meta, indent=2, ensure_ascii=False)
        except OSError:
            pass

    data_npy = root / "data.npy"
    data_npz = root / "data.npz"
    if data_npy.is_file():
        r.data_path = str(data_npy.relative_to(repo))
        try:
            r.shape = read_npy_shape(data_npy)
        except Exception as e:
            r.notes.append(f"data.npy: {e}")
    elif data_npz.is_file():
        r.data_path = str(data_npz.relative_to(repo))
        try:
            k, shp = read_npz_array_key_shape(data_npz, ["data", "X", "arr_0"])
            r.data_key = k
            r.shape = shp
        except Exception as e:
            r.notes.append(f"data.npz: {e}")
    else:
        r.notes.append("no data.npy or data.npz")

    for adj_name in ("adj.npy", "adj.npz"):
        p = root / adj_name
        if not p.is_file():
            continue
        r.adj_path = str(p.relative_to(repo))
        try:
            if p.suffix == ".npy":
                r.adj_key = "adj" if p.stem == "adj" else p.stem
                r.adj_shape = read_npy_shape(p)
            else:
                k, shp = read_npz_array_key_shape(p, ["adj", "A", "graph", "data"])
                r.adj_key = k
                r.adj_shape = shp
        except Exception as e:
            r.notes.append(f"{adj_name}: {e}")
        break
    return r


def _impl(a: argparse.Namespace) -> int:
    repo: Path = a.repo.resolve()
    out_path: Path = a.out or (repo / "reports" / "cross_domain_scale_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dirs = [
        "LargeST",
        "LargeST_full",
        "KnowAir",
        "ETTm1",
        "ETTm1_small",
        "ETTm2",
        "ETTm2_small",
        "Weather",
    ]
    reports = [analyze_dir(repo, d) for d in dirs]
    in_len, out_len = a.input_len, a.output_len

    def windows_for_t(t: int) -> int:
        if t < in_len + out_len - 1:
            return 0
        return t - in_len - out_len + 1

    def soft_weights(
        vmap: dict[str, float], keys: list[str], temperature: float
    ) -> dict[str, float]:
        wts = {k: 1.0 / (max(vmap[k], 1.0) ** temperature) for k in keys}
        s = sum(wts.values())
        return {k: wts[k] / s for k in keys}

    lines: list[str] = []
    lines.append("# Cross-domain dataset scale (BasicSTFM)\n\n")
    lines.append(
        f"Config: `input_len={in_len}`, `output_len={out_len}` → "
        f"`windows = T - {in_len} - {out_len} + 1`.\n\n"
    )
    lines.append("## Table 1 — measured scale per directory\n\n")
    vols: dict[str, float] = {}
    wdict: dict[str, int] = {}

    headers = [
        "Dataset",
        "data file",
        "key",
        "T, N, C",
        "adj file",
        "adj key",
        "adj shape",
        "windows",
        "volume = w×N×C",
    ]
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|\n")

    for r in reports:
        wv = 0
        vol = 0.0
        tnc = "—"
        if r.shape and len(r.shape) == 3:
            t, n, c = r.shape[0], r.shape[1], r.shape[2]
            tnc = f"({t}, {n}, {c})"
            wv = windows_for_t(t)
            vol = float(wv) * n * c
            vols[r.name] = vol
        wdict[r.name] = wv
        dpath = f"`{r.data_path}`" if r.data_path else "—"
        dkey = f"`{r.data_key}`" if r.data_key is not None else "—"
        ap = f"`{r.adj_path}`" if r.adj_path else "—"
        ak = f"`{r.adj_key}`" if r.adj_key else "—"
        aj = f"`{r.adj_shape}`" if r.adj_shape else "—"
        vols_str = f"{vol:.6e}" if vol > 0 else "—"
        lines.append(
            f"| {r.name} | {dpath} | {dkey} | {tnc} | {ap} | {ak} | {aj} | {wv} | {vols_str} |\n"
        )

    lines.append("\n## README snippets (if any)\n\n")
    for r in reports:
        for n in r.notes:
            lines.append(f"- **{r.name}**: {n}\n")
        if r.readme_excerpt:
            lines.append(f"### {r.name}\n\n")
            lines.append(
                "<pre>\n" + html.escape(r.readme_excerpt.rstrip()) + "\n</pre>\n\n"
            )

    caps = {
        "LargeST": 8_192,
        "LargeST_full": 8_192,
        "KnowAir": 4_096,
        "ETTm1": 16_000,
        "ETTm2": 16_000,
        "ETTm1_small": 2_000,
        "ETTm2_small": 2_000,
        "Weather": 12_000,
    }

    def v_eff_map(keys: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in keys:
            sh = next((x.shape for x in reports if x.name == k), None)
            if not sh or len(sh) != 3:
                out[k] = 1.0
                continue
            t, n, c = sh
            wmax = windows_for_t(t)
            wcap = min(caps.get(k, wmax), wmax)
            out[k] = float(wcap) * n * c
        return out

    ve_a = v_eff_map(["LargeST", "KnowAir", "ETTm1", "Weather"])
    wts_a = soft_weights(ve_a, list(ve_a), 0.35)
    ve_b = v_eff_map(["LargeST_full", "KnowAir", "ETTm1", "Weather"])
    wts_b = soft_weights(ve_b, list(ve_b), 0.35)

    def fmt_w(w: dict[str, float]) -> str:
        return " ".join(f"`{k}`={w[k]:.3f}" for k in sorted(w))

    lines.append("## Table 2 — joint pretrain mixing (heuristic)\n\n")
    lines.append(
        "In this tree, `LargeST` and `LargeST_full` have the same `(T, N, C)`; `LargeST` is a `.npz` "
        "containing a single giant `.npy`, while `LargeST_full` is a standalone `.npy` for mmap. "
        "Uncapped, traffic **dominates** `volume` vs air / ETT / weather by orders of magnitude — **cap** traffic windows per epoch when mixing.\n\n"
    )
    lines.append("### A — prefer `data/LargeST/`\n\n")
    lines.append("Suggested **max windows / domain / epoch** (tune in your sampler / `steps_per_cap`):\n\n")
    lines.append("| Domain | cap |\n| --- | ---: |\n")
    for k in [
        "LargeST",
        "LargeST_full",
        "KnowAir",
        "ETTm1",
        "ETTm2",
        "Weather",
        "ETTm1_small",
        "ETTm2_small",
    ]:
        if k in caps:
            lines.append(f"| {k} | {caps[k]} |\n")
    lines.append(
        f"\nSampling weights (from **capped** effective `volume`, pow index 0.35): {fmt_w(wts_a)}.\n"
    )
    lines.append(
        f"Effective volumes used: {', '.join(f'{k}={ve_a[k]:.4e}' for k in ve_a)}.\n"
    )
    lines.append(
        "\n- **ETTm `*_small`**: use for smoke tests / fast debugging; for real pretrain prefer full `ETTm1` or `ETTm2` (more `T`, same `N`).\n\n"
    )
    lines.append("### B — `data/LargeST_full/`\n\n")
    lines.append("Same scale as `LargeST` here; I/O may differ. Use **identical** caps. Weights: ")
    lines.append(f"{fmt_w(wts_b)}.\n")
    lines.append(
        f"Effective volumes: {', '.join(f'{k}={ve_b[k]:.4e}' for k in ve_b)}.\n\n"
    )

    # Dominance: uncapped, traffic vs the three others
    tr = vols.get("LargeST", 0.0)
    oth = sum(vols.get(n, 0) for n in ("KnowAir", "ETTm1", "Weather"))
    if tr + oth > 0:
        lines.append(
            f"**Uncapped volume share** (LargeST vs KnowAir+ETTm1+Weather only): "
            f"**{100 * tr / (tr + oth):.1f}%** in LargeST — mixing **without** caps lets traffic govern gradients.\n\n"
        )
    t_ratio = 0.0
    if oth + tr > 0:
        t_ratio = tr / (tr + oth)
    lines.append("## Synthesis\n\n")
    lines.append(
        f"1. **First run**: joint pretrain on **`KnowAir` + `Weather` + one ETT (`ETTm1` or `ETTm2`)**; add **`LargeST` (or `LargeST_full`) only with 4k–8k max traffic windows/epoch** or you re-enter a ~**{t_ratio*100 if t_ratio else 0:.0f}%**-traffic regime by volume.\n"
    )
    lines.append(
        "2. **`LargeST_full`**: not smaller than `LargeST` in this report — skip only if your pipeline prefers `.npz` zip I/O; otherwise use the **same** caps either way. Do **not** use `full` to “fix” domain imbalance without subsampling.\n"
    )
    lines.append(
        "3. **Slowest** forward/backward: **traffic (N=8600)**. Smallest-`T` ETTm `small` is fast per epoch but **thin** for representation learning.\n"
    )

    text = "".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n[Saved] {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--input-len", type=int, default=96)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--out", type=Path, default=None)
    return _impl(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
