#!/usr/bin/env python3
"""Build KnowAir / PM2.5-GNN-style adjacency and BasicSTFM data.npz + adj.npz.

Spatial graph: Haversine kNN (default k=8), Gaussian kernel on distances or binary kNN,
symmetrized with max, diagonal set to 1.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

NodeLevel = Literal["auto", "city", "station"]


@dataclass(frozen=True)
class NodeMeta:
    ids: list[str]
    names: list[str]
    lon: np.ndarray  # [N]
    lat: np.ndarray  # [N]
    source_file: str


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Pairwise haversine; supports broadcasting when one side is 1D."""
    r = 6371.0
    p1 = math.pi / 180.0
    d_lat = (lat2 - lat1) * p1
    d_lon = (lon2 - lon1) * p1
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1 * p1) * np.cos(lat2 * p1) * np.sin(d_lon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return r * c


def pairwise_haversine_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Full [N, N] distance matrix in km."""
    lon = np.asarray(lon, dtype=np.float64).reshape(-1, 1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1, 1)
    return _haversine_km(lon, lat, lon.T, lat.T)


_CITY_LINE_RE = re.compile(r"^(\d+)\s+(.+?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def parse_city_file(path: Path) -> NodeMeta:
    ids: list[str] = []
    names: list[str] = []
    lons: list[float] = []
    lats: list[float] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = _CITY_LINE_RE.match(line)
        if not m:
            raise ValueError(f"{path}: line {line_no} does not match 'index name lon lat': {raw!r}")
        idx, name, lon_s, lat_s = m.groups()
        ids.append(idx)
        names.append(name.strip())
        lons.append(float(lon_s))
        lats.append(float(lat_s))
    return NodeMeta(
        ids=ids,
        names=names,
        lon=np.asarray(lons, dtype=np.float64),
        lat=np.asarray(lats, dtype=np.float64),
        source_file=str(path.resolve()),
    )


def parse_station_csv(path: Path) -> NodeMeta:
    ids: list[str] = []
    names: list[str] = []
    lons: list[float] = []
    lats: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        lower = {h.lower(): h for h in reader.fieldnames}
        for req in ("lon", "lat"):
            if req not in lower:
                raise ValueError(f"{path}: need a '{req}' column, got {reader.fieldnames}")
        id_key = lower.get("id") or lower.get("station_id")
        name_key = lower.get("stations_name") or lower.get("name") or lower.get("station_name")
        if id_key is None:
            raise ValueError(f"{path}: need ID column (id / station_id), got {reader.fieldnames}")
        if name_key is None:
            name_key = id_key
        for row in reader:
            ids.append(str(row[id_key]).strip())
            names.append(str(row[name_key]).strip())
            lons.append(float(row[lower["lon"]]))
            lats.append(float(row[lower["lat"]]))
    return NodeMeta(
        ids=ids,
        names=names,
        lon=np.asarray(lons, dtype=np.float64),
        lat=np.asarray(lats, dtype=np.float64),
        source_file=str(path.resolve()),
    )


def load_and_canonicalize_npy(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return array [T, N, C] and original shape."""
    arr = np.load(path, mmap_mode="r")
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path}: expected a single-array .npy file, got {type(arr)}")
    orig_shape = tuple(int(x) for x in arr.shape)
    x = np.asarray(arr)
    if x.ndim == 2:
        raise ValueError(
            f"{path}: 2D array {orig_shape} is ambiguous (expected 3D [T,N,C]). "
            "Reshape manually or extend this script with --reshape."
        )
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    if x.ndim != 3:
        raise ValueError(f"{path}: need 3D [T,N,C] after cleanup, got shape {x.shape}")
    return x.astype(np.float32), orig_shape


def infer_node_level(n: int) -> Literal["city", "station"]:
    d_city = min(abs(n - 184), abs(n - 196))
    d_station = abs(n - 1498)
    return "city" if d_city <= d_station else "station"


def resolve_metadata(
    n: int,
    node_level: NodeLevel,
    metadata_dir: Path,
    city_file: Path | None,
    station_file: Path | None,
) -> NodeMeta:
    city_txt = metadata_dir / "city.txt"
    city_196 = metadata_dir / "city_196.txt"
    station_csv = metadata_dir / "station_info.csv"

    if city_file is not None:
        meta = parse_city_file(city_file)
        if len(meta.ids) != n:
            raise SystemExit(
                f"节点数 N={n} 与 --city-meta 行数 {len(meta.ids)} 不一致。\n"
                f"  文件: {meta.source_file}\n"
                f"  提示: KnowAir 官方数组通常为 184 城，对应 PM2.5-GNN 的 data/city.txt；"
                f"city_196.txt 为 196 城。"
            )
        return meta

    if station_file is not None:
        meta = parse_station_csv(station_file)
        if len(meta.ids) != n:
            raise SystemExit(
                f"节点数 N={n} 与 --station-meta 行数 {len(meta.ids)} 不一致。\n"
                f"  文件: {meta.source_file}"
            )
        return meta

    level = infer_node_level(n) if node_level == "auto" else node_level

    if level == "station":
        meta = parse_station_csv(station_csv)
        if len(meta.ids) != n:
            raise SystemExit(_diagnose_mismatch(n, metadata_dir, city_txt, city_196, station_csv))
        return meta

    # city
    meta184 = parse_city_file(city_txt)
    meta196 = parse_city_file(city_196)
    if n == len(meta184.ids):
        return meta184
    if n == len(meta196.ids):
        return meta196
    raise SystemExit(_diagnose_mismatch(n, metadata_dir, city_txt, city_196, station_csv))


def _diagnose_mismatch(
    n: int,
    metadata_dir: Path,
    city_txt: Path,
    city_196: Path,
    station_csv: Path,
) -> str:
    meta184 = parse_city_file(city_txt)
    meta196 = parse_city_file(city_196)
    meta_s = parse_station_csv(station_csv)
    lines = [
        f"无法为 N={n} 自动匹配元数据（节点数与元数据行数不一致）。",
        "",
        "候选文件行数:",
        f"  city.txt          -> {len(meta184.ids)}  (PM2.5-GNN 图构建实际使用的城市列表)",
        f"  city_196.txt      -> {len(meta196.ids)}",
        f"  station_info.csv  -> {len(meta_s.ids)}  (站点)",
        "",
        f"metadata 目录: {metadata_dir.resolve()}",
        "",
        "请使用 --node-level / --city-meta / --station-meta 显式指定，或检查 .npy 是否对应其它预处理版本。",
    ]
    return "\n".join(lines)


def knn_graph_from_dist(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed kNN mask [N,N] and distances on those arcs (0 elsewhere)."""
    n = dist.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    k_eff = min(k, n - 1)
    np.fill_diagonal(dist, math.inf)
    idx = np.argsort(dist, axis=1)[:, :k_eff]
    mask = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n, dtype=np.int64)[:, None]
    mask[rows, idx] = 1.0
    d_sel = np.zeros((n, n), dtype=np.float64)
    d_sel[rows, idx] = dist[rows, idx]
    return mask, d_sel


def build_adjacency(
    dist_full: np.ndarray,
    k: int,
    weighted: bool,
    sigma: float | Literal["auto"],
) -> tuple[np.ndarray, float | None]:
    """Return dense adj [N,N] and sigma used（仅高斯加权时有效）。"""
    dist_work = dist_full.astype(np.float64).copy()
    mask, d_sel = knn_graph_from_dist(dist_work, k)
    edges = d_sel[mask > 0]
    if edges.size == 0:
        raise RuntimeError("kNN 未产生任何边（检查 k 与 N）。")

    sigma_used: float | None = None
    if weighted:
        if sigma == "auto":
            sig = float(np.median(edges))
            if sig <= 0:
                raise RuntimeError(f"sigma 中位数无效: {sig}")
        else:
            sig = float(sigma)
            if sig <= 0:
                raise ValueError("sigma must be positive")
        sigma_used = sig
        w = np.zeros_like(d_sel)
        nz = d_sel > 0
        w[nz] = np.exp(-(d_sel[nz] ** 2) / (sig**2))
        adj = w
    else:
        adj = mask.copy()

    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32), sigma_used


def adj_stats(adj: np.ndarray) -> dict[str, float]:
    n = adj.shape[0]
    off = adj.copy()
    np.fill_diagonal(off, 0.0)
    nnz = int(np.count_nonzero(off))
    denom = n * (n - 1)
    sparsity_off = 1.0 - (nnz / max(denom, 1))
    degree = off.astype(np.float64).sum(axis=1)
    return {
        "nnz_offdiag": float(nnz),
        "density_offdiag": float(nnz) / float(max(denom, 1)),
        "sparsity_offdiag": float(sparsity_off),
        "mean_degree": float(degree.mean()),
    }


def str2bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KnowAir / PM2.5-GNN -> BasicSTFM data.npz + adj.npz")
    p.add_argument("--input-npy", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--node-level", choices=("auto", "city", "station"), default="auto")
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--weighted",
        type=str2bool,
        default=True,
        help="true: Gaussian kernel on kNN distances; false: binary kNN (0/1).",
    )
    p.add_argument(
        "--sigma",
        default="auto",
        help="'auto' -> median kNN edge distance; else positive float (Gaussian only).",
    )
    p.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Directory containing city.txt, city_196.txt, station_info.csv (PM2.5-GNN data/).",
    )
    p.add_argument("--city-meta", type=Path, default=None, help="Override city metadata file.")
    p.add_argument("--station-meta", type=Path, default=None, help="Override station_info.csv path.")
    p.add_argument(
        "--build-binary-knn",
        action="store_true",
        help="Alias: same as --weighted false (binary kNN).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.build_binary_knn:
        args.weighted = False

    sigma_arg: float | Literal["auto"]
    if str(args.sigma).lower() == "auto":
        sigma_arg = "auto"
    else:
        sigma_arg = float(args.sigma)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = Path(__file__).resolve().parents[2] / "data/raw_data/KnowAir/pm25_gnn_metadata"

    data, orig_shape = load_and_canonicalize_npy(args.input_npy)
    t, n, c = data.shape

    meta = resolve_metadata(n, args.node_level, metadata_dir, args.city_meta, args.station_meta)
    dist = pairwise_haversine_km(meta.lon, meta.lat)
    adj, sigma_used = build_adjacency(dist, args.k, args.weighted, sigma_arg)
    stats = adj_stats(adj)

    np.savez_compressed(out_dir / "data.npz", data=data)
    np.savez_compressed(out_dir / "adj.npz", adj=adj)

    level_guess = infer_node_level(n)
    readme = [
        "# KnowAir / BasicSTFM 数据说明",
        "",
        "## 原始与处理后数组",
        f"- 原始 `.npy` shape: `{orig_shape}`",
        f"- 处理后 `data.npz['data']` shape: `{tuple(data.shape)}` （期望 `[T, N, C]`）",
        "",
        "## 元数据",
        f"- 节点粒度判定（auto 启发式）: **{level_guess}**（N={n} 更接近城市 184/196 或站点 1498）",
        f"- 实际使用的元数据文件: `{meta.source_file}`",
        f"- CLI `--node-level`: `{args.node_level}`",
        "",
        "## 图构造",
        f"- kNN k = {args.k}",
        f"- 距离: Haversine（km）",
        f"- 加权: **{'高斯核 exp(-d^2/sigma^2)' if args.weighted else '二值 kNN（0/1）'}**",
        f"- sigma: **{sigma_used if sigma_used is not None else 'N/A（二值图不使用）'}** "
        f"（km；加权图下 auto 为有向 kNN 边上的距离中位数）",
        "- 对称化: `A = max(A, A.T)`",
        "- 对角线: 置为 1",
        "",
        "## 邻接矩阵统计（不含对角线）",
        f"- `adj` shape: `{tuple(adj.shape)}`",
        f"- 非零率（off-diagonal density）: {stats['density_offdiag']:.6f}",
        f"- 稀疏度（off-diagonal sparsity）: {stats['sparsity_offdiag']:.6f}",
        f"- 平均度（off-diagonal 权重和/节点）: {stats['mean_degree']:.6f}",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("=== KnowAir / BasicSTFM 导出完成 ===")
    print(f"原始 shape: {orig_shape} -> 处理后 [T,N,C]: {tuple(data.shape)}")
    print(f"节点粒度: {level_guess}；元数据: {meta.source_file}")
    print(f"adj shape: {tuple(adj.shape)}")
    print(
        f"off-diagonal 非零率: {stats['density_offdiag']:.6f}；"
        f"平均度: {stats['mean_degree']:.6f}"
    )
    print(f"输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
