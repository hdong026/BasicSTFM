#!/usr/bin/env python3
"""Offline audit: data stats, StandardScaler, RevIN batch stats, naive baselines, protocol flags.

Run from BasicSTFM repo root ( paths like ``data/METR-LA/data.npz`` resolve relative to cwd )::

    python scripts/analysis/audit_factost_protocol.py \\
        configs/monash/dpm_sr_monash15_then_traffic_sharded_transfer_96_factost_protocol.yaml

Use ``--json-only`` for machine-readable output without the human summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Downstream datasets that should use FactoST-style RevIN when comparing to FactoST benchmarks.
FACTOST_EXPECT_USE_REVIN: Tuple[str, ...] = (
    "METR-LA",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "PEMS-BAY",
    "ETTh2",
    "Weather",
    "Electricity",
)


def _tty_red(text: str, *, stream: Any = None) -> str:
    """ANSI red when writing to a TTY (no markup in JSON)."""

    s = sys.stdout if stream is None else stream
    try:
        if s.isatty():
            return f"\033[91m{text}\033[0m"
    except Exception:
        pass
    return text


def _compute_raw_quality_warnings(label: str, arr: np.ndarray) -> Tuple[Dict[str, Any], List[str]]:
    """Flag sentinel-like mins / absurd scale maxima on the full stored tensor."""

    flat_all = np.asarray(arr.reshape(-1), dtype=np.float64)
    nan_frac = float(np.mean(~np.isfinite(flat_all))) if flat_all.size else 1.0
    finite = flat_all[np.isfinite(flat_all)]
    qw: List[str] = []
    if finite.size == 0:
        qw.append(f"{label}: no finite values in data array")
        stats = {"min": float("nan"), "max": float("nan"), "non_finite_frac": nan_frac}
        return stats, qw

    g_min = float(finite.min())
    g_max = float(finite.max())
    if g_min <= -999:
        qw.append(f"{label}: raw min={g_min} <= -999 (sentinel/outlier risk)")
    if abs(g_max) > 1e6:
        qw.append(f"{label}: raw max={g_max} has abs>1e6 (scale/outlier risk)")
    if nan_frac > 0:
        qw.append(f"{label}: raw npz non-finite fraction={nan_frac:.6f}")
    stats = {"min": g_min, "max": g_max, "non_finite_frac": nan_frac}
    return stats, qw


def _ensure_repo_on_path() -> None:
    r = str(REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def _load_array(path: Path, *, mmap_mode: Optional[str] = None) -> np.ndarray:
    p = path.expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    loaded = np.load(str(p), allow_pickle=False, mmap_mode=mmap_mode)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        arr = loaded["data"].copy()
        loaded.close()
    else:
        arr = loaded
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr.astype(np.float32, copy=False)


def _split_lengths(total: int, split: Sequence[float | int]) -> Tuple[int, int, int]:
    if len(split) != 3:
        raise ValueError("split must have length 3")
    if all(isinstance(x, float) and x <= 1.0 for x in split):
        tr = int(total * split[0])
        va = int(total * split[1])
        te = total - tr - va
        return tr, va, te
    tr, va, te = (int(x) for x in split)
    return tr, va, te


def _masked_mean(err_sq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return err_sq.mean()
    while mask.ndim < err_sq.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=err_sq.dtype, device=err_sq.device).expand_as(err_sq)
    return (err_sq * mask).sum() / mask.sum().clamp_min(1.0)


def _report_dataset(
    *,
    label: str,
    data_path: str,
    input_len: int,
    output_len: int,
    split: Sequence[float | int],
    factost_split: bool,
    scaler_type: str,
    use_revin: bool,
    factost_original_scale: bool,
    stage_results_path: Optional[str],
) -> Dict[str, Any]:
    from basicstfm.data.factost_split import resolve_factost_split_lengths, split_triple_to_split_field
    from basicstfm.data.revin import factost_value_revin_normalize
    from basicstfm.data.scaler import StandardScaler

    path = Path(data_path)
    arr = _load_array(path, mmap_mode="r" if path.suffix.lower() == ".npz" else None)
    t, n, c = arr.shape
    flat = np.asarray(arr[..., 0].reshape(-1), dtype=np.float64)
    global_stats, quality_warnings = _compute_raw_quality_warnings(label, np.asarray(arr))
    report: Dict[str, Any] = {
        "label": label,
        "data_path": str(path),
        "shape_TNC": [t, n, c],
        "raw_value_global_stats": global_stats,
        "quality_warnings": quality_warnings,
        "raw_value_ch0_stats": {
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "std": float(np.nanstd(flat)),
            "p01": float(np.nanquantile(flat, 0.01)),
            "p50": float(np.nanquantile(flat, 0.50)),
            "p99": float(np.nanquantile(flat, 0.99)),
        },
        "protocol": {
            "scaler_type": scaler_type,
            "use_revin": use_revin,
            "factost_original_scale": factost_original_scale,
            "factost_split_requested": factost_split,
        },
    }

    split_spec: Sequence[float | int] = split
    fs_meta: Dict[str, Any] = {}
    if factost_split:
        triple, fs_meta = resolve_factost_split_lengths(str(path), t, split)
        if triple is not None:
            split_spec = split_triple_to_split_field(triple)
    report["factost_split_meta"] = fs_meta
    tr, va, te = _split_lengths(t, split_spec)
    report["split_lengths"] = {"train": tr, "val": va, "test": te}

    train_raw = np.asarray(arr[:tr], dtype=np.float32)
    test_raw = np.asarray(arr[tr + va : tr + va + te], dtype=np.float32)

    scaler: Any
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.fit(train_raw)
        report["standard_scaler"] = {
            "mean_ch0": float(scaler.mean[..., 0].mean()),
            "std_ch0": float(scaler.std[..., 0].mean()),
        }
    else:
        scaler = None
        report["standard_scaler"] = None

    window = input_len + output_len
    if len(test_raw) < window:
        report["note"] = "test split shorter than window; skipping window baselines"
        return report

    # First valid test window (start at 0 within test_raw).
    win = np.asarray(test_raw[:window], dtype=np.float32)
    x_np = win[:input_len]
    y_np = win[input_len:]

    x = torch.from_numpy(x_np)[None, ...]
    y = torch.from_numpy(y_np)[None, ...]

    if scaler_type == "standard" and scaler is not None:
        x_s = scaler.transform(x)
        y_s = scaler.transform(y)
    else:
        x_s = x
        y_s = y

    batch: Dict[str, Any] = {}
    if use_revin:
        x_r, y_r = factost_value_revin_normalize(x_s, y_s, batch, value_channels=0, eps=1e-5, scaled_std_floor=0.05)
        report["revin_batch_after_normalize"] = {
            "x_mean": float(x_r.mean()),
            "x_std": float(x_r.std(unbiased=False)),
            "y_mean": float(y_r.mean()),
            "y_std": float(y_r.std(unbiased=False)),
        }
    else:
        x_r, y_r = x_s, y_s

    last = x_np[-1:, :, :].copy()
    persistence = np.tile(last, (output_len, 1, 1))
    zero = np.zeros_like(y_np)

    def _mae_rmse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        d = torch.from_numpy(np.abs(a - b))
        mae = float(d.mean())
        rmse = float(torch.sqrt((d**2).mean()))
        return mae, rmse

    report["baselines_raw_space_ch_all"] = {
        "persistence": dict(zip(("mae", "rmse"), _mae_rmse(persistence, y_np))),
        "zero": dict(zip(("mae", "rmse"), _mae_rmse(zero, y_np))),
    }

    if scaler_type == "standard" and scaler is not None:
        pers_s = scaler.transform(torch.from_numpy(persistence.astype(np.float32))[None, ...])
        zero_s = scaler.transform(torch.from_numpy(zero.astype(np.float32))[None, ...])
        report["baselines_dataset_scaled_space"] = {
            "persistence": dict(
                zip(("mae", "rmse"), _mae_rmse_tensor_all(pers_s, y_s, None))
            ),
            "zero": dict(zip(("mae", "rmse"), _mae_rmse_tensor_all(zero_s, y_s, None))),
        }

    if stage_results_path:
        p = Path(stage_results_path)
        if p.is_file():
            payload = json.loads(p.read_text(encoding="utf-8"))
            metrics = _extract_latest_metrics(payload, label=data_path)
            report["stage_results_metrics"] = metrics
        else:
            report["stage_results_metrics"] = {"note": f"missing file {p}"}
    else:
        report["stage_results_metrics"] = None

    return report


def _mae_rmse_tensor(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return _masked_mean((pred - tgt).abs(), mask)


def _rmse_rmse_tensor(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_masked_mean((pred - tgt).pow(2), mask).clamp_min(1e-12))


def _mae_rmse_tensor_all(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[float, float]:
    return float(_mae_rmse_tensor(pred, tgt, mask)), float(_rmse_rmse_tensor(pred, tgt, mask))


def _extract_latest_metrics(payload: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return out
    ds_name = Path(label).parent.name
    for st in reversed(stages):
        if not isinstance(st, dict):
            continue
        rd = st.get("resolved_data")
        dp = rd.get("data_path") if isinstance(rd, dict) else None
        if dp is None:
            continue
        if Path(str(dp)).parent.name != ds_name:
            continue
        test_block = st.get("test")
        if not isinstance(test_block, dict):
            continue
        keys = (
            "test/metric/mae",
            "test/metric/rmse",
            "test/metric/mae_norm",
            "test/metric/rmse_norm",
            "test/metric/mae_original",
            "test/metric/rmse_original",
            "test/metric/mae_revin_raw",
            "test/metric/rmse_revin_raw",
        )
        for k in keys:
            if k in test_block:
                out[k] = test_block[k]
        break
    return out


def _primary_mae_mapping(*, use_revin: bool, factost_original_scale: bool) -> str:
    if not use_revin:
        return "metric/mae on dataset scaler outputs (no RevIN path)"
    if factost_original_scale:
        return (
            "metric/mae_original / inverse RevIN + inverse dataset scaler "
            "(FactoST original_scale=1)"
        )
    return (
        "metric/mae_revin_raw / inverse RevIN only, still in dataset-standardized space "
        "(FactoST original_scale=0)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Audit FactoST evaluation protocol per YAML targets.")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument(
        "--stage-results",
        default=None,
        help="Optional stage_results.json to pull logged dual metrics",
    )
    parser.add_argument("--cwd", default=str(REPO_ROOT), help="Working directory for relative data paths")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON payload (skip human-readable summary / warnings preamble)",
    )
    args = parser.parse_args(argv)

    os.chdir(args.cwd)

    from basicstfm.config import load_config
    from basicstfm.engines.stage import StagePlan

    cfg = load_config(args.config)
    audit_rows = StagePlan.describe_factost_protocol_audit(cfg)

    eval_targets: List[Dict[str, Any]] = []
    for row in audit_rows:
        if not row.get("eval_only"):
            continue
        eval_targets.append(row)

    reports: List[Dict[str, Any]] = []
    cfg_fallback = cfg.get("data", {})
    warnings: List[str] = []

    for row in eval_targets:
        dk = row.get("dataset_key")
        if dk in FACTOST_EXPECT_USE_REVIN and not row.get("use_revin"):
            warnings.append(
                f"WARNING: stage={row.get('name')!r} dataset_key={dk!r} should align with FactoST "
                f"benchmarks but task.use_revin is false."
            )

        dp = row.get("data_path")
        if not dp:
            reports.append(
                {
                    "stage_name": row.get("name"),
                    "dataset_key": dk,
                    "error": "missing data_path after merge",
                    "protocol_row": row,
                }
            )
            continue

        il = row.get("input_len") or cfg_fallback.get("input_len")
        ol = row.get("output_len") or cfg_fallback.get("output_len")
        if il is None or ol is None:
            il = cfg["pipeline"]["stages"][0].get("data", {}).get("input_len")
            ol = cfg["pipeline"]["stages"][0].get("data", {}).get("output_len")
        split = tuple(row.get("split") or cfg_fallback.get("split", (0.7, 0.1, 0.2)))

        detail = _report_dataset(
            label=str(row.get("dataset_key")),
            data_path=str(dp),
            input_len=int(il),
            output_len=int(ol),
            split=split,
            factost_split=bool(row.get("factost_split")),
            scaler_type=str(row.get("data_scaler_type") or "standard"),
            use_revin=bool(row.get("use_revin")),
            factost_original_scale=bool(row.get("factost_original_scale")),
            stage_results_path=args.stage_results,
        )
        for qw in detail.get("quality_warnings") or []:
            warnings.append(qw)
        reports.append(
            {
                "stage_name": row.get("name"),
                "dataset_key": dk,
                "input_len": il,
                "output_len": ol,
                "split": list(split),
                "test_metric_mae_maps_to": _primary_mae_mapping(
                    use_revin=bool(row.get("use_revin")),
                    factost_original_scale=bool(row.get("factost_original_scale")),
                ),
                **detail,
            }
        )

    payload = {"targets": reports, "warnings": warnings}

    if not args.json_only:
        print("=== FactoST protocol audit (eval_only stages) ===")
        print(f"config: {args.config}")
        if not warnings:
            print("(no protocol or raw-quality warnings)")
        else:
            for w in warnings:
                print(_tty_red(w))
        for item in reports:
            if "error" in item:
                print(f"- {item.get('stage_name')}: ERROR {item['error']}")
                continue
            prot = item.get("protocol", {})
            print(
                f"- {item.get('stage_name')} | dataset_key={item.get('dataset_key')} | "
                f"scaler={prot.get('scaler_type')} use_revin={prot.get('use_revin')} "
                f"factost_original_scale={prot.get('factost_original_scale')}"
            )
            print(f"    test/metric/mae → {item.get('test_metric_mae_maps_to')}")
        print()

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
