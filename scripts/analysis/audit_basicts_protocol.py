#!/usr/bin/env python3
"""Offline audit for BasicTS-style traffic protocol (StandardScaler + no RevIN).

Example::

    python scripts/analysis/audit_basicts_protocol.py \\
        configs/monash/dpm_sr_monash15_then_traffic_sharded_transfer_12_basicts_protocol.yaml \\
        --dataset PEMS04 \\
        --json-only

Persistence repeats the **last input timestep** for **every** horizon
``pred[h] = x[input_len - 1]`` for ``h = 0 .. output_len - 1``.
Metrics aggregate **all** valid **test-split** sliding windows (full test).

Trainer mapping (denormalized supervision, no RevIN):
``test/metric/mae`` equals ``metric/mae_original_after_inverse_standard_scaler``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    r = str(REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def _load_array(path: Path) -> np.ndarray:
    p = path.expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    loaded = np.load(str(p), allow_pickle=False)
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


def _horizon_metrics_aggregate(
    pred: np.ndarray,
    tgt: np.ndarray,
    *,
    sum_abs: np.ndarray,
    sum_sq: np.ndarray,
) -> None:
    """Accumulate sums for MAE/RMSE per horizon (pred,tgt shape [H,N,C])."""

    assert pred.shape == tgt.shape and pred.ndim == 3
    h_len, n, c = pred.shape
    err = pred - tgt
    abs_err = np.abs(err).reshape(h_len, -1).sum(axis=1)
    sq_err = (err ** 2).reshape(h_len, -1).sum(axis=1)
    sum_abs[:] += abs_err
    sum_sq[:] += sq_err


def _finalize_horizon_metrics(sum_abs: np.ndarray, sum_sq: np.ndarray, *, denom: float) -> Tuple[np.ndarray, np.ndarray]:
    mae_h = sum_abs / denom
    rmse_h = np.sqrt(sum_sq / denom)
    return mae_h.astype(np.float64), rmse_h.astype(np.float64)


def _report_one_target(
    *,
    stage_name: str,
    dataset_key: Optional[str],
    data_path: str,
    input_len: int,
    output_len: int,
    split: Sequence[float | int],
    split_mode: Optional[str],
    use_revin: bool,
    primary_supervision_space: Optional[str],
    basicts_scale_logging: bool,
    scaler_type: str,
    stage_results_path: Optional[str],
) -> Dict[str, Any]:
    from basicstfm.data.basicts_split import resolve_basicts_split_lengths
    from basicstfm.data.factost_split import split_triple_to_split_field
    from basicstfm.data.scaler import StandardScaler

    path = Path(data_path)
    arr = _load_array(path)
    t, n, c = arr.shape

    split_spec: Sequence[float | int] = split
    basicts_meta: Dict[str, Any] = {}
    if split_mode == "basicts":
        triple, basicts_meta = resolve_basicts_split_lengths(
            total_timesteps=t,
            dataset_key=str(dataset_key or path.parent.name),
            fallback_split=tuple(float(x) for x in split),
        )
        split_spec = split_triple_to_split_field(triple)

    tr, va, te = _split_lengths(t, split_spec)
    train_raw = np.asarray(arr[:tr], dtype=np.float32)
    test_raw = np.asarray(arr[tr + va : tr + va + te], dtype=np.float32)

    scaler_summary: Optional[Dict[str, Any]] = None
    scaler: Any = None
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.fit(train_raw)
        scaler_summary = {
            "mean_shape": list(scaler.mean.shape),
            "std_shape": list(scaler.std.shape),
            "mean_ch0_global_mean": float(np.mean(scaler.mean[..., 0])),
            "std_ch0_global_mean": float(np.mean(scaler.std[..., 0])),
        }

    window = input_len + output_len
    baselines: Dict[str, Any] = {"num_windows_test": 0}
    if len(test_raw) >= window:
        n_win = len(test_raw) - window + 1
        denom = float(n_win * n * c)
        sum_abs_z = np.zeros(output_len, dtype=np.float64)
        sum_sq_z = np.zeros(output_len, dtype=np.float64)
        sum_abs_p = np.zeros(output_len, dtype=np.float64)
        sum_sq_p = np.zeros(output_len, dtype=np.float64)

        for start in range(n_win):
            seg = test_raw[start : start + window]
            x_np = seg[:input_len]
            y_np = seg[input_len:]
            last = x_np[-1:, :, :]
            persistence = np.tile(last, (output_len, 1, 1))
            zero = np.zeros_like(y_np)
            _horizon_metrics_aggregate(zero, y_np, sum_abs=sum_abs_z, sum_sq=sum_sq_z)
            _horizon_metrics_aggregate(persistence, y_np, sum_abs=sum_abs_p, sum_sq=sum_sq_p)

        mae_z, rmse_z = _finalize_horizon_metrics(sum_abs_z, sum_sq_z, denom=denom)
        mae_p, rmse_p = _finalize_horizon_metrics(sum_abs_p, sum_sq_p, denom=denom)

        def _pick(idx: int) -> Dict[str, float]:
            return {"mae": float(mae_p[idx]), "rmse": float(rmse_p[idx])}

        horizon_keys = {1: 0, 3: 2, 6: 5, 12: 11}
        persistence_horizons = {
            f"horizon_{k}": _pick(idx) for k, idx in horizon_keys.items() if idx < output_len
        }
        baselines = {
            "num_windows_test": int(n_win),
            "zero_original_scale": {
                "per_horizon_mae": [float(x) for x in mae_z],
                "per_horizon_rmse": [float(x) for x in rmse_z],
                "average_over_horizons_mae": float(mae_z.mean()),
                "average_over_horizons_rmse": float(rmse_z.mean()),
                "average_over_12_mae": float(mae_z.mean()),
                "average_over_12_rmse": float(rmse_z.mean()),
            },
            "persistence_original_scale": {
                "per_horizon_mae": [float(x) for x in mae_p],
                "per_horizon_rmse": [float(x) for x in rmse_p],
                "average_over_horizons_mae": float(mae_p.mean()),
                "average_over_horizons_rmse": float(rmse_p.mean()),
                "average_over_12_mae": float(mae_p.mean()),
                "average_over_12_rmse": float(rmse_p.mean()),
                **persistence_horizons,
            },
        }

    metric_note = (
        "When task.use_revin=false and primary_supervision_space=denormalized, "
        "logged test/metric/mae equals metric/mae_original_after_inverse_standard_scaler "
        "(inverse StandardScaler on predictions & targets before reduction)."
    )

    stage_metrics = None
    if stage_results_path:
        rp = Path(stage_results_path)
        if rp.is_file():
            payload = json.loads(rp.read_text(encoding="utf-8"))
            stage_metrics = _extract_latest_metrics(payload, data_path=data_path)
        else:
            stage_metrics = {"note": f"missing file {rp}"}

    return {
        "stage_name": stage_name,
        "dataset_key": dataset_key,
        "data_path": str(path),
        "shape_TNC": [t, n, c],
        "selected_channel_note": "channel 0 only when building *_BasicTS NPZs from multi-channel sources",
        "split_yaml_fallback": list(float(x) for x in split) if split_mode == "basicts" else list(split),
        "split_mode": split_mode,
        "basicts_split_meta": basicts_meta,
        "split_lengths_resolved": {"train": tr, "val": va, "test": te},
        "protocol_task": {
            "use_revin": use_revin,
            "primary_supervision_space": primary_supervision_space,
            "basicts_scale_logging": basicts_scale_logging,
            "scaler_type": scaler_type,
        },
        "standard_scaler_train_fit": scaler_summary,
        "baselines_original_scale": baselines,
        "metric_space_audit": {
            "test_metric_mae_maps_to": "metric/mae_original_after_inverse_standard_scaler",
            "condition": metric_note,
            "expected_use_revin": False,
        },
        "stage_results_metrics": stage_metrics,
    }


def _extract_latest_metrics(payload: Dict[str, Any], *, data_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return out
    ds_name = Path(data_path).parent.name
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
            "metric/mae_original_after_inverse_standard_scaler",
            "metric/rmse_original_after_inverse_standard_scaler",
        )
        for k in keys:
            if k in test_block:
                out[k] = test_block[k]
        break
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    parser = argparse.ArgumentParser(description="Audit BasicTS-style traffic protocol.")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument("--dataset", default=None, help="Filter by registry dataset_key (e.g. PEMS04)")
    parser.add_argument(
        "--stage-results",
        default=None,
        help="Optional stage_results.json from a finished run",
    )
    parser.add_argument("--cwd", default=str(REPO_ROOT), help="Working directory for relative paths")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(args.cwd)

    from basicstfm.config import load_config
    from basicstfm.engines.stage import StagePlan

    cfg = load_config(args.config)
    rows = StagePlan.describe_factost_protocol_audit(cfg)
    cfg_fallback = cfg.get("data") or {}

    targets: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("eval_only"):
            continue
        dk = row.get("dataset_key")
        if args.dataset and dk != args.dataset:
            continue
        targets.append(row)

    reports: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for row in targets:
        dp = row.get("data_path")
        if not dp:
            reports.append({"stage_name": row.get("name"), "error": "missing data_path"})
            continue
        il = row.get("input_len") or cfg_fallback.get("input_len")
        ol = row.get("output_len") or cfg_fallback.get("output_len")
        split = tuple(row.get("split") or cfg_fallback.get("split", (0.6, 0.2, 0.2)))
        sm = row.get("split_mode")
        if isinstance(sm, str):
            sm = sm.strip().lower() or None

        rep = _report_one_target(
            stage_name=str(row.get("name")),
            dataset_key=row.get("dataset_key"),
            data_path=str(dp),
            input_len=int(il),
            output_len=int(ol),
            split=split,
            split_mode=sm,
            use_revin=bool(row.get("use_revin")),
            primary_supervision_space=row.get("primary_supervision_space"),
            basicts_scale_logging=bool(row.get("basicts_scale_logging")),
            scaler_type=str(row.get("data_scaler_type") or "standard"),
            stage_results_path=args.stage_results,
        )
        if rep["protocol_task"]["use_revin"]:
            warnings.append(
                f"{rep['stage_name']}: use_revin=true but BasicTS protocol expects false."
            )
        if rep["split_mode"] != "basicts":
            warnings.append(
                f"{rep['stage_name']}: split_mode is not 'basicts' (got {rep['split_mode']!r})."
            )
        reports.append(rep)

    payload = {"config": args.config, "targets": reports, "warnings": warnings}

    if not args.json_only:
        print("=== BasicTS protocol audit (eval_only stages) ===")
        print(f"config: {args.config}")
        if args.dataset:
            print(f"filter dataset_key={args.dataset!r}")
        for w in warnings:
            print(f"WARNING: {w}")
        for item in reports:
            if "error" in item:
                print(f"- ERROR {item}")
                continue
            print(
                f"- {item['stage_name']} | {item['dataset_key']} | "
                f"TNC={item['shape_TNC']} | split_mode={item.get('split_mode')}"
            )
        print()

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
