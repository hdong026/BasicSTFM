#!/usr/bin/env python3
"""Audit BasicTS-protocol baseline configs (DPM-SR + OpenCity + FactoST + UniST).

Checks YAML stages for data paths (*_BasicTS), scaler, RevIN, split_mode, metrics,
source groups, and few-shot ratios. Optionally counts model parameters by building
the model on CPU (requires deps + data layout for shape inference).

Usage:
  python scripts/analysis/audit_baseline_suite.py
  python scripts/analysis/audit_baseline_suite.py --configs /path/to/a.yaml /path/to/b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS = [
    REPO_ROOT / "configs/monash/dpm_sr_monash15_then_traffic_sharded_transfer_12_basicts_protocol.yaml",
    REPO_ROOT / "configs/basicts/opencity_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml",
    REPO_ROOT / "configs/basicts/factost_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml",
    REPO_ROOT / "configs/basicts/unist_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml",
]

TRAFFIC_KEYS_BASIC = frozenset(
    {"METR-LA", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS-BAY"}
)

EXPECTED_SOURCES = [
    "LargeST_xd_part0",
    "LargeST_xd_part1",
    "KnowAir",
    "Weather",
    "ETTm1",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_target_traffic_stage(stage: Dict[str, Any], registry: Dict[str, Any]) -> bool:
    data = stage.get("data") or {}
    key = data.get("dataset_key")
    if key is None:
        return False
    if str(key) not in TRAFFIC_KEYS_BASIC:
        return False
    ent = registry.get(str(key)) or {}
    dp = str(ent.get("data_path") or "")
    return "_BasicTS/data.npz" in dp.replace("\\", "/")


def _norm_relpath(p: str) -> str:
    return str(p).replace("\\", "/")


def _audit_stage(
    stage: Dict[str, Any],
    *,
    baseline_name: str,
    registry: Dict[str, Any],
) -> List[str]:
    errs: List[str] = []
    name = stage.get("name", "<unnamed>")
    data = stage.get("data") or {}
    task = stage.get("task") or {}
    dk = data.get("dataset_key")

    if not _is_target_traffic_stage(stage, registry):
        return errs

    ent = registry.get(str(dk)) or {}
    dp = _norm_relpath(str(ent.get("data_path") or ""))
    if not dp.endswith("_BasicTS/data.npz"):
        errs.append(f"{name}: data_path must be *_BasicTS/data.npz for {dk}, got {dp!r}")

    if str(dk) == "PEMS04":
        abs_p = REPO_ROOT / dp
        if abs_p.is_file():
            try:
                z = np.load(str(abs_p), mmap_mode="r")
                arr = z["data"]
                sh = tuple(arr.shape)
                z.close()
                if sh != (16992, 307, 1):
                    errs.append(f"{name}: PEMS04 BasicTS shape expected (16992,307,1) got {sh}")
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{name}: could not verify PEMS04 shape ({exc})")

    sc = (data.get("scaler") or {}).get("type")
    if sc != "standard":
        errs.append(f"{name}: scaler.type must be 'standard', got {sc!r}")

    if bool(task.get("use_revin", False)):
        errs.append(f"{name}: target use_revin must be false for BasicTS protocol (got true)")

    if data.get("split_mode") != "basicts":
        errs.append(
            f"{name}: split_mode must be 'basicts' for BasicTS traffic targets "
            f"(got {data.get('split_mode')!r})"
        )

    if task.get("primary_supervision_space") != "denormalized":
        errs.append(
            f"{name}: task.primary_supervision_space should be 'denormalized' "
            f"(got {task.get('primary_supervision_space')!r})"
        )

    if not bool(task.get("basicts_scale_logging", False)):
        errs.append(f"{name}: task.basicts_scale_logging should be true for metric audit mapping")

    if "max_test_windows" in data:
        errs.append(f"{name}: must not set max_test_windows (full test)")

    return errs


def _few_shot_ratios_by_slug(stages: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    slugs = ("metr_la", "pems03", "pems04", "pems07", "pems08", "pems_bay", "etth2", "electricity", "weather_tgt")
    out: Dict[str, List[float]] = {s: [] for s in slugs}
    for st in stages:
        name = str(st.get("name", ""))
        r = st.get("few_shot_ratio")
        if r is None:
            continue
        for slug in slugs:
            if name.startswith(f"{slug}_"):
                out[slug].append(float(r))
    return out


def _pretrain_sources(cfg: Dict[str, Any]) -> Optional[List[str]]:
    groups = cfg.get("dataset_groups") or {}
    src = groups.get("cross_domain_sharded_sources")
    if isinstance(src, list):
        return [str(x) for x in src]
    return None


def audit_config(path: Path, *, count_params: bool) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    registry = cfg.get("dataset_registry") or {}
    stages = list((cfg.get("pipeline") or {}).get("stages") or [])
    baseline = path.stem

    errs: List[str] = []
    for st in stages:
        errs.extend(_audit_stage(st, baseline_name=baseline, registry=registry))

    src = _pretrain_sources(cfg)
    if src != EXPECTED_SOURCES:
        errs.append(
            f"cross_domain_sharded_sources mismatch: expected {EXPECTED_SOURCES} got {src}"
        )

    fs = _few_shot_ratios_by_slug(stages)
    for slug in ("metr_la", "pems03", "pems04", "pems07", "pems08", "pems_bay"):
        ratios = fs.get(slug, [])
        if not ratios:
            errs.append(f"missing few-shot stages for slug {slug!r}")
            continue
        if 0.05 not in ratios:
            errs.append(f"{slug!r}: few_shot_ratio 0.05 missing (got {sorted(set(ratios))})")
        if 0.10 not in ratios:
            errs.append(f"{slug!r}: few_shot_ratio 0.10 missing (got {sorted(set(ratios))})")

    status = "FAIL" if errs else "OK"

    total: Optional[int] = None
    trainable: Optional[int] = None
    perr: Optional[str] = None
    if count_params:
        mtype = str((cfg.get("model") or {}).get("type", ""))
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from basicstfm.builders import import_builtin_components  # type: ignore
            from basicstfm.registry import MODELS  # type: ignore

            import_builtin_components()
            mcfg = dict(cfg.get("model") or {})
            if "SRDSTFM" in mtype:
                fill = {
                    "num_nodes": 307,
                    "input_dim": 1,
                    "output_dim": 1,
                    "input_len": 12,
                    "output_len": 12,
                    "num_datasets": 5,
                }
            else:
                fill = {
                    "num_nodes": 307,
                    "input_dim": 18,
                    "output_dim": 1,
                    "input_len": 12,
                    "output_len": 12,
                }
            for k, v in fill.items():
                if mcfg.get(k) in (None, "auto"):
                    mcfg[k] = v
            model = MODELS.build(mcfg)
            total = int(sum(p.numel() for p in model.parameters()))
            trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        except Exception as exc:  # noqa: BLE001
            perr = str(exc)

    return {
        "config": str(path.relative_to(REPO_ROOT)),
        "status": status,
        "errors": errs,
        "params_total": total,
        "params_trainable_default": trainable,
        "param_count_error": perr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", type=Path, default=None)
    ap.add_argument("--no-param-count", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    paths = [Path(p).resolve() for p in (args.configs or DEFAULT_CONFIGS)]
    rows = [audit_config(p, count_params=not args.no_param_count) for p in paths]
    if args.json:
        print(json.dumps(rows, indent=2))
        raise SystemExit(0 if all(r["status"] == "OK" for r in rows) else 1)

    fail = False
    for r in rows:
        print(f"== {r['config']} [{r['status']}] ==")
        if r.get("params_total") is not None:
            print(f"  model params: {r['params_total']:,} (all trainable by default: {r['params_trainable_default']:,})")
        if r.get("param_count_error"):
            print(f"  param count skipped: {r['param_count_error']}")
        for e in r["errors"]:
            print(f"  FAIL: {e}")
            fail = True
        if not r["errors"]:
            print("  (target-stage BasicTS checks passed)")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
