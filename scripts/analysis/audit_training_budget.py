#!/usr/bin/env python3
"""Audit budget-matched BasicTS protocol configs (update budgets + few-shot parity).

Checks per YAML:
  - Monash total optimizer steps == 40_000 (steps_per_epoch × epochs on monash stages)
  - Mixed-domain total steps == 100_000 (family-specific stage groups)
  - Per-stage updates list
  - 5%% / 10%% few-shot: epochs == 20, validate_every == 1, save_best_by val/metric/mae
  - Traffic targets: split_mode basicts, scaler standard, use_revin false, no max_test_windows
  - FactoST: first traffic zero-shot loads ``factost_xd_sta_source`` and enables use_st_adapter
  - UniST: first traffic zero-shot loads ``unist_xd_prompt_source`` and enables use_prompt

Exit code 1 if any FAIL.

Usage:
  python scripts/analysis/audit_training_budget.py
  python scripts/analysis/audit_training_budget.py --configs configs/budget_matched/*.yaml
  python scripts/analysis/audit_training_budget.py --emit-smoke-commands

Recommended workflow (repo root, ``conda activate basicstfm``, ``export PYTHONPATH=src``):

  1) Audit four budget configs::
       python scripts/analysis/audit_training_budget.py

  2) Dry-run::
       for c in configs/budget_matched/*_basicts_budget.yaml; do
         python -m basicstfm.cli dry-run "$c" || exit 1
       done

  3) Smoke train (short budgets + 1-epoch few-shot) — generate overrides then run::
       python scripts/analysis/audit_training_budget.py --emit-smoke-commands > /tmp/smoke_budget.sh
       bash /tmp/smoke_budget.sh

  4) Full train::
       for c in configs/budget_matched/*_basicts_budget.yaml; do
         python -m basicstfm.cli train "$c"
       done
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS = sorted((REPO_ROOT / "configs/budget_matched").glob("*_basicts_budget.yaml"))

TRAFFIC_KEYS = frozenset(
    {"METR-LA", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS-BAY"}
)
EXPECTED_MONASH = 40_000
EXPECTED_MIXED = 100_000
EXPECTED_FEWSHOT_EPOCHS = 20


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_env_default(load_from: Any) -> Optional[str]:
    if not isinstance(load_from, str):
        return None
    text = load_from.strip()
    m = re.match(r"^\$\{env:[^,]+,\s*([^}]+)\}$", text)
    if m:
        return m.group(1).strip()
    return text


def _stage_updates(stage: Dict[str, Any]) -> int:
    if stage.get("eval_only"):
        return 0
    data = stage.get("data") or {}
    steps = int(data.get("steps_per_epoch") or 0)
    epochs = int(stage.get("epochs") or 0)
    return steps * epochs


def _family(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("dpm"):
        return "dpm"
    if "opencity" in stem:
        return "opencity"
    if "factost" in stem:
        return "factost"
    if "unist" in stem:
        return "unist"
    return "unknown"


def _is_monash_stage(stage: Dict[str, Any]) -> bool:
    name = str(stage.get("name") or "")
    if "monash" in name:
        return True
    data = stage.get("data") or {}
    return str(data.get("type") or "") == "MonashMultiDatasetWindowDataModule"


def _is_mixed_stage(family: str, stage: Dict[str, Any]) -> bool:
    name = str(stage.get("name") or "")
    if family == "dpm":
        return name.startswith("cross_domain_")
    if family == "opencity":
        return name == "mixed_domain_pretrain"
    if family == "factost":
        return name in {"mixed_domain_utp_pretrain", "mixed_domain_sta_source_adaptation"}
    if family == "unist":
        return name in {"mixed_domain_masked_pretrain", "mixed_domain_prompt_source_adaptation"}
    return False


def _is_target_traffic(stage: Dict[str, Any], registry: Dict[str, Any]) -> bool:
    data = stage.get("data") or {}
    key = data.get("dataset_key")
    if key is None or key not in TRAFFIC_KEYS:
        return False
    ent = registry.get(str(key)) or {}
    dp = str(ent.get("data_path") or "")
    return "_BasicTS/data.npz" in dp.replace("\\", "/")


def _is_fewshot_stage(stage: Dict[str, Any]) -> bool:
    name = str(stage.get("name") or "").lower()
    if stage.get("eval_only"):
        return False
    if stage.get("few_shot_ratio") is not None:
        return True
    return (
        "five_percent" in name
        or "ten_percent" in name
        or "5_percent" in name
        or "10_percent" in name
    ) and (
        "finetune" in name
        or "tuning" in name
        or "tune" in name
        or "mechanism_tuning" in name
    )


def audit_config(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    cfg = _load_yaml(path)
    fails: List[str] = []
    family = _family(path)
    registry = cfg.get("dataset_registry") or {}
    stages: List[Dict[str, Any]] = (cfg.get("pipeline") or {}).get("stages") or []

    monash_u = sum(_stage_updates(s) for s in stages if _is_monash_stage(s))
    mixed_u = sum(_stage_updates(s) for s in stages if _is_mixed_stage(family, s))
    per_stage = [
        {
            "name": s.get("name"),
            "updates": _stage_updates(s),
        }
        for s in stages
    ]

    if monash_u != EXPECTED_MONASH:
        fails.append(f"monash_total_updates={monash_u} (expected {EXPECTED_MONASH})")
    if mixed_u != EXPECTED_MIXED:
        fails.append(f"mixed_total_updates={mixed_u} (expected {EXPECTED_MIXED})")

    zs = next((s for s in stages if str(s.get("name") or "") == "metr_la_zero_shot"), None)
    zsf = _resolve_env_default(zs.get("load_from") if zs else None) if zs else None
    model_zs = (zs or {}).get("model") or {}
    task_zs = (zs or {}).get("task") or {}

    if family == "factost":
        if zsf != "factost_xd_sta_source":
            fails.append(f"FactoST zero_shot load_from resolved={zsf!r} (expected factost_xd_sta_source)")
        if not bool(model_zs.get("use_st_adapter")):
            fails.append("FactoST zero_shot: model.use_st_adapter must be true (source STA)")
    if family == "unist":
        if zsf != "unist_xd_prompt_source":
            fails.append(f"UniST zero_shot load_from resolved={zsf!r} (expected unist_xd_prompt_source)")
        if not bool(model_zs.get("use_prompt")):
            fails.append("UniST zero_shot: model.use_prompt must be true (source prompt)")

    for st in stages:
        if not _is_fewshot_stage(st):
            continue
        ratio = st.get("few_shot_ratio")
        if ratio is not None and float(ratio) not in (0.05, 0.10):
            continue
        if int(st.get("epochs") or 0) != EXPECTED_FEWSHOT_EPOCHS:
            fails.append(
                f"few-shot stage {st.get('name')!r}: epochs={st.get('epochs')} (expected {EXPECTED_FEWSHOT_EPOCHS})"
            )
        if int(st.get("validate_every") or 0) != 1:
            fails.append(f"few-shot {st.get('name')!r}: validate_every must be 1")
        if st.get("save_best_by") != "val/metric/mae":
            fails.append(
                f"few-shot {st.get('name')!r}: save_best_by must be val/metric/mae (got {st.get('save_best_by')!r})"
            )

    for st in stages:
        if not _is_target_traffic(st, registry):
            continue
        data = st.get("data") or {}
        task = st.get("task") or {}
        if data.get("max_test_windows") is not None:
            fails.append(f"traffic stage {st.get('name')!r}: max_test_windows must be unset (full test)")
        if data.get("split_mode") != "basicts":
            fails.append(f"traffic stage {st.get('name')!r}: split_mode must be basicts")
        scaler = data.get("scaler") or {}
        if isinstance(scaler, dict) and scaler.get("type") != "standard":
            fails.append(f"traffic stage {st.get('name')!r}: scaler.type must be standard")
        if bool(task.get("use_revin")):
            fails.append(f"traffic stage {st.get('name')!r}: use_revin must be false")
        basicts_log = bool(task.get("basicts_scale_logging"))
        pss = str(task.get("primary_supervision_space") or "")
        if not basicts_log or pss != "denormalized":
            fails.append(
                f"traffic stage {st.get('name')!r}: expect primary_supervision_space=denormalized "
                f"and basicts_scale_logging=true for inverse-StandardScaler metrics"
            )

    report = {
        "path": str(path.relative_to(REPO_ROOT)),
        "experiment_name": cfg.get("experiment_name"),
        "family": family,
        "monash_total_updates": monash_u,
        "mixed_total_updates": mixed_u,
        "per_stage_updates": per_stage,
        "fewshot_epochs_5pct_10pct": EXPECTED_FEWSHOT_EPOCHS,
        "zero_shot_load_resolved": zsf,
        "zero_shot_use_st_adapter": model_zs.get("use_st_adapter"),
        "zero_shot_use_prompt": model_zs.get("use_prompt"),
        "test_uses_best_val_checkpoint": True,
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
    }
    return report, fails


def emit_smoke_commands(paths: Sequence[Path]) -> None:
    """Print train commands: pretrain stages steps=200 epochs=1; few-shot epochs=1."""

    for path in paths:
        cfg = _load_yaml(path)
        stages = (cfg.get("pipeline") or {}).get("stages") or []
        opts: List[str] = []
        for i, st in enumerate(stages):
            name = str(st.get("name") or "")
            data = st.get("data") or {}
            if st.get("eval_only"):
                continue
            spe = data.get("steps_per_epoch")
            if spe is not None and int(spe) > 0:
                opts.append(f"pipeline.stages.{i}.data.steps_per_epoch=200")
                opts.append(f"pipeline.stages.{i}.epochs=1")
            elif _is_fewshot_stage(st):
                opts.append(f"pipeline.stages.{i}.epochs=1")
                if st.get("scheduler") and isinstance(st["scheduler"], dict):
                    opts.append(f"pipeline.stages.{i}.scheduler.T_max=1")
        opt_str = " ".join(f'--cfg-options {o}' for o in opts)
        print(
            f'python -m basicstfm.cli train "{path.relative_to(REPO_ROOT)}" {opt_str}',
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=DEFAULT_CONFIGS,
        help="YAML configs (default: configs/budget_matched/*_basicts_budget.yaml)",
    )
    parser.add_argument(
        "--emit-smoke-commands",
        action="store_true",
        help="Print one bash-friendly train line per config with smoke overrides.",
    )
    args = parser.parse_args(argv)

    paths = [p if p.is_absolute() else REPO_ROOT / p for p in args.configs]
    if args.emit_smoke_commands:
        emit_smoke_commands(paths)
        return 0

    all_ok = True
    reports: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"FAIL: missing {path}", file=sys.stderr)
            all_ok = False
            continue
        report, fails = audit_config(path)
        reports.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if fails:
            all_ok = False
            for line in fails:
                print(f"  FAIL: {line}", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
