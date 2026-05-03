#!/usr/bin/env python3
"""Audit few-shot / adapter-tune stages across budget YAMLs: trainable params, load route, protocol.

Example commands (from repo root):

1) Audit five-model few-shot routes::

    python scripts/analysis/audit_fewshot_trainability.py \\
      configs/budget_matched/dpm_sr_monash15_then_mixed_12_basicts_budget_stage2_only_fs.yaml \\
      configs/budget_matched/dpm_srpp_monash15_then_mixed_12_basicts_budget_stage2_only_fs.yaml \\
      configs/budget_matched/opencity_monash15_then_mixed_12_basicts_budget.yaml \\
      configs/budget_matched/factost_monash15_then_mixed_12_basicts_budget.yaml \\
      configs/budget_matched/unist_monash15_then_mixed_12_basicts_budget_lite.yaml \\
      --output-md reports/fewshot_trainability_audit.md \\
      --output-json reports/fewshot_trainability_audit.json

2) Dry-run one config::

    basicstfm dry-run configs/budget_matched/opencity_monash15_then_mixed_12_basicts_budget.yaml

3) Smoke train (shorten epochs via overrides)::

    basicstfm train configs/budget_matched/opencity_monash15_then_mixed_12_basicts_budget.yaml \\
      --cfg-options trainer.work_dir=runs/smoke_opencity pipeline.stages[].epochs=1

4) Full train::

    basicstfm train configs/budget_matched/opencity_monash15_then_mixed_12_basicts_budget.yaml
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

KEYWORDS = (
    "five_percent",
    "ten_percent",
    "few_shot",
    "finetune",
    "fine_tune",
    "tune",
    "mechanism_tuning",
)

AUDIT_DIMS = {
    "num_nodes": 207,
    "input_dim": 1,
    "output_dim": 1,
    "input_len": 12,
    "output_len": 12,
}

DPM_SR_EXPECTED_UNFREEZE = frozenset(
    {
        "residual_event_encoder.*",
        "diffusion_mechanism_learner.*",
        "fusion_predictor.*",
        "calibration_head.*",
    }
)

SRPP_EXTRA_UNFREEZE = frozenset(
    {
        "event_gate.*",
        "regime_gate.*",
        "graph_event_conditioner.*",
        "propagation_gate.*",
        "dynamic_adapter.*",
    }
)

FACTOST_ALLOWED_UNFREEZE_PREFIXES = (
    "prompt_",
    "prompt.",
    "prompt_adapter.",
    "metadata_gate.",
    "st_",
    "forecast_head.",
    "cpr.",
    "domain_prompt.",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(p == "all" or fnmatch.fnmatch(name, p) for p in patterns)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _stage_matches_keywords(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in KEYWORDS)


def _is_pure_eval(stage: Mapping[str, Any]) -> bool:
    return bool(stage.get("eval_only")) and int(stage.get("epochs") or 0) == 0


def _freeze_unfreeze(stage: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    fr = stage.get("freeze") or []
    un = stage.get("unfreeze") or []
    if isinstance(fr, str):
        fr = [fr]
    if isinstance(un, str):
        un = [un]
    return [str(x) for x in fr], [str(x) for x in un]


def apply_trainability(model: Any, freeze: Sequence[str], unfreeze: Sequence[str]) -> None:
    for p in model.parameters():
        p.requires_grad = True
    for name, p in model.named_parameters():
        if _matches_any(name, freeze):
            p.requires_grad = False
        if _matches_any(name, unfreeze):
            p.requires_grad = True


def _prepare_model_cfg(cfg: Dict[str, Any], stage: Mapping[str, Any]) -> Dict[str, Any]:
    base = cfg.get("model") or {}
    merged = _merge_dicts(base, stage.get("model") or {})
    out = dict(merged)
    for key, val in AUDIT_DIMS.items():
        if key not in out or _is_auto(out.get(key)):
            out[key] = val
    return out


def _detect_family(cfg: Dict[str, Any], path: Path) -> str:
    exp = str(cfg.get("experiment_name") or "").lower()
    p = path.name.lower()
    m = cfg.get("model") or {}
    mtype = str(m.get("type") or "")
    if "dpm_srpp" in exp or "srpp" in p or "sr++" in exp:
        return "dpm_srpp"
    if exp.startswith("dpm_sr") or "dpm_sr" in p:
        return "dpm_sr"
    if mtype == "OpenCityFoundationModel":
        return "opencity"
    if mtype == "FactoSTFoundationModel":
        return "factost"
    if mtype == "UniSTFoundationModel":
        return "unist_lite"
    return "unknown"


def _trainer_reloads_best_before_test() -> bool:
    t = REPO / "src" / "basicstfm" / "engines" / "trainer.py"
    if not t.is_file():
        return False
    txt = t.read_text(encoding="utf-8")
    return "Reloading best validation checkpoint for test" in txt


def _protocol_rows(stage: Mapping[str, Any]) -> Tuple[List[str], bool]:
    """Return (issues, original_scale_ok)."""
    issues: List[str] = []
    data = stage.get("data") or {}
    task = stage.get("task") or {}
    ttype = str(task.get("type") or "")
    if data.get("type") != "WindowDataModule":
        return issues, True
    if data.get("max_test_windows") is not None:
        issues.append("data.max_test_windows must be null for full BasicTS test")
    if str((data.get("scaler") or {}).get("type") or "").lower() not in {"standard", "zscore"}:
        issues.append("data.scaler.type should be standard")
    if data.get("split_mode") != "basicts":
        issues.append("data.split_mode must be basicts")
    if task.get("use_revin") is True:
        issues.append("task.use_revin must be false for main-table protocol")
    if ttype == "ForecastingTask":
        orig_ok = (
            str(task.get("primary_supervision_space") or "").lower().replace("-", "_") == "denormalized"
        )
        if not orig_ok:
            issues.append("ForecastingTask.primary_supervision_space must be denormalized")
        if not task.get("basicts_scale_logging"):
            issues.append("ForecastingTask.basicts_scale_logging should be true")
    else:
        orig_ok = True
    return issues, orig_ok


def _audit_route(
    family: str,
    cfg: Dict[str, Any],
    path: Path,
    stage: Mapping[str, Any],
    trainable_ratio: float,
    trainable_names: Sequence[str],
) -> Tuple[bool, str, List[str]]:
    """Returns (ok, label, failures)."""
    fails: List[str] = []
    name = str(stage.get("name") or "")
    frz, unf = _freeze_unfreeze(stage)
    task = stage.get("task") or {}
    task_type = str(task.get("type") or "")
    load_method = str(stage.get("load_method") or "checkpoint")

    label = "custom"
    if family == "dpm_sr":
        label = "DPM-SR Stage-II-only (event + diffusion mechanism + fusion + calibration)"
        if set(unf) != DPM_SR_EXPECTED_UNFREEZE:
            fails.append(f"{name}: unfreeze must match DPM-SR few-shot set")
        for p in unf:
            if "stable_trunk" in p:
                fails.append(f"{name}: must not unfreeze stable_trunk*")
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all]")
        if task_type != "StableResidualForecastingTask":
            fails.append(f"{name}: task.type must be StableResidualForecastingTask")
        if str(task.get("phase") or "") != "joint":
            fails.append(f"{name}: task.phase must be joint")
        required = (
            ("stable_weight", 0.0),
            ("spectral_weight", 0.0),
            ("cross_cov_weight", 0.0),
            ("residual_weight", 0.1),
            ("final_weight", 1.0),
        )
        for k, v in required:
            if k not in task:
                fails.append(f"{name}: missing task.{k}")
            elif not math.isclose(float(task[k]), float(v), rel_tol=0, abs_tol=1e-9):
                fails.append(f"{name}: task.{k} must be {v}, got {task[k]!r}")
        lr = float((stage.get("optimizer") or {}).get("lr") or 0)
        if lr not in (0.0003, 0.0004, 0.0005):
            fails.append(f"{name}: optimizer.lr must be in [3e-4,4e-4,5e-4], got {lr}")

    elif family == "dpm_srpp":
        label = "DPM-SR++ Stage-II-only (+ optional event gates); stable law frozen"
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all]")
        if not DPM_SR_EXPECTED_UNFREEZE.issubset(set(unf)):
            fails.append(
                f"{name}: unfreeze must include core set {sorted(DPM_SR_EXPECTED_UNFREEZE)}"
            )
        extras = set(unf) - DPM_SR_EXPECTED_UNFREEZE
        if not extras.issubset(SRPP_EXTRA_UNFREEZE):
            fails.append(
                f"{name}: unexpected extra unfreeze patterns {sorted(extras - SRPP_EXTRA_UNFREEZE)}"
            )
        for p in unf:
            pl = p.lower()
            for bad in ("stable_trunk", "persistence_anchor", "robust_stable", "stable_delta_head"):
                if bad in pl:
                    fails.append(f"{name}: forbidden few-shot pattern (stable-law): {p!r}")
        if task_type != "StableResidualForecastingTaskV5":
            fails.append(f"{name}: task.type must be StableResidualForecastingTaskV5")
        if str(task.get("phase") or "") != "joint":
            fails.append(f"{name}: task.phase must be joint")
        required = (
            ("stable_weight", 0.0),
            ("spectral_weight", 0.0),
            ("cross_cov_weight", 0.0),
            ("residual_weight", 0.1),
            ("final_weight", 1.0),
        )
        for k, v in required:
            if k not in task:
                fails.append(f"{name}: missing task.{k}")
            elif not math.isclose(float(task[k]), float(v), rel_tol=0, abs_tol=1e-9):
                fails.append(f"{name}: task.{k} must be {v}, got {task[k]!r}")
        if task.get("robust_stage1") is True:
            fails.append(f"{name}: few-shot must set robust_stage1 false")
        rw = float(task.get("robust_lambda", 0.0))
        if rw > 1e-12:
            fails.append(f"{name}: few-shot must set robust_lambda 0.0, got {rw}")
        lr = float((stage.get("optimizer") or {}).get("lr") or 0)
        if lr not in (0.0003, 0.0004, 0.0005):
            fails.append(f"{name}: optimizer.lr must be in [3e-4,5e-4], got {lr}")
        for bad_sub in ("persistence_anchor", "robust_stable", "stable_delta_head"):
            for tname in trainable_names:
                if bad_sub in tname:
                    fails.append(f"{name}: trainable param {tname} violates frozen stable-law narrative")

    elif family == "opencity":
        label = "OpenCity head-only tuning"
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all] for head-only")
        if tuple(unf) != ("forecast_head.*",):
            fails.append(f"{name}: unfreeze must be exactly [forecast_head.*], got {unf!r}")
        if trainable_ratio > 0.25:
            fails.append(f"{name}: trainable_ratio {trainable_ratio:.4f} suggests full finetune")

    elif family == "factost":
        label = "FactoST adapter + prompt + head tuning"
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all]")
        if load_method.lower() in {"load_backbone_weights", "backbone"}:
            fails.append(f"{name}: load_method must be checkpoint/full load, not backbone-only")
        lm = load_method.lower()
        if lm not in {"checkpoint", "state_dict", "full_checkpoint"}:
            fails.append(f"{name}: load_method should be checkpoint/state_dict (got {load_method!r})")
        for p in unf:
            ok = False
            for pref in FACTOST_ALLOWED_UNFREEZE_PREFIXES:
                if fnmatch.fnmatch(p, pref + "*") or p == pref.rstrip(".*"):
                    ok = True
                    break
            if not ok and "*" not in p:
                if any(p.startswith(pref.rstrip("*")) for pref in ("prompt", "st_", "forecast_head", "metadata_gate", "domain_prompt", "cpr")):
                    ok = True
            if not ok:
                fails.append(f"{name}: unexpected unfreeze pattern {p!r} for FactoST few-shot")
        lr = float((stage.get("optimizer") or {}).get("lr") or 0)
        if not math.isclose(lr, 1e-3, rel_tol=0.05, abs_tol=1e-4):
            fails.append(f"{name}: FactoST few-shot lr expected ~1e-3, got {lr}")

    elif family == "unist_lite":
        is_head_cfg = "lite_head" in str(cfg.get("experiment_name") or "") or "lite_head" in path.name
        if is_head_cfg:
            label = "UniST-lite+Head (ablation)"
            if not any("prompt" in u for u in unf):
                fails.append(f"{name}: unfreeze must include prompt.*")
            if not any("reconstruction_head" in u for u in unf):
                fails.append(f"{name}: lite_head must unfreeze reconstruction_head.*")
        else:
            label = "UniST-lite prompt-only"
            if set(unf) != {"prompt.*"}:
                fails.append(f"{name}: main-table unfreeze must be {{prompt.*}}, got {unf!r}")
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all]")
        for forb in ("reconstruction_head", "forecast_head", "encoder.", "decoder.", "backbone."):
            if not is_head_cfg and any(forb in u for u in unf):
                fails.append(f"{name}: main UniST-lite must not unfreeze {forb}")
        m = cfg.get("model") or {}
        if int(m.get("num_memory_spatial") or 0) != 128 or int(m.get("num_memory_temporal") or 0) != 128:
            fails.append(f"{name}: UniST-lite expects num_memory_spatial/temporal 128")

    ok = not fails
    return ok, label, fails


def _sta_stage_audit(cfg: Dict[str, Any]) -> List[str]:
    """FactoST route: STA source must be ForecastingTask + use_st_adapter."""
    fails: List[str] = []
    for st in (cfg.get("pipeline") or {}).get("stages") or []:
        if str(st.get("save_artifact") or "") != "factost_xd_sta_source":
            continue
        tk = st.get("task") or {}
        if str(tk.get("type") or "") != "ForecastingTask":
            fails.append(
                f"stage {st.get('name')}: factost_xd_sta_source must use ForecastingTask, "
                f"got {tk.get('type')!r}"
            )
        mod = st.get("model") or {}
        if mod.get("use_st_adapter") is not True:
            fails.append(f"stage {st.get('name')}: STA source must set model.use_st_adapter: true")
    return fails


def audit_config(path: Path) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """Returns (summary dict, global failures, row dicts)."""
    from basicstfm.utils.results import pretty_model_name

    cfg = _load_yaml(path)
    experiment_name = str(cfg.get("experiment_name") or path.stem)
    family = _detect_family(cfg, path)
    global_f: List[str] = []
    if family == "unknown":
        global_f.append(f"unknown model family for {path.name}")

    if family == "factost":
        global_f.extend(_sta_stage_audit(cfg))

    rows: List[Dict[str, Any]] = []
    stages = (cfg.get("pipeline") or {}).get("stages") or []

    from basicstfm.builders import import_builtin_components
    from basicstfm.registry import MODELS

    import_builtin_components()

    for idx, st in enumerate(stages):
        name = str(st.get("name") or f"stage_{idx}")
        if not _stage_matches_keywords(name):
            continue
        if _is_pure_eval(st):
            continue
        if int(st.get("epochs") or 0) < 1:
            continue

        frz, unf = _freeze_unfreeze(st)
        mcfg = _prepare_model_cfg(cfg, st)
        try:
            model = MODELS.build(mcfg)
        except Exception as ex:  # noqa: BLE001
            rows.append(
                {
                    "config_path": str(path),
                    "experiment_name": experiment_name,
                    "display_name": pretty_model_name(
                        {"experiment_name": experiment_name, "model_type": mcfg.get("type")}
                    ),
                    "stage_name": name,
                    "error": f"MODEL BUILD FAILED: {ex}",
                }
            )
            global_f.append(f"{name}: could not build model — {ex}")
            continue

        apply_trainability(model, frz, unf)
        tnames = [n for n, p in model.named_parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = float(trainable) / float(total) if total else 0.0

        task = st.get("task") or {}
        proto_issues, orig_ok = _protocol_rows(st)
        opt = st.get("optimizer") or {}
        save_best = st.get("save_best", True)
        save_best_by = str(st.get("save_best_by") or "")

        suspected_full_ft = frz != ["all"] or (ratio > 0.85 and family not in {"opencity", "unist_lite"})

        route_ok, route_label, route_fails = _audit_route(
            family, cfg, path, st, ratio, tnames
        )
        row_fails = list(proto_issues) + route_fails
        global_f.extend(row_fails)

        rows.append(
            {
                "config_path": str(path.resolve()),
                "experiment_name": experiment_name,
                "display_name": pretty_model_name(
                    {"experiment_name": experiment_name, "model_type": mcfg.get("type")}
                ),
                "stage_name": name,
                "stage_index": idx,
                "model_family": family,
                "model_type": mcfg.get("type"),
                "task_type": task.get("type"),
                "load_from": st.get("load_from"),
                "load_method": str(st.get("load_method") or "checkpoint"),
                "freeze": frz,
                "unfreeze": unf,
                "trainable_param_names": tnames,
                "trainable_param_count": int(trainable),
                "total_param_count": int(total),
                "trainable_ratio": round(ratio, 6),
                "optimizer_lr": opt.get("lr"),
                "epochs": int(st.get("epochs") or 0),
                "save_best_by": save_best_by,
                "save_best_enabled": bool(save_best),
                "metric_original_scale": orig_ok,
                "suspected_full_finetune": suspected_full_ft,
                "recommended_route": route_label,
                "route_ok": len(row_fails) == 0,
                "failures": row_fails,
            }
        )

    return (
        {
            "config_path": str(path.resolve()),
            "experiment_name": experiment_name,
            "model_family": family,
        },
        global_f,
        rows,
    )


def _md_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "experiment",
        "stage",
        "family",
        "task",
        "trainable",
        "total",
        "ratio",
        "lr",
        "ep",
        "save_best_by",
        "route_ok",
        "fails",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        if "error" in r:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("experiment_name", "")),
                        str(r.get("stage_name", "")),
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "ERROR",
                        str(r.get("error", ""))[:80],
                    ]
                )
                + " |"
            )
            continue
        fails = "; ".join(r.get("failures") or [])[:120]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("experiment_name", ""))[:40],
                    str(r.get("stage_name", ""))[:36],
                    str(r.get("model_family", "")),
                    str(r.get("task_type", ""))[:28],
                    str(r.get("trainable_param_count", "")),
                    str(r.get("total_param_count", "")),
                    f'{float(r.get("trainable_ratio") or 0):.4f}',
                    str(r.get("optimizer_lr", "")),
                    str(r.get("epochs", "")),
                    str(r.get("save_best_by", ""))[:20],
                    "OK" if r.get("route_ok") else "FAIL",
                    fails.replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="YAML configs to audit")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    missing: List[str] = []
    failures: List[str] = []

    if not _trainer_reloads_best_before_test():
        failures.append("trainer.py may not reload best-val checkpoint before test")

    for raw in args.configs:
        path = raw if raw.is_absolute() else REPO / raw
        if not path.exists():
            missing.append(str(path))
            print(f"WARNING: missing config {path}", file=sys.stderr)
            continue
        summ, gf, rows = audit_config(path)
        summaries.append(summ)
        all_rows.extend(rows)
        failures.extend(gf)

    out_payload = {
        "summaries": summaries,
        "rows": all_rows,
        "missing_configs": missing,
        "global_failures": sorted(set(failures)),
        "trainer_best_val_reload_ok": _trainer_reloads_best_before_test(),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    md = "# Few-shot trainability audit\n\n"
    md += _md_table(all_rows) + "\n\n"
    names = sorted({r.get("display_name") for r in all_rows if r.get("display_name")})
    if names:
        md += "## Model display names\n" + "\n".join(f"- {n}" for n in names) + "\n\n"
    if missing:
        md += "## Missing configs\n" + "\n".join(f"- `{m}`" for m in missing) + "\n\n"
    if failures:
        md += "## Failures\n" + "\n".join(f"- {f}" for f in failures) + "\n\n"

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")

    print(md)
    route_bad = any(r.get("route_ok") is False for r in all_rows)
    build_bad = any("error" in r for r in all_rows)
    if failures or route_bad or build_bad:
        print("FAIL: see route_ok=false rows, build errors, or failures list", file=sys.stderr)
        return 1
    print("PASS: few-shot route audit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
