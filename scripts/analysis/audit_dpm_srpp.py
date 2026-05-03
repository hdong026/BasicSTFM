#!/usr/bin/env python3
"""Audit DPM-SR++ budget YAMLs: persistence anchor, residual stable target, Stage-II-only FS, robust flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
TRAINER_PY = REPO / "src/basicstfm/engines/trainer.py"

EXPECTED_FS_UNFREEZE: Set[str] = {
    "residual_event_encoder.*",
    "diffusion_mechanism_learner.*",
    "fusion_predictor.*",
    "calibration_head.*",
}


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fewshot_stages(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in (cfg.get("pipeline") or {}).get("stages") or []:
        if st.get("few_shot_ratio") is None:
            continue
        if "mechanism_tuning" not in str(st.get("name") or ""):
            continue
        if st.get("eval_only"):
            continue
        out.append(st)
    return out


def _trainer_best_reload() -> bool:
    if not TRAINER_PY.is_file():
        return False
    t = TRAINER_PY.read_text(encoding="utf-8")
    return "Reloading best validation checkpoint for test" in t


def audit(path: Path, *, require_persistence: bool = True) -> Tuple[bool, List[str]]:
    fails: List[str] = []
    cfg = _load(path)
    m = cfg.get("model") or {}
    if str(m.get("type") or "") != "SRDSTFMBackbone":
        fails.append("model.type must be SRDSTFMBackbone")

    use_pa = m.get("use_persistence_anchor")
    if require_persistence and use_pa is not True:
        fails.append("model.use_persistence_anchor must be true")
    if (not require_persistence) and use_pa is True:
        fails.append("ablation expects use_persistence_anchor false")

    st_stable = None
    for st in (cfg.get("pipeline") or {}).get("stages") or []:
        if st.get("name") == "cross_domain_stable_trunk_pretraining":
            st_stable = st
            break
    if st_stable is None:
        fails.append("missing cross_domain_stable_trunk_pretraining")
    else:
        tk = st_stable.get("task") or {}
        if tk.get("type") != "StableResidualForecastingTaskV5":
            fails.append("Stage I task should be StableResidualForecastingTaskV5")
        if tk.get("robust_logsumexp_excess") is not True and require_persistence:
            if "no_robust_residual" not in path.name:
                fails.append("Stage I missing robust_logsumexp_excess: true")

    for fs in _fewshot_stages(cfg):
        name = fs.get("name", "")
        frz = list(fs.get("freeze") or [])
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all]")
        unf = {str(u) for u in (fs.get("unfreeze") or [])}
        if "no_stage2_only_fs" in path.name:
            if "stable_trunk.forecast_head.*" not in unf:
                fails.append(f"{name}: ablation no_stage2_only_fs expects stable_trunk.forecast_head.* in unfreeze")
        else:
            if unf != EXPECTED_FS_UNFREEZE:
                fails.append(f"{name}: unfreeze must be Stage-II-only set, got {sorted(unf)}")
            for p in unf:
                if "stable_trunk" in p:
                    fails.append(f"{name}: must not unfreeze stable_trunk*, got {p}")

        tk = fs.get("task") or {}
        if "no_stage2_only_fs" not in path.name:
            if float(tk.get("stable_weight", -1)) != 0.0:
                fails.append(f"{name}: stable_weight must be 0")
            if float(tk.get("spectral_weight", -1)) != 0.0:
                fails.append(f"{name}: spectral_weight must be 0")
        if not fs.get("save_best"):
            fails.append(f"{name}: save_best must be true")

    if not _trainer_best_reload():
        fails.append("trainer must reload best-val ckpt before test")

    return not fails, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO / "configs/budget_matched/dpm_srpp_monash15_then_mixed_12_basicts_budget.yaml",
    )
    parser.add_argument(
        "--no-require-persistence",
        action="store_true",
        help="For dpm_srpp_no_persistence_anchor.yaml",
    )
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO / args.config
    if not path.exists():
        print(f"FAIL: missing {path}", file=sys.stderr)
        return 1
    try:
        import yaml  # noqa: F401
    except ImportError:
        print("FAIL: PyYAML required", file=sys.stderr)
        return 1
    req_p = not args.no_require_persistence and "no_persistence" not in path.name
    ok, fails = audit(path, require_persistence=req_p)
    print(f"audit_dpm_srpp: {'PASS' if ok else 'FAIL'} ({path})")
    for line in fails:
        print(f"  - {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
