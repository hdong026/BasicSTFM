#!/usr/bin/env python3
"""Audit DPM-SR budget few-shot stages: Stage-II-only unfreeze, loss weights, save-best -> test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[2]

TRAINER_PY = REPO / "src/basicstfm/engines/trainer.py"

EXPECTED_UNFREEZE: Set[str] = {
    "residual_event_encoder.*",
    "diffusion_mechanism_learner.*",
    "fusion_predictor.*",
    "calibration_head.*",
}


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _few_shot_mechanism_stages(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in (cfg.get("pipeline") or {}).get("stages") or []:
        name = str(st.get("name") or "")
        if "mechanism_tuning" not in name:
            continue
        fs = st.get("few_shot_ratio")
        if fs is None:
            continue
        if float(fs) not in (0.05, 0.1):
            continue
        if st.get("eval_only"):
            continue
        out.append(st)
    return out


def _trainer_documents_best_reload() -> bool:
    if not TRAINER_PY.is_file():
        return False
    text = TRAINER_PY.read_text(encoding="utf-8")
    return "Reloading best validation checkpoint for test" in text and "stage.save_best" in text


def audit(path: Path) -> Tuple[bool, List[str]]:
    fails: List[str] = []
    cfg = _load(path)
    m = cfg.get("model") or {}
    if str(m.get("type") or "") != "SRDSTFMBackbone":
        fails.append("root model.type should be SRDSTFMBackbone for this audit")

    few = _few_shot_mechanism_stages(cfg)
    if not few:
        fails.append("no few-shot mechanism_tuning stages found")
        return False, fails

    phases = {str((s.get("task") or {}).get("phase") or "") for s in few}
    if len(phases) != 1:
        fails.append(f"inconsistent task.phase across few-shot stages: {sorted(phases)}")
        phase: str | None = None
    else:
        phase = next(iter(phases))

    for st in few:
        name = st.get("name", "<unnamed>")
        frz = list(st.get("freeze") or [])
        if frz != ["all"]:
            fails.append(f"{name}: freeze must be [all], got {frz!r}")

        unf = {str(u) for u in (st.get("unfreeze") or [])}
        if unf != EXPECTED_UNFREEZE:
            fails.append(f"{name}: unfreeze must equal {sorted(EXPECTED_UNFREEZE)}, got {sorted(unf)}")

        for p in unf:
            if "stable_trunk" in p:
                fails.append(f"{name}: unfreeze must not include stable_trunk*, got {p!r}")

        if not st.get("save_best"):
            fails.append(f"{name}: save_best must be true")
        if str(st.get("save_best_by") or "") != "val/metric/mae":
            fails.append(f"{name}: save_best_by must be val/metric/mae, got {st.get('save_best_by')!r}")

        tk = st.get("task") or {}
        if str(tk.get("type") or "") != "StableResidualForecastingTask":
            fails.append(f"{name}: task.type must be StableResidualForecastingTask")

        if phase is None:
            continue
        if phase == "joint":
            if float(tk.get("stable_weight", -1)) != 0.0:
                fails.append(f"{name}: task.stable_weight must be 0.0 for joint Stage-II-only FS")
            if float(tk.get("spectral_weight", -1)) != 0.0:
                fails.append(f"{name}: task.spectral_weight must be 0.0")
            if float(tk.get("cross_cov_weight", -1)) != 0.0:
                fails.append(f"{name}: task.cross_cov_weight must be 0.0")
            if float(tk.get("residual_weight", -1)) != 0.1:
                fails.append(f"{name}: task.residual_weight must be 0.1, got {tk.get('residual_weight')!r}")
            if str(tk.get("model_mode") or "") != "forecast":
                fails.append(f"{name}: task.model_mode must be forecast for joint few-shot")
        elif phase == "diffusion":
            if float(tk.get("stable_weight", -1)) != 0.0:
                fails.append(f"{name}: task.stable_weight must be 0.0 (documented; phase=diffusion ignores)")
            if float(tk.get("spectral_weight", -1)) != 0.0:
                fails.append(f"{name}: task.spectral_weight must be 0.0")
            if float(tk.get("cross_cov_weight", -1)) != 0.0:
                fails.append(f"{name}: task.cross_cov_weight must be 0.0")
            if str(tk.get("model_mode") or "") != "diffusion_pretrain":
                fails.append(
                    f"{name}: residual-phase ablation expects model_mode=diffusion_pretrain, "
                    f"got {tk.get('model_mode')!r}"
                )
        else:
            fails.append(f"{name}: unsupported few-shot task.phase {phase!r} (expected joint or diffusion)")

        opt = st.get("optimizer") or {}
        lr = float(opt.get("lr", 0.0))
        if lr not in (0.0003, 0.0004, 0.0005):
            fails.append(f"{name}: optimizer.lr expected 3e-4–5e-4 for few-shot, got {lr}")

    if not _trainer_documents_best_reload():
        fails.append(
            "trainer.py does not appear to reload best-val weights before test; "
            f"check {TRAINER_PY.relative_to(REPO)}"
        )

    return not fails, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO / "configs/budget_matched/dpm_sr_monash15_then_mixed_12_basicts_budget_stage2_only_fs.yaml",
        help="Path to DPM-SR budget YAML",
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

    ok, fails = audit(path)
    print(f"audit_dpm_fewshot_route: {'PASS' if ok else 'FAIL'} ({path})")
    for line in fails:
        print(f"  - {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
