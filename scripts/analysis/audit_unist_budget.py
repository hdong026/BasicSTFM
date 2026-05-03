#!/usr/bin/env python3
"""Audit UniST-lite budget YAML: fair memory, 80k/20k mixed updates, aligned zero-shot, prompt-only few-shot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]

MASK_STAGE = "mixed_domain_masked_pretrain"
PROMPT_STAGE = "mixed_domain_prompt_source_adaptation"
PROMPT_ARTIFACT = "unist_xd_prompt_source"
EXP_LITE = "unist_monash15_then_mixed_12_basicts_budget_lite"
EXP_LITE_HEAD = "unist_monash15_then_mixed_12_basicts_budget_lite_head"


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_unist(cfg: Dict[str, Any]) -> bool:
    m = cfg.get("model") or {}
    return str(m.get("type") or "") == "UniSTFoundationModel"


def _stage_updates(st: Dict[str, Any]) -> int:
    data = st.get("data") or {}
    steps = int(data.get("steps_per_epoch") or 0)
    epochs = int(st.get("epochs") or 0)
    return steps * epochs


def _is_target_window(st: Dict[str, Any]) -> bool:
    data = st.get("data") or {}
    return data.get("type") == "WindowDataModule"


def _is_zero_shot(st: Dict[str, Any]) -> bool:
    return "zero_shot" in str(st.get("name") or "") and _is_target_window(st)


def _is_few_shot(st: Dict[str, Any]) -> bool:
    if "zero_shot" in str(st.get("name") or ""):
        return False
    if not _is_target_window(st):
        return False
    if st.get("few_shot_ratio") is not None:
        return True
    n = str(st.get("name") or "")
    return "five_percent" in n or "ten_percent" in n


def _unfreeze_list(st: Dict[str, Any]) -> List[str]:
    u = st.get("unfreeze") or []
    if isinstance(u, str):
        return [u]
    return [str(x) for x in u]


def _forbidden_tuning_patterns(unfreeze: Sequence[str]) -> List[str]:
    """Weights that should not be tuned in fair UniST-lite few-shot (main table)."""
    bad: List[str] = []
    joined = " ".join(unfreeze).lower()
    for token in ("forecast_head", "encoder", "decoder", "backbone"):
        if any(token in str(p).lower() for p in unfreeze):
            bad.append(token)
    return bad


def audit(path: Path) -> Tuple[bool, List[str]]:
    fails: List[str] = []
    cfg = _load(path)
    is_head = path.name.endswith("_lite_head.yaml") or str(
        cfg.get("experiment_name") or ""
    ).endswith("_lite_head")

    if not _is_unist(cfg):
        fails.append("root model.type must be UniSTFoundationModel")

    model = cfg.get("model") or {}
    if int(model.get("num_memory_spatial") or -1) != 128:
        fails.append(
            f"model.num_memory_spatial must be 128, got {model.get('num_memory_spatial')!r}"
        )
    if int(model.get("num_memory_temporal") or -1) != 128:
        fails.append(
            f"model.num_memory_temporal must be 128, got {model.get('num_memory_temporal')!r}"
        )

    stages = (cfg.get("pipeline") or {}).get("stages") or []
    mask_st = None
    prompt_st = None
    for st in stages:
        n = str(st.get("name") or "")
        if n == MASK_STAGE:
            mask_st = st
        if n == PROMPT_STAGE:
            prompt_st = st

    if mask_st is None:
        fails.append(f"missing stage {MASK_STAGE}")
    else:
        if int(mask_st.get("epochs") or 0) != 8:
            fails.append(f"{MASK_STAGE}: expected epochs=8, got {mask_st.get('epochs')!r}")
        if _stage_updates(mask_st) != 80_000:
            fails.append(
                f"{MASK_STAGE}: expected 80k updates (steps_per_epoch*epochs), got {_stage_updates(mask_st)}"
            )
        tk = (mask_st.get("task") or {}).get("type")
        if tk != "MaskedReconstructionTask":
            fails.append(f"{MASK_STAGE}: task.type must be MaskedReconstructionTask, got {tk!r}")
        mod = mask_st.get("model") or {}
        if mod.get("use_prompt") is not False:
            fails.append(f"{MASK_STAGE}: model.use_prompt must be false, got {mod.get('use_prompt')!r}")

    if prompt_st is None:
        fails.append(f"missing stage {PROMPT_STAGE}")
    else:
        if int(prompt_st.get("epochs") or 0) != 2:
            fails.append(f"{PROMPT_STAGE}: expected epochs=2, got {prompt_st.get('epochs')!r}")
        if _stage_updates(prompt_st) != 20_000:
            fails.append(
                f"{PROMPT_STAGE}: expected 20k updates, got {_stage_updates(prompt_st)}"
            )
        tk = (prompt_st.get("task") or {}).get("type")
        if tk != "MaskedForecastCompletionTask":
            fails.append(f"{PROMPT_STAGE}: task.type must be MaskedForecastCompletionTask, got {tk!r}")
        mod = prompt_st.get("model") or {}
        if mod.get("use_prompt") is not True:
            fails.append(f"{PROMPT_STAGE}: model.use_prompt must be true, got {mod.get('use_prompt')!r}")

    mixed_total = 0
    if mask_st is not None:
        mixed_total += _stage_updates(mask_st)
    if prompt_st is not None:
        mixed_total += _stage_updates(prompt_st)
    if mixed_total != 100_000 and mask_st and prompt_st:
        fails.append(f"mixed-domain total updates must be 100k, got {mixed_total}")

    for st in stages:
        name = str(st.get("name") or "")
        if _is_zero_shot(st):
            tk = st.get("task") or {}
            if tk.get("type") != "MaskedForecastCompletionTask":
                fails.append(f"{name}: zero-shot task.type must be MaskedForecastCompletionTask")
            if str(tk.get("output_key") or "") != "reconstruction":
                fails.append(f"{name}: task.output_key must be reconstruction")
            if str(tk.get("model_mode") or "") != "reconstruct":
                fails.append(f"{name}: task.model_mode must be reconstruct")
            lf = str(st.get("load_from") or "")
            if PROMPT_ARTIFACT not in lf:
                fails.append(f"{name}: load_from must reference {PROMPT_ARTIFACT}, got {lf!r}")
            mod = st.get("model") or {}
            if mod.get("use_prompt") is not True:
                fails.append(f"{name}: model.use_prompt must be true")
            data = st.get("data") or {}
            if str(data.get("scaler", {}).get("type") or "") != "standard":
                fails.append(f"{name}: data.scaler.type must be standard")
            if data.get("split_mode") != "basicts":
                fails.append(f"{name}: data.split_mode must be basicts")
            if data.get("max_test_windows") is not None:
                fails.append(f"{name}: must not set max_test_windows (full BasicTS test)")
            if int(data.get("input_len") or 0) != 12 or int(data.get("output_len") or 0) != 12:
                fails.append(f"{name}: input_len/output_len must be 12/12")

        if _is_few_shot(st):
            unf = _unfreeze_list(st)
            has_prompt = any("prompt" in p.lower() for p in unf)
            if not has_prompt:
                fails.append(f"{name}: few-shot unfreeze must include prompt.*")
            if not is_head:
                if any("reconstruction_head" in p.lower() for p in unf):
                    fails.append(
                        f"{name}: UniST-lite main few-shot must not unfreeze reconstruction_head.*"
                    )
            else:
                if not any("reconstruction_head" in p.lower() for p in unf):
                    fails.append(
                        f"{name}: UniST-lite+Head few-shot must unfreeze reconstruction_head.*"
                    )
            bad = _forbidden_tuning_patterns(unf)
            if bad:
                fails.append(f"{name}: few-shot must not unfreeze {bad}")

            data = st.get("data") or {}
            if str(data.get("scaler", {}).get("type") or "") != "standard":
                fails.append(f"{name}: data.scaler.type must be standard")
            if data.get("split_mode") != "basicts":
                fails.append(f"{name}: data.split_mode must be basicts")
            if data.get("max_test_windows") is not None:
                fails.append(f"{name}: must not set max_test_windows")
            opt = st.get("optimizer") or {}
            lr = float(opt.get("lr") or 0)
            if abs(lr - 3e-4) > 1e-9:
                fails.append(f"{name}: few-shot optimizer.lr must be 3e-4, got {opt.get('lr')!r}")

    exp = str(cfg.get("experiment_name") or "")
    if not is_head and exp != EXP_LITE:
        fails.append(
            f"experiment_name should be {EXP_LITE!r} for main lite config, got {exp!r}"
        )
    if is_head and exp != EXP_LITE_HEAD:
        fails.append(
            f"experiment_name should be {EXP_LITE_HEAD!r} for head ablation, got {exp!r}"
        )

    ok = not fails
    return ok, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO / f"configs/budget_matched/{EXP_LITE}.yaml",
        help="Path to UniST-lite budget YAML",
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
    print(f"audit_unist_budget: {'PASS' if ok else 'FAIL'} ({path})")
    for line in fails:
        print(f"  - {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
