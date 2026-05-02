#!/usr/bin/env python3
"""Audit UniST budget-matched YAML: MaskedForecastCompletion on source + targets, no forecast_head unfreeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]

TRAFFIC = frozenset(
    {"METR-LA", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS-BAY"}
)


def _load_from_refs_unist_prompt_source(lf: Any) -> bool:
    if lf is None:
        return False
    if lf == "unist_xd_prompt_source":
        return True
    if isinstance(lf, str) and "unist_xd_prompt_source" in lf:
        return True
    return False


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_target_window(stage: Dict[str, Any]) -> bool:
    data = stage.get("data") or {}
    return data.get("type") == "WindowDataModule" and data.get("dataset_key") in TRAFFIC


def audit(path: Path) -> Tuple[bool, List[str]]:
    fails: List[str] = []
    cfg = _load(path)
    stages = (cfg.get("pipeline") or {}).get("stages") or []

    for st in stages:
        name = st.get("name", "")
        task = (st.get("task") or {}).get("type")

        if name == "mixed_domain_prompt_source_adaptation":
            if task != "MaskedForecastCompletionTask":
                fails.append(f"{name}: expected MaskedForecastCompletionTask, got {task!r}")

        lm = st.get("load_method")
        if name in {"mixed_domain_masked_pretrain", "mixed_domain_prompt_source_adaptation"}:
            if lm != "foundation_channel_inflate":
                fails.append(f"{name}: load_method must be foundation_channel_inflate, got {lm!r}")

        data = st.get("data") or {}
        if _is_target_window(st):
            if task != "MaskedForecastCompletionTask":
                fails.append(
                    f"{name}: BasicTS traffic stage must use MaskedForecastCompletionTask "
                    f"(not ForecastingTask), got {task!r}"
                )
            if lm is not None and lm not in ("checkpoint",):
                fails.append(
                    f"{name}: traffic load_from prior stage must use load_method unset or "
                    f"'checkpoint' (not foundation inflate), got {lm!r}"
                )
            if "zero_shot" in name:
                lf_zs = st.get("load_from")
                if not _load_from_refs_unist_prompt_source(lf_zs):
                    fails.append(
                        f"{name}: load_from must reference unist_xd_prompt_source, got {lf_zs!r}"
                    )
            if "five_percent" in name or "ten_percent" in name:
                unf = st.get("unfreeze") or []
                joined = " ".join(str(u) for u in unf)
                if "forecast_head" in joined:
                    fails.append(f"{name}: unfreeze must not include forecast_head.*")
                if not any(str(u).startswith("prompt.") for u in unf):
                    fails.append(f"{name}: unfreeze must include prompt.*")
                if "reconstruction_head" not in joined:
                    fails.append(f"{name}: unfreeze must include reconstruction_head.*")
            scaler = data.get("scaler") or {}
            if isinstance(scaler, dict) and scaler.get("type") != "standard":
                fails.append(f"{name}: scaler.type must be standard")
            if data.get("split_mode") != "basicts":
                fails.append(f"{name}: split_mode must be basicts")
            tk = st.get("task") or {}
            mod = st.get("model") or {}
            if tk.get("use_revin") or mod.get("use_revin"):
                fails.append(f"{name}: use_revin must be false/absent for BasicTS traffic stages")

    ok = not fails
    return ok, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO / "configs/budget_matched/unist_monash15_then_mixed_12_basicts_budget.yaml",
        help="Path to unist budget YAML",
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
    print(f"audit_unist_route: {'PASS' if ok else 'FAIL'} ({path})")
    for line in fails:
        print(f"  FAIL: {line}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
