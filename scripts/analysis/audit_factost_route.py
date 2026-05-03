#!/usr/bin/env python3
"""Audit FactoST budget-matched YAML: STA trains ForecastingTask; targets load full STA checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]

STA_ARTIFACT = "factost_xd_sta_source"
UTP_TRUNK = "factost_xd_utp_trunk"
MONASH_WARMUP = "factost_monash15_warmup"
STA_STAGE = "mixed_domain_sta_source_adaptation"

_ALLOWED_ZERO_SHOT_LOAD = frozenset({None, "checkpoint", "state_dict"})
_FORBIDDEN_LOAD = "load_backbone_weights"


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_factost(cfg: Dict[str, Any]) -> bool:
    m = cfg.get("model") or {}
    return str(m.get("type") or "") == "FactoSTFoundationModel"


def _load_from_str(load_from: Any) -> str:
    if load_from is None:
        return ""
    return str(load_from)


def _env_default_contains(pattern: str, load_from: Any) -> bool:
    s = _load_from_str(load_from)
    for m in re.finditer(r"\$\{env:[^,]+,([^}]*)\}", s):
        if pattern in m.group(1):
            return True
    return False


def _references_sta_artifact(load_from: Any) -> bool:
    s = _load_from_str(load_from)
    if STA_ARTIFACT in s:
        return True
    return _env_default_contains(STA_ARTIFACT, load_from)


def _is_utp_only_checkpoint_ref(load_from: Any) -> bool:
    s = _load_from_str(load_from)
    if not s:
        return False
    risky = UTP_TRUNK in s or MONASH_WARMUP in s or _env_default_contains(UTP_TRUNK, load_from)
    if not risky:
        return False
    return STA_ARTIFACT not in s and not _env_default_contains(STA_ARTIFACT, load_from)


def _is_factost_zero_shot_stage(st: Dict[str, Any]) -> bool:
    name = str(st.get("name") or "")
    if "zero_shot" not in name:
        return False
    data = st.get("data") or {}
    return data.get("type") == "WindowDataModule"


def _is_factost_few_shot_stage(st: Dict[str, Any]) -> bool:
    name = str(st.get("name") or "")
    if "zero_shot" in name:
        return False
    data = st.get("data") or {}
    if data.get("type") != "WindowDataModule":
        return False
    if st.get("few_shot_ratio") is not None:
        return True
    return "five_percent" in name or "ten_percent" in name


def _unfreeze_has_required(unfreeze: Sequence[Any]) -> Tuple[bool, List[str]]:
    """Require explicit glob entries covering forecast head, metadata gate, ST adapter, prompt adapter."""
    patterns = [str(p) for p in unfreeze]

    def has(pat_substr: str) -> bool:
        return any(pat_substr in p for p in patterns)

    missing: List[str] = []
    if not has("forecast_head"):
        missing.append("unfreeze must include forecast_head.* (or forecast_head*)")
    if not has("metadata_gate"):
        missing.append("unfreeze must include metadata_gate.* (or metadata_gate*)")
    if not any(p == "st_*" or p.startswith("st_*") for p in patterns):
        missing.append("unfreeze must include st_*")
    if not has("prompt_adapter"):
        missing.append("unfreeze must include prompt_adapter.*")
    if not has("prompt_"):
        missing.append("unfreeze must include prompt_* (prompt_u / prompt_v tokens)")
    return not missing, missing


def audit(path: Path) -> Tuple[bool, List[str]]:
    fails: List[str] = []
    cfg = _load(path)
    if not _is_factost(cfg):
        fails.append("root model.type must be FactoSTFoundationModel for this audit")

    stages = (cfg.get("pipeline") or {}).get("stages") or []

    sta_seen = False
    for st in stages:
        name = str(st.get("name") or "")
        task = (st.get("task") or {}).get("type")

        if name == STA_STAGE:
            sta_seen = True
            if task == "FactoSTUTPTask":
                fails.append(f"{name}: must not use FactoSTUTPTask")
            if task != "ForecastingTask":
                fails.append(f"{name}: expected ForecastingTask, got {task!r}")
            tk = st.get("task") or {}
            if str(tk.get("model_mode") or "") != "forecast":
                fails.append(f"{name}: task.model_mode must be 'forecast', got {tk.get('model_mode')!r}")
            mod = st.get("model") or {}
            if mod.get("use_st_adapter") is not True:
                fails.append(f"{name}: model.use_st_adapter must be true")
            if mod.get("output_dim") != 1:
                fails.append(f"{name}: model.output_dim must be 1, got {mod.get('output_dim')!r}")
            lm = st.get("load_method")
            if lm != "foundation_channel_inflate":
                fails.append(f"{name}: load_method must be foundation_channel_inflate, got {lm!r}")
            if str(st.get("save_artifact") or "") != STA_ARTIFACT:
                fails.append(
                    f"{name}: save_artifact must be {STA_ARTIFACT!r} (STA must be a forecasting checkpoint)"
                )

        if _is_factost_zero_shot_stage(st):
            if task != "ForecastingTask":
                fails.append(f"{name}: FactoST zero-shot must use ForecastingTask, got {task!r}")
            lf = st.get("load_from")
            if not _references_sta_artifact(lf):
                fails.append(
                    f"{name}: load_from must reference {STA_ARTIFACT} (literal or env default), got {lf!r}"
                )
            if _is_utp_only_checkpoint_ref(lf):
                fails.append(
                    f"{name}: load_from must not be UTP/warmup-only without STA, got {lf!r}"
                )
            lm = st.get("load_method")
            if lm == _FORBIDDEN_LOAD:
                fails.append(f"{name}: zero-shot must not use load_method={_FORBIDDEN_LOAD!r}")
            elif lm not in _ALLOWED_ZERO_SHOT_LOAD:
                fails.append(
                    f"{name}: zero-shot load_method must be checkpoint/state_dict (full weights), got {lm!r}"
                )
            if st.get("strict_load") is True:
                fails.append(f"{name}: strict_load should be false for zero-shot (channel/layout tolerance)")
            mod = st.get("model") or {}
            if mod.get("use_st_adapter") is not True:
                fails.append(f"{name}: model.use_st_adapter must be true for zero-shot")

        if _is_factost_few_shot_stage(st):
            lm = st.get("load_method", "checkpoint")
            if lm == _FORBIDDEN_LOAD:
                fails.append(f"{name}: few-shot must not use load_method={_FORBIDDEN_LOAD!r}")
            lf = st.get("load_from")
            if not _references_sta_artifact(lf):
                fails.append(
                    f"{name}: few-shot load_from must reference {STA_ARTIFACT}, got {lf!r}"
                )
            unfreeze = st.get("unfreeze") or []
            ok_u, miss_u = _unfreeze_has_required(unfreeze if isinstance(unfreeze, list) else [unfreeze])
            if not ok_u:
                for m in miss_u:
                    fails.append(f"{name}: {m}")

    if not sta_seen:
        fails.append(f"missing stage {STA_STAGE}")

    ok = not fails
    return ok, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO / "configs/budget_matched/factost_monash15_then_mixed_12_basicts_budget.yaml",
        help="Path to FactoST budget YAML",
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
    print(f"audit_factost_route: {'PASS' if ok else 'FAIL'} ({path})")
    for line in fails:
        print(f"  - {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
