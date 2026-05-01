#!/usr/bin/env python3
"""One-shot generator for configs/budget_matched/*_basicts_budget.yaml.

Run from repo root:
  python scripts/analysis/materialize_budget_matched_basicts_configs.py

Re-run after editing templates in configs/basicts/ or configs/monash/.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "configs" / "budget_matched"

FEWSHOT_EPOCHS = 20
MONASH_STEPS = 10_000
MONASH_EPOCHS = 4
MIXED_STEPS = 10_000

# Only these task classes accept ``use_revin`` in ``__init__``.
_TASK_TYPES_WITH_REVIN = frozenset({"ForecastingTask", "StableResidualForecastingTask"})


def _task_set_use_revin_false(task: Dict[str, Any]) -> None:
    if task.get("type") in _TASK_TYPES_WITH_REVIN:
        task["use_revin"] = False


def _dump(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        cfg,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    path.write_text(text, encoding="utf-8")


def _patch_opencity() -> None:
    src = REPO / "configs/basicts/opencity_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment_name"] = "opencity_monash15_then_mixed_12_basicts_budget"
    cfg["trainer"]["work_dir"] = "runs/opencity_monash15_then_mixed_12_basicts_budget"
    header = (
        "# Budget-matched BasicTS traffic benchmark (6 targets). See audit_training_budget.py.\n"
        "# Monash: 40k updates | Mixed: 100k (10×10k) | Few-shot: 20 epochs @ 5%%/10%% | best-val MAE -> test.\n\n"
    )
    stages = cfg["pipeline"]["stages"]
    for st in stages:
        name = st.get("name", "")
        data = st.get("data") or {}
        if st.get("few_shot_ratio") is not None or "finetune" in name:
            st["epochs"] = FEWSHOT_EPOCHS
            if st.get("scheduler"):
                st["scheduler"]["T_max"] = FEWSHOT_EPOCHS
            st["save_best_by"] = "val/metric/mae"
            st["save_best"] = True
        if data.get("type") == "WindowDataModule" and data.get("dataset_key") in {
            "METR-LA",
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "PEMS-BAY",
        }:
            data.pop("max_test_windows", None)
            data.pop("max_val_windows", None)
        if st.get("task"):
            _task_set_use_revin_false(st["task"])
    out = OUT_DIR / "opencity_monash15_then_mixed_12_basicts_budget.yaml"
    _dump(cfg, out)
    content = header + out.read_text(encoding="utf-8")
    out.write_text(content, encoding="utf-8")


def _patch_dpm() -> None:
    src = REPO / "configs/monash/dpm_sr_monash15_then_traffic_sharded_transfer_12_basicts_protocol.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment_name"] = "dpm_sr_monash15_then_mixed_12_basicts_budget"
    cfg["trainer"]["work_dir"] = "runs/dpm_sr_monash15_then_mixed_12_basicts_budget"
    # Drop non-traffic registry entries used only for removed stages
    for key in ("Electricity", "ETTh2"):
        cfg["dataset_registry"].pop(key, None)

    stable_template = yaml.safe_load(
        (REPO / "configs/monash/dpm_sr_monash15_stable_pretrain_12.yaml").read_text(encoding="utf-8")
    )
    for key, val in stable_template.get("dataset_registry", {}).items():
        cfg.setdefault("dataset_registry", {})[key] = deepcopy(val)
    cfg.setdefault("dataset_groups", {})["monash15_sources"] = deepcopy(
        stable_template["dataset_groups"]["monash15_sources"]
    )
    monash_stage = deepcopy(stable_template["pipeline"]["stages"][0])
    monash_stage["name"] = "monash_temporal_warmup"
    monash_stage["save_artifact"] = "dpm_budget_monash_warmup"
    monash_stage["epochs"] = MONASH_EPOCHS
    monash_stage["data"]["steps_per_epoch"] = MONASH_STEPS
    monash_stage["scheduler"]["T_max"] = MONASH_EPOCHS

    # Remove extra benchmark stages (etth2 / weather / electricity probes)
    stages: List[Dict[str, Any]] = cfg["pipeline"]["stages"]
    filtered: List[Dict[str, Any]] = []
    for st in stages:
        n = st["name"]
        if n.startswith("etth2_") or n.startswith("weather_tgt_") or n.startswith("electricity_"):
            continue
        filtered.append(st)
    cfg["pipeline"]["stages"] = filtered

    # Insert monash warmup and retarget stable trunk load
    xd_stages = cfg["pipeline"]["stages"]
    xd_stable = xd_stages[0]
    xd_stable["load_from"] = "${env:MONASH_STAGE0_CKPT,dpm_budget_monash_warmup}"
    xd_stable["load_method"] = "stable_trunk_channel_inflate"
    cfg["pipeline"]["stages"] = [monash_stage] + xd_stages

    # Mixed-domain budgets (40k + 40k + 20k)
    for st in cfg["pipeline"]["stages"]:
        name = st.get("name", "")
        if name == "cross_domain_stable_trunk_pretraining":
            st["epochs"] = 4
            st["data"]["steps_per_epoch"] = 10_000
            if st.get("scheduler"):
                st["scheduler"]["T_max"] = 4
        elif name == "cross_domain_residual_diffusion_pretraining":
            st["epochs"] = 4
            st["data"]["steps_per_epoch"] = 10_000
            if st.get("scheduler"):
                st["scheduler"]["T_max"] = 4
        elif name == "cross_domain_joint_refinement":
            st["epochs"] = 2
            st["data"]["steps_per_epoch"] = 10_000
            if st.get("scheduler"):
                st["scheduler"]["T_max"] = 2
        data = st.get("data") or {}
        if data.get("type") == "WindowDataModule" and data.get("dataset_key") in {
            "METR-LA",
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "PEMS-BAY",
        }:
            data.pop("max_test_windows", None)
            data.pop("max_val_windows", None)
        if "mechanism_tuning" in name:
            st["epochs"] = FEWSHOT_EPOCHS
            if st.get("scheduler"):
                st["scheduler"]["T_max"] = FEWSHOT_EPOCHS
            st["save_best_by"] = "val/metric/mae"
            st["save_best"] = True
        if st.get("task"):
            _task_set_use_revin_false(st["task"])

    header = (
        "# DPM-SR budget-matched BasicTS traffic (6 targets). Monash warmup in-run (40k).\n"
        "# Mixed: 40k stable + 40k diffusion + 20k joint = 100k. Few-shot: 20 epochs, best-val -> test.\n\n"
    )
    out = OUT_DIR / "dpm_sr_monash15_then_mixed_12_basicts_budget.yaml"
    _dump(cfg, out)
    out.write_text(header + out.read_text(encoding="utf-8"), encoding="utf-8")


def _patch_factost() -> None:
    src = REPO / "configs/basicts/factost_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment_name"] = "factost_monash15_then_mixed_12_basicts_budget"
    cfg["trainer"]["work_dir"] = "runs/factost_monash15_then_mixed_12_basicts_budget"

    stages: List[Dict[str, Any]] = cfg["pipeline"]["stages"]
    utp_pre = None
    mix_idx = -1
    for i, st in enumerate(stages):
        if st["name"] == "mixed_domain_utp_pretrain":
            utp_pre = deepcopy(st)
            mix_idx = i
            break
    if utp_pre is None or mix_idx < 0:
        raise RuntimeError("mixed_domain_utp_pretrain not found")

    utp_pre["epochs"] = 5
    utp_pre["data"]["steps_per_epoch"] = MIXED_STEPS
    utp_pre["scheduler"]["T_max"] = 5
    utp_pre["save_artifact"] = "factost_xd_utp_trunk"

    sta = deepcopy(utp_pre)
    sta["name"] = "mixed_domain_sta_source_adaptation"
    sta["save_artifact"] = "factost_xd_sta_source"
    sta["load_from"] = "${env:FACTOST_UTP_CKPT,factost_xd_utp_trunk}"
    sta["load_method"] = "foundation_channel_inflate"
    sta["strict_load"] = False
    sta["epochs"] = 5
    sta["data"]["steps_per_epoch"] = MIXED_STEPS
    sta["scheduler"]["T_max"] = 5
    sta.setdefault("model", {})["use_st_adapter"] = True

    new_stages = stages[:mix_idx] + [utp_pre, sta] + stages[mix_idx + 1 :]
    for st in new_stages:
        if st["name"] == "monash_temporal_warmup":
            st["epochs"] = MONASH_EPOCHS
            st["data"]["steps_per_epoch"] = MONASH_STEPS
            st["scheduler"]["T_max"] = MONASH_EPOCHS

    for st in new_stages:
        n = st.get("name", "")
        lf = st.get("load_from")
        if (
            isinstance(lf, str)
            and not lf.lstrip().startswith("${")
            and ("factost_xd_mixed" in lf or lf.strip() == "factost_xd_utp_trunk")
        ):
            st["load_from"] = "factost_xd_sta_source"
        if "adapter_tune" in n or "five_percent" in n or "ten_percent" in n:
            st["epochs"] = FEWSHOT_EPOCHS
            st["scheduler"]["T_max"] = FEWSHOT_EPOCHS
            st["save_best_by"] = "val/metric/mae"
            st["save_best"] = True
        if n.endswith("_zero_shot"):
            st.setdefault("model", {})["use_st_adapter"] = True
        data = st.get("data") or {}
        if data.get("type") == "WindowDataModule" and data.get("dataset_key") in {
            "METR-LA",
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "PEMS-BAY",
        }:
            data.pop("max_test_windows", None)
            data.pop("max_val_windows", None)
        if st.get("task"):
            _task_set_use_revin_false(st["task"])

    cfg["pipeline"]["stages"] = new_stages
    header = (
        "# FactoST budget-matched: UTP 50k + STA(source adapter) 50k; zero-shot loads STA artifact.\n\n"
    )
    out = OUT_DIR / "factost_monash15_then_mixed_12_basicts_budget.yaml"
    _dump(cfg, out)
    out.write_text(header + out.read_text(encoding="utf-8"), encoding="utf-8")


def _patch_unist() -> None:
    src = REPO / "configs/basicts/unist_monash15_then_mixed_sharded_transfer_12_basicts_protocol.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment_name"] = "unist_monash15_then_mixed_12_basicts_budget"
    cfg["trainer"]["work_dir"] = "runs/unist_monash15_then_mixed_12_basicts_budget"

    stages: List[Dict[str, Any]] = cfg["pipeline"]["stages"]
    mixed_idx = next(i for i, st in enumerate(stages) if st["name"] == "mixed_domain_pretrain")
    mixed = deepcopy(stages[mixed_idx])
    mixed["name"] = "mixed_domain_masked_pretrain"
    mixed["save_artifact"] = "unist_xd_masked_trunk"
    mixed["epochs"] = 5
    mixed["data"]["steps_per_epoch"] = MIXED_STEPS
    mixed["scheduler"]["T_max"] = 5
    mixed.setdefault("model", {})["use_prompt"] = False

    prompt_adapt = deepcopy(mixed)
    prompt_adapt["name"] = "mixed_domain_prompt_source_adaptation"
    prompt_adapt["save_artifact"] = "unist_xd_prompt_source"
    prompt_adapt["load_from"] = "${env:UNIST_MASKED_CKPT,unist_xd_masked_trunk}"
    prompt_adapt["load_method"] = "foundation_channel_inflate"
    prompt_adapt["strict_load"] = False
    prompt_adapt["epochs"] = 5
    prompt_adapt["data"]["steps_per_epoch"] = MIXED_STEPS
    prompt_adapt["scheduler"]["T_max"] = 5
    prompt_adapt.setdefault("model", {})["use_prompt"] = True

    new_stages = stages[:mixed_idx] + [mixed, prompt_adapt] + stages[mixed_idx + 1 :]

    for st in new_stages:
        if st["name"] == "monash_temporal_warmup":
            st["epochs"] = MONASH_EPOCHS
            st["data"]["steps_per_epoch"] = MONASH_STEPS
            st["scheduler"]["T_max"] = MONASH_EPOCHS
        n = st.get("name", "")
        lf = st.get("load_from")
        if isinstance(lf, str) and not lf.lstrip().startswith("${"):
            if lf in ("unist_xd_mixed", "unist_xd_masked_trunk"):
                st["load_from"] = "unist_xd_prompt_source"
        if "zero_shot" in n:
            st.setdefault("model", {})["use_prompt"] = True
        if "prompt_tune" in n:
            st["epochs"] = FEWSHOT_EPOCHS
            st["scheduler"]["T_max"] = FEWSHOT_EPOCHS
            st["save_best_by"] = "val/metric/mae"
            st["save_best"] = True
            st["load_from"] = "unist_xd_prompt_source"
        data = st.get("data") or {}
        if data.get("type") == "WindowDataModule" and data.get("dataset_key") in {
            "METR-LA",
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "PEMS-BAY",
        }:
            data.pop("max_test_windows", None)
            data.pop("max_val_windows", None)
        if st.get("task"):
            _task_set_use_revin_false(st["task"])

    cfg["pipeline"]["stages"] = new_stages
    header = (
        "# UniST budget-matched: masked 50k + prompt(source) 50k; zero-shot loads prompt artifact.\n\n"
    )
    out = OUT_DIR / "unist_monash15_then_mixed_12_basicts_budget.yaml"
    _dump(cfg, out)
    out.write_text(header + out.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _patch_dpm()
    _patch_opencity()
    _patch_factost()
    _patch_unist()
    print(f"Wrote 4 configs under {OUT_DIR}")


if __name__ == "__main__":
    main()
