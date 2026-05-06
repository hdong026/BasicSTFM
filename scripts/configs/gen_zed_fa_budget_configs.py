#!/usr/bin/env python3
"""Generate ZED Factorized Adaptation (FA) budget configs.

Outputs:
  - One YAML with pretrain + zero-shot + **both** 5%% and 10%% mechanism stages (single command).
  - Optional split YAMLs: 5%%-only and 10%%-only (smaller runs / A-B dirs).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/budget_matched/dpm_srpp_zed_few_shot_monash15_then_mixed_12.yaml"

FEW_SHOT_FA = {
    "load_from_zero_shot": True,
    "adaptation_mode": "factorized_adapter",
    "enable_factorized_adaptation": True,
    "train_target_prompt": True,
    "target_prompt_dim": 64,
    "max_target_prompt_datasets": 64,
    "target_spatial_adapter": True,
    "spatial_adapter_bottleneck_ratio": 8,
    "spatial_adapter_scale": 0.25,
    "affine_calibration": True,
    "calibration": {
        "mode": "node_channel",
        "init_alpha": 1.0,
        "init_beta": 0.0,
        "node_channel_param_cap": 4096,
    },
    "router_offset": True,
    "adaptation_mode": "router_gate_adapter",
    "train_router": True,
    "train_fusion_gate": True,
    "train_router_feat_adapter": True,
    "train_target_adapter": True,
    "train_full_target_diffusion": False,
    "few_shot_mode": "calibration",
    "anchor_to_zero_shot": True,
    "zed_fa_loss": True,
    "loss": {
        "lambda_final": 1.0,
        "lambda_residual": 0.1,
        "lambda_anchor": 0.05,
        "lambda_adapter_reg": 1.0e-4,
    },
    "lambda_gate": 0.0,
    "few_shot_epochs": 5,
    "few_shot_lr": 1.0e-4,
}

UNFREEZE_FA = [
    "zed_router.*",
    "zed_route_gate.*",
    "zed_router_feat_adapter.*",
    "zed_fewshot_adapter.*",
    "zed_fa_prompt_embed.*",
    "zed_fa_prompt_proj.*",
    "zed_fa_router_bias",
    "zed_fa_spatial_adapter.*",
    "zed_fa_affine.*",
]


def _filter_stages(stages: list, keep: str) -> list:
    out = []
    for s in stages:
        nm = str(s.get("name", ""))
        if keep == "5":
            if "_ten_percent_" in nm:
                continue
        else:
            if "_five_percent_" in nm:
                continue
        out.append(deepcopy(s))
    return out


def _patch_mechanism_epochs(stage: dict) -> None:
    if "mechanism_tuning" not in str(stage.get("name", "")):
        return
    if stage.get("few_shot_ratio") is None:
        return
    ep = int(FEW_SHOT_FA.get("few_shot_epochs", 5))
    stage["epochs"] = ep
    stage["early_stop_patience"] = 2
    stage["gradient_clip_val"] = 5.0
    stage["unfreeze"] = list(UNFREEZE_FA)
    opt = stage.get("optimizer")
    if isinstance(opt, dict):
        opt["lr"] = float(FEW_SHOT_FA["few_shot_lr"])
        opt["weight_decay"] = 1.0e-4
    sch = stage.get("scheduler")
    if isinstance(sch, dict) and sch.get("type") == "CosineAnnealingLR":
        sch["T_max"] = max(1, ep)


def build(*, exp: str, work: str, frac: Optional[str]) -> dict:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    cfg["experiment_name"] = exp
    cfg["trainer"]["work_dir"] = work
    cfg["model"]["name"] = "DPM_SRPP_ZED_FA"
    cfg["model"]["few_shot"] = deepcopy(FEW_SHOT_FA)
    cfg["model"]["stage2"]["enable_fewshot_adapter"] = True
    if frac is None:
        st = [deepcopy(s) for s in cfg["pipeline"]["stages"]]
    else:
        st = _filter_stages(cfg["pipeline"]["stages"], frac)
    for stage in st:
        _patch_mechanism_epochs(stage)
    cfg["pipeline"]["stages"] = st
    return cfg


def main() -> None:
    out_dir = ROOT / "configs/budget_matched"
    h = (
        "# ZED-FA (Factorized Adaptation): few-shot target prompt + spatial adapter + affine calib.\n"
        "# Load ZS/joint weights via stage load_from / ZED_ZERO_SHOT_CKPT; FA modules init fresh.\n"
        "# Effective freeze: stable_trunk, source experts (adapters+shared diffusion), event encoder\n"
        "# stay frozen (freeze: [all] minus FA/router/gate/router_feat_adapter/fewshot_adapter).\n\n"
    )
    h_full = (
        h
        + "# --- ALL-IN-ONE ---\n"
        + "# Monash warmup -> XD pretrain/joint -> traffic zero-shot evals -> 5%% + 10%% FA mechanism.\n"
        + "# Single command: basicstfm train configs/budget_matched/dpm_srpp_zed_fa_few_shot_monash15_then_mixed_12.yaml\n\n"
    )
    c_all = build(
        exp="dpm_srpp_zed_fa_few_shot_monash15_then_mixed_12",
        work="runs/dpm_srpp_zed_fa_few_shot_monash15_then_mixed_12",
        frac=None,
    )
    c5 = build(exp="dpm_srpp_zed_fa_fs5", work="runs/dpm_srpp_zed_fa_fs5", frac="5")
    c10 = build(exp="dpm_srpp_zed_fa_fs10", work="runs/dpm_srpp_zed_fa_fs10", frac="10")
    p_all = out_dir / "dpm_srpp_zed_fa_few_shot_monash15_then_mixed_12.yaml"
    p5 = out_dir / "dpm_srpp_zed_fa_few_shot_5_monash15_then_mixed_12.yaml"
    p10 = out_dir / "dpm_srpp_zed_fa_few_shot_10_monash15_then_mixed_12.yaml"
    p_all.write_text(h_full + yaml.dump(c_all, sort_keys=False, allow_unicode=True), encoding="utf-8")
    p5.write_text(h + yaml.dump(c5, sort_keys=False, allow_unicode=True), encoding="utf-8")
    p10.write_text(h + yaml.dump(c10, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {p_all}\nWrote {p5}\nWrote {p10}")


if __name__ == "__main__":
    main()
