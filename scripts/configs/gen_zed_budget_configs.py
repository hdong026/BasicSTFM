#!/usr/bin/env python3
"""Generate DPM-SR++ ZED (zero-shot expert diffusion) budget configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs/budget_matched/dpm_srpp_dsd_few_shot_monash15_then_mixed_12.yaml"

ZED_MODEL_BASE = {
    "type": "SRDSTFMBackboneZED",
    "name": "DPM_SRPP_ZED",
    "num_nodes": "auto",
    "input_dim": "auto",
    "output_dim": "auto",
    "input_len": "auto",
    "output_len": "auto",
    "hidden_dim": 128,
    "residual_mode": "forecast",
    "fusion_mode": "additive",
    "use_frequency_branch": True,
    "num_datasets": 5,
    "diffusion_enabled": True,
    "use_inertia_gate": True,
    "use_attenuation_gate": True,
    "use_persistence_anchor": True,
    "stage1": {
        "freeze": True,
        "load_checkpoint": "${env:ZED_STAGE1_CKPT,optional_path_to_stage1.pt}",
    },
    "stage2": {
        "expert_bank": True,
        "expert_source": "source_domains",
        "expert_domains": [
            "LargeST_xd_part0",
            "LargeST_xd_part1",
            "KnowAir",
            "Weather",
            "ETTm1",
        ],
        "routing_key": "auto",
        "head_mode": "adapter",
        "num_experts": "auto",
        "share_event_abstraction": True,
        "share_diffusion_base": True,
        "zero_shot_router": True,
        "router_inputs": [
            "graph_signature",
            "temporal_statistics",
            "residual_event_signature",
        ],
        "router_type": "softmax_mlp",
        "top_k_experts": 3,
        "per_dataset_fusion_gate": False,
        "router_fusion_gate": True,
        "fallback_head": "mixture",
        "fusion_gate_init": 0.1,
        "adapter_bottleneck": 32,
        "adapter_scale": 0.25,
        "enable_fewshot_adapter": False,
    },
}

ZED_FEW_SHOT_MODEL = {
    "load_from_zero_shot": True,
    "adaptation_mode": "router_gate_adapter",
    "freeze_stable_trunk": True,
    "freeze_source_experts": True,
    "train_router": True,
    "train_fusion_gate": True,
    "train_target_adapter": True,
    "train_full_target_diffusion": False,
    "few_shot_mode": "calibration",
    "train_router_feat_adapter": True,
    "anchor_to_zero_shot": True,
    "lambda_anchor": 0.1,
    "lambda_gate": 0.01,
    "few_shot_epochs": 3,
    "few_shot_lr": 1.0e-4,
}

# Mechanism stages: default loads same-run joint checkpoint (one yaml = one work_dir, no second exp.).
# Override with ZED_ZERO_SHOT_CKPT=/path/to.pt when you only run few-shot stages or want an external ZS bundle.
ZED_MECHANISM_LOAD_FROM = "${env:ZED_ZERO_SHOT_CKPT,srd_xd_joint}"

ZED_UNFREEZE_DIFFUSION = [
    "residual_event_encoder.*",
    "diffusion_mechanism_learner.*",
    "zed_router.*",
    "zed_route_gate.*",
    "zed_expert_adapters.*",
]
ZED_UNFREEZE_JOINT = ZED_UNFREEZE_DIFFUSION + [
    "fusion_predictor.*",
    "calibration_head.*",
]

ZED_UNFREEZE_FS_FULL_TARGET = [
    "residual_event_encoder.*",
    "diffusion_mechanism_learner.*",
    "zed_expert_adapters.*",
    "fusion_predictor.*",
    "calibration_head.*",
]


def _strip_fewshot_stages(stages: list) -> list:
    return [
        s
        for s in stages
        if "_five_percent_" not in s["name"] and "_ten_percent_" not in s["name"]
    ]


def _patch_unfreeze(stage: dict, patterns: list) -> None:
    stage["unfreeze"] = list(patterns)


def _fewshot_unfreeze_patterns(fs: Dict[str, Any]) -> list[str]:
    pats: list[str] = []
    if fs.get("train_router", True):
        pats.append("zed_router.*")
    if fs.get("train_fusion_gate", True):
        pats.append("zed_route_gate.*")
    if fs.get("train_router_feat_adapter", True):
        pats.append("zed_router_feat_adapter.*")
    if fs.get("train_target_adapter", True):
        pats.append("zed_fewshot_adapter.*")
    if fs.get("few_shot_mode") == "train_target_head" or fs.get("train_full_target_diffusion"):
        pats.extend(list(ZED_UNFREEZE_FS_FULL_TARGET))
    return pats or ["zed_router.*"]


def _patch_zed_mechanism_stage(st: dict, fs: Dict[str, Any]) -> None:
    nm = str(st.get("name", ""))
    if "mechanism_tuning" not in nm:
        return
    if st.get("few_shot_ratio") is None:
        return
    st["load_from"] = ZED_MECHANISM_LOAD_FROM
    st["strict_load"] = False
    epochs = int(fs.get("few_shot_epochs", 3))
    st["epochs"] = epochs
    opt = st.get("optimizer")
    if isinstance(opt, dict):
        opt["lr"] = float(fs.get("few_shot_lr", 1.0e-4))
    sch = st.get("scheduler")
    if isinstance(sch, dict) and sch.get("type") == "CosineAnnealingLR":
        sch["T_max"] = max(1, epochs)
    _patch_unfreeze(st, _fewshot_unfreeze_patterns(fs))


def _apply_zed_to_cfg(cfg: dict, *, protocol: str, exp_name: str, work_dir: str, fewshot: bool) -> dict:
    c = deepcopy(cfg)
    c["experiment_name"] = exp_name
    c["experiment"] = {"protocol": protocol}
    c["trainer"]["work_dir"] = work_dir
    c["model"] = deepcopy(ZED_MODEL_BASE)
    if fewshot:
        c["model"]["stage2"]["enable_fewshot_adapter"] = True
        c["model"]["few_shot"] = deepcopy(ZED_FEW_SHOT_MODEL)

    stages = c["pipeline"]["stages"]
    if not fewshot:
        stages = _strip_fewshot_stages(stages)
    c["pipeline"]["stages"] = stages

    fs_model: Dict[str, Any] = (c["model"].get("few_shot") or {}) if fewshot else {}

    for st in c["pipeline"]["stages"]:
        nm = str(st.get("name", ""))
        task = st.get("task")
        if isinstance(task, dict):
            task["log_zed_metrics"] = True
            task["log_stage2_per_domain"] = True
            task["dsd_routing_key"] = "auto"
        if nm == "cross_domain_residual_diffusion_pretraining":
            _patch_unfreeze(st, ZED_UNFREEZE_DIFFUSION)
        elif nm == "cross_domain_joint_refinement":
            _patch_unfreeze(st, ZED_UNFREEZE_JOINT)
        elif fewshot:
            _patch_zed_mechanism_stage(st, fs_model)

    return c


def main() -> None:
    base = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    out_dir = ROOT / "configs/budget_matched"

    z0 = _apply_zed_to_cfg(
        base,
        protocol="zero_shot",
        exp_name="dpm_srpp_zed_zero_shot_monash15_then_mixed_12",
        work_dir="runs/dpm_srpp_zed_zero_shot_monash15_then_mixed_12",
        fewshot=False,
    )
    z1 = _apply_zed_to_cfg(
        base,
        protocol="few_shot",
        exp_name="dpm_srpp_zed_few_shot_monash15_then_mixed_12",
        work_dir="runs/dpm_srpp_zed_few_shot_monash15_then_mixed_12",
        fewshot=True,
    )

    h0 = (
        "# DPM_SRPP_ZED — main experiment: source-only expert bank + router; zero-shot on traffic targets (no target-head training).\n"
        "# Experts align with cross_domain_sharded_sources; routing uses graph + temporal + residual signatures (no target labels).\n\n"
    )
    h1 = (
        "# DPM_SRPP_ZED — full pipeline in one work_dir: pretrain → traffic zero-shot evals → 5%%/10%% few-shot.\n"
        "# Few-shot mechanism default load_from: srd_xd_joint (same run). Set env ZED_ZERO_SHOT_CKPT only for an external .pt.\n"
        "# Calibration: router + fusion gate + router feat adapter + zed_fewshot_adapter (see model.few_shot).\n\n"
    )

    (out_dir / "dpm_srpp_zed_zero_shot_monash15_then_mixed_12.yaml").write_text(
        h0 + yaml.dump(z0, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out_dir / "dpm_srpp_zed_few_shot_monash15_then_mixed_12.yaml").write_text(
        h1 + yaml.dump(z1, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("Wrote ZED configs to", out_dir)


if __name__ == "__main__":
    main()
