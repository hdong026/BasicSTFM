#!/usr/bin/env python3
"""Generate DPM-SR++ ZED (zero-shot expert diffusion) budget configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs/budget_matched/dpm_srpp_dsd_few_shot_monash15_then_mixed_12.yaml"

ZED_MODEL = {
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
ZED_UNFREEZE_FS = [
    "zed_router.*",
    "zed_route_gate.*",
    "zed_fewshot_adapter.*",
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


def _apply_zed_to_cfg(cfg: dict, *, protocol: str, exp_name: str, work_dir: str, fewshot: bool) -> dict:
    c = deepcopy(cfg)
    c["experiment_name"] = exp_name
    c["experiment"] = {"protocol": protocol}
    c["trainer"]["work_dir"] = work_dir
    c["model"] = deepcopy(ZED_MODEL)
    if fewshot:
        c["model"]["stage2"]["enable_fewshot_adapter"] = True

    stages = c["pipeline"]["stages"]
    if not fewshot:
        stages = _strip_fewshot_stages(stages)
    c["pipeline"]["stages"] = stages

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
        elif fewshot and "mechanism_tuning" in nm and st.get("few_shot_ratio") is not None:
            _patch_unfreeze(st, ZED_UNFREEZE_FS)

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
        "# DPM_SRPP_ZED — few-shot secondary: load ZED checkpoint; tune router + gate (+ optional zed_fewshot_adapter); stable trunk frozen.\n\n"
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
