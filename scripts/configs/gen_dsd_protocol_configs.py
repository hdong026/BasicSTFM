#!/usr/bin/env python3
"""Generate DPM-SR++ DSD protocol YAMLs (zero_shot / few_shot / full_shot) from the stage2-only FS template."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

SRC_DEFAULT = (
    "configs/budget_matched/dpm_srpp_dsd_monash15_then_mixed_12_basicts_budget_stage2_only_fs.yaml"
)

HEAD_FEW = """# DPM_SRPP_DSD — few_shot protocol (default main experiment): Stage-I unchanged; Stage-II per-dataset adapters.
# experiment.protocol drives logging / result labels; pipeline matches this protocol."""

HEAD_ZERO = """# DPM_SRPP_DSD — zero_shot: no target-domain adaptation stages (eval only on targets with shared/fallback heads)."""

HEAD_FULL = """# DPM_SRPP_DSD — full_shot: supervised upper bound — full BasicTS train split for dataset-specific heads."""


def inject_meta(cfg: dict, protocol: str, exp_name: str, work_dir: str) -> dict:
    c = deepcopy(cfg)
    out: dict = {}
    for k, v in c.items():
        if k == "experiment_name":
            out[k] = exp_name
            out["experiment"] = {"protocol": protocol}
        elif k == "trainer":
            nv = deepcopy(v)
            nv["work_dir"] = work_dir
            out[k] = nv
        elif k == "model":
            mo = deepcopy(v)
            typ = mo.get("type")
            mo.pop("name", None)
            new_m: dict = {"type": typ, "name": "DPM_SRPP_DSD"}
            for kk, vv in mo.items():
                if kk != "type":
                    new_m[kk] = vv
            out[k] = new_m
        else:
            out[k] = v
    return out


def strip_fewshot_stages(cfg: dict) -> dict:
    c = deepcopy(cfg)
    c["pipeline"]["stages"] = [
        s
        for s in c["pipeline"]["stages"]
        if "_five_percent_" not in s["name"] and "_ten_percent_" not in s["name"]
    ]
    return c


def merge_fullshot_stages(cfg: dict) -> dict:
    c = deepcopy(cfg)
    stages = c["pipeline"]["stages"]
    out: list = []
    i = 0
    while i < len(stages):
        s = stages[i]
        nm = s["name"]
        if nm.endswith("_five_percent_mechanism_tuning"):
            ten = stages[i + 1]
            if not str(ten["name"]).endswith("_ten_percent_mechanism_tuning"):
                raise AssertionError(f"Expected ten-percent stage after {nm!r}, got {ten['name']!r}")
            prefix = nm[: -len("_five_percent_mechanism_tuning")]
            fs = deepcopy(s)
            fs["name"] = f"{prefix}_full_shot_adaptation"
            fs.pop("few_shot_ratio", None)
            fs.pop("train_fraction", None)
            fs["epochs"] = 30
            sched = fs.get("scheduler")
            if isinstance(sched, dict) and sched.get("type") == "CosineAnnealingLR":
                sched["T_max"] = 30
            out.append(fs)
            i += 2
        else:
            out.append(s)
            i += 1
    c["pipeline"]["stages"] = out
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out-dir", default="configs/budget_matched")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    src = root / args.src
    out_dir = root / args.out_dir
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    specs = [
        (
            "dpm_srpp_dsd_few_shot_monash15_then_mixed_12.yaml",
            HEAD_FEW,
            inject_meta(cfg, "few_shot", "dpm_srpp_dsd_few_shot_monash15_then_mixed_12", "runs/dpm_srpp_dsd_few_shot_monash15_then_mixed_12"),
        ),
        (
            "dpm_srpp_dsd_zero_shot_monash15_then_mixed_12.yaml",
            HEAD_ZERO,
            strip_fewshot_stages(
                inject_meta(
                    cfg,
                    "zero_shot",
                    "dpm_srpp_dsd_zero_shot_monash15_then_mixed_12",
                    "runs/dpm_srpp_dsd_zero_shot_monash15_then_mixed_12",
                )
            ),
        ),
        (
            "dpm_srpp_dsd_full_shot_monash15_then_mixed_12.yaml",
            HEAD_FULL,
            merge_fullshot_stages(
                inject_meta(
                    cfg,
                    "full_shot",
                    "dpm_srpp_dsd_full_shot_monash15_then_mixed_12",
                    "runs/dpm_srpp_dsd_full_shot_monash15_then_mixed_12",
                )
            ),
        ),
    ]
    for fname, head, body in specs:
        path = out_dir / fname
        dumped = yaml.dump(body, default_flow_style=False, sort_keys=False, allow_unicode=True)
        path.write_text(head + "\n\n" + dumped, encoding="utf-8")
        print("Wrote", path.relative_to(root))


if __name__ == "__main__":
    main()
