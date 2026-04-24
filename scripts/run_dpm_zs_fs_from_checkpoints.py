#!/usr/bin/env python3
"""
从指定 checkpoint 跑与主配置一致的 zero-shot + 5% mechanism tuning（原版 DPM / v2 / v3）。

阶段定义、数据划分、batch、优化器、freeze/unfreeze 等均从对应的
``dpm_*_largest_pretrain_zero_fewshot_single_target.yaml`` 中截取；仅替换各阶段的
``load_from`` 为给定 .pt 路径，并把 ``model.hidden_dim=192``、``stable_mixer_layers=6``
与当前磁盘上的预训练权重对齐（原版 SRDSTFM 与 v2/v3 的 resolved_model 一致）。

示例（在仓库根目录、已 conda activate basicstfm）::

    # 只跑 METR-LA，few-shot 1 个 epoch（冒烟）
    python scripts/run_dpm_zs_fs_from_checkpoints.py --presets all --datasets METR-LA --few-shot-epochs 1

    # 全部预设（6 个：原版 stable/diffusion + v2 + v3），与默认训练一致（耗时长）
    python scripts/run_dpm_zs_fs_from_checkpoints.py --presets all

    # 仅原版 DPM 两个中间 checkpoint
    python scripts/run_dpm_zs_fs_from_checkpoints.py --presets dpm_orig_stable dpm_orig_diffusion

    # 只生成配置不训练
    python scripts/run_dpm_zs_fs_from_checkpoints.py --presets dpm_v2_diffusion --dry-run --dump-config-dir /tmp/dpm_zsfs_yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("需要 PyYAML：pip install pyyaml") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]

V1_YAML = REPO_ROOT / "configs/foundation/dpm_stfm_largest_pretrain_zero_fewshot_single_target.yaml"
V2_YAML = REPO_ROOT / "configs/foundation/dpm_v2_largest_pretrain_zero_fewshot_single_target.yaml"
V3_YAML = REPO_ROOT / "configs/foundation/dpm_v3_largest_pretrain_zero_fewshot_single_target.yaml"

# 与 runs/*/results/stage_results.json 中 resolved_model 一致
MODEL_OVERRIDES = {"hidden_dim": 192, "stable_mixer_layers": 6}

DATASET_TO_PREFIX = {
    "METR-LA": "metr_la_",
    "PEMS-BAY": "pems_bay_",
    "PEMS04": "pems04_",
    "PEMS07": "pems07_",
    "PEMS08": "pems08_",
}

PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "dpm_v2_stable": {
        "yaml": V2_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_v2_largest_transfer_single_target/checkpoints/largest_graph_stable_trunk_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_v2_from_stable_last",
    },
    "dpm_v2_diffusion": {
        "yaml": V2_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_v2_largest_transfer_single_target/checkpoints/largest_graph_residual_diffusion_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_v2_from_diffusion_last",
    },
    "dpm_v3_stable": {
        "yaml": V3_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_v3_largest_transfer_single_target/checkpoints/largest_graph_stable_trunk_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_v3_from_stable_last",
    },
    "dpm_v3_diffusion": {
        "yaml": V3_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_v3_largest_transfer_single_target/checkpoints/largest_graph_residual_diffusion_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_v3_from_diffusion_last",
    },
    "dpm_orig_stable": {
        "yaml": V1_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_largest_transfer_single_target/checkpoints/largest_graph_stable_trunk_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_orig_from_stable_last",
    },
    "dpm_orig_diffusion": {
        "yaml": V1_YAML,
        "checkpoint": REPO_ROOT
        / "runs/dpm_largest_transfer_single_target/checkpoints/largest_graph_residual_diffusion_pretraining_last.pt",
        "work_dir": REPO_ROOT / "runs/zs_fs_eval/dpm_orig_from_diffusion_last",
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _zs_fs_stages(full_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages = full_cfg["pipeline"]["stages"]
    out: List[Dict[str, Any]] = []
    for s in stages:
        name = s.get("name", "")
        if name.endswith("_zero_shot") or name.endswith("_five_percent_mechanism_tuning"):
            out.append(deepcopy(s))
    return out


def _filter_by_datasets(stages: List[Dict[str, Any]], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    prefixes = [DATASET_TO_PREFIX[d] for d in datasets]
    filtered: List[Dict[str, Any]] = []
    for s in stages:
        if any(s["name"].startswith(p) for p in prefixes):
            filtered.append(s)
    return filtered


def _patch_ckpt_and_epochs(
    stages: List[Dict[str, Any]],
    ckpt: Path,
    few_shot_epochs: int,
) -> None:
    ckpt_s = str(ckpt.resolve())
    for s in stages:
        s["load_from"] = ckpt_s
        if s["name"].endswith("_five_percent_mechanism_tuning"):
            s["epochs"] = int(few_shot_epochs)
            sched = s.get("scheduler")
            if isinstance(sched, dict) and "T_max" in sched:
                sched["T_max"] = int(few_shot_epochs)


def build_run_config(
    source_yaml: Path,
    checkpoint: Path,
    experiment_name: str,
    work_dir: Path,
    datasets: Sequence[str] | None,
    few_shot_epochs: int,
) -> Dict[str, Any]:
    base = _load_yaml(source_yaml)
    stages = _zs_fs_stages(base)
    if datasets:
        stages = _filter_by_datasets(stages, datasets)
    if not stages:
        raise ValueError("筛选后没有阶段：请检查 --datasets 是否与 YAML 中一致")

    _patch_ckpt_and_epochs(stages, checkpoint, few_shot_epochs)

    model = deepcopy(base["model"])
    model.update(MODEL_OVERRIDES)

    cfg: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "seed": base.get("seed", 42),
        "trainer": {
            **deepcopy(base["trainer"]),
            "work_dir": str(work_dir),
            "auto_resume": False,
        },
        "dataset_registry": deepcopy(base["dataset_registry"]),
        "data": deepcopy(base["data"]),
        "model": model,
        "pipeline": {"stages": stages},
    }
    return cfg


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从指定 checkpoint 跑 DPM zero/few-shot（与主 YAML 一致）")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["all"],
        help="预设名：dpm_v2_* / dpm_v3_* / dpm_orig_*（原版 SRDSTFM）；或 all（共 6 个）",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(DATASET_TO_PREFIX.keys()),
        help="只跑这些数据集（默认五个全跑）",
    )
    parser.add_argument(
        "--few-shot-epochs",
        type=int,
        default=3,
        help="5%% mechanism tuning 的 epoch 数（与主配置默认 3 一致；可改为 1 缩短时间）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅调用 basicstfm.cli dry-run，不训练",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="只打印将执行的命令，不运行",
    )
    parser.add_argument(
        "--dump-config-dir",
        type=Path,
        default=None,
        help="若指定，则把每个 preset 的完整 YAML 写入该目录（{preset}.yaml），便于检查",
    )
    args = parser.parse_args(argv)

    preset_keys = args.presets
    if preset_keys == ["all"]:
        preset_keys = list(PRESET_SPECS.keys())

    for key in preset_keys:
        if key not in PRESET_SPECS:
            print(f"未知 preset: {key!r}，可选: {', '.join(PRESET_SPECS)}", file=sys.stderr)
            return 2

    for key in preset_keys:
        spec = PRESET_SPECS[key]
        ckpt: Path = spec["checkpoint"]
        if not ckpt.is_file():
            print(f"[跳过] {key}: 找不到 checkpoint {ckpt}", file=sys.stderr)
            continue

        work_dir: Path = spec["work_dir"]
        work_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_run_config(
            spec["yaml"],
            ckpt,
            experiment_name=key,
            work_dir=work_dir,
            datasets=args.datasets,
            few_shot_epochs=args.few_shot_epochs,
        )

        if args.print_commands:
            print(
                f"\n# {key}: work_dir={work_dir} ckpt={ckpt}\n"
                f"cd {REPO_ROOT} && {sys.executable} -m basicstfm.cli train <临时生成的.yaml>"
            )
            continue

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False, allow_unicode=True)
            tmp_path = Path(tmp.name)

        cmd = [
            sys.executable,
            "-m",
            "basicstfm.cli",
            "dry-run" if args.dry_run else "train",
            str(tmp_path),
        ]
        print("\n>>>", " ".join(cmd))

        try:
            if args.dump_config_dir:
                args.dump_config_dir.mkdir(parents=True, exist_ok=True)
                dump_path = args.dump_config_dir / f"{key}.yaml"
                dump_path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"已保存配置 {dump_path}")

            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                print(f"preset {key} 失败，exit {proc.returncode}", file=sys.stderr)
                return proc.returncode
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    print("\n结果 JSON：各 preset 的 work_dir 下 results/stage_results.json")
    print("汇总示例：")
    print(
        "  python scripts/results/export_results.py "
        "--input-roots runs/zs_fs_eval/dpm_v2_from_stable_last "
        "runs/zs_fs_eval/dpm_v2_from_diffusion_last "
        "runs/zs_fs_eval/dpm_v3_from_stable_last "
        "runs/zs_fs_eval/dpm_v3_from_diffusion_last "
        "runs/zs_fs_eval/dpm_orig_from_stable_last "
        "runs/zs_fs_eval/dpm_orig_from_diffusion_last "
        "--split test --metrics test/metric/mae test/metric/rmse --print-table"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
