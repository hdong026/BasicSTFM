"""Structured result discovery, summarization, and benchmark visualization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def discover_stage_result_files(roots: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file() and root_path.name == "stage_results.json":
            files.append(root_path)
            continue
        files.extend(sorted(root_path.rglob("stage_results.json")))
    return sorted(set(files))


def load_stage_result_payload(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload in {path!r}")
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"Result payload {path!r} does not contain a 'stages' list")
    return payload


def flatten_stage_results(
    payload: Dict[str, Any],
    source_path: str | Path,
) -> List[Dict[str, Any]]:
    source_path = Path(source_path)
    rows: List[Dict[str, Any]] = []
    experiment_name = payload.get("experiment_name")
    work_dir = payload.get("work_dir")
    for stage in payload.get("stages", []):
        row: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "work_dir": work_dir,
            "source_path": str(source_path),
            "stage_name": stage.get("name"),
            "stage_index": stage.get("stage_index"),
            "task": stage.get("task"),
            "model_type": stage.get("model_type"),
            "data_type": stage.get("data_type"),
            "eval_only": stage.get("eval_only"),
            "load_from": stage.get("load_from"),
            "load_method": stage.get("load_method"),
            "save_artifact": stage.get("save_artifact"),
            "train_fraction": stage.get("train_fraction"),
            "train_windows": stage.get("train_windows"),
            "checkpoint": stage.get("checkpoint"),
            "dataset": _infer_dataset_name(stage.get("resolved_data")),
            "dataset_names": _infer_dataset_names(stage.get("resolved_data")),
            "data_path": _read_nested(stage, "resolved_data", "data_path"),
            "graph_path": _read_nested(stage, "resolved_data", "graph_path"),
            "input_len": _read_nested(stage, "resolved_data", "input_len"),
            "output_len": _read_nested(stage, "resolved_data", "output_len")
            or _read_nested(stage, "resolved_data", "target_len"),
            "batch_size": _read_nested(stage, "resolved_data", "batch_size"),
        }
        _merge_metric_block(row, stage.get("train"), prefix="train")
        _merge_metric_block(row, stage.get("val"), prefix="val")
        _merge_metric_block(row, stage.get("test"), prefix="test")
        rows.append(row)
    return rows


def filter_stage_rows(
    rows: Sequence[Dict[str, Any]],
    stages: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    experiment: Optional[str] = None,
) -> List[Dict[str, Any]]:
    stage_set = set(stages or [])
    filtered = []
    for row in rows:
        if stage_set and row.get("stage_name") not in stage_set:
            continue
        if dataset and row.get("dataset") != dataset:
            continue
        if experiment and row.get("experiment_name") != experiment:
            continue
        filtered.append(dict(row))
    return filtered


def summarize_stage_rows(
    rows: Sequence[Dict[str, Any]],
    split: str = "test",
    metrics: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    metric_columns = [normalize_metric_name(metric, split=split) for metric in (metrics or ("metric/mae", "metric/rmse"))]
    summary: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "experiment_name": row.get("experiment_name"),
            "stage_name": row.get("stage_name"),
            "dataset": row.get("dataset"),
            "model_type": row.get("model_type"),
            "task": row.get("task"),
            "train_fraction": row.get("train_fraction"),
            "checkpoint": row.get("checkpoint"),
        }
        for metric in metric_columns:
            item[metric] = row.get(metric)
        summary.append(item)
    return summary


def infer_stage_regime(row: Mapping[str, Any]) -> str:
    stage_name = str(row.get("stage_name") or "").lower()
    train_fraction = row.get("train_fraction")
    eval_only = bool(row.get("eval_only"))

    if eval_only or "zero_shot" in stage_name:
        return "zero_shot"
    if train_fraction not in (None, "", 0, 0.0):
        return "few_shot"
    if any(
        token in stage_name
        for token in (
            "few_shot",
            "prompt_tuning",
            "adapter",
            "head_tuning",
            "finetune",
            "fine_tune",
        )
    ):
        return "few_shot"
    return "pretrain"


def pretty_model_name(row: Mapping[str, Any]) -> str:
    experiment_name_raw = str(row.get("experiment_name") or "")
    experiment_name = experiment_name_raw.lower()
    # Presets from scripts/run_dpm_zs_fs_from_checkpoints.py (one row per ckpt; must not collapse).
    zs_fs_ckpt_eval = {
        "dpm_orig_stable": "DPM-original (stable ckpt)",
        "dpm_orig_diffusion": "DPM-original (diffusion ckpt)",
        "dpm_v2_stable": "DPM-v2 (stable ckpt)",
        "dpm_v2_diffusion": "DPM-v2 (diffusion ckpt)",
        "dpm_v3_stable": "DPM-v3 (stable ckpt)",
        "dpm_v3_diffusion": "DPM-v3 (diffusion ckpt)",
    }
    if experiment_name_raw in zs_fs_ckpt_eval:
        return zs_fs_ckpt_eval[experiment_name_raw]
    experiment_mapping = {
        "opencity_traffic_benchmark": "OpenCity",
        "opencity_largest_transfer": "OpenCity-LargeST",
        "opencity_traffic_benchmark_single_target": "OpenCity-1Ch",
        "opencity_largest_transfer_single_target": "OpenCity-LargeST-1Ch",
        "srd_stfm_foundation_transfer": "DPM-STFM",
        "srd_stfm_joint_from_scratch": "DPM-Scratch",
        "srd_stfm_ablation_stable_only": "DPM-StableOnly",
        "srd_stfm_ablation_no_diffusion": "DPM-NoDiffusion",
        "srd_stfm_ablation_no_disentangle": "DPM-NoDisentangle",
        # Cross-domain (XD) pretrain recipes under configs/cross_domain and foundation
        "dpm_cross_domain_pretrain_transfer": "DPM-v3 (XD)",
        "dpm_v3_cross_domain_pretrain_transfer": "DPM-v3 (XD)",
        "dpm_v2_cross_domain_pretrain_transfer": "DPM-v2 (XD)",
        "dpm_stfm_cross_domain_pretrain_transfer": "DPM-SR (XD)",
        "opencity_cross_domain_pretrain_transfer": "OpenCity (XD)",
        "factost_cross_domain_pretrain_transfer": "FactoST (XD)",
        "unist_cross_domain_pretrain_transfer": "UniST (XD)",
        # Sharded cross-domain transfer (configs/cross_domain/*_sharded_transfer.yaml)
        "dpm_v2_cross_domain_sharded_transfer": "DPM-v2 (XD)",
        "dpm_v3_cross_domain_sharded_transfer": "DPM-v3 (XD)",
        "dpm_stfm_cross_domain_sharded_transfer": "DPM-SR (XD)",
        "dpm_stfm_v4_cross_domain_e2e_transfer": "DPM-SR v4 (XD E2E)",
        "opencity_cross_domain_sharded_transfer": "OpenCity (XD)",
        "factost_cross_domain_sharded_transfer": "FactoST (XD)",
        "unist_cross_domain_sharded_transfer": "UniST (XD)",
    }
    if experiment_name in experiment_mapping:
        return experiment_mapping[experiment_name]

    model_type = str(row.get("model_type") or "")
    mapping = {
        "OpenCityFoundationModel": "OpenCity",
        "FactoSTFoundationModel": "FactoST",
        "UniSTFoundationModel": "UniST",
        "SRDSTFMBackbone": "DPM-STFM",
        "DPMV2Backbone": "DPM-STFM-v2",
        "DPMV3Backbone": "DPM-STFM-v3",
        "DPMSTFMBackbone": "DPM-STFM",
        "MADSTFMBackbone": "DPM-STFM",
    }
    if model_type in mapping:
        return mapping[model_type]

    if "opencity" in experiment_name:
        return "OpenCity"
    if "factost" in experiment_name:
        return "FactoST"
    if "unist" in experiment_name:
        return "UniST"
    return model_type or "Unknown"


def build_paper_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    split: str = "test",
    metric: str = "metric/mae",
    datasets: Optional[Sequence[str]] = None,
    model_order: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    metric_key = normalize_metric_name(metric, split=split)
    filtered = [dict(row) for row in rows if infer_stage_regime(row) in {"zero_shot", "few_shot"}]
    if datasets is None:
        dataset_names = []
        for row in filtered:
            dataset = row.get("dataset")
            if dataset and dataset not in dataset_names:
                dataset_names.append(str(dataset))
    else:
        dataset_names = [str(item) for item in datasets]

    model_names = [pretty_model_name(row) for row in filtered]
    if model_order is None:
        ordered_models = []
        for name in model_names:
            if name not in ordered_models:
                ordered_models.append(name)
    else:
        ordered_models = [str(item) for item in model_order if str(item) in model_names]
        for name in model_names:
            if name not in ordered_models:
                ordered_models.append(name)

    value_map: Dict[Tuple[str, str, str], Optional[float]] = {}
    for row in filtered:
        model = pretty_model_name(row)
        dataset = row.get("dataset")
        if dataset not in dataset_names:
            continue
        regime = infer_stage_regime(row)
        value = row.get(metric_key)
        value_map[(model, str(dataset), regime)] = None if value is None else float(value)

    summary: List[Dict[str, Any]] = []
    for model in ordered_models:
        item: Dict[str, Any] = {"Model": model}
        zs_values: List[float] = []
        fs_values: List[float] = []
        gains: List[float] = []
        for dataset in dataset_names:
            zs = value_map.get((model, dataset, "zero_shot"))
            fs = value_map.get((model, dataset, "few_shot"))
            gain = None if zs is None or fs is None else float(zs) - float(fs)
            item[f"{dataset} ZS"] = zs
            item[f"{dataset} FS"] = fs
            item[f"{dataset} Gain"] = gain
            if zs is not None:
                zs_values.append(float(zs))
            if fs is not None:
                fs_values.append(float(fs))
            if gain is not None:
                gains.append(float(gain))
        item["Avg ZS"] = None if not zs_values else sum(zs_values) / len(zs_values)
        item["Avg FS"] = None if not fs_values else sum(fs_values) / len(fs_values)
        item["Avg Gain"] = None if not gains else sum(gains) / len(gains)
        summary.append(item)
    return dataset_names, summary


def normalize_metric_name(metric: str, split: str = "test") -> str:
    metric = str(metric)
    if metric.startswith(("train/", "val/", "test/")):
        return metric
    return f"{split}/{metric.lstrip('/')}"


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = list(fieldnames or [])
    elif fieldnames is None:
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames or []))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify_csv_value(row.get(key)) for key in writer.fieldnames})


def build_markdown_table(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    if not rows:
        columns = list(columns or [])
        if not columns:
            return ""
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        return "\n".join((header, divider))

    columns = list(columns or _ordered_fieldnames(rows))
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_escape_markdown(row.get(column)) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def write_markdown(path: str | Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_table(rows, columns=columns), encoding="utf-8")


def _merge_metric_block(row: Dict[str, Any], block: Any, prefix: str) -> None:
    if not isinstance(block, dict):
        return
    for key, value in block.items():
        normalized = key if str(key).startswith(f"{prefix}/") else f"{prefix}/{key}"
        row[normalized] = value


def _infer_dataset_name(data_cfg: Any) -> Optional[str]:
    if not isinstance(data_cfg, dict):
        return None
    datasets = _infer_dataset_names(data_cfg)
    if datasets:
        return "+".join(datasets)
    data_path = data_cfg.get("data_path")
    if not data_path:
        return None
    return Path(str(data_path)).parent.name or None


def _infer_dataset_names(data_cfg: Any) -> Optional[List[str]]:
    if not isinstance(data_cfg, dict):
        return None
    datasets = data_cfg.get("datasets")
    if isinstance(datasets, list):
        names = []
        for item in datasets:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name:
                names.append(str(name))
                continue
            data_path = item.get("data_path")
            if data_path:
                names.append(Path(str(data_path)).parent.name or Path(str(data_path)).stem)
        return names or None
    dataset_names = data_cfg.get("dataset_names")
    if isinstance(dataset_names, list):
        return [str(item) for item in dataset_names]
    return None


def _read_nested(mapping: Dict[str, Any], root: str, key: str) -> Any:
    payload = mapping.get(root)
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _ordered_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "experiment_name",
        "stage_name",
        "dataset",
        "model_type",
        "task",
        "train_fraction",
        "checkpoint",
    ]
    seen = []
    for key in preferred:
        if any(key in row for row in rows):
            seen.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _stringify_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _escape_markdown(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
