"""Structured result discovery and export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
    data_path = data_cfg.get("data_path")
    if not data_path:
        return None
    return Path(str(data_path)).parent.name or None


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
