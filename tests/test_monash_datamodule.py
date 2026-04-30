"""Smoke tests for Monash series window datamodules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from basicstfm.builders import import_builtin_components
from basicstfm.registry import DATAMODULES


def _write_toy_monash(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.0, 9.9, 500, dtype=np.float32)
    lengths = np.array([240, 260], dtype=np.int64)
    offsets = np.array([0, 240], dtype=np.int64)
    np.save(dest / "values.npy", values)
    np.save(dest / "offsets.npy", offsets)
    np.save(dest / "lengths.npy", lengths)
    (dest / "meta.json").write_text(
        json.dumps({"dataset_name": dest.name, "n_series": 2, "total_timesteps": 500}), encoding="utf-8"
    )


def test_monash_multi_setup_and_collate(tmp_path):
    import_builtin_components()
    ds_a = tmp_path / "ds_a"
    ds_b = tmp_path / "ds_b"
    _write_toy_monash(ds_a)
    _write_toy_monash(ds_b)

    cfg = {
        "type": "MonashMultiDatasetWindowDataModule",
        "datasets": [
            {"name": "ds_a", "monash_root": str(ds_a), "batch_size": 8},
            {"name": "ds_b", "monash_root": str(ds_b), "batch_size": 8},
        ],
        "input_len": 12,
        "output_len": 8,
        "split": [0.7, 0.1, 0.2],
        "stride": 2,
        "norm_mode": "global_standard",
        "steps_per_epoch": 4,
    }
    dm = DATAMODULES.build(cfg)
    dm.setup()
    train = dm.train_dataloader()
    batch = next(iter(train))

    assert batch["x"].shape[1:] == (12, 1, 1)
    assert batch["y"].shape[1:] == (8, 1, 1)
    assert "dataset_index" in batch
    assert batch["graph"].shape == (1, 1)
    assert isinstance(batch["dataset_index"], torch.Tensor)

    scaler = dm.get_scaler()
    assert scaler is not None


def test_monash_series_norm_instance(tmp_path):
    import_builtin_components()
    root = tmp_path / "one"
    _write_toy_monash(root)

    dm = DATAMODULES.build(
        {
            "type": "MonashSeriesWindowDataModule",
            "monash_root": str(root),
            "input_len": 24,
            "output_len": 16,
            "norm_mode": "instance",
            "batch_size": 4,
            "stride": 1,
            "split": [0.7, 0.1, 0.2],
        }
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert torch.isfinite(batch["x"]).all()
    assert torch.isfinite(batch["y"]).all()


def test_monash_series_norm_instance_standard(tmp_path):
    import_builtin_components()
    root = tmp_path / "one"
    _write_toy_monash(root)

    dm = DATAMODULES.build(
        {
            "type": "MonashSeriesWindowDataModule",
            "monash_root": str(root),
            "input_len": 24,
            "output_len": 16,
            "norm_mode": "instance_standard",
            "scaler": {"type": "identity"},
            "batch_size": 4,
            "stride": 1,
            "split": [0.7, 0.1, 0.2],
        }
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x0 = batch["x"][0].numpy().reshape(-1)
    assert np.isfinite(x0).all()
    assert abs(float(np.mean(x0))) < 0.05
    assert 0.8 < float(np.std(x0)) < 1.2
