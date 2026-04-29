"""Monash TSF as univariate windows: tensors [T,1,1], graph diagonal 1×1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from basicstfm.data.datamodule import (
    InterleavedNamedLoaders,
    SequentialNamedLoaders,
    WindowDataModule,
    _sanitize_tensor,
    _split_lengths,
)
from basicstfm.data.scaler import DatasetAwareScaler, build_scaler
from basicstfm.registry import DATAMODULES


NormMode = Literal["global_standard", "per_series_standard", "instance"]


def _resolve_split_edges(length: int, split: Sequence[float]) -> Tuple[int, int, int]:
    tr, va, te = _split_lengths(int(length), split)
    total = tr + va + te
    if length < total:
        raise ValueError(f"series length {length} < split sum {total}")
    te += length - total
    return int(tr), int(va), int(te)


def _finite_mu_sigma(block: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(block, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    mu = float(np.mean(vals))
    sig = float(np.std(vals))
    if sig < 1e-6:
        sig = 1e-6
    return mu, sig


def enumerate_window_placements(
    offsets: np.ndarray,
    lengths: np.ndarray,
    split: Sequence[float],
    input_len: int,
    target_len: int,
    stride: int,
    part: Literal["train", "val", "test"],
) -> List[Tuple[int, int]]:
    wl = int(input_len) + int(target_len)
    stride_v = max(1, int(stride))
    out: List[Tuple[int, int]] = []

    lengths = np.asarray(lengths, dtype=np.int64)
    part_str = str(part)

    for s_idx in range(len(lengths)):
        ln = int(lengths[s_idx])
        tr_len, va_len, te_len = _resolve_split_edges(ln, split)

        if part_str == "train":
            seg_len, seg_start = tr_len, 0
        elif part_str == "val":
            seg_len, seg_start = va_len, tr_len
        else:
            seg_len, seg_start = te_len, tr_len + va_len

        if seg_len < wl:
            continue
        local_last = seg_len - wl
        for ls in range(0, local_last + 1, stride_v):
            out.append((s_idx, seg_start + ls))

    return out


def train_segment_concat(
    values: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    split: Sequence[float],
) -> np.ndarray:
    blobs: List[np.ndarray] = []
    for s_idx in range(len(lengths)):
        off = int(offsets[s_idx])
        ln = int(lengths[s_idx])
        tr_len, _, _ = _resolve_split_edges(ln, split)
        seg = np.asarray(values[off : off + tr_len], dtype=np.float32).reshape(tr_len, 1, 1)
        blobs.append(seg)
    if not blobs:
        return np.zeros((1, 1, 1), dtype=np.float32)
    return np.concatenate(blobs, axis=0)


def fit_series_mu_sigma_train(
    values: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    split: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    n_series = len(lengths)
    mus = np.zeros(n_series, dtype=np.float32)
    sigs = np.ones(n_series, dtype=np.float32)
    for s_idx in range(n_series):
        off = int(offsets[s_idx])
        ln = int(lengths[s_idx])
        tr_len, _, _ = _resolve_split_edges(ln, split)
        block = np.asarray(values[off : off + tr_len], dtype=np.float64)
        vals = block[np.isfinite(block)]
        if vals.size > 0:
            mus[s_idx] = float(np.mean(vals))
            s = float(np.std(vals))
            sigs[s_idx] = float(s if s >= 1e-6 else 1e-6)
    return mus, sigs


class MonashSeriesWindowDataset(Dataset):
    """One window indexed by placements `(series_idx, start_offset_in_series)`."""

    def __init__(
        self,
        values: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        placements: Sequence[Tuple[int, int]],
        input_len: int,
        target_len: int,
        stride: int,
        *,
        norm_mode: NormMode = "global_standard",
        series_mu: Optional[np.ndarray] = None,
        series_sigma: Optional[np.ndarray] = None,
    ) -> None:
        if input_len <= 0 or target_len <= 0:
            raise ValueError("input_len and target_len must be positive")
        self.values = values
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.placements = list(placements)
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.stride = max(1, int(stride))
        self.norm_mode: NormMode = norm_mode
        self.series_mu = None if series_mu is None else np.asarray(series_mu, dtype=np.float32)
        self.series_sigma = None if series_sigma is None else np.asarray(series_sigma, dtype=np.float32)
        if norm_mode == "per_series_standard":
            if self.series_mu is None or self.series_sigma is None:
                raise ValueError("per_series_standard requires series_mu/sigma")

    def __len__(self) -> int:
        return len(self.placements)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s_idx, start_in_series = self.placements[int(index)]
        off = int(self.offsets[s_idx])
        mid = start_in_series + self.input_len
        end = mid + self.target_len

        xv = np.asarray(self.values[off + start_in_series : off + mid], dtype=np.float64).reshape(-1)
        yv = np.asarray(self.values[off + mid : off + end], dtype=np.float64).reshape(-1)

        if self.norm_mode == "instance":
            mu, sig = _finite_mu_sigma(np.concatenate([xv, yv]))
            xv = ((xv - mu) / sig).astype(np.float32)
            yv = ((yv - mu) / sig).astype(np.float32)
        elif self.norm_mode == "per_series_standard":
            mu = float(self.series_mu[s_idx])
            sig = float(self.series_sigma[s_idx])
            xv = ((xv - mu) / max(sig, 1e-6)).astype(np.float32)
            yv = ((yv - mu) / max(sig, 1e-6)).astype(np.float32)
        else:
            xv = xv.astype(np.float32)
            yv = yv.astype(np.float32)

        x_np = xv.reshape(self.input_len, 1, 1)
        y_np = yv.reshape(self.target_len, 1, 1)
        adj = np.eye(1, dtype=np.float32)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x_np)),
            "y": torch.from_numpy(np.ascontiguousarray(y_np)),
            "adj": torch.from_numpy(adj),
            "series_index": torch.tensor(int(s_idx), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def load_monash_dir(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    root = Path(root)
    vals = np.load(root / "values.npy", mmap_mode="r")
    offs = np.load(root / "offsets.npy")
    lens = np.load(root / "lengths.npy")
    meta: Dict[str, Any] = {}
    mp = root / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return vals, offs, lens, meta


def _collate_named(
    *,
    attach_graph: bool,
    attach_dataset_meta: Optional[Tuple[int, str]],
) -> Any:
    def _fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_collate(samples)
        batch.pop("adj", None)
        batch["x"], x_mask = _sanitize_tensor(batch["x"])
        batch["y"], y_mask = _sanitize_tensor(batch["y"])
        batch["x_mask"] = x_mask
        batch["y_mask"] = y_mask
        graph = torch.eye(1, dtype=torch.float32)
        if attach_graph:
            batch["graph"] = graph
        if attach_dataset_meta is not None:
            idx, ds_name = attach_dataset_meta
            n = batch["x"].shape[0]
            batch["dataset_name"] = ds_name
            batch["dataset_index"] = torch.full((n,), int(idx), dtype=torch.long)
            batch["num_nodes"] = torch.full((n,), 1, dtype=torch.long)
            batch["num_channels"] = torch.full((n,), 1, dtype=torch.long)
        return batch

    return _fn


def _limit_train_three(
    train_ds: Dataset,
    *,
    train_fraction: Optional[float],
    train_windows: Optional[int],
    max_train_windows: Optional[int],
) -> Dataset:
    limited = WindowDataModule._limit_dataset(
        train_ds,
        fraction=train_fraction,
        windows=train_windows,
    )
    if max_train_windows is not None:
        limited = WindowDataModule._limit_dataset(limited, windows=max_train_windows)
    return limited


@DATAMODULES.register()
class MonashSeriesWindowDataModule:
    """One ``data/Monash15/<subset>/`` directory."""

    def __init__(
        self,
        monash_root: str,
        input_len: int,
        output_len: Optional[int] = None,
        target_len: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        split: Sequence[float] = (0.7, 0.1, 0.2),
        stride: int = 1,
        scaler: Optional[dict] = None,
        norm_mode: str = "global_standard",
        shuffle_train: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        name: Optional[str] = None,
        max_train_windows: Optional[int] = None,
        max_val_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("MonashSeriesWindowDataModule requires target_len or output_len")
        self.monash_root = str(monash_root)
        self.name = None if name is None else str(name)
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.split = tuple(float(x) for x in split)
        self.stride = int(stride)

        nm = str(norm_mode).lower().replace("-", "_")
        if nm in {"global_standard", "standard"}:
            self._norm_enum: NormMode = "global_standard"
            self.scaler_cfg = scaler or {"type": "standard"}
        elif nm == "per_series_standard":
            self._norm_enum = "per_series_standard"
            self.scaler_cfg = scaler or {"type": "identity"}
        elif nm in {"instance", "instance_norm"}:
            self._norm_enum = "instance"
            self.scaler_cfg = scaler or {"type": "identity"}
        else:
            raise ValueError(
                f"norm_mode must be global_standard | per_series_standard | instance, got {norm_mode!r}"
            )

        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.train_fraction = train_fraction if train_fraction is not None else few_shot_ratio
        self.train_windows = train_windows if train_windows is not None else few_shot_windows
        self.distributed = bool(distributed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.max_train_windows = None if max_train_windows is None else int(max_train_windows)
        self.max_val_windows = None if max_val_windows is None else int(max_val_windows)
        self.max_test_windows = None if max_test_windows is None else int(max_test_windows)
        self._epoch = 0
        self.scaler = build_scaler(self.scaler_cfg)
        self.datasets: Dict[str, Dataset] = {}

    def setup(self) -> None:
        values, offsets, lengths, _ = load_monash_dir(Path(self.monash_root))

        placements_tr = enumerate_window_placements(
            offsets, lengths, self.split, self.input_len, self.target_len, self.stride, "train"
        )
        placements_va = enumerate_window_placements(
            offsets, lengths, self.split, self.input_len, self.target_len, self.stride, "val"
        )
        placements_te = enumerate_window_placements(
            offsets, lengths, self.split, self.input_len, self.target_len, self.stride, "test"
        )

        series_mu = series_sig = None
        if self._norm_enum == "per_series_standard":
            series_mu, series_sig = fit_series_mu_sigma_train(values, offsets, lengths, self.split)

        kw = dict(norm_mode=self._norm_enum, series_mu=series_mu, series_sigma=series_sig)

        ds_train = MonashSeriesWindowDataset(
            values, offsets, lengths, placements_tr, self.input_len, self.target_len, self.stride, **kw
        )
        ds_val = MonashSeriesWindowDataset(
            values, offsets, lengths, placements_va, self.input_len, self.target_len, self.stride, **kw
        )
        ds_test = MonashSeriesWindowDataset(
            values, offsets, lengths, placements_te, self.input_len, self.target_len, self.stride, **kw
        )

        if self._norm_enum == "global_standard":
            self.scaler.fit(train_segment_concat(values, offsets, lengths, self.split))
        else:
            self.scaler.fit(np.zeros((1, 1, 1), dtype=np.float32))

        train_ds = _limit_train_three(
            ds_train,
            train_fraction=self.train_fraction,
            train_windows=self.train_windows,
            max_train_windows=self.max_train_windows,
        )
        if self.max_val_windows is not None:
            ds_val = WindowDataModule._limit_dataset(ds_val, windows=self.max_val_windows)
        if self.max_test_windows is not None:
            ds_test = WindowDataModule._limit_dataset(ds_test, windows=self.max_test_windows)

        self.datasets = {"train": train_ds, "val": ds_val, "test": ds_test}

    def train_dataloader(
        self,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ) -> DataLoader:
        frac = train_fraction if train_fraction is not None else few_shot_ratio
        wins = train_windows if train_windows is not None else few_shot_windows
        dataset = WindowDataModule._limit_dataset(self.datasets["train"], fraction=frac, windows=wins)

        sampler = self._sampler(dataset, shuffle=self.shuffle_train, drop_last=self.drop_last)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=_collate_named(attach_graph=True, attach_dataset_meta=None),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=_collate_named(attach_graph=True, attach_dataset_meta=None),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=_collate_named(attach_graph=True, attach_dataset_meta=None),
        )

    def _sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> Optional[DistributedSampler]:
        if not self.distributed:
            return None
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.seed,
        )
        sampler.set_epoch(self._epoch)
        return sampler

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def get_scaler(self) -> object:
        return self.scaler

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.name or Path(self.monash_root).name,
            "num_nodes": 1,
            "num_channels": 1,
            "input_len": self.input_len,
            "target_len": self.target_len,
        }


@DATAMODULES.register()
class MonashMultiDatasetWindowDataModule:
    """Interleaved Monash subsets (parallel to MultiDatasetWindowDataModule)."""

    def __init__(
        self,
        datasets: Sequence[dict],
        input_len: int,
        target_len: Optional[int] = None,
        output_len: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        split: Sequence[float] = (0.7, 0.1, 0.2),
        stride: int = 1,
        scaler: Optional[dict] = None,
        norm_mode: str = "global_standard",
        shuffle_train: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_strategy: str = "round_robin",
        eval_strategy: str = "per_dataset",
        steps_per_epoch: Optional[int] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
        seed: int = 42,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        max_val_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("MonashMultiDatasetWindowDataModule requires target_len or output_len")
        if not datasets:
            raise ValueError("datasets list must not be empty")
        self.dataset_cfgs = [dict(x) for x in datasets]

        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.default_split = tuple(split)
        self.default_stride = int(stride)
        nm = str(norm_mode).lower().replace("-", "_")
        self.norm_mode_raw = nm
        if nm in {"global_standard", "standard"}:
            self._norm_enum: NormMode = "global_standard"
            self.default_scaler_cfg = scaler or {"type": "standard"}
        elif nm == "per_series_standard":
            self._norm_enum = "per_series_standard"
            self.default_scaler_cfg = scaler or {"type": "identity"}
        elif nm in {"instance", "instance_norm"}:
            self._norm_enum = "instance"
            self.default_scaler_cfg = scaler or {"type": "identity"}
        else:
            raise ValueError(f"unsupported norm_mode: {norm_mode!r}")

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.train_strategy = str(train_strategy)
        self.eval_strategy = str(eval_strategy)
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)
        self.dataset_weights = dict(dataset_weights or {})
        self.train_fraction = train_fraction if train_fraction is not None else few_shot_ratio
        self.train_windows = train_windows if train_windows is not None else few_shot_windows
        self.seed = int(seed)
        self.distributed = bool(distributed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.max_val_windows = None if max_val_windows is None else int(max_val_windows)
        self.max_test_windows = None if max_test_windows is None else int(max_test_windows)
        self._epoch = 0

        self.datasets_by_split: Dict[str, Dict[str, Dataset]] = {"train": {}, "val": {}, "test": {}}
        self.dataset_names: List[str] = []
        self.name_to_index: Dict[str, int] = {}
        self.dataset_batch_sizes: Dict[str, int] = {}
        self.scaler: Optional[DatasetAwareScaler] = None

    def _normalize_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        root = cfg.get("monash_root") or cfg.get("data_path")
        if not root:
            raise ValueError('Each Monash entry needs monash_root or data_path to a Monash subset folder')
        name = str(cfg.get("name") or Path(root).name)
        mtw = cfg.get("max_train_windows")
        if mtw is not None:
            mtw = int(mtw)
            if mtw <= 0:
                raise ValueError("max_train_windows must be positive when set")
        return {
            "name": name,
            "monash_root": str(root),
            "split": tuple(cfg.get("split", self.default_split)),
            "stride": int(cfg.get("stride", self.default_stride)),
            "scaler": dict(cfg.get("scaler", self.default_scaler_cfg)),
            "batch_size": int(cfg.get("batch_size", self.batch_size)),
            "norm_mode": str(cfg.get("norm_mode", self.norm_mode_raw)),
            "max_train_windows": mtw,
        }

    def setup(self) -> None:
        self.datasets_by_split = {"train": {}, "val": {}, "test": {}}
        self.dataset_names = []
        self.name_to_index = {}
        self.dataset_batch_sizes = {}
        fitted: List[object] = []

        for raw in self.dataset_cfgs:
            cfg = self._normalize_cfg(raw)
            nm = cfg["norm_mode"].lower().replace("-", "_")
            if nm in {"global_standard", "standard"}:
                norm_enum: NormMode = "global_standard"
            elif nm == "per_series_standard":
                norm_enum = "per_series_standard"
            elif nm in {"instance", "instance_norm"}:
                norm_enum = "instance"
            else:
                norm_enum = self._norm_enum

            vals, offs, lens, _ = load_monash_dir(Path(cfg["monash_root"]))
            scaler_i = build_scaler(dict(cfg["scaler"]))
            if norm_enum == "global_standard":
                scaler_i.fit(train_segment_concat(vals, offs, lens, cfg["split"]))
            else:
                scaler_i.fit(np.zeros((1, 1, 1), dtype=np.float32))

            placements_tr = enumerate_window_placements(
                offs, lens, cfg["split"], self.input_len, self.target_len, cfg["stride"], "train"
            )
            placements_va = enumerate_window_placements(
                offs, lens, cfg["split"], self.input_len, self.target_len, cfg["stride"], "val"
            )
            placements_te = enumerate_window_placements(
                offs, lens, cfg["split"], self.input_len, self.target_len, cfg["stride"], "test"
            )

            mu = sig = None
            if norm_enum == "per_series_standard":
                mu, sig = fit_series_mu_sigma_train(vals, offs, lens, cfg["split"])

            kw = dict(norm_mode=norm_enum, series_mu=mu, series_sigma=sig)
            ds_tr = MonashSeriesWindowDataset(
                vals, offs, lens, placements_tr, self.input_len, self.target_len, cfg["stride"], **kw
            )
            ds_va = MonashSeriesWindowDataset(
                vals, offs, lens, placements_va, self.input_len, self.target_len, cfg["stride"], **kw
            )
            ds_te = MonashSeriesWindowDataset(
                vals, offs, lens, placements_te, self.input_len, self.target_len, cfg["stride"], **kw
            )

            name = cfg["name"]
            if name in self.name_to_index:
                raise ValueError(f"duplicate Monash subset name {name}")

            ds_tr_work = ds_tr
            max_tw = cfg.get("max_train_windows")
            if max_tw is not None:
                ds_tr_work = WindowDataModule._limit_dataset(ds_tr_work, windows=int(max_tw))

            if self.max_val_windows is not None:
                ds_va = WindowDataModule._limit_dataset(ds_va, windows=self.max_val_windows)
            if self.max_test_windows is not None:
                ds_te = WindowDataModule._limit_dataset(ds_te, windows=self.max_test_windows)

            self.name_to_index[name] = len(self.dataset_names)
            self.dataset_names.append(name)
            self.dataset_batch_sizes[name] = cfg["batch_size"]
            fitted.append(scaler_i)
            self.datasets_by_split["train"][name] = ds_tr_work
            self.datasets_by_split["val"][name] = ds_va
            self.datasets_by_split["test"][name] = ds_te

        self.scaler = DatasetAwareScaler.from_scalers(
            names=self.dataset_names,
            scalers=fitted,
            max_channels=1,
        )

    def train_dataloader(
        self,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ):
        frac = train_fraction if train_fraction is not None else few_shot_ratio
        wins = train_windows if train_windows is not None else few_shot_windows
        loaders = self._split_loaders_train(frac=frac, windows=wins, shuffle=self.shuffle_train)
        return InterleavedNamedLoaders(
            loaders=loaders,
            strategy=self.train_strategy,
            steps_per_epoch=self.steps_per_epoch,
            dataset_weights=self.dataset_weights,
            seed=self.seed,
        )

    def val_dataloader(self):
        loaders = self._split_loaders_eval("val", shuffle=False)
        if self.eval_strategy == "per_dataset":
            return loaders
        return SequentialNamedLoaders(loaders)

    def test_dataloader(self):
        loaders = self._split_loaders_eval("test", shuffle=False)
        if self.eval_strategy == "per_dataset":
            return loaders
        return SequentialNamedLoaders(loaders)

    def _split_loaders_train(self, frac: Optional[float], windows: Optional[int], shuffle: bool):
        loaders: Dict[str, DataLoader] = {}
        for name, ds_base in self.datasets_by_split["train"].items():
            materialized = WindowDataModule._limit_dataset(ds_base, fraction=frac, windows=windows)

            sampler = self._sampler(materialized, shuffle=shuffle, drop_last=self.drop_last)
            collate = _collate_named(
                attach_graph=True,
                attach_dataset_meta=(self.name_to_index[name], name),
            )
            loaders[name] = DataLoader(
                materialized,
                batch_size=self.dataset_batch_sizes[name],
                shuffle=shuffle if sampler is None else False,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                collate_fn=collate,
            )
        return loaders

    def _split_loaders_eval(self, split: str, shuffle: bool):
        loaders: Dict[str, DataLoader] = {}
        for name, ds in self.datasets_by_split[split].items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.dataset_batch_sizes[name],
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                collate_fn=_collate_named(
                    attach_graph=True,
                    attach_dataset_meta=(self.name_to_index[name], name),
                ),
            )
        return loaders

    def _sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> Optional[DistributedSampler]:
        if not self.distributed:
            return None
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.seed,
        )
        sampler.set_epoch(self._epoch)
        return sampler

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def get_scaler(self) -> object:
        if self.scaler is None:
            raise RuntimeError("scaler unavailable before setup()")
        return self.scaler

    def get_metadata(self) -> Dict[str, Any]:
        if not self.dataset_names:
            raise RuntimeError("metadata unavailable before setup()")
        return {
            "num_nodes": 1,
            "num_channels": 1,
            "input_len": self.input_len,
            "target_len": self.target_len,
            "num_datasets": len(self.dataset_names),
            "dataset_names": list(self.dataset_names),
        }