"""Data modules that expose train/val/test dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate

from basicstfm.data.scaler import build_scaler
from basicstfm.data.window_dataset import WindowDataset
from basicstfm.registry import DATAMODULES


def _split_lengths(total: int, split: Sequence[float]) -> Tuple[int, int, int]:
    if len(split) != 3:
        raise ValueError("split must contain train/val/test ratios or lengths")
    if all(isinstance(x, float) and x <= 1.0 for x in split):
        train = int(total * split[0])
        val = int(total * split[1])
        test = total - train - val
        return train, val, test
    train, val, test = (int(x) for x in split)
    if train + val + test > total:
        raise ValueError("Explicit split lengths exceed total timesteps")
    return train, val, test


def _load_numpy(path: str, key: str = "data") -> np.ndarray:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    loaded = np.load(data_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key not in loaded:
            available = ", ".join(loaded.files)
            loaded.close()
            raise KeyError(f"Key {key!r} not found in {path}. Available: {available}")
        array = loaded[key].copy()
        loaded.close()
        return array
    return loaded


@DATAMODULES.register()
class WindowDataModule:
    """Load a single [T, N, C] array and expose sliding-window loaders."""

    def __init__(
        self,
        data_path: str,
        input_len: int,
        target_len: Optional[int] = None,
        output_len: Optional[int] = None,
        input_key: str = "data",
        graph_path: Optional[str] = None,
        graph_key: str = "adj",
        batch_size: int = 32,
        num_workers: int = 0,
        split: Sequence[float] = (0.7, 0.1, 0.2),
        stride: int = 1,
        scaler: Optional[dict] = None,
        shuffle_train: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("WindowDataModule requires target_len or output_len")
        self.data_path = data_path
        self.input_key = input_key
        self.graph_path = graph_path
        self.graph_key = graph_key
        self.input_len = input_len
        self.target_len = int(target_len)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.stride = stride
        self.scaler_cfg = scaler or {"type": "standard"}
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_fraction = train_fraction if train_fraction is not None else few_shot_ratio
        self.train_windows = train_windows if train_windows is not None else few_shot_windows
        self.scaler = build_scaler(self.scaler_cfg)
        self.graph: Optional[torch.Tensor] = None
        self.datasets: Dict[str, Dataset] = {}
        self.data_shape: Optional[Tuple[int, int, int]] = None

    def setup(self) -> None:
        array = _load_numpy(self.data_path, self.input_key)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"Expected [T, N, C] data, got {array.shape}")
        self.data_shape = tuple(int(x) for x in array.shape)

        train_len, val_len, test_len = _split_lengths(len(array), self.split)
        train_raw = array[:train_len]
        val_raw = array[train_len : train_len + val_len]
        test_raw = array[train_len + val_len : train_len + val_len + test_len]

        self.scaler.fit(train_raw)

        train_dataset = WindowDataset(train_raw, self.input_len, self.target_len, self.stride)
        self.datasets = {
            "train": self._limit_train_dataset(train_dataset),
            "val": WindowDataset(val_raw, self.input_len, self.target_len, self.stride),
            "test": WindowDataset(test_raw, self.input_len, self.target_len, self.stride),
        }

        if self.graph_path:
            graph = _load_numpy(self.graph_path, self.graph_key)
            self.graph = torch.as_tensor(graph, dtype=torch.float32)

    def train_dataloader(
        self,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ) -> DataLoader:
        dataset = self._limit_dataset(
            self.datasets["train"],
            fraction=train_fraction if train_fraction is not None else few_shot_ratio,
            windows=train_windows if train_windows is not None else few_shot_windows,
        )
        return self._make_loader(dataset, shuffle=self.shuffle_train, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return self._make_loader(
            self.datasets[split],
            shuffle=shuffle,
            drop_last=self.drop_last if split == "train" else False,
        )

    def _make_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate,
        )

    def _collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_collate(samples)
        if self.graph is not None:
            batch["graph"] = self.graph
        return batch

    def _limit_train_dataset(self, dataset: Dataset) -> Dataset:
        return self._limit_dataset(
            dataset,
            fraction=self.train_fraction,
            windows=self.train_windows,
        )

    @staticmethod
    def _limit_dataset(
        dataset: Dataset,
        fraction: Optional[float] = None,
        windows: Optional[int] = None,
    ) -> Dataset:
        length = len(dataset)
        limit = length
        if fraction is not None:
            fraction = float(fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("train_fraction/few_shot_ratio must be in (0, 1]")
            limit = min(limit, max(1, int(length * fraction)))
        if windows is not None:
            windows = int(windows)
            if windows <= 0:
                raise ValueError("train_windows/few_shot_windows must be positive")
            limit = min(limit, windows)
        if limit == length:
            return dataset
        return Subset(dataset, list(range(limit)))

    def get_scaler(self) -> object:
        return self.scaler

    def get_metadata(self) -> Dict[str, Any]:
        if self.data_shape is None:
            raise RuntimeError("DataModule metadata is unavailable before setup")
        _, num_nodes, num_channels = self.data_shape
        return {
            "data_shape": self.data_shape,
            "num_nodes": num_nodes,
            "num_channels": num_channels,
            "input_len": self.input_len,
            "target_len": self.target_len,
        }


@DATAMODULES.register()
class SyntheticDataModule(WindowDataModule):
    """Generate smooth multi-node signals for smoke tests and demos."""

    def __init__(
        self,
        num_timesteps: int = 720,
        num_nodes: int = 32,
        num_channels: int = 2,
        input_len: int = 24,
        target_len: Optional[int] = None,
        output_len: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        split: Sequence[float] = (0.7, 0.1, 0.2),
        stride: int = 1,
        scaler: Optional[dict] = None,
        shuffle_train: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
        seed: int = 42,
        noise_std: float = 0.05,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            target_len = 12
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.seed = seed
        self.noise_std = noise_std
        super().__init__(
            data_path="",
            input_len=input_len,
            target_len=target_len,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
            stride=stride,
            scaler=scaler,
            shuffle_train=shuffle_train,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_fraction=train_fraction,
            train_windows=train_windows,
            few_shot_ratio=few_shot_ratio,
            few_shot_windows=few_shot_windows,
        )

    def setup(self) -> None:
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.num_timesteps, dtype=np.float32)[:, None, None]
        node_phase = np.linspace(0.0, np.pi, self.num_nodes, dtype=np.float32)[None, :, None]
        channel_scale = np.linspace(0.8, 1.2, self.num_channels, dtype=np.float32)[None, None, :]
        daily = np.sin(2 * np.pi * t / 24.0 + node_phase)
        weekly = np.cos(2 * np.pi * t / (24.0 * 7.0) + node_phase / 2.0)
        trend = 0.001 * t * channel_scale
        noise = rng.normal(
            0.0,
            self.noise_std,
            size=(self.num_timesteps, self.num_nodes, self.num_channels),
        )
        array = (daily + 0.5 * weekly + trend + noise).astype(np.float32)
        self.data_shape = tuple(int(x) for x in array.shape)

        distances = np.abs(np.arange(self.num_nodes)[:, None] - np.arange(self.num_nodes)[None, :])
        graph = np.exp(-distances / max(1.0, self.num_nodes / 8.0)).astype(np.float32)
        self.graph = torch.from_numpy(graph)

        train_len, val_len, test_len = _split_lengths(len(array), self.split)
        train_raw = array[:train_len]
        val_raw = array[train_len : train_len + val_len]
        test_raw = array[train_len + val_len : train_len + val_len + test_len]

        self.scaler.fit(train_raw)
        train_dataset = WindowDataset(
            train_raw, self.input_len, self.target_len, self.stride
        )
        self.datasets = {
            "train": self._limit_train_dataset(train_dataset),
            "val": WindowDataset(
                val_raw, self.input_len, self.target_len, self.stride
            ),
            "test": WindowDataset(
                test_raw, self.input_len, self.target_len, self.stride
            ),
        }
