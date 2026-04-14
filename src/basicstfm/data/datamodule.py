"""Data modules that expose train/val/test dataloaders."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from basicstfm.data.scaler import DatasetAwareScaler, build_scaler
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


def _load_numpy(
    path: str,
    key: str = "data",
    mmap_mode: Optional[str] = None,
) -> np.ndarray:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    effective_mmap = mmap_mode if data_path.suffix.lower() == ".npy" else None
    loaded = np.load(data_path, allow_pickle=False, mmap_mode=effective_mmap)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key not in loaded:
            available = ", ".join(loaded.files)
            loaded.close()
            raise KeyError(f"Key {key!r} not found in {path}. Available: {available}")
        array = loaded[key].copy()
        loaded.close()
        return array
    return loaded


def _sanitize_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finite_mask = torch.isfinite(value)
    sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized, finite_mask


class PartitionedWindowDataset(Dataset):
    """Forecasting windows over a node subset of a larger [T, N, C] array."""

    def __init__(
        self,
        array: np.ndarray,
        node_ids: np.ndarray,
        input_len: int,
        target_len: int,
        stride: int = 1,
    ) -> None:
        if array.ndim != 3:
            raise ValueError(f"Expected data with shape [T, N, C], got {array.shape}")
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("PartitionedWindowDataset requires at least one node id")
        if input_len <= 0 or target_len <= 0:
            raise ValueError("input_len and target_len must be positive")
        self.array = array.astype(np.float32, copy=False)
        self.node_ids = node_ids
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.stride = int(stride)
        self.window_len = self.input_len + self.target_len
        if len(self.array) < self.window_len:
            raise ValueError(
                f"Not enough timesteps ({len(self.array)}) for input_len={input_len} "
                f"and target_len={target_len}"
            )

    def __len__(self) -> int:
        return (len(self.array) - self.window_len) // self.stride + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = index * self.stride
        mid = start + self.input_len
        end = mid + self.target_len
        window = np.asarray(self.array[start:end, self.node_ids, :], dtype=np.float32)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(window[: self.input_len])),
            "y": torch.from_numpy(np.ascontiguousarray(window[self.input_len :])),
            "index": torch.tensor(index, dtype=torch.long),
        }


def _partition_node_ids(
    num_nodes: int,
    partition_size: int,
    *,
    graph: Optional[np.ndarray] = None,
    strategy: str = "graph_greedy",
    overlap: int = 0,
    max_partitions: Optional[int] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    if partition_size <= 0:
        raise ValueError("partition_size must be positive")
    if partition_size >= num_nodes:
        return [np.arange(num_nodes, dtype=np.int64)]
    strategy = str(strategy).lower()
    overlap = int(overlap)
    if overlap < 0 or overlap >= partition_size:
        raise ValueError("partition_overlap must satisfy 0 <= overlap < partition_size")

    if strategy == "contiguous":
        step = max(1, partition_size - overlap)
        partitions = [
            np.arange(start, min(start + partition_size, num_nodes), dtype=np.int64)
            for start in range(0, num_nodes, step)
        ]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        order = np.arange(num_nodes, dtype=np.int64)
        rng.shuffle(order)
        step = max(1, partition_size - overlap)
        partitions = [
            np.sort(order[start : start + partition_size])
            for start in range(0, num_nodes, step)
            if order[start : start + partition_size].size > 0
        ]
    elif strategy == "graph_greedy":
        if graph is None:
            raise ValueError("partition_strategy='graph_greedy' requires a graph")
        adjacency = np.asarray(graph) != 0
        np.fill_diagonal(adjacency, False)
        degrees = adjacency.sum(axis=1)
        remaining = np.ones(num_nodes, dtype=np.bool_)
        partitions = []

        while remaining.any():
            remaining_idx = np.flatnonzero(remaining)
            seed_node = int(remaining_idx[np.argmax(degrees[remaining_idx])])
            queue = [seed_node]
            local_seen = {seed_node}
            part: List[int] = []

            while queue and len(part) < partition_size:
                node = queue.pop(0)
                if not remaining[node]:
                    continue
                remaining[node] = False
                part.append(node)
                neighbors = np.flatnonzero(adjacency[node] & remaining)
                if neighbors.size:
                    ordered = neighbors[np.argsort(-degrees[neighbors])]
                    for neighbor in ordered.tolist():
                        if neighbor not in local_seen:
                            queue.append(neighbor)
                            local_seen.add(neighbor)

            if len(part) < partition_size:
                fill = np.flatnonzero(remaining)
                if fill.size:
                    ordered_fill = fill[np.argsort(-degrees[fill])]
                    needed = partition_size - len(part)
                    extra = ordered_fill[:needed]
                    remaining[extra] = False
                    part.extend(int(idx) for idx in extra.tolist())

            partitions.append(np.asarray(sorted(part), dtype=np.int64))
    else:
        raise ValueError(f"Unsupported partition strategy: {strategy}")

    partitions = [part for part in partitions if part.size > 0]
    if max_partitions is not None:
        partitions = partitions[: int(max_partitions)]
    return partitions


class InterleavedNamedLoaders:
    """Interleave batches from multiple loaders and attach dataset identity."""

    def __init__(
        self,
        loaders: Mapping[str, DataLoader],
        strategy: str = "round_robin",
        steps_per_epoch: Optional[int] = None,
        dataset_weights: Optional[Mapping[str, float]] = None,
        seed: int = 42,
    ) -> None:
        if not loaders:
            raise ValueError("InterleavedNamedLoaders requires at least one loader")
        self.loaders = dict(loaders)
        self.strategy = str(strategy)
        self.steps_per_epoch = (
            int(steps_per_epoch)
            if steps_per_epoch is not None
            else int(sum(max(1, len(loader)) for loader in self.loaders.values()))
        )
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.dataset_weights = {
            name: float(dataset_weights.get(name, len(loader)) if dataset_weights else len(loader))
            for name, loader in self.loaders.items()
        }
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict]:
        names = list(self.loaders.keys())
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        rng = random.Random(self.seed)
        weight_vector = [max(self.dataset_weights.get(name, 0.0), 1e-6) for name in names]

        for step in range(self.steps_per_epoch):
            if self.strategy == "round_robin":
                name = names[step % len(names)]
            elif self.strategy in {"proportional", "size_proportional"}:
                name = rng.choices(names, weights=weight_vector, k=1)[0]
            elif self.strategy in {"uniform", "random"}:
                name = rng.choice(names)
            elif self.strategy == "sequential":
                cursor = step
                for candidate in names:
                    loader_len = max(1, len(self.loaders[candidate]))
                    if cursor < loader_len:
                        name = candidate
                        break
                    cursor -= loader_len
                else:
                    name = names[-1]
            else:
                raise ValueError(f"Unknown train mixing strategy: {self.strategy}")

            iterator = iterators[name]
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loaders[name])
                iterators[name] = iterator
                batch = next(iterator)
            yield batch


class SequentialNamedLoaders:
    """Yield validation/test batches dataset by dataset."""

    def __init__(self, loaders: Mapping[str, DataLoader]) -> None:
        self.loaders = dict(loaders)

    def __len__(self) -> int:
        return int(sum(len(loader) for loader in self.loaders.values()))

    def __iter__(self) -> Iterator[dict]:
        for loader in self.loaders.values():
            yield from loader


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
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        name: Optional[str] = None,
        mmap_mode: Optional[str] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("WindowDataModule requires target_len or output_len")
        self.name = None if name is None else str(name)
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
        self.distributed = bool(distributed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.mmap_mode = mmap_mode
        self._epoch = 0
        self.scaler = build_scaler(self.scaler_cfg)
        self.graph: Optional[torch.Tensor] = None
        self.datasets: Dict[str, Dataset] = {}
        self.data_shape: Optional[Tuple[int, int, int]] = None

    def setup(self) -> None:
        array = _load_numpy(self.data_path, self.input_key, mmap_mode=self.mmap_mode)
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
            graph = _load_numpy(self.graph_path, self.graph_key, mmap_mode=self.mmap_mode)
            if graph.ndim != 2 or graph.shape[0] != array.shape[1] or graph.shape[1] != array.shape[1]:
                raise ValueError(
                    f"Graph shape {tuple(graph.shape)} does not match data node count "
                    f"{array.shape[1]} for dataset {self.data_path}"
                )
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
        return self._make_loader(
            dataset,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last,
            distributed=self.distributed,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return self._make_loader(
            self.datasets[split],
            shuffle=shuffle,
            drop_last=self.drop_last if split == "train" else False,
            distributed=False,
        )

    def _make_loader(
        self,
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool,
        distributed: bool,
    ) -> DataLoader:
        sampler = self._build_sampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            distributed=distributed,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate,
        )

    def _collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_collate(samples)
        batch["x"], x_mask = _sanitize_tensor(batch["x"])
        batch["y"], y_mask = _sanitize_tensor(batch["y"])
        batch["x_mask"] = x_mask
        batch["y_mask"] = y_mask
        if self.graph is not None:
            batch["graph"] = self.graph
        return batch

    def _limit_train_dataset(self, dataset: Dataset) -> Dataset:
        return self._limit_dataset(
            dataset,
            fraction=self.train_fraction,
            windows=self.train_windows,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
        distributed: bool,
    ) -> Optional[DistributedSampler]:
        if not distributed:
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
            "dataset_name": self.name,
            "data_shape": self.data_shape,
            "num_nodes": num_nodes,
            "num_channels": num_channels,
            "input_len": self.input_len,
            "target_len": self.target_len,
        }


@DATAMODULES.register()
class PartitionedWindowDataModule:
    """Window datamodule that trains on node partitions of one large graph."""

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
        train_strategy: str = "round_robin",
        steps_per_epoch: Optional[int] = None,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        name: Optional[str] = None,
        mmap_mode: Optional[str] = "r",
        partition_size: int = 1024,
        partition_overlap: int = 0,
        partition_strategy: str = "graph_greedy",
        max_partitions: Optional[int] = None,
        scaler_fit_timesteps: Optional[int] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("PartitionedWindowDataModule requires target_len or output_len")
        self.name = None if name is None else str(name)
        self.data_path = data_path
        self.input_key = input_key
        self.graph_path = graph_path
        self.graph_key = graph_key
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.split = tuple(split)
        self.stride = int(stride)
        self.scaler_cfg = scaler or {"type": "standard"}
        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.train_strategy = str(train_strategy)
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)
        self.train_fraction = train_fraction if train_fraction is not None else few_shot_ratio
        self.train_windows = train_windows if train_windows is not None else few_shot_windows
        self.distributed = bool(distributed)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.mmap_mode = mmap_mode
        self.partition_size = int(partition_size)
        self.partition_overlap = int(partition_overlap)
        self.partition_strategy = str(partition_strategy)
        self.max_partitions = None if max_partitions is None else int(max_partitions)
        self.scaler_fit_timesteps = (
            None if scaler_fit_timesteps is None else int(scaler_fit_timesteps)
        )
        self._epoch = 0
        self.scaler = build_scaler(self.scaler_cfg)
        self.data_shape: Optional[Tuple[int, int, int]] = None
        self.graph: Optional[torch.Tensor] = None
        self.partition_names: List[str] = []
        self.partition_to_index: Dict[str, int] = {}
        self.partition_graphs: Dict[str, Optional[torch.Tensor]] = {}
        self.partition_batch_sizes: Dict[str, int] = {}
        self.datasets_by_split: Dict[str, Dict[str, Dataset]] = {"train": {}, "val": {}, "test": {}}

    def setup(self) -> None:
        array = _load_numpy(self.data_path, self.input_key, mmap_mode=self.mmap_mode)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"Expected [T, N, C] data, got {array.shape}")
        self.data_shape = tuple(int(x) for x in array.shape)

        graph_np = None
        if self.graph_path:
            graph_np = _load_numpy(self.graph_path, self.graph_key, mmap_mode=self.mmap_mode)
            if (
                graph_np.ndim != 2
                or graph_np.shape[0] != array.shape[1]
                or graph_np.shape[1] != array.shape[1]
            ):
                raise ValueError(
                    f"Graph shape {tuple(graph_np.shape)} does not match data node count "
                    f"{array.shape[1]} for dataset {self.data_path}"
                )
            self.graph = torch.as_tensor(np.asarray(graph_np, dtype=np.float32), dtype=torch.float32)

        train_len, val_len, test_len = _split_lengths(len(array), self.split)
        train_raw = array[:train_len]
        val_raw = array[train_len : train_len + val_len]
        test_raw = array[train_len + val_len : train_len + val_len + test_len]

        fit_raw = train_raw
        if self.scaler_fit_timesteps is not None and self.scaler_fit_timesteps < len(train_raw):
            fit_raw = train_raw[: self.scaler_fit_timesteps]
        self.scaler.fit(fit_raw)

        partitions = _partition_node_ids(
            num_nodes=int(array.shape[1]),
            partition_size=self.partition_size,
            graph=graph_np,
            strategy=self.partition_strategy,
            overlap=self.partition_overlap,
            max_partitions=self.max_partitions,
            seed=self.seed,
        )
        if not partitions:
            raise RuntimeError("PartitionedWindowDataModule failed to create any partitions")

        self.partition_names = []
        self.partition_to_index = {}
        self.partition_graphs = {}
        self.partition_batch_sizes = {}
        self.datasets_by_split = {"train": {}, "val": {}, "test": {}}

        for idx, node_ids in enumerate(partitions):
            name = f"partition_{idx:03d}"
            self.partition_names.append(name)
            self.partition_to_index[name] = idx
            self.partition_batch_sizes[name] = self.batch_size
            self.datasets_by_split["train"][name] = WindowDataModule._limit_dataset(
                PartitionedWindowDataset(
                    train_raw,
                    node_ids=node_ids,
                    input_len=self.input_len,
                    target_len=self.target_len,
                    stride=self.stride,
                ),
                fraction=self.train_fraction,
                windows=self.train_windows,
            )
            self.datasets_by_split["val"][name] = PartitionedWindowDataset(
                val_raw,
                node_ids=node_ids,
                input_len=self.input_len,
                target_len=self.target_len,
                stride=self.stride,
            )
            self.datasets_by_split["test"][name] = PartitionedWindowDataset(
                test_raw,
                node_ids=node_ids,
                input_len=self.input_len,
                target_len=self.target_len,
                stride=self.stride,
            )
            if graph_np is None:
                self.partition_graphs[name] = None
            else:
                subgraph = np.asarray(graph_np[np.ix_(node_ids, node_ids)], dtype=np.float32)
                self.partition_graphs[name] = torch.as_tensor(subgraph, dtype=torch.float32)

    def train_dataloader(
        self,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ):
        fraction = train_fraction if train_fraction is not None else few_shot_ratio
        windows = train_windows if train_windows is not None else few_shot_windows
        loaders = self._make_split_loaders(
            split="train",
            fraction=fraction,
            windows=windows,
            shuffle=self.shuffle_train,
        )
        return InterleavedNamedLoaders(
            loaders,
            strategy=self.train_strategy,
            steps_per_epoch=self.steps_per_epoch,
            seed=self.seed,
        )

    def val_dataloader(self):
        return self._make_split_loaders(split="val", shuffle=False)

    def test_dataloader(self):
        return self._make_split_loaders(split="test", shuffle=False)

    def _make_split_loaders(
        self,
        split: str,
        fraction: Optional[float] = None,
        windows: Optional[int] = None,
        shuffle: bool = False,
    ) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        for name, dataset in self.datasets_by_split[split].items():
            materialized = (
                WindowDataModule._limit_dataset(dataset, fraction=fraction, windows=windows)
                if split == "train"
                else dataset
            )
            sampler = self._build_sampler(
                materialized,
                shuffle=shuffle,
                drop_last=self.drop_last if split == "train" else False,
                distributed=self.distributed and split == "train",
            )
            loaders[name] = DataLoader(
                materialized,
                batch_size=self.partition_batch_sizes[name],
                shuffle=shuffle if sampler is None else False,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last if split == "train" else False,
                collate_fn=self._make_collate(name),
            )
        return loaders

    def _make_collate(self, name: str):
        graph = self.partition_graphs[name]
        partition_index = int(self.partition_to_index[name])

        def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch = default_collate(samples)
            batch["x"], x_mask = _sanitize_tensor(batch["x"])
            batch["y"], y_mask = _sanitize_tensor(batch["y"])
            batch["x_mask"] = x_mask
            batch["y_mask"] = y_mask
            batch["dataset_name"] = name
            batch["dataset_index"] = torch.full(
                (batch["x"].shape[0],),
                partition_index,
                dtype=torch.long,
            )
            batch["num_nodes"] = torch.full((batch["x"].shape[0],), batch["x"].shape[2], dtype=torch.long)
            batch["num_channels"] = torch.full(
                (batch["x"].shape[0],),
                batch["x"].shape[-1],
                dtype=torch.long,
            )
            if graph is not None:
                batch["graph"] = graph
            return batch

        return collate

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
        distributed: bool,
    ) -> Optional[DistributedSampler]:
        if not distributed:
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

    def get_scaler(self) -> object:
        return self.scaler

    def get_metadata(self) -> Dict[str, Any]:
        if self.data_shape is None:
            raise RuntimeError("DataModule metadata is unavailable before setup")
        max_nodes = max((graph.shape[0] for graph in self.partition_graphs.values() if graph is not None), default=0)
        if max_nodes == 0:
            max_nodes = min(self.partition_size, self.data_shape[1])
        return {
            "dataset_name": self.name or Path(self.data_path).stem,
            "data_shape": self.data_shape,
            "num_nodes": max_nodes,
            "num_channels": self.data_shape[2],
            "input_len": self.input_len,
            "target_len": self.target_len,
            "num_partitions": len(self.partition_names),
        }


@DATAMODULES.register()
class MultiDatasetWindowDataModule:
    """Joint pretraining data module for multiple window datasets.

    The module keeps one scaler and one loader per dataset, then exposes a
    configurable interleaving strategy for training. Validation and test can be
    returned either as a combined sequential loader or as a mapping of
    per-dataset loaders so the trainer can compute macro metrics.
    """

    def __init__(
        self,
        datasets: Sequence[dict],
        input_len: int,
        target_len: Optional[int] = None,
        output_len: Optional[int] = None,
        input_key: str = "data",
        graph_key: str = "adj",
        batch_size: int = 32,
        num_workers: int = 0,
        split: Sequence[float] = (0.7, 0.1, 0.2),
        stride: int = 1,
        scaler: Optional[dict] = None,
        shuffle_train: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_strategy: str = "round_robin",
        eval_strategy: str = "per_dataset",
        steps_per_epoch: Optional[int] = None,
        dataset_weights: Optional[dict] = None,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
        seed: int = 42,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        mmap_mode: Optional[str] = None,
    ) -> None:
        if output_len is not None:
            target_len = output_len
        if target_len is None:
            raise ValueError("MultiDatasetWindowDataModule requires target_len or output_len")
        if not datasets:
            raise ValueError("MultiDatasetWindowDataModule requires a non-empty datasets list")
        self.dataset_cfgs = [dict(item) for item in datasets]
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.default_input_key = input_key
        self.default_graph_key = graph_key
        self.default_split = tuple(split)
        self.default_stride = int(stride)
        self.default_scaler_cfg = scaler or {"type": "standard"}
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
        self._epoch = 0
        self.mmap_mode = mmap_mode

        self.datasets_by_split: Dict[str, Dict[str, Dataset]] = {"train": {}, "val": {}, "test": {}}
        self.graphs: Dict[str, Optional[torch.Tensor]] = {}
        self.dataset_names: List[str] = []
        self.dataset_batch_sizes: Dict[str, int] = {}
        self.data_shapes: Dict[str, Tuple[int, int, int]] = {}
        self.dataset_num_channels: Dict[str, int] = {}
        self.dataset_metadata: Dict[str, Dict[str, Any]] = {}
        self.name_to_index: Dict[str, int] = {}
        self.max_num_nodes = 0
        self.max_num_channels = 0
        self.scaler: Optional[DatasetAwareScaler] = None

    def setup(self) -> None:
        self.datasets_by_split = {"train": {}, "val": {}, "test": {}}
        self.graphs = {}
        self.dataset_names = []
        self.dataset_batch_sizes = {}
        self.data_shapes = {}
        self.dataset_num_channels = {}
        self.dataset_metadata = {}
        self.name_to_index = {}
        self.max_num_nodes = 0
        self.max_num_channels = 0
        fitted_scalers = []

        for raw_cfg in self.dataset_cfgs:
            cfg = self._normalize_dataset_cfg(raw_cfg)
            name = cfg["name"]
            if name in self.name_to_index:
                raise ValueError(f"Duplicate dataset name in multi-dataset config: {name}")

            array = _load_numpy(cfg["data_path"], cfg["input_key"], mmap_mode=self.mmap_mode)
            if array.ndim == 2:
                array = array[..., None]
            if array.ndim != 3:
                raise ValueError(f"Expected [T, N, C] data for {name}, got {array.shape}")

            train_len, val_len, test_len = _split_lengths(len(array), cfg["split"])
            train_raw = array[:train_len]
            val_raw = array[train_len : train_len + val_len]
            test_raw = array[train_len + val_len : train_len + val_len + test_len]

            scaler = build_scaler(cfg["scaler"])
            scaler.fit(train_raw)
            fitted_scalers.append(scaler)

            self.name_to_index[name] = len(self.dataset_names)
            self.dataset_names.append(name)
            self.dataset_batch_sizes[name] = int(cfg["batch_size"])
            self.data_shapes[name] = tuple(int(x) for x in array.shape)
            self.dataset_num_channels[name] = int(array.shape[2])
            self.max_num_nodes = max(self.max_num_nodes, int(array.shape[1]))
            self.max_num_channels = max(self.max_num_channels, int(array.shape[2]))

            self.datasets_by_split["train"][name] = WindowDataset(
                train_raw,
                self.input_len,
                self.target_len,
                int(cfg["stride"]),
            )
            self.datasets_by_split["val"][name] = WindowDataset(
                val_raw,
                self.input_len,
                self.target_len,
                int(cfg["stride"]),
            )
            self.datasets_by_split["test"][name] = WindowDataset(
                test_raw,
                self.input_len,
                self.target_len,
                int(cfg["stride"]),
            )

            graph = None
            if cfg["graph_path"]:
                loaded_graph = _load_numpy(
                    cfg["graph_path"],
                    cfg["graph_key"],
                    mmap_mode=self.mmap_mode,
                )
                if (
                    loaded_graph.ndim != 2
                    or loaded_graph.shape[0] != array.shape[1]
                    or loaded_graph.shape[1] != array.shape[1]
                ):
                    raise ValueError(
                        f"Graph shape {tuple(loaded_graph.shape)} does not match data node count "
                        f"{array.shape[1]} for dataset {name}"
                    )
                graph = torch.as_tensor(loaded_graph, dtype=torch.float32)
            self.graphs[name] = graph
            self.dataset_metadata[name] = {
                "data_path": cfg["data_path"],
                "graph_path": cfg["graph_path"],
                "num_nodes": int(array.shape[1]),
                "num_channels": int(array.shape[2]),
                "input_len": self.input_len,
                "target_len": self.target_len,
                "batch_size": int(cfg["batch_size"]),
                "split": list(cfg["split"]),
                "stride": int(cfg["stride"]),
            }

        self.scaler = DatasetAwareScaler.from_scalers(
            names=self.dataset_names,
            scalers=fitted_scalers,
            max_channels=self.max_num_channels,
        )

    def train_dataloader(
        self,
        train_fraction: Optional[float] = None,
        train_windows: Optional[int] = None,
        few_shot_ratio: Optional[float] = None,
        few_shot_windows: Optional[int] = None,
    ):
        loaders = self._make_split_loaders(
            "train",
            fraction=train_fraction if train_fraction is not None else few_shot_ratio,
            windows=train_windows if train_windows is not None else few_shot_windows,
            shuffle=self.shuffle_train,
        )
        return InterleavedNamedLoaders(
            loaders=loaders,
            strategy=self.train_strategy,
            steps_per_epoch=self.steps_per_epoch,
            dataset_weights=self.dataset_weights,
            seed=self.seed,
        )

    def val_dataloader(self):
        loaders = self._make_split_loaders("val", shuffle=False)
        if self.eval_strategy == "per_dataset":
            return loaders
        if self.eval_strategy == "combined":
            return SequentialNamedLoaders(loaders)
        raise ValueError(f"Unknown eval_strategy: {self.eval_strategy}")

    def test_dataloader(self):
        loaders = self._make_split_loaders("test", shuffle=False)
        if self.eval_strategy == "per_dataset":
            return loaders
        if self.eval_strategy == "combined":
            return SequentialNamedLoaders(loaders)
        raise ValueError(f"Unknown eval_strategy: {self.eval_strategy}")

    def get_scaler(self) -> object:
        if self.scaler is None:
            raise RuntimeError("DataModule scaler is unavailable before setup")
        return self.scaler

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def get_metadata(self) -> Dict[str, Any]:
        if not self.dataset_names:
            raise RuntimeError("DataModule metadata is unavailable before setup")
        return {
            "num_nodes": self.max_num_nodes,
            "num_channels": self.max_num_channels,
            "input_len": self.input_len,
            "target_len": self.target_len,
            "num_datasets": len(self.dataset_names),
            "dataset_names": list(self.dataset_names),
            "datasets": dict(self.dataset_metadata),
        }

    def _normalize_dataset_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        data_path = cfg.get("data_path")
        if not data_path:
            raise ValueError("Each multi-dataset entry must define data_path")
        name = str(cfg.get("name") or Path(data_path).parent.name or Path(data_path).stem)
        return {
            "name": name,
            "data_path": str(data_path),
            "graph_path": cfg.get("graph_path"),
            "input_key": str(cfg.get("input_key", self.default_input_key)),
            "graph_key": str(cfg.get("graph_key", self.default_graph_key)),
            "split": tuple(cfg.get("split", self.default_split)),
            "stride": int(cfg.get("stride", self.default_stride)),
            "scaler": dict(cfg.get("scaler", self.default_scaler_cfg)),
            "batch_size": int(cfg.get("batch_size", self.batch_size)),
        }

    def _make_split_loaders(
        self,
        split: str,
        fraction: Optional[float] = None,
        windows: Optional[int] = None,
        shuffle: bool = False,
    ) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        for name, dataset in self.datasets_by_split[split].items():
            materialized = (
                WindowDataModule._limit_dataset(dataset, fraction=fraction, windows=windows)
                if split == "train"
                else dataset
            )
            sampler = self._build_sampler(
                materialized,
                shuffle=shuffle,
                drop_last=self.drop_last if split == "train" else False,
                distributed=self.distributed and split == "train",
            )
            loaders[name] = DataLoader(
                materialized,
                batch_size=self.dataset_batch_sizes[name],
                shuffle=shuffle if sampler is None else False,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last if split == "train" else False,
                collate_fn=self._make_collate(name),
            )
        return loaders

    def _build_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
        distributed: bool,
    ) -> Optional[DistributedSampler]:
        if not distributed:
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

    def _make_collate(self, name: str):
        dataset_index = int(self.name_to_index[name])
        graph = self.graphs[name]
        num_channels = int(self.dataset_num_channels[name])

        def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch = default_collate(samples)
            batch["x"], x_finite = _sanitize_tensor(batch["x"])
            batch["y"], y_finite = _sanitize_tensor(batch["y"])
            batch["x"] = self._pad_channels(batch["x"], self.max_num_channels)
            batch["y"] = self._pad_channels(batch["y"], self.max_num_channels)
            x_finite = self._pad_channels(x_finite.to(dtype=batch["x"].dtype), self.max_num_channels).bool()
            y_finite = self._pad_channels(y_finite.to(dtype=batch["y"].dtype), self.max_num_channels).bool()
            batch["x_mask"] = self._channel_mask(batch["x"], num_channels).bool() & x_finite
            batch["y_mask"] = self._channel_mask(batch["y"], num_channels).bool() & y_finite
            batch["dataset_name"] = name
            batch["dataset_index"] = torch.full(
                (batch["x"].shape[0],),
                dataset_index,
                dtype=torch.long,
            )
            batch["num_nodes"] = torch.full((batch["x"].shape[0],), batch["x"].shape[2], dtype=torch.long)
            batch["num_channels"] = torch.full((batch["x"].shape[0],), num_channels, dtype=torch.long)
            if graph is not None:
                batch["graph"] = graph
            return batch

        return collate

    @staticmethod
    def _pad_channels(value: torch.Tensor, target_channels: int) -> torch.Tensor:
        if value.shape[-1] == target_channels:
            return value
        if value.shape[-1] > target_channels:
            raise ValueError(
                f"Cannot pad channels from {value.shape[-1]} down to {target_channels}"
            )
        pad_shape = list(value.shape)
        pad_shape[-1] = target_channels - value.shape[-1]
        pad = value.new_zeros(pad_shape)
        return torch.cat([value, pad], dim=-1)

    @staticmethod
    def _channel_mask(value: torch.Tensor, num_channels: int) -> torch.Tensor:
        mask = value.new_zeros(value.shape)
        mask[..., :num_channels] = 1
        return mask


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
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
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
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            seed=seed,
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
