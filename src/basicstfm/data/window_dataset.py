"""Sliding-window datasets for spatio-temporal tensors."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Create forecasting windows from an array shaped [T, N, C]."""

    def __init__(
        self,
        array: np.ndarray,
        input_len: int,
        target_len: int,
        stride: int = 1,
    ) -> None:
        if array.ndim != 3:
            raise ValueError(f"Expected data with shape [T, N, C], got {array.shape}")
        if input_len <= 0 or target_len <= 0:
            raise ValueError("input_len and target_len must be positive")
        self.array = array.astype(np.float32, copy=False)
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
        return {
            "x": torch.from_numpy(self.array[start:mid]),
            "y": torch.from_numpy(self.array[mid:end]),
            "index": torch.tensor(index, dtype=torch.long),
        }
