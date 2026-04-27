"""Base helpers for trainable task flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


class Task(ABC):
    """A task owns the per-batch training logic for a stage."""

    def set_scaler(self, scaler: object) -> None:
        self.scaler = scaler

    def transform(self, value: torch.Tensor, batch: Dict[str, Any] | None = None) -> torch.Tensor:
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            return value
        try:
            return scaler.transform(value, batch=batch)
        except TypeError:
            return scaler.transform(value)

    def inverse_transform(self, value: torch.Tensor, batch: Dict[str, Any] | None = None) -> torch.Tensor:
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            return value
        try:
            return scaler.inverse_transform(value, batch=batch)
        except TypeError:
            return scaler.inverse_transform(value)

    @staticmethod
    def merge_masks(*masks: Any) -> Any:
        merged = None
        for mask in masks:
            if mask is None:
                continue
            mask = mask.bool()
            merged = mask if merged is None else (merged & mask)
        return merged

    @staticmethod
    def align_prediction_target(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        if pred.shape[-1] == target.shape[-1]:
            return target, mask
        if pred.shape[-1] < target.shape[-1]:
            target = target[..., : pred.shape[-1]]
            if mask is not None:
                mask = mask[..., : pred.shape[-1]]
            return target, mask

        pad_channels = pred.shape[-1] - target.shape[-1]
        target_pad = target.new_zeros(*target.shape[:-1], pad_channels)
        target = torch.cat([target, target_pad], dim=-1)
        if mask is None:
            mask = target.new_ones(*target.shape[:-1], target.shape[-1] - pad_channels, dtype=torch.bool)
        mask = mask.bool()
        mask_pad = mask.new_zeros(*mask.shape[:-1], pad_channels)
        mask = torch.cat([mask, mask_pad], dim=-1)
        return target, mask

    @staticmethod
    def slice_prediction_to_data_channels(
        pred: torch.Tensor,
        data_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Match last-dim (channels) to the dataset before ``inverse_transform``.

        Same idea as ``StableResidualForecastingTask._align_pred_target`` in *scaled* space:
        a backbone trained on a max width (e.g. 18) may be evaluated on fewer channels (e.g. 3);
        the scaler mean/std only cover the data channels, so trim or pad *pred* to ``data_ref``'s width.
        """

        c = data_ref.shape[-1]
        if pred.shape[-1] == c:
            return pred
        if pred.shape[-1] > c:
            return pred[..., :c]
        pad = pred.new_zeros(*pred.shape[:-1], c - pred.shape[-1])
        return torch.cat([pred, pad], dim=-1)

    @abstractmethod
    def step(self, model: torch.nn.Module, batch: Dict[str, Any], losses, device: torch.device):
        raise NotImplementedError
