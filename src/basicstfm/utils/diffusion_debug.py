"""Debug utilities for diffusion-mechanism introspection."""

from __future__ import annotations

from typing import Any, Dict

import torch


def tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    """Return compact scalar statistics for logging."""

    value = value.detach()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def build_diffusion_debug_payload(outputs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Collect summary stats of key stable-residual-diffusion tensors."""

    tracked = {
        "stable_component": outputs.get("stable_forecast"),
        "residual_energy": outputs.get("residual_energy"),
        "event_activation": outputs.get("event_activation"),
        "diffusion_gate": outputs.get("diffusion_gate"),
        "inertia_gate": outputs.get("inertia_gate"),
        "attenuation_gate": outputs.get("attenuation_gate"),
        "propagation_map": outputs.get("propagation_map"),
        "fusion_weight": outputs.get("fusion_weight"),
    }
    payload: Dict[str, Dict[str, float]] = {}
    for key, value in tracked.items():
        if isinstance(value, torch.Tensor):
            payload[key] = tensor_stats(value)
    return payload
