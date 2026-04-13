"""Utility helpers for extension models."""

from basicstfm_ext.utils.dataset_stats import compute_dataset_descriptor

__all__ = ["compute_dataset_descriptor"]
"""Utility helpers for BasicSTFM extensions."""

from basicstfm_ext.utils.dataset_stats import DescriptorCache, DescriptorConfig, compute_dataset_descriptor
from basicstfm_ext.utils.protocol_debug import debug_tensor_shape, ensure_protocol_shape
from basicstfm_ext.utils.temporal_queries import HorizonQueryEncoder, LearnedTemporalQueries, temporal_index_features

__all__ = [
    "DescriptorCache",
    "DescriptorConfig",
    "HorizonQueryEncoder",
    "LearnedTemporalQueries",
    "compute_dataset_descriptor",
    "debug_tensor_shape",
    "ensure_protocol_shape",
    "temporal_index_features",
]
