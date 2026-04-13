"""Plug-in extensions for BasicSTFM.

Importing :mod:`basicstfm_ext` is enough to register the extension models,
tasks, and auxiliary losses through BasicSTFM's existing registries.
"""

from basicstfm_ext import losses, models, tasks

__all__ = ["losses", "models", "tasks"]
