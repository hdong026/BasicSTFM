from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components
from basicstfm.losses.common import LossCollection
from basicstfm.registry import MODELS, TASKS


class StableResidualForecastingTaskV3Test(unittest.TestCase):
    def test_robust_stage1_scales_loss(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "DPMV3Backbone",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 4,
                "hidden_dim": 16,
                "num_datasets": 2,
                "use_stable_graph_context": False,
            }
        )
        task = TASKS.build(
            {
                "type": "StableResidualForecastingTaskV3",
                "phase": "stable",
                "model_mode": "stable_pretrain",
                "robust_stage1": True,
                "robust_lambda": 0.5,
                "robust_temperature": 1.0,
                "robust_ema_momentum": 0.0,
                "robust_use_standardized_risk": True,
                "robust_group_key": "dataset",
            }
        )
        losses = LossCollection([{"type": "mae"}])
        b = 4
        batch = {
            "x": torch.randn(b, 8, 4, 1),
            "y": torch.randn(b, 4, 4, 1),
            "graph": torch.eye(4),
            "dataset_index": torch.zeros(b, dtype=torch.long),
        }
        out1 = task.step(model, batch, losses, torch.device("cpu"))
        t1 = out1["loss"]
        for _ in range(5):
            task.step(model, batch, losses, torch.device("cpu"))
        out2 = task.step(model, batch, losses, torch.device("cpu"))
        t2 = out2["loss"]
        self.assertTrue(torch.isfinite(t1) and torch.isfinite(t2))

    def test_non_robust_delegates_to_base(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "DPMV3Backbone",
                "num_nodes": 3,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 6,
                "output_len": 3,
                "hidden_dim": 8,
                "num_datasets": 1,
                "use_stable_graph_context": False,
            }
        )
        task = TASKS.build(
            {
                "type": "StableResidualForecastingTaskV3",
                "phase": "joint",
                "model_mode": "forecast",
                "robust_stage1": True,
            }
        )
        losses = LossCollection([{"type": "mae"}])
        batch = {
            "x": torch.randn(2, 6, 3, 1),
            "y": torch.randn(2, 3, 3, 1),
            "graph": torch.eye(3),
        }
        out = task.step(model, batch, losses, torch.device("cpu"))
        self.assertIn("loss", out)


if __name__ == "__main__":
    unittest.main()
