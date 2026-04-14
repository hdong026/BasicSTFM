from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components
from basicstfm.registry import MODELS


class SRDSTFMShapeTest(unittest.TestCase):
    def test_srdstfm_outputs_expected_shapes(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "SRDSTFMBackbone",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "num_datasets": 4,
            }
        )
        x = torch.randn(2, 8, 5, 2)
        target = torch.randn(2, 3, 5, 2)
        graph = torch.eye(5)
        dataset_index = torch.tensor([0, 1], dtype=torch.long)

        out = model(
            x,
            graph=graph,
            target=target,
            dataset_index=dataset_index,
            mode="forecast",
        )
        self.assertEqual(tuple(out["forecast"].shape), (2, 3, 5, 2))
        self.assertEqual(tuple(out["stable_forecast"].shape), (2, 3, 5, 2))
        self.assertEqual(tuple(out["residual_forecast"].shape), (2, 3, 5, 2))
        self.assertEqual(tuple(out["stable_reconstruction"].shape), (2, 8, 5, 2))
        self.assertEqual(tuple(out["event_score"].shape), (2, 8, 5, 1))
        self.assertEqual(tuple(out["event_locality"].shape), (2, 8, 5, 1))
        self.assertEqual(tuple(out["propagation_map"].shape), (2, 3, 5, 5))
        self.assertEqual(tuple(out["inertia_gate"].shape), (2, 3, 5, 1))
        self.assertEqual(tuple(out["attenuation_gate"].shape), (2, 3, 5, 1))
        self.assertEqual(tuple(out["fusion_weight"].shape), (2, 3, 5, 1))

    def test_stable_mode_disables_diffusion_rollout(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "SRDSTFMBackbone",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 6,
                "output_len": 2,
                "hidden_dim": 8,
            }
        )
        x = torch.randn(1, 6, 4, 1)
        target = torch.randn(1, 2, 4, 1)
        out = model(x, graph=torch.eye(4), target=target, mode="stable_pretrain")
        self.assertEqual(float(out["propagation_map"].abs().sum().item()), 0.0)
        self.assertEqual(float(out["residual_forecast"].abs().sum().item()), 0.0)

    def test_encode_mode_returns_embedding(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "SRDSTFMBackbone",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 6,
                "output_len": 2,
                "hidden_dim": 8,
            }
        )
        x = torch.randn(2, 6, 4, 1)
        out = model(x, mode="encode")
        self.assertEqual(tuple(out["embedding"].shape), (2, 6, 4, 8))


if __name__ == "__main__":
    unittest.main()
