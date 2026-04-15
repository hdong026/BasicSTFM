from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components
from basicstfm.losses.common import LossCollection
from basicstfm.registry import MODELS, TASKS
from basicstfm.tasks.masking import sample_spatiotemporal_mask, temporal_tail_mask


class StageTaskProtocolTest(unittest.TestCase):
    def test_mixed_mask_sampler_returns_nontrivial_mask(self):
        x = torch.randn(2, 8, 5, 1)
        mask = sample_spatiotemporal_mask(
            x,
            mask_ratio=0.5,
            strategy="mixed",
            strategies=("random", "temporal", "tube", "block"),
        )
        self.assertEqual(mask.shape, x.shape)
        self.assertTrue(mask.any())
        self.assertTrue((~mask).any())

    def test_temporal_tail_mask_matches_future_suffix(self):
        x = torch.randn(2, 10, 4, 1)
        mask = temporal_tail_mask(x, future_steps=3)
        self.assertFalse(mask[:, :7].any())
        self.assertTrue(mask[:, 7:].all())

    def test_masked_forecast_completion_task_runs_with_unist(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "UniSTFoundationModel",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 6,
                "output_len": 4,
                "hidden_dim": 16,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "use_prompt": True,
                "num_memory_spatial": 4,
                "num_memory_temporal": 4,
            }
        )
        task = TASKS.build({"type": "MaskedForecastCompletionTask"})
        losses = LossCollection([{"type": "mae"}]).to(torch.device("cpu"))
        batch = {
            "x": torch.randn(2, 6, 4, 1),
            "y": torch.randn(2, 4, 4, 1),
        }
        out = task.step(model, batch, losses, torch.device("cpu"))
        self.assertEqual(tuple(out["pred"].shape), (2, 4, 4, 1))
        self.assertEqual(tuple(out["target"].shape), (2, 4, 4, 1))

    def test_joint_pretraining_task_supports_spectral_terms(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "TinySTFoundationModel",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 4,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
            }
        )
        task = TASKS.build(
            {
                "type": "JointReconstructionForecastTask",
                "mask_ratio": 0.5,
                "mask_strategy": "temporal",
                "reconstruction_spectral_weight": 0.1,
                "forecast_spectral_weight": 0.1,
                "spectral_scales": [1, 2],
            }
        )
        losses = LossCollection([{"type": "mse"}]).to(torch.device("cpu"))
        batch = {
            "x": torch.randn(2, 8, 4, 1),
            "y": torch.randn(2, 4, 4, 1),
        }
        out = task.step(model, batch, losses, torch.device("cpu"))
        self.assertIn("reconstruction/loss/spectral", out["logs"])
        self.assertIn("forecast/loss/spectral", out["logs"])

    def test_stable_residual_task_supports_trend_target(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "SRDSTFMBackbone",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 4,
                "hidden_dim": 16,
                "stable_summary_mode": "attention",
                "stable_frequency_low_ratio": 0.4,
                "use_calibration_head": False,
            }
        )
        task = TASKS.build(
            {
                "type": "StableResidualForecastingTask",
                "phase": "joint",
                "stable_target": "trend",
                "trend_scale": 2,
                "stable_low_ratio": 0.4,
                "stable_num_low_bins": 2,
            }
        )
        losses = LossCollection([{"type": "mae"}]).to(torch.device("cpu"))
        batch = {
            "x": torch.randn(2, 8, 4, 1),
            "y": torch.randn(2, 4, 4, 1),
            "graph": torch.eye(4),
        }
        out = task.step(model, batch, losses, torch.device("cpu"))
        self.assertIn("loss/total", out["logs"])
        self.assertEqual(tuple(out["pred"].shape), (2, 4, 4, 1))
        self.assertEqual(tuple(out["target"].shape), (2, 4, 4, 1))


if __name__ == "__main__":
    unittest.main()
