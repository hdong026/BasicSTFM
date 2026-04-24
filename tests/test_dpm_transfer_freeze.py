from __future__ import annotations

import fnmatch
import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.builders import import_builtin_components
from basicstfm.engines.stage import StageSpec
from basicstfm.engines.trainer import MultiStageTrainer
from basicstfm.registry import MODELS


class SRDSTFMTransferFreezeTest(unittest.TestCase):
    def test_few_shot_freeze_unfreeze_targets_diffusion_path(self):
        import_builtin_components()
        cfg = {
            "data": {
                "type": "SyntheticDataModule",
                "input_len": 6,
                "output_len": 3,
            },
            "model": {
                "type": "SRDSTFMBackbone",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 6,
                "output_len": 3,
                "hidden_dim": 8,
            },
            "pipeline": {
                "stages": [
                    {
                        "name": "dummy",
                        "task": {"type": "ForecastingTask"},
                    }
                ]
            },
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)
        trainer.model = MODELS.build(cfg["model"])

        stage = StageSpec.from_dict(
            {
                "name": "few_shot",
                "task": {"type": "StableResidualForecastingTask"},
                "freeze": ["all"],
                "unfreeze": [
                    "residual_event_encoder.*",
                    "diffusion_mechanism_learner.*",
                    "fusion_predictor.*",
                    "calibration_head.*",
                ],
            }
        )
        trainer._apply_trainability(stage)

        model = trainer.model
        if model is None:
            self.fail("model should not be None")

        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}

        self.assertTrue(trainable)
        self.assertTrue(any(name.startswith("diffusion_mechanism_learner.") for name in trainable))
        self.assertTrue(any(name.startswith("fusion_predictor.") for name in trainable))
        self.assertTrue(any(name.startswith("calibration_head.") for name in trainable))
        self.assertTrue(any(name.startswith("stable_trunk.") for name in frozen))

        allow_patterns = stage.unfreeze
        for name in trainable:
            self.assertTrue(any(fnmatch.fnmatch(name, pat) for pat in allow_patterns))


class DPMV2BackboneSmokeTest(unittest.TestCase):
    def test_forward_with_graph_matches_shapes(self):
        import_builtin_components()
        import torch

        model = MODELS.build(
            {
                "type": "DPMV2Backbone",
                "num_nodes": 5,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 4,
                "hidden_dim": 16,
                "use_stable_graph_context": True,
                "stable_graph_num_layers": 2,
                "stable_graph_share_weights": True,
                "num_datasets": 1,
            }
        )
        b, t, n = 2, 8, 5
        x = torch.randn(b, t, n, 1)
        g = torch.eye(n)
        out = model(x, graph=g, mode="forecast", target=torch.randn(b, 4, n, 1))
        self.assertEqual(tuple(out["forecast"].shape), (b, 4, n, 1))

        out_no = model(x, graph=None, mode="forecast", target=torch.randn(b, 4, n, 1))
        self.assertEqual(tuple(out_no["forecast"].shape), (b, 4, n, 1))

        off = MODELS.build(
            {
                "type": "DPMV2Backbone",
                "num_nodes": 5,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 4,
                "hidden_dim": 16,
                "use_stable_graph_context": False,
                "num_datasets": 1,
            }
        )
        out_ab = off(x, graph=g, mode="forecast", target=torch.randn(b, 4, n, 1))
        self.assertEqual(tuple(out_ab["forecast"].shape), (b, 4, n, 1))


if __name__ == "__main__":
    unittest.main()
