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


if __name__ == "__main__":
    unittest.main()
