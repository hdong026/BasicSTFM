from __future__ import annotations

import importlib.util
import unittest

from basicstfm.engines.stage import StagePlan


class EvalOnlyStageTest(unittest.TestCase):
    def test_eval_only_stage_may_have_zero_epochs(self):
        plan = StagePlan.from_config(
            {
                "pipeline": {
                    "stages": [
                        {
                            "name": "zero_shot",
                            "epochs": 0,
                            "eval_only": True,
                            "task": {"type": "ForecastingTask"},
                        }
                    ]
                }
            }
        )
        self.assertTrue(plan.stages[0].eval_only)
        self.assertEqual(plan.stages[0].epochs, 0)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class FewShotDataModuleTest(unittest.TestCase):
    def test_stage_level_few_shot_loader_limits_training_windows(self):
        from basicstfm.data.datamodule import SyntheticDataModule

        dm = SyntheticDataModule(
            num_timesteps=80,
            num_nodes=4,
            num_channels=1,
            input_len=8,
            output_len=4,
            batch_size=2,
            shuffle_train=False,
        )
        dm.setup()
        full = len(dm.datasets["train"])
        limited_loader = dm.train_dataloader(few_shot_windows=3)
        self.assertGreater(full, 3)
        self.assertEqual(len(limited_loader.dataset), 3)


if __name__ == "__main__":
    unittest.main()
