import unittest

from basicstfm.engines.stage import StagePlan, StageSpec


class StagePlanTest(unittest.TestCase):
    def test_stage_accepts_max_epochs_alias(self):
        stage = StageSpec.from_dict(
            {
                "name": "pretrain",
                "max_epochs": 2,
                "task": {"type": "MaskedReconstructionTask"},
                "losses": {"type": "mse"},
            }
        )
        self.assertEqual(stage.epochs, 2)
        self.assertEqual(stage.losses, [{"type": "mse"}])
        self.assertEqual(stage.save_every, 1)
        self.assertTrue(stage.save_last)
        self.assertTrue(stage.save_best)
        self.assertTrue(stage.save_epoch_checkpoints)

    def test_plan_from_config(self):
        plan = StagePlan.from_config(
            {
                "pipeline": {
                    "stages": [
                        {"name": "a", "task": {"type": "ForecastingTask"}},
                        {"name": "b", "task": {"type": "ForecastingTask"}, "epochs": 3},
                    ]
                }
            }
        )
        self.assertEqual([stage.name for stage in plan.stages], ["a", "b"])
        self.assertEqual(plan.stages[1].epochs, 3)

    def test_stage_pipeline_fields(self):
        stage = StageSpec.from_dict(
            {
                "name": "adapter_stage",
                "task": {"type": "ForecastingTask"},
                "model": {"type": "UniSTFoundationModel", "use_prompt": True},
                "data": {"few_shot_ratio": 0.05},
                "load_from": "unist_pretrain",
                "load_method": "checkpoint",
                "save_artifact": "unist_prompt",
                "freeze": "all",
                "unfreeze": ["prompt.*"],
            }
        )
        self.assertEqual(stage.model["type"], "UniSTFoundationModel")
        self.assertTrue(stage.model["use_prompt"])
        self.assertEqual(stage.data["few_shot_ratio"], 0.05)
        self.assertEqual(stage.load_from, "unist_pretrain")
        self.assertEqual(stage.load_method, "checkpoint")
        self.assertEqual(stage.save_artifact, "unist_prompt")
        self.assertEqual(stage.freeze, ["all"])
        self.assertEqual(stage.unfreeze, ["prompt.*"])


if __name__ == "__main__":
    unittest.main()
