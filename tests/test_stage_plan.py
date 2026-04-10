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


if __name__ == "__main__":
    unittest.main()
