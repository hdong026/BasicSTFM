from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.config import load_config
from basicstfm.engines.stage import StagePlan


class SRDSTFMStageProtocolTest(unittest.TestCase):
    def test_protocol_config_contains_stable_diffusion_joint_transfer(self):
        cfg_path = Path("configs/foundation/dpm_stfm_pretrain_zero_fewshot.yaml")
        cfg = load_config(str(cfg_path))
        plan = StagePlan.from_config(cfg)

        names = [stage.name for stage in plan.stages]
        self.assertEqual(names[0], "stable_trunk_pretraining")
        self.assertEqual(names[1], "residual_diffusion_pretraining")
        self.assertEqual(names[2], "joint_refinement")
        self.assertIn("metr_la_zero_shot", names)
        self.assertIn("metr_la_five_percent_mechanism_tuning", names)

        stage1 = plan.stages[0]
        stage2 = plan.stages[1]
        stage3 = plan.stages[2]

        self.assertEqual(stage1.task["type"], "StableResidualForecastingTask")
        self.assertEqual(stage1.task["phase"], "stable")
        self.assertEqual(stage2.task["phase"], "diffusion")
        self.assertEqual(stage3.task["phase"], "joint")

        self.assertIn("stable_trunk.*", stage2.freeze)

    def test_zero_shot_stages_are_eval_only(self):
        cfg_path = Path("configs/foundation/dpm_stfm_pretrain_zero_fewshot.yaml")
        cfg = load_config(str(cfg_path))
        plan = StagePlan.from_config(cfg)

        zero_shot = [stage for stage in plan.stages if "zero_shot" in stage.name]
        self.assertTrue(zero_shot)
        for stage in zero_shot:
            self.assertTrue(stage.eval_only)
            self.assertEqual(stage.epochs, 0)


if __name__ == "__main__":
    unittest.main()
