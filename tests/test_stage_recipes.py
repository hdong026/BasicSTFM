import unittest
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.engines.trainer import MultiStageTrainer


class StageRecipeCompilationTest(unittest.TestCase):
    def test_model_recipes_inherit_and_reset(self):
        cfg = {
            "data": {
                "type": "SyntheticDataModule",
                "input_len": 12,
                "output_len": 6,
            },
            "model": {
                "type": "TinySTFoundationModel",
                "num_nodes": 8,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 12,
                "output_len": 6,
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 64,
            },
            "pipeline": {
                "stages": [
                    {
                        "name": "pretrain",
                        "task": {"type": "MaskedReconstructionTask"},
                        "model": {"hidden_dim": 48},
                    },
                    {
                        "name": "adapter",
                        "task": {"type": "ForecastingTask"},
                        "model": {"ffn_dim": 128},
                    },
                    {
                        "name": "fresh_family_variant",
                        "task": {"type": "ForecastingTask"},
                        "reset_model": True,
                        "model": {"hidden_dim": 96},
                    },
                ]
            },
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)

        self.assertEqual(trainer._stage_model_recipes[0]["hidden_dim"], 48)
        self.assertEqual(trainer._stage_model_recipes[1]["hidden_dim"], 48)
        self.assertEqual(trainer._stage_model_recipes[1]["ffn_dim"], 128)
        self.assertEqual(trainer._stage_model_recipes[2]["hidden_dim"], 96)
        self.assertEqual(trainer._stage_model_recipes[2]["ffn_dim"], 64)

    def test_data_recipes_inherit_and_reset(self):
        cfg = {
            "data": {
                "type": "SyntheticDataModule",
                "input_len": 24,
                "output_len": 12,
                "batch_size": 16,
            },
            "model": {
                "type": "TinySTFoundationModel",
                "num_nodes": 8,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 24,
                "output_len": 12,
            },
            "pipeline": {
                "stages": [
                    {
                        "name": "stage_1",
                        "task": {"type": "ForecastingTask"},
                        "data": {"batch_size": 64},
                    },
                    {
                        "name": "stage_2",
                        "task": {"type": "ForecastingTask"},
                        "data": {"few_shot_ratio": 0.1},
                    },
                    {
                        "name": "stage_3",
                        "task": {"type": "ForecastingTask"},
                        "reset_data": True,
                        "data": {"batch_size": 8},
                    },
                ]
            },
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)

        self.assertEqual(trainer._stage_data_recipes[0]["batch_size"], 64)
        self.assertEqual(trainer._stage_data_recipes[1]["batch_size"], 64)
        self.assertEqual(trainer._stage_data_recipes[1]["few_shot_ratio"], 0.1)
        self.assertEqual(trainer._stage_data_recipes[2]["batch_size"], 8)
        self.assertNotIn("few_shot_ratio", trainer._stage_data_recipes[2])

    def test_switching_model_type_requires_explicit_reset(self):
        cfg = {
            "data": {"type": "SyntheticDataModule", "input_len": 12, "output_len": 6},
            "model": {
                "type": "TinySTFoundationModel",
                "num_nodes": 8,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 12,
                "output_len": 6,
            },
            "pipeline": {
                "stages": [
                    {"name": "stage_1", "task": {"type": "ForecastingTask"}},
                    {
                        "name": "stage_2",
                        "task": {"type": "ForecastingTask"},
                        "model": {"type": "UniSTFoundationModel"},
                    },
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "reset_model"):
            MultiStageTrainer(cfg, dry_run=True)

    def test_switching_data_type_requires_explicit_reset(self):
        cfg = {
            "data": {"type": "SyntheticDataModule", "input_len": 12, "output_len": 6},
            "model": {
                "type": "TinySTFoundationModel",
                "num_nodes": 8,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 12,
                "output_len": 6,
            },
            "pipeline": {
                "stages": [
                    {"name": "stage_1", "task": {"type": "ForecastingTask"}},
                    {
                        "name": "stage_2",
                        "task": {"type": "ForecastingTask"},
                        "data": {"type": "WindowDataModule"},
                    },
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "reset_data"):
            MultiStageTrainer(cfg, dry_run=True)


if __name__ == "__main__":
    unittest.main()
