import unittest
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.engines.trainer import MultiStageTrainer


class FakeDataModule:
    def get_metadata(self):
        return {
            "data_shape": (100, 207, 1),
            "num_nodes": 207,
            "num_channels": 1,
            "input_len": 12,
            "target_len": 12,
        }


class ModelConfigResolutionTest(unittest.TestCase):
    def test_auto_model_dimensions_are_inferred_from_data(self):
        cfg = {
            "data": {"type": "unused"},
            "model": {"type": "TinySTFoundationModel"},
            "pipeline": {"stages": [{"name": "forecasting", "task": {"type": "ForecastingTask"}}]},
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)
        trainer.datamodule = FakeDataModule()

        resolved = trainer._resolve_model_config(
            {
                "type": "TinySTFoundationModel",
                "num_nodes": "auto",
                "input_dim": "auto",
                "output_dim": "auto",
                "input_len": "auto",
                "output_len": "auto",
            }
        )

        self.assertEqual(resolved["num_nodes"], 207)
        self.assertEqual(resolved["input_dim"], 1)
        self.assertEqual(resolved["output_dim"], 1)
        self.assertEqual(resolved["input_len"], 12)
        self.assertEqual(resolved["output_len"], 12)

    def test_explicit_model_dimensions_are_preserved(self):
        cfg = {
            "data": {"type": "unused"},
            "model": {"type": "TinySTFoundationModel"},
            "pipeline": {"stages": [{"name": "forecasting", "task": {"type": "ForecastingTask"}}]},
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)
        trainer.datamodule = FakeDataModule()

        resolved = trainer._resolve_model_config(
            {
                "type": "TinySTFoundationModel",
                "num_nodes": 10,
                "input_dim": 3,
                "output_dim": 3,
                "input_len": 6,
                "output_len": 2,
            }
        )

        self.assertEqual(resolved["num_nodes"], 10)
        self.assertEqual(resolved["input_dim"], 3)
        self.assertEqual(resolved["output_dim"], 3)
        self.assertEqual(resolved["input_len"], 6)
        self.assertEqual(resolved["output_len"], 2)

    def test_auto_horizons_follow_datamodule_metadata(self):
        cfg = {
            "data": {"type": "unused"},
            "model": {"type": "TinySTFoundationModel"},
            "pipeline": {"stages": [{"name": "forecasting", "task": {"type": "ForecastingTask"}}]},
        }
        trainer = MultiStageTrainer(cfg, dry_run=True)

        class HorizonDataModule:
            def get_metadata(self):
                return {
                    "data_shape": (100, 10, 2),
                    "num_nodes": 10,
                    "num_channels": 2,
                    "input_len": 36,
                    "target_len": 18,
                }

        trainer.datamodule = HorizonDataModule()
        resolved = trainer._resolve_model_config(
            {
                "type": "TinySTFoundationModel",
                "num_nodes": "auto",
                "input_dim": "auto",
                "output_dim": "auto",
                "input_len": "auto",
                "output_len": "auto",
            }
        )

        self.assertEqual(resolved["input_len"], 36)
        self.assertEqual(resolved["output_len"], 18)


if __name__ == "__main__":
    unittest.main()
