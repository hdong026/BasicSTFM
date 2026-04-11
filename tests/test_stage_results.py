from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.builders import import_builtin_components
from basicstfm.engines.trainer import MultiStageTrainer


class StageResultsTest(unittest.TestCase):
    def test_trainer_writes_structured_stage_results(self):
        import_builtin_components()
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "run"
            cfg = {
                "experiment_name": "stage_results_test",
                "data": {
                    "type": "SyntheticDataModule",
                    "num_timesteps": 72,
                    "num_nodes": 4,
                    "num_channels": 1,
                    "input_len": 8,
                    "output_len": 4,
                    "batch_size": 4,
                    "split": [0.6, 0.2, 0.2],
                    "shuffle_train": False,
                },
                "model": {
                    "type": "TinySTFoundationModel",
                    "num_nodes": "auto",
                    "input_dim": "auto",
                    "output_dim": "auto",
                    "input_len": "auto",
                    "output_len": "auto",
                    "hidden_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 16,
                },
                "pipeline": {
                    "stages": [
                        {
                            "name": "pretrain",
                            "epochs": 1,
                            "save_artifact": "backbone",
                            "task": {"type": "MaskedReconstructionTask", "mask_ratio": 0.25},
                            "losses": [{"type": "mse"}],
                            "metrics": [{"type": "mae"}],
                            "optimizer": {"type": "AdamW", "lr": 0.001},
                        },
                        {
                            "name": "zero_shot_eval",
                            "eval_only": True,
                            "epochs": 0,
                            "load_from": "backbone",
                            "task": {"type": "ForecastingTask"},
                            "losses": [{"type": "mae"}],
                            "metrics": [{"type": "mae"}],
                        },
                    ]
                },
            }

            trainer = MultiStageTrainer(
                cfg,
                work_dir=str(work_dir),
                device="cpu",
                log_every=0,
            )
            trainer.run()

            result_path = work_dir / "results" / "stage_results.json"
            self.assertTrue(result_path.exists())

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_name"], "stage_results_test")
            self.assertEqual([stage["name"] for stage in payload["stages"]], ["pretrain", "zero_shot_eval"])
            self.assertIn("train", payload["stages"][0])
            self.assertIn("test", payload["stages"][0])
            self.assertTrue(payload["stages"][1]["eval_only"])
            self.assertIsNone(payload["stages"][1]["train"])


if __name__ == "__main__":
    unittest.main()
