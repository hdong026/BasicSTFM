from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.builders import import_builtin_components, import_custom_modules
from basicstfm.engines.trainer import MultiStageTrainer


class OpenCityInterfaceStageFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import_builtin_components()
        import_custom_modules(["basicstfm_ext"])

    def test_few_shot_stage_can_freeze_backbone_and_tune_heads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "seed": 7,
                "experiment_name": "opencity_interface_freeze_test",
                "data": {
                    "type": "SyntheticDataModule",
                    "num_timesteps": 480,
                    "num_nodes": 6,
                    "num_channels": 1,
                    "input_len": 24,
                    "output_len": 18,
                    "batch_size": 4,
                    "split": [0.7, 0.1, 0.2],
                },
                "model": {
                    "type": "OpenCityVariableInterfaceWrapper",
                    "num_nodes": "auto",
                    "input_dim": "auto",
                    "output_dim": "auto",
                    "input_len": "auto",
                    "output_len": "auto",
                    "backbone_cfg": {
                        "input_len": 12,
                        "output_len": 12,
                        "hidden_dim": 16,
                        "num_layers": 1,
                        "num_heads": 4,
                        "ffn_dim": 32,
                        "dropout": 0.0,
                    },
                    "interface_cfg": {
                        "variant": "C",
                        "head_type": "gru",
                        "hidden_dim": 12,
                        "bottleneck_dim": 6,
                    },
                    "conditioning_cfg": {
                        "embedding_dim": 16,
                        "stats_hidden_dim": 24,
                        "rank": 4,
                    },
                },
                "pipeline": {
                    "stages": [
                        {
                            "name": "pretrain",
                            "epochs": 1,
                            "task": {"type": "InterfaceForecastingTask"},
                            "losses": [{"type": "mae"}],
                            "metrics": [{"type": "mae"}],
                        },
                        {
                            "name": "few_shot",
                            "epochs": 1,
                            "few_shot_ratio": 0.1,
                            "freeze": ["all"],
                            "unfreeze": ["input_head.*", "output_head.*", "conditioning.*"],
                            "task": {"type": "InterfaceForecastingTask"},
                            "losses": [{"type": "mae"}],
                            "metrics": [{"type": "mae"}],
                        },
                    ]
                },
            }
            trainer = MultiStageTrainer(
                cfg,
                work_dir=str(Path(tmpdir) / "run"),
                device="cpu",
                log_every=0,
                dry_run=False,
                strategy="auto",
            )
            trainer.setup()
            stage = trainer.plan.stages[1]
            trainer._prepare_stage_components(stage, 1)
            trainer._apply_trainability(stage)
            assert trainer.model is not None
            model = trainer.model

            param_map = {name: param.requires_grad for name, param in model.named_parameters()}
            self.assertTrue(any(name.startswith("input_head.") and value for name, value in param_map.items()))
            self.assertTrue(any(name.startswith("output_head.") and value for name, value in param_map.items()))
            self.assertTrue(any(name.startswith("conditioning.") and value for name, value in param_map.items()))
            self.assertTrue(
                all(not value for name, value in param_map.items() if name.startswith("backbone."))
            )


if __name__ == "__main__":
    unittest.main()
