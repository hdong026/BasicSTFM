from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components, import_custom_modules
from basicstfm.registry import MODELS


class OpenCityDIdentityModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import_builtin_components()
        import_custom_modules(["basicstfm_ext"])

    def test_matched_length_residual_identity_stays_close_to_backbone(self) -> None:
        model = MODELS.build(
            {
                "type": "OpenCityProtocolAdapterWrapper",
                "num_nodes": 4,
                "input_dim": 1,
                "output_dim": 1,
                "input_len": 12,
                "output_len": 12,
                "backbone_cfg": {
                    "input_len": 12,
                    "output_len": 12,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "ffn_dim": 32,
                    "dropout": 0.0,
                },
                "adapter_cfg": {
                    "input_backend": "cross_attention",
                    "output_backend": "query_attention",
                    "hidden_dim": 16,
                    "bottleneck_dim": 8,
                    "num_heads": 4,
                    "residual_scale_init": 0.0,
                },
                "calibration_cfg": {
                    "enable_conditioning": False,
                    "embedding_dim": 16,
                    "stats_hidden_dim": 24,
                },
                "distill_cfg": {
                    "teacher_source": "backbone",
                    "lambda_distill_matched": 1.0,
                    "lambda_identity_in": 1.0,
                    "lambda_identity_out": 1.0,
                },
            }
        )
        outputs = model(
            torch.randn(2, 12, 4, 1),
            graph=torch.eye(4),
            dataset_context={"dataset_name": "SRC", "metadata": {"target_len": 12}},
        )
        self.assertTrue(torch.allclose(outputs["forecast"], outputs["backbone_forecast"], atol=1e-6))
        self.assertIn("teacher_forecast", outputs)
        self.assertLess(float(outputs["aux_losses"]["identity_in"].item()), 1e-6)
        self.assertLess(float(outputs["aux_losses"]["identity_out"].item()), 1e-6)


if __name__ == "__main__":
    unittest.main()
