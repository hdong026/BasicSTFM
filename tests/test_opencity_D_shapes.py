from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components, import_custom_modules
from basicstfm.registry import MODELS


class OpenCityDShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import_builtin_components()
        import_custom_modules(["basicstfm_ext"])

    def test_protocol_wrapper_supports_mismatch_shapes(self) -> None:
        model = MODELS.build(
            {
                "type": "OpenCityProtocolAdapterWrapper",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 24,
                "output_len": 18,
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
                },
                "calibration_cfg": {
                    "embedding_dim": 16,
                    "stats_hidden_dim": 24,
                },
                "distill_cfg": {
                    "lambda_distill_matched": 0.0,
                    "lambda_identity_in": 0.0,
                    "lambda_identity_out": 0.0,
                },
            }
        )
        outputs = model(
            torch.randn(2, 24, 5, 2),
            graph=torch.eye(5),
            dataset_context={"dataset_name": "TARGET", "metadata": {"target_len": 18}},
        )
        self.assertEqual(tuple(outputs["forecast"].shape), (2, 18, 5, 2))
        self.assertEqual(tuple(outputs["backbone_forecast"].shape), (2, 18, 5, 2))
        self.assertEqual(tuple(outputs["fixed_backbone_forecast"].shape), (2, 12, 5, 2))
        self.assertEqual(tuple(outputs["shared_feat"].shape), (2, 16))
        self.assertEqual(outputs["dataset_embedding"].shape[0], 16)


if __name__ == "__main__":
    unittest.main()
