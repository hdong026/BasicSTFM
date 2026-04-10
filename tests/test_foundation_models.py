from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components
from basicstfm.registry import MODELS


class FoundationModelShapeTest(unittest.TestCase):
    def test_forecast_reconstruct_and_encode_shapes(self):
        import_builtin_components()
        cases = [
            (
                "OpenCityFoundationModel",
                {"num_layers": 1, "num_heads": 2, "ffn_dim": 32},
            ),
            (
                "FactoSTFoundationModel",
                {
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "patch_len": 4,
                    "num_prompt_tokens": 2,
                },
            ),
            (
                "UniSTFoundationModel",
                {
                    "encoder_layers": 1,
                    "decoder_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "num_memory_spatial": 4,
                    "num_memory_temporal": 4,
                },
            ),
        ]

        for model_type, extra in cases:
            with self.subTest(model_type=model_type):
                model = MODELS.build(
                    {
                        "type": model_type,
                        "num_nodes": 5,
                        "input_dim": 2,
                        "output_dim": 2,
                        "input_len": 8,
                        "output_len": 3,
                        "hidden_dim": 16,
                        **extra,
                    }
                )
                x = torch.randn(2, 8, 5, 2)
                graph = torch.eye(5)

                out = model(x, graph=graph, mode="both")
                self.assertEqual(tuple(out["forecast"].shape), (2, 3, 5, 2))
                self.assertEqual(tuple(out["reconstruction"].shape), (2, 8, 5, 2))

                encoded = model(x, graph=graph, mode="encode")
                self.assertEqual(encoded["embedding"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
