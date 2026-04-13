from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components, import_custom_modules
from basicstfm.registry import MODELS


class OpenCityInterfaceShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import_builtin_components()
        import_custom_modules(["basicstfm_ext"])

    def _build_model(self, variant: str, head_type: str = "gru"):
        return MODELS.build(
            {
                "type": "OpenCityVariableInterfaceWrapper",
                "num_nodes": 5,
                "input_dim": 1,
                "output_dim": 1,
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
                "interface_cfg": {
                    "variant": variant,
                    "head_type": head_type,
                    "hidden_dim": 12,
                    "bottleneck_dim": 6,
                    "num_layers": 1,
                },
                "conditioning_cfg": {
                    "source_datasets": ["SRC_A", "SRC_B"],
                    "embedding_dim": 16,
                    "stats_hidden_dim": 24,
                    "rank": 4,
                },
                "regularization_cfg": {
                    "lambda_adv": 0.0,
                    "lambda_ortho": 0.01,
                    "lambda_red": 0.01,
                },
            }
        )

    def test_all_variants_produce_runtime_horizon(self) -> None:
        x = torch.randn(2, 24, 5, 1)
        graph = torch.eye(5)
        dataset_context = {
            "dataset_name": "TARGET",
            "metadata": {"target_len": 18},
        }
        for variant in ("A", "B", "C"):
            with self.subTest(variant=variant):
                model = self._build_model(variant=variant)
                outputs = model(x, graph=graph, dataset_context=dataset_context)
                self.assertEqual(tuple(outputs["forecast"].shape), (2, 18, 5, 1))
                self.assertEqual(tuple(outputs["shared_feat"].shape), (2, 16))
                self.assertEqual(outputs["dataset_embedding"].shape[0], 16)
                self.assertIn("aux_losses", outputs)
                if variant == "A":
                    self.assertIn("prototype_weights", outputs)

    def test_mamba_backend_falls_back_cleanly(self) -> None:
        model = self._build_model(variant="C", head_type="mamba")
        outputs = model(
            torch.randn(2, 24, 5, 1),
            graph=torch.eye(5),
            dataset_context={"dataset_name": "TARGET", "metadata": {"target_len": 18}},
        )
        self.assertEqual(tuple(outputs["forecast"].shape), (2, 18, 5, 1))


if __name__ == "__main__":
    unittest.main()
