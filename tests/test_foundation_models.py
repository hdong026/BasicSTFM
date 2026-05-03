from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch
import torch.nn.functional as F

from basicstfm.builders import import_builtin_components
from basicstfm.registry import MODELS
from basicstfm.utils.checkpoint import save_checkpoint


class FoundationModelShapeTest(unittest.TestCase):
    def test_factost_pads_narrow_inputs_like_open_city(self):
        """Native-width batches must match explicit trailing channel zeros (STA padding contract)."""
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "FactoSTFoundationModel",
                "num_nodes": 5,
                "input_dim": 4,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "patch_len": 4,
                "num_prompt_tokens": 2,
                "use_st_adapter": False,
            }
        )
        model.eval()
        torch.manual_seed(0)
        x_narrow = torch.randn(2, 8, 5, 1)
        x_explicit = F.pad(x_narrow, (0, 3))
        with torch.no_grad():
            o_n = model(x_narrow, mode="forecast")["forecast"]
            o_ex = model(x_explicit, mode="forecast")["forecast"]
        torch.testing.assert_close(o_n, o_ex)

    def test_opencity_supports_multichannel_input_with_single_target_output(self):
        import_builtin_components()
        model = MODELS.build(
            {
                "type": "OpenCityFoundationModel",
                "num_nodes": 5,
                "input_dim": 3,
                "output_dim": 1,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
            }
        )
        x = torch.randn(2, 8, 5, 3)
        graph = torch.eye(5)
        out = model(x, graph=graph, mode="forecast")
        self.assertEqual(tuple(out["forecast"].shape), (2, 3, 5, 1))

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

    def test_backbone_weight_loaders_support_stage_transfer(self):
        import_builtin_components()

        factost_pretrain = MODELS.build(
            {
                "type": "FactoSTFoundationModel",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "patch_len": 4,
                "use_st_adapter": False,
            }
        )
        factost_adapter = MODELS.build(
            {
                "type": "FactoSTFoundationModel",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "patch_len": 4,
                "use_st_adapter": True,
            }
        )
        unist_pretrain = MODELS.build(
            {
                "type": "UniSTFoundationModel",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "use_prompt": False,
                "num_memory_spatial": 4,
                "num_memory_temporal": 4,
            }
        )
        unist_prompt = MODELS.build(
            {
                "type": "UniSTFoundationModel",
                "num_nodes": 5,
                "input_dim": 2,
                "output_dim": 2,
                "input_len": 8,
                "output_len": 3,
                "hidden_dim": 16,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "use_prompt": True,
                "num_memory_spatial": 4,
                "num_memory_temporal": 4,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factost_path = Path(tmpdir) / "factost.pt"
            unist_path = Path(tmpdir) / "unist.pt"
            save_checkpoint(str(factost_path), factost_pretrain, extra={"stage": "utp"})
            save_checkpoint(str(unist_path), unist_pretrain, extra={"stage": "pretrain"})

            gate_before = factost_adapter.state_dict()["metadata_gate.0.weight"].clone()
            st_before = factost_adapter.state_dict()["st_node_proj.weight"].clone()
            prompt_before = unist_prompt.state_dict()["prompt.spatial_memory.keys"].clone()

            factost_missing, factost_unexpected = factost_adapter.load_backbone_weights(
                str(factost_path),
                strict=False,
            )
            unist_missing, unist_unexpected = unist_prompt.load_backbone_weights(
                str(unist_path),
                strict=False,
            )

        self.assertEqual(factost_unexpected, [])
        self.assertEqual(unist_unexpected, [])
        self.assertEqual(factost_missing, [])
        self.assertEqual(unist_missing, [])
        torch.testing.assert_close(
            factost_adapter.state_dict()["patch_proj.weight"],
            factost_pretrain.state_dict()["patch_proj.weight"],
        )
        torch.testing.assert_close(
            factost_adapter.state_dict()["metadata_gate.0.weight"],
            gate_before,
        )
        torch.testing.assert_close(
            factost_adapter.state_dict()["st_node_proj.weight"],
            st_before,
        )
        torch.testing.assert_close(
            unist_prompt.state_dict()["prompt.spatial_memory.keys"],
            prompt_before,
        )
        torch.testing.assert_close(
            unist_prompt.state_dict()["value_proj.weight"],
            unist_pretrain.state_dict()["value_proj.weight"],
        )


if __name__ == "__main__":
    unittest.main()
