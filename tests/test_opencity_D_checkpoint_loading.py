from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components, import_custom_modules
from basicstfm.models.foundation.opencity import OpenCityFoundationModel
from basicstfm.registry import MODELS
from basicstfm.utils.checkpoint import save_checkpoint


class OpenCityDCheckpointLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import_builtin_components()
        import_custom_modules(["basicstfm_ext"])

    def test_protocol_wrapper_loads_vanilla_backbone_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "opencity_backbone.pt"
            backbone = OpenCityFoundationModel(
                num_nodes=6,
                input_dim=1,
                output_dim=1,
                input_len=12,
                output_len=12,
                hidden_dim=16,
                num_layers=1,
                num_heads=4,
                ffn_dim=32,
                dropout=0.0,
            )
            save_checkpoint(str(ckpt_path), backbone)

            wrapper = MODELS.build(
                {
                    "type": "OpenCityProtocolAdapterWrapper",
                    "num_nodes": 6,
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
                        "hidden_dim": 16,
                        "bottleneck_dim": 8,
                        "num_heads": 4,
                    },
                    "calibration_cfg": {
                        "embedding_dim": 16,
                        "stats_hidden_dim": 24,
                    },
                }
            )

            missing, unexpected = wrapper.load_backbone_weights(str(ckpt_path), strict=True)
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            for name, tensor in backbone.state_dict().items():
                self.assertTrue(torch.equal(tensor, wrapper.backbone.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
