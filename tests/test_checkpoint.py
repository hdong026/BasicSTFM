from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.models.dataset_modulation import DatasetConditionedFiLM
from basicstfm.models.diffusion_mechanism_learner import DiffusionMechanismLearner
from basicstfm.models.dpm_stfm import SRDSTFMBackbone
from basicstfm.models.fusion_predictor import FusionPredictor
from basicstfm.models.residual_event_encoder import ResidualEventEncoder
from basicstfm.models.stable_trunk_encoder import StableTrunkEncoder
from basicstfm.utils.checkpoint import adapt_checkpoint_state_dict, load_checkpoint, restore_rng_state


class WrappedTrunk(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.stable_trunk = StableTrunkEncoder(
            input_dim=input_dim,
            hidden_dim=32,
            output_len=12,
            output_dim=output_dim,
            use_frequency_branch=True,
            stable_mixer_layers=1,
        )


class WrappedTrunkAndResidual(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.stable_trunk = StableTrunkEncoder(
            input_dim=input_dim,
            hidden_dim=32,
            output_len=12,
            output_dim=output_dim,
            use_frequency_branch=True,
            stable_mixer_layers=1,
        )
        self.residual_event_encoder = ResidualEventEncoder(input_dim=input_dim, hidden_dim=32)


class CheckpointTest(unittest.TestCase):
    def test_load_checkpoint_can_skip_rng_restore(self):
        model = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "rng_state": {"torch": [1, 2, 3, 4]},
                },
                path,
            )
            other = torch.nn.Linear(4, 2)
            info = load_checkpoint(str(path), other, strict=True, restore_rng=False)
            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["unexpected_keys"], [])

    def test_stable_trunk_channel_inflate_single_to_multi_channel(self):
        pre = WrappedTrunk(input_dim=1, output_dim=1)
        tgt = WrappedTrunk(input_dim=18, output_dim=18)
        ckpt_state = {k: v for k, v in pre.state_dict().items() if k.startswith("stable_trunk.")}
        adapted = adapt_checkpoint_state_dict(
            tgt,
            ckpt_state,
            stable_trunk_channel_inflate=True,
        )
        local = adapted["stable_trunk.local_branch.0.weight"]
        self.assertEqual(tuple(local.shape), (32, 18, 3, 1))
        torch.testing.assert_close(
            local[:, 0:1, :, :],
            pre.state_dict()["stable_trunk.local_branch.0.weight"].to(local.dtype),
        )

    def test_diffusion_output_proj_output_dim_inflate(self):
        class M(torch.nn.Module):
            def __init__(self, output_dim: int) -> None:
                super().__init__()
                self.diffusion_mechanism_learner = DiffusionMechanismLearner(
                    hidden_dim=32,
                    output_dim=output_dim,
                    num_datasets=3,
                )

        pre = M(1)
        tgt = M(18)
        ckpt = {
            k: v
            for k, v in pre.state_dict().items()
            if "diffusion_mechanism_learner.output_proj" in k
        }
        adapted = adapt_checkpoint_state_dict(tgt, ckpt, stable_trunk_channel_inflate=True)
        w = adapted["diffusion_mechanism_learner.output_proj.weight"]
        self.assertEqual(tuple(w.shape), (18, 32))
        torch.testing.assert_close(
            w[0:1, :],
            pre.state_dict()["diffusion_mechanism_learner.output_proj.weight"].to(w.dtype),
        )

    def test_dataset_modulation_embedding_row_align(self):
        class M(torch.nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                self.diffusion_mechanism_learner = torch.nn.Module()
                self.diffusion_mechanism_learner.dataset_modulation = DatasetConditionedFiLM(
                    hidden_dim=32,
                    num_datasets=n,
                )

        pre = M(15)
        tgt = M(5)
        ckpt = {"diffusion_mechanism_learner.dataset_modulation.embedding.weight": pre.state_dict()[
            "diffusion_mechanism_learner.dataset_modulation.embedding.weight"
        ]}
        adapted = adapt_checkpoint_state_dict(tgt, ckpt, stable_trunk_channel_inflate=True)
        w = adapted["diffusion_mechanism_learner.dataset_modulation.embedding.weight"]
        self.assertEqual(tuple(w.shape), (5, 32))
        exp = pre.state_dict()["diffusion_mechanism_learner.dataset_modulation.embedding.weight"][:5].to(
            w.dtype
        )
        torch.testing.assert_close(w, exp)

    def test_residual_value_proj_channel_inflate_with_stable_trunk(self):
        pre = WrappedTrunkAndResidual(input_dim=1, output_dim=1)
        tgt = WrappedTrunkAndResidual(input_dim=18, output_dim=18)
        ckpt_state = {
            k: v
            for k, v in pre.state_dict().items()
            if k.startswith("stable_trunk.") or k.startswith("residual_event_encoder.value_proj")
        }
        adapted = adapt_checkpoint_state_dict(
            tgt,
            ckpt_state,
            stable_trunk_channel_inflate=True,
        )
        w = adapted["residual_event_encoder.value_proj.weight"]
        self.assertEqual(tuple(w.shape), (32, 18))
        torch.testing.assert_close(
            w[:, 0:1],
            pre.state_dict()["residual_event_encoder.value_proj.weight"].to(w.dtype),
        )

    def test_fusion_predictor_additive_logit_inflate(self):
        class M(torch.nn.Module):
            def __init__(self, odim: int) -> None:
                super().__init__()
                self.fusion_predictor = FusionPredictor(output_dim=odim, fusion_mode="additive")

        pre = M(1)
        post = M(18)
        ckpt = {"fusion_predictor.additive_logit": pre.state_dict()["fusion_predictor.additive_logit"]}
        out = adapt_checkpoint_state_dict(post, ckpt, stable_trunk_channel_inflate=True)
        t = out["fusion_predictor.additive_logit"]
        self.assertEqual(tuple(t.shape), (1, 1, 1, 18))
        torch.testing.assert_close(
            t[..., 0:1],
            pre.state_dict()["fusion_predictor.additive_logit"].to(t.dtype),
        )

    def test_srd_backbone_full_monash_to_mixed_adapt(self):
        pre = SRDSTFMBackbone(
            num_nodes=32,
            input_dim=1,
            output_dim=1,
            input_len=96,
            output_len=96,
            hidden_dim=64,
            num_datasets=15,
            use_calibration_head=True,
        )
        post = SRDSTFMBackbone(
            num_nodes=32,
            input_dim=18,
            output_dim=18,
            input_len=96,
            output_len=96,
            hidden_dim=64,
            num_datasets=5,
            use_calibration_head=True,
        )
        ckpt = pre.state_dict()
        merged = adapt_checkpoint_state_dict(post, ckpt, stable_trunk_channel_inflate=True)
        self.assertEqual(set(merged.keys()), set(post.state_dict().keys()))
        post.load_state_dict(merged, strict=True)

    def test_stable_trunk_channel_inflate_mismatch_still_raises(self):
        pre = WrappedTrunk(input_dim=1, output_dim=1)
        tgt = WrappedTrunk(input_dim=18, output_dim=18)
        bad = dict(tgt.state_dict())
        bad["stable_trunk.branch_logits"] = torch.zeros(5)
        with self.assertRaises(RuntimeError):
            adapt_checkpoint_state_dict(
                tgt,
                bad,
                stable_trunk_channel_inflate=True,
            )

    def test_foundation_channel_inflate_opencity_value_proj(self):
        from basicstfm.models.foundation.opencity import OpenCityFoundationModel

        pre = OpenCityFoundationModel(
            num_nodes=1,
            input_dim=1,
            output_dim=1,
            input_len=12,
            output_len=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_num_nodes=8,
        )
        post = OpenCityFoundationModel(
            num_nodes=1,
            input_dim=18,
            output_dim=18,
            input_len=12,
            output_len=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_num_nodes=8,
        )
        ck = pre.state_dict()
        adapted = adapt_checkpoint_state_dict(post, ck, foundation_channel_inflate=True)
        post.load_state_dict(adapted, strict=True)
        w = post.value_proj.weight
        self.assertEqual(tuple(w.shape), (32, 18))
        self.assertTrue(torch.allclose(w[:, 0], ck["value_proj.weight"][:, 0].to(w.dtype)))

    def test_restore_rng_state_accepts_python_lists(self):
        state = {"torch": torch.get_rng_state().tolist()}
        restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()
