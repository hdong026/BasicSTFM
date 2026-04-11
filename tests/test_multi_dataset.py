from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.builders import import_builtin_components
from basicstfm.data.datamodule import MultiDatasetWindowDataModule, WindowDataModule
from basicstfm.engines.trainer import MultiStageTrainer
from basicstfm.losses.common import LossCollection
from basicstfm.registry import MODELS, TASKS


class MultiDatasetDataModuleTest(unittest.TestCase):
    def test_joint_pretrain_loader_exposes_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_a = root / "A"
            ds_b = root / "B"
            ds_a.mkdir()
            ds_b.mkdir()

            np.savez(ds_a / "data.npz", data=np.random.randn(48, 5, 1).astype(np.float32))
            np.savez(ds_a / "adj.npz", adj=np.eye(5, dtype=np.float32))
            np.savez(ds_b / "data.npz", data=np.random.randn(48, 7, 2).astype(np.float32))
            np.savez(ds_b / "adj.npz", adj=np.eye(7, dtype=np.float32))

            datamodule = MultiDatasetWindowDataModule(
                datasets=[
                    {"name": "A", "data_path": str(ds_a / "data.npz"), "graph_path": str(ds_a / "adj.npz")},
                    {"name": "B", "data_path": str(ds_b / "data.npz"), "graph_path": str(ds_b / "adj.npz")},
                ],
                input_len=4,
                output_len=2,
                batch_size=3,
                split=(0.6, 0.2, 0.2),
                train_strategy="round_robin",
                eval_strategy="per_dataset",
                steps_per_epoch=4,
            )
            datamodule.setup()
            metadata = datamodule.get_metadata()
            self.assertEqual(metadata["num_nodes"], 7)
            self.assertEqual(metadata["num_channels"], 2)

            train_loader = datamodule.train_dataloader()
            seen = set()
            for batch in train_loader:
                seen.add(batch["dataset_name"])
                self.assertIn("x_mask", batch)
                self.assertIn("y_mask", batch)
                self.assertEqual(batch["x"].shape[-1], 2)
                self.assertEqual(batch["y"].shape[-1], 2)
            self.assertEqual(seen, {"A", "B"})

            val_loaders = datamodule.val_dataloader()
            self.assertIsInstance(val_loaders, dict)
            self.assertEqual(set(val_loaders.keys()), {"A", "B"})

    def test_multi_dataset_batches_flow_through_foundation_task(self):
        import_builtin_components()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_a = root / "A"
            ds_b = root / "B"
            ds_a.mkdir()
            ds_b.mkdir()

            np.savez(ds_a / "data.npz", data=np.random.randn(64, 5, 1).astype(np.float32))
            np.savez(ds_a / "adj.npz", adj=np.eye(5, dtype=np.float32))
            np.savez(ds_b / "data.npz", data=np.random.randn(64, 7, 2).astype(np.float32))
            np.savez(ds_b / "adj.npz", adj=np.eye(7, dtype=np.float32))

            datamodule = MultiDatasetWindowDataModule(
                datasets=[
                    {"name": "A", "data_path": str(ds_a / "data.npz"), "graph_path": str(ds_a / "adj.npz")},
                    {"name": "B", "data_path": str(ds_b / "data.npz"), "graph_path": str(ds_b / "adj.npz")},
                ],
                input_len=6,
                output_len=3,
                batch_size=2,
                split=(0.6, 0.2, 0.2),
                train_strategy="round_robin",
                eval_strategy="combined",
                steps_per_epoch=2,
            )
            datamodule.setup()
            metadata = datamodule.get_metadata()

            model = MODELS.build(
                {
                    "type": "OpenCityFoundationModel",
                    "num_nodes": metadata["num_nodes"],
                    "input_dim": metadata["num_channels"],
                    "output_dim": metadata["num_channels"],
                    "input_len": metadata["input_len"],
                    "output_len": metadata["target_len"],
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                }
            )
            task = TASKS.build({"type": "ForecastingTask"})
            task.set_scaler(datamodule.get_scaler())
            losses = LossCollection([{"type": "mae"}]).to(torch.device("cpu"))

            for batch in datamodule.train_dataloader():
                out = task.step(model, batch, losses, torch.device("cpu"))
                self.assertEqual(out["pred"].shape[-1], metadata["num_channels"])
                self.assertEqual(out["target"].shape[-1], metadata["num_channels"])
                self.assertIsNotNone(out["mask"])

    def test_named_eval_logs_are_aggregated_macro_style(self):
        logs = MultiStageTrainer._aggregate_named_eval_logs(
            {
                "A": {"val/loss/total": 2.0, "val/metric/mae": 1.0},
                "B": {"val/loss/total": 4.0, "val/metric/mae": 3.0},
            },
            prefix="val",
        )
        self.assertAlmostEqual(logs["val/loss/total"], 3.0)
        self.assertAlmostEqual(logs["val/metric/mae"], 2.0)
        self.assertEqual(logs["val/dataset/A/loss/total"], 2.0)
        self.assertEqual(logs["val/dataset/B/metric/mae"], 3.0)

    def test_single_dataset_module_uses_distributed_sampler_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.savez(root / "data.npz", data=np.random.randn(64, 5, 1).astype(np.float32))
            np.savez(root / "adj.npz", adj=np.eye(5, dtype=np.float32))

            datamodule = WindowDataModule(
                data_path=str(root / "data.npz"),
                graph_path=str(root / "adj.npz"),
                input_len=6,
                output_len=3,
                batch_size=4,
                split=(0.6, 0.2, 0.2),
                distributed=True,
                world_size=2,
                rank=0,
            )
            datamodule.setup()
            loader = datamodule.train_dataloader()
            self.assertIsInstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)

    def test_multi_dataset_module_uses_distributed_sampler_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_a = root / "A"
            ds_b = root / "B"
            ds_a.mkdir()
            ds_b.mkdir()

            np.savez(ds_a / "data.npz", data=np.random.randn(48, 5, 1).astype(np.float32))
            np.savez(ds_b / "data.npz", data=np.random.randn(48, 7, 1).astype(np.float32))

            datamodule = MultiDatasetWindowDataModule(
                datasets=[
                    {"name": "A", "data_path": str(ds_a / "data.npz")},
                    {"name": "B", "data_path": str(ds_b / "data.npz")},
                ],
                input_len=4,
                output_len=2,
                batch_size=2,
                split=(0.6, 0.2, 0.2),
                distributed=True,
                world_size=2,
                rank=0,
            )
            datamodule.setup()
            loader = datamodule.train_dataloader()
            self.assertTrue(loader.loaders)
            for subloader in loader.loaders.values():
                self.assertIsInstance(
                    subloader.sampler,
                    torch.utils.data.distributed.DistributedSampler,
                )


if __name__ == "__main__":
    unittest.main()
