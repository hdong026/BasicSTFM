from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

from basicstfm.builders import import_builtin_components
from basicstfm.registry import DATAMODULES


class PartitionedWindowDataModuleTest(unittest.TestCase):
    def test_partitioned_datamodule_builds_partition_loaders(self):
        import_builtin_components()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = np.random.randn(64, 10, 1).astype(np.float32)
            graph = np.eye(10, dtype=np.float32)
            graph[0, 1] = 1.0
            graph[1, 0] = 1.0
            graph[4, 5] = 1.0
            graph[5, 4] = 1.0
            np.save(root / "data.npy", data)
            np.save(root / "adj.npy", graph)

            datamodule = DATAMODULES.build(
                {
                    "type": "PartitionedWindowDataModule",
                    "data_path": str(root / "data.npy"),
                    "graph_path": str(root / "adj.npy"),
                    "input_len": 12,
                    "output_len": 6,
                    "batch_size": 3,
                    "partition_size": 4,
                    "partition_strategy": "graph_greedy",
                    "mmap_mode": "r",
                }
            )
            datamodule.setup()

            metadata = datamodule.get_metadata()
            self.assertEqual(metadata["num_channels"], 1)
            self.assertEqual(metadata["input_len"], 12)
            self.assertEqual(metadata["target_len"], 6)
            self.assertGreaterEqual(metadata["num_partitions"], 3)

            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))
            self.assertEqual(tuple(batch["x"].shape[1:]), (12, 4, 1))
            self.assertEqual(tuple(batch["y"].shape[1:]), (6, 4, 1))
            self.assertEqual(tuple(batch["graph"].shape), (4, 4))
            self.assertEqual(tuple(batch["x_mask"].shape), tuple(batch["x"].shape))
            self.assertEqual(tuple(batch["y_mask"].shape), tuple(batch["y"].shape))

            val_loaders = datamodule.val_dataloader()
            self.assertIsInstance(val_loaders, dict)
            self.assertTrue(val_loaders)
            first_val_batch = next(iter(next(iter(val_loaders.values()))))
            self.assertEqual(tuple(first_val_batch["graph"].shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
