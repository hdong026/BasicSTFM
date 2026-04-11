from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.utils.checkpoint import load_checkpoint, restore_rng_state


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

    def test_restore_rng_state_accepts_python_lists(self):
        state = {"torch": torch.get_rng_state().tolist()}
        restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()
