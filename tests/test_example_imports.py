from __future__ import annotations

import importlib
import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")


class ExampleImportTest(unittest.TestCase):
    def test_examples_package_is_importable(self):
        custom_loss = importlib.import_module("examples.custom_loss")
        custom_task = importlib.import_module("examples.custom_task")

        self.assertTrue(hasattr(custom_loss, "SmoothL1MAEMix"))
        self.assertTrue(hasattr(custom_task, "ForecastWithAuxEmbeddingPenalty"))


if __name__ == "__main__":
    unittest.main()
