import json
import os
import tempfile
import unittest

from basicstfm.config import deep_merge, expand_dataset_registry, load_config, set_by_dotted_key


class ConfigTest(unittest.TestCase):
    def test_json_config_and_overrides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
            json.dump({"trainer": {"device": "auto"}, "stages": [{"epochs": 3}]}, fp)
            path = fp.name
        try:
            cfg = load_config(path, overrides=["trainer.device=\"cpu\"", "stages.0.epochs=1"])
        finally:
            os.unlink(path)
        self.assertEqual(cfg["trainer"]["device"], "cpu")
        self.assertEqual(cfg["stages"][0]["epochs"], 1)

    def test_deep_merge_keeps_nested_values(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        self.assertEqual(merged, {"a": {"b": 9, "c": 2}})

    def test_set_by_dotted_key_creates_dicts(self):
        cfg = {}
        set_by_dotted_key(cfg, "a.b.c", 7)
        self.assertEqual(cfg["a"]["b"]["c"], 7)

    def test_expand_dataset_registry_for_single_and_multi_dataset_recipes(self):
        cfg = {
            "dataset_registry": {
                "METR-LA": {
                    "name": "METR-LA",
                    "data_path": "data/METR-LA/data.npz",
                    "graph_path": "data/METR-LA/adj.npz",
                },
                "PEMS-BAY": {
                    "name": "PEMS-BAY",
                    "data_path": "data/PEMS-BAY/data.npz",
                    "graph_path": "data/PEMS-BAY/adj.npz",
                },
                "PEMS08": {
                    "name": "PEMS08",
                    "data_path": "data/PEMS08/data.npz",
                    "graph_path": "data/PEMS08/adj.npz",
                },
            },
            "dataset_groups": {
                "sources": ["METR-LA", "PEMS-BAY"],
            },
            "data": {
                "type": "WindowDataModule",
                "dataset_key": "PEMS08",
                "input_len": 12,
                "output_len": 12,
            },
            "pipeline": {
                "stages": [
                    {
                        "name": "pretrain",
                        "task": {"type": "MaskedReconstructionTask"},
                        "data": {
                            "type": "MultiDatasetWindowDataModule",
                            "dataset_group": "sources",
                            "input_len": 12,
                            "output_len": 12,
                        },
                    }
                ]
            },
        }

        expanded = expand_dataset_registry(cfg)
        self.assertNotIn("dataset_key", expanded["data"])
        self.assertEqual(expanded["data"]["data_path"], "data/PEMS08/data.npz")
        stage_data = expanded["pipeline"]["stages"][0]["data"]
        self.assertNotIn("dataset_group", stage_data)
        self.assertEqual(stage_data["datasets"][0]["name"], "METR-LA")
        self.assertEqual(stage_data["datasets"][1]["graph_path"], "data/PEMS-BAY/adj.npz")


if __name__ == "__main__":
    unittest.main()
