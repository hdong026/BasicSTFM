import json
import os
import tempfile
import unittest

from basicstfm.config import deep_merge, load_config, set_by_dotted_key


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


if __name__ == "__main__":
    unittest.main()
