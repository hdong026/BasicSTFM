from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from basicstfm.utils.results import (
    build_markdown_table,
    discover_stage_result_files,
    filter_stage_rows,
    flatten_stage_results,
    load_stage_result_payload,
    summarize_stage_rows,
)


class ResultsExportTest(unittest.TestCase):
    def test_discover_flatten_and_summarize_stage_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "runs" / "exp_a" / "results" / "stage_results.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "experiment_name": "exp_a",
                "work_dir": str(root / "runs" / "exp_a"),
                "stages": [
                    {
                        "name": "zero_shot_test",
                        "stage_index": 1,
                        "task": "ForecastingTask",
                        "model_type": "UniSTFoundationModel",
                        "data_type": "WindowDataModule",
                        "resolved_data": {
                            "data_path": "data/METR-LA/data.npz",
                            "graph_path": "data/METR-LA/adj.npz",
                            "input_len": 12,
                            "output_len": 12,
                            "batch_size": 16,
                        },
                        "test": {
                            "test/metric/mae": 3.21,
                            "test/metric/rmse": 5.43,
                        },
                    }
                ],
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            files = discover_stage_result_files([root / "runs"])
            self.assertEqual(files, [result_path])

            loaded = load_stage_result_payload(result_path)
            rows = flatten_stage_results(loaded, result_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "METR-LA")
            self.assertEqual(rows[0]["test/metric/mae"], 3.21)

            filtered = filter_stage_rows(rows, stages=["zero_shot_test"], dataset="METR-LA")
            self.assertEqual(len(filtered), 1)

            summary = summarize_stage_rows(filtered, split="test", metrics=["metric/mae", "metric/rmse"])
            self.assertEqual(summary[0]["test/metric/mae"], 3.21)
            self.assertEqual(summary[0]["test/metric/rmse"], 5.43)

            markdown = build_markdown_table(summary)
            self.assertIn("experiment_name", markdown)
            self.assertIn("test/metric/mae", markdown)
            self.assertIn("exp_a", markdown)


if __name__ == "__main__":
    unittest.main()
