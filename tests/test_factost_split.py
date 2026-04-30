import unittest

from basicstfm.data.factost_split import parse_dataset_datesplit


class FactostSplitParseTest(unittest.TestCase):
    def test_three_int_lengths(self) -> None:
        self.assertEqual(parse_dataset_datesplit(1000, [700, 100, 200]), (700, 100, 200))

    def test_train_val_test_ranges_contiguous(self) -> None:
        spec = {"train": [0, 700], "val": [700, 800], "test": [800, 1000]}
        self.assertEqual(parse_dataset_datesplit(1000, spec), (700, 100, 200))

    def test_train_end_cuts(self) -> None:
        spec = {"train_end": 600, "val_end": 800, "test_end": 1000}
        self.assertEqual(parse_dataset_datesplit(1000, spec), (600, 200, 200))


if __name__ == "__main__":
    unittest.main()
