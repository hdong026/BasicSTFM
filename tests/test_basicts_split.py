"""BasicTS-aligned traffic split ratios."""

from __future__ import annotations

import unittest

from basicstfm.data.basicts_split import (
    BASICTS_TRAFFIC_TRAIN_VAL_TEST_RATIO,
    basicts_lengths_from_ratio,
    canonical_basicts_traffic_key,
    resolve_basicts_split_lengths,
)


class TestBasictsSplit(unittest.TestCase):
    def test_canonical_strip_suffix(self) -> None:
        self.assertEqual(canonical_basicts_traffic_key("PEMS04_BasicTS"), "PEMS04")

    def test_pems04_ratio_matches_basic_ts_script(self) -> None:
        self.assertEqual(BASICTS_TRAFFIC_TRAIN_VAL_TEST_RATIO["PEMS04"], (0.6, 0.2, 0.2))

    def test_metr_la_ratio_differs_from_pems_flow(self) -> None:
        self.assertEqual(BASICTS_TRAFFIC_TRAIN_VAL_TEST_RATIO["METR-LA"], (0.7, 0.1, 0.2))

    def test_lengths_integer_floor_like_basic_ts(self) -> None:
        tr, va, te = basicts_lengths_from_ratio(16992, (0.6, 0.2, 0.2))
        self.assertEqual(tr, int(16992 * 0.6))
        self.assertEqual(va, int(16992 * 0.2))
        self.assertEqual(tr + va + te, 16992)

    def test_resolve_known_dataset(self) -> None:
        triple, meta = resolve_basicts_split_lengths(
            total_timesteps=1000,
            dataset_key="PEMS04",
            fallback_split=(0.7, 0.1, 0.2),
        )
        self.assertEqual(meta["resolution"], "basic_ts_official")
        self.assertEqual(sum(triple), 1000)


if __name__ == "__main__":
    unittest.main()
