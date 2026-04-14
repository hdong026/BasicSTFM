import unittest

import numpy as np

from basicstfm.data.scaler import StandardScaler


class ScalerTest(unittest.TestCase):
    def test_standard_scaler_round_trip_numpy(self):
        array = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        scaler = StandardScaler().fit(array)

        transformed = scaler.transform(array)
        restored = scaler.inverse_transform(transformed)

        np.testing.assert_allclose(restored, array, rtol=1e-6, atol=1e-6)

    def test_standard_scaler_broadcasts_batch_shape(self):
        train = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        batch = np.arange(48, dtype=np.float32).reshape(2, 4, 3, 2)
        scaler = StandardScaler().fit(train)

        transformed = scaler.transform(batch)
        restored = scaler.inverse_transform(transformed)

        np.testing.assert_allclose(restored, batch, rtol=1e-6, atol=1e-6)

    def test_standard_scaler_ignores_nan_statistics_and_imputes_missing_values(self):
        train = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        train[0, 0, 0] = np.nan
        batch = train.copy()
        batch[1, 1, 1] = np.nan

        scaler = StandardScaler().fit(train)
        transformed = scaler.transform(batch)

        self.assertTrue(np.isfinite(scaler.mean).all())
        self.assertTrue(np.isfinite(scaler.std).all())
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
