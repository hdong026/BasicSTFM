from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from basicstfm.data.revin import factost_value_revin_inverse, factost_value_revin_normalize


class FactostRevinTest(unittest.TestCase):
    def test_normalize_inverse_round_trip(self):
        b, t, n, c = 3, 12, 5, 3
        x = torch.randn(b, t, n, c)
        y = torch.randn(b, 4, n, c)
        batch: dict = {}
        xv, yv = factost_value_revin_normalize(x, y, batch, value_channels=0, eps=1e-5)
        xr = factost_value_revin_inverse(xv, batch)
        yr = factost_value_revin_inverse(yv, batch)
        self.assertTrue(torch.allclose(x[..., 0:1], xr[..., 0:1], atol=1e-4, rtol=1e-3))
        self.assertTrue(torch.allclose(x[..., 1:], xv[..., 1:], atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(y[..., 0:1], yr[..., 0:1], atol=1e-4, rtol=1e-3))

    def test_std_uses_input_time_only(self):
        b, t, n = 2, 8, 4
        x = torch.arange(b * t * n, dtype=torch.float32).view(b, t, n, 1) * 0.1 + 1.0
        y = torch.ones(b, 3, n, 1) * 999.0
        batch = {}
        _, yv = factost_value_revin_normalize(x, y, batch, value_channels=0, eps=1e-5)
        m = batch["revin_mean"]
        s = batch["revin_std"]
        self.assertEqual(tuple(m.shape), (b, 1, n, 1))
        exp = (y[0, 0, 0, 0] - m[0, 0, 0, 0]) / s[0, 0, 0, 0]
        self.assertAlmostEqual(float(yv[0, 0, 0, 0]), float(exp), places=4)
