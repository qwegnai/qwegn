#!/usr/bin/env python3
# Unit tests for Scripts/quantize/emeans.py (arXiv:2611.09501 contract).
from __future__ import annotations

import unittest
import numpy as np
import torch

from emeans import (
    EmeansCodebook,
    EmeansRegime,
    RegimeViolationError,
    assign_emeans,
    emeans_schedule,
    fit_emeans,
)


def _lloyd(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32).copy()
    x32 = x.astype(np.float32)
    x_norm2 = (x32 * x32).sum(axis=1, keepdims=True)
    for _ in range(iters):
        c_norm2 = (c * c).sum(axis=1)
        dist = x_norm2 + c_norm2[None, :] - 2.0 * (x32 @ c.T)
        idx = dist.argmin(axis=1)
        for j in range(k):
            m = idx == j
            if m.any():
                c[j] = x32[m].mean(axis=0)
    c_norm2 = (c * c).sum(axis=1)
    mse = float((x_norm2 + c_norm2[None, :] - 2.0 * (x32 @ c.T)).min(axis=1).mean())
    return c, mse


def _eval_mse(x: np.ndarray, cb: EmeansCodebook) -> float:
    idx = assign_emeans(x, cb, device="cpu")
    diff = x.astype(np.float32) - cb.centroids[idx]
    return float((diff * diff).sum(axis=1).mean())


class EmeansScheduleTests(unittest.TestCase):
    def test_refuses_sub_threshold_n_over_k(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((128, 16), dtype=np.float32)
        with self.assertRaises(RegimeViolationError):
            fit_emeans(x, k=64, regime=EmeansRegime.IN_SAMPLE_STATIC, device="cpu")

    def test_power_law_schedule_bounds(self):
        b_knee, a_knee = emeans_schedule(n=100_000, k=256, epochs=2.0, batch_mode="knee")
        b_val, a_val = emeans_schedule(n=100_000, k=256, epochs=2.0, batch_mode="valley")
        self.assertEqual(b_knee & (b_knee - 1), 0)
        self.assertEqual(b_val & (b_val - 1), 0)
        self.assertLessEqual(b_val, b_knee)
        self.assertGreater(a_knee, 0.0)
        self.assertLessEqual(a_knee, 1.0)
        self.assertGreater(a_val, 0.0)
        self.assertLessEqual(a_val, 1.0)


class EmeansConvergenceTests(unittest.TestCase):
    def _clustered_data(self, n: int = 16_384, d: int = 16, k: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((k, d), dtype=np.float32) * 2.5
        ids = rng.integers(0, k, size=n)
        noise = rng.standard_normal((n, d), dtype=np.float32) * 0.6
        return (centers[ids] + noise).astype(np.float32)

    def test_two_epoch_emeans_beats_or_matches_lloyd25(self):
        x = self._clustered_data(n=16_384, d=16, k=64, seed=7)
        _, lloyd_mse = _lloyd(x, k=64, iters=25, seed=7)
        cb = fit_emeans(x, k=64, regime=EmeansRegime.IN_SAMPLE_STATIC,
                        epochs=2.5, seed=7, device="cpu")
        em_mse = _eval_mse(x, cb)
        self.assertLessEqual(em_mse, lloyd_mse * 1.01,
                             f"e-means MSE {em_mse:.6f} exceeded Lloyd@25 {lloyd_mse:.6f}")

    def test_dead_cluster_revival_eliminates_empty_clusters(self):
        rng = np.random.default_rng(11)
        # Skewed mixture: 60 clusters dense + 4 far outliers so random init leaves dead centroids.
        main = rng.standard_normal((16_000, 8), dtype=np.float32)
        outliers = rng.standard_normal((384, 8), dtype=np.float32) * 0.05 + 50.0
        x = np.concatenate([main, outliers], axis=0)
        cb = fit_emeans(x, k=128, regime=EmeansRegime.IN_SAMPLE_STATIC,
                        epochs=3.0, seed=11, device="cpu")
        idx = assign_emeans(x, cb, device="cpu")
        counts = np.bincount(idx, minlength=128)
        self.assertEqual(int((counts == 0).sum()), 0)

    def test_last_epoch_mean_vs_slow_ema_regimes(self):
        rng = np.random.default_rng(23)
        x_train = rng.standard_normal((8_192, 4), dtype=np.float32)
        x_test = rng.standard_normal((8_192, 4), dtype=np.float32)
        cb_static = fit_emeans(x_train, k=64, regime=EmeansRegime.IN_SAMPLE_STATIC,
                               epochs=2.0, seed=23, device="cpu")
        cb_calib = fit_emeans(x_train, k=64, regime=EmeansRegime.OUT_OF_SAMPLE_CALIB,
                              epochs=2.0, seed=23, device="cpu")
        self.assertIsNone(cb_static.biases)
        self.assertLessEqual(_eval_mse(x_train, cb_static), _eval_mse(x_train, cb_calib) * 1.005)
        self.assertLess(_eval_mse(x_test, cb_calib), 1.5)

    def test_balanced_routing_produces_uniform_cluster_loads(self):
        # Strongly skewed data where unconstrained k-means creates >4x load imbalance.
        rng = np.random.default_rng(99)
        dense = rng.standard_normal((12_000, 16), dtype=np.float32) * 0.5
        diffuse = rng.standard_normal((4_384, 16), dtype=np.float32) * 3.0 + 4.0
        x = np.concatenate([dense, diffuse], axis=0)
        cb = fit_emeans(x, k=32, regime=EmeansRegime.BALANCED_ROUTING,
                        epochs=3.0, seed=99, device="cpu")
        self.assertIsNotNone(cb.biases)
        idx = assign_emeans(x, cb, device="cpu")
        counts = np.bincount(idx, minlength=32)
        target = len(x) / 32.0
        self.assertLess(counts.max() / target, 1.35, f"max/target={counts.max()/target:.3f}")
        self.assertGreater(counts.min() / target, 0.65, f"min/target={counts.min()/target:.3f}")

    def test_casted_norm_fp16_matches_fp32_closely(self):
        x = self._clustered_data(n=8_192, d=32, k=64, seed=5)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        cb32 = fit_emeans(x, k=64, regime=EmeansRegime.IN_SAMPLE_STATIC,
                          epochs=2.0, seed=5, device="cpu", compute_dtype="float32")
        cb16 = fit_emeans(x, k=64, regime=EmeansRegime.IN_SAMPLE_STATIC,
                          epochs=2.0, seed=5, device=dev, compute_dtype="float16")
        mse32 = _eval_mse(x, cb32)
        mse16 = _eval_mse(x, cb16)
        self.assertLess(abs(mse16 - mse32) / mse32, 0.02)


if __name__ == "__main__":
    unittest.main()
