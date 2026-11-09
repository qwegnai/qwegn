"""HQ8_G32_M5 cooker tests: symmetric Hadamard-rotated INT8 planar planes."""
import unittest

import numpy as np

from pack import (GROUP, HQ8_PAYLOAD_BYTES, dequant_hq8_g32_m5,
                  dequant_hq8_g32_m5_rot, quant_hq8_g32_m5)


class HQ8CookerTests(unittest.TestCase):
    def test_shapes_ranges_and_symmetry(self):
        rng = np.random.default_rng(7)
        w = rng.standard_normal((64, 2560)).astype(np.float32)
        qs, sc = quant_hq8_g32_m5(w)
        self.assertEqual(qs.shape, w.shape)
        self.assertEqual(qs.dtype, np.int8)
        self.assertEqual(sc.shape, (64, 2560 // GROUP))
        self.assertEqual(sc.dtype, np.float16)
        self.assertGreaterEqual(int(qs.min()), -127)
        self.assertLessEqual(int(qs.max()), 127)
        self.assertEqual(HQ8_PAYLOAD_BYTES, 32)

    def test_roundtrip_beats_hq4_error_class(self):
        rng = np.random.default_rng(11)
        w = rng.standard_normal((64, 2560)).astype(np.float32)
        w += (rng.standard_t(3, w.shape) * 0.05).astype(np.float32)
        w_hat = dequant_hq8_g32_m5(*quant_hq8_g32_m5(w))
        rel = float(np.sqrt(((w - w_hat) ** 2).mean()) / np.sqrt((w ** 2).mean()))
        self.assertLess(rel, 0.015, f"relative RMSE {rel}")

    def test_zero_groups_stay_zero(self):
        w = np.zeros((1, 64), np.float32)
        w[0, :32] = 1e-3
        qs, sc = quant_hq8_g32_m5(w)
        self.assertEqual(int(np.abs(qs[0, 32:]).max()), 0)
        self.assertEqual(float(sc[0, 1]), 0.0)

    def test_rot_dequant_matches_scale_law(self):
        rng = np.random.default_rng(3)
        w = rng.standard_normal((8, 96)).astype(np.float32)
        qs, sc = quant_hq8_g32_m5(w)
        g = dequant_hq8_g32_m5_rot(qs, sc).reshape(8, 3, 32)
        for row in range(8):
            for grp in range(3):
                s = float(sc[row, grp])
                if s == 0.0:
                    self.assertTrue(np.all(g[row, grp] == 0.0))
                else:
                    self.assertTrue(np.all(np.abs(g[row, grp]) <= 127.5 * s))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            quant_hq8_g32_m5(np.zeros((4, 30), np.float32))
        bad = np.zeros((4, 32), np.float32)
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            quant_hq8_g32_m5(bad)


    def test_target_selects_plain_gemv_dense_only(self):
        from pack import hq8_target
        ok = ["a.self_attn.q_proj.weight", "a.self_attn.o_proj.weight",
              "a.linear_attn.out_proj.weight", "a.ple.key_proj.weight"]
        deny = ["a.self_attn.mlp.experts.gate", "a.ngram_embedding.w",
                "a.linear_attn.in_proj_qkv.weight", "a.linear_attn.in_proj_z.weight",
                "mtp.layers.0.self_attn.q_proj.weight", "a.embed_tokens.weight"]
        for name in ok:
            self.assertTrue(hq8_target(name), name)
        for name in deny:
            self.assertFalse(hq8_target(name), name)


if __name__ == "__main__":
    unittest.main()
