import unittest

import numpy as np
import torch

from hq_blocks import dequant_hq_blocks, parse_quant_spec, quant_hq_blocks
from hq_fakequant import fake_quant

SPECS = ("hq4_g32_m5", "hq4_g32", "hq4_g64", "hq4_g128", "hq3_g32", "hq3_g64", "hq6_g32", "hq8_g32", "hq2_g32")


class FakeQuantTests(unittest.TestCase):
    def test_matches_byte_writers(self):
        gen = torch.Generator().manual_seed(7)
        w = torch.randn(96, 640, generator=gen) * 0.02
        w[3, 17] = 0.4
        w[5, 64:128] = 0.0
        for spec in SPECS:
            bits, group, m5 = parse_quant_spec(spec)
            blocks, eff = quant_hq_blocks(w, bits=bits, group=group, is_m5=m5)
            expected = dequant_hq_blocks(blocks, bits=bits, group=eff, is_m5=m5)
            actual = fake_quant(w, spec).numpy()
            mismatch = np.mean(np.abs(actual - expected) > 1e-6)
            self.assertLess(mismatch, 1e-3, spec)

    def test_group_resolution_and_expert_stacks(self):
        w = torch.randn(4, 8, 320)
        self.assertEqual(fake_quant(w, "hq4_g128").shape, w.shape)
        flat = fake_quant(w.reshape(32, 320), "hq4_g128").reshape(w.shape)
        self.assertTrue(torch.equal(flat, fake_quant(w, "hq4_g128")))

    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            fake_quant(torch.full((2, 32), float("nan")), "hq4_g32")
        with self.assertRaises(ValueError):
            fake_quant(torch.randn(2, 30), "hq4_g32")


if __name__ == "__main__":
    unittest.main()
