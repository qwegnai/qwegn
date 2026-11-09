"""Unit tests for HQ{4,6,8}_G{32,64,128,256} block quantizer, mixed pack, and verifier."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from hq4 import quant_hq4_blocks, dequant_hq4_g32_m5
from hq_blocks import (
    VALID_BITS,
    VALID_GROUPS,
    parse_quant_spec,
    format_quant_spec,
    resolve_group_size,
    hq_block_bytes,
    quant_hq_blocks,
    dequant_hq_blocks,
)
from pack import pack_file
from verify import audit_pack, read_header


class HQBlocksTests(unittest.TestCase):
    def test_parse_and_format_quant_spec(self) -> None:
        self.assertEqual(parse_quant_spec("hq4"), (4, 32, True))
        self.assertEqual(parse_quant_spec("hq4_g32_m5"), (4, 32, True))
        self.assertEqual(parse_quant_spec("hq4_g32"), (4, 32, False))
        self.assertEqual(parse_quant_spec("hq4_g64"), (4, 64, False))
        self.assertEqual(parse_quant_spec("hq6", default_group=128), (6, 128, False))
        self.assertEqual(parse_quant_spec("hq8_g32"), (8, 32, False))
        self.assertEqual(format_quant_spec(4, 32, is_m5=True), "hq4_g32_m5")
        self.assertEqual(format_quant_spec(4, 32, is_m5=False), "hq4_g32")
        self.assertEqual(format_quant_spec(8, 32), "hq8_g32")
        self.assertEqual(format_quant_spec(6, 128), "hq6_g128")

    def test_incumbent_hq4_g32_m5_is_byte_identical(self) -> None:
        rng = np.random.default_rng(42)
        w = rng.standard_normal((16, 256), dtype=np.float32) * 0.03
        raw_inc = quant_hq4_blocks(w)
        raw_new, eff_g = quant_hq_blocks(w, 4, 32, is_m5=True)
        self.assertEqual(eff_g, 32)
        self.assertEqual(raw_inc.tobytes(), raw_new.tobytes())
        w_inc = dequant_hq4_g32_m5(raw_inc)
        w_new = dequant_hq_blocks(raw_new, 4, 32, is_m5=True)
        np.testing.assert_allclose(w_inc, w_new, rtol=0, atol=0)

    def test_all_twelve_pairs_monotonic_fidelity(self) -> None:
        rng = np.random.default_rng(123)
        w = rng.standard_normal((32, 512), dtype=np.float32) * 0.02
        mse_table: dict[tuple[int, int], float] = {}
        for bits in VALID_BITS:
            for group in VALID_GROUPS:
                raw, eff_g = quant_hq_blocks(w, bits, group, is_m5=False)
                self.assertEqual(eff_g, group)
                expected_bytes = 32 * (512 // group) * hq_block_bytes(bits, group)
                self.assertEqual(raw.size, expected_bytes)
                w_hat = dequant_hq_blocks(raw, bits, group, is_m5=False)
                mse = float(np.mean((w - w_hat) ** 2))
                mse_table[(bits, group)] = mse
        for group in VALID_GROUPS:
            self.assertLess(mse_table[(3, group)], mse_table[(2, group)] * 0.45)
            self.assertLess(mse_table[(4, group)], mse_table[(3, group)] * 0.35)
            self.assertLess(mse_table[(6, group)], mse_table[(4, group)] * 0.1)
            self.assertLess(mse_table[(8, group)], mse_table[(6, group)] * 0.1)
        for bits in VALID_BITS:
            self.assertLessEqual(mse_table[(bits, 32)], mse_table[(bits, 64)])
            self.assertLessEqual(mse_table[(bits, 64)], mse_table[(bits, 128)])
            self.assertLessEqual(mse_table[(bits, 128)], mse_table[(bits, 256)])

    def test_gcd_group_size_clamping(self) -> None:
        self.assertEqual(resolve_group_size(320, 256), 64)
        self.assertEqual(resolve_group_size(320, 128), 64)
        self.assertEqual(resolve_group_size(320, 64), 64)
        self.assertEqual(resolve_group_size(320, 32), 32)

    def test_mixed_pack_and_verify_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        dense_f32 = rng.standard_normal((16, 320), dtype=np.float32) * 0.02
        exp_f32 = rng.standard_normal((4, 8, 256), dtype=np.float32) * 0.02
        norm_f32 = np.ones((256,), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            qwp_path = Path(tmp) / "model.qwp"
            pack_file(
                [
                    ("model.layers.0.dense.weight", dense_f32, "hq8_g128"),
                    ("model.layers.0.mlp.experts.down_proj", exp_f32, "hq4_g64"),
                    ("model.layers.0.input_layernorm.weight", norm_f32, True),
                ],
                qwp_path,
                provenance={
                    "source": "unit-test",
                    "quant": "dense_hq8_g128__expert_hq4_g64",
                    "geometry_hash": "0" * 64,
                },
            )
            ok, msg = audit_pack(
                qwp_path,
                expected_shapes={
                    "model.layers.0.dense.weight": [16, 320],
                    "model.layers.0.mlp.experts.down_proj": [4, 8, 256],
                    "model.layers.0.input_layernorm.weight": [256],
                },
            )
            self.assertTrue(ok, msg)
            hdr, _ = read_header(qwp_path)
            self.assertEqual(hdr["tensors"]["model.layers.0.dense.weight"]["hq"], {"bits": 8, "group": 64})
            self.assertEqual(hdr["tensors"]["model.layers.0.mlp.experts.down_proj"]["hq"], {"bits": 4, "group": 64})


if __name__ == "__main__":
    unittest.main()
