# Unit tests for SHVQ{2,3,4}_G{32,64,128,256} quantizer, packer, and verifier.
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hq_blocks import dequant_hq_blocks, hq_block_bytes, quant_hq_blocks
from pack import pack_file
from shvq import (
    SHVQ_VALID_BITS,
    SHVQ_VALID_GROUPS,
    dequant_shvq_blocks,
    emit_metal_and_swift_tables,
    load_shvq_tables,
    parse_shvq_spec,
    quant_shvq_blocks,
    shvq_block_bytes,
)
from verify import audit_pack, read_header


class TestSHVQ(unittest.TestCase):
    def test_tables_positive_orthant_invariants(self):
        tbls = load_shvq_tables()
        self.assertEqual(tbls["lut2_bits"].shape, (256, 8))
        self.assertEqual(tbls["lut3_bits"].shape, (256, 4))
        self.assertEqual(tbls["lut4_u8"].shape, (4096, 4))
        self.assertEqual(tbls["step_map_bits"].shape, (256,))
        # Every stored Float16 bit pattern must have sign bit == 0 (positive orthant).
        self.assertTrue(np.all((tbls["lut2_bits"] & 0x8000) == 0))
        self.assertTrue(np.all((tbls["lut3_bits"] & 0x8000) == 0))
        self.assertTrue(np.all((tbls["step_map_bits"] & 0x8000) == 0))

    def test_routed_half_table_matches_packed_conversion_and_is_reproducible(self):
        tables = load_shvq_tables()
        step_bits = tables["step_map_bits"]
        scale = np.float32(float(step_bits[255].view(np.float16)) / 255.0)
        converted = (np.arange(256, dtype=np.float32) * scale).astype(np.float16).view(np.uint16)
        np.testing.assert_array_equal(converted, step_bits)
        expected = step_bits[tables["lut4_u8"]].reshape(-1)
        with tempfile.TemporaryDirectory() as td:
            metal = Path(td) / "qw_shvq_tables.metalh"
            swift = Path(td) / "QWSHVQTables.swift"
            emit_metal_and_swift_tables(metal, swift)
            routed = metal.with_name("qw_shvq_routed_lut.metalh")
            actual = np.array([int(v, 16) for v in re.findall(
                r"0x([0-9A-Fa-f]{4})u", routed.read_text())], dtype=np.uint16)
            np.testing.assert_array_equal(actual, expected)
            root = Path(__file__).resolve().parents[2]
            self.assertEqual(routed.read_bytes(),
                             (root / "Sources/QWMetal/include" / routed.name).read_bytes())
            self.assertEqual(metal.read_bytes(),
                             (root / "Sources/QWMetal/include" / metal.name).read_bytes())
            self.assertEqual(swift.read_bytes(),
                             (root / "Sources/Foundation/QWQuant" / swift.name).read_bytes())

    def test_all_12_pairs_byte_law_and_roundtrip(self):
        rng = np.random.default_rng(12345)
        w = (rng.standard_normal((16, 256)) * 0.04).astype(np.float32)
        for bits in SHVQ_VALID_BITS:
            for group in SHVQ_VALID_GROUPS:
                bbytes = shvq_block_bytes(bits, group)
                self.assertEqual(bbytes, 2 + (group * bits) // 8)
                self.assertLess(bbytes, hq_block_bytes(bits, group))
                blks, eff_g = quant_shvq_blocks(w, bits=bits, group=group, device="cpu")
                self.assertEqual(eff_g, group)
                self.assertEqual(blks.shape, (16, 256 // group, bbytes))
                dq = dequant_shvq_blocks(blks, bits=bits, group=group)
                self.assertEqual(dq.shape, w.shape)
                self.assertTrue(np.all(np.isfinite(dq)))

    def test_rate_distortion_beats_scalar_hq_across_2_3_4_bits(self):
        rng = np.random.default_rng(20261110)
        w = (rng.standard_normal((128, 512)) * 0.025).astype(np.float32)
        norm_w = float(np.linalg.norm(w))
        for bits, shvq_g, hq_g in [(2, 64, 128), (3, 64, 64), (4, 64, 64)]:
            s_blks, _ = quant_shvq_blocks(w, bits=bits, group=shvq_g, device="cpu")
            s_dq = dequant_shvq_blocks(s_blks, bits=bits, group=shvq_g)
            s_rel = float(np.linalg.norm(w - s_dq)) / norm_w

            h_blks, _ = quant_hq_blocks(w, bits=bits, group=hq_g, is_m5=False)
            h_dq = dequant_hq_blocks(h_blks, bits=bits, group=hq_g, is_m5=False)
            h_rel = float(np.linalg.norm(w - h_dq)) / norm_w

            self.assertLessEqual(shvq_block_bytes(bits, shvq_g) * (512 // shvq_g),
                                 hq_block_bytes(bits, hq_g) * (512 // hq_g))
            self.assertLess(s_rel, h_rel * 0.92,
                            f"SHVQ{bits}_G{shvq_g} rel_L2={s_rel:.4f} must beat HQ{bits}_G{hq_g} rel_L2={h_rel:.4f}")

    def test_pack_and_verify_mixed_shvq_2d_and_3d_experts(self):
        rng = np.random.default_rng(99)
        w_dense = (rng.standard_normal((32, 128)) * 0.02).astype(np.float32)
        w_exp_gu = (rng.standard_normal((4, 64, 128)) * 0.02).astype(np.float32)
        w_exp_dn = (rng.standard_normal((4, 128, 64)) * 0.02).astype(np.float32)
        w_norm = np.ones((128,), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            out_qwp = Path(td) / "model.qwp"
            tensors = [
                ("layers.0.attn.o_proj.weight", w_dense, "shvq4_g32"),
                ("layers.0.mlp.experts.gate_up_proj", w_exp_gu, "shvq2_g64"),
                ("layers.0.mlp.experts.down_proj", w_exp_dn, "shvq3_g64"),
                ("norm.weight", w_norm, True),
            ]
            pack_file(tensors, out_qwp, {"model_id": "test"})
            counts = audit_pack(out_qwp)
            hdr, _ = read_header(out_qwp)
            self.assertEqual(hdr["tensors"]["layers.0.attn.o_proj.weight"]["hq"],
                             {"bits": 4, "group": 32, "vq": "shvq1"})
            self.assertEqual(hdr["tensors"]["layers.0.mlp.experts.gate_up_proj"]["hq"],
                             {"bits": 2, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr["tensors"]["layers.0.mlp.experts.down_proj"]["hq"],
                             {"bits": 3, "group": 64, "vq": "shvq1"})
            self.assertEqual(parse_shvq_spec("shvq3"), (3, 64))
            self.assertIsNotNone(counts)


if __name__ == "__main__":
    unittest.main()
