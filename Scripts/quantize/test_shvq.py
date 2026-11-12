# Unit tests for SHVQ{1,15,2,3,4}_G{32,64,128,256} quantizer, packer, and verifier.
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
from shvq_sub2 import (
    SHVQ15_STEP_SCALE_F32,
    emit_sub2_metal_and_swift_tables,
    load_shvq_sub2_tables,
)
from verify import audit_pack, read_header


class TestSHVQ(unittest.TestCase):
    def test_tables_positive_orthant_invariants(self):
        tbls = load_shvq_tables()
        self.assertEqual(tbls["lut2_bits"].shape, (256, 8))
        self.assertEqual(tbls["lut3_bits"].shape, (256, 4))
        self.assertEqual(tbls["lut4_u8"].shape, (4096, 4))
        self.assertEqual(tbls["step_map_bits"].shape, (256,))
        self.assertTrue(np.all((tbls["lut2_bits"] & 0x8000) == 0))
        self.assertTrue(np.all((tbls["lut3_bits"] & 0x8000) == 0))
        self.assertTrue(np.all((tbls["step_map_bits"] & 0x8000) == 0))

    def test_sub2_tables_nonzero_and_0ulp_step_map_invariants(self):
        sub2 = load_shvq_sub2_tables()
        self.assertEqual(sub2["lut1_bits"].shape, (128, 8))
        self.assertEqual(sub2["step_map15_bits"].shape, (256,))
        self.assertEqual(sub2["lut15_u8"].shape, (2048, 8))
        self.assertTrue(np.all((sub2["lut1_bits"] & 0x7FFF) >= 0x0400))
        self.assertTrue(np.all((sub2["lut1_bits"] & 0x7FFF) < 0x7C00))
        self.assertTrue(np.all(sub2["lut15_u8"] != 0))
        i8 = np.arange(256, dtype=np.uint8).view(np.int8)
        conv = (i8.astype(np.float32) * SHVQ15_STEP_SCALE_F32).astype(np.float16).view(np.uint16)
        np.testing.assert_array_equal(conv, sub2["step_map15_bits"])
        direct = sub2["step_map15_bits"][sub2["lut15_u8"]]
        self.assertTrue(np.all((direct & 0x7FFF) >= 0x0400))
        self.assertTrue(np.all((direct & 0x7FFF) < 0x7C00))
        with tempfile.TemporaryDirectory() as td:
            metal = Path(td) / "qw_shvq_sub2_tables.metalh"
            swift = Path(td) / "QWSHVQSub2Tables.swift"
            emit_sub2_metal_and_swift_tables(metal, swift)
            routed = metal.with_name("qw_shvq_sub2_routed_lut.metalh")
            actual = np.array(
                [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]{4})u", routed.read_text())],
                dtype=np.uint16,
            )
            np.testing.assert_array_equal(actual, direct.reshape(-1))
            root = Path(__file__).resolve().parents[2]
            self.assertEqual(routed.read_bytes(),
                             (root / "Sources/QWMetal/include" / routed.name).read_bytes())
            self.assertEqual(metal.read_bytes(),
                             (root / "Sources/QWMetal/include" / metal.name).read_bytes())
            self.assertEqual(swift.read_bytes(),
                             (root / "Sources/Foundation/QWQuant" / swift.name).read_bytes())

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

    def test_all_20_pairs_byte_law_cpu_and_metal_roundtrip(self):
        rng = np.random.default_rng(12345)
        w = (rng.standard_normal((16, 256)) * 0.04).astype(np.float32)
        for bits in SHVQ_VALID_BITS:
            for group in SHVQ_VALID_GROUPS:
                bbytes = shvq_block_bytes(bits, group)
                want_pay = (group * 3) // 16 if bits == 15 else (group * bits) // 8
                self.assertEqual(bbytes, 2 + want_pay)
                self.assertEqual(bbytes % 2, 0)
                blks_cpu, eff_g = quant_shvq_blocks(w, bits=bits, group=group, device="cpu")
                self.assertEqual(eff_g, group)
                self.assertEqual(blks_cpu.shape, (16, 256 // group, bbytes))
                dq_cpu = dequant_shvq_blocks(blks_cpu, bits=bits, group=group)
                self.assertEqual(dq_cpu.shape, w.shape)
                self.assertTrue(np.all(np.isfinite(dq_cpu)))
                blks_mtl, _ = quant_shvq_blocks(w, bits=bits, group=group, device="mps")
                dq_mtl = dequant_shvq_blocks(blks_mtl, bits=bits, group=group)
                rel_cpu = float(np.linalg.norm(w - dq_cpu) / np.linalg.norm(w))
                rel_mtl = float(np.linalg.norm(w - dq_mtl) / np.linalg.norm(w))
                self.assertAlmostEqual(rel_cpu, rel_mtl, delta=2e-3)

    def test_rate_distortion_beats_scalar_hq_and_monotone_sub2(self):
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

        b15, _ = quant_shvq_blocks(w, bits=15, group=64, device="cpu")
        rel15 = float(np.linalg.norm(w - dequant_shvq_blocks(b15, bits=15, group=64))) / norm_w
        b1, _ = quant_shvq_blocks(w, bits=1, group=64, device="cpu")
        rel1 = float(np.linalg.norm(w - dequant_shvq_blocks(b1, bits=1, group=64))) / norm_w
        self.assertLess(rel15, 0.43)
        self.assertLess(rel15, rel1 * 0.78)

    def test_pack_and_verify_mixed_shvq_including_sub2(self):
        rng = np.random.default_rng(99)
        w_dense = (rng.standard_normal((32, 128)) * 0.02).astype(np.float32)
        w_draft = (rng.standard_normal((32, 128)) * 0.02).astype(np.float32)
        w_exp_gu = (rng.standard_normal((4, 64, 128)) * 0.02).astype(np.float32)
        w_exp_dn = (rng.standard_normal((4, 128, 64)) * 0.02).astype(np.float32)
        w_norm = np.ones((128,), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            out_qwp = Path(td) / "model.qwp"
            tensors = [
                ("layers.0.attn.o_proj.weight", w_dense, "shvq15_g64"),
                ("mtp.fc.weight", w_draft, "shvq1_g64"),
                ("layers.0.mlp.experts.gate_up_proj", w_exp_gu, "shvq2_g64"),
                ("layers.0.mlp.experts.down_proj", w_exp_dn, "shvq3_g64"),
                ("norm.weight", w_norm, True),
            ]
            pack_file(tensors, out_qwp, {"model_id": "test"})
            counts = audit_pack(out_qwp)
            hdr, _ = read_header(out_qwp)
            self.assertEqual(hdr["tensors"]["layers.0.attn.o_proj.weight"]["hq"],
                             {"bits": 15, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr["tensors"]["mtp.fc.weight"]["hq"],
                             {"bits": 1, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr["tensors"]["layers.0.mlp.experts.gate_up_proj"]["hq"],
                             {"bits": 2, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr["tensors"]["layers.0.mlp.experts.down_proj"]["hq"],
                             {"bits": 3, "group": 64, "vq": "shvq1"})
            self.assertEqual(parse_shvq_spec("shvq15"), (15, 64))
            self.assertEqual(parse_shvq_spec("shvq1"), (1, 64))
            self.assertEqual(parse_shvq_spec("shvq3"), (3, 64))
            self.assertIsNotNone(counts)

    def test_pack_dflash_sub2_shvq(self):
        import json
        import torch
        from safetensors.torch import save_file
        from pack_dflash import quantize_dflash_model

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            cfg = {
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 128,
                "vocab_size": 64,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "block_size": 16,
                "dflash_config": {"selected_prefix_code_bridge": {"code_size": 64, "rank": 64}},
            }
            (src / "config.json").write_text(json.dumps(cfg))
            save_file(
                {
                    "fc.weight": torch.randn(128, 256, dtype=torch.bfloat16) * 0.02,
                    "lm_head.weight": torch.randn(64, 128, dtype=torch.bfloat16) * 0.02,
                    "norm.weight": torch.ones(128, dtype=torch.bfloat16),
                    "selected_prefix_bridge.code_embedding": torch.randn(64, 128, dtype=torch.bfloat16) * 0.02,
                    "draft_block_causal_version": torch.tensor(1, dtype=torch.int64),
                },
                str(src / "model.safetensors"),
            )
            for qspec, want_bits in [("shvq15_g64", 15), ("shvq1_g64", 1)]:
                dst = Path(td) / f"out_{qspec}"
                stats = quantize_dflash_model(src, dst, quant=qspec)
                self.assertEqual(stats[qspec], 2)
                self.assertEqual(stats["bf16"], 2)
                hdr, _ = read_header(dst / "model.qwp")
                self.assertEqual(hdr["provenance"]["quant_recorded"], qspec)
                self.assertEqual(hdr["tensors"]["fc.weight"]["hq"],
                                 {"bits": want_bits, "group": 64, "vq": "shvq1"})
                self.assertEqual(hdr["tensors"]["lm_head.weight"]["hq"],
                                 {"bits": want_bits, "group": 64, "vq": "shvq1"})
                self.assertEqual(hdr["tensors"]["norm.weight"]["kind"], "bf16")


if __name__ == "__main__":
    unittest.main()
