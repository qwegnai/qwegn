import math
import unittest
from pathlib import Path

import numpy as np
import torch

import flashnext_teacher_kl as kl
from hq4 import dequant_hq4_g32_m5_rot, quant_hq4_blocks
from pack_ngram import _encode_q8

PACK = Path("~/.qwg/models/Qwen3.8-Flash-Next-hq4g32/model.qwp").expanduser()


class TeacherKLTests(unittest.TestCase):
    def test_mlx_dequant_matches_mlx(self):
        import mlx.core as mx
        rng = np.random.default_rng(3)
        for bits, group in ((4, 32), (4, 64), (8, 64), (2, 32)):
            w = rng.normal(size=(6, 256)).astype(np.float32)
            q, s, b = mx.quantize(mx.array(w), group_size=group, bits=bits)
            ref = np.array(mx.dequantize(q, s, b, group_size=group, bits=bits))
            got = kl.mlx_dequant(np.array(q), torch.from_numpy(np.array(s.astype(mx.float32))),
                                 torch.from_numpy(np.array(b.astype(mx.float32))), 256).numpy()
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6, err_msg=f"{bits}/{group}")

    def test_q8_rows_match_pack_encoder(self):
        rows = np.random.default_rng(5).normal(scale=0.03, size=(64, 160)).astype(np.float32)
        rows[2, :32] = 0.0
        raw = np.frombuffer(_encode_q8(rows, np), np.uint8).reshape(64, 5, 34)
        codes = raw[..., :32].copy().view(np.int8).astype(np.float32)
        scales = raw[..., 32:].copy().view(np.float16).astype(np.float32)
        expected = (codes * scales).reshape(64, 160)
        np.testing.assert_array_equal(kl.q8_rows(torch.from_numpy(rows)).numpy(), expected)

    def test_q4_rows_match_unrotated_blocks(self):
        rows = np.random.default_rng(6).normal(scale=0.03, size=(64, 160)).astype(np.float32)
        expected = dequant_hq4_g32_m5_rot(quant_hq4_blocks(rows, rotate=False))
        got = kl.q4_rows(torch.from_numpy(rows)).numpy()
        self.assertLess(np.mean(np.abs(got - expected) > 1e-6), 1e-3)

    def test_ngram_file_bytes(self):
        self.assertEqual(kl.ngram_file_bytes("q8", 320001536), 54613602304)
        self.assertEqual(kl.ngram_file_bytes("q4", 40), 2 * 4096)
        self.assertEqual(kl.ngram_file_bytes("bf16", 13), 3 * 4096)

    def test_variant_cost(self):
        header = {"tensors": {
            "model.language_model.layers.0.mlp.experts.down_proj": {
                "shape": [4, 64, 128], "experts": 4, "hq4_off": 0, "e_hq4": 64 * 4 * 20},
            "model.language_model.layers.0.self_attn.o_proj.weight": {"shape": [64, 128], "hq4_off": 0},
            "model.language_model.norm.weight": {"shape": [64], "raw_off": 0, "kind": "bf16"},
        }}
        ngram = {"q8": 1000}
        cost = kl.variant_cost(header, ("bf16", "hq4_g32_m5", "hq8_g32", "q8"), ngram)
        expert = 4 * 64 * 128 // 32 * 36
        self.assertEqual(cost["model_bytes"], expert + 64 * 128 * 2 + 128)
        self.assertEqual(cost["total_bytes"], cost["model_bytes"] + 1000)

    @unittest.skipUnless(PACK.exists(), "incumbent pack not installed")
    def test_classification_matches_incumbent(self):
        from verify import read_header
        header, _ = read_header(str(PACK))
        wrong = [n for n, e in header["tensors"].items()
                 if ("raw_off" not in e) != kl.is_quantized_name(n, e["shape"])]
        self.assertEqual(wrong, [])
        incumbent = kl.variant_cost(header, kl.VARIANTS["incumbent"], {"q8": 0})
        payload = sum(math.prod(e["shape"]) // 32 * 20 if "raw_off" not in e else 0
                      for e in header["tensors"].values())
        self.assertGreater(incumbent["model_bytes"], payload)
        self.assertLess(incumbent["model_bytes"], PACK.stat().st_size)


if __name__ == "__main__":
    unittest.main()
