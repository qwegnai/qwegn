"""Unit tests for GigaChat4 and MLA/GatedNorm DFlash2 QWP1 packers."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

QUANTIZE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(QUANTIZE_DIR))
sys.path.insert(0, str(QUANTIZE_DIR.parent / "oracle"))

from emit_pack_identity import geo_v2  # noqa: E402
from gigachat4_adapter import (  # noqa: E402
    _deinterleave_gdn_qkvz_ba,
    _deinterleave_mla_q,
    quantize_gigachat4_model,
)
from pack import quantize_model  # noqa: E402
from pack_dflash import derive_dflash_model_id, quantize_dflash_model  # noqa: E402
from verify import read_header  # noqa: E402


def _add_gated_norm(tensors: dict[str, torch.Tensor], prefix: str, dim: int, g: torch.Generator) -> None:
    tensors[prefix + "weight"] = torch.ones(dim, dtype=torch.bfloat16)
    tensors[prefix + "gate_up_projection.weight"] = (torch.randn(16, dim, generator=g) * 0.02).to(torch.bfloat16)
    tensors[prefix + "gate_down_projection.weight"] = (torch.randn(dim, 16, generator=g) * 0.02).to(torch.bfloat16)


def _write_synthetic_gigachat4_checkpoint(src: Path) -> dict:
    src.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["GigaChat4ForCausalLM"],
        "model_type": "gigachat4",
        "hidden_size": 128,
        "intermediate_size": 256,
        "vocab_size": 256,
        "num_hidden_layers": 2,
        "full_attention_layers": [1],
        "linear_attention_type": "gated_deltanet",
        "linear_gating_type": "gated_rmsnorm_sigmoid",
        "linear_sigmoid_gate_scale": 2.0,
        "linear_attn_o_norm_eps": 1e-6,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 32,
        "qk_head_dim": 64,
        "v_head_dim": 32,
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "index_n_heads": 2,
        "index_head_dim": 32,
        "index_topk": 64,
        "norm_type": "GatedNorm",
        "indexer_norm_type": "LlamaRMSNorm",
        "layernorm_type": "pre_post",
        "layernorm_gating_weight": 2.0,
        "gated_attention": True,
        "use_mla_scaling_factor": True,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "use_shared_expert_sigmoid": False,
        "num_nextn_predict_layers": 1,
        "nextn_is_sparse": False,
        "nextn_is_linear": False,
        "final_logit_softcapping": 30.0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 100000.0,
        "max_position_embeddings": 32768,
        "eos_token_id": 2,
    }
    (src / "config.json").write_text(json.dumps(cfg))
    (src / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {"<s>": 1, "</s>": 2, "a": 3}, "merges": []},
    }))

    g = torch.Generator().manual_seed(17)
    h, vocab, inter, moe_inter = 128, 256, 256, 64
    emb = (torch.randn(vocab, h, generator=g) * 0.02).to(torch.bfloat16)
    lm_head = (torch.randn(vocab, h, generator=g) * 0.02).to(torch.bfloat16)
    t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": emb,
        "lm_head.weight": lm_head,
    }
    _add_gated_norm(t, "model.norm.", h, g)

    # Layer 0: GDN + dense MLP.
    b0 = "model.layers.0."
    for nm in ("input_layernorm.", "post_self_attn_layernorm.", "post_attention_layernorm.", "post_feedforward_layernorm."):
        _add_gated_norm(t, b0 + nm, h, g)
    s0 = b0 + "self_attn."
    t[s0 + "in_proj_qkvz.weight"] = (torch.randn(2 * (32 + 32 + 64 + 64), h, generator=g) * 0.02).to(torch.bfloat16)
    t[s0 + "in_proj_ba.weight"] = (torch.randn(2 * 4, h, generator=g) * 0.02).to(torch.bfloat16)
    t[s0 + "conv1d.weight"] = (torch.randn(256, 1, 4, generator=g) * 0.02).to(torch.bfloat16)
    t[s0 + "norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
    t[s0 + "A_log"] = torch.zeros(4, dtype=torch.float32)
    t[s0 + "dt_bias"] = torch.ones(4, dtype=torch.float32)
    t[s0 + "out_proj.weight"] = (torch.randn(h, 128, generator=g) * 0.02).to(torch.bfloat16)
    for p, sh in (("gate_proj", (inter, h)), ("up_proj", (inter, h)), ("down_proj", (h, inter))):
        t[b0 + f"mlp.{p}.weight"] = (torch.randn(*sh, generator=g) * 0.02).to(torch.bfloat16)

    # Layer 1: MLA + DSA indexer + MoE.
    def populate_mla(blk: str, *, with_dsa: bool) -> None:
        for nm in ("input_layernorm.", "post_self_attn_layernorm.", "post_attention_layernorm.", "post_feedforward_layernorm."):
            _add_gated_norm(t, blk + nm, h, g)
        ap = blk + "self_attn."
        t[ap + "q_a_proj.weight"] = (torch.randn(64, h, generator=g) * 0.02).to(torch.bfloat16)
        _add_gated_norm(t, ap + "q_a_layernorm.", 64, g)
        t[ap + "q_b_proj.weight"] = (torch.randn(4 * 64, 64, generator=g) * 0.02).to(torch.bfloat16)
        t[ap + "kv_a_proj_with_mqa.weight"] = (torch.randn(64 + 32, h, generator=g) * 0.02).to(torch.bfloat16)
        _add_gated_norm(t, ap + "kv_a_layernorm.", 64, g)
        t[ap + "kv_b_proj.weight"] = (torch.randn(4 * 64, 64, generator=g) * 0.02).to(torch.bfloat16)
        t[ap + "gate_proj.weight"] = (torch.randn(4 * 32, h, generator=g) * 0.02).to(torch.bfloat16)
        t[ap + "o_proj.weight"] = (torch.randn(h, 4 * 32, generator=g) * 0.02).to(torch.bfloat16)
        if with_dsa:
            t[ap + "indexer.wk.weight"] = (torch.randn(32, h, generator=g) * 0.02).to(torch.bfloat16)
            t[ap + "indexer.wq_b.weight"] = (torch.randn(64, 64, generator=g) * 0.02).to(torch.bfloat16)
            t[ap + "indexer.k_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
            t[ap + "indexer.weights_proj.weight"] = (torch.randn(2, h, generator=g) * 0.02).to(torch.float32)

    b1 = "model.layers.1."
    populate_mla(b1, with_dsa=True)
    t[b1 + "mlp.gate.weight"] = (torch.randn(4, h, generator=g) * 0.02).to(torch.float32)
    t[b1 + "mlp.gate.e_score_correction_bias"] = torch.zeros(4, dtype=torch.float32)
    for p, sh in (("gate_proj", (moe_inter, h)), ("up_proj", (moe_inter, h)), ("down_proj", (h, moe_inter))):
        t[b1 + f"mlp.shared_experts.{p}.weight"] = (torch.randn(*sh, generator=g) * 0.02).to(torch.bfloat16)
    for e in range(4):
        for p, sh in (("gate_proj", (moe_inter, h)), ("up_proj", (moe_inter, h)), ("down_proj", (h, moe_inter))):
            t[b1 + f"mlp.experts.{e}.{p}.weight"] = (torch.randn(*sh, generator=g) * 0.02).to(torch.bfloat16)

    # Layer 2: NextN MTP (dense MLA without DSA + dense MLP + duplicate tied embed/head).
    b2 = "model.layers.2."
    for nm in ("enorm.", "hnorm.", "shared_head.norm."):
        _add_gated_norm(t, b2 + nm, h, g)
    t[b2 + "eh_proj.weight"] = (torch.randn(h, 2 * h, generator=g) * 0.02).to(torch.bfloat16)
    t[b2 + "embed_tokens.weight"] = emb.clone()
    t[b2 + "shared_head.head.weight"] = lm_head.clone()
    populate_mla(b2, with_dsa=False)
    for p, sh in (("gate_proj", (inter, h)), ("up_proj", (inter, h)), ("down_proj", (h, inter))):
        t[b2 + f"mlp.{p}.weight"] = (torch.randn(*sh, generator=g) * 0.02).to(torch.bfloat16)

    save_file(t, str(src / "model.safetensors"))
    return cfg


class GigaChat4AdapterTests(unittest.TestCase):
    def test_deinterleave_gdn_ba_places_b_second_and_a_first(self) -> None:
        qkvz = torch.arange(2 * (4 + 4 + 8 + 8) * 16, dtype=torch.float32).reshape(2 * 24, 16)
        ba = torch.zeros(2, 4, 16, dtype=torch.float32)
        ba[:, :2, :] = -7.0  # b comes first in Megatron/GigaChat4 in_proj_ba
        ba[:, 2:, :] = 11.0  # a comes second
        qkv, z, a, b = _deinterleave_gdn_qkvz_ba(
            qkvz, ba.reshape(8, 16), qk_heads=2, v_heads=4, k_dim=4, v_dim=4, hidden=16
        )
        self.assertEqual(list(qkv.shape), [32, 16])
        self.assertEqual(list(z.shape), [16, 16])
        self.assertEqual(list(a.shape), [4, 16])
        self.assertEqual(list(b.shape), [4, 16])
        self.assertTrue(torch.all(a == 11.0))
        self.assertTrue(torch.all(b == -7.0))

    def test_deinterleave_mla_q_splits_per_head_nope_and_rope(self) -> None:
        raw = torch.zeros(3, 6, 8, dtype=torch.float32)
        raw[:, :4, :] = 2.5
        raw[:, 4:, :] = -3.5
        nope, rope = _deinterleave_mla_q(raw.reshape(18, 8), q_heads=3, nope_d=4, rope_d=2, in_dim=8)
        self.assertEqual(list(nope.shape), [12, 8])
        self.assertEqual(list(rope.shape), [6, 8])
        self.assertTrue(torch.all(nope == 2.5))
        self.assertTrue(torch.all(rope == -3.5))

    def test_pack_dispatch_mixed_and_shvq4_and_raw_gdn_ab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            _write_synthetic_gigachat4_checkpoint(src)

            out_mixed = root / "out_mixed"
            quantize_model(src, out_mixed, dense_quant="hq8_g32", expert_quant="shvq4_g32")
            hdr, _ = read_header(out_mixed / "model.qwp")
            prov = hdr["provenance"]
            tensors = hdr["tensors"]
            self.assertEqual(prov["model_id"], "gigachat4")
            self.assertEqual(prov["quant_recorded"], "dense_hq8_g32__expert_shvq4_g32")
            self.assertTrue(prov["mtp_shared_embed_and_head"])
            self.assertNotIn("model.layers.2.embed_tokens.weight", tensors)
            self.assertNotIn("model.layers.2.shared_head.head.weight", tensors)
            self.assertNotIn("model.layers.2.self_attn.indexer.wk.weight", tensors)
            self.assertIn("model.layers.1.self_attn.q_b_nope_proj.weight", tensors)
            self.assertIn("model.layers.1.self_attn.q_b_rope_proj.weight", tensors)
            self.assertEqual(tensors["model.norm.gate_up_projection.weight"]["kind"], "bf16")
            self.assertEqual(tensors["model.layers.0.linear_attn.in_proj_a.weight"]["hq"]["bits"], 8)
            self.assertEqual(tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]["hq"]["vq"], "shvq1")

            out_fast = root / "out_shvq4"
            quantize_model(src, out_fast, quant="shvq4_g32")
            hdr_fast, _ = read_header(out_fast / "model.qwp")
            self.assertEqual(hdr_fast["provenance"]["quant_recorded"], "shvq4_g32")

            out_raw_ab = root / "out_raw_ab"
            quantize_gigachat4_model(src, out_raw_ab, quant="shvq4_g32", raw_gdn_ab=True)
            hdr_ab, _ = read_header(out_raw_ab / "model.qwp")
            self.assertEqual(hdr_ab["tensors"]["model.layers.0.linear_attn.in_proj_a.weight"]["kind"], "bf16")
            self.assertEqual(hdr_ab["tensors"]["model.layers.0.linear_attn.in_proj_b.weight"]["kind"], "bf16")

    def test_pack_dflash_supports_gigachat4_mla_and_gated_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = root / "models--ai-sage--GigaChat4-37B-A4B" / "snapshots" / ("a" * 40) / "dflash_draft"
            snap.mkdir(parents=True)
            cfg = {
                "architectures": ["GigaChat4DFlash2DraftModel"],
                "model_type": "gigachat4",
                "hidden_size": 128,
                "intermediate_size": 256,
                "vocab_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 32,
                "v_head_dim": 32,
                "q_lora_rank": 64,
                "kv_lora_rank": 64,
                "norm_type": "GatedNorm",
                "layernorm_type": "pre_post",
                "gated_attention": True,
                "use_mla_scaling_factor": True,
                "dflash_config": {
                    "block_size": 8,
                    "selector_rank": 64,
                    "selector_top_k": 16,
                    "attention_mode": "mla",
                },
            }
            (snap / "config.json").write_text(json.dumps(cfg))
            self.assertEqual(derive_dflash_model_id(cfg, snap, is_dspine=False), "dflash2_gigachat4_37b_a4b")

            g = torch.Generator().manual_seed(23)
            t: dict[str, torch.Tensor] = {
                "fc.weight": (torch.randn(128, 256, generator=g) * 0.02).to(torch.bfloat16),
                "candidate_selector.hidden_projection.weight": (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16),
                "layers.0.self_attn.q_a_proj.weight": (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16),
                "layers.0.self_attn.q_b_proj.weight": (torch.randn(4 * 64, 64, generator=g) * 0.02).to(torch.bfloat16),
                "layers.0.mlp.gate_proj.weight": (torch.randn(256, 128, generator=g) * 0.02).to(torch.bfloat16),
            }
            _add_gated_norm(t, "hidden_norm.", 128, g)
            _add_gated_norm(t, "norm.", 128, g)
            _add_gated_norm(t, "layers.0.input_layernorm.", 128, g)
            save_file(t, str(snap / "model.safetensors"))

            out = root / "out_dflash"
            quantize_dflash_model(snap, out, quant="shvq4_g32")
            hdr, _ = read_header(out / "model.qwp")
            self.assertEqual(hdr["provenance"]["model_id"], "dflash2_gigachat4_37b_a4b")
            expected_geo = "qwg-dflash2-v1:128|256|256|1|4|4|64|8|64|16"
            self.assertEqual(hdr["provenance"]["geometry_hash"], hashlib.sha256(expected_geo.encode()).hexdigest())
            self.assertIn("layers.0.self_attn.q_b_nope_proj.weight", hdr["tensors"])
            self.assertIn("layers.0.self_attn.q_b_rope_proj.weight", hdr["tensors"])
            self.assertEqual(hdr["tensors"]["hidden_norm.gate_up_projection.weight"]["kind"], "bf16")

    def test_real_gigachat4_geo_v2_digest_matches_pack_if_present(self) -> None:
        cfg_path = (
            Path.home()
            / ".cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B"
            / "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc/config.json"
        )
        if not cfg_path.is_file():
            self.skipTest("local GigaChat4-37B-A4B config.json not present")
        cfg = json.loads(cfg_path.read_text())
        digest = hashlib.sha256(geo_v2(cfg).encode()).hexdigest()
        self.assertEqual(digest, "6aa3f3e504094dc6aa5cf32623fd59f7b24917da480c5528987202f320c2b92d")


if __name__ == "__main__":
    unittest.main()
