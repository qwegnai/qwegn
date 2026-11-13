"""Unit tests for ERNIE-5.5-Flash-Base QWP1 packer, F32 raw planes, and QLSA0001."""
from __future__ import annotations

import json, struct, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_release import audit_release
from pack import quantize_model
from verify import read_header


def build_mini_ernie55_hf_checkpoint(src: Path) -> dict:
    h, v, inter, moe_inter, lat = 128, 256, 256, 64, 64
    n_exp, topk, streams = 4, 2, 4
    cfg = {
        "architectures": ["Ernie5_5_MoeForCausalLM"],
        "model_type": "ernie5_5_moe",
        "hidden_size": h,
        "vocab_size": v,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "intermediate_size": inter,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "mtp_shared_last_layer": True,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": n_exp,
        "num_experts_per_tok": topk,
        "n_shared_experts": 1,
        "moe_intermediate_size": moe_inter,
        "moe_latent_size": lat,
        "moe_split_feature_routing": True,
        "routed_scaling_factor": 2.5,
        "routed_scaling_factor_learnable": True,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "enable_hyper_connections": True,
        "hc_mult": streams,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "multimax": "lm_head",
        "tie_word_embeddings": False,
        "compress_ratios": [4, -2, -2],
        "compress_rope_theta": 160000.0,
        "hca_rope_type": "yarn",
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "o_groups": 2,
        "o_lora_rank": 64,
        "sliding_window": 8,
        "use_vha_attention": True,
        "add_full_attention_sink_bias": True,
        "use_gated_attn": True,
        "hybrid_mla_attention": "mqa_dsa",
        "hybrid_mla_num_attention_heads": 4,
        "hybrid_mla_num_key_value_heads": 4,
        "hybrid_mla_q_lora_rank": 64,
        "hybrid_mla_kv_lora_rank": 64,
        "hybrid_mla_qk_nope_head_dim": 32,
        "hybrid_mla_qk_rope_head_dim": 32,
        "hybrid_mla_v_head_dim": 64,
        "index_n_heads": 4,
        "index_head_dim": 32,
        "index_topk": 16,
        "use_fast_hadamard": True,
        "rope_interleave": True,
        "rope_theta": 8000000.0,
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 256,
        },
    }
    (src / "config.json").write_text(json.dumps(cfg, indent=2))
    (src / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": "bos:{{ messages | length }}"}))

    gen = torch.Generator().manual_seed(42)

    def bf16(shape):
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    def f32(shape, scale=0.02):
        return torch.randn(shape, generator=gen, dtype=torch.float32) * scale

    t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": bf16((v, h)),
        "model.norm.weight": (torch.ones(h) + f32((h,), 0.01)).to(torch.bfloat16),
        "lm_head.weight": bf16((v, h)),
        "model.lm_head.multimax_ranges": torch.tensor([14.875, 15.8125, 8.75, 0.3046875], dtype=torch.float32),
        "model.lm_head.multimax_ts": torch.tensor([-0.095703125, 0.470703125, -0.15625, -0.015319824], dtype=torch.float32),
        "model.mhc_contract.hc_head_fn": f32((streams * h, streams)),
        "model.mhc_contract.hc_head_scale": torch.tensor([1.0 + 2.0**-18], dtype=torch.float32),
        "model.mhc_contract.hc_head_base": torch.tensor([0.25 + 2.0**-19] * streams, dtype=torch.float32),
    }

    def add_mhc(pfx: str):
        for sub in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            t[f"{pfx}{sub}.mapping_proj.weight"] = f32((streams * h, 24))
            t[f"{pfx}{sub}.bias"] = f32((24,))
            t[f"{pfx}{sub}.alpha_pre"] = torch.tensor([0.5], dtype=torch.float32)
            t[f"{pfx}{sub}.alpha_post"] = torch.tensor([0.75], dtype=torch.float32)
            t[f"{pfx}{sub}.alpha_res"] = torch.tensor([1.0], dtype=torch.float32)

    b0 = "model.layers.0."
    t[b0 + "input_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    t[b0 + "post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    add_mhc(b0)
    t[b0 + "self_attn.linear_q_down_proj.weight"] = bf16((64, h))
    t[b0 + "self_attn.linear_kv_proj.weight"] = bf16((64, h))
    t[b0 + "self_attn.core_attention.compressor.linear_wkv.weight"] = bf16((64, h))
    t[b0 + "self_attn.core_attention.compressor.linear_wgate.weight"] = bf16((64, h))
    t[b0 + "self_attn.q_layernorm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    t[b0 + "self_attn.kv_layernorm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    t[b0 + "self_attn.core_attention.compressor.norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    t[b0 + "self_attn.core_attention.compressor.ape"] = f32((4, 64)) + (2.0**-19)
    t[b0 + "self_attn.linear_q_up_proj.weight"] = bf16((4 * 64, 64))
    t[b0 + "self_attn.gate_proj.weight"] = bf16((2 * 64, 64))
    t[b0 + "self_attn.linear_o_group_proj"] = bf16((2 * 64, 2 * 64))
    t[b0 + "self_attn.o_proj.weight"] = bf16((h, 2 * 64))
    t[b0 + "self_attn.vha_postmix_U"] = bf16((4, 2))
    t[b0 + "self_attn.vha_postmix_V"] = bf16((4, 2))
    t[b0 + "self_attn.core_attention.attn_sink"] = f32((4,))
    t[b0 + "mlp.gate_proj.weight"] = bf16((inter, h))
    t[b0 + "mlp.up_proj.weight"] = bf16((inter, h))
    t[b0 + "mlp.down_proj.weight"] = bf16((h, inter))

    for l in (1, 2):
        bl = f"model.layers.{l}."
        t[bl + "input_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
        t[bl + "post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
        if l == 1:
            add_mhc(bl)
        else:
            add_mhc(bl + "transformer_layer.")
        t[bl + "self_attn.q_a_proj.weight"] = bf16((64, h))
        t[bl + "self_attn.kv_a_proj_with_mqa.weight"] = bf16((64 + 32, h))
        t[bl + "self_attn.core_attention.indexer.wk.weight"] = bf16((32, h))
        t[bl + "self_attn.q_a_layernorm.weight"] = torch.ones(64, dtype=torch.bfloat16)
        t[bl + "self_attn.kv_a_layernorm.weight"] = torch.ones(64, dtype=torch.bfloat16)
        t[bl + "self_attn.core_attention.indexer.k_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
        t[bl + "self_attn.core_attention.indexer.k_norm.bias"] = torch.zeros(32, dtype=torch.bfloat16)
        t[bl + "self_attn.core_attention.indexer.weights_proj.weight"] = bf16((4, h))
        t[bl + "self_attn.q_b_proj.weight"] = bf16((4 * (32 + 32), 64))
        t[bl + "self_attn.gate_proj.weight"] = bf16((4 * 64, 64))
        t[bl + "self_attn.core_attention.indexer.wq_b.weight"] = bf16((4 * 32, 64))
        t[bl + "self_attn.kv_b_proj.weight"] = bf16((4 * (32 + 64), 64))
        t[bl + "self_attn.o_proj.weight"] = bf16((h, 4 * 64))
        t[bl + "self_attn.vha_postmix_U"] = bf16((4, 2))
        t[bl + "self_attn.vha_postmix_V"] = bf16((4, 2))
        t[bl + "self_attn.core_attention.softmax_offset"] = bf16((4,))
        t[bl + "block_sparse_moe.gate.weight"] = torch.full((n_exp, h), 1.0 + 2.0**-19, dtype=torch.float32)
        t[bl + "block_sparse_moe.gate.weight_1"] = torch.full((n_exp, h), -0.5 + 2.0**-20, dtype=torch.float32)
        t[bl + "block_sparse_moe.e_score_correction_bias"] = f32((n_exp,)) + float(l)
        t[bl + "block_sparse_moe.gate.routed_scaling_factor_param"] = torch.ones(n_exp, dtype=torch.float32) * 1.5
        t[bl + "block_sparse_moe.fc1_latent_proj.weight"] = bf16((lat, h))
        t[bl + "block_sparse_moe.fc2_latent_proj.weight"] = bf16((h, lat))
        t[bl + "block_sparse_moe.latent_norm.weight"] = torch.ones(lat, dtype=torch.bfloat16)
        t[bl + "block_sparse_moe.shared_experts.w1.weight"] = bf16((moe_inter, h))
        t[bl + "block_sparse_moe.shared_experts.w2.weight"] = bf16((h, moe_inter))
        t[bl + "block_sparse_moe.shared_experts.w3.weight"] = bf16((moe_inter, h))
        for e in range(n_exp):
            t[bl + f"block_sparse_moe.experts.{e}.w1.weight"] = bf16((moe_inter, lat))
            t[bl + f"block_sparse_moe.experts.{e}.w2.weight"] = bf16((lat, moe_inter))
            t[bl + f"block_sparse_moe.experts.{e}.w3.weight"] = bf16((moe_inter, lat))

    b2 = "model.layers.2."
    t[b2 + "enorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    t[b2 + "hnorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    t[b2 + "e_proj.weight"] = bf16((h, h))
    t[b2 + "h_proj.weight"] = bf16((h, h))
    t[b2 + "hc_head_fn"] = f32((streams * h, streams))
    t[b2 + "hc_head_scale"] = torch.tensor([1.0], dtype=torch.float32)
    t[b2 + "hc_head_base"] = torch.zeros(streams, dtype=torch.float32)
    t[b2 + "shared_head.norm.weight"] = torch.ones(h, dtype=torch.bfloat16)

    keys = sorted(t.keys())
    mid = len(keys) // 2
    s1_name, s2_name = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file({k: t[k] for k in keys[:mid]}, str(src / s1_name))
    save_file({k: t[k] for k in keys[mid:]}, str(src / s2_name))
    wmap = {k: (s1_name if i < mid else s2_name) for i, k in enumerate(keys)}
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return cfg


class TestErnie55Adapter(unittest.TestCase):

    def test_sharded_pack_audit_f32_and_qlsa(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hf_src"
            src.mkdir()
            build_mini_ernie55_hf_checkpoint(src)
            out = Path(tmp) / "out_pack"
            quantize_model(src, out)

            info = audit_release(out)
            self.assertTrue(info["ok"])
            self.assertEqual(info["counts"]["hq8_g32"], 21)
            self.assertEqual(info["counts"]["shvq15_g64"], 8)
            self.assertEqual(info["counts"]["shvq3_g64"], 4)

            qwp = out / "model.qwp"
            hdr, _ = read_header(qwp)
            self.assertEqual(
                hdr["provenance"]["quant_recorded"],
                "dense_hq8_g32__expert_shvq15_g64__expdown_shvq3_g64",
            )
            tensors = hdr["tensors"]
            self.assertNotIn("model.layers.2.self_attn.q_a_proj.weight", tensors)
            self.assertNotIn("model.layers.2.mlp.experts.0.gate_proj.weight", tensors)
            self.assertIn("model.layers.2.e_proj.weight", tensors)
            self.assertIn("model.layers.2.mlp.gate.e_score_correction_bias", tensors)

            gate_entry = tensors["model.layers.1.mlp.gate.weight"]
            self.assertEqual(gate_entry["kind"], "f32")
            with qwp.open("rb") as f:
                f.seek(gate_entry["raw_off"])
                vals = np.frombuffer(f.read(8 * 128 * 4), dtype="<f4")
            self.assertEqual(vals[0], np.float32(1.0 + 2.0**-19))
            self.assertEqual(vals[4 * 128], np.float32(-0.5 + 2.0**-20))

            qlsa = out / "assets" / "model.qlsa"
            self.assertTrue(qlsa.is_file())
            raw_qlsa = qlsa.read_bytes()
            self.assertEqual(raw_qlsa[:8], b"QLSA0001")
            hlen = struct.unpack_from("<Q", raw_qlsa, 8)[0]
            qhdr = json.loads(raw_qlsa[-hlen:].decode("utf-8"))
            self.assertEqual(qhdr["provenance"]["geometry_digest"], hdr["provenance"]["geometry_digest"])
            wq_meta = qhdr["tensors"]["layers.0.wq_idx"]
            self.assertEqual(wq_meta["shape"], [32, 32])
            q_mat = np.frombuffer(
                raw_qlsa[wq_meta["raw_off"]: wq_meta["raw_off"] + wq_meta["n_bytes"]],
                dtype=np.float16,
            ).astype(np.float32).reshape(32, 32)
            gram = q_mat @ q_mat.T
            self.assertLess(float(np.max(np.abs(gram - np.eye(32, dtype=np.float32)))), 2e-2)

    def test_base_checkpoint_without_chat_template_synthesizes_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hf_base"
            src.mkdir()
            build_mini_ernie55_hf_checkpoint(src)
            (src / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|ERNIE_BEGIN|>"}))
            (src / "tokenizer.json").write_text(json.dumps({
                "version": "1.0",
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [{"SpecialToken": {"id": "<|ERNIE_MASK|>", "type_id": 0}}],
                },
            }))
            out = Path(tmp) / "out_base_pack"
            quantize_model(src, out)
            info = audit_release(out)
            self.assertTrue(info["ok"])
            self.assertTrue((out / "assets" / "chat_template.jinja").is_file())


if __name__ == "__main__":
    unittest.main()
