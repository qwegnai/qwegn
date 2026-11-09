#!/usr/bin/env python3
"""Emit the QWP1-identity provenance fields for a Flash-Next pack (plan §2,
FN-AD-7): {"model_id", "geometry_digest"} where geometry_digest is sha256 of
the canonical `qwg-geo-v2` string built by Sources/QWCore/QWModelDescriptor.

Python mirror of the Swift digest-fields order. Cross-language equality is
PINNED by Tests/QWCoreTests/QWModelDescriptorTests (the Swift test asserts
this file's _canonical_string byte-for-byte), so drift on either side fails
the gate — same discipline as the v1 geometry_hash pinning between pack.py
and QWPack. Floats are hashed by IEEE bit pattern (0x… hex), never by
language-dependent text printing: Swift prints 1e-6 as "0.000001", Python as
"1e-06" — both wrong for a cross-language digest.
Usage: python3 Scripts/oracle/emit_pack_identity.py [config.json] [out.json]
"""
import hashlib, json, struct, sys
from pathlib import Path

def f64hex(x) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "x")

def f32hex(x) -> str:
    return "0x" + format(struct.unpack(">I", struct.pack(">f", float(x)))[0], "x")

def geo_v2(tc: dict) -> str:
    num_layers = tc["num_hidden_layers"]
    raw_layer_types = tc.get("layer_types") or (["full_attention"] * num_layers)
    layer_kinds = "".join("F" if t in ("full_attention", "deepseek_sparse_attention") else "L" for t in raw_layer_types)
    if tc.get("full_attention_interval") is not None:
        fa_interval = tc["full_attention_interval"]
    elif "F" in layer_kinds:
        fa_interval = layer_kinds.index("F") + 1
    else:
        fa_interval = 1
    hd = tc.get("head_dim") or tc.get("qk_head_dim") or (
        (tc.get("qk_nope_head_dim") or 0) + (tc.get("qk_rope_head_dim") or 0)
    ) or (tc["hidden_size"] // tc["num_attention_heads"])
    rope_params = tc.get("rope_parameters") or {}
    rot_f = tc.get("partial_rotary_factor")
    if rot_f is None:
        rot_f = rope_params.get("partial_rotary_factor", 1.0)
    theta = tc.get("rope_theta") or rope_params.get("rope_theta") or 10000.0
    gate_on = (bool(tc.get("attn_output_gate"))
               or tc.get("output_gate_type") == "sigmoid"
               or tc.get("model_type") in ("qwen3_5", "qwen3_5_text", "qwen4_exp", "qwen4_exp_text"))
    qgate = 1 if gate_on else 0
    lin_cfg = tc.get("linear_attn_config") or {}
    gdn_qk = tc.get("linear_num_key_heads") or lin_cfg.get("num_heads") or 0
    gdn_v = tc.get("linear_num_value_heads") or lin_cfg.get("num_heads") or 0
    gdn_hd = tc.get("linear_key_head_dim") or lin_cfg.get("head_dim") or 0
    gdn_ck = tc.get("linear_conv_kernel_dim") or lin_cfg.get("short_conv_kernel_size") or 0
    mtp_layers = tc.get("mtp_num_hidden_layers")
    if mtp_layers is None:
        mtp_layers = tc.get("num_nextn_predict_layers", 0) or 0
    fields = [
        f"model_type={tc['model_type']}",
        f"hidden={tc['hidden_size']}",
        f"vocab={tc['vocab_size']}",
        f"layers={num_layers}",
        f"layer_kinds={layer_kinds}",
        f"fa_interval={fa_interval}",
        f"q={tc['num_attention_heads']}",
        f"kv={tc['num_key_value_heads']}",
        f"hd={hd}",
        f"qgate={qgate}",
        f"rot_f={f64hex(rot_f)}",
        f"theta={f64hex(theta)}",
        f"gdn={gdn_qk},{gdn_v},{gdn_hd},{gdn_ck}",
        f"mtp={mtp_layers},{1 if tc.get('mtp_use_dedicated_embeddings') else 0}",
        f"eos={tc['eos_token_id'][0] if isinstance(tc['eos_token_id'], list) else tc['eos_token_id']}",
        f"rms={f32hex(tc['rms_norm_eps'])}",
        f"maxpos={tc['max_position_embeddings']}",
    ]
    z = lambda k: tc.get(k) if tc.get(k) is not None else 0
    idx_heads = tc.get("indexer_n_heads") if tc.get("indexer_n_heads") is not None else (tc.get("index_n_heads") or 0)
    idx_kv = tc.get("indexer_kv_heads") if tc.get("indexer_kv_heads") is not None else (1 if idx_heads else 0)
    idx_hd = tc.get("indexer_head_dim") if tc.get("indexer_head_dim") is not None else (tc.get("index_head_dim") or 0)
    idx_budget = tc.get("indexer_budget") if tc.get("indexer_budget") is not None else (tc.get("index_topk") or 0)
    idx_comp = tc.get("indexer_compress_ratio")
    if idx_comp is None:
        idx_comp = 4 if tc.get("index_kpool_compress") else (1 if idx_heads else 0)
    fields.append("idx={},{},{},{},{}".format(idx_heads, idx_kv, idx_hd, idx_budget, idx_comp))
    eg = tc.get("scoring_func") or (
        "sigmoid" if tc.get("model_type") in ("qwen4_exp", "qwen4_exp_text", "glm5_next", "glm5_next_text") else "-"
    )
    num_exp = tc.get("num_experts") if tc.get("num_experts") is not None else (tc.get("n_routed_experts") or 0)
    moe_int = z("moe_intermediate_size")
    shared_int = tc.get("shared_expert_intermediate_size")
    if shared_int is None:
        n_sh = tc.get("n_shared_experts") or 0
        shared_int = (n_sh * moe_int) if n_sh > 0 else 0
    fields.append("moe={},{},{},{},{}".format(num_exp, z("num_experts_per_tok"),
                  moe_int, shared_int, eg))
    fields.append("hc={},{}".format(z("hc_count"), z("hc_lowrank")))
    ple_ids = tc.get("ple_layer_ids") or []
    fields.append("ple={},{},{},{},{},{},{},{}".format(
        ",".join(str(i) for i in ple_ids), z("ple_embed_dim"), z("ple_conv_kernel_size"),
        z("ngram_size"), z("heads_per_ngram"), z("ngram_vocab_size_base"),
        z("make_ngram_vocab_size_divisible_by"), z("split_ngram_parts")))
    return "qwg-geo-v2:" + "|".join(fields)

MODEL_ID = {
    "qwen4_exp_text": "qwen3.8-flash-next",
    "qwen4_exp": "qwen3.8-flash-next",
    "qwen3_5": "qwen3.8-27b",
    "qwen3_5_text": "qwen3.8-27b",
    "qwen3": "qwen3-dense",
    "qwen3_moe": "qwen3-moe",
    "qwen3_5_moe": "qwen3-moe",
    "qwen3_5_moe_text": "qwen3-moe",
    "glm5_next": "glm5_next",
    "glm5_next_text": "glm5_next",
    "qwen2": "qwen2-dense",
    "llama": "llama-dense",
    "mistral": "mistral-dense",
}

def main() -> int:
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "Scripts/oracle/config_flashnext_golden.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "Scripts/oracle/flashnext_pack_identity.json")
    cfg = json.loads(cfg_path.read_text())
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    geo = geo_v2(tc)
    identity = {
        "model_id": MODEL_ID[tc["model_type"]],
        "geometry_digest": hashlib.sha256(geo.encode()).hexdigest(),
        "_canonical_string": geo,
        "_note": "pack.py successor injects model_id+geometry_digest into provenance (FN-AD-7); _canonical_string/_note are documentation, NOT packed",
    }
    out_path.write_text(json.dumps(identity, indent=1) + "\n")
    print(f"wrote {out_path}\nmodel_id        = {identity['model_id']}\ngeometry_digest = {identity['geometry_digest']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
