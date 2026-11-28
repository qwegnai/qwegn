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
from __future__ import annotations

import hashlib, json, struct, sys
from pathlib import Path

def f64hex(x) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "x")

def f32hex(x) -> str:
    return "0x" + format(struct.unpack(">I", struct.pack(">f", float(x)))[0], "x")

def geo_v2(tc: dict, parent_mt: str | None = None) -> str:
    mt = tc.get("model_type") or parent_mt or ""
    num_layers = tc["num_hidden_layers"]
    raw_ratios = tc.get("compress_ratios")
    if tc.get("layer_types") is not None:
        raw_layer_types = tc["layer_types"]
    elif raw_ratios and len(raw_ratios) >= num_layers:
        raw_layer_types = [
            "compressed_attention" if r > 0 else "full_attention"
            for r in raw_ratios[:num_layers]
        ]
    elif tc.get("layer_group_size"):
        gs = int(tc["layer_group_size"])
        raw_layer_types = [
            "full_attention" if (i + 1) % gs == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    elif tc.get("full_attention_layers") is not None or tc.get("linear_attention_type") is not None:
        fa_set = {int(x) for x in (tc.get("full_attention_layers") or [])}
        raw_layer_types = [
            "full_attention" if i in fa_set else "linear_attention"
            for i in range(num_layers)
        ]
    else:
        raw_layer_types = ["full_attention"] * num_layers
    layer_kinds = "".join(
        "F" if t in ("full_attention", "deepseek_sparse_attention")
        else ("C" if t == "compressed_attention" else ("S" if t == "sliding_attention" else "L"))
        for t in raw_layer_types
    )
    if tc.get("full_attention_interval") is not None:
        fa_interval = tc["full_attention_interval"]
    elif tc.get("layer_group_size") is not None:
        fa_interval = int(tc["layer_group_size"])
    elif any(ch in ("F", "S") for ch in layer_kinds):
        fa_indices = [i for i, ch in enumerate(layer_kinds) if ch in ("F", "S")]
        cand = fa_indices[0] + 1
        fa_interval = cand if all((i + 1) % cand == 0 for i in fa_indices) else 0
    else:
        fa_interval = 1
    hd = tc.get("head_dim") or tc.get("attention_head_size") or tc.get("qk_head_dim") or (
        (tc.get("qk_nope_head_dim") or 0) + (tc.get("qk_rope_head_dim") or 0)
    ) or (tc["hidden_size"] // tc["num_attention_heads"])
    rope_params = tc.get("rope_parameters") or {}
    fa_rope = rope_params.get("full_attention") or {}
    sa_rope = rope_params.get("sliding_attention") or {}
    rot_f = tc.get("partial_rotary_factor")
    if rot_f is None:
        rot_f = rope_params.get("partial_rotary_factor")
    if rot_f is None:
        rot_f = fa_rope.get("partial_rotary_factor")
    if rot_f is None:
        rot_f = sa_rope.get("partial_rotary_factor")
    if rot_f is None:
        rot_f = tc.get("rotary_percent")
    if rot_f is None:
        rot_f = 1.0
    theta = (
        tc.get("rope_theta")
        or rope_params.get("rope_theta")
        or fa_rope.get("rope_theta")
        or tc.get("global_rope_theta")
        or tc.get("local_rope_theta")
        or 10000.0
    )
    if "attn_output_gate" in tc and tc["attn_output_gate"] is not None:
        gate_on = bool(tc["attn_output_gate"])
    elif "use_gated_attn" in tc and tc["use_gated_attn"] is not None:
        gate_on = bool(tc["use_gated_attn"])
    elif "gated_attention" in tc and tc["gated_attention"] is not None:
        gate_on = bool(tc["gated_attention"])
    else:
        gran = (tc.get("gated_attention_proj_granularity_type") or "").lower()
        gate_on = (tc.get("output_gate_type") is not None
                   or gran in ("head_wise", "element_wise")
                   or mt in (
                       "qwen3_5", "qwen3_5_text", "qwen4_exp", "qwen4_exp_text",
                       "lfm3", "lfm3_text", "lfm3_vl",
                   ))
    qgate = 1 if gate_on else 0
    lin_cfg = tc.get("linear_attn_config") or {}
    has_lin = any(t == "linear_attention" for t in raw_layer_types)
    lin_kv = tc.get("num_kv_heads_for_linear_attn")
    def_lin_qk = lin_kv if (lin_kv is not None and lin_kv > 0) else tc["num_attention_heads"]
    gdn_qk = (tc.get("linear_num_key_heads") or lin_cfg.get("num_heads") or def_lin_qk) if has_lin else 0
    gdn_v = (tc.get("linear_num_value_heads") or lin_cfg.get("num_heads") or tc["num_attention_heads"]) if has_lin else 0
    gdn_hd = (tc.get("linear_key_head_dim") or lin_cfg.get("head_dim") or hd) if has_lin else 0
    gdn_ck = (tc.get("linear_conv_kernel_dim") or lin_cfg.get("short_conv_kernel_size") or tc.get("short_conv_kernel_size") or 0) if has_lin else 0
    mtp_layers = tc.get("mtp_num_hidden_layers")
    if mtp_layers is None:
        mtp_layers = tc.get("num_nextn_predict_layers")
    if mtp_layers is None:
        mtp_layers = tc.get("mtp_num_layers", 0) or 0
    kv_heads = tc.get("num_key_value_heads", tc["num_attention_heads"])
    eos_raw = tc.get("eos_token_id", 0)
    eos_id = eos_raw[0] if isinstance(eos_raw, list) else eos_raw
    norm_eps = tc.get("rms_norm_eps") or tc.get("norm_eps") or tc.get("layer_norm_eps") or 1e-5
    max_pos = tc.get("max_position_embeddings", 8192)
    fields = [
        f"model_type={mt}",
        f"hidden={tc['hidden_size']}",
        f"vocab={tc['vocab_size']}",
        f"layers={num_layers}",
        f"layer_kinds={layer_kinds}",
        f"fa_interval={fa_interval}",
        f"q={tc['num_attention_heads']}",
        f"kv={kv_heads}",
        f"hd={hd}",
        f"qgate={qgate}",
        f"rot_f={f64hex(rot_f)}",
        f"theta={f64hex(theta)}",
        f"gdn={gdn_qk},{gdn_v},{gdn_hd},{gdn_ck}",
        f"mtp={mtp_layers},{1 if tc.get('mtp_use_dedicated_embeddings') else 0}",
        f"eos={eos_id}",
        f"rms={f32hex(norm_eps)}",
        f"maxpos={max_pos}",
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
    eg = tc.get("scoring_func") or tc.get("router_score_function") or tc.get("score_function") or (
        "sigmoid" if mt in (
            "qwen4_exp", "qwen4_exp_text", "glm5_next", "glm5_next_text",
            "ernie5_5_moe", "ernie5_5_moe_text", "maxwell_moe",
            "bailing_moe_v3", "bailing_moe_v3_vl", "bailing_moe_v3_text",
            "gigachat4",
        ) else "-"
    )
    num_exp = tc.get("num_experts") if tc.get("num_experts") is not None else (tc.get("n_routed_experts") or 0)
    moe_int = z("moe_intermediate_size")
    shared_int = tc.get("shared_expert_intermediate_size")
    if shared_int is None and tc.get("moe_shared_expert_intermediate_size") is not None:
        n_sh = tc.get("num_shared_experts") if tc.get("num_shared_experts") is not None else (tc.get("n_shared_experts") or 1)
        shared_int = tc["moe_shared_expert_intermediate_size"] * max(1, n_sh)
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
    moe_lat = tc.get("moe_latent_size") or 0
    if moe_lat > 0:
        mtp_shared = 1 if tc.get("mtp_shared_last_layer") else 0
        multimax = 1 if tc.get("multimax") == "lm_head" else 0
        fields.append(f"lat={moe_lat},{mtp_shared},{multimax}")
    if "C" in layer_kinds and raw_ratios:
        pos_ratios = [r for r in raw_ratios if r > 0]
        def_ratio = pos_ratios[0] if pos_ratios else 128
        o_groups = tc.get("o_groups") or 8
        o_lora = tc.get("o_lora_rank") or 1024
        sw = tc.get("sliding_window") or 128
        fields.append(f"hca={def_ratio},{o_groups},{o_lora},{sw}")
    return "qwg-geo-v2:" + "|".join(fields)

MODEL_ID = {
    "qwen4_exp_text": "qwen3.8-flash-next",
    "qwen4_exp": "qwen3.8-flash-next",
    "qwen3_5": "qwen3.8-27b",
    "qwen3_5_text": "qwen3.8-27b",
    "qwen3": "qwen3-dense",
    "qwen3_moe": "qwen3-moe",
    "qwen3_5_moe": "qwen3.8-moe",
    "qwen3_5_moe_text": "qwen3.8-moe",
    "sensenova_loop": "sensenova_loop",
    "sensenova_loop_text": "sensenova_loop",
    "glm5_next": "glm5_next",
    "glm5_next_text": "glm5_next",
    "ernie5_5_moe": "ernie5_5_moe",
    "ernie5_5_moe_text": "ernie5_5_moe",
    "maxwell_moe": "maxwell_moe",
    "bailing_moe_v3": "bailing_moe_v3",
    "bailing_moe_v3_vl": "bailing_moe_v3",
    "bailing_moe_v3_text": "bailing_moe_v3",
    "qwen2": "qwen2-dense",
    "llama": "llama-dense",
    "mistral": "mistral-dense",
    "lfm2": "lfm2",
    "lfm3": "lfm3",
    "lfm3_vl": "lfm3_vl",
    "lfm2_encoder": "lfm2_encoder",
    "d1_omni": "d1_omni",
    "modernbert": "modernbert",
    "deberta_v2": "deberta_v2",
    "gigachat4": "gigachat4",
}

def main() -> int:
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "Scripts/oracle/config_flashnext_golden.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "Scripts/oracle/flashnext_pack_identity.json")
    cfg = json.loads(cfg_path.read_text())
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    mt = tc.get("model_type") or cfg.get("model_type", "")
    geo = geo_v2(tc, parent_mt=cfg.get("model_type"))
    identity = {
        "model_id": MODEL_ID[mt],
        "geometry_digest": hashlib.sha256(geo.encode()).hexdigest(),
        "_canonical_string": geo,
        "_note": "pack.py successor injects model_id+geometry_digest into provenance (FN-AD-7); _canonical_string/_note are documentation, NOT packed",
    }
    out_path.write_text(json.dumps(identity, indent=1) + "\n")
    print(f"wrote {out_path}\nmodel_id        = {identity['model_id']}\ngeometry_digest = {identity['geometry_digest']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
