#!/usr/bin/env python3
"""FN-AD-8 §4 golden generator: run the PINNED reference implementation
(transformers 5.17.0 models/qwen4_exp, CPU, fp32) on deterministic synthetic
inputs and emit .npy fixtures + a manifest the Swift differential-parity
tests read. The Swift side must reproduce these tensors within declared
tolerances before any optimized kernel is trusted.

Components fixed here (plan §4 order):
  hyper_connection — grouped RMSNorm(1+w), silu(down/4), sigmoid(up), mean-mix,
                     2·sigmoid(inject), lane write-back
  moe_router       — softmax fp32 → top-10 → /sum → bf16-cast weights;
                     per-token SwiGLU + shared expert + sigmoid gate accumulation
  gdn              — chunked gated delta rule (qk l2norm eps 1e-6, scaling),
                     conv k=4 silu, RMSNormGated(norm-before-gate), out_proj;
                     recurrent-vs-chunk parity is part of the fixture
  ple              — hash head ids (cross-check vs derive_ple_hash_replica)

Numeric policy: inputs bf16 (checkpoint dtype), math fp32 exactly as the
reference does; tolerances recorded per component in the manifest.
"""
import json, math, sys
from pathlib import Path

import math
import numpy as np
import torch
torch.manual_seed(20261011)

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # pinned reference

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "Tests/QWCoreTests/Fixtures/fn")
ROOT.mkdir(parents=True, exist_ok=True)
CFG = json.loads(Path("Scripts/oracle/config_flashnext_golden.json").read_text())["text_config"]

class MiniCfg:
    """FN-faithful ratios shrunk for CPU determinism: same GDN head geometry,
    hc_count/lowrank, activation rules; smaller hidden/experts/layers."""
    model_type = "qwen4_exp_text"
    hidden_size = 64
    vocab_size = 320
    num_hidden_layers = 2
    layer_types = ["linear_attention", "full_attention"]
    full_attention_interval = 2
    rms_norm_eps = 1e-6
    max_position_embeddings = 512
    num_attention_heads = 4; num_key_value_heads = 2; head_dim = 16
    partial_rotary_factor = 0.25; rope_theta = 10000000.0
    output_gate_type = "sigmoid"; hidden_act = "silu"
    linear_key_head_dim = 8; linear_num_key_heads = 2
    linear_num_value_heads = 6; linear_value_head_dim = 8
    linear_conv_kernel_dim = 4
    num_experts = 8; num_experts_per_tok = 3; norm_topk_prob = True
    moe_intermediate_size = 16; shared_expert_intermediate_size = 16
    hc_count = 4; hc_lowrank = 8
    mtp_num_hidden_layers = 1; mtp_use_dedicated_embeddings = False
    eos_token_id = 248044
    seed = 1234
    bos_token_id = 248044
    ngram_size = 3; heads_per_ngram = 8; ngram_vocab_size_base = 17
    make_ngram_vocab_size_divisible_by = 128; split_ngram_parts = 128
    ple_layer_ids = [2]; ple_embed_dim = 64; ple_conv_kernel_size = 4
    indexer_budget = 8; indexer_compress_ratio = 2; indexer_head_dim = 8
    indexer_kv_heads = 1; indexer_n_heads = 2
    rope_parameters = {"rope_type": "default", "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25}
    attention_bias = False; attention_dropout = 0.0
    _attn_implementation = "eager"    # inference default in this pinned version
    initializer_range = 0.02; tie_word_embeddings = False
    use_cache = True; output_router_logits = False
    _experts_implementation = "eager"   # integrations/moe.py hook attr

def save(name, t: torch.Tensor):
    a = t.detach().cpu().contiguous()   # non-contiguous views save as fortran_order=True
    if a.dtype == torch.bfloat16:
        # numpy has no bf16: store the raw IEEE bit patterns as uint16 and
        # mark the manifest so readers reinterpret as bf16 (exact, lossless).
        bits = a.view(torch.uint16).numpy()
        np.save(ROOT / f"{name}.npy", bits)
        return {"shape": list(bits.shape), "dtype": "bf16"}
    a = a.numpy()
    np.save(ROOT / f"{name}.npy", a)
    return {"shape": list(a.shape), "dtype": str(a.dtype)}

def det(shape, scale=0.4, kind="bf16"):
    t = torch.randn(*shape, dtype=torch.float32) * scale
    return t.to(torch.bfloat16) if kind == "bf16" else t

def to_bf16(module):
    """Real model IO is bf16 (config torch_dtype=bfloat16); the reference casts
    to fp32 internally where it needs to. Match the shipping dtype."""
    module.to(torch.bfloat16)
    return module

manifest = {"note": "FN §4 reference-parity fixtures; generated from transformers qwen4_exp on CPU fp32 math, bf16 IO; seed 20261011",
            "components": {}}

def dump_params(prefix, module):
    for n, t in list(module.named_parameters()) + list(module.named_buffers()):
        save(f"{prefix}__{n.replace('.', '__')}", t)

# ---------------- hyper-connection (Qwen4ExpTextGatedResidual) --------------
cfg = MiniCfg()
hc = to_bf16(M.Qwen4ExpTextGatedResidual(cfg))
with torch.no_grad():
    for p in hc.parameters(): p.normal_(0, 0.15)
    hc.hc_norm.weight.data.normal_(0, 0.1)   # note: (1+w) rule
x = det((1, 5, cfg.hc_count * cfg.hidden_size))
mixed, hyper_in, inj = hc(x)
manifest["components"]["hyper_connection"] = {
    "input": save("hc_input", x),
    "mixed": save("hc_mixed", mixed),
    "hyper_input_echo": save("hc_hyper_echo", hyper_in),
    "injection_weights": save("hc_inject", inj),
    "tolerance": {"abs": 1e-2, "rel": 3e-2, "why": "bf16 in/out, fp32 math, GEMMs of size 20/2560 scale down to 64 — Metal f16 accumulate differs in last bits"},
}
dump_params("hc", hc)
# write-back composition: hidden = hyper_input + (out ⊗ inject).sum-lane
out = det((1, 5, cfg.hidden_size), 0.5)
writeback = (hyper_in + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2))
save("hc_writeback_out", out); save("hc_writeback", writeback)

# ---------------- MoE (router + experts + shared + gate) --------------------
moe = to_bf16(M.Qwen4ExpTextSparseMoeBlock(cfg))
with torch.no_grad():
    for p in moe.parameters(): p.normal_(0, 0.2)
x = det((1, 6, cfg.hidden_size))
y = moe(x)
_, rw, sel = moe.gate(x.view(-1, cfg.hidden_size))
manifest["components"]["moe"] = {
    "input": save("moe_input", x),
    "output": save("moe_output", y),
    "routing_weights": save("moe_routing", rw),
    "selected_experts": save("moe_selected", sel.to(torch.int64)),
    "tolerance": {"abs": 1e-2, "rel": 3e-2},
    "tie_law": "torch.topk order = descending value, stable by index; Swift must match (documented in FN-AD-8)",
}
dump_params("moe", moe)

# ---------------- GDN (chunk vs recurrent parity + full block) --------------
gdn = to_bf16(M.Qwen4ExpTextGatedDeltaNet(cfg, layer_idx=0))
with torch.no_grad():
    for n, p in gdn.named_parameters():
        if "A_log" in n: p.uniform_(0.01, 16).log_()
        elif "dt_bias" in n: p.fill_(0.0)
        elif n == "norm.weight": p.fill_(1.0)
        else: p.normal_(0, 0.2)
x = det((1, 70, cfg.hidden_size))     # 70 > chunk 64 → multi-chunk path
# build q,k,v exactly as the module does for the core scan:
import torch.nn.functional as F
mixed_qkv = gdn.in_proj_qkv(x).transpose(1, 2)
# nn.Conv1d(padding=k-1) places causal taps at out[t] = Σⱼ in[t−k+1+j]·w[j]
# (zero left-pad); the causal window is out[:, :, :L] — taking [k-1:] would
# look k−1 taps into the FUTURE (caught by differential-parity 2026-10-11).
conv = gdn.conv1d(mixed_qkv)[:, :, :x.shape[1]]
conv = F.silu(conv).transpose(1, 2)
qd, kd, v = torch.split(conv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
q = qd.reshape(1, -1, gdn.num_k_heads, gdn.head_k_dim)
k = kd.reshape(1, -1, gdn.num_k_heads, gdn.head_k_dim)
v = v.reshape(1, -1, gdn.num_v_heads, gdn.head_v_dim)
b = gdn.in_proj_b(x).sigmoid()
a = gdn.in_proj_a(x)
g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
rep = gdn.num_v_heads // gdn.num_k_heads
q = q.repeat_interleave(rep, dim=2); k = k.repeat_interleave(rep, dim=2)
out_chunk, st_chunk = M.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=b,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True, chunk_size=64)
out_rec, st_rec = M.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=b,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
z = gdn.in_proj_z(x).reshape(1, -1, gdn.num_v_heads, gdn.head_v_dim)
save("gdn_z", z)   # isolate the norm law from in_proj_z GEMM accumulation order
core_normed = gdn.norm(out_chunk.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
core_flat = core_normed.reshape(1, -1, gdn.value_dim)
out_proj = gdn.out_proj(core_flat)
parity = (out_chunk - out_rec).abs().max().item()
manifest["components"]["gdn"] = {
    "input": save("gdn_input", x),
    "conv_mixed": save("gdn_conv_mixed", conv),   # [1, T, C] post-silu (was double-transposed)
    "beta": save("gdn_beta", b), "g": save("gdn_g", g),
    "out_chunk": save("gdn_out_chunk", out_chunk),
    "out_recurrent": save("gdn_out_recurrent", out_rec),
    "state_final": save("gdn_state_final", st_chunk),
    "normed": save("gdn_normed", core_flat),
    "out_proj": save("gdn_out_proj", out_proj),
    "chunk_vs_recurrent_maxabs": parity,
    "tolerance": {"abs": 6e-3, "rel": 5e-2,
                  "why": "chunk-vs-recurrent parity is fp32 algorithmic (UT-transform vs per-token); Swift reference mirrors the RECURRENT walk; kernel parity measured against this fixture"},
}
dump_params("gdn", gdn)

# ---------------- QSA indexer (selection law) ----------------
idx = to_bf16(M.Qwen4ExpTextQSAIndexer(cfg, layer_idx=0))
xi = det((1, 24, cfg.hidden_size))
pos = torch.arange(xi.shape[1])
L = xi.shape[1]
causal = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)
# The tables come from the REFERENCE rotary module itself: inv_freq from
# rope_parameters, 3×mrope position stack (identical rows for pure text),
# cos/sin in fp32, returned .to(dtype=x.dtype) → bf16 (torch rounds each op).
rotary = M.Qwen4ExpTextRotaryEmbedding(cfg)
pos_stack = pos.view(1, 1, -1).expand(3, 1, -1)     # (3, bs, L) text rows identical
cos, sin = rotary(xi, pos_stack)                    # each [1, L, rotary_dim] bf16
with torch.no_grad():
    sel_mask = idx(xi, (cos, sin), causal, None)              # [1,1,L,L] bool
save("qsa_input", xi)
save("qsa_positions", pos.to(torch.int64))
save("qsa_cos", cos.float()); save("qsa_sin", sin.float())
# The mask IS the selection contract (selected rows are recoverable: the True
# positions per query, ascending). dump_params stores weights.
save("qsa_mask", sel_mask.view(L, L).to(torch.int64))
dump_params("qsa", idx)
manifest["components"]["qsa_indexer"] = {
    "mask": {"shape": [L, L], "dtype": "int64-bool"},
    "tolerance": {"exact": True},
    "law": "per-query selected-token mask; Swift mirror must reproduce the mask bit-exactly (selection integer contract)",
}

# ---------------- QSA full attention (dense math + value gate) --------------
# The math the full-attention layers run (block-sparse attention uses the same
# kernels with a selected-subset K/V; the dense path is its correctness floor).
# FN-AD-8: intermediates come from the REFERENCE'S OWN ops (never a paraphrase).
import torch.nn as nn
qh, kh, hd_ = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
xat = det((1, 24, cfg.hidden_size))
attn_mod = to_bf16(M.Qwen4ExpTextAttention(cfg, layer_idx=0))
# cos/sin are the canonical bf16 tables from the reference rotary module
# (built once in the indexer section; the layer's cos[:, -seq:, :] slice is a
# no-op at seq == L).
# full module forward (eager): mask = causal float additive (0 / -inf)
NEG = torch.finfo(torch.float32).min
full_mask = torch.where(causal, torch.tensor(0.0), torch.tensor(NEG)).view(1, 1, L, L).to(torch.float32)
with torch.no_grad():
    (full_out, full_attn_w) = attn_mod(xat, (cos, sin), full_mask, None)
    # the layer's OWN indexer mask (combined into scores by the reference)
    sel_mask_f = attn_mod.indexer(xat, (cos, sin), full_mask, None)
    combined_mask = full_mask + sel_mask_f
# step-by-step with the reference's OWN functions for intermediate fixtures
w_q = attn_mod.q_proj.weight.detach(); w_k = attn_mod.k_proj.weight.detach()
w_v = attn_mod.v_proj.weight.detach(); w_o = attn_mod.o_proj.weight.detach()
qkv = attn_mod.q_proj(xat)
qst, gate = torch.chunk(qkv.view(1, 24, qh, hd_ * 2), 2, dim=-1)
query_states = attn_mod.q_norm(qst.reshape(1, 24, qh, hd_).transpose(1, 2))
key_states = attn_mod.k_norm(attn_mod.k_proj(xat).reshape(1, 24, kh, hd_).transpose(1, 2))
value_states = attn_mod.v_proj(xat).reshape(1, 24, kh, hd_).transpose(1, 2)
q_rot, k_rot = M.apply_rotary_pos_emb(query_states, key_states, cos, sin)   # module unsqueezes (dim=1)
gate_s = torch.sigmoid(gate.reshape(1, 24, qh * hd_))
k_rep = k_rot.repeat_interleave(qh // kh, dim=1)
v_rep = value_states.repeat_interleave(qh // kh, dim=1)
scores = torch.matmul(q_rot, k_rep.transpose(2, 3)) * attn_mod.scaling
scores = scores + combined_mask      # causal + indexer selection (reference law)
attnw = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q_rot.dtype)
ctx = torch.matmul(attnw, v_rep).transpose(1, 2).contiguous()
out_ref = attn_mod.o_proj((ctx.reshape(1, 24, qh * hd_)) * gate_s.to(ctx.reshape(1, 24, qh * hd_).dtype))
assert torch.equal(out_ref, full_out) or (out_ref - full_out).abs().max() < 1e-2, "manual vs module forward"
save("attn_input", xat)
save("attn_q_rot", q_rot.float())
save("attn_k_rot", k_rot.float())
save("attn_gate", gate.float())
save("attn_attn", attnw.float())
save("attn_ctx", ctx.reshape(1, 24, qh * hd_).float())
save("attn_out", full_out.float())
save("attn_combined_mask", combined_mask.float())
save("attn_scores", scores.float())
save("attn_v_proj_w", w_v.float())
save("attn_o_proj_w", w_o.float())
dump_params("attn_q", attn_mod.q_proj); dump_params("attn_qn", attn_mod.q_norm)
dump_params("attn_k", attn_mod.k_proj); dump_params("attn_kn", attn_mod.k_norm)
manifest["components"]["attn"] = {
    "out": {"shape": [1, 24, cfg.hidden_size], "dtype": "float32", "npy": "attn_out.npy"},
    "tolerance": {"abs": 1e-2, "rel": 1e-2},
    "law": "q_proj bf16 → view(t,qh,hd·2) chunk(Q|gate) → q RMSNorm(1+w) fp32→bf16; k= k_proj→k_norm; partial RoPE via apply_rotary_pos_emb with fp32 tables (PROMOTES to fp32; rotated part only, rest passthrough per reference concat law) → scores=q_rot·k_rot^T·scaling fp32 + combined(causal&indexer) mask → softmax fp32→bf16 → ·V bf16 → transpose→reshape bf16; out = o_proj(ctx × bf16(sigmoid(fp32 gate)))",
}

# ---------------- PLE layer (lookup + gating + dilated short conv) ----------
# FN-AD-8: run the reference Qwen4ExpTextPLELayer itself; every intermediate
# from its own submodules. Laws (pinned in the manifest):
#  embeds  = ngram_embedding(ngram_ids) (pure gather; ids law already bit-exact
#            via flashnext_ple_hash_golden.json / QWFnPLE)
#  key     = key_proj(embeds) → norm_key grouped RMSNorm (group=hidden, (1+w))
#  value   = value_proj(embeds)
#  q_n     = norm_query(hidden) grouped RMSNorm
#  gate    = bf16(Σ_lane bf16(k*q) fp32-accum) / √hidden → bf16; symmetric sqrt
#            gate = bf16(sqrt(max(|g|,1e-6)))·sign(g); sigmoid bf16; ×value bf16
#  n_c     = norm_conv(gated flat); short conv: pad LEFT (k−1)·ngram dilation,
#            depthwise dilation=ngram, silu; output = gated + conv (bf16 round)
ple_layer = to_bf16(M.Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0))
# short conv window law (reference _short_conv): pad LEFT (k−1)·dilation = 9,
# slice to −(state+seq), dilated depthwise conv (span (k−1)·dilation+1 = 10) →
# output length is exactly seq — causal, zero-padded history, no truncation.
# (seq=10 chosen so the ngram segment/eos law is exercised over a longer span.)
ple_hidden = det((1, 10, cfg.hc_count * cfg.hidden_size))
ple_ids = torch.tensor([[5, 77, 300, 248044, 9, 250, 31, 4, 88, 12]], dtype=torch.int64)  # mid-seq eos → segment law
with torch.no_grad():
    ple_out = ple_layer(ple_hidden, ple_ids, None, conv_mask=None)
    emb = ple_layer.ple_embedding(ple_ids, None)                       # (1,6,ple_embed_dim)
    ngram_ids_probe = None
    # recompute ngram ids with the reference's own internals (heads law):
    ih = ple_ids.long()
    ctx = torch.full((1, ple_layer.ple_embedding.context_len), cfg.eos_token_id, dtype=torch.int64)
    hist = torch.cat([ctx, ih], dim=-1)
    shifts = [ple_layer.ple_embedding._shift_right_ignore_eos(hist, s) for s in range(cfg.ngram_size)]
    blocks = []
    for ng in range(2, cfg.ngram_size + 1):
        mixed = shifts[0] * ple_layer.ple_embedding.layer_multipliers[0]
        for pos in range(1, ng):
            mixed = torch.bitwise_xor(mixed, shifts[pos] * ple_layer.ple_embedding.layer_multipliers[pos])
        s0 = (ng - 2) * cfg.heads_per_ngram
        blocks.append(torch.remainder(mixed.unsqueeze(-1),
                        ple_layer.ple_embedding.ngram_heads_vocab_sizes[s0:s0+cfg.heads_per_ngram].view(1,1,-1))
                      + ple_layer.ple_embedding.ngram_heads_offsets[s0:s0+cfg.heads_per_ngram].view(1,1,-1))
    ngram_ids_probe = torch.cat(blocks, dim=-1)[:, -ih.shape[1]:]
    key_n = ple_layer.norm_key(ple_layer.key_proj(emb)).view(1, 10, cfg.hc_count, cfg.hidden_size)
    value = ple_layer.value_proj(emb)
    q_n = ple_layer.norm_query(ple_hidden).view(1, 10, cfg.hc_count, cfg.hidden_size)
    gate = (key_n * q_n).sum(dim=-1, keepdim=True) / math.sqrt(cfg.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_flat = gated.flatten(-2)
    gated_normed = ple_layer.norm_conv(gated_flat)
    conv_in = F.pad(gated_normed.transpose(1, 2), (ple_layer.short_conv_state_len, 0))
    conv_in = conv_in[..., -(ple_layer.short_conv_state_len + 10):]
    conv_out = F.silu(ple_layer.conv1d(conv_in)).transpose(1, 2)
    ref_out = gated_flat + conv_out
    assert (ref_out - ple_out).abs().max().item() < 1e-1, "manual PLE replay diverged from module forward"
save("ple_input_hidden", ple_hidden)
save("ple_input_ids", ple_ids)
save("ple_ngram_ids", ngram_ids_probe.to(torch.int64))
save("ple_embeds", emb.float() if emb.dtype != torch.bfloat16 else emb)
save("ple_key_normed", key_n.float())
save("ple_value", value.float())
save("ple_query_normed", q_n.float())
save("ple_gate", gate.float())
save("ple_gated", gated_flat.float())
save("ple_gated_normed", gated_normed.float())
save("ple_short_conv", conv_out.float())
save("ple_out", ple_out.float())
save("ple_padded_conv_in", conv_in.float())
dump_params("ple_kp", ple_layer.key_proj); dump_params("ple_vp", ple_layer.value_proj)
dump_params("ple_nk", ple_layer.norm_key); dump_params("ple_nq", ple_layer.norm_query)
dump_params("ple_nc", ple_layer.norm_conv); dump_params("ple_c1", ple_layer.conv1d)
dump_params("ple_emb", ple_layer.ple_embedding.ngram_embedding)
manifest["components"]["ple_layer"] = {
    "out": {"shape": [1, 10, cfg.hc_count * cfg.hidden_size], "dtype": "float32"},
    "tolerance": {"abs": 1e-2, "rel": 1e-2},
    "law": "embeds=gather(ngram_ids); key=key_proj→grouped RMSNorm(group=hidden,(1+w)); value=value_proj; q_n=grouped RMSNorm(hidden); gate=bf16(fp32 Σ_hidden bf16(k·q))·(1/√hidden)→bf16, then symmetric sqrt law bf16(sqrt(max(|g|,1e-6)))·sign(g); gated=bf16(sigmoid(gate)·value); norm_conv on flat; short conv = causal pad-left (k−1)·ngram, depthwise dilation=ngram, silu(bf16); out = bf16(gated + conv)",
}

# ---------------- FULL DECODER LAYER (plan §4 gate: layer-by-layer) ----------
# Instantiate the reference Qwen4ExpTextDecoderLayer ITSELF (MiniCfg layer 1 =
# full_attention + PLE, exactly FN-shaped: ple_layer_ids=[2] is one-indexed →
# layer_idx 1). Every stage activation from the reference's own ops; the Swift
# composition must reproduce the stage chain under {abs 1e-2, rel 1e-2}.
import torch.nn as nn
layer = to_bf16(M.Qwen4ExpTextDecoderLayer(cfg, layer_idx=1))
# STRUCTURE LAW (plan §4): the decoder layer stream is the FOUR-LANE tensor
# (b, s, hc_count·hidden) = width 256. hc.read compresses lanes → mixed width
# 64 for attn/mlp; write-back re-expands to 256. PLE adds into the LANED stream
# (its norm_query has group_size=hidden over width hc·hidden — proven by init).
lay_x = det((1, 10, cfg.hc_count * cfg.hidden_size))       # laned layer input
lay_ids = torch.tensor([[5, 77, 300, 248044, 9, 250, 31, 4, 88, 12]], dtype=torch.int64)
causal_10 = torch.tril(torch.ones(10, 10, dtype=torch.bool))
lay_mask = torch.where(causal_10, torch.tensor(0.0), torch.tensor(NEG)).view(1, 1, 10, 10).to(torch.float32)
# tables built BY THE REFERENCE ROTARY MODULE for positions 0..9 (the shared
# 24-row table would feed the wrong positions to a 10-row layer). rotary() is
# the module instance created in the indexer section; its 3-row mrope stack is
# identical rows for pure text (recomposition is a no-op there).
lay_pos10 = torch.arange(10).view(1, 1, -1).expand(3, 1, -1)
lay_cos, lay_sin = rotary(lay_x, lay_pos10)          # each [1, 10, rot] bf16
with torch.no_grad():
    stage = {}
    hs = lay_x                                                    # (1,10,256) laned
    # embedding lookup for the composition test (hash law input, Swift side
    # recomputes row ids via QWFnPLE and gathers — exact contract)
    lay_embeds = layer.ple.ple_embedding(lay_ids, None)           # (1,10,ple_embed_dim)
    save("lay_ple_embeds", lay_embeds.float() if lay_embeds.dtype != torch.bfloat16 else lay_embeds)
    # ngram row ids for the Swift hash-mirror exact check (same law as the
    # standalone ple section, but layer.ple has its OWN prime tables/seed chain
    # — never mix instances)
    li = lay_ids.long()
    lctx = torch.full((1, layer.ple.ple_embedding.context_len), cfg.eos_token_id, dtype=torch.int64)
    lhist = torch.cat([lctx, li], dim=-1)
    lsh = [layer.ple.ple_embedding._shift_right_ignore_eos(lhist, s) for s in range(cfg.ngram_size)]
    lblocks = []
    for ng in range(2, cfg.ngram_size + 1):
        mixed = lsh[0] * layer.ple.ple_embedding.layer_multipliers[0]
        for pos in range(1, ng):
            mixed = torch.bitwise_xor(mixed, lsh[pos] * layer.ple.ple_embedding.layer_multipliers[pos])
        s0 = (ng - 2) * cfg.heads_per_ngram
        lblocks.append(torch.remainder(mixed.unsqueeze(-1),
                        layer.ple.ple_embedding.ngram_heads_vocab_sizes[s0:s0+cfg.heads_per_ngram].view(1,1,-1))
                       + layer.ple.ple_embedding.ngram_heads_offsets[s0:s0+cfg.heads_per_ngram].view(1,1,-1))
    lay_ngram_ids = torch.cat(lblocks, dim=-1)[:, -li.shape[1]:]
    # cross-check: ids gather from the table must equal the embedding output
    tbl = layer.ple.ple_embedding.ngram_embedding.weight
    assert torch.equal(tbl[lay_ngram_ids], lay_embeds.view(*lay_ngram_ids.shape, -1)), \
        "lay embeds gather mismatch — ids law wrong"
    save("lay_ngram_ids", lay_ngram_ids.to(torch.int64))
    # stage 0: PLE add (present at layer_idx 1; ple_layer_index = 0)
    ple_delta = layer.ple(hs, lay_ids, None, conv_mask=None)       # (1,10,256)
    stage["ple_add"] = (hs + ple_delta).to(torch.bfloat16)
    hs = stage["ple_add"]
    # stage 1: attn hyper-connection read (lanes → mixed 64)
    mixed, hyper_input, inj_w = layer.attn_hyper_connection(hs)    # mixed (1,10,64)
    stage["hc1_mixed"] = mixed
    # stage 2: self-attention — call with the RAW causal mask; the layer's own
    # forward combines (indexer ∘ causal) INTERNALLY, so the staged replay must
    # pass the raw mask exactly like the reference does (double-combining would
    # add the indexer mask twice).
    sel_mask_f = layer.self_attn.indexer(mixed, (lay_cos, lay_sin), lay_mask, None)
    save("lay_sel_mask", sel_mask_f.view(10, 10).to(torch.int64))
    # raw selected-token rows (block-major, tail appended, prefix-capped at
    # budget+ratio−1, −1 padding) — the Swift select() contract verbatim
    # Raw selected-token rows (block-major, tail appended, prefix-capped at
    # budget+ratio−1, −1 padding) — the Swift select() contract verbatim. The
    # reference builds only the mask internally; this replay recomputes the
    # rows with the SAME law the Swift mirror implements and asserts their
    # scatter reproduces the reference mask (scatter −1 → dropped column).
    # Verbatim copy of the reference QSAIndexer.forward loop (transformers
    # 5.17.0, modeling_qwen4_exp.py:699-778) — the ONLY change is that it also
    # keeps selected_token_indices so the Swift select() contract is pinned.
    idx = layer.self_attn.indexer
    bs, sl = 1, 10
    hidden_shape = (bs, sl, -1, idx.index_head_dim)
    full_cos, full_sin = lay_cos[None] if lay_cos.dim() == 2 else (lay_cos, lay_sin)
    current_cos, current_sin = full_cos[:, -sl:, :], full_sin[:, -sl:, :]
    qk = idx.index_qk_proj(mixed)
    q_t, token_k = torch.split(qk, [idx.index_n_heads * idx.index_head_dim,
                                    idx.index_kv_heads * idx.index_head_dim], dim=-1)
    q_t, raw_keys = q_t.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q_t = idx.q_layernorm(q_t)
    q_t = M.apply_rotary_pos_emb(q_t, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    visible = (lay_mask == 0)
    sel_rows = torch.full((bs, sl, idx.token_budget + idx.compress_ratio - 1), -1,
                          dtype=torch.int32)
    for query_idx in range(sl):
        loc = torch.nonzero(visible[0, 0, query_idx], as_tuple=False).flatten()
        ncb = loc.shape[-1] // idx.compress_ratio
        if ncb > 0:
            bti = loc[: ncb * idx.compress_ratio].view(ncb, idx.compress_ratio)
            kg = raw_keys[0].index_select(0, bti.flatten())
            kg = kg.view(*bti.shape, idx.index_head_dim)
            pk = kg.float().mean(dim=1).to(raw_keys.dtype)
            pk = idx.k_layernorm(pk)
            gs = bti[:, 0]
            bks = M.apply_rotary_pos_emb(pk.unsqueeze(1),
                                         cos=full_cos[0].index_select(0, gs),
                                         sin=full_sin[0].index_select(0, gs)).squeeze(1)
            sc = torch.matmul(q_t[0, query_idx].float(), bks.float().transpose(-1, -2)) \
                 .transpose(-1, -2)
            sc = torch.relu(sc).sum(dim=-1) / math.sqrt(idx.index_head_dim)
            # reference quirk kept verbatim: topk(dim=0) transposes indices
            sbi = sc.topk(min(idx.block_topk, ncb), dim=0).indices
            sbi = sbi.flatten()
            sel_tokens = bti.index_select(0, sbi).flatten()
        else:
            sel_tokens = torch.tensor([], dtype=torch.int32)
        tail = loc[ncb * idx.compress_ratio:]
        sel_tokens = torch.cat([sel_tokens, tail]).to(torch.int32)
        sel_rows[0, query_idx, : sel_tokens.numel()] = sel_tokens
    sel_rows = sel_rows[0]
    sc_pos = torch.where(sel_rows >= 0, sel_rows, torch.tensor(10))
    boolm = torch.zeros(10, 11, dtype=torch.bool).scatter(-1, sc_pos, True)[..., :10]
    # POLARITY: lay_sel_mask holds the reference's FLOAT additive mask — 0.0
    # marks an ALLOWED token, -finfo.min a blocked one (int64 dump clamps the
    # negative to INT64_MIN; the Swift test reads allowed = (v == 0)). The mask
    # bool is therefore `== 0`, never `.bool()` of the float.
    allowed_ref = sel_mask_f.view(10, 10) == 0
    assert torch.equal(boolm, allowed_ref), "lay selected-rows law != reference mask"
    save("lay_selected", sel_rows.to(torch.int64))
    save("lay_combined_mask", (lay_mask.view(10, 10) + sel_mask_f.view(10, 10).float()).float())
    attn_out, _ = layer.self_attn(mixed, (lay_cos, lay_sin), lay_mask, None)   # (1,10,64)
    stage["attn_out"] = attn_out
    # stage 3: write-back to the laned stream
    injection = attn_out.unsqueeze(-2) * inj_w.unsqueeze(-1)
    stage["hc1_out"] = (hyper_input + injection.flatten(-2)).to(torch.bfloat16)
    hs = stage["hc1_out"]
    # stage 4: mlp hyper-connection read
    mixed2, hyper_input2, inj_w2 = layer.mlp_hyper_connection(hs)
    stage["hc2_mixed"] = mixed2
    # stage 5: MoE — the block returns ONE tensor; routing comes from its gate
    # submodule (returns router_logits, routing_weights, selected_experts).
    _, routing_w, sel_e = layer.mlp.gate(mixed2.reshape(-1, cfg.hidden_size))
    moe_out = layer.mlp(mixed2)
    stage["moe_out"] = moe_out
    save("lay_routing_weights", routing_w.float())
    save("lay_selected_experts", sel_e.to(torch.int64))
    # stage 6: final write-back = layer output
    injection2 = moe_out.unsqueeze(-2) * inj_w2.unsqueeze(-1)
    stage["layer_out"] = (hyper_input2 + injection2.flatten(-2)).to(torch.bfloat16)
    # cross-check against the layer's own forward (plain tensor return)
    full_layer_out = layer(lay_x, (lay_cos, lay_sin), attention_mask=lay_mask,
                           past_key_values=None, ple_input_ids=lay_ids)
    assert isinstance(full_layer_out, torch.Tensor), "unexpected layer forward return type"
save("lay_input", lay_x)
save("lay_ids", lay_ids)
save("lay_mask", lay_mask)
save("lay_cos", lay_cos.float()); save("lay_sin", lay_sin.float())
for name in ["ple_delta", "ple_add", "hc1_mixed", "attn_out", "hc1_out", "hc2_mixed", "moe_out", "layer_out"]:
    if name == "ple_delta":
        save("lay_" + name, ple_delta.float()); continue
    save("lay_" + name, stage[name].float())
save("lay_full_forward", full_layer_out.float())
maxfd = (full_layer_out.float() - stage["layer_out"].float()).abs().max().item()
print("  full-layer forward vs staged composition maxdiff:", maxfd)
assert maxfd < 1e-1, "staged composition diverged from reference layer forward"
dump_params("lay_hc1", layer.attn_hyper_connection); dump_params("lay_hc2", layer.mlp_hyper_connection)
dump_params("lay_sa_q", layer.self_attn.q_proj); dump_params("lay_sa_qn", layer.self_attn.q_norm)
dump_params("lay_sa_k", layer.self_attn.k_proj); dump_params("lay_sa_kn", layer.self_attn.k_norm)
dump_params("lay_sa_v", layer.self_attn.v_proj); dump_params("lay_sa_o", layer.self_attn.o_proj)
dump_params("lay_mlp", layer.mlp)
dump_params("lay_ple", layer.ple)
dump_params("lay_idx", layer.self_attn.indexer)
manifest["components"]["layer"] = {
    "layer_out": {"shape": [1, 10, cfg.hidden_size], "dtype": "float32"},
    "tolerance": {"abs": 1e-2, "rel": 1e-2},
    "law": "hs+=ple(hs,ids) → hc1.read → self_attn(indexer-combined mask) → writeback(bf16(out·w)+hyper) → hc2.read → mlp → writeback; each stage pinned",
}

# ---------------- PLE head ids (already byte-verified) ----------------------
tok_path = "Scripts/oracle/flashnext_ple_hash_golden.json"
tok = json.loads(Path(tok_path).read_text())
save("ple_tokens", torch.tensor(tok["tokens"], dtype=torch.int64))
save("ple_head_ids", torch.tensor([v["head_row_ids"] for v in tok["head_ids_by_token"]], dtype=torch.int64))
manifest["components"]["ple_hash"] = {"note": "head_row_ids from derive_ple_hash_replica.py (multipliers byte-verified vs checkpoint)",
                                      "tolerance": {"exact": True}}

# ---------------- PLE collision + missing-context (plan §4: hash-collision
# and missing-context tests). Pinned collision found by exhaustive scan of the
# MiniCfg ngram-2 law (mults/divisors are deterministic): pairA=(1,68) and
# pairB=(2,22) satisfy mixed(pairA) ≡ mixed(pairB) (mod 17) — head 0 lands the
# SAME table row for two disjoint token pairs; the other ngram-2 heads do NOT
# collide, pinning that the collision is the mathematical congruence, not a
# bug. Contract: the replica must reproduce the collision identically.
emb2 = layer.ple.ple_embedding
mults = [int(m) for m in emb2.layer_multipliers]
def wrap64(v): return ((v + 2**63) % 2**64) - 2**63        # torch int64 overflow
def mixed_pair(prev, cur): return wrap64(wrap64(cur * mults[0]) ^ wrap64(prev * mults[1]))
PAIR_A, PAIR_B = (1, 68), (2, 22)
assert mixed_pair(*PAIR_A) % 17 == mixed_pair(*PAIR_B) % 17, "pinned collision moved"
assert mixed_pair(*PAIR_A) % 19 != mixed_pair(*PAIR_B) % 19, "head1 unexpectedly collides"

def rows_of(emb_m, seq):
    # Pure-Python big-int replay of the hash law: products may exceed int64
    # (248044·multiplier wraps in torch), so wrap to SIGNED int64 before the
    # modulo, then Python % (Euclidean for positive divisors) == torch.
    # remainder. XOR on big ints matches torch's wrapped XOR in the low 64
    # bits; torch tensors + Python 2**63 raise OverflowError — never route
    # this law through tensor arithmetic.
    hist = torch.cat([torch.full((1, 2), cfg.eos_token_id, dtype=torch.int64), seq], dim=-1)
    sh = [[int(x) for x in emb_m._shift_right_ignore_eos(hist, s)[0].tolist()]
          for s in range(cfg.ngram_size)]
    mu = [int(v) for v in emb_m.layer_multipliers]
    vs = [int(v) for v in emb_m.ngram_heads_vocab_sizes]
    ofs = [int(v) for v in emb_m.ngram_heads_offsets]
    out = []
    for i in range(seq.shape[1]):
        ids_ = []
        for ni, ng in enumerate((2, 3)):
            m = sh[0][2 + i] * mu[0]
            for pos in range(1, ng):
                m ^= sh[pos][2 + i] * mu[pos]
            m = ((m + 2**63) % 2**64) - 2**63      # torch int64 two's-complement wrap
            for h in range(cfg.heads_per_ngram):
                gi = ni * cfg.heads_per_ngram + h
                ids_.append(m % vs[gi] + ofs[gi])  # Python % = torch.remainder (v>0)
        out.append(ids_)
    return torch.tensor(out, dtype=torch.int64)

with torch.no_grad():
    seqA = torch.tensor([list(PAIR_A)], dtype=torch.int64)
    seqB = torch.tensor([list(PAIR_B)], dtype=torch.int64)
    rA, rB = rows_of(emb2, seqA), rows_of(emb2, seqB)
    eA = emb2(seqA, None).view(2, 64).to(torch.bfloat16)
    eB = emb2(seqB, None).view(2, 64).to(torch.bfloat16)
    tbl2 = emb2.ngram_embedding.weight
    # rows_of is a big-int PARAPHRASE of the reference loop — pin it the same
    # way as lay_ngram_ids: gathering the table by the replayed rows must
    # reproduce the embedding module's own output for EVERY head/token.
    assert torch.equal(tbl2[rA], eA.view(2, 16, 4)), "rows_of replay != embedding output (A)"
    assert torch.equal(tbl2[rB], eB.view(2, 16, 4)), "rows_of replay != embedding output (B)"
    col_heads = [h for h in range(8) if torch.equal(tbl2[rA[1, h]], tbl2[rB[1, h]])]
    assert col_heads == [0], f"collision heads changed: {col_heads}"
    # ngram-3 heads: with 47..79-sized divisors a chance row match between two
    # random trigrams is expected law (17/79 per head) — compute the truth and
    # PIN it rather than assert a guess; the Swift test replays equality per
    # head from the fixture, so whichever side is true is the contract.
    ngram3_equal = [h for h in range(8) if torch.equal(tbl2[rA[1, 8 + h]], tbl2[rB[1, 8 + h]])]
    save("ple_col_rowsA", rA.reshape(-1)); save("ple_col_rowsB", rB.reshape(-1))
    save("ple_col_embedsA", eA); save("ple_col_embedsB", eB)
    save("ple_col_seqA", seqA.reshape(-1)); save("ple_col_seqB", seqB.reshape(-1))
    save("ple_col_heads", torch.tensor(col_heads, dtype=torch.int64))
    save("ple_col_ng3_equal", torch.tensor(ngram3_equal, dtype=torch.int64))
    # missing-context / segment law: in [7, eos, 9], token 9 must hash with
    # FRESH context (== a standalone [9] run) — the eos resets the segment.
    seqM = torch.tensor([[7, cfg.eos_token_id, 9]], dtype=torch.int64)
    seqF = torch.tensor([[9]], dtype=torch.int64)
    rM = rows_of(emb2, seqM)
    rF = rows_of(emb2, seqF)
    assert torch.equal(rM[2], rF[0]), "segment restart law broken"
    eM = emb2(seqM, None).view(3, 64).to(torch.bfloat16)
    save("ple_seg_ids", seqM.reshape(-1)); save("ple_seg_rows", rM.reshape(-1))
    save("ple_seg_embeds", eM)
    save("ple_fresh_ids", seqF.reshape(-1)); save("ple_fresh_rows", rF.reshape(-1))
    save("ple_fresh_embeds", eM[2:3].contiguous())
manifest["components"]["ple_collision"] = {
    "tolerance": {"exact": True},
    "law": "pinned ngram-2 congruence pairA(1,68) pairB(2,22): ngram2 head0 congruent mod 17 → identical table row and embedding vector (heads 1-7 non-congruent, replayed per-head); ngram-3 head equality per ple_col_ng3_equal (row-identity pinned by fixture); token after eos hashes as fresh segment (eos-pad prefix == standalone run); all head rows and gathered embeds exact"}

(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"fixtures written to {ROOT}")
for c, m in manifest["components"].items():
    print(f"  {c:18} {m.get('tolerance', {})}")
print(f"  gdn chunk-vs-recurrent maxabs {parity:.3e}")
