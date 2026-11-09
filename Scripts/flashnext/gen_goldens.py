#!/usr/bin/env python3
"""Golden generator for the FlashNext (qwen4_exp_text) Swift CPU reference.

Source of truth: HF transformers `models/qwen4_exp/modeling_qwen4_exp.py` (pinned in
specs/30-flashnext-architecture.md §0). Runs in the dedicated golden venv
(`~/Developer/ai/qwegine.oracle-fn`, torch 2.14.1, transformers 5.17.0, numpy 2.4.3).

Modules with small random weights (fixed seed):
  - Qwen4ExpTextRMSNorm (group)           -> rmsnorm
  - Qwen4ExpTextGatedResidual (combine)   -> hc_gated_residual
  - Qwen4ExpTextSparseMoeBlock            -> sparse_moe
Exact integer module at the REAL FlashNext hash constants:
  - N-gram ids (Qwen4ExpTextNGramEmbedding.forward mixing) -> ngram_ids

Output: Tests/QWFlashNextTests/fixtures/fn_goldens.json (flat float lists + shapes).
"""
import json
import math
import os
import sys

import torch

from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "Tests", "QWFlashNextTests", "fixtures", "fn_goldens.json")
SEED = 20261020


def flat(t):
    return {"shape": list(t.shape), "data": t.detach().to(torch.float32).reshape(-1).tolist()}


def export_state(module, prefix):
    return {f"{prefix}{k}": flat(v) for k, v in module.state_dict().items()}


def small_config():
    return Qwen4ExpTextConfig(
        hidden_size=16, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=12,
        shared_expert_intermediate_size=12, hidden_act="silu", norm_topk_prob=True,
    )


def gen_rmsnorm(out):
    torch.manual_seed(SEED)
    cfg = small_config()
    H, hc = cfg.hidden_size, cfg.hc_count
    norm = M.Qwen4ExpTextRMSNorm(hc * H, group_size=H, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.5 + 1.0)
    x = torch.randn(2, 3, hc * H)
    y = norm(x)
    out["rmsnorm"] = {"eps": cfg.rms_norm_eps, "group": H, "weights": export_state(norm, ""),
                      "input": flat(x), "output": flat(y)}


def gen_hc(out):
    torch.manual_seed(SEED + 1)
    cfg = small_config()
    hc_mod = M.Qwen4ExpTextGatedResidual(cfg, use_combine=True)
    x = torch.randn(2, 3, cfg.hc_count * cfg.hidden_size)
    with torch.no_grad():
        for p in hc_mod.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    mixed, x_in, inj = hc_mod(x)
    out["hc_gated_residual"] = {"hc": cfg.hc_count, "hidden": cfg.hidden_size, "lowrank": cfg.hc_lowrank,
                                "eps": cfg.rms_norm_eps, "weights": export_state(hc_mod, ""),
                                "input": flat(x), "mixed": flat(mixed), "injection": flat(inj)}


def gen_moe(out):
    torch.manual_seed(SEED + 2)
    cfg = small_config()
    moe = M.Qwen4ExpTextSparseMoeBlock(cfg)
    with torch.no_grad():
        for p in moe.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    x = torch.randn(2, 3, cfg.hidden_size)
    y = moe(x)
    out["sparse_moe"] = {"hidden": cfg.hidden_size, "experts": cfg.num_experts, "top_k": cfg.num_experts_per_tok,
                         "expert_intermediate": cfg.moe_intermediate_size,
                         "shared_intermediate": cfg.shared_expert_intermediate_size,
                         "norm_topk_prob": cfg.norm_topk_prob,
                         "weights": export_state(moe, ""), "input": flat(x), "output": flat(y)}


def gen_ngram(out):
    # Real FlashNext constants (config.json): vocab 248320, seed 1234, ngram_size 3, heads_per_ngram 8,
    # ngram_vocab_size_base 20000000, eos 248044, PLE layer index 0 (decoder layer 1, ple_layer_ids=[2]).
    vocab, seed, ngram, hpn, base, eos = 248320, 1234, 3, 8, 20000000, 248044
    ple_layer_index = 0
    ngram_heads = (ngram - 1) * hpn
    mult = M._build_layer_multipliers(vocab, ngram, ple_layer_index, seed).tolist()
    sizes, offsets, total = [], [], 0
    for head in range(ngram_heads):
        g = ple_layer_index * ngram_heads + head
        size = M._find_nth_prime_after(base - 1, g + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    # Deterministic token streams, including EOS-delimited segments and large ids.
    g = torch.Generator().manual_seed(SEED + 3)
    seqs = []
    for length in (7, 16):
        ids = torch.randint(0, vocab, (1, length), generator=g)
        ids[0, length // 3] = eos
        seqs.append(ids)

    class Stub:  # unbound helper needs only eos_token_id
        eos_token_id = eos

    results = []
    for ids in seqs:
        # Forward semantics with an empty cache: history = eos-initialised context ++ ids.
        context_len = ngram - 1
        history = torch.cat([torch.full((1, context_len), eos, dtype=torch.long), ids.long()], dim=-1)
        shifted = [M.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos(Stub, history, s) for s in range(ngram)]
        blocks = []
        for n in range(2, ngram + 1):
            start = (n - 2) * hpn
            end = start + hpn
            mixed = shifted[0] * mult[0]
            for pos in range(1, n):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * mult[pos])
            hv = torch.tensor(sizes[start:end], dtype=torch.long).view(1, 1, -1)
            ho = torch.tensor(offsets[start:end], dtype=torch.long).view(1, 1, -1)
            blocks.append(torch.remainder(mixed.unsqueeze(-1), hv) + ho)
        rows = torch.cat(blocks, dim=-1)[:, -ids.shape[1]:]
        results.append({"tokens": ids[0].tolist(), "ids": rows[0].tolist()})
    out["ngram_ids"] = {"vocab": vocab, "seed": seed, "ngram_size": ngram, "heads_per_ngram": hpn,
                        "ngram_vocab_size_base": base, "eos": eos, "ple_layer_index": ple_layer_index,
                        "multipliers": mult, "head_sizes": sizes, "head_offsets": offsets,
                        "total_rows": total, "cases": results}


def gdn_config():
    return Qwen4ExpTextConfig(
        hidden_size=16, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="sigmoid",
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, num_hidden_layers=4, layer_types=["linear_attention"] * 4,
    )


def gen_gdn(out):
    torch.manual_seed(SEED + 4)
    cfg = gdn_config()
    gdn = M.Qwen4ExpTextGatedDeltaNet(cfg, layer_idx=0)
    with torch.no_grad():
        for name, p in gdn.named_parameters():
            if name == "A_log":
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4.0)))
            else:
                p.copy_(torch.randn_like(p) * 0.2)
    x = torch.randn(2, 5, cfg.hidden_size)
    y = gdn(x)
    out["gated_delta_net"] = {"hidden": cfg.hidden_size, "k_heads": cfg.linear_num_key_heads,
                              "v_heads": cfg.linear_num_value_heads, "k_dim": cfg.linear_key_head_dim,
                              "v_dim": cfg.linear_value_head_dim, "conv_kernel": cfg.linear_conv_kernel_dim,
                              "eps": cfg.rms_norm_eps, "gate_activation": cfg.output_gate_type,
                              "weights": {k: flat(v) for k, v in gdn.named_parameters()},
                              "input": flat(x), "output": flat(y)}


def ple_config():
    return Qwen4ExpTextConfig(
        hidden_size=16, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6, hidden_act="silu",
        vocab_size=20, ngram_vocab_size_base=30, make_ngram_vocab_size_divisible_by=8, ngram_size=3,
        heads_per_ngram=2, seed=1234, eos_token_id=19, ple_embed_dim=8, ple_conv_kernel_size=4,
        ple_layer_ids=[2], num_hidden_layers=4, layer_types=["linear_attention"] * 4,
    )


def gen_ple(out):
    torch.manual_seed(SEED + 5)
    cfg = ple_config()
    ple = M.Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0)
    with torch.no_grad():
        for name, p in ple.named_parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    ids = torch.randint(0, cfg.vocab_size, (2, 6), generator=torch.Generator().manual_seed(SEED + 6))
    ids[0, 2] = cfg.eos_token_id
    h = torch.randn(2, 6, cfg.hc_count * cfg.hidden_size)
    emb = ple.ple_embedding(ids, None)
    y = ple(h, ids, None)
    table = ple.ple_embedding.ngram_embedding.weight
    out["ple_layer"] = {"hidden": cfg.hidden_size, "hc": cfg.hc_count, "ple_embed_dim": cfg.ple_embed_dim,
                        "conv_kernel": cfg.ple_conv_kernel_size, "dilation": cfg.ngram_size,
                        "eps": cfg.rms_norm_eps, "ngram_size": cfg.ngram_size, "heads_per_ngram": cfg.heads_per_ngram,
                        "vocab": cfg.vocab_size, "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
                        "seed": cfg.seed, "eos": cfg.eos_token_id, "ple_layer_index": 0,
                        "ngram_table": flat(table),
                        "weights": {k: flat(v) for k, v in ple.named_parameters() if not k.startswith("ple_embedding")},
                        "ids": ids.tolist(), "hidden_input": flat(h), "embedding": flat(emb), "output": flat(y)}


def attn_config():
    cfg = Qwen4ExpTextConfig(
        hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=16, attention_bias=False,
        attention_dropout=0.0, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="sigmoid",
        rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
        num_hidden_layers=4, layer_types=["full_attention"] * 4,
    )
    cfg._attn_implementation = "eager"
    return cfg


def indexer_block_scores(ix, cfg, x_b, cos_b, sin_b, t):
    """HF QSA block scores for query t of one sequence (mirrors Qwen4ExpTextQSAIndexer.forward)."""
    L = x_b.shape[0]
    H = cfg.indexer_n_heads * cfg.indexer_head_dim
    qk = ix.index_qk_proj(x_b)
    q = ix.q_layernorm(qk[:, :H].reshape(L, cfg.indexer_n_heads, cfg.indexer_head_dim))
    q = M.apply_rotary_pos_emb(q, cos=cos_b, sin=sin_b, unsqueeze_dim=1)
    nb = (t + 1) // cfg.indexer_compress_ratio
    r = cfg.indexer_compress_ratio
    pooled = torch.stack([qk[bi * r:(bi + 1) * r, H:].mean(0) for bi in range(nb)])
    pooled = ix.k_layernorm(pooled)
    starts = [bi * r for bi in range(nb)]
    pooled = M.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos_b[starts], sin=sin_b[starts]).squeeze(1)
    return torch.relu(q[t] @ pooled.T).sum(0) / math.sqrt(cfg.indexer_head_dim)


def attention_case(seed):
    torch.manual_seed(seed)
    cfg = attn_config()
    attn = M.Qwen4ExpTextAttention(cfg, layer_idx=3)
    rotary = M.Qwen4ExpTextRotaryEmbedding(cfg)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    B, L = 2, 20
    x = torch.randn(B, L, cfg.hidden_size)
    position_ids = torch.arange(L).view(1, L).expand(B, L)
    cos, sin = rotary(x, position_ids.unsqueeze(0).expand(3, B, L))
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    mask = torch.zeros(B, 1, L, L).masked_fill(~causal, torch.finfo(torch.float32).min)
    return cfg, attn, x, cos, sin, mask


def cutoff_is_tie_free(cfg, attn, x, cos, sin, margin=1e-4):
    """True when every query's top-k block cutoff is strictly separated (no exact or near ties)."""
    k = cfg.indexer_budget // cfg.indexer_compress_ratio
    with torch.no_grad():
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                if (t + 1) // cfg.indexer_compress_ratio <= k:
                    continue
                scores = indexer_block_scores(attn.indexer, cfg, x[b], cos[b], sin[b], t)
                v = torch.sort(scores, descending=True).values
                if (v[k - 1] - v[k]).abs().item() < margin:
                    return False
    return True


def gen_attention(out):
    # Seed search: the golden must be tie-free at the QSA block cutoff (see FNQSAIndexer tie notes).
    for attempt in range(64):
        cfg, attn, x, cos, sin, mask = attention_case(SEED + 7 + attempt)
        if cutoff_is_tie_free(cfg, attn, x, cos, sin):
            break
    else:
        raise SystemExit("no tie-free seed found")
    with torch.no_grad():
        selected = attn.indexer(x, (cos, sin), mask, None)
        out_attn, _ = attn(x, (cos, sin), mask, None)
    B, L = x.shape[0], x.shape[1]
    sel01 = (selected == 0).to(torch.int32)  # float mask: 0 = selected/visible, min = dropped
    out["attention"] = {
        "seed": SEED + 7 + attempt,
        "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim, "rotary_dim": int(cfg.head_dim * 0.25), "rope_theta": 10000000.0,
        "indexer": {"heads": cfg.indexer_n_heads, "kv_heads": cfg.indexer_kv_heads,
                    "head_dim": cfg.indexer_head_dim, "budget": cfg.indexer_budget,
                    "compress": cfg.indexer_compress_ratio, "eps": cfg.rms_norm_eps},
        "eps": cfg.rms_norm_eps,
        "weights": {k: flat(v) for k, v in attn.named_parameters()},
        "rope_cos": flat(cos[0]), "rope_sin": flat(sin[0]),
        "input": flat(x), "selected": sel01.reshape(B, L, L).tolist(),
        "output": flat(out_attn),
    }


def model_config():
    cfg = Qwen4ExpTextConfig(
        hidden_size=16, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="sigmoid",
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, attention_bias=False, attention_dropout=0.0,
        rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=12,
        shared_expert_intermediate_size=12, norm_topk_prob=True, vocab_size=20, eos_token_id=19,
        ngram_vocab_size_base=30, make_ngram_vocab_size_divisible_by=8, ngram_size=3, heads_per_ngram=2,
        seed=1234, ple_embed_dim=8, ple_conv_kernel_size=4, ple_layer_ids=[1], num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def model_case(seed):
    torch.manual_seed(seed)
    cfg = model_config()
    model = M.Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4.0)))
            else:
                p.copy_(torch.randn_like(p) * 0.2)
    ids = torch.randint(0, cfg.vocab_size, (2, 20), generator=torch.Generator().manual_seed(SEED + 20))
    ids[0, 4] = cfg.eos_token_id
    ids[1, 9] = cfg.eos_token_id
    return cfg, model, ids


def gen_model(out):
    for attempt in range(64):
        seed = SEED + 100 + attempt
        cfg, model, ids = model_case(seed)
        captured = []
        handles = []
        for i, layer in enumerate(model.model.layers):
            if cfg.layer_types[i] == "full_attention":
                handles.append(layer.self_attn.indexer.register_forward_pre_hook(
                    lambda mod, args, store=captured: store.append((mod, args[0].detach(), args[1]))))
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        for h in handles:
            h.remove()
        if all(cutoff_is_tie_free_pre(mod, cfg, x, pe) for mod, x, pe in captured):
            break
    else:
        raise SystemExit("no tie-free model seed found")
    streams = []
    hooks = [layer.register_forward_hook(lambda m, a, o: streams.append(o.detach()))
             for layer in model.model.layers]
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    for h in hooks:
        h.remove()
    ple_layer = [i for i, layer in enumerate(model.model.layers) if layer.ple is not None][0]
    weights = {k: flat(v) for k, v in model.named_parameters() if not k.endswith("ngram_embedding.weight")}
    out["model"] = {
        "seed": seed, "hidden": cfg.hidden_size, "hc": cfg.hc_count, "vocab": cfg.vocab_size,
        "eos": cfg.eos_token_id, "layer_types": cfg.layer_types, "ple_layer": ple_layer,
        "ngram": {"vocab": cfg.vocab_size, "ngram_size": cfg.ngram_size, "heads_per_ngram": cfg.heads_per_ngram,
                  "base": cfg.ngram_vocab_size_base, "seed": cfg.seed, "eos": cfg.eos_token_id,
                  "ple_layer_index": 0},
        "ple_table": flat(model.model.layers[ple_layer].ple.ple_embedding.ngram_embedding.weight),
        "weights": weights, "ids": ids.tolist(),
        "streams": [flat(st) for st in streams], "logits": flat(logits),
    }


def cutoff_is_tie_free_pre(mod, cfg, x, pe, margin=1e-4):
    cos, sin = pe
    k = cfg.indexer_budget // cfg.indexer_compress_ratio
    with torch.no_grad():
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                if (t + 1) // cfg.indexer_compress_ratio <= k:
                    continue
                scores = indexer_block_scores(mod, cfg, x[b], cos[b], sin[b], t)
                v = torch.sort(scores, descending=True).values
                if (v[k - 1] - v[k]).abs().item() < margin:
                    return False
    return True


def main():
    out = {"generator": "scripts/flashnext/gen_goldens.py", "torch": torch.__version__}
    gen_rmsnorm(out)
    gen_hc(out)
    gen_moe(out)
    gen_ngram(out)
    gen_gdn(out)
    gen_ple(out)
    gen_attention(out)
    gen_model(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(OUT), os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    sys.exit(main())
