#!/usr/bin/env python3
"""Real-length golden for FlashNext full attention (layer 3) with the PRODUCTION indexer budget (2048).

Sequence length > 2048 so the top-512 block cutoff is active. HF scoring formulas, computed once for all queries
(q and block keys after their RMSNorm and RoPE, scores = sum_h relu(q_h . k_b) / sqrt(d)). Block selection uses
the MLX/oMLX contract (last k of a stable ascending sort by (score, index)). Everything else is HF Qwen4ExpTextAttention.

Output (outside the repo): $FN_LONG_DIR or ~/Developer/ai/qwegine.oracle-fn/goldens/flashnext_attn_long
  meta.json, in.f32 (T x hidden), out.f32 (T x hidden), sel.u8 (T x T), 
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_real_layer0 as base  # noqa: E402

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

LAYER = 3
T = int(os.environ.get("FN_LONG_T", "2100"))
SEED = 20261025
OUT = os.environ.get("FN_LONG_DIR", os.path.expanduser(
    "~/Developer/ai/qwegine.oracle-fn/goldens/flashnext_attn_long"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def selection_rows(scores, k, complete_counts):
    """MLX contract per query row: last k of stable ascending sort by (score, index) among the complete blocks."""
    T_, nb = scores.shape
    chosen = []
    ties = 0
    for t in range(T_):
        c = complete_counts[t]
        if c == 0:
            chosen.append([])
            continue
        row = scores[t, :c]
        order = sorted(range(c), key=lambda b: (-row[b], -b))
        kk = min(k, c)
        if c > kk and row[order[kk - 1]] == row[order[kk]]:
            ties += 1
        chosen.append(order[:kk])
    return chosen, ties


def main():
    os.makedirs(OUT, exist_ok=True)
    pack_path = os.path.join(base.PACK_DIR, "model.qwp")
    hdr = base.load_header(pack_path)
    assets = json.load(open(os.path.join(base.PACK_DIR, "assets", "config.json")))
    cfg = Qwen4ExpTextConfig.from_dict(assets.get("text_config", assets))
    cfg._attn_implementation = "eager"
    r = cfg.indexer_compress_ratio
    k = cfg.indexer_budget // r
    torch.manual_seed(SEED)
    attn = M.Qwen4ExpTextAttention(cfg, layer_idx=LAYER).eval()
    prefix = f"{base.PREFIX}layers.{LAYER}.self_attn."
    state = {}
    with open(pack_path, "rb") as f:
        for name in attn.state_dict().keys():
            state[name] = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, prefix + name)))
    attn.load_state_dict(state, strict=True)
    del state
    log(f"attention layer {LAYER} loaded; T={T} budget={cfg.indexer_budget} blockTopK={k}")

    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, T, cfg.hidden_size, generator=gen) * 0.5
    rotary = M.Qwen4ExpTextRotaryEmbedding(cfg)
    pos = torch.arange(T).view(1, T)
    cos, sin = rotary(x, pos.unsqueeze(0).expand(3, 1, T))
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    mask = torch.zeros(1, 1, T, T).masked_fill(~causal, torch.finfo(torch.float32).min)

    # Indexer tensors, computed once with the HF modules (same math as Qwen4ExpTextQSAIndexer.forward).
    ix = attn.indexer
    with torch.no_grad():
        H = cfg.indexer_n_heads * cfg.indexer_head_dim
        xs = x[0]
        qk = ix.index_qk_proj(xs)
        q = ix.q_layernorm(qk[:, :H].reshape(T, cfg.indexer_n_heads, cfg.indexer_head_dim))
        q = M.apply_rotary_pos_emb(q, cos=cos[0], sin=sin[0], unsqueeze_dim=1)           # T x heads x d
        nb = T // r
        raw = qk[:, H:]
        pooled = raw[: nb * r].reshape(nb, r, cfg.indexer_head_dim).mean(1)
        pooled = ix.k_layernorm(pooled)
        starts = torch.arange(nb) * r
        pooled = M.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0][starts], sin=sin[0][starts]).squeeze(1)
        # scores[t, b] = sum_h relu(q[t,h] . k[b]) / sqrt(d)
        dots = torch.einsum("thd,bd->tbh", q, pooled)
        scores = torch.relu(dots).sum(-1) / float(cfg.indexer_head_dim) ** 0.5
    complete = [(t + 1) // r for t in range(T)]
    chosen, ties = selection_rows(scores.double().numpy(), k, complete)
    sel = np.zeros((T, T), dtype=np.uint8)
    for t in range(T):
        for b in chosen[t]:
            sel[t, b * r:(b + 1) * r] = 1
        sel[t, complete[t] * r:t + 1] = 1          # causal tail stays visible
    log(f"selection built: ties at cutoff = {ties} queries; last query keeps {int(sel[-1].sum())} of {T} keys")

    # HF attention with the selection injected as the indexer's float mask (0 = selected, min = dropped).
    class Fixed(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
            return torch.zeros(1, 1, T, T).masked_fill(torch.from_numpy(sel == 0), torch.finfo(torch.float32).min)

    attn.indexer = Fixed()
    with torch.no_grad():
        y, _ = attn(x, (cos, sin), mask, None)
    x[0].numpy().astype("<f4").tofile(os.path.join(OUT, "in.f32"))
    y[0].numpy().astype("<f4").tofile(os.path.join(OUT, "out.f32"))
    sel.tofile(os.path.join(OUT, "sel.u8"))
    meta = {"layer": LAYER, "tokens": T, "hidden": cfg.hidden_size, "budget": cfg.indexer_budget,
            "compress": r, "seed": SEED, "ties_at_cutoff_queries": ties,
            "selected_last_query": int(sel[-1].sum()), "out_abs_mean": float(y.abs().mean())}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    log(f"wrote {meta}")


if __name__ == "__main__":
    main()
