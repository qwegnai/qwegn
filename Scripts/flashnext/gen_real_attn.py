#!/usr/bin/env python3
"""Real-weight golden for FlashNext full attention (layer 3): HF Qwen4ExpTextAttention on pack-dequantized
weights at real dims (24 heads, head_dim 256, partial RoPE 64, GQA 2, QSA indexer 4 heads x 128).

Indexer budget is overridden to 16 (blockTopK = 4) so the top-k selection is active on a 40-token sequence
(the real budget 2048 needs > 2048 tokens). Block selection uses the MLX/oMLX contract (qsa_fast.py): the last
k of a stable ascending sort by (score, index), i.e. equal scores prefer the HIGHER block index. Scores use the
HF formulas (relu, sum over heads, / sqrt(head_dim)). Everything else is HF.

Writes Tests/QWFlashNextTests/fixtures/fn_attn3_{in,out}.f32, fn_attn3_sel.u8, fn_attn3_meta.json.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_real_layer0 as base  # noqa: E402  (pack reader + dequant law)
import gen_goldens as gg  # noqa: E402  (indexer_block_scores mirrors HF scoring)

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

LAYER = 3
T = 40
BUDGET = 16
SEED = 20261022
FIXTURES = base.FIXTURES


def mlx_block_selection(scores, k):
    """Indices of the k largest scores; equal scores prefer the higher index (MLX argpartition [-k:])."""
    order = sorted(range(len(scores)), key=lambda b: (-scores[b], -b))
    return order[:k]


class MLXSelectingIndexer(torch.nn.Module):
    """Stands in for Qwen4ExpTextQSAIndexer: same output contract (float mask, 0 = selected, min = dropped)."""

    def __init__(self, indexer, cfg):
        super().__init__()
        self.ix = indexer
        self.cfg = cfg
        self.ties_at_cutoff = 0

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        cfg = self.cfg
        cos, sin = position_embeddings
        x = hidden_states[0]
        T_ = x.shape[0]
        r = cfg.indexer_compress_ratio
        k = cfg.indexer_budget // r
        sel = torch.zeros(T_, T_, dtype=torch.bool)
        with torch.no_grad():
            for t in range(T_):
                complete = (t + 1) // r
                if complete > 0:
                    scores = gg.indexer_block_scores(self.ix, cfg, x, cos[0], sin[0], t).tolist()
                    chosen = mlx_block_selection(scores, min(k, complete))
                    if complete > k and sorted(scores, reverse=True)[k - 1] == sorted(scores, reverse=True)[k]:
                        self.ties_at_cutoff += 1
                    for b in chosen:
                        sel[t, b * r:(b + 1) * r] = True
                sel[t, complete * r:t + 1] = True  # causal tail stays visible
        out = torch.zeros(1, 1, T_, T_).masked_fill(~sel, torch.finfo(torch.float32).min)
        return out


def main():
    pack_path = os.path.join(base.PACK_DIR, "model.qwp")
    hdr = base.load_header(pack_path)
    assets = base.json.load(open(os.path.join(base.PACK_DIR, "assets", "config.json")))
    tc = assets.get("text_config", assets)
    cfg = Qwen4ExpTextConfig.from_dict(tc)
    cfg._attn_implementation = "eager"
    cfg.indexer_budget = BUDGET
    torch.manual_seed(SEED)
    attn = M.Qwen4ExpTextAttention(cfg, layer_idx=LAYER).eval()
    prefix = f"{base.PREFIX}layers.{LAYER}.self_attn."
    state = {}
    with open(pack_path, "rb") as f:
        for k in attn.state_dict().keys():
            state[k] = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, prefix + k)))
    attn.load_state_dict(state, strict=True)
    rotary = M.Qwen4ExpTextRotaryEmbedding(cfg)
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, T, cfg.hidden_size, generator=gen) * 0.5
    pos = torch.arange(T).view(1, T)
    cos, sin = rotary(x, pos.unsqueeze(0).expand(3, 1, T))
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    mask = torch.zeros(1, 1, T, T).masked_fill(~causal, torch.finfo(torch.float32).min)
    selector = MLXSelectingIndexer(attn.indexer, cfg)
    attn.indexer = selector
    with torch.no_grad():
        y, _ = attn(x, (cos, sin), mask, None)
    # Recompute the selection mask the attention actually used, for the Swift selection check.
    selector.ties_at_cutoff = 0
    with torch.no_grad():
        sel_mask = selector(x, (cos, sin), mask, None)
    sel = (sel_mask[0, 0] == 0).to(torch.uint8).numpy()
    os.makedirs(FIXTURES, exist_ok=True)
    x[0].numpy().astype("<f4").tofile(os.path.join(FIXTURES, "fn_attn3_in.f32"))
    y[0].numpy().astype("<f4").tofile(os.path.join(FIXTURES, "fn_attn3_out.f32"))
    sel.tofile(os.path.join(FIXTURES, "fn_attn3_sel.u8"))
    meta = {"layer": LAYER, "tokens": T, "hidden": cfg.hidden_size, "budget": BUDGET,
            "compress": cfg.indexer_compress_ratio, "seed": SEED,
            "selected_per_last_query": int(sel[-1].sum()), "causal_per_last_query": T,
            "out_abs_mean": float(y.abs().mean()), "ties_at_cutoff_queries": selector.ties_at_cutoff}
    base.json.dump(meta, open(os.path.join(FIXTURES, "fn_attn3_meta.json"), "w"), indent=1)
    print("wrote", meta)


if __name__ == "__main__":
    main()
