#!/usr/bin/env python3
"""Real-dimension golden for the FlashNext PLE layer (decoder layer 1, ple_layer_index 0).

- PLE weights: real HQ4/bf16 pack tensors `layers.1.ple.*` (dequantized with the verified law).
- N-gram rows: read from the source safetensors shards via `assets/ple_ngram_locations.json` (byte offsets),
  after checking each shard's `base_off` against its safetensors header. The 320M-row table is never materialized.
- N-gram ids: HF helpers (`_build_layer_multipliers`, `_find_nth_prime_after`, `_shift_right_ignore_eos`) with the
  real constants; the router constants in the locations JSON are checked against them first.
- HF `Qwen4ExpTextPLELayer.forward` runs on the real weights; its `ple_embedding` is replaced by a module that
  returns the precomputed embedding (the real module would allocate the full table).

Writes Tests/QWFlashNextTests/fixtures/fn_ple1_{in,out}.f32, fn_ple1_ids.i32, fn_ple1_meta.json.
"""
import glob
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_real_layer0 as base  # noqa: E402

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

LAYER = 1
PLE_INDEX = 0
T = 12
SEED = 20261023
VOCAB, NGRAM, HPN, BASE, NSEED, EOS = 248320, 3, 8, 20000000, 1234, 248044
ROW_BYTES = 320
SNAPSHOT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/"))[0]
LOC = json.load(open(os.path.join(base.PACK_DIR, "assets", "ple_ngram_locations.json")))


class FakeEmbedding(torch.nn.Module):
    """Stands in for Qwen4ExpTextNGramEmbedding: returns the precomputed (1, T, ple_embed_dim) embedding."""

    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, input_ids, past_key_values=None):
        return self.emb


def check_shard_offsets():
    """Each shard's base_off must equal 8 + header_len + data_offsets[key][0] in its safetensors file."""
    seen = {}
    for sh in LOC["shards"]:
        if sh["file"] in seen:
            continue
        path = os.path.join(SNAPSHOT, sh["file"])
        with open(path, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            meta = json.loads(f.read(hl))
        seen[sh["file"]] = (hl, meta)
    for sh in LOC["shards"]:
        hl, meta = seen[sh["file"]]
        off = 8 + hl + meta[sh["key"]]["data_offsets"][0]
        assert off == sh["base_off"], (sh["key"], off, sh["base_off"])
    return len(LOC["shards"])


def read_row(row):
    for sh in LOC["shards"]:
        if sh["first_row"] <= row < sh["first_row"] + sh["shape"][0]:
            with open(os.path.join(SNAPSHOT, sh["file"]), "rb") as f:
                f.seek(sh["base_off"] + (row - sh["first_row"]) * ROW_BYTES)
                raw = f.read(ROW_BYTES)
            return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    raise IndexError(row)


def ngram_ids(tokens, multipliers, sizes, offsets):
    """HF Qwen4ExpTextNGramEmbedding forward ids for one fresh sequence (EOS-initialised context)."""

    class Stub:
        eos_token_id = EOS

    heads = (NGRAM - 1) * HPN
    history = torch.cat([torch.full((1, NGRAM - 1), EOS, dtype=torch.long), tokens.view(1, -1).long()], dim=-1)
    shifted = [M.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos(Stub, history, s) for s in range(NGRAM)]
    blocks = []
    for n in range(2, NGRAM + 1):
        start = (n - 2) * HPN
        mixed = shifted[0] * multipliers[0]
        for pos in range(1, n):
            mixed = torch.bitwise_xor(mixed, shifted[pos] * multipliers[pos])
        hv = torch.tensor(sizes[start:start + HPN], dtype=torch.long).view(1, 1, -1)
        ho = torch.tensor(offsets[start:start + HPN], dtype=torch.long).view(1, 1, -1)
        blocks.append(torch.remainder(mixed.unsqueeze(-1), hv) + ho)
    rows = torch.cat(blocks, dim=-1)[:, -tokens.numel():]
    assert rows.shape[-1] == heads
    return rows[0]


def main():
    n_shards = check_shard_offsets()
    mult = M._build_layer_multipliers(VOCAB, NGRAM, PLE_INDEX, NSEED).tolist()
    heads = (NGRAM - 1) * HPN
    sizes, offsets, total = [], [], 0
    for h in range(heads):
        s = M._find_nth_prime_after(BASE - 1, PLE_INDEX * heads + h + 1)
        sizes.append(s)
        offsets.append(total)
        total += s
    rc = LOC["router_constants"]["layers"][str(LAYER)]
    assert mult == rc["layer_multipliers"] and offsets == rc["ngram_heads_offsets"] \
        and sizes == rc["ngram_heads_vocab_sizes"], "router constants disagree with HF helpers"

    gen = torch.Generator().manual_seed(SEED)
    tokens = torch.randint(0, VOCAB, (T,), generator=gen)
    tokens[3] = EOS
    tokens[8] = EOS
    ids = ngram_ids(tokens, mult, sizes, offsets)
    emb_rows = []
    for t in range(T):
        emb_rows.append(np.concatenate([read_row(int(r)) for r in ids[t].tolist()]))
    emb = torch.from_numpy(np.stack(emb_rows).astype(np.float32)).unsqueeze(0)

    pack_path = os.path.join(base.PACK_DIR, "model.qwp")
    hdr = base.load_header(pack_path)
    assets = json.load(open(os.path.join(base.PACK_DIR, "assets", "config.json")))
    cfg = Qwen4ExpTextConfig.from_dict(assets.get("text_config", assets))
    cfg.ngram_vocab_size_base = 30          # small construction; the table is replaced below
    cfg.make_ngram_vocab_size_divisible_by = 8
    cfg._attn_implementation = "eager"
    torch.manual_seed(SEED)
    ple = M.Qwen4ExpTextPLELayer(cfg, layer_idx=LAYER, ple_layer_index=PLE_INDEX).eval()
    ple.ple_embedding = FakeEmbedding(emb)
    prefix = f"{base.PREFIX}layers.{LAYER}.ple."
    state = {}
    with open(pack_path, "rb") as f:
        for k in ple.state_dict().keys():
            if k.startswith("ple_embedding."):
                continue
            state[k] = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, prefix + k)))
    missing, unexpected = ple.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("ple_embedding.") for k in missing), (missing, unexpected)

    x = torch.randn(1, T, cfg.hc_count * cfg.hidden_size, generator=gen) * 0.5
    with torch.no_grad():
        y = ple(x, tokens.view(1, T), None)
    os.makedirs(base.FIXTURES, exist_ok=True)
    x[0].numpy().astype("<f4").tofile(os.path.join(base.FIXTURES, "fn_ple1_in.f32"))
    y[0].numpy().astype("<f4").tofile(os.path.join(base.FIXTURES, "fn_ple1_out.f32"))
    np.asarray(tokens.numpy(), dtype="<i4").tofile(os.path.join(base.FIXTURES, "fn_ple1_ids.i32"))
    meta = {"layer": LAYER, "ple_index": PLE_INDEX, "tokens": T, "width": cfg.hc_count * cfg.hidden_size,
            "embed_dim": cfg.ple_embed_dim, "seed": SEED, "eos": EOS, "vocab": VOCAB, "ngram_size": NGRAM,
            "heads_per_ngram": HPN, "base": BASE, "seed_const": NSEED, "shards_checked": n_shards,
            "out_abs_mean": float(y.abs().mean()), "total_rows_heads": total, "total_rows_json": LOC["total_rows"]}
    base.json.dump(meta, open(os.path.join(base.FIXTURES, "fn_ple1_meta.json"), "w"), indent=1)
    print("wrote", meta)


if __name__ == "__main__":
    main()
