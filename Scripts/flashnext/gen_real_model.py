#!/usr/bin/env python3
"""Full 48-layer golden for FlashNext: HF Qwen4ExpForCausalLM semantics, run layer by layer on the real HQ4 pack.

The full HF model cannot be instantiated (experts alone ~480 GB fp32), so each decoder layer is built, loaded
from the pack (verified dequant law), applied to the stream, and released. Same math as the HF forward:
  stream = tile(embed(ids), hc); for each layer: [PLE residual on layer 1 with real N-gram rows] -> decoder layer;
  logits = lm_head(hyper_connection_mixer(stream)).

Outputs go OUTSIDE the repo (large): $GOLDEN_DIR (default ~/Developer/ai/qwegine.oracle-fn/goldens/flashnext_full):
  meta.json, ids.i32, streams.f32 (num_layers x T x hc*H), logits.f32 (T x vocab), progress lines to stdout.
"""
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_real_layer0 as base  # noqa: E402
import gen_real_ple1 as ple1  # noqa: E402

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

T = 4
SEED = 20261024
OUT = os.environ.get("GOLDEN_DIR", os.path.expanduser(
    "~/Developer/ai/qwegine.oracle-fn/goldens/flashnext_full"))


class FakeEmbedding(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, input_ids, past_key_values=None):
        return self.emb


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    pack_path = os.path.join(base.PACK_DIR, "model.qwp")
    hdr = base.load_header(pack_path)
    assets = json.load(open(os.path.join(base.PACK_DIR, "assets", "config.json")))
    cfg = Qwen4ExpTextConfig.from_dict(assets.get("text_config", assets))
    cfg._attn_implementation = "eager"
    num_layers = cfg.num_hidden_layers
    width = cfg.hc_count * cfg.hidden_size
    small = Qwen4ExpTextConfig.from_dict(assets.get("text_config", assets))
    small._attn_implementation = "eager"
    small.ngram_vocab_size_base = 30           # PLE table is replaced; keeps construction small
    small.make_ngram_vocab_size_divisible_by = 8

    gen = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, cfg.vocab_size, (T,), generator=gen)
    ids[2] = cfg.eos_token_id

    with open(pack_path, "rb") as f:
        embed_w = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, base.PREFIX + "embed_tokens.weight")))
        stream = embed_w[ids].unsqueeze(0).repeat(1, 1, cfg.hc_count)
        del embed_w
        ple_emb = None
        pos = torch.arange(T).view(1, T)
        rotary = M.Qwen4ExpTextRotaryEmbedding(small)
        cos, sin = rotary(stream, pos.unsqueeze(0).expand(3, 1, T))
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        mask = torch.zeros(1, 1, T, T).masked_fill(~causal, torch.finfo(torch.float32).min)

        # PLE rows for layer 1 (real shards), via the verified N-gram helpers.
        mult = M._build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, 0, ple1.NSEED).tolist()
        heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
        sizes, offsets, total = [], [], 0
        for h in range(heads):
            s_ = M._find_nth_prime_after(cfg.ngram_vocab_size_base - 1, h + 1)
            sizes.append(s_)
            offsets.append(total)
            total += s_
        tok = ids.clone()
        ids_ngram = ple1.ngram_ids(tok, mult, sizes, offsets)
        emb_rows = [np.concatenate([ple1.read_row(int(r)) for r in ids_ngram[t].tolist()]) for t in range(T)]
        ple_emb = torch.from_numpy(np.stack(emb_rows).astype(np.float32)).unsqueeze(0)
        log(f"ple rows read: {T * heads}")

        streams = np.zeros((num_layers, T, width), dtype="<f4")
        for i in range(num_layers):
            t0 = time.time()
            layer = M.Qwen4ExpTextDecoderLayer(small, layer_idx=i).eval()
            if layer.ple is not None:
                layer.ple.ple_embedding = FakeEmbedding(ple_emb)
            prefix = f"{base.PREFIX}layers.{i}."
            state = {}
            for k in layer.state_dict().keys():
                state[k] = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, prefix + k)))
            layer.load_state_dict(state, strict=True)
            del state
            with torch.no_grad():
                stream = layer(stream, position_embeddings=(cos, sin), attention_mask=mask,
                               conv_mask=None, past_key_values=None, ple_input_ids=ids.view(1, T))
            streams[i] = stream[0].numpy()
            log(f"layer {i} ({cfg.layer_types[i]}{' +PLE' if layer.ple is not None else ''}) done in {time.time() - t0:.1f}s")
            del layer
            gc.collect()

        mixer = M.Qwen4ExpTextGatedResidual(small, use_combine=False).eval()
        mstate = {k: torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, base.PREFIX + "hyper_connection_mixer." + k)))
                  for k in mixer.state_dict().keys()}
        mixer.load_state_dict(mstate, strict=True)
        with torch.no_grad():
            mixed = mixer(stream)
        lm = torch.from_numpy(np.ascontiguousarray(base.pack_tensor(f, hdr, "lm_head.weight")))
        with torch.no_grad():
            logits = mixed @ lm.T

    ids.numpy().astype("<i4").tofile(os.path.join(OUT, "ids.i32"))
    streams.tofile(os.path.join(OUT, "streams.f32"))
    logits[0].numpy().astype("<f4").tofile(os.path.join(OUT, "logits.f32"))
    meta = {"tokens": T, "layers": num_layers, "width": width, "vocab": cfg.vocab_size, "seed": SEED,
            "eos": cfg.eos_token_id, "ids": ids.tolist(), "logits_abs_max": float(logits.abs().max()),
            "layer_types": cfg.layer_types}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    log(f"wrote golden to {OUT}: logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
