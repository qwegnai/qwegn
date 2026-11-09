#!/usr/bin/env python3
"""Real-dimension golden for FlashNext decoder layer 0 (GatedDeltaNet + sparse MoE at Qwen3.8-Flash-Next dims).

Weights come from the real HQ4 pack (model.qwp) dequantized with the verified law (nibble map,
(nib+1024)*scale+bias, orthonormal FWHT per group-32; expert e at hq4_off + e*e_hq4). The HF reference
(transformers qwen4_exp) runs the decoder layer in fp32 on a fixed random stream.

Writes fixtures (float32 little-endian) and meta under Tests/QWFlashNextTests/fixtures/:
  fn_layer0_in.f32, fn_layer0_out.f32, fn_layer0_meta.json
Run with the golden venv: env -u PYTHONPATH PYTHONNOUSERSITE=1 .../qwegine.oracle-fn/bin/python ...
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

PACK_DIR = os.path.expanduser("~/.qwg/models/Qwen3.8-Flash-Next-hq4g32")
PREFIX = "model.language_model."
LAYER = 0
T = 8
SEED = 20261021
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "Tests", "QWFlashNextTests", "fixtures")
BLOCK = 20
GROUP = 32


def load_header(pack_path):
    size = os.path.getsize(pack_path)
    with open(pack_path, "rb") as f:
        f.seek(4)
        hlen = int.from_bytes(f.read(8), "little")
        f.seek(size - hlen)
        return json.loads(f.read(hlen))


def dequant_blocks(raw, rows, cols):
    """HQ4_G32 rows -> float32 (rows, cols), exactly the Swift dequantHQ4Row law."""
    blk = np.frombuffer(raw, dtype=np.dtype([("s", "<f2"), ("b", "<f2"), ("q", "<u4", (4,))]))
    assert blk.size == rows * (cols // GROUP)
    s = blk["s"].astype(np.float32).reshape(-1, 1)
    b = blk["b"].astype(np.float32).reshape(-1, 1)
    q = blk["q"].reshape(-1, 4)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    nib = np.zeros((blk.size, GROUP), dtype=np.float32)
    for j in range(4):
        for k in range(8):
            nib[:, 8 * j + k] = ((q[:, j] >> shifts[k]) & 0xF).astype(np.float32)
    v = ((nib + np.float32(1024.0)) * s + b).reshape(-1, GROUP)
    h = 1
    while h < GROUP:
        v = v.reshape(-1, GROUP // (2 * h), 2, h)
        u = v[:, :, 0, :].copy()
        w = v[:, :, 1, :].copy()
        v[:, :, 0, :] = u + w
        v[:, :, 1, :] = u - w
        v = v.reshape(-1, GROUP)
        h *= 2
    return (v * np.float32(1.0 / np.sqrt(GROUP))).reshape(rows, cols)


def read_at(f, off, n):
    f.seek(off)
    return f.read(n)


def pack_tensor(f, hdr, name):
    e = hdr["tensors"][name]
    shape = e["shape"]
    if e.get("kind") == "bf16":
        n = int(np.prod(shape))
        raw = np.frombuffer(read_at(f, e["raw_off"], 2 * n), dtype="<u2").astype(np.uint32) << 16
        return raw.view(np.float32).reshape(shape)
    if len(shape) == 2:
        rows, cols = shape
        return dequant_blocks(read_at(f, e["hq4_off"], rows * (cols // GROUP) * BLOCK), rows, cols)
    if len(shape) == 3 and e.get("experts"):
        count, rows, cols = shape
        per = e["e_hq4"]
        assert per == rows * (cols // GROUP) * BLOCK
        out = np.empty(shape, dtype=np.float32)
        for ex in range(count):
            out[ex] = dequant_blocks(read_at(f, e["hq4_off"] + ex * per, per), rows, cols)
        return out
    raise ValueError(f"unhandled pack entry {name}: {e}")


def main():
    pack_path = os.path.join(PACK_DIR, "model.qwp")
    hdr = load_header(pack_path)
    assets = json.load(open(os.path.join(PACK_DIR, "assets", "config.json")))
    tc = assets.get("text_config", assets)
    cfg = Qwen4ExpTextConfig.from_dict(tc)
    cfg._attn_implementation = "eager"
    cfg.num_hidden_layers = cfg.num_hidden_layers
    torch.manual_seed(SEED)
    layer = M.Qwen4ExpTextDecoderLayer(cfg, layer_idx=LAYER).eval()
    prefix = f"{PREFIX}layers.{LAYER}."
    want = {k: None for k in layer.state_dict().keys()}
    state = {}
    with open(pack_path, "rb") as f:
        for k in want:
            name = prefix + k
            if name not in hdr["tensors"]:
                sys.exit(f"missing pack tensor {name}")
            state[k] = torch.from_numpy(np.ascontiguousarray(pack_tensor(f, hdr, name)))
    missing, unexpected = layer.load_state_dict(state, strict=True), None
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, T, cfg.hc_count * cfg.hidden_size, generator=gen) * 0.5
    with torch.no_grad():
        y = layer(x, position_embeddings=(None, None))
    os.makedirs(FIXTURES, exist_ok=True)
    x.numpy().astype("<f4").tofile(os.path.join(FIXTURES, "fn_layer0_in.f32"))
    y.numpy().astype("<f4").tofile(os.path.join(FIXTURES, "fn_layer0_out.f32"))
    meta = {"layer": LAYER, "tokens": T, "width": cfg.hc_count * cfg.hidden_size, "seed": SEED,
            "layer_type": cfg.layer_types[LAYER], "tensors": len(state),
            "out_abs_mean": float(y.abs().mean()), "out_abs_max": float(y.abs().max())}
    json.dump(meta, open(os.path.join(FIXTURES, "fn_layer0_meta.json"), "w"), indent=1)
    print("wrote", meta, missing)


if __name__ == "__main__":
    main()
