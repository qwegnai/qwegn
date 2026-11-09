#!/usr/bin/env python3
"""Build an HQ4_G32_M5 draft-head sidecar for the MTP speculative path (L1).

The draft LM head is a DRAFT-only artifact: the target verification uses
the target head. This tool quantizes the source `lm_head.weight` with the SAME
`quant_hq4_g32_m5` the pack uses for every trunk weight, so the device kernel's
unpack path (C04 golden) is identical.

The sidecar is a real QWP1 pack (one tensor: lm_head.weight) built through
pack.py's own `pack_file` — one wire, one writer (C04 law): the Swift
QWQuant.Pack reader validates magic, geometry hash, and layout for free.

Usage:
  python3 Scripts/quantize/build_draft_head.py \
      --model /path/to/Qwen3.8-27B-snapshot \
      --out $HOME/.qwg/models/Qwen3.8-27B-hq4g32/lm_head.q4g
  python3 Scripts/quantize/build_draft_head.py --out ... --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pack import BLOCK_BYTES, GROUP, dequant_hq4_g32_m5, pack_file  # noqa: E402

HEAD_NAME = "lm_head.weight"


def read_lm_head_bf16(model_dir: Path) -> np.ndarray:
    """Read lm_head.weight (bf16) from the source snapshot via its index."""
    from safetensors import safe_open
    import torch

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][HEAD_NAME]
    with safe_open(str(model_dir / shard), framework="pt") as f:
        t = f.get_tensor(HEAD_NAME)
        assert t.dtype == torch.bfloat16, f"{HEAD_NAME} dtype {t.dtype}"
        return t.float().numpy()


def build(model_dir: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    w = read_lm_head_bf16(model_dir)
    provenance = {
        "source": str(model_dir.resolve()),
        "quant_recorded": "hq4_g32_m5",
        "note": "MTP draft-only LM head sidecar (L1, HQ4_G32_M5)",
    }
    pack_file([(HEAD_NAME, w)], out, provenance)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.3f} GB) — "
          f"{HEAD_NAME} hq4_g32_m5 {w.shape[0]}x{w.shape[1]}")


def verify(out: Path, model_dir: Path | None) -> None:
    """CPU-side dequant sanity on the sidecar's own wire."""
    with open(out, "rb") as f:
        assert f.read(4) == b"QWP1", "bad magic"
        hlen = int.from_bytes(f.read(8), "little")
        f.seek(-hlen, 2)
        h = json.loads(f.read(hlen))
        e = h["tensors"][HEAD_NAME]
        N, K = e["shape"]
        G = K // GROUP
        f.seek(e["hq4_off"])
        blocks = np.frombuffer(f.read(N * G * BLOCK_BYTES), dtype=np.uint8).reshape(N, G, BLOCK_BYTES)
        scale = np.ascontiguousarray(blocks[..., 0:2]).view(np.float16).astype(np.float32)
        bias = np.ascontiguousarray(blocks[..., 2:4]).view(np.float16).astype(np.float32)
        print(f"planes ok: {N}x{K} groups={G} "
              f"scale range [{scale.min():.4g}, {scale.max():.4g}] "
              f"bias range [{bias.min():.4g}, {bias.max():.4g}]")
    if model_dir is not None:
        w = read_lm_head_bf16(model_dir)
        rows = np.linspace(0, N - 1, 64).astype(int)
        dq = dequant_hq4_g32_m5(blocks[rows])
        err = np.abs(dq - w[rows])
        rel = err.max() / (np.abs(w[rows]).max() + 1e-9)
        print(f"sampled parity (64 rows): max abs err {err.max():.5g}, "
              f"max rel {rel:.5g}")
    print(f"file size: {out.stat().st_size} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", metavar="DIR", help="source snapshot dir")
    ap.add_argument("--out", metavar="FILE", required=True)
    ap.add_argument("--verify", action="store_true", help="CPU dequant check")
    a = ap.parse_args()
    if a.verify:
        verify(Path(a.out), Path(a.model) if a.model else None)
        sys.exit(0)
    if not a.model:
        ap.error("--model required unless --verify")
    build(Path(a.model), Path(a.out))
