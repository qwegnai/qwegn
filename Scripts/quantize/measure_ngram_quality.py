"""Measure the QNGRAM v2 row formats against the real table.

The figures published in `specs/04-qwquant.md` come from this script, run
against the shipped `assets/ngram.bin` of a real model. Re-run it to check
them; do not quote a number this script does not print.

    python Scripts/quantize/measure_ngram_quality.py \
        ~/.qwg/models/Qwen3.8-Flash-Next-hq4g32/assets/ngram.bin

The reader's rule is one line: value = code * scale. This script applies it
to the rows the writer would have produced, and reports what it costs.
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hq4 import quant_hq4_blocks  # noqa: E402
from pack_ngram import (  # noqa: E402
    _F16_MIN_NORMAL,
    _encode_q8,
    _encode_vq2,
    decode_vq2_rows,
    train_vq2_codebook,
)

GROUP = 32
HEAD_DIM = 160
GROUPS = HEAD_DIM // GROUP


def deq_q8(rows: np.ndarray) -> np.ndarray:
    """Decode a Q8_G32 row the way the reader will: code * stored scale."""
    enc = np.frombuffer(_encode_q8(rows, np), dtype=np.uint8)
    enc = enc.reshape(-1, GROUPS, GROUP + 2)
    codes = enc[:, :, :GROUP].view(np.int8).astype(np.float32)
    scale = np.ascontiguousarray(enc[:, :, GROUP:]).view(np.float16)
    scale = scale.reshape(-1, GROUPS).astype(np.float32)
    return (codes * scale[:, :, None]).reshape(rows.shape)


def deq_q4(rows: np.ndarray) -> np.ndarray:
    """Decode a Q4_G32 row: the block layout's own reconstruction."""
    blocks = quant_hq4_blocks(rows, rotate=False)
    n5 = (rows.shape[0], GROUPS)
    scale = np.ascontiguousarray(blocks[..., 0:2]).view(np.float16).reshape(n5)
    bias = np.ascontiguousarray(blocks[..., 2:4]).view(np.float16).reshape(n5)
    words = np.ascontiguousarray(blocks[..., 4:20]).view(np.uint32)
    q = np.empty((rows.shape[0], GROUPS, GROUP), dtype=np.float32)
    for j in range(4):
        b = 8 * j
        raw = words[..., j]
        q[..., b + 0] = (raw >> 0) & 0xF
        q[..., b + 1] = (raw >> 16) & 0xF
        q[..., b + 2] = (raw >> 4) & 0xF
        q[..., b + 3] = (raw >> 20) & 0xF
        q[..., b + 4] = (raw >> 8) & 0xF
        q[..., b + 5] = (raw >> 24) & 0xF
        q[..., b + 6] = (raw >> 12) & 0xF
        q[..., b + 7] = (raw >> 28) & 0xF
    out = (q + np.float32(1024.0)) * scale.astype(np.float32)[..., None]
    out = out + bias.astype(np.float32)[..., None]
    return out.reshape(rows.shape)


def deq_vq2(train_rows: np.ndarray, eval_rows: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Train a 1 KiB e-means VQ2_G32 codebook on train_rows and decode eval_rows."""
    cb = train_vq2_codebook(train_rows, seed=0, device=device)
    enc = _encode_vq2(eval_rows, cb, np, device=device)
    return decode_vq2_rows(enc, cb, eval_rows.shape[0], HEAD_DIM)


def sample_rows(path: Path, sectors: int) -> np.ndarray:
    """Read rows spread evenly across the table's sectors (v1 BF16 or v2 Q8_G32)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(4096)
        entry_count, dim, row_bytes = struct.unpack_from("<QHH", head, 16)
        data_offset, = struct.unpack_from("<Q", head, 32)
        version, flags, dtype = struct.unpack_from("<HHI", head, 8)
        if (version, dtype) not in ((1, 0), (2, 1)):
            raise SystemExit(f"{path}: expected v1 BF16 or v2 Q8_G32 table, got "
                             f"version={version} dtype={dtype}")
        rps = 4096 // row_bytes
        total = (size - data_offset) // 4096
        picks = sorted(set(int(i * (total - 1) / max(sectors - 1, 1))
                            for i in range(sectors)))
        rows = []
        for s in picks:
            f.seek(data_offset + s * 4096)
            blk = f.read(rps * row_bytes)
            if dtype == 0:
                u16 = np.frombuffer(blk, dtype="<u2").astype("<u4") << np.uint32(16)
                rows.append(u16.view("<f4").reshape(-1, dim))
            else:
                g = np.frombuffer(blk, dtype=np.uint8).reshape(rps, dim // GROUP, GROUP + 2)
                codes = g[:, :, :GROUP].view(np.int8).astype(np.float32)
                sc = np.ascontiguousarray(g[:, :, GROUP:]).view("<f2").reshape(rps, dim // GROUP).astype(np.float32)
                rows.append((codes * sc[:, :, None]).reshape(rps, dim))
    return np.concatenate(rows)


def report(name: str, src: np.ndarray, got: np.ndarray) -> None:
    rn = np.linalg.norm(src, axis=-1)
    err = np.linalg.norm(got - src, axis=-1) / np.maximum(rn, 1e-300)
    den = np.linalg.norm(got, axis=-1) * rn
    cos = np.ones_like(rn)
    ok = den > 0
    cos[ok] = (got * src).sum(axis=-1)[ok] / den[ok]
    g = src.reshape(-1, GROUPS, GROUP)
    peak_in = np.abs(g).max(axis=-1)
    peak_out = np.abs(got.reshape(-1, GROUPS, GROUP)).max(axis=-1)
    clipped = (peak_out < peak_in * 0.999) & (peak_in > 0)
    ratio = np.where(clipped, peak_out / np.maximum(peak_in, 1e-300), 1.0)
    print(f"\n{name}:")
    print(f"  row relative L2 error: mean={err.mean():.6f} "
          f"p99={np.percentile(err, 99):.6f} max={err.max():.6f}")
    print(f"  cosine similarity: mean={cos.mean():.8f} min={cos.min():.8f} "
          f"p0.1={np.percentile(cos, 0.1):.8f}")
    print(f"  groups whose peak is clipped: {int(clipped.sum()):,} of "
          f"{peak_in.size:,} ({100.0 * clipped.mean():.2f}%)")
    if clipped.any():
        print(f"    peak kept: mean={ratio[clipped].mean():.4f} "
              f"min={ratio[clipped].min():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", metavar="NGRAM_BIN",
                    help="the ngram.bin of a real model")
    ap.add_argument("--sectors", type=int, default=800,
                    help="how many sectors to sample across the table")
    a = ap.parse_args()
    path = Path(os.path.expanduser(a.table))
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    src = sample_rows(path, a.sectors)
    norm = np.linalg.norm(src, axis=-1)
    live = src[norm > 0]
    print(f"{path}: sampled {src.shape[0]:,} rows, {live.shape[0]:,} non-degenerate")
    am = np.abs(live.reshape(-1, GROUPS, GROUP)).max(axis=-1)
    cliff = 127.0 * _F16_MIN_NORMAL
    print(f"per-group peak: median={np.median(am):.4e} min={am.min():.4e} "
          f"max={am.max():.4e}")
    print(f"groups below the f16-normal cliff ({cliff:.4e}): "
          f"{(am < cliff).sum():,} of {am.size:,} ({100.0 * (am < cliff).mean():.4f}%)")
    report("Q8_G32", live, deq_q8(live))
    report("Q4_G32", live, deq_q4(live))
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    train_live, eval_live = live[0::2], live[1::2]
    report("Q4_G32 (held-out split)", eval_live, deq_q4(eval_live))
    report("VQ2_G32 (e-means, held-out split)", eval_live, deq_vq2(train_live, eval_live, device=dev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
