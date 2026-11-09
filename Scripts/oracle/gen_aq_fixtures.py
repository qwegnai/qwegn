#!/usr/bin/env python3
"""The load-time writers' goldens: the quantiser and the repacker.

Two fixtures, both committed under Tests/QWQuantTests/fixtures/:

  aq_quantizer_golden.json — a BF16 weight matrix and the HQ4_G32 blocks
    the INCUMBENT Python packer writes for it. The packer is
    Scripts/quantize/hq4.py; this is what the GPU quantiser must match.

  aq_repack_golden.json — an MLX affine tensor, MLX's own codes and MLX's
    own dequantised values. MLX is the witness for the repacker's layout:
    the generator asserts that MLX's codes reproduce `mx.dequantize` before
    it writes them, so the fixture cannot encode a wrong order.

Run: python3 Scripts/oracle/gen_aq_fixtures.py
"""
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "Scripts" / "quantize"))
import hq4  # noqa: E402

OUT = ROOT / "Tests" / "QWQuantTests" / "fixtures"


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16, nearest-even, as the hardware would."""
    bits = values.astype(np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) & 0xFFFF0000).astype(np.uint32) >> 16


def build_quantizer_golden():
    rows, cols = 4, 64
    rng = np.random.default_rng(20261026)
    weights = (rng.standard_normal((rows, cols)) * 0.7).astype(np.float32)
    bf16 = f32_to_bf16(weights)
    # The device reads bf16, so the packer must quantise the bf16 values.
    as_f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
    blocks = hq4.quant_hq4_g32_m5(as_f32)
    return {
        "rows": rows,
        "cols": cols,
        "bf16": bf16.astype("<u2").tobytes().hex(),
        "blocks": np.ascontiguousarray(blocks).tobytes().hex(),
    }


def write_quantizer_golden():
    doc = build_quantizer_golden()
    path = OUT / "aq_quantizer_golden.json"
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} ({doc['rows'] * doc['cols'] // 32} blocks)")


def build_repack_golden():
    import mlx.core as mx

    rows, cols, group, bits = 2, 128, 64, 4
    mx.random.seed(20261026)
    w = mx.random.normal((rows, cols)).astype(mx.float32)
    qw, scales, biases = mx.quantize(w, group_size=group, bits=bits, mode="affine")
    groups = cols // group

    packed = np.frombuffer(np.array(qw.view(mx.uint32)).tobytes(), dtype=np.uint8)
    packed = packed.reshape(rows, -1)
    sc = np.array(scales).reshape(rows, groups).astype(np.float32)
    bi = np.array(biases).reshape(rows, groups).astype(np.float32)
    values = np.array(mx.dequantize(qw, scales, biases, group_size=group, bits=bits))

    # MLX's codes, read the way MLX packs 4 bits: byte b holds elements
    # 2b and 2b+1. The witness is MLX's own dequantise — if the order were
    # wrong, the values would not come back.
    codes = np.zeros((rows, cols), dtype=np.uint8)
    gbytes = group * bits // 8
    for r in range(rows):
        for g in range(groups):
            at = g * gbytes
            for i in range(group):
                byte = packed[r][at + i // 2]
                codes[r][g * group + i] = (byte >> (4 * (i % 2))) & 0xF
            got = sc[r][g] * codes[r][g * group:(g + 1) * group].astype(np.float32) + bi[r][g]
            want = values[r][g * group:(g + 1) * group]
            err = float(np.abs(got - want).max())
            assert err < 1e-5, f"MLX's 4-bit order does not reproduce mx.dequantize (err {err})"

    return {
        "bits": bits,
        "group": group,
        "rows": rows,
        "cols": cols,
        "packed": np.ascontiguousarray(packed).tobytes().hex(),
        "scales": np.ascontiguousarray(sc).tobytes().hex(),
        "biases": np.ascontiguousarray(bi).tobytes().hex(),
        "codes": np.ascontiguousarray(codes).tobytes().hex(),
        "values": np.ascontiguousarray(values).tobytes().hex(),
    }


def write_repack_golden():
    doc = build_repack_golden()
    path = OUT / "aq_repack_golden.json"
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} ({doc['rows'] * doc['cols'] // doc['group']} groups)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_quantizer_golden()
    write_repack_golden()


if __name__ == "__main__":
    main()
