#!/usr/bin/env python3
"""C14 MLP fixtures (spec 14; parity ≤5e-4 f16-out at all M incl MTP M=8).

Geometry (QWGeometry): H=5120, I=17408. Packed gate‖up per the C05
silu_pair kernel contract: packed row layout [0,I)=gate, [I,2I)=up, so the
fused GEMV produces h[i]=silu(gate_i)·up_i in ONE pass. Down: [H, I]
(inner dim I ⇒ 136 quant groups; the down GEMV's xGroupSum lives over h).

The quantizer is bit-identical to gen_s1_q4_fixtures.py (asym g128 over the
f16-rounded weight, groups WITHIN a row only) — the reference y is computed
from the EXACT stored pack values in fp32, so engine tests measure GEMV
math, not quantization error (the torch-oracle ≤5e-4 gate runs at S4 on
real ckpt weights; here the anchor is the dequant-GEMV itself).

Files (deterministic RNG; same weights for every M; x varies by rows):
  mlp_gate.bin   mlp_gate_sc/mn     q4-g128 [I, 40, 64] + [I,40] f16 planes
  mlp_up.bin     mlp_up_sc/mn       ditto
  mlp_gu.bin     mlp_gu_sc/mn       PACKED [2I, 40, 64] (gate rows [0,I),
                                   up rows [I,2I)) — planes proven EQUAL to
                                   the per-tensor files (row-local groups)
  mlp_down.bin   mlp_down_sc/mn     [H, 136, 64] + [H,136] planes
  mlp_x_M{M}.bin        f16 [M, 5120]        (M ∈ {1,2,8}; 8 = MTP batch)
  mlp_gsum_M{M}.bin     f32 [M, 40]          per-row x group sums
  mlp_gsum1.bin         f32 [40]             (M=1: xgroupsum kernel ref)
  mlp_h_M{M}.bin        f32 [M, 17408]       silu(gate·x)·(up·x)
  mlp_hsum_M{M}.bin     f32 [M, 136]         per-row h group sums (down GEMV)
  mlp_hsum1.bin         f32 [136]            (M=1)
  mlp_y_M{M}.bin        f32 [M, 5120]        down·h — the ≤5e-4 parity target
  mlp_h1packed.bin      f32 [17408]          (M=1: silu_pair kernel's out)"""
import numpy as np, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "quantize"))
from pack import quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

H, I, K = 5120, 17408, 5120
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(14)

def q4g(W, name):
    """HQ4_G32_M5 packer over W; returns exact rotated dequant matrix matching Metal."""
    blocks = quant_hq4_g32_m5(W)
    (out / f"{name}.bin").write_bytes(blocks.tobytes())
    return dequant_hq4_g32_m5_rot(blocks, f16_round=True)

gate = rng.uniform(-0.02, 0.02, size=(I, K)).astype(np.float32)
up   = rng.uniform(-0.02, 0.02, size=(I, K)).astype(np.float32)
down = rng.uniform(-0.02, 0.02, size=(H, I)).astype(np.float32)

gq = q4g(gate, "mlp_gate")
uq = q4g(up, "mlp_up")
# packed [2I, K]: gate rows [0,I), up rows [I,2I) — C05 kernel contract.
# Quantizing the concat re-uses per-row stats ⇒ planes MUST be bit-equal to
# the per-tensor files (groups never span rows; assert = layout proof).
pq = q4g(np.concatenate([gate, up], axis=0), "mlp_gu")
assert np.array_equal(pq[:I], gq) and np.array_equal(pq[I:], uq), \
    "packed planes must equal per-tensor planes"
dq = q4g(down, "mlp_down")
assert dq.shape == (H, I)

for M in (1, 2, 8):
    # x ~ N(0,1): post_attention_layernorm output is unit-scale in the real
    # net — with the old 0.02·unif the whole y landed under the 5e-4 abs
    # tolerance (VACUOUS parity test: garbage passes). Kernel math does not
    # care; the TEST does.
    x = rng.standard_normal(size=(M, K)).astype(np.float32).astype(np.float16)
    xf = x.astype(np.float32)
    (out / f"mlp_x_M{M}.bin").write_bytes(x.view(np.uint16).tobytes())
    gsum = xf.reshape(M, K // 128, 128).sum(axis=2).astype(np.float32)
    (out / f"mlp_gsum_M{M}.bin").write_bytes(gsum.tobytes())
    hg = (xf @ gq.T)                                   # [M, I] fp32 gate·Wgᵀ
    hu = (xf @ uq.T)                                   # up·Wuᵀ
    h = hg / (1.0 + np.exp(-hg)) * hu                  # silu exact form (fp32)
    # spec-14 line 10: intermediate activations f16. The batched gu kernel
    # STORES h as f16 and the down GEMV reads it back — so the reference y
    # must multiply the f16-rounded h (else h's own f16 rounding, ~1e-3 rel,
    # breaks parity independent of kernel correctness). h dump is f16.
    h16 = h.astype(np.float16)
    hsum = h16.astype(np.float32).reshape(M, I // 128, 128).sum(axis=2).astype(np.float32)
    y = h16.astype(np.float32) @ dq.T                  # [M, H] f32 (down out dumps f16)
    (out / f"mlp_h_M{M}.bin").write_bytes(h16.view(np.uint16).tobytes())
    (out / f"mlp_hsum_M{M}.bin").write_bytes(hsum.tobytes())
    (out / f"mlp_y_M{M}.bin").write_bytes(y.astype(np.float32).tobytes())
    if M == 1:
        (out / "mlp_gsum1.bin").write_bytes(gsum[0].tobytes())
        (out / "mlp_hsum1.bin").write_bytes(hsum[0].tobytes())
        (out / "mlp_h1packed.bin").write_bytes(h16[0].view(np.uint16).tobytes())
    print(f"M={M}: |h|max {np.abs(h).max():.5f} |y|max {np.abs(y).max():.5f}")
print("c14 fixtures done")
