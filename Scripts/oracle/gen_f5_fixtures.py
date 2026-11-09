#!/usr/bin/env python3
"""gen_f5_fixtures.py — C05-F5 skinny-GEMM race fixtures (MTP verify batch).

Family law (QWGemmShape): m ≤ 8 decode GEMMs = F5. The verify window
commits ≤ 1+N tokens/round (N drafts ⇒ M = 1..3 at N=2; the race sweeps
M ∈ {2,3,4,8}). The gate (spec 05/16): ONE launch reading each weight
plane ONCE and dotting M x-rows must beat the C05 F1 winner run per-row
×M on wall time — C14's batched attempt FAILED this scaling law (M=8 ≈
4.8× M=1 time, ≈62 GB/s; DECISIONS records the anomaly). If F5 can't win,
speculation stays off (spec 16's own line) — that is a legitimate honest
outcome of this race, recorded as such.

Shapes (real trunk geometry at the verify hot door):
  q   [12288, 5120]   q_proj (Q‖gate packed rows)
  kv  [1024, 5120]    k/v_proj (shared planes; C09 takes the same rows)
  o   [5120, 6144]    o_proj
  fc  [5120, 10240]   mtp fc (draft leg — batched for N>1 streaming too)
  gu  [34816, 5120]   packed gate‖up rows (epi .siluMulPair variant raced
                      separately; here .none math y = W_deq·x)
  dn  [5120, 17408]   down_proj
  hd  [61440, 5120]   lm_head surrogate (quarter vocab; rows structurally
                      identical — the FULL-vocab bench loads real planes)
Planes: C15 bit-identical q4-g128 packer over f16-rounded weights.
Per (shape, M): x f16 [M,K]; gs f32 [M,K/128] (torch-sum order — parity
absorbs the last-ulp fold delta on y, law recorded in C16 leg-1);
yref f32 [M,N] = dequant-plane math on the f16 x (f32 accumulate; F5 y
parity law: ≤5e-4 + f16-store spacing, C14 pattern).
"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "quantize"))
from pack import quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(17)

def q4g(W, name):
    """HQ4_G32_M5 packer over W; returns exact rotated dequant matrix matching Metal."""
    blocks = quant_hq4_g32_m5(W)
    (out / f"{name}.bin").write_bytes(blocks.tobytes())
    return dequant_hq4_g32_m5_rot(blocks, f16_round=True)

SHAPES = {
    "f5q":  (12288, 5120),
    "f5kv": (1024, 5120),
    "f5o":  (5120, 6144),
    "f5fc": (5120, 10240),
    "f5gu": (34816, 5120),
    "f5dn": (5120, 17408),
    "f5hd": (61440, 5120),
}
MS = (2, 3, 4, 8)

total = 0
for name, (n, k) in SHAPES.items():
    # decode-weight init scale ~ 1/sqrt(k) (init_small law, std 0.02-class)
    W = (rng.standard_normal((n, k)) * (k ** -0.5)).astype(np.float32)
    deq = q4g(W, name)
    del W
    for M in MS:
        x = rng.standard_normal(k * M).astype(np.float32).astype(np.float16)
        gs = x.astype(np.float32).reshape(M, k // 128, 128).sum(2).astype(np.float32)
        y = np.zeros((M, n), np.float32)
        xf = x.astype(np.float32)                     # [M·K]
        CH = 16384                                    # chunked f32-accum dot (no [N,K] intermediates)
        for c0 in range(0, k, CH):
            wc = deq[:, c0:c0 + CH]                   # [N,CH] f32 view of the dequant plane
            xc = xf.reshape(M, k)[:, c0:c0 + CH]      # [M,CH]
            y += wc @ xc.T if M == 1 else np.einsum("nk,mk->mn", wc, xc, optimize=True)
        yref = y.reshape(-1).astype(np.float32)       # [M·N] row-major
        (out / f"{name}_x_M{M}.bin").write_bytes(x.view(np.uint16).tobytes())
        (out / f"{name}_gs_M{M}.bin").write_bytes(gs.tobytes())
        (out / f"{name}_y_M{M}.bin").write_bytes(yref.tobytes())
        if M <= 3:   # F5 v6 (TG-staged, M=2/3) races with f16-store output
            (out / f"{name}_y16_M{M}.bin").write_bytes(yref.astype(np.float16).view(np.uint16).tobytes())
        assert np.isfinite(yref).all() and yref.size == M * n
        del x, xf, gs, yref
    del deq
    total += 1
    print(f"{name}: [{n},{k}] planes + M∈{MS} fixtures ok")
print(f"f5 fixtures: {total} shapes × {len(MS)} M — race ready")
