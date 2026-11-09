#!/usr/bin/env python3
"""S1 HQ4_G32_M5 race fixtures — numpy does the O(N·K) work the race must NOT
measure in scalar Swift. For each N in {1024, 6144, 10240, 12288, 17408}
(K=5120) writes:
  s1_q4_N{N}.bin    — HQ4_G32_M5 blocks [N, K/32, 20] (the only weight plane)
  s1_q4_N{N}_x.bin  — shared x f16 [K]
  s1_q4_N{N}_gs.bin — legacy x group sums f32 [40]
  s1_q4_N{N}_y.bin  — fp32 reference y[N] (numpy dequant-GEMV, same math as kernel)
Deterministic RNG across all N. Quantizer = pack.py's HQ4_G32_M5."""
import numpy as np, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "quantize"))
from pack import quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

K = 5120
NS = [1024, 6144, 10240, 12288, 17408]
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
out.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
x = rng.uniform(-0.02, 0.02, size=K).astype(np.float32).astype(np.float16)
xf = x.astype(np.float32)
gs = xf.reshape(K // 128, 128).sum(axis=1).astype(np.float32)

for N in NS:
    W = rng.uniform(-0.02, 0.02, size=(N, K)).astype(np.float32)
    blocks = quant_hq4_g32_m5(W)
    wq = dequant_hq4_g32_m5_rot(blocks, f16_round=True)
    y = wq @ xf
    (out / f"s1_q4_N{N}.bin").write_bytes(blocks.tobytes())
    (out / f"s1_q4_N{N}_x.bin").write_bytes(x.view(np.uint16).tobytes())
    (out / f"s1_q4_N{N}_gs.bin").write_bytes(gs.view(np.uint8).tobytes())
    (out / f"s1_q4_N{N}_y.bin").write_bytes(y.astype(np.float32).tobytes())
    print(f"N={N}: blocks {blocks.nbytes/1e6:.1f} MB, |y|max {np.abs(y).max():.4f}")
print("fixtures done")
