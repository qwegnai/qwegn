#!/usr/bin/env python3
"""C06 fixtures — torch RMSNorm fp32 parity corpus (specs/06-qwnorm.md gate):
for rows in {1, 512, 4096}, hidden 5120:
  rms_N{R}_x.bin     f16 [R,H]        rms_N{R}_g.bin  f16 [H] gamma
  rms_N{R}_y.bin     f32 ref: x·rsqrt(mean(x²)+1e-6)·(1+γ) in fp32, cast f16
#                      (OFFICIAL Qwen3_5RMSNorm PLUS-ONE law — γ bins hold RAW
#                      checkpoint weights w≈0; raw law stays for GDN gated-norm)
  rms_N{R}_res_in.bin  f16 residual   rms_N{R}_res_y.bin f32: f16(res+x) then norm
  rms_N{R}_rs.bin      f32 row-scale  (rstd per row, prologue-scale variant)
Adversarial high-variance row (row 0) exercises the fp32-accum overflow law.
γ = 1 + small noise so the weight actually matters in parity."""
import numpy as np, pathlib, math, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "quantize"))
from pack import fwht32_ortho

H = 5120
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(11)

# N=2 appended last: the rng stream is consumed in order, so [1,512,4096]
# fixtures stay bit-identical; N=2 feeds testHeadNormNotHere's arena (R=2).
for R in [1, 512, 4096, 2]:
    x = rng.standard_normal((R, H)).astype(np.float32)
    if R >= 1:
        x[0] *= 60.0                     # adversarial high-variance row
    g = (rng.standard_normal(H) * 0.05).astype(np.float32)  # RAW w (plus-one in the ref)
    res = rng.standard_normal((R, H)).astype(np.float32) * 0.5
    x16 = x.astype(np.float16)
    g16 = g.astype(np.float16)
    res16 = res.astype(np.float16)
    xf = x16.astype(np.float32)
    gf = g16.astype(np.float32)
    resf = res16.astype(np.float32)
    # variant 1: norm of x + fused FWHT-32 (HQ4_G32_M5 spec §6.1)
    ms = (xf * xf).mean(-1, keepdims=True)
    gf1 = (gf + 1.0).astype(np.float32)
    y1 = fwht32_ortho((xf * (1.0 / np.sqrt(ms + 1e-6)) * gf1).astype(np.float32)).astype(np.float16)
    # variant 2: residual first (f16 add rounding!), then norm + fused FWHT-32
    res2 = (resf + xf).astype(np.float16)
    res2f = res2.astype(np.float32)
    ms2 = (res2f * res2f).mean(-1, keepdims=True)
    y2 = fwht32_ortho((res2f * (1.0 / np.sqrt(ms2 + 1e-6)) * gf1).astype(np.float32)).astype(np.float16)
    # variant 3: row scales = rstd of variant-2-style input (x directly)
    rstd = (1.0 / np.sqrt(ms + 1e-6))[..., 0].astype(np.float32)
    out.joinpath(f"rms_N{R}_x.bin").write_bytes(x16.view(np.uint16).tobytes())
    out.joinpath(f"rms_N{R}_g.bin").write_bytes(g16.view(np.uint16).tobytes())
    out.joinpath(f"rms_N{R}_y.bin").write_bytes(y1.astype(np.float32).view(np.uint8).tobytes())
    out.joinpath(f"rms_N{R}_res_in.bin").write_bytes(res16.view(np.uint16).tobytes())
    out.joinpath(f"rms_N{R}_res_y.bin").write_bytes(y2.astype(np.float32).view(np.uint8).tobytes())
    out.joinpath(f"rms_N{R}_res_out.bin").write_bytes(res2.view(np.uint16).tobytes())
    out.joinpath(f"rms_N{R}_rs.bin").write_bytes(rstd.view(np.uint8).tobytes())
    print(f"R={R}: |y1|max {np.abs(y1.astype(float)).max():.4f} rstd[0] {rstd[0]:.3e}")
print("norm fixtures done")
