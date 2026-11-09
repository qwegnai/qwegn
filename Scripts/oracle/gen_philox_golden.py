#!/usr/bin/env python3
"""Emit ONLY head_philox_golden.bin (the Philox-4x32-10 u-stream golden).
Same law as gen_head_fixtures.py's tail — kept as a tiny standalone so the
golden can be refreshed without regenerating the 700 MB head planes."""
import numpy as np, pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
M32 = (1 << 32) - 1
D1, D2 = 0xD2511F53, 0xCD9E8D57
SEED = 0x0123456789ABCDEF

def philox4(c0, c1, c2, c3, seed):
    k0, k1 = (seed >> 32) & M32, seed & M32
    for _ in range(10):
        p0, p1 = c0 * D1, c2 * D2
        hi0, lo0 = (p0 >> 32), (p0 & M32)
        hi1, lo1 = (p1 >> 32), (p1 & M32)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & M32, lo1, (hi0 ^ c3 ^ k1) & M32, lo0
        k0 = (k0 + D2) & M32
        k1 = (k1 + D1) & M32
    return c0, c1, c2, c3

u = np.empty(64, np.float32)
for n in range(64):
    c = philox4(n & M32, (n >> 32) & M32, 0, 0, SEED)
    u[n] = np.float32(c[0]) * np.float32(2.0 ** -32)
assert u.min() >= 0.0 and u.max() < 1.0
(out / "head_philox_golden.bin").write_bytes(u.tobytes())
print("philox golden u[0..3]:", u[:4])
