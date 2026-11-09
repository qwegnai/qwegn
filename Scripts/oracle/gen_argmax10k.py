#!/usr/bin/env python3
"""C15 argmax golden: 10k positions × N=8192 logits + u32 argmax per row
(spec-15 gate: "T=0 argmax parity with torch == 100% on 10k golden
positions"). Covers: random sign/magnitude spread, crafted TIES (spec law:
ties → LOWEST id, numpy/torch argmax first-occurrence), all-negative rows
(sortable's ~b branch), ±80 adversarial rails, maxima at tile-boundary
indices (255/256/257, N−2/N−1 — the fold's geometry-critical slots),
id-0 ties, near-ties decided by 1 ulp, subnormal-scale rows.

LAW (why NO ±0.0 pair): the engine consumes GEMV output; exact ±0.0 logits
can never both appear (and the key law orders 0.0 > −0.0 while argmax
treats them equal) — a fixture with both would bind a contract the engine
does not owe. Zero-rows carry a positive spike instead.

Files:
  head_amax10k_logits.bin  f32 [10000, 8192]  (320 MB — bench/suite time)
  head_amax10k_ids.bin     u32 [10000]        numpy argmax per row
"""
import numpy as np, pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
rng = np.random.default_rng(151)
P, N = 10_000, 8192
logits = (rng.standard_normal((P, N)) * 4.0).astype(np.float32)

def put(p, i, v): logits[p, int(i)] = v

for p in range(P):
    kind = p % 10
    if kind == 0:            # clean interior max
        put(p, rng.integers(0, N), 40.0)
    elif kind == 1:          # crafted tie: two equal maxima, lowest id wins
        a, b = sorted(rng.integers(0, N, size=2).tolist())
        put(p, a, 45.0)
        put(p, b, 45.0)
    elif kind == 2:          # all-negative rows (sortable's ~b branch)
        logits[p] = -40.0
        put(p, rng.integers(0, N), -12.0)
    elif kind == 3:          # ±80 adversarial rails
        logits[p] = -80.0
        put(p, rng.integers(0, N), 80.0)
    elif kind == 4:          # max at tile boundaries
        put(p, [255, 256, 257, N - 2, N - 1][p % 5], 50.0)
    elif kind == 5:          # zero row + one positive spike
        logits[p] = 0.0
        put(p, rng.integers(0, N), 3.0)
    elif kind == 6:          # near-tie decided by 1 ulp (higher value wins)
        v = 33.0
        a, b = sorted(rng.integers(0, N, size=2).tolist())
        if a == b: b = (b + 1) % N
        put(p, a, v)                                  # the winner
        put(p, b, np.nextafter(v, np.float32(-np.inf)))
    elif kind == 7:          # duplicate maxima incl. id 0 (id-0 tie law)
        logits[p, 0] = 60.0
        put(p, rng.integers(1, N), 60.0)
    elif kind == 8:          # wide-range mixed (default noise dominates)
        pass
    else:                    # subnormal-scale row + tiny max
        logits[p] *= np.float32(1e-38)
        put(p, rng.integers(0, N), np.float32(5e-38))

ids = np.argmax(logits, axis=1).astype(np.uint32)
# numpy tie law check (documented dependency): argmax = first occurrence
assert int(np.argmax(np.array([1.0, 7.0, 7.0]))) == 1
(out / "head_amax10k_logits.bin").write_bytes(logits.tobytes())
(out / "head_amax10k_ids.bin").write_bytes(ids.tobytes())
ties = sum(1 for p in range(P) if p % 10 == 1)
print(f"10k argmax golden done; tie rows {ties}, ids[:6] {ids[:6]}")
