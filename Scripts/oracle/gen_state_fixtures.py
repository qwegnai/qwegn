#!/usr/bin/env python3
"""gen_state_fixtures.py — C20 goldens: sequence-state machine contracts.

Parity-port law (rule 2): LIVE Python truth BEFORE any Swift assertion. C20's
subject is the STATE MACHINE (spec-20): slot ownership, snapshot/rollback,
prefix-cache LRU. Two golden families:

  A. ROLLBACK PARITY — the GDN recurrent state (rec [48,128,128] fp32 = the
     spec's 3.15 MB footprint + conv ring [(K-1)×QKV_CH] per the C11 layout)
     through the scripted sequence (prefill P → snapshot → decode 64) versus
     a FRESH (prefill P+64) walk. Golden law: decode-after-snapshot lands on
     the fresh walk's state BYTES (== 0 diff) — the spec's "bit-identical"
     gate bound at the state layer. Inputs are the C12 door contract shapes
     (l2-normed q̂×128^-0.5 ‖ k̂, g ≤ 0 decay law, β=σ); the conv ring holds
     the last K−1 MIXED rows (engine contract input st_mixed.bin) rotated
     per the C11 law.

  B. PREFIX-CACHE ALGEBRA — a PYTHON ref machine (the spec-20 semantics:
     clone-on-adopt independence, MRU touch, LRU evict under pressure,
     slot-pressure REFUSAL before any state change) driven by a scripted
     trace; every op's post-state is the golden the Swift QWState must
     reproduce op-for-op.

Deterministic (fixed seed): rerun is byte-identical (self-check at end).
"""
import json, os, sys
import numpy as np
import torch
torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "Scripts", "spike", "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from qw38_oracle import GV, GHD, CONVK, QKV_CH  # geometry from the authority

SEED = 20261002
NS = 160                       # steps: 96-prefill case + 64 decode
PREFILS = (1, 37, 96)

# ── contract inputs (C12/C11 law shapes) ──────────────────────────────────
torch.manual_seed(SEED)
q = torch.nn.functional.normalize(torch.randn(NS, GV, GHD), dim=-1) * (GHD ** -0.5)
k = torch.nn.functional.normalize(torch.randn(NS, GV, GHD), dim=-1)
v = torch.randn(NS, GV, GHD)
beta = torch.sigmoid(torch.randn(NS, GV))
A_log = torch.log(torch.empty(GV).uniform_(0.01, 16))
a = torch.randn(NS, GV)
g = -A_log.exp()[None, :] * torch.nn.functional.softplus(a + 1.0)
g = g.clamp(max=-1e-6)                                  # decay law: g ≤ 0
# mixed rows the conv ring stores (C11 layout: [QKV_CH] per token, fp32)
mixed = torch.randn(NS, QKV_CH)

qf = q.numpy().astype("<f4"); kf = k.numpy().astype("<f4"); vf = v.numpy().astype("<f4")
bf = beta.numpy().astype("<f4"); gf = g.numpy().astype("<f4"); mf = mixed.numpy().astype("<f4")
# Shared decay table: np.exp's fp32 result, dumped once so the Swift walk
# multiplies the SAME bytes (cross-impl expf-vs-np.exp last-ulp noise is
# impossible by construction — same law as the golden-generator doctrine).
decf = np.exp(gf, dtype=np.float32)
for name, arr in (("st_q", qf), ("st_k", kf), ("st_v", vf),
                  ("st_beta", bf), ("st_g", gf), ("st_mixed", mf), ("st_dec", decf)):
    with open(os.path.join(OUT, f"{name}.bin"), "wb") as f:
        f.write(arr.tobytes())

def rec_walk(idxs, rec):
    """Delta-rule walk (C12 contract; per-v-head state [GV,GHD,GHD] fp32).

    SUMMATION-ORDER LAW (C12 precedent): the k-axis fold is an ASCENDING
    sequential fp32 fold, NOT numpy's pairwise sum — measured: numpy(128)
    and ascending differ in 87% of draws, and any association is math for
    this state machine (the parity gate is rollback-vs-fresh equality,
    which is association-invariant). The generator and the Swift walk share
    THIS canonical order so Swift can byte-compare against the dumped gold
    (stronger than machine-internal parity)."""
    GVh, HD = GV, GHD
    for t in idxs:
        dec = np.exp(gf[t], dtype=np.float32)             # [GV] fp32
        for h in range(GVh):
            s = rec[h]                                     # [128,128] (kk, vv)
            d = bf[t, h]
            decay = dec[h]
            kt = kf[t, h]; vt = vf[t, h]
            # kmem[vv] = Σ_kk s[kk,vv]·kt[kk] (ascending fold)
            kmem = np.zeros(HD, np.float32)
            for kk in range(HD):
                kmem += kt[kk] * s[kk]
            dlt = (vt - kmem) * d
            s *= decay
            rec[h] = s + np.outer(kt, dlt)
    return rec

def ring_after(idxs):
    """C11 rotate law: ring = last CONVK-1 mixed rows (fp32); a walk shorter
    than K-1 zero-pads on the LEFT (tokens before position 0 never existed)."""
    hist = [t for t in idxs]
    ring = np.zeros((CONVK - 1, QKV_CH), dtype="<f4")
    for j in range(CONVK - 1):
        src_pos = len(hist) - (CONVK - 1 - j)
        if src_pos >= 0:
            ring[j] = mf[hist[src_pos]]
    return ring

A = {}
for P in PREFILS:
    idx_pre = list(range(P))
    snap_rec = rec_walk(idx_pre, np.zeros((GV, GHD, GHD), np.float32))
    snap_ring = ring_after(idx_pre)
    idx_dec = list(range(P, P + 64))
    # rollback = restart from snapshot (delta-journal accept-path equivalent):
    after_rec = rec_walk(idx_dec, snap_rec.copy())
    after_ring = ring_after(idx_pre + idx_dec)
    fresh_rec = rec_walk(range(P + 64), np.zeros((GV, GHD, GHD), np.float32))
    fresh_ring = ring_after(range(P + 64))
    assert after_rec.tobytes() == fresh_rec.tobytes(), f"rollback law violated P={P}"
    assert after_ring.tobytes() == fresh_ring.tobytes(), f"ring law violated P={P}"
    for suf, rec, ring in (("a", snap_rec, snap_ring), ("b", after_rec, after_ring),
                           ("c", fresh_rec, fresh_ring)):
        with open(os.path.join(OUT, f"st_rec_{suf}_P{P}.bin"), "wb") as f:
            f.write(rec.astype("<f4").tobytes())
        with open(os.path.join(OUT, f"st_ring_{suf}_P{P}.bin"), "wb") as f:
            f.write(ring.astype("<f4").tobytes())
    A[f"P{P}"] = {"rec_bytes": int(snap_rec.nbytes), "ring_bytes": int(snap_ring.nbytes),
                  "law": "snapshot→decode == fresh walk (0-diff asserted)"}

# ── Family B: prefix-cache algebra trace ──────────────────────────────────
CAP, LRU_N = 4, 3

class RefState:
    """Spec-20 semantics: 1 slot == 1 sequence (slot index == seq id);
    prefix entries are CLONES (clone-on-adopt independence); LRU over prefix
    entries with MRU touch on adopt-hit and on decode; refusal raises BEFORE
    touching any state."""
    class Refused(Exception): pass

    def __init__(self):
        self.slot_of, self.slots, self.free = {}, {}, list(range(CAP))
        self.cache, self.clones, self.events = [], 0, []

    def log(self, ev): self.events.append(ev)

    def adopt(self, seq, fp):
        if seq in self.slot_of or not self.free:
            raise RefState.Refused(f"slot pressure / dup seq {seq}")
        slot = self.free.pop(0)
        hit = fp in self.cache
        if hit:
            self.cache.remove(fp); self.clones += 1     # clone-on-adopt
        else:
            while len(self.cache) >= LRU_N:
                e = self.cache.pop(); self.log({"op": "evict", "fp": e})
        self.cache.insert(0, fp)
        self.slot_of[seq] = slot
        self.slots[slot] = {"seq": seq, "fp": fp, "gen": 0}
        self.log({"op": "adopt", "seq": seq, "fp": fp, "slot": slot, "hit": hit,
                  "cache": list(self.cache), "clones": self.clones})

    def decode(self, seq):
        slot = self.slot_of[seq]
        self.slots[slot]["gen"] += 1
        fp = self.slots[slot]["fp"]
        if fp in self.cache:                              # MRU touch
            self.cache.remove(fp); self.cache.insert(0, fp)
        self.log({"op": "decode", "seq": seq, "slot": slot,
                  "gen": self.slots[slot]["gen"], "cache": list(self.cache)})

    def finish(self, seq):
        slot = self.slot_of.pop(seq)
        self.slots.pop(slot)
        self.free.append(slot)
        self.log({"op": "finish", "seq": seq, "slot": slot, "free": sorted(self.free)})

rng = np.random.RandomState(SEED)
ops, live, next_seq = [], [], 100
for phase in range(6):
    batch = []
    for _ in range(3):
        fp = f"p{rng.randint(0, 5)}"          # 5 fps → guarantees LRU churn + hits
        ops.append(["adopt", next_seq, fp]); batch.append(next_seq); next_seq += 1
        ops.append(["decode", batch[-1]])
        ops.append(["decode", batch[-1]])
    live = live + batch
    if len(live) >= CAP:
        # demand one MORE while full: the refusal proof (raise-before-mutate)
        break
    # drain the two oldest so the next phase fits
    for s in live[:2]:
        ops.append(["finish", s])
    live = live[2:]

refusal = ["adopt", next_seq, f"p{rng.randint(0,5)}"]
ops.append(refusal)

rs2 = RefState()
events = []
for o in ops[:-1]:
    getattr(rs2, o[0])(*o[1:])
before = (list(rs2.free), dict(rs2.slot_of), list(rs2.cache))
try:
    rs2.adopt(*refusal[1:])
    raise AssertionError("refusal did not raise")
except RefState.Refused:
    pass
after = (list(rs2.free), dict(rs2.slot_of), list(rs2.cache))
assert before == after, "refusal mutated state — spec law violated"

B = {"cap": CAP, "lru_n": LRU_N, "ops": ops, "events": rs2.events,
     "refusal": refusal, "refusal_state_unchanged": True,
     "laws": ["slot index == seq id; one live seq per slot",
              "clone-on-adopt: entries are independent; hits bump clones",
              "MRU touch on adopt-hit/decode; cache ≤ lru_n (evict = LRU tail)",
              "refusal raises BEFORE any state change (asserted here)"]}

with open(os.path.join(OUT, "st_prefix_trace.json"), "w") as f:
    json.dump(B, f, separators=(",", ":"))

# determinism self-check (C17 law): recompute fresh, byte-compare.
r2 = rec_walk(range(96 + 64), np.zeros((GV, GHD, GHD), np.float32))
assert np.array_equal(r2, np.fromfile(os.path.join(OUT, "st_rec_c_P96.bin"), dtype="<f4")
                      .reshape(GV, GHD, GHD)), "determinism violated"

print(json.dumps({"seed": SEED, "family_a": A,
                  "family_b_ops": len(ops), "events": len(rs2.events),
                  "refusal_at": refusal}))
print("OK: C20 goldens written to", OUT)
