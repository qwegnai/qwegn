#!/usr/bin/env python3
"""C15 head fixtures (spec 15: split-K lm_head GEMV + sampler parity).

Full lm_head is [248320, 5120] (2.54 GB BF16). Parity fixtures use N=61440
(a quarter of the vocab; rows are structurally identical — the FULL-vocab
bench runs synthetic planes in-process, the parity TEST binds the math).

SAMPLER CONTRACT (spec: "distribution parity vs torch multinomial ≤1e-4 —
a violation is a bug in either sampler, never noise"): the reference is
generated with TORCH ops (F.softmax, torch.topk), not a numpy stand-in —
the pipeline is softmax(logits/T) → top-k 64 → top-p 0.9 nucleus → renorm.
Tie law pinned (boundary values are unique in these fixtures — asserted —
so the clauses below are provable no-ops here but the kernel still ships
them for adversarial ties): keep iff probs_i > kth, or == kth with
ascending-index rank < slack; nucleus = shortest desc-order (ties ascending
index) prefix whose cumsum ≥ p, i.e. count(cumsum < p) + 1.

Files:
  head_x.bin          f16 [5120]   post-final-norm activation
  head_x_gsum.bin     f32 [40]     Σ per 128-group (q4 min term)
  head_w.bin          q4-g128 nibbles [N, 40, 64] + sc/mn (bit-identical
                      packer — asym g128 over f16-rounded weights)
  head_w16.bin        f16 [N, 5120] the f16-rounded W (bf16-default-quality
                      anchor path raced by bench; f16 escape-hatch GEMV)
  head_y.bin          f32 [N]      y = W_deq·x fp32 (q4 parity anchor)
  head_y16.bin        f32 [N]      y = W16·x fp32 (f16-path anchor)
  head_greedy.bin     u32          argmax(head_y)
  head_adv_logits.bin f32 [N]      adversarial ±80 (fp32 overflow safety)
  head_adv_greedy.bin u32          argmax of the adversarial logits
  head_dist_logits.bin f32 [N]     peaked serving logits (real nucleus)
  head_dist_p.bin     f32 [N]      torch reference distribution (dense)
  head_dist_idx.bin   i32 [S]      survivor ids ascending (S ≈ 11)
  head_dist_probs.bin f32 [S]      survivor renormalized p (parallel)"""
import numpy as np, pathlib, sys, torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "quantize"))
from pack import quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

N, K = 61440, 5120
out = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(15)
torch.manual_seed(15)

def q4g(W, name):
    blocks = quant_hq4_g32_m5(W)
    (out / f"{name}.bin").write_bytes(blocks.tobytes())
    return dequant_hq4_g32_m5_rot(blocks, f16_round=True)

# ── GEMV fixtures ──
x = rng.standard_normal(K).astype(np.float32).astype(np.float16)
(out / "head_x.bin").write_bytes(x.view(np.uint16).tobytes())
(out / "head_x_gsum.bin").write_bytes(x.astype(np.float32).reshape(K // 128, 128).sum(1).tobytes())
W = rng.standard_normal((N, K)).astype(np.float32) * (5120 ** -0.5)   # lm_head init scale
wq = q4g(W, "head_w")
w16 = W.astype(np.float16)
(out / "head_w16.bin").write_bytes(w16.view(np.uint16).tobytes())
xf = x.astype(np.float32)
y = (wq @ xf).astype(np.float32)
y16 = (w16.astype(np.float32) @ xf).astype(np.float32)
(out / "head_y.bin").write_bytes(y.tobytes())
(out / "head_y16.bin").write_bytes(y16.tobytes())
greedy = np.argmax(y).astype(np.uint32)
(out / "head_greedy.bin").write_bytes(greedy.tobytes())
print(f"y: |max| {np.abs(y).max():.3f} greedy {greedy}  |y16-y|max {np.abs(y16 - y).max():.2e}")

# ── adversarial ±80 (fp32 softmax overflow safety; argmax at extremes) ──
adv = rng.random(N).astype(np.float32) * 4.0 - 80.0          # bulk near −80
adv[rng.choice(N, size=8, replace=False)] = 80.0
adv[3] = 80.0
adv[7] = 79.9998
(out / "head_adv_logits.bin").write_bytes(adv.tobytes())
(out / "head_adv_greedy.bin").write_bytes(np.uint32(np.argmax(adv)).tobytes())

# ── torch ground truth for the sampler contract ──
def stable_top_p(probs: torch.Tensor, topk: int, topp: float):
    """torch-exact top-k → top-p nucleus → renorm; tie law documented in the
    module docstring. Returns (idx ascending, renormalized probs)."""
    v, _ = torch.topk(probs, topk)
    kth = v[-1]
    keep_k = probs > kth
    slack_k = topk - int(keep_k.sum())
    if slack_k > 0:
        eq = torch.nonzero(probs == kth, as_tuple=False).squeeze(-1)
        keep_k[eq[:slack_k]] = True
    idx_k = torch.nonzero(keep_k, as_tuple=False).squeeze(-1)
    pk = probs[idx_k]
    order = torch.argsort(-pk, stable=True)          # desc, ties asc index
    sv = pk[order]
    cum = torch.cumsum(sv, 0)
    need = int((cum < topp).sum()) + 1               # pre-< topp, +1
    q = sv[need - 1]
    keep = idx_k[probs[idx_k] > q]                   # strict > q among top-k
    slack_q = need - int(keep.numel())
    if slack_q > 0:
        eq = idx_k[probs[idx_k] == q]                # ascending index
        keep = torch.cat([keep, eq[:slack_q]])
    keep, _ = torch.sort(keep)
    p = probs[keep] / probs[keep].sum()
    # boundary uniqueness ASSERT (fixture law — tie clauses provable no-ops)
    bound = probs[keep]
    assert torch.unique(bound).numel() == bound.numel(), "boundary tie in survivors"
    assert probs[keep].numel() == need, "nucleus size mismatch"
    return keep.numpy().astype(np.int32), p.numpy().astype(np.float32)

logits = np.zeros(N, np.float32)
spike_idx = rng.choice(N, size=60, replace=False)
logits[spike_idx] = rng.uniform(12.0, 20.0, size=60).astype(np.float32)
logits += rng.standard_normal(N).astype(np.float32) * 0.05      # thin floor
logits = logits.astype(np.float32)
probs_t = F.softmax(torch.from_numpy(logits) / 0.7, dim=-1)
idx, p = stable_top_p(probs_t, 64, 0.9)
(out / "head_dist_logits.bin").write_bytes(logits.tobytes())
(out / "head_dist_idx.bin").write_bytes(idx.tobytes())
(out / "head_dist_probs.bin").write_bytes(p.tobytes())
full = np.zeros(N, np.float32); full[idx] = p
(out / "head_dist_p.bin").write_bytes(full.tobytes())
print(f"dist survivors {len(idx)}, k_eff≈{1 / np.sum(p ** 2):.1f}, first {idx[:4]}")

# ── Philox-4x32-10 golden (the C15 RNG law: Swift sampler AND the MSL
#    GPU path must both reproduce THIS stream bit-exactly — the fixture is
#    the canonical spec algorithm, so it pins BOTH ports, not one's copy).
#    Stream: u(n) = philox4(seed, ctr=(n, n≫32, 0, 0)).x · 2⁻³²;
#    D.1=0xD2511F53 D.2=0xCD9E8D57 W=0x9E3779B9 (canonical constants; the
#    Metal port folds W into k0+=D2, k1+=D1 per round, h = high word of the
#    64-bit product).
M32 = (1 << 32) - 1
D1, D2, WCONST = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9
SEED = 0x0123456789ABCDEF

def philox4(c0, c1, c2, c3, seed):
    # canonical round: hi0,lo0 = D1*c0; hi1,lo1 = D2*c2
    #   out = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0);  k0 += D2; k1 += D1
    # (identical law to QWPhilox.swift and the C15 kernel port)
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
print("c15 fixtures done")
