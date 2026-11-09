"""C16 reject-sampling speculation reference (Leviathan et al. '23 — the
distribution-preservation law the engine MUST reproduce stream-wise).

DRAWS CONTRACT (so Swift with Philox-4x32-10 binds the same stream): one
uniform per decision, consumed in order, f64 in [0,1). A round of N drafts
CONSUMES accepted+1 uniforms (a reject stops the accept-scan; the
resample/bonus draw is the one more). Uniform stream = one Philox stream
seeded per run; NO per-position reseeding.

SHAPE LAW: p_target has N+1 entries — p_target[i] is the TARGET's
distribution for the position d_{i+1} was drafted at (the verify forward
over [t_s, d_1..d_N] yields marginals for s+1..s+N+1); p_target[N] is the
bonus distribution (only consumed when all N accepted). p_draft has N
entries — the DRAFT's distribution that produced each d_i.

ACCEPT: keep draft d_i iff r_i < min(1, p_target[i-1][d_i] / p_draft[i-1][d_i]).
REJECT at i: resample t from norm(max(0, p_target[i-1] − p_draft[i-1])) —
committed; positions i+1..N untouched (their KV rolls back — C20 ring).
ALL ACCEPT: t = sample from p_target[N] (the bonus token).
Committed after a round = prefix d_1..d_{i-1} + t (t ALWAYS commits ⇒
progress ≥ 1 token/round).

THEOREM (what the 100k-draw gate exercises): if drafts come from q and
verification from p, the committed-token marginal equals p EXACTLY —
rounding-only KL vs an independent p-sampler. A wiring bug shows as KL
≥ 1e-3; the gate is ≤ 1e-4 over 100k draws, pinned seeds (spec 16).
"""
import math
import numpy as np


def accept_round(p_draft, p_target, token_ids, uniforms):
    """One speculative round, B=1. p_draft: list of N f64 arrays (length V);
    p_target: list of N+1 f64 arrays (length V). token_ids: drafted ids [N].
    uniforms: indexable f64 sequence (consumed from position 0).
    Returns (accepted_prefix_ids, final_token, n_drawn)."""
    n = len(token_ids)
    assert len(p_draft) == n and len(p_target) == n + 1, "shape law: N drafts ⇒ N+1 targets"
    drawn = 0
    for i in range(n):
        d = int(token_ids[i])
        ratio = p_target[i][d] / p_draft[i][d] if p_draft[i][d] > 0 else math.inf
        r = uniforms[drawn]; drawn += 1
        if r < min(1.0, ratio):
            continue                                # accept d_i
        res = np.maximum(p_target[i] - p_draft[i], 0.0)
        s = res.sum()
        assert s > 0.0, "zero residual mass — q≥p pointwise everywhere (impossible unless q==p)"
        res = res / s
        t = sample_from(res, uniforms[drawn]); drawn += 1
        return list(token_ids[:i]), t, drawn
    t = sample_from(p_target[n], uniforms[drawn]); drawn += 1
    return list(token_ids), t, drawn


def _scan(p, u):
    """Deterministic inverse-CDF sample, ASCENDING-cumsum strict law:
    smallest idx with cumsum > u. Swift must use the same scan order
    (f64 accumulate left→right, strict <)."""
    c = 0.0
    for i in range(len(p)):
        c += p[i]
        if u < c:
            return i
    return len(p) - 1        # u within fp tail of 1.0


def sample_from(p, u):
    return _scan(np.asarray(p, dtype=np.float64), float(u))
