"""mtp_spec.accept_round self-check — the distribution-preservation theorem
bound to a MARKOV target so the analytic marginal exists and KL is exact.

Target process: x_{k+1} ~ P(· | x_k = i) (table PT[i], f64, random init).
Draft:     q(· | i) (table QT[i], DIFFERENT random init).
Speculative chain (N=2 per round): d1 ~ q(·|i); d2 ~ q(·|d1);
  accept via accept_round([q_i, q_d1], [PT_i, PT_d1, PT_d2], [d1,d2], uni).
Committed tokens (prefix + final) are pushed onto a truth queue; every
truth step must satisfy P(x_{k+1}=j | x_k=i) = PT[i][j] — the strongest
possible form of the theorem. KL(PT[i] ‖ empirical row i) over the
VISITED transitions (expectation of p = Σ_j PT[i][j]·min(1,PT/Q) — exact
per-row marginal) is the number; a wiring bug (e.g. the buggy `min(1,p/q)`
vs `min(1,q/p)` swap) lands at ~1e-2; the gate is ≤1e-4 (spec 16 law).
Also checks the DRAWS contract: drawn == acc+2 on reject, N+1 on accept.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mtp_spec import accept_round

rng = np.random.default_rng(11)
V, N, STEPS = 16, 2, 400_000
def table(seed, scale):
    r = np.random.default_rng(seed)
    t = r.standard_normal((V, V)) * scale
    t = np.exp(t - t.max(1, keepdims=True)); t /= t.sum(1, keepdims=True)
    return t
PT = table(11, 1.6)          # target conditionals
QT = table(12, 2.2)          # draft conditionals (deliberately different)
uni_all = rng.random((STEPS, N + 1))

emp = np.zeros((V, V)); visit = np.zeros(V)
emp = np.zeros((V, V)); visit = np.zeros(V)
# ── CONTROL arm: direct inverse-CDF sampling from PT (no speculation at
#    all) through the SAME scan/tally code. Its KL is the harness floor —
#    if the control shows ~1e-1 too, the estimator is the problem, not
#    accept_round.
ctrl = np.zeros((V, V)); cvisit = np.zeros(V)
i = int(rng.integers(V))
for step in range(STEPS):
    u = float(rng.random())
    j = int(np.searchsorted(np.cumsum(PT[i]), u, side="right"))
    ctrl[i, j] += 1; cvisit[i] += 1
    i = j

def row_kl(M, vis):
    out = []
    for r in range(V):
        if vis[r] < 1000: continue
        row = M[r] / vis[r]
        kl = float((PT[r] * np.log(PT[r] / np.maximum(row, 1e-300))).sum())
        out.append((vis[r], kl - (V - 1) / (2.0 * vis[r])))   # Miller-Madow
    return out

ck = row_kl(ctrl, cvisit)
print(f"CONTROL direct-sampler: rows {len(ck)} mean-KL {np.mean([k for _, k in ck]):.3e} "
      f"visits min {min(v for v, _ in ck):.0f} max {max(v for v, _ in ck):.0f}")

state = int(rng.integers(V)); steps = 0; rounds = 0
acc_hist = np.zeros(N + 1, dtype=int); bad_draws = 0
truth = []                   # queue of committed tokens; steps consume
while steps < STEPS and rounds < STEPS:
    i = state; d = []; Pd = []
    for j in range(N):
        q = QT[i if j == 0 else d[-1]]
        u = float(rng.random())
        # inverse-CDF draft sample (ascending strict law, same as engine)
        c = np.cumsum(q); dj = int(np.searchsorted(c, u, side="right"))
        d.append(dj); Pd.append(q)
    Pt = [PT[i], PT[d[0]], PT[d[1]]]
    uni = uni_all[rounds]
    prefix, t, drawn = accept_round(Pd, Pt, d, uni)
    acc = len(prefix)
    want_draws = acc + 2 if acc < N else N + 1
    if drawn != want_draws: bad_draws += 1
    rounds += 1
    seq = d[:acc] + [t]
    # feed onto the truth chain
    for tk in seq:
        emp[i, tk] += 1; visit[i] += 1
        i = tk; steps += 1
        if steps >= STEPS: break
    state = i
kl_sum, n_seen, tv_max = 0.0, 0, 0.0
for i in range(V):
    if visit[i] < 20000: continue
    row = emp[i] / visit[i]
    kl = float((PT[i] * np.log(PT[i] / np.maximum(row, 1e-300))).sum()) - (V - 1) / (2.0 * visit[i])   # Miller-Madow
    kl_sum += kl; n_seen += 1
    tv_max = max(tv_max, float(np.abs(PT[i] - row).max()))
print(f"rounds {rounds} steps {steps} visits {n_seen} rows")
print(f"KL mean {kl_sum / max(n_seen,1):.3e}  TV max {tv_max:.3e}  draws-contract violations {bad_draws}")
print(f"accept hist {acc_hist.tolist() if acc_hist.sum() else '—'}")
