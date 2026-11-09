#!/usr/bin/env python3
"""gen_gdn_prefill_fixtures.py — C13 GDN chunked-prefill parity fixtures.

Golden: qw38_oracle.py recurrent GatedDeltaNet (T0.3) — the naive walk IS
the golden (spec-13: "Parity is the admission gate: state + output vs naive
C12 path"; the torch fp32 walk is the engine-naive twin). THE CHUNKED
FORM (validated here, generator-side, BEFORE any dump — never trust the
sketch):

  per chunk t0..t0+L−1, per v-head (q̂,k̂ normed upstream; state S fp32):
    γ_t = Σ_{j≤t} g_j                      (intra-chunk cumsum, fp32)
    K̂_e = k̂ ⊙ e^γ                          (row-scaled)
    M   = I + tril(D_β·(K̂_e·Kᵀ), −1)       K=RAW k̂ rows; unit-diag
    solve M·Ṽ = v̂   (forward substitution, fp32 — spec: fp32 solve
                      mandatory even if GEMMs run f16)
    fold Ṽ' = Ṽ·e^{γL−γ} − (tril(Q̂·Kᵀ,−1)·Ṽ)·e^{γL−γ}   (intra-out into Ṽ')
    S  ← S·e^{γL} + (K̂⊙e^{γL−γ})ᵀ · Ṽ'ᵀ     (one GEMM: C = Σ_t k̂·wᵀ)
    O  = (Q̂·S_old)·e^γ + S_new·…            → done as: O_raw = e^γ·(Q̂·S_old)
                                               + tril-intra above… the K3
                                               kernel folds exactly:
                                               O_raw = e^γ·(Q̂·S_old) + G̃·Ṽ
    then GatedRMSNorm per (t,head) row with z (fp32 gate), f16 store.

Adversarial coverage (spec Risks: "β-saturated heads → ill-conditioned
I−B̃K̃ᵀ"): a v-head slice with β∈(0.94,1.0) must survive the whole walk.

Dump (torch-naive references, engine layout — SAME conventions as C12):
  y [T][10240] f16 (q̂16|k̂16|v48 post-l2norm), z [T][6144] f16,
  beta/glog [T][48] f32, state0 [48][128][128] f32, w 128 f16,
  o_t{X} [48][128] f32 for boundary token indices, o16_t{X} f16,
  state [48][128][128] f32 (walk FINAL, adjacency @1e-4 at 1K).
T ∈ {130 (ragged: 2×64 + 2 tail), 1024 (spec parity @1K)}. Chunk length is
a KERNEL parameter (L=64 shipped first; the fixtures are chunk-agnostic —
any L partitions the same dump).
"""
import pathlib
import sys
import torch
def rot32(t):
    """Engine output law: o is stored in the H_32-rotated basis (orthonormal FWHT per 32-block)."""
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < 32:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / 32 ** 0.5
    dt = t.dtype
    return ((t.float().reshape(-1, 32) @ H).reshape(t.shape)).to(dt)
import torch.nn.functional as F

torch.set_grad_enabled(False)
OUT = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
OUT.mkdir(parents=True, exist_ok=True)
D, NH, NV = 128, 16, 48
EPS, SCALE = 1e-6, D ** -0.5


def l2norm(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + EPS)


def rmsnorm_gated(o, w, z):
    x = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * w
    return x * F.silu(z.float())


def save(t, name):
    t = t.detach().cpu().contiguous()
    (OUT / name).write_bytes(bytes(t.view(torch.uint8).flatten()))
    return t.numel() * t.element_size()


def step(S, qh, kh, v, beta, g):
    S = S * torch.exp(g)
    kv = (S * kh[:, None]).sum(dim=0)
    delta = (v - kv) * beta
    S = S + kh[:, None] * delta[None, :]
    o = (S * qh[:, None]).sum(dim=0)
    return S, o


def naive_walk(S0, q, k, v, z, beta, glog, w, keep):
    """torch fp32 naive walk batched over heads — the oracle's exact loop
    form (per-t, all 48 heads). Returns final state + o rows @ keep."""
    T = q.shape[0]
    qh = q[:, [h // 3 for h in range(NV)]]              # [T,48,128] share law
    kh = k[:, [h // 3 for h in range(NV)]]
    S = S0.clone()
    o_rows = {}
    for t in range(T):
        S = S * torch.exp(glog[t])[..., None, None]
        kv = (S * kh[t].unsqueeze(-1)).sum(dim=-2)
        delta = (v[t] - kv) * beta[t].unsqueeze(-1)
        S = S + kh[t].unsqueeze(-1) * delta.unsqueeze(-2)
        if (t + 1) in keep:
            o = (S * qh[t].unsqueeze(-1)).sum(dim=-2)   # [48,128]
            o_rows[t + 1] = torch.stack(
                [rmsnorm_gated(o[h], w, z[t, h]) for h in range(NV)])
    return S, o_rows


def chunked_walk(S0, k, v, beta, glog, L, o_out=None, q=None):
    """chunked reference (the C13 kernel math), fp64, per-head explicit loops.

    EXACT derivation (validated by (a) an L=1 self-check that must reproduce
    one naive step bit-for-bit, (b) the generator assert chunked==naive in
    fp64). Local t∈[0,L), Γ_t=Σ_{j≤t}g_j, Γ_{−1}=0:
      unroll:  S^(t) = S_0·e^{Γ_t} + Σ_{i≤t} k̂_i·δ_iᵀ·e^{Γ_t−Γ_i}
      δ-rule:  δ_t = β_t(v_t − k̂_tᵀS^(t−1))
      ⇒ unit-diag lower-triangular solve per v-dim d:
         M[t,i] = β_t·(k̂_t·k̂_i)·e^{Γ_{t−1}−Γ_i}  (i<t), diag 1 ⇒ no divides
         rhs[t] = β_t·(v_t − e^{Γ_{t−1}}·(S_0ᵀk̂_t))
      carry:   S_L = S_0·e^{Γ_{L−1}} + Σ_i k̂_i·δ_iᵀ·e^{Γ_{L−1}−Γ_i}
      output:  o_t = e^{Γ_t}·(S_0ᵀq̂_t) + Σ_{i≤t}(q̂_t·k̂_i)·e^{Γ_t−Γ_i}·δ_i
      (o optional: the K3 kernel's fold; dumped o rows still come from the
      naive walk so Swift validates the kernel against the naive golden)."""
    T = k.shape[0]
    qk = [h // 3 for h in range(NV)]
    S = S0.clone().double()                              # [48,128,128]
    for t0 in range(0, T, L):
        Lc = min(L, T - t0)
        for h in range(NV):
            i = qk[h]
            kk = k[t0:t0 + Lc, i].double()               # [L,128]
            vv = v[t0:t0 + Lc, h].double()               # [L,128]
            bb = beta[t0:t0 + Lc, h].double()            # [L]
            gg = glog[t0:t0 + Lc, h].double()
            Gam = torch.cumsum(gg, 0)                    # Γ_t
            gL = Gam[Lc - 1]
            GK = kk @ kk.t()                             # [L,L] (k̂_t·k̂_s)
            # EXACT exponents (re-derived from δ_t = β_t(v_t − e^{g_t}·
            # S^(t−1)ᵀk̂_t and S^(t−1) = S_0e^{Γ_{t−1}} + Σ k̂_iδ_iᵀe^{Γ_{t−1}−Γ_i};
            # multiplying by e^{g_t} folds it: e^{g_t+Γ_{t−1}} = e^{Γ_t}):
            #   M[t,i] = β_t·(k̂_t·k̂_i)·e^{Γ_t−Γ_i}    (i<t), diag 1 ⇒ no divide
            #   rhs[t] = β_t·(v_t − e^{Γ_t}·(S_0ᵀk̂_t))
            # An e^{Γ_{t−1}−Γ_i} draft fails the L=1 bit-check below (δ_0
            # off by e^{g_0}) — the check is the tie-breaker, not memory.
            Mm = torch.eye(Lc, dtype=torch.float64)
            for r in range(1, Lc):
                Mm[r, :r] = bb[r] * GK[r, :r] * torch.exp(Gam[r] - Gam[:r])
            KSo = kk @ S[h]                              # [L,128] (S_0ᵀk̂_t)[d]
            rhs = bb[:, None] * (vv - torch.exp(Gam)[:, None] * KSo)
            delta = torch.zeros(Lc, D, dtype=torch.float64)
            for r in range(Lc):                          # unit diag, no divide
                delta[r] = rhs[r] - (Mm[r, :r] @ delta[:r] if r else 0.0)
            if o_out is not None:
                qq = q[t0:t0 + Lc, i].double()
                QS = qq @ S[h]                           # [L,128] S_0ᵀq̂_t
                QK = qq @ kk.t()
                for t in range(Lc):
                    intra = (QK[t, :t + 1] * torch.exp(Gam[t] - Gam[:t + 1])) @ delta[:t + 1]
                    o_out[t0 + t, h] = torch.exp(Gam[t]) * QS[t] + intra
            S[h] = S[h] * torch.exp(gL) + (kk * torch.exp(gL - Gam)[:, None]).t() @ delta
    return S.float()


def chunked_selfcheck(q, k, v, beta, glog, S0):
    """L=1 chunks must reproduce the naive step EXACTLY (fp64 vs fp64) —
    isolates exponent bookkeeping bugs from both fp32 rounding and L>1
    solve interactions, one assert per risk class."""
    T = 7
    S_a = S0.clone().double()
    for h in range(NV):
        i = h // 3
        for t in range(T):
            g, b = glog[t, h].double(), beta[t, h].double()
            kk, vv, qq = k[t, i].double(), v[t, h].double(), q[t, i].double()
            S_a[h] = S_a[h] * torch.exp(g)
            kv = (S_a[h] * kk[:, None]).sum(0)
            d = (vv - kv) * b
            S_a[h] = S_a[h] + kk[:, None] * d[None, :]
    o_a = torch.zeros(T, NV, D, dtype=torch.float64)
    S_b = S0.clone().double()
    o_b = torch.zeros(T, NV, D, dtype=torch.float64)
    # same L=1 pass through chunked_walk's code path (monkey T via slice)
    Sc = chunked_walk(S0[:NV], k[:T], v[:T], beta[:T], glog[:T], 1, o_out=o_b, q=q[:T])
    errS = (Sc.double() - S_a).abs().max().item()
    return errS


def naive_state64(S0, k, v, beta, glog):
    """fp64 state-only naive walk — the pure-math twin of chunked_walk so the
    generator's math gate is 1e-6 (isolates formula bugs from fp32 rounding;
    the DUMPED golden stays the fp32 naive that the kernel is validated
    against in Swift, exactly as spec-13 frames the admission gate)."""
    T = k.shape[0]
    qk = [h // 3 for h in range(NV)]
    S = S0.clone().double()
    for t in range(T):
        for h in range(NV):
            i = qk[h]
            kk, vv = k[t, i].double(), v[t, h].double()
            b, g = beta[t, h].double(), glog[t, h].double()
            S[h] = S[h] * torch.exp(g)
            kv = (S[h] * kk[:, None]).sum(0)
            d = (vv - kv) * b
            S[h] = S[h] + kk[:, None] * d[None, :]
    return S


def naive_o64(S0, q, k, v, beta, glog):
    """fp64 naive walk returning PRE-NORM o for every t — the twin that
    validates the chunked OUTPUT fold (o_out), separate from the state gate."""
    T = k.shape[0]
    qk = [h // 3 for h in range(NV)]
    S = S0.clone().double()
    O = torch.zeros(T, NV, D, dtype=torch.float64)
    for t in range(T):
        for h in range(NV):
            i = qk[h]
            kk, vv = k[t, i].double(), v[t, h].double()
            b, g = beta[t, h].double(), glog[t, h].double()
            S[h] = S[h] * torch.exp(g)
            kv = (S[h] * kk[:, None]).sum(0)
            d = (vv - kv) * b
            S[h] = S[h] + kk[:, None] * d[None, :]
            O[t, h] = (S[h] * q[t, i].double()[:, None]).sum(0)
    return O


def gen(tag, T, seed, L_check, state0=None):
    torch.manual_seed(seed)
    S0 = torch.zeros(NV, D, D) if state0 is None else state0.clone()
    q = torch.zeros(T, NH, D); k = torch.zeros(T, NH, D)
    v = torch.zeros(T, NV, D); z = torch.zeros(T, NV, D)
    beta = torch.zeros(T, NV); glog = torch.zeros(T, NV)
    w = (torch.randn(D) * 0.3 + 1.0)
    keep = set()
    for t in range(T):
        q[t] = (l2norm(torch.randn(NH, D) * 0.9) * SCALE).half().float()
        k[t] = l2norm(torch.randn(NH, D) * 0.9).half().float()
        v[t] = (torch.randn(NV, D) * 0.9).half().float()
        z[t] = (torch.randn(NV, D) * 0.8).half().float()
        b = torch.sigmoid(torch.randn(NV) * 0.5)
        b[:8] = torch.sigmoid(torch.rand(8) * 0.3 + 3.5)     # β ∈ (0.96,1.0) adversarial
        beta[t] = b.half().float()
        glog[t] = (-torch.rand(NV) * 0.02).half().float()
        for idx in {1, 2, 63, 64, 65, 66, 128, 129, 130, T - 1, T}:
            if 0 < idx <= T:
                keep.add(idx)
    w16 = w.half().float()
    S_n, o_ref = naive_walk(S0, q, k, v, z, beta, glog, w16, keep)
    assert torch.isfinite(S_n).all(), f"{tag}: adversarial β destabilised the walk — retune inputs"
    # CHUNKED state must equal naive state (fp32-level): the math gate of
    # this generator (kernel-level parity is the Swift test's job).
    # MATH gate (fp64 vs fp64, 1e-6): isolates any exponent/solve bug from
    # fp32 rounding — a wrong chunked formula is O(1), never 1e-6.
    S_c = chunked_walk(S0.clone(), k, v, beta, glog, L_check)
    S_n64 = naive_state64(S0.clone(), k, v, beta, glog)
    err64 = (S_c.double() - S_n64).abs().max().item()
    assert err64 < 1e-6, f"{tag}: chunked(L={L_check}) state vs fp64 naive: {err64:.2e}"
    # OUTPUT-fold gate: chunked o_out (2-GEMM+tril fold) vs fp64 naive o.
    O_c = torch.zeros(T, NV, D, dtype=torch.float64)
    chunked_walk(S0.clone(), k, v, beta, glog, L_check, o_out=O_c, q=q)
    O_n64 = naive_o64(S0.clone(), q, k, v, beta, glog)
    errO = (O_c - O_n64).abs().max().item()
    assert errO < 1e-6, f"{tag}: chunked o-fold vs fp64 naive: {errO:.2e}"
    # L=1 unit self-check: separates FORMULA bugs (the off-by-one-g class is
    # O(e^{0.02}−1)≈2e-2) from fp summation-order (naive scales S then sums;
    # chunked sums then scales — identical math, ~1e-8 after T steps of state
    # growth). 1e-5 sits 3 orders above rounding, 3 below any real bug.
    errL1 = chunked_selfcheck(q[:7], k[:7], v[:7], beta[:7], glog[:7], S0)
    assert errL1 < 1e-5, f"{tag}: L=1 self-check {errL1:.2e}"
    # the fp32 dump's OWN error vs fp64 (report; kernel parity is 1e-4 vs this)
    err_fp32 = (S_n - S_n64.float()).abs().max().item()
    print(f"{tag}: T={T} L={L_check} chunked state {err64:.1e} / o-fold {errO:.1e} / L1 {errL1:.1e}; fp32 dump err {err_fp32:.2e}")
    # dumps (f16 engine storage; β/g stay fp32)
    yrows = torch.cat([q.half().reshape(T, -1), k.half().reshape(T, -1),
                       v.half().reshape(T, -1)], dim=1).half()
    assert save(yrows, f"gdn_pre_y_{tag}.bin") == T * 10240 * 2
    assert save(z.half(), f"gdn_pre_z_{tag}.bin") == T * 6144 * 2
    assert save(beta, f"gdn_pre_beta_{tag}.bin") == T * NV * 4
    assert save(glog, f"gdn_pre_g_{tag}.bin") == T * NV * 4
    assert save(S0, f"gdn_pre_state0_{tag}.bin") == NV * D * D * 4
    assert save(S_n, f"gdn_pre_state_{tag}.bin") == NV * D * D * 4
    assert save(w.half(), f"gdn_pre_w_{tag}.bin") == D * 2
    for t, o in sorted(o_ref.items()):
        assert save(rot32(o), f"gdn_pre_o_t{t}_{tag}.bin") == NV * D * 4
        assert save(rot32(o).half(), f"gdn_pre_o16_t{t}_{tag}.bin") == NV * D * 2
    print(f"{tag}: dumped {len(keep)} o-rows")




def dump_uwab(tag="r", T=130, L=64, seed=31):
    """Dumps K1's expected outputs (U,W,A,B) + the engine inputs for the
    EXACT gen(tag)'s RNG stream (Swift bisect test reads these). The
    formulas are dbg_ab-validated (o err 3e-17 vs fp64 naive): per chunk,
    M = I + tril(β_r·(k̂_r·k̂_i)e^{Γ_r−Γ_i},−1); M·U = β∘V; M·W = diag(βe^Γ)K̂;
    A = tril(QKg)·U; B = e^Γ·q̂ − tril(QKg)·W; QKg[r,i]=e^{Γ_r−Γ_i}·q̂_r·k̂_i."""
    torch.manual_seed(seed)
    S0 = torch.zeros(NV, D, D)
    q = torch.zeros(T, NH, D); k = torch.zeros(T, NH, D)
    v = torch.zeros(T, NV, D); z = torch.zeros(T, NV, D)
    beta = torch.zeros(T, NV); glog = torch.zeros(T, NV)
    w = (torch.randn(D) * 0.3 + 1.0)
    for t in range(T):
        q[t] = (l2norm(torch.randn(NH, D) * 0.9) * SCALE).half().float()
        k[t] = l2norm(torch.randn(NH, D) * 0.9).half().float()
        v[t] = (torch.randn(NV, D) * 0.9).half().float()
        z[t] = (torch.randn(NV, D) * 0.8).half().float()
        b = torch.sigmoid(torch.randn(NV) * 0.5)
        b[:8] = torch.sigmoid(torch.rand(8) * 0.3 + 3.5)
        beta[t] = b.half().float()
        glog[t] = (-torch.rand(NV) * 0.02).half().float()
    qk = [h // 3 for h in range(NV)]
    Ud = torch.zeros(T, NV, D); Wd = torch.zeros(T, NV, D)
    Ad = torch.zeros(T, NV, D); Bd = torch.zeros(T, NV, D)
    for c in range((T + L - 1) // L):
        t0 = c * L; Lc = min(L, T - t0)
        for h in range(NV):
            i = qk[h]
            kk = k[t0:t0 + Lc, i].double(); qq = q[t0:t0 + Lc, i].double()
            vv = v[t0:t0 + Lc, h].double()
            bb = beta[t0:t0 + Lc, h].double(); gg = glog[t0:t0 + Lc, h].double()
            Gam = torch.cumsum(gg, 0)
            GK = kk @ kk.t()
            Mm = torch.eye(Lc, dtype=torch.float64)
            for r in range(1, Lc):
                Mm[r, :r] = bb[r] * GK[r, :r] * torch.exp(Gam[r] - Gam[:r])
            U = torch.zeros(Lc, D, dtype=torch.float64)
            Wm = torch.zeros(Lc, D, dtype=torch.float64)
            for r in range(Lc):
                U[r] = bb[r] * vv[r] - (Mm[r, :r] @ U[:r] if r else 0.0)
                Wm[r] = bb[r] * torch.exp(Gam[r]) * kk[r] - (Mm[r, :r] @ Wm[:r] if r else 0.0)
            QKg = torch.zeros(Lc, Lc, dtype=torch.float64)
            for r in range(Lc):
                QKg[r, :r + 1] = (qq[r] @ kk[:r + 1].t()) * torch.exp(Gam[r] - Gam[:r + 1])
            A = torch.tril(QKg) @ U
            B = torch.exp(Gam)[:, None] * qq - torch.tril(QKg) @ Wm
            Ud[t0:t0 + Lc, h] = U.float(); Wd[t0:t0 + Lc, h] = Wm.float()
            Ad[t0:t0 + Lc, h] = A.float(); Bd[t0:t0 + Lc, h] = B.float()
    assert torch.isfinite(Ud).all() and torch.isfinite(Wd).all()
    assert save(torch.cat([q.half().reshape(T, -1), k.half().reshape(T, -1),
                           v.half().reshape(T, -1)], dim=1).half(), f"gdn_probe_y.bin") == T * 10240 * 2
    assert save(z.half(), "gdn_probe_z.bin") == T * 6144 * 2
    assert save(beta, "gdn_probe_beta.bin") == T * NV * 4
    assert save(glog, "gdn_probe_g.bin") == T * NV * 4
    assert save(S0, "gdn_probe_state0.bin") == NV * D * D * 4
    assert save(w.half(), "gdn_probe_w.bin") == D * 2
    # state golden from the validated chunked_walk (same as gen("r")'s dump)
    assert save(chunked_walk(S0.clone(), k, v, beta, glog, L), "gdn_probe_state.bin") == NV * D * D * 4
    # CASE COLLISION LAW (macOS APFS is case-INSENSITIVE): "W.bin" silently
    # overwrote the 256-byte norm "w.bin" and poisoned every K3 check (inf
    # w-values). Matrix names must not differ from scalar names by case only.
    for nm, tarr in [("U", Ud), ("Wmat", Wd), ("A", Ad), ("B", Bd)]:
        assert save(tarr, f"gdn_probe_{nm}.bin") == T * NV * D * 4
    # ── o_scr gate refs (K2's pre-norm output): o_t = A_t + B_t·S_chunk,
    # S at chunk start obtained EXACTLY via shorter-chunk walks (chunk 1
    # start = L=63 walk; chunk 2 start = L=128 walk — L=1 walks are the
    def state_at(n):
        return chunked_walk(S0.clone(), k[:n], v[:n], beta[:n], glog[:n], 1).double()
    Sst = {c: state_at(c * L) for c in range(3) if c * L <= T}
    oscr = torch.zeros(T, NV, D)
    for c in range((T + L - 1) // L):
        t0 = c * L; Lc = min(L, T - t0)
        for t in range(t0, t0 + Lc):
            oscr[t] = (Ad[t].double() + torch.stack(
                [Bd[t].double() @ Sst[c][h] for h in range(NV)])).float()
    assert save(oscr, "gdn_probe_oscr.bin") == T * NV * D * 4
    print(f"probe dumps written ({T} tokens, L={L}) + o_scr refs")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        dump_uwab()
    else:
        gen("r", T=130, seed=31, L_check=64)
        gen("k", T=1024, seed=41, L_check=64)
        print("gdn prefill fixtures done")

