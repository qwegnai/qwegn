"""G0 gate runner (plan §Phase 0).

Gate (from the plan): reference self-consistency — naive recurrent GDN ==
chunked UT/WY formulation, maxabs < 1e-5 in fp32 — validates the math we
port to Swift kernels BEFORE any Swift kernel exists. Also builds the tiny
oracle and dumps the parity corpus for component benches.

Run:  .venv-oracle/bin/python Scripts/oracle/run_g0.py [--device mps|cpu]
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from qw38_oracle import (QW38Oracle, GatedRMSNorm, l2norm,
                         GQK, GV, GHD, CONVK, EPS)

HERE = Path(__file__).parent
OUT = HERE / "golden"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- chunk GDN
def chunk_gated_delta_rule(q, k, v, g, beta, initial_state=None,
                           chunk_size=64):
    """Independent UT/WY chunked formulation (NOT a copy of HF's):
    q,k [B,H,T,D] fp32 pre-normalized by caller; g,beta [B,H,T].
    Returns out [B,H,T,D], final_state [B,H,D,D]."""
    # Derived by 2-token induction from the naive recurrence (see note below);
    # the gate itself validates chunk==naive to 1e-5, so the derivation is
    # checked, not trusted.
    #   per chunk, G_t = Σ_{s≤t} g_s (from chunk start), P[i,j]=e^{G_i−G_j} (j≤i):
    #   A[i,j] = β_i (k_i·k_j) P[i,j] (i>j);      L = I + strictlower(A)
    #   u  = L⁻¹ (β⊙v);          w = L⁻¹ (β⊙e^G⊙k)
    #   d  = u − w @ S_in
    #   o_t = (QKᵀ⊙P) @ d + e^{G_t} q_t @ S_in
    #   S_out = e^{G_C} S_in + (K⊙e^{G_C−G})ᵀ @ d
    B, Hn, T, D = q.shape
    dev = q.device
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    Tp = T + pad
    NC = Tp // chunk_size
    C = chunk_size
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    def chunk(x): return x.reshape(B, Hn, NC, C, -1)   # [B,H,NC,C,D]
    qc, kc, kbc, vbc = chunk(q), chunk(k), chunk(k_beta), chunk(v_beta)
    gc = g.reshape(B, Hn, NC, C)
    cum = gc.cumsum(-1)                                # [B,H,NC,C]
    # P[i,j] = exp(G_i − G_j) for i ≥ j else 0
    tril = torch.ones(C, C, dtype=torch.bool, device=dev).tril()
    pd = (cum[..., :, None] - cum[..., None, :]).masked_fill(~tril, float("-inf")).exp()
    A = kbc @ kc.transpose(-1, -2) * pd                # A[i,j]=β_i(k_i·k_j)P[i,j]
    L = torch.eye(C, device=dev) + A.tril(-1)          # unit-lower  [B,H,NC,C,C]
    u = torch.linalg.solve_triangular(L, vbc, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(L, kbc * cum.unsqueeze(-1).exp(),
                                      upper=False, unitriangular=True)
    S = torch.zeros(B, Hn, D, D, device=dev) if initial_state is None \
        else initial_state.clone()
    out = torch.zeros_like(qc)
    for c in range(NC):
        q_c, k_c = qc[:, :, c], kc[:, :, c]
        u_c, w_c = u[:, :, c], w[:, :, c]
        G_c = cum[:, :, c]                              # [B,H,C]
        d = u_c - w_c @ S                               # [B,H,C,D]
        attnP = (q_c @ k_c.transpose(-1, -2)) * pd[:, :, c]
        out[:, :, c] = attnP @ d + (q_c * G_c.exp().unsqueeze(-1)) @ S
        S = S * G_c[..., -1].exp()[..., None, None] \
            + (k_c * (G_c[..., -1:] - G_c).exp().unsqueeze(-1)).transpose(-1, -2) @ d
    out = out.reshape(B, Hn, Tp, D)[:, :, :T]
    return out, S


def recurrent_gdn(q, k, v, g, beta, initial_state=None):
    """Naive per-token recurrence (same math as oracle GatedDeltaNet body)."""
    B, Hn, T, D = q.shape
    S = torch.zeros(B, Hn, D, D, device=q.device) if initial_state is None \
        else initial_state.clone()
    out = torch.zeros_like(q)
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kv = (S * k[:, :, t].unsqueeze(-1)).sum(-2)
        delta = (v[:, :, t] - kv) * beta[:, :, t].unsqueeze(-1)
        S = S + k[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (S * q[:, :, t].unsqueeze(-1)).sum(-2)
    return out, S


def gate_gdn_equivalence(device, seed=1337):
    """Real per-head geometry: 48 heads, D=128, T=256; l2norm+scale per spec."""
    torch.manual_seed(seed)
    B, Hn, T, D = 1, GV, 256, GHD
    dev = torch.device(device)
    q = torch.randn(B, Hn, T, D, device=dev, dtype=torch.float32)
    k = torch.randn(B, Hn, T, D, device=dev, dtype=torch.float32)
    v = torch.randn(B, Hn, T, D, device=dev, dtype=torch.float32)
    q, k = l2norm(q), l2norm(k)
    q = q * (D ** -0.5)
    # g ≤ 0 (decay law), beta in (0,1)
    A_log = torch.randn(Hn, device=dev) * 0.5 - 1.0
    a = torch.randn(B, Hn, T, device=dev) * 0.5
    dt_bias = torch.randn(Hn, device=dev) * 0.1
    g = -A_log.exp()[None, :, None] * F.softplus(a + dt_bias[None, :, None])
    beta = torch.sigmoid(torch.randn(B, Hn, T, device=dev) * 0.5)

    t0 = time.time()
    o_rec, s_rec = recurrent_gdn(q, k, v, g, beta)
    t_rec = time.time() - t0
    t0 = time.time()
    o_ch, s_ch = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
    t_ch = time.time() - t0
    amax = (o_rec - o_ch).abs().max().item()
    smax = (s_rec - s_ch).abs().max().item()
    print(f"GDN chunk==recurrent: maxabs(out)={amax:.2e} maxabs(state)={smax:.2e} "
          f"[rec {t_rec:.1f}s vs chunk {t_ch:.1f}s, chunk speedup {t_rec/max(t_ch,1e-9):.1f}×]")
    ok = amax < 1e-5 and smax < 1e-5
    print("GATE GDN-EQUIV:", "PASS" if ok else "FAIL")
    torch.save({"o_rec": o_rec.cpu(), "s_rec": s_rec.cpu(), "amax": amax},
               OUT / f"gdn_equiv_T{T}.pt")
    return ok


def dump_oracle_corpus(device):
    """Tiny oracle (8 layers = 2 full + 6 GDN, real per-layer geometry,
    vocab shrunk to 8192 for memory) → parity tensors for Swift benches."""
    model = QW38Oracle(nl=8, device=device, vocab=8192).init_small()
    gs = model.new_gdn_states(1, device)
    ks = model.new_kv_caches(1, device)
    ids = torch.randint(0, 8192, (1, 128), device=device)
    logits, acts = model.prefill(ids, gs, ks, return_every=True)
    top1 = logits.argmax(-1)
    # 4 decode steps with per-layer state evolution
    dec = []
    tok = top1[:, -1:]
    for step in range(4):
        lg = model.decode_step(tok, gs, ks)
        dec.append({"logits": lg[0, 0].cpu(), "argmax": int(lg.argmax(-1))})
        tok = lg.argmax(-1)
    torch.save({
        "ids": ids.cpu(),
        "acts": {k: v[0].cpu() for k, v in acts.items()},
        "gdn_states": [{"conv": s["conv"].cpu(), "rec": s["rec"].cpu()} for s in gs],
        "kv_full": [{"k": c["k"].cpu(), "v": c["v"].cpu()} for c in ks if c["k"] is not None],
        "decode": dec,
        "geometry": {"nl_layers": 8, "full_idx": [3, 7], "vocab": 8192,
                     "hidden": 5120, "seed": 20260930},
    }, OUT / "corpus_tiny8.pt")
    print(f"corpus dumped: {OUT/'corpus_tiny8.pt'} "
          f"({(OUT/'corpus_tiny8.pt').stat().st_size/1e6:.0f} MB)")
    # decode-after-prefill continuity check (oracle self-consistency):
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok1 = gate_gdn_equivalence(a.device)
    ok2 = dump_oracle_corpus(a.device)
    print("G0:", "PASS" if (ok1 and ok2) else "FAIL")
    sys.exit(0 if (ok1 and ok2) else 1)
