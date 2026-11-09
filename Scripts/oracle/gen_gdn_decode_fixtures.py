#!/usr/bin/env python3
"""gen_gdn_decode_fixtures.py — C12 GDN recurrent decode-step parity fixtures.

Golden: qw38_oracle.py GatedDeltaNet (T0.3 = HF modeling_qwen3_5
torch_recurrent_gated_delta_rule). Per V-head h, state S [128 k, 128 v] fp32:
    S     ← S·exp(g)                                   (g scalar/head, ≤0)
    kv[v] = Σ_k S[k,v]·k̂[k]
    δ[v]  = (v[v] − kv[v])·β                           (β scalar/head ∈ (0,1))
    S[k,v] += k̂[k]·δ[v]
    o[v]  = Σ_k S[k,v]·q̂[k]              AFTER the update (token included)
then RMSNormGated: o·rsqrt(mean(o²)+1e-6)·w ⊙ silu(z_f32) (oracle
GatedRMSNorm), stored f16 (engine storage).

HEAD-SHARING LAW (oracle repeat_interleave(GV//GQK=3, dim=2), spec-12 §3):
q̂/k̂ exist for 16 KEY heads; v-heads {3i,3i+1,3i+2} share key-head i ⇒ kernel
maps h → i = h/3. (v,z,β,g are per-v-head, 48 each.) A per-v-head q̂ dump is
self-consistent math but NOT the engine contract — dump THE ENGINE LAYOUT.

NORMING CONTRACT: the kernel receives POST-l2norm k̂ and q̂ pre-scaled by
1/√128 (oracle :119-120; producer = upstream prologue, recorded as a C12
line). Normed k̂ is also the STABILITY proof: the k-direction factor is
e^g − β·‖k‖² ∈ (−1,1) exactly when ‖k‖=1 (un-normed k diverged the 1000-step
walk — an earlier draft of this file proved it, hence this contract).

ENGINE LAYOUT (what the kernel binds; S[k][v] row-major per head):
  y       [steps][10240] f16 — q̂(16×128) ‖ k̂(16×128) ‖ v(48×128) per step
  zrow    [steps][6144]  f16 — per-v-head z
  betas   [steps][48]    f32 ;  glog [steps][48] f32 (g ≤ 0)
  state0  [48][128][128] f32 (zeros for head a; head b is pre-seeded = a
                              REAL long-sequence state; state fp32 law)
  state   [48][128][128] f32 — final state (adjacency 1e-4, spec gate)
  o_tX    [48][128]      f32 for X ∈ {1,2,3,10,999,1000} (a) / {1,2,3} (b)
  o16_tX  same, f16-rounded store view (adjacency @1e-3)
Kernel per head h: q̂=y[·,(h/3)·128], k̂=y[·,2048+(h/3)·128], v=y[·,4096+h·128],
z=zrow[·,h·128], β=betas[·,h], g=glog[·,h]; out o rows [48][128] f16.

Generator asserts (never trust the dump): state finite across the whole walk
(the stability proof), dumps byte-exact.
"""
import pathlib
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
D, NH, NV = 128, 16, 48        # head dim, key heads, value heads
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


def gen(tag, steps, seed, state0=None):
    torch.manual_seed(seed)
    # THE WALK STARTS AT state0 (None → zeros: the fresh-engine-state law).
    S0 = (torch.zeros(NV, D, D) if state0 is None else state0.clone()).float()
    S = S0.clone()
    w = (torch.randn(D) * 0.3 + 1.0).half()
    yrows, zrows, betas, gs = [], [], [], []
    o_ref = {}
    keep = {1, 2, 3, 10, 999, 1000} & set(range(1, steps + 1))
    for t in range(1, steps + 1):
        qh = (l2norm(torch.randn(NH, D) * 0.9) * SCALE).half()   # ‖q‖=128^-0.5
        kh = l2norm(torch.randn(NH, D) * 0.9).half()             # ‖k‖=1 (contract)
        v = (torch.randn(NV, D) * 0.9).half()
        z = (torch.randn(NV, D) * 0.8).half()
        beta = torch.sigmoid(torch.randn(NV) * 0.5)
        g = -torch.rand(NV) * 0.02
        yrows.append(torch.cat([qh.reshape(-1), kh.reshape(-1), v.reshape(-1)]).half())  # [10240]
        zrows.append(z)
        betas.append(beta); gs.append(g)
        qf = qh.float(); kf = kh.float()
        for h in range(NV):
            i = h // 3                                            # key-head share
            S[h] = S[h] * torch.exp(g[h])
            kv = (S[h] * kf[i][:, None]).sum(dim=0)
            delta = (v[h].float() - kv) * beta[h]
            S[h] = S[h] + kf[i][:, None] * delta[None, :]
            o = (S[h] * qf[i][:, None]).sum(dim=0)
            if t in keep:
                o_ref.setdefault(t, torch.zeros(NV, D))[h] = \
                    rmsnorm_gated(o, w.float(), z[h])
    assert torch.isfinite(S).all(), f"{tag}: state diverged (stability broken)"
    assert save(torch.cat([r.view(torch.uint8) for r in yrows]).view(torch.uint8)
                if False else
                torch.cat([r for r in yrows]).half(), f"gdn_dec_y_{tag}.bin") == steps * 10240 * 2
    assert save(torch.cat([r for r in zrows]).half(), f"gdn_dec_z_{tag}.bin") == steps * 6144 * 2
    assert save(torch.stack(betas).float(), f"gdn_dec_beta_{tag}.bin") == steps * NV * 4
    assert save(torch.stack(gs).float(), f"gdn_dec_g_{tag}.bin") == steps * NV * 4
    assert save(S, f"gdn_dec_state_{tag}.bin") == NV * D * D * 4
    assert save(S0, f"gdn_dec_state0_{tag}.bin") == NV * D * D * 4   # walk START
    assert save(w, f"gdn_dec_w_{tag}.bin") == D * 2   # per-head file (seeds differ; shared name = silent overwrite)
    for t, o in sorted(o_ref.items()):
        assert save(rot32(o), f"gdn_dec_o_t{t}_{tag}.bin") == NV * D * 4
        assert save(rot32(o).half(), f"gdn_dec_o16_t{t}_{tag}.bin") == NV * D * 2
    print(f"{tag}: {steps} steps stable; o dumps {sorted(o_ref)}")


if __name__ == "__main__":
    gen("a", steps=1000, seed=11)
    torch.manual_seed(99)
    gen("b", steps=3, seed=22, state0=torch.randn(NV, D, D) * 0.5)
    print("gdn decode fixtures done (engine layout)")
