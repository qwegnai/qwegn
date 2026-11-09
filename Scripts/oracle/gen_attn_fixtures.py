#!/usr/bin/env python3
"""gen_attn_fixtures.py — C08 prefill-attention parity fixtures (torch native).

Fixtures are the kernel CONTRACT inputs — what `qwg_attn_prefill_gate` sees
AFTER C06.4 norm + C07 rope (spec 08: "rope already applied upstream; kernel
sees plain q/k"). Math mirrors the causal-fixed oracle in qw38_oracle.py /
HF modeling_qwen3_5.py EXACTLY:
    logits = q̂ @ k̂ᵀ/√256          (fp32 math on f16-rounded inputs — C06 law)
    causal tril mask BEFORE softmax (never after — nan-row trap)
    P = softmax_row(logits) ; o = P @ v̂
    out = o ⊙ σ(gate)               (HF :818 sigmoid multiply, NOT silu(gate)⊙o)

Layouts (all token-major, f16 = engine storage law, C07 parity law):
  attn_qgate_S{S}.bin  [S][12288]  interleaved [q̂(256)‖gate(256)] per head —
                        byte-identical contract to C07 qOut output.
  attn_k_S{S}.bin      [S][1024]   k̂ (normed+roped, from C07 kOut)
  attn_v_S{S}.bin      [S][1024]   v (v_proj output, untouched by C06/C07)
  attn_o_S{S}.bin      [S][6144]   out = o ⊙ σ(gate) — the kernel's contract
                        (o_proj input). f16 stored; parity test = 1-ulp
                        adjacency law.
Sizes S ∈ {7, 16, 32, 128} (non-multiples of the 32-lane q-tile) + adversarial
scale ×30 at S=32 (spec Risk 2: high-variance heads — logits std ≈ 30 after
÷16, exp() overflows without pre-subtraction; kernel must stay finite) and a
degenerate all-equal-keys set at S=32 (softmax ties; mask boundaries).

PREFIX LAW: same seed → torch fills row-major → fixtures of size S1 < S2 are
row-prefixes. The Swift suite uses q_o_S16 vs q_o_S32 equality of the first 16
rows as the CAUSAL LAW test (output for token t must not depend on t'>t).

Generator-side asserts (never trust the dump): all outputs finite, softmax
rows sum to 1 under mask, prefix property verified before write.
"""
import math
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

torch.set_grad_enabled(False)
OUT = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
OUT.mkdir(parents=True, exist_ok=True)

QH, KVH, HD = 24, 4, 256           # QWGeometry (checked vs geometry_golden at runtime)
GQA = QH // KVH                     # 6
EPS = 1e-6                          # unused here (norms are upstream) — see header


def head_norm(x):
    """HeadRMSNorm fp32 (oracle :HeadRMSNorm) with weight=1 (fixture case)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def attention_oracle(qh, gate, kh, v):
    """qh [S,QH,HD], gate [S,QH,HD], kh/v [S,KVH,HD] — f16-rounded inputs.
    Returns (out [S,QH,HD] fp32 pre-f16-store, att [S,QH,S] fp32 for asserts).
    q̂/k̂ here are the C07 OUTPUTS as-is (norm+rope already applied upstream —
    spec 08: "kernel sees plain q/k"; C07 fixtures carry their own parity),
    gate is the raw q_proj half C07 passed through untouched.
    Math EXACTLY per fixed oracle: fp32 from f16-rounded inputs."""
    q = qh.float().transpose(0, 1)                     # [QH,S,HD]
    g = gate.float().transpose(0, 1)
    k = kh.float().transpose(0, 1)                     # [KVH,S,HD] — already k̂
    vr = v.float().transpose(0, 1)
    k_r = k.repeat_interleave(GQA, dim=0)              # GQA 6:1 broadcast
    v_r = vr.repeat_interleave(GQA, dim=0)
    logits = (q @ k_r.transpose(-1, -2)) / math.sqrt(HD)      # [QH,S,S] fp32
    n = logits.shape[-1]
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, float("-inf"))       # BEFORE softmax
    att = torch.softmax(logits, dim=-1)
    o = att @ v_r                                            # [QH,S,HD]
    out = (o * torch.sigmoid(g)).transpose(0, 1)             # HF :818 epilogue
    return out, att


def save(t, name, dtype=None):
    """Torch-native bytes (this venv's numpy is broken — buffer protocol only)."""
    t = t.detach().cpu().contiguous()
    if dtype is not None:
        t = t.to(dtype)
    raw = bytes(t.view(torch.uint8).flatten())      # reinterpret, no numpy
    (OUT / name).write_bytes(raw)
    return len(raw)


def gen(S, scale=1.0, equal_keys=False, seed=20260930, tag=""):
    torch.manual_seed(seed)
    qh = (torch.randn(S, QH, HD) * scale).half()       # post rope+norm (contract)
    gate = torch.randn(S, QH, HD).half()
    kh = torch.randn(S, KVH, HD).half()
    if equal_keys:
        kh[:] = kh[0:1].clone()                        # degenerate ties
    v = torch.randn(S, KVH, HD).half()

    out, att = attention_oracle(qh, gate, kh, v)

    # ── generator-side truth checks (a fixture that lies poisons every test) ──
    assert torch.isfinite(out).all(), f"S={S}{tag}: non-finite oracle output"
    rowsum = att.sum(-1)                               # [QH,S]
    assert torch.isfinite(rowsum).all() and (rowsum - 1).abs().max() < 1e-4, \
        f"S={S}{tag}: masked softmax rows must sum to 1 (no nan rows)"
    assert att.shape == (QH, S, S)

    # q̂ in the fixture is ALREADY head-normed (the contract says post-C06);
    # re-apply norm to qh so the stored bytes match what C07 actually emits?
    # NO — C07 stores xn = normed; q̂ here stands for that output. gate is the
    # RAW q_proj half (untouched by C07's rope/norm math — passthrough), so
    # gate is stored as drawn.
    qg_interleaved = torch.stack([qh, gate], dim=2).reshape(S, QH * HD * 2)  # [q̂‖gate] per head
    n1 = save(qg_interleaved, f"attn_qgate_S{S}{tag}.bin")
    n2 = save(kh, f"attn_k_S{S}{tag}.bin")             # k̂ = "already C07'd" f16
    n3 = save(v, f"attn_v_S{S}{tag}.bin")
    n4 = save(rot32(out), f"attn_o_S{S}{tag}.bin", torch.float16)  # f16 engine storage law
    # EXACT byte-count assertions (a truncated fixture silently zeroes tails)
    assert n1 == S * QH * HD * 2 * 2 and n2 == S * KVH * HD * 2, f"qgate/k bytes S={S}"
    assert n3 == S * KVH * HD * 2 and n4 == S * QH * HD * 2, f"v/o bytes S={S}"
    print(f"S={S}{tag}: scale={scale} out finite, softmax rows ok "
          f"(max |rowsum-1| = {(rowsum - 1).abs().max():.2e})")


def prefix_check():
    """Row-prefix law: same seed → S16 is the exact prefix of S32 inputs."""
    torch.manual_seed(20260930)
    a = torch.randn(16, QH, HD).half()
    torch.manual_seed(20260930)
    b = torch.randn(32, QH, HD).half()
    assert torch.equal(a, b[:16]), "prefix law broken — causal-law test invalid"


if __name__ == "__main__":
    prefix_check()
    for S in (7, 16, 32, 128):
        gen(S)
    gen(32, scale=30.0, tag="_adv30")          # spec Risk 2 adversarial heads
    gen(32, equal_keys=True, tag="_ties")      # degenerate mask/softmax ties
    gen(512)                                   # non-finite + curve anchor
    print("attn fixtures done")
