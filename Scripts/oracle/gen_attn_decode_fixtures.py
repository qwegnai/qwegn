#!/usr/bin/env python3
"""gen_attn_decode_fixtures.py — C09 decode-attention parity fixtures.

Contract (spec 09): ONE query token attends over a paged KV cache of T-1 past
rows PLUS its own new K/V row (the kernel attends the new row explicitly and
writes it to the page table in its epilogue — "decode never launches a
standalone KV-write"). Inputs are post-C06.4/C07 exactly like C08: the kernel
sees plain q̂/k̂ (norm+rope applied upstream); gate passes through untouched.

Math mirrors the fixed oracle (qw38_oracle.py / HF modeling_qwen3_5.py):
    logits[t] = q̂·k̂[t]/√256            (fp32 from f16-rounded inputs)
    softmax over the FULL key axis [0..T-1] in fp32 (decode is inherently
      fully-unmasked causal: a single query attends ALL past + itself)
    o = att @ v̂ ; out = o ⊙ σ(gate)      (HF :818 sigmoid multiply)
Split-K/flash ordering (splits merged by log-sum-exp) is fp32-reorder noise
at the 1e-7 level — inside the C06 1-ulp adjacency law (absTol 1e-3).

Layouts (f16 = engine storage law):
  attn_qgate_dec_t{T}.bin   [1][12288] interleaved [q̂(256)‖gate(256)]×24 heads
                            — BYTE-IDENTICAL contract to the prefill per-token row.
  attn_kcache_t{T}.bin      [T-1][1024] k̂ of the PAST keys (paged rows the
                            kernel gathers via the page table; fixtures lay
                            them contiguous — the table in tests is the
                            identity so the gather path gets real coverage
                            when tests permute it).
  attn_vcache_t{T}.bin      [T-1][1024]
  attn_knew_t{T}.bin        [1024]  k̂ of the NEW token (already normed+roped
                            by C06/C07 — contract identical to k̂ rows)
  attn_vnew_t{T}.bin        [1024]
  attn_o_t{T}.bin           [1][6144] out = o ⊙ σ(gate), f16 (o_proj input).
T ∈ {2, 257, 1000, 2049, 4321}: T-1 ∈ {1, 256, 999, 2048, 4320} — page
boundaries (256 | T-1 at T=257 puts the new token at offset 0 of page 1),
non-multiples, and a multi-split case. Plus ×30 adversarial logits at T=1000
(exp-overflow trap; kernel must stay finite) and ties (all keys equal).

Generator-side asserts (never trust the dump): finite out; softmax rows sum
to 1; exact byte counts; the new-k row is NOT inside the cache file (the
kernel writes it — reading it back is a test in itself).
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

QH, KVH, HD = 24, 4, 256           # QWGeometry
GQA = QH // KVH                    # 6


def decode_oracle(qh, gate, kcache, knew, vcache, vnew):
    """qh/gate [QH,HD] fp16, kcache [T-1,KVH,HD], knew [KVH,HD] — f16 contract
    inputs. Returns out [QH,HD] fp32 (pre-f16-store) and att [QH,T] fp32."""
    k_all = torch.cat([kcache.float(), knew.float().unsqueeze(0)], dim=0)   # [T,KVH,HD]
    v_all = torch.cat([vcache.float(), vnew.float().unsqueeze(0)], dim=0)
    q = qh.float().unsqueeze(0).transpose(0, 1)            # [QH,1,HD]
    # repeat_interleave maps kv head h → q heads [h·6 … h·6+5]: q head index
    # h_q uses kv head h_q/6 — SAME convention as the kernel (kvh = head/6).
    k_r = k_all.float().transpose(0, 1).repeat_interleave(GQA, dim=0)      # [QH,T,HD]
    v_r = v_all.float().transpose(0, 1).repeat_interleave(GQA, dim=0)
    logits = (q @ k_r.transpose(-1, -2)) / math.sqrt(HD)   # [QH,1,T] fp32
    att = torch.softmax(logits, dim=-1)                    # fully unmasked row
    o = (att @ v_r)[:, 0, :]                               # [QH,HD]
    out = o * torch.sigmoid(gate.float())                  # HF :818
    return out, att[:, 0, :]


def save(t, name, dtype=None):
    t = t.detach().cpu().contiguous()
    if dtype is not None:
        t = t.to(dtype)
    raw = bytes(t.view(torch.uint8).flatten())      # reinterpret, no numpy
    (OUT / name).write_bytes(raw)
    return len(raw)


def gen(T, scale=1.0, ties=False, seed=20261001, tag=""):
    assert T >= 2
    torch.manual_seed(seed)
    C = T - 1                                   # cache (past) rows
    qh = (torch.randn(QH, HD) * scale).half()   # post rope+norm (contract)
    gate = torch.randn(QH, HD).half()
    kcache = torch.randn(C, KVH, HD).half()
    vcache = torch.randn(C, KVH, HD).half()
    knew = torch.randn(KVH, HD).half()
    if ties:                                    # degenerate: every key equal
        kcache[:] = kcache[0:1].clone()
        knew[:] = kcache[0].clone()
    vnew = torch.randn(KVH, HD).half()

    out, att = decode_oracle(qh, gate, kcache, knew, vcache, vnew)

    # ── generator-side truth checks ─────────────────────────────────────────
    assert torch.isfinite(out).all(), f"T={T}{tag}: non-finite oracle output"
    rowsum = att.sum(-1)
    assert torch.isfinite(rowsum).all() and (rowsum - 1).abs().max() < 1e-4, \
        f"T={T}{tag}: softmax rows must sum to 1 (T-wide row)"
    assert att.shape == (QH, T)

    qg_interleaved = torch.stack([qh, gate], dim=1).reshape(QH * HD * 2)   # [q̂‖gate] per head
    assert save(qg_interleaved, f"attn_qgate_dec_t{T}{tag}.bin") == QH * HD * 2 * 2
    if C > 0:
        assert save(kcache, f"attn_kcache_t{T}{tag}.bin") == C * KVH * HD * 2
        assert save(vcache, f"attn_vcache_t{T}{tag}.bin") == C * KVH * HD * 2
    assert save(knew, f"attn_knew_t{T}{tag}.bin") == KVH * HD * 2
    assert save(vnew, f"attn_vnew_t{T}{tag}.bin") == KVH * HD * 2
    assert save(rot32(out), f"attn_o_t{T}{tag}.bin", torch.float16) == QH * HD * 2
    print(f"T={T}{tag}: scale={scale} ties={ties} keys/row-sum-1="
          f"{(rowsum - 1).abs().max():.2e}")


if __name__ == "__main__":
    for T in (2, 257, 1000, 2049, 4321):
        gen(T)
    gen(1000, scale=30.0, tag="_adv30")        # exp-overflow trap (spec Risk 2)
    gen(257, ties=True, tag="_ties")           # degenerate softmax ties
    print("attn decode fixtures done")
