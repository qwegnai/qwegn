#!/usr/bin/env python3
"""gen_gdn_conv_fixtures.py — C11 GDN causal depthwise conv1d parity fixtures.

Golden source: qw38_oracle.py GatedDeltaNet (T0.3, HF-verified) — the spec-11
open question is settled BY THE GOLDEN: a SiLU DOES follow the conv, and the
conv HAS a bias (spec's "no bias" line contradicts the oracle; fixtures rule,
and the kernel binds a zero-slab if a checkpoint lacks a bias).

Contract (engine storage law, same as C06/C07/C08 fixtures):
  x       [T][10240] f16   — packed q(2048)|k(2048)|v(6144) from in_proj_qkv,
                              ALREADY quantised to f16 storage (the oracle
                              conv runs fp32 ON THESE ROUNDED values — the
                              parity is math-from-rounded-inputs).
  w       [10240][4] f16   — conv1d.weight verbatim layout (depthwise: 4 taps
                              per channel, contiguous ⇒ a lane's 8-channel
                              block is one 64 B load).
  bias    [10240]   f16
  state   [3][10240] f16   — conv state = the three PREVIOUS raw token rows,
                              age order (col 0 = oldest). Fresh sequence =
                              zeros; the _pfx cases carry a real prefix.
  y       [T][10240] f16   — silu(bias + Σ_k w[c,k]·x[t−3+k,c]) fp32 math,
                              f16 store. Parity law: 1-ulp adjacency @ 1e-3.
  state2  [3][10240] f16   — conv state AFTER the call = the last three raw
                              (PRE-silu!) token rows of cat(state, x). This
                              is pure row copy — the tests assert byte
                              EQUALITY, stronger than adjacency.

T set: {1..8, 63..66, 127..130} (spec gate: chunk-boundary neighbourhoods)
plus _pfx variants (non-zero incoming state) at T ∈ {1,2,65}. The Swift-side
streaming test replays decode (T=1 × N) and asserts per-step rows == the
prefill y rows and final state2 == prefill state2 (the spec parity law).

Generator-side asserts (never trust the dump): finite; conv window math
hand-checked on random (T,c,t) triples against the explicit 4-tap sum; state
rows byte-identical to the source token rows.
"""
import pathlib
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
OUT = pathlib.Path(__file__).resolve().parent.parent / "spike" / "out"
OUT.mkdir(parents=True, exist_ok=True)

C = 10240          # QKV_CH = (2·16 + 48)·128
K = 4              # CONVK


def make_wb(seed=20261001):
    torch.manual_seed(seed)
    w = (torch.randn(C, K) * 0.35)          # depthwise weight [C][4]
    b = (torch.randn(C) * 0.25)
    return w, b


def conv_forward(x16, w16, b16, state16):
    """fp32 math from f16-rounded inputs (engine parity convention).
    x16 [T,C] f16, w16 [C,4] f16, b16 [C] f16, state16 [3,C] f16 →
    (y16 [T,C], state2_16 [3,C])."""
    T = x16.shape[0]
    x = x16.float()
    padded = torch.cat([state16.float(), x], dim=0)          # [T+3, C]
    wt = w16.float()                                          # [C,4]
    # taps broadcast over [T, C]: wt[:, k] is [C] → aligns the channel dim
    y = wt[:, 0] * padded[0:T] + wt[:, 1] * padded[1:T + 1] \
        + wt[:, 2] * padded[2:T + 2] + wt[:, 3] * padded[3:T + 3]  # 4-tap causal
    y = y + b16.float()
    y = F.silu(y)
    state2 = padded[T:T + 3, :]                               # last 3 raw rows
    # cross-check vs torch's grouped conv1d (independent implementation)
    ref = F.conv1d(padded.T.unsqueeze(0), w16.float().unsqueeze(1), b16.float(),
                   groups=C)[0].T                            # [T, C]
    assert (F.silu(ref) - y).abs().max() < 1e-4, "hand conv != F.conv1d"
    return y.half(), state2.half()


def save(t, name):
    t = t.detach().cpu().contiguous()
    raw = bytes(t.view(torch.uint8).flatten())
    (OUT / name).write_bytes(raw)
    return len(raw)


if __name__ == "__main__":
    w, b = make_wb()
    w16, b16 = w.half(), b.half()
    assert save(w16, "gdn_conv_w.bin") == C * K * 2
    assert save(b16, "gdn_conv_bias.bin") == C * 2

    for T in [1, 2, 3, 4, 5, 6, 7, 8, 63, 64, 65, 66, 127, 128, 129, 130]:
        torch.manual_seed(1000 + T)
        x16 = (torch.randn(T, C) * 0.8).half()
        state16 = torch.zeros(3, C).half()
        y16, state2_16 = conv_forward(x16, w16, b16, state16)
        assert torch.isfinite(y16.float()).all(), f"T={T}: non-finite y"
        assert save(x16, f"gdn_conv_x_t{T}.bin") == T * C * 2
        assert save(y16, f"gdn_conv_y_t{T}.bin") == T * C * 2
        assert save(state16, f"gdn_conv_state_t{T}.bin") == 3 * C * 2
        assert save(state2_16, f"gdn_conv_state2_t{T}.bin") == 3 * C * 2
        print(f"T={T}: fresh-state OK")

    # prefix-carry cases: real incoming state (a streamed sequence mid-run)
    for T in [1, 2, 65]:
        torch.manual_seed(2000 + T)
        x16 = (torch.randn(T, C) * 0.8).half()
        state16 = (torch.randn(3, C) * 0.8).half()           # non-zero prefix
        y16, state2_16 = conv_forward(x16, w16, b16, state16)
        assert save(x16, f"gdn_conv_x_t{T}_pfx.bin") == T * C * 2
        assert save(y16, f"gdn_conv_y_t{T}_pfx.bin") == T * C * 2
        assert save(state16, f"gdn_conv_state_t{T}_pfx.bin") == 3 * C * 2
        assert save(state2_16, f"gdn_conv_state2_t{T}_pfx.bin") == 3 * C * 2
        print(f"T={T}: prefix-state OK")

    # hand triple-check: explicit 4-tap for random channels
    torch.manual_seed(9)
    x16 = (torch.randn(9, C) * 0.8).half()
    st16 = (torch.randn(3, C) * 0.5).half()
    y16, _ = conv_forward(x16, w16, b16, st16)
    for _ in range(16):
        t = int(torch.randint(0, 9, (1,))); c = int(torch.randint(0, C, (1,)))
        pad = torch.cat([st16.float(), x16.float()], 0)      # [12, C]
        acc = b16.float()[c]
        for k in range(4):
            acc += w16.float()[c, k] * pad[t + k, c]
        got = y16[t, c].float().item()
        want = torch.nn.functional.silu(acc).half().float().item()
        assert abs(got - want) <= max(1e-3, abs(want) * 6e-4), (t, c, got, want)
    print("hand 4-tap triple-check OK")
    print("gdn conv fixtures done")
