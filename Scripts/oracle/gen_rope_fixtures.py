"""C07 QWRope fixtures — HF-exact rope tables + fused norm+rope apply outputs.

Semantics copied line-by-line from qw38_oracle.py / HF transformers 5.17:
  inv_freq[i] = theta^(-2i/64)                       (i < 32)
  freqs = pos * inv_freq ; emb = cat[freqs,freqs]    (cos[k] == cos[k+32])
  y[:64] = xn[:64]·cos + rot_half(xn[:64])·sin       NeoX half-rotate
  y[64:] = xn[64:]                                   (passthrough)
  xn = head-rmsnorm(x)·(1+w) (fp32 math, eps 1e-6; PLUS-ONE law)
YaRN per HF _compute_yarn_parameters (model ships rope_type "default"; YaRN
is the 1M-extension candidate → fixture proves our recipe == HF recipe).
Pure torch (the oracle venv's numpy import is flaky → zero numpy here).
Stored as the ENGINE stores: tables f32, apply outputs f16 (parity gate
asserts 1-ulp adjacency, not raw abs — the C06 law).
"""
import math, pathlib
import torch

torch.manual_seed(7)
H, QH, KVH, HD, ROD = 5120, 24, 4, 256, 64
THETA, EPS = 1e7, 1e-6
out = pathlib.Path(__file__).parent.parent / "spike/out"
out.mkdir(parents=True, exist_ok=True)

def save(t: torch.Tensor, name: str):
    """raw little-endian bytes, no numpy (f16/f32 → uint8 view)."""
    t = t.detach().cpu().contiguous()
    assert t.dtype in (torch.float16, torch.float32), str(t.dtype)
    out.joinpath(name).write_bytes(bytes(bytearray(t.view(torch.uint8).reshape(-1).tolist())))

# ── tables: plain + YaRN(factor=4) ──────────────────────────────────────────
def inv_freq_plain():
    return 1.0 / (THETA ** (torch.arange(0, ROD, 2, dtype=torch.float32) / ROD))

def inv_freq_yarn(factor, orig_max=262144, base=THETA, dim=ROD, beta_fast=32, beta_slow=1, truncate=True):
    # verbatim HF _compute_yarn_parameters math (float32)
    pos_freqs = torch.tensor([base ** (i / dim) for i in range(0, dim, 2)], dtype=torch.float32)
    invf_ex = 1.0 / pos_freqs
    invf_in = 1.0 / (factor * pos_freqs)
    def fcd(nr):  # find_correction_dim
        return (dim * math.log(orig_max / (nr * 2 * math.pi))) / (2 * math.log(base))
    lo, hi = fcd(beta_fast), fcd(beta_slow)
    if truncate: lo, hi = math.floor(lo), math.ceil(hi)
    low = max(lo, 0.0); high = min(hi, float(dim - 1))
    if low == high: high += 0.001        # HF: prevent singularity
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    eef = 1 - ramp
    return invf_in * (1 - eef) + invf_ex * eef

def cos_sin(pos_ids, inv_freq, scaling=1.0):
    # HF forward parity: cos = freqs.cos() * attention_scaling (yarn mscale)
    freqs = pos_ids[:, None].float() * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * scaling)[None], (emb.sin() * scaling)[None]

POS_S = 512
pos_ids = torch.randperm(262144)[:POS_S].float()
for name, invf, scal in [("plain", inv_freq_plain(), 1.0),
                         ("yarn4", inv_freq_yarn(4.0), 0.1 * math.log(4.0) + 1.0)]:
    c, s = cos_sin(pos_ids, invf, scal)
    save(invf, f"rope_{name}_invfreq.bin")
    save(c, f"rope_{name}_cos.bin")
    save(s, f"rope_{name}_sin.bin")
save(pos_ids, "rope_pos_ids.bin")

# ── apply fixtures: fused headnorm + partial rope on q (interleaved) & k ────
def head_rmsnorm(x, w):          # x [...,256], RAW w [256] fp32
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)

def apply_partial(x, c_row, s_row):     # x [B,S,H,256] fp32; c/s [1,S,64] oracle form
    rot = x[..., :ROD]; d = ROD // 2
    x1, x2 = rot[..., :d], rot[..., d:]
    rot_half = torch.cat([-x2, x1], dim=-1)
    c_row = c_row.reshape(1, -1, 1, ROD)   # [1,S,64]→[1,S,1,64] broadcasts over B,H
    s_row = s_row.reshape(1, -1, 1, ROD)
    y = rot * c_row + rot_half * s_row
    return torch.cat([y, x[..., ROD:]], dim=-1)

# S=4096 prefill blobs are ~380 MB of fixtures → the apply-parity gate samples
# S=64 (row-offset math is size-independent; spec BW gate lives in bench_rope).
for S in [1, 8, 64]:
    B = 1
    c, s = cos_sin(torch.arange(S).float(), inv_freq_plain())
    save(c, f"rope_table_cos_S{S}.bin")     # EXACT table rows the kernel must use
    save(s, f"rope_table_sin_S{S}.bin")
    q_raw = torch.randn(B, S, QH * HD * 2)                    # Q‖gate interleaved
    k_raw = torch.randn(B, S, KVH * HD)
    qw = torch.randn(HD) * 0.05
    kw = torch.randn(KVH * HD) * 0.05
    # engine stores activations f16 → fixture inputs are f16-rounded
    q_in = q_raw.half(); k_in = k_raw.half()
    qf = q_in.float(); kf = k_in.float()
    qh = qf.view(B, S, QH, HD * 2)
    q_rot = qh[..., :HD]                                      # rotated 256 of each 512 slot
    xn = head_rmsnorm(q_rot, qw)
    yq = apply_partial(xn, c, s)
    q_out = torch.cat([yq, qh[..., HD:]], dim=-1).reshape(B, S, QH * HD * 2).half()
    xn_k = head_rmsnorm(kf.view(B, S, KVH, HD), kw.view(1, 1, KVH, HD))
    yk = apply_partial(xn_k, c, s)
    k_out = yk.half()
    # stage intermediates: norm stored f16 = ENGINE storage law (the f16
    # 1-ulp adjacency parity law requires BOTH endpoints f16-representable;
    # an f32 ref would make rounding noise count as violation).
    save(xn.reshape(B, S, QH, HD).half(), f"rope_q_norm_S{S}.bin")
    save(xn_k.half(), f"rope_k_norm_S{S}.bin")
    save(q_out, f"rope_q_out_S{S}.bin")
    save(k_out, f"rope_k_out_S{S}.bin")
    save(q_in, f"rope_q_in_S{S}.bin")
    save(k_in, f"rope_k_in_S{S}.bin")
    save(qw.half(), f"rope_qw_S{S}.bin")
    save(kw.half(), f"rope_kw_S{S}.bin")
    print(f"S={S}: q|y|max {q_out.float().abs().max():.4f} k|y|max {k_out.float().abs().max():.4f}")

# YaRN needle at simulated 1M position (extrapolation sanity, table-level only)
pos1m = torch.tensor([999_000.0, 1_000_000.0])
inv_y = inv_freq_yarn(4.0)
cy, sy = cos_sin(pos1m, inv_y, 0.1 * math.log(4.0) + 1.0)
save(cy, "rope_yarn4_cos_1m.bin")
save(sy, "rope_yarn4_sin_1m.bin")
print("fixtures done; yarn1m cos[0,:3] =", cy[0, :3].tolist())
