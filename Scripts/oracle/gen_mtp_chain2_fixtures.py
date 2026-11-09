#!/usr/bin/env python3
"""gen_mtp_chain2_fixtures.py — C16 leg-2b/2c: the FULL draft-step chain's
weights + per-rung refs, golden round-0 state (seed 16, V=4096, isize
1024, P=16, mtpA stream) under HQ4_G32_M5.

CHAIN (MTPBlock.forward, step j=0 — engine HQ4_G32_M5 storage law):
  proj  = cat[enorm(e_f16), hnorm_from_rot(zprev_rot_f16)]   (rotated H_32)
  x1    = fc_deq_rot·proj  (f32 accumulate) → cast f16       (standard basis)
  x1h   = rmsnorm(x1, ln1)                                   (rotated H_32)
  qg32  = Wq_deq_rot·x1h   [12288] f32; slots [q̂256 ‖ gate256] per head
  q̂     = per-head rmsnorm(qg32[:256], wq) (eps 1e-6, head dim 256)
  k̂     = head_rmsnorm(k_deq_rot, wk)=(norm·(1+wk)) then PARTIAL ROPE (32 pairs, NeoX)
  vnew  = cast(Wv_deq_rot·x1h)                               (raw, no norm)
  att   = fwht32(half(FA(q̂, K, V) ⊙ σ(gate)))               (rotated H_32)
  o     = Wo_deq_rot·att → res1 = half(x1 + half(o))         (standard basis)
  ln2   = rmsnorm(res1, ln2_w)                               (rotated H_32)
  h     = fwht32(half(silu(Wgate_rot·ln2) ⊙ (Wup_rot·ln2)))  (rotated H_32)
  dn    = Wdown_deq_rot·h → res2 = half(res1 + half(dn))     (standard basis)
  z_step= rmsnorm(res2, postnorm_w)                          (rotated H_32)
  logits= Whead_deq_rot·z_step                               (f32)
"""
import pathlib, sys, math
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quantize"))
from qw38_oracle import QW38Oracle, H, QH, KVH, HD, ROD, EPS
from pack import fwht32_ortho, quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

OUT = HERE.parent / "spike" / "out"
torch.set_grad_enabled(False)

def fwht32_t(x_t):
    return torch.from_numpy(fwht32_ortho(x_t.detach().cpu().numpy().astype(np.float32)))

torch.manual_seed(16)
m = QW38Oracle(nl=4, vocab=4096, mtp=True, isize=1024).init_small(seed=16)
ids = torch.randint(0, 4096, (1, 16))
gs_, kc_ = m.new_gdn_states(), m.new_kv_caches()
logits, outs = m.prefill(ids, gs_, kc_)
tok = int(logits[:, -1:].argmax(-1))
P = 16
START = P                                              # step-0 abs position
# Trunk final_norm output in engine is rotated by H_32
zprev_rot = fwht32_t(outs["final_norm"][0, P - 1].float())

def w(name, a):
    a = np.ascontiguousarray(a)
    (OUT / name).write_bytes(a.astype(np.uint8).tobytes() if a.dtype == np.uint8
                             else (a.view(np.uint16).tobytes() if a.dtype == np.float16 else a.tobytes()))

def hq4(W, name):
    """HQ4_G32_M5 packer: quant_hq4_g32_m5 rotates 32-elem blocks by H_32 internally, return rotated dequant f32."""
    w_f32 = W.detach().numpy().astype(np.float16).astype(np.float32)
    blocks = quant_hq4_g32_m5(w_f32)
    (OUT / f"{name}.bin").write_bytes(blocks.tobytes())
    return dequant_hq4_g32_m5_rot(blocks, f16_round=True)

mt = m.mtp
# ── weights → HQ4_G32_M5 1-plane blocks + f16 norms ──
fc_deq   = hq4(mt.eh_proj.weight, "mtpc_fc")          # [5120, 10240]
q_deq    = hq4(mt.self_attn.q_proj.weight, "mtpc_q")  # [12288, 5120]
k_deq    = hq4(mt.self_attn.k_proj.weight, "mtpc_k")  # [1024, 5120]
v_deq    = hq4(mt.self_attn.v_proj.weight, "mtpc_v")
o_deq    = hq4(mt.self_attn.o_proj.weight, "mtpc_o")  # [5120, 6144]
gu_deq   = hq4(torch.cat([mt.mlp.gate_proj.weight, mt.mlp.up_proj.weight], 0), "mtpc_gu")
dn_deq   = hq4(mt.mlp.down_proj.weight, "mtpc_dn")    # [5120, 1024] (isize!)
head_deq = hq4(m.lm_head.weight, "mtpc_head")         # [4096, 5120]

for nm, t in [("ln1", mt.input_layernorm.weight), ("ln2", mt.post_attention_layernorm.weight),
              ("postnorm", mt.post_norm.weight), ("wq", mt.self_attn.q_norm.weight)]:
    w(f"mtpc_{nm}.bin", t.detach().half().numpy())
w("mtpc_wk.bin", np.tile(mt.self_attn.k_norm.weight.detach().half().numpy(), KVH))  # [1024] head-blocks

h16 = lambda t: t.half().float()      # SEAM: f16 storage rounding, f32 math
def gemv(plane, x_f16_rounded):
    return torch.mv(torch.from_numpy(plane), x_f16_rounded)

def headnorm(x, w32, d=HD):           # per-head norms (q_norm, k_norm — NO fwht32!)
    r = x.view(-1, d).float()
    return r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w32.float())

def fullnorm_rot(x_f16_rounded, w_f16):   # qwg_rmsnorm: standard in -> rotated H_32 f16 out
    xf = x_f16_rounded.float()
    wf = w_f16.half().float()
    yf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wf)
    return fwht32_t(yf).half()

def fullnorm_from_rot(x_rot_f16_rounded, w_f16):  # qwg_rmsnorm_from_rot: rotated H_32 in -> rotated H_32 f16 out
    unrot = fwht32_t(x_rot_f16_rounded.float()).half().float()
    wf = w_f16.half().float()
    yf = unrot * torch.rsqrt(unrot.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wf)
    return fwht32_t(yf).half()

# ── chain to the rope door (engine arithmetic on HQ4_G32_M5 planes) ──
e_f16 = m.embed(torch.tensor([tok])).float()[0].half()
enorm_out = fullnorm_rot(e_f16.float(), mt.enorm.weight)
hnorm_out = fullnorm_from_rot(zprev_rot.half().float(), mt.hnorm.weight)
proj = torch.cat([enorm_out, hnorm_out])                   # [10240] f16 (rotated H_32)
w("mtpc_proj.bin", proj.numpy())
x1_32 = gemv(fc_deq, proj.float())                         # f32 accum on stored f16 in
x1 = x1_32.half()
w("mtpc_x1_f32.bin", x1_32.numpy()); w("mtpc_x1.bin", x1.numpy())

x1h_f16 = fullnorm_rot(x1.float(), mt.input_layernorm.weight)
x1hd = x1h_f16.float()
w("mtpc_x1h_f32.bin", x1hd.numpy())
w("mtpc_x1h.bin", x1h_f16.numpy())

qg32 = gemv(q_deq, x1hd)                                   # [12288] f32
w("mtpc_qg_f32.bin", qg32.numpy())
qg = qg32.half().view(QH, HD * 2)
qraw = qg[:, :HD].float()
gate32 = qg[:, HD:]                                        # passthrough
qhat = headnorm(qraw, mt.self_attn.q_norm.weight.half().float())  # [24,256]
k32 = gemv(k_deq, x1hd)
kraw = k32.half().view(KVH, HD).float()
kh = headnorm(kraw, mt.self_attn.k_norm.weight.half().float())    # [4,256]

# partial rope, NeoX, 32 pairs — table rows for START..START+2
inv = 1.0 / (1e7 ** (torch.arange(0, ROD, 2, dtype=torch.float32) / ROD))
pos = torch.arange(START, START + 3, dtype=torch.float32)
freqs = pos[:, None] * inv[None, :]                        # [3,32]
emb = torch.cat([freqs, freqs], -1)                        # [3,64]
cos_t, sin_t = emb.cos(), emb.sin()
assert torch.equal(cos_t[:, :32], cos_t[:, 32:]), "HF emb=cat[f,f] law"
w("mtpc_rope_tab.bin", torch.cat([cos_t[:, :32], sin_t[:, :32]], -1).numpy())

def rope_partial(x, cos, sin):                             # x [h,256] rows
    x1_, x2_ = x[:, :ROD // 2], x[:, ROD // 2:ROD]
    rot = torch.cat([-x2_, x1_], -1)
    out = x[:, :ROD] * cos + rot * sin
    return torch.cat([out, x[:, ROD:]], -1)

qhat_r = rope_partial(qhat, cos_t[0], sin_t[0])
kh_r = rope_partial(kh, cos_t[0], sin_t[0])
q_door_ref = torch.cat([qhat_r.half(), gate32.half()], -1).reshape(-1)
w("mtpc_qdoor_ref.bin", q_door_ref.numpy())                # f16 [12288]
w("mtpc_knew_ref.bin", kh_r.half().reshape(-1).numpy())    # f16 [1024]
vnew = gemv(v_deq, x1hd).half()
w("mtpc_vnew_ref.bin", vnew.numpy())                       # f16 [1024]

# ── oracle mechanism check ──
mkv = m.new_mtp_kv()
zpost = outs["final_norm"]
m.mtp_prefill(ids, zpost, mkv)
w("mtpc_seedkv_k.bin", mkv["k"][0].half().numpy())
w("mtpc_seedkv_v.bin", mkv["v"][0].half().numpy())
emb_step = m.embed(torch.tensor([tok])).float()[:, None, :]
z0 = outs["final_norm"][0, P - 1].float()[None, None, :]
cos1, sin1 = m._rope_span(START, 1, "cpu")
z1 = m.mtp(emb_step, z0, cos1, sin1, mkv)[0, 0]
lg0 = m.lm_head(z1[None, None])
w("mtpc_z1_oracle.bin", z1.half().numpy())
w("mtpc_dft0_oracle.bin", lg0[0, 0].argmax(-1).int().numpy())

invf = 1.0 / (1e7 ** (torch.arange(0, ROD, 2, dtype=torch.float32) / ROD))
def rope_rows(x, p):
    fr = torch.tensor(float(p)) * invf
    c = torch.cat([fr.cos(), fr.cos()]); s = torch.cat([fr.sin(), fr.sin()])
    x1_, x2_ = x[:, :ROD // 2], x[:, ROD // 2:ROD]
    rot = torch.cat([-x2_, x1_], -1)
    return torch.cat([x[:, :ROD] * c + rot * s, x[:, ROD:]], -1)

# ── MTP prefill rows on ENGINE HQ4_G32_M5 planes (pair law: row t = (e_t, h_{t-1}),
#    rope pos = t, t = 1..P-1) ──
embW = m.embed.weight.detach().float()
zpost_rows = zpost[0]
seedK, seedV = [], []
for t in range(1, P):
    e_t = embW[ids[0, t]].half().float()
    z_t_rot = fwht32_t(zpost_rows[t - 1].float()).half().float()
    proj_t = torch.cat([fullnorm_rot(e_t, mt.enorm.weight),
                        fullnorm_from_rot(z_t_rot, mt.hnorm.weight)]).float()
    x1_t = h16(gemv(fc_deq, proj_t))
    x1h_t = fullnorm_rot(x1_t, mt.input_layernorm.weight).float()
    k_t = rope_rows(headnorm(h16(gemv(k_deq, x1h_t)).view(KVH, HD),
                             mt.self_attn.k_norm.weight.half().float()), t).half()
    v_t = h16(gemv(v_deq, x1h_t)).view(KVH, HD)
    seedK.append(k_t); seedV.append(v_t)
w("mtpc_seedkq_k.bin", torch.stack(seedK).half().reshape(P - 1, KVH, HD).permute(1, 0, 2).numpy())
w("mtpc_seedkq_v.bin", torch.stack(seedV).half().reshape(P - 1, KVH, HD).permute(1, 0, 2).numpy())
print("seed rows on HQ4_G32_M5 planes:", len(seedK))

# ════════════════════════════════════════════════════════════════════════
# LEG 2c: the back half on HQ4_G32_M5 planes
# ════════════════════════════════════════════════════════════════════════
q16 = q_door_ref.view(QH, HD * 2)
qh_h = q16[:, :HD].float()
g_h = q16[:, HD:].float()
Knew = kh_r.half()
Vnew = vnew.half().view(KVH, HD)
seedK_t = torch.from_numpy(np.ascontiguousarray(
    torch.stack(seedK).half().reshape(P - 1, KVH, HD).permute(1, 0, 2).numpy())).float()
seedV_t = torch.from_numpy(np.ascontiguousarray(
    torch.stack(seedV).half().reshape(P - 1, KVH, HD).permute(1, 0, 2).numpy())).float()
att_rows = []
for h in range(QH):
    g = h // (QH // KVH)
    K_all = torch.cat([seedK_t[g], Knew[g].float()[None]], 0)
    V_all = torch.cat([seedV_t[g], Vnew[g].float()[None]], 0)
    lg = (qh_h[h] @ K_all.T) / math.sqrt(HD)
    pr = torch.softmax(lg, -1)
    att_rows.append((pr @ V_all) * torch.sigmoid(g_h[h]))
att_6144 = torch.cat(att_rows)
# qwg_attn_decode_combine rounds to half, then applies fwht32_inplace
att_out = fwht32_t(att_6144.half().float()).half()
w("mtpc_att_out.bin", att_out.numpy())

res1_32 = gemv(o_deq, att_out.float())
res1 = (x1.float() + res1_32.half().float()).half()
w("mtpc_res1.bin", res1.numpy())

ln2 = fullnorm_rot(res1.float(), mt.post_attention_layernorm.weight)
w("mtpc_ln2a.bin", ln2.numpy())
ln2d = ln2.float()

gu_out32 = gemv(gu_deq, ln2d)
I_sz = mt.mlp.gate_proj.weight.shape[0]
g_part, u_part = gu_out32[:I_sz], gu_out32[I_sz:]
h_32 = torch.nn.functional.silu(g_part) * u_part
# Engine casts silu_pair output to f16, then runs qwg_fwht32_f16 in-place
h_16 = fwht32_t(h_32.half().float()).half()
w("mtpc_h.bin", h_16.numpy())

dn_out32 = gemv(dn_deq, h_16.float())
res2 = (res1.float() + dn_out32.half().float()).half()
w("mtpc_res2.bin", res2.numpy())

z_step = fullnorm_rot(res2.float(), mt.post_norm.weight)
w("mtpc_z_step.bin", z_step.numpy())

logits32 = gemv(head_deq, z_step.float())
w("mtpc_logits.bin", logits32.numpy())
d0 = int(logits32.argmax())
w("mtpc_dft0_q4.bin", torch.tensor(d0, dtype=torch.int32).numpy())
print(f"leg2c HQ4_G32_M5 draft-0: {d0} (oracle plane {int(lg0[0,0].argmax())}) "
      f"|logits| {logits32.abs().max():.3f}")
