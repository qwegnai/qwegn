#!/usr/bin/env python3
"""gen_mtp_fc_fixtures.py — C16 draft-leg fc GEMV fixtures (HQ4_G32_M5).

mtp.fc = [5120, 10240] (n=5120, k=10240) HQ4_G32_M5 — an F1 decode-GEMV
family shape (m=1, n≠vocab) ⇒ ships on the C05 ADR-001 winner via
QWGemm.encode.

The input x is the cat [e_norm ‖ h_norm] of the two pre-norms in rotated
H_32 basis — the golden round-0 state: e = fwht32(enorm(emb(tok))),
h = fwht32(hnorm(z_prev)). We dump a REAL round-0 fc input from the MTP
golden state (seed 16, V=4096, P=16) so the fixture ALSO pins the wiring
order [e-block(5120) ‖ h-block(5120)].

Files:
  mtpfc_x.bin   f16 [10240]        post-norm cat input (rotated H_32, round-0 state)
  mtpfc_w.bin   Block_HQ4_G32      [5120, 320] blocks of 20 B (102,400,000 B)
  mtpfc_y.bin   f32 [5120]         y = W_deq·x fp32
"""
import pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quantize"))
from qw38_oracle import QW38Oracle, H
from pack import fwht32_ortho, quant_hq4_g32_m5, dequant_hq4_g32_m5_rot

OUT = HERE.parent / "spike" / "out"
OUT.mkdir(parents=True, exist_ok=True)
N, K = 5120, 10240

torch.set_grad_enabled(False)
torch.manual_seed(16)
m = QW38Oracle(nl=4, vocab=4096, mtp=True, isize=1024).init_small(seed=16)
m.eval()

# real round-0 state: same seeds as gen_mtp_fixtures → identical tensors
ids = torch.randint(0, 4096, (1, 16))
gs_, kc_ = m.new_gdn_states(), m.new_kv_caches()
logits, outs = m.prefill(ids, gs_, kc_)
zpost = outs["final_norm"]
tok = logits[:, -1:].argmax(-1)
z_prev = zpost[:, 15:16]
e_std = m.mtp.enorm(m.embed(tok).float()).reshape(-1).numpy()
h_std = m.mtp.hnorm(z_prev).reshape(-1).numpy()
e_rot = fwht32_ortho(e_std).astype(np.float16)
h_rot = fwht32_ortho(h_std).astype(np.float16)
x = np.concatenate([e_rot, h_rot], axis=-1)   # f16 [10240] — wiring pin
assert x.shape == (K,)

# HQ4_G32_M5 weight plane (quant_hq4_g32_m5 rotates 32-elem input blocks by H_32 internally)
W = m.mtp.eh_proj.weight.detach().float().numpy()     # [5120, 10240]
w_f32 = W.astype(np.float16).astype(np.float32)
blocks = quant_hq4_g32_m5(w_f32)
wq_rot = dequant_hq4_g32_m5_rot(blocks, f16_round=True)

xf = x.astype(np.float32)
y = (wq_rot @ xf).astype(np.float32)

(OUT / "mtpfc_x.bin").write_bytes(np.ascontiguousarray(x).tobytes())
(OUT / "mtpfc_w.bin").write_bytes(blocks.tobytes())
(OUT / "mtpfc_y.bin").write_bytes(np.ascontiguousarray(y).tobytes())
assert blocks.nbytes == N * (K // 32) * 20 and y.shape == (N,)
print(f"mtpfc fixtures (HQ4_G32_M5): |y|max {np.abs(y).max():.4f}  wiring-eq e-block sum {np.abs(x[:H].astype(np.float32)).sum():.4f}")
