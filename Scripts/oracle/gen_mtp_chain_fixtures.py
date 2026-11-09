#!/usr/bin/env python3
"""gen_mtp_chain_fixtures.py — C16 leg-2a: the draft chain's first rungs
(enorm / hnorm through the shipped QWNorm door) + the f32→f16 seam cast
door, with REAL MTP weights from the mtpA golden state under HQ4_G32_M5.

Chain law this binds (plans/c16-mtp-golden.md): step j consumes
(tok_j, z_{j-1}); z_prev arrives in rotated H_32 basis (from trunk
final_norm or previous draft post_norm) → the seam cast makes it engine
f16 → hnorm (qwg_rmsnorm_from_rot: un-rotates via fwht32_inplace, applies
RMSNorm*(1+w), re-rotates via fwht32). e = embed(tok) arrives as the f16
embed row in standard basis → enorm (qwg_rmsnorm: applies RMSNorm*(1+w)
and rotates via fwht32).

Files:
  mtpc_e.bin          f16 [5120]   embed row of the golden round-0 token (standard basis)
  mtpc_zprev.bin      f32 [5120]   z_prev (rotated H_32 basis)
  mtpc_enorm_w.bin    f16 [5120]   mtp.pre_fc_norm_embedding weight
  mtpc_hnorm_w.bin    f16 [5120]   mtp.pre_fc_norm_hidden weight
  mtpc_enorm_ref.bin  f16 [5120]   fwht32(rmsnorm(e, enorm_w)) f32→f16
  mtpc_hnorm_ref.bin  f16 [5120]   fwht32(rmsnorm(fwht32_f16(cast(zprev)), hnorm_w)) f32→f16
  mtpc_cast_f32.bin   f32 [5120]   == mtpc_zprev (cast door input)
  mtpc_cast_f16.bin   f16 [5120]   torch RNE of the same values (rotated H_32 basis)
"""
import pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quantize"))
from qw38_oracle import QW38Oracle, H
from pack import fwht32_ortho

OUT = HERE.parent / "spike" / "out"
torch.set_grad_enabled(False)

def fwht32_t(x_t):
    return torch.from_numpy(fwht32_ortho(x_t.detach().cpu().numpy().astype(np.float32)))

# same seed as gen_mtp_fixtures → identical tensors, identical token
torch.manual_seed(16)
m = QW38Oracle(nl=4, vocab=4096, mtp=True, isize=1024).init_small(seed=16)
ids = torch.randint(0, 4096, (1, 16))
gs_, kc_ = m.new_gdn_states(), m.new_kv_caches()
logits, outs = m.prefill(ids, gs_, kc_)
tok = int(logits[:, -1:].argmax(-1))
# Trunk final_norm output in engine is rotated by H_32
zprev_f32 = fwht32_t(outs["final_norm"][0, 15].float())

def w(name, t):
    a = np.ascontiguousarray(t)
    (OUT / name).write_bytes(a.tobytes() if a.dtype == np.float32 else a.view(np.uint16).tobytes() if a.dtype == np.float16 else a.astype(np.uint8).tobytes())

e_row = m.embed(torch.tensor([tok])).float()[0]        # fp32 embed of tok (standard basis)
e_f16 = e_row.half()                                    # engine embed storage
enorm_w = m.mtp.enorm.weight.half()
hnorm_w = m.mtp.hnorm.weight.half()

def rmsnorm_rot_out(x_f32, w_f32):
    y_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w_f32)
    return fwht32_t(y_f32)

# qwg_rmsnorm on standard-basis e_f16
enorm_ref = rmsnorm_rot_out(e_f16.float(), enorm_w.float()).half()
# qwg_rmsnorm_from_rot on rotated z16: un-rotate via fwht32_inplace (rounds to f16), then rmsnorm + fwht32
z16_unrot_f32 = fwht32_t(zprev_f32.half().float()).half().float()
hnorm_ref = rmsnorm_rot_out(z16_unrot_f32, hnorm_w.float()).half()

assert enorm_ref.isfinite().all() and hnorm_ref.isfinite().all()
w("mtpc_e.bin", e_f16.numpy())
w("mtpc_zprev.bin", zprev_f32.numpy())
w("mtpc_enorm_w.bin", enorm_w.numpy())
w("mtpc_hnorm_w.bin", hnorm_w.numpy())
w("mtpc_enorm_ref.bin", enorm_ref.numpy())
w("mtpc_hnorm_ref.bin", hnorm_ref.numpy())
w("mtpc_cast_f32.bin", zprev_f32.numpy())
w("mtpc_cast_f16.bin", zprev_f32.half().numpy())
print(f"chain rungs (HQ4_G32_M5): tok={tok} |e|max {e_f16.abs().max():.4f} "
      f"|enorm| {enorm_ref.abs().max():.4f} |hnorm| {hnorm_ref.abs().max():.4f}")
