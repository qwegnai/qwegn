"""S2 — dtype race (CPU, torch): does fp16 hurt vs bf16, and is fp32 GDN
state actually required (the config law says yes — we quantify the cost).

Runs the tiny-8 oracle prefill+decode in:
  ref      : fp32 weights, fp32 math
  act_f16  : fp16 weights/activations (matmul in fp16, acc fp32 by torch)
  act_bf16 : bf16 weights/activations
  state_f16: fp32 weights, GDN recurrent state stored fp16 (quantize per step)
Compares logits vs fp32 ref: maxabs, top-1 agreement, and per-position
logit RMSE after 4 decode steps (quality metric that predicts ppl drift).
"""
import sys, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from qw38_oracle import QW38Oracle

torch.manual_seed(1337)
DEV = "cpu"
S = 96

model = QW38Oracle(nl=8, device=DEV, vocab=8192).init_small()
ids = torch.randint(0, 8192, (1, S), device=DEV)


def run(model, dt=None, state_f16=False, steps=4):
    """Prefill + `steps` decode; returns final-step logits [1,1,V] fp32."""
    m = model
    gs = m.new_gdn_states(1, DEV)
    ks = m.new_kv_caches(1, DEV)
    if dt:
        m = copy.deepcopy(m).to(dt)
        gs = [{k: v.to(dt) if torch.is_tensor(v) else v for k, v in s.items()} for s in gs]
    with torch.no_grad():
        logits, _ = m.prefill(ids, gs, ks)
        tok = logits[:, -1:].argmax(-1)
        out = None
        for _ in range(steps):
            if state_f16:
                for s in gs:  # quantize recurrent state to f16 after each prefill
                    s["rec"] = s["rec"].half().float()
            lg = m.decode_step(tok, gs, ks)
            if state_f16:
                for s in gs:
                    s["rec"] = s["rec"].half().float()
                    s["conv"] = s["conv"].half().float()
            out = lg
            tok = lg[:, -1:].argmax(-1)
    return out.float()


ref = run(model)
f16 = run(model, dt=torch.float16)
bf16 = run(model, dt=torch.bfloat16)
s16 = run(model, state_f16=True)

def cmp(name, x):
    d = (x - ref).abs()
    top_match = (x.argmax(-1) == ref.argmax(-1)).float().mean().item()
    kl = F.kl_div(F.log_softmax(x, -1), F.softmax(ref, -1), reduction="batchmean").item()
    print(f"{name:10s} maxabs={d.max().item():.4e}  rmse={d.pow(2).mean().sqrt().item():.3e}  "
          f"top1_agree={top_match:.2f}  KL={kl:.4f}")
    return d.max().item(), kl

# ---- long-horizon compounding: greedy decode 64 steps, state f16 vs f32 ----
def run_long(model, state_f16=False, steps=64):
    gs = model.new_gdn_states(1, DEV); ks = model.new_kv_caches(1, DEV)
    with torch.no_grad():
        logits, _ = model.prefill(ids, gs, ks)
        tok = logits[:, -1:].argmax(-1)
        drift = []
        import math as _m
        for st in range(steps):
            if state_f16:
                for s in gs:
                    s["rec"] = s["rec"].half().float(); s["conv"] = s["conv"].half().float()
            lg = model.decode_step(tok, gs, ks)
            if state_f16:
                for s in gs:
                    s["rec"] = s["rec"].half().float()
            lg32 = lg.float()
            drift.append((lg32 - logits.new_zeros(())).abs().max().item() if False else lg32)
            tok = lg[:, -1:].argmax(-1)
    return drift

def run_long_ref(model, steps=64):
    gs = model.new_gdn_states(1, DEV); ks = model.new_kv_caches(1, DEV)
    with torch.no_grad():
        logits, _ = model.prefill(ids, gs, ks)
        tok = logits[:, -1:].argmax(-1)
        outs = []
        for st in range(steps):
            lg = model.decode_step(tok, gs, ks)
            outs.append(lg.float()[0, 0])
            tok = lg[:, -1:].argmax(-1)
    return outs

ref64 = run_long_ref(model)
f16s64 = run_long(model, state_f16=True)
import torch as _t
kls = [_t.nn.functional.kl_div(_t.log_softmax(f16s64[i], -1),
                               _t.softmax(ref64[i], -1), reduction="batchmean").item()
       for i in range(64)]
print(f"compounding KL: mean step<5 {sum(kls[:5])/5:.6f}  "
      f"step~32 {sum(kls[27:37])/10:.6f}  step~64 {sum(kls[54:])/10:.6f}")
agree = sum(int(f16s64[i].argmax() == ref64[i].argmax()) for i in range(64)) / 64
print(f"top-1 agreement over 64 greedy steps: {agree:.2f}")
print()
print("== activation dtype (final decode-step logits vs fp32 ref) ==")
m4, kl4 = cmp("fp16", f16)
mb, klb = cmp("bf16", bf16)
print("== GDN recurrent state fp16 (quantized per step) ==")
ms, kls = cmp("state_f16", s16)
print("\nS2 verdict inputs:")
print(f"  fp16 vs bf16 KL ratio: {kl4/max(klb,1e-12):.2f}× (fp16/bf16)")
print(f"  state_f16 KL: {kls:.5f} — the law says fp32; cost of fp16 state quantified above")
if kls > 5e-3:
    print("  → fp32 state CONFIRMED required (KL would compound over 10k-step sessions)")
else:
    print("  → fp16 state candidate for a later ADR (halves the 23% state-BW slice)")
