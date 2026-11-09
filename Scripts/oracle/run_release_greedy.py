#!/usr/bin/env python3
"""Q4g oracle replay; optional legacy Q8 comparison via QW_Q8_REFERENCE.

New model directories contain no Q8 planes. A Q4g-only replay is not a
quantization quality gate. An explicitly supplied legacy reference enables
the historical high-margin greedy comparison.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(12)
from transformers import AutoTokenizer
from qw38_oracle import QW38Oracle, load_pack_weights

PACK = os.environ.get("QW_PACK", os.path.expanduser("~/.qwg/models/Qwen3.8-27B-hhq432"))
ASSETS = os.environ.get("QW_ASSETS", os.path.join(PACK, "assets")
                        if os.path.isdir(PACK) else PACK + ".assets")
tok = AutoTokenizer.from_pretrained(ASSETS)   # tokenizer bytes sha-bind the pack
PROMPT = os.environ.get("QW_PROMPT") or (
    "The capital of France is Paris. The largest ocean on Earth is the "
    "Pacific Ocean. Water boils at 100 degrees Celsius at sea level. "
    "The planet known as the red planet is")
# QW_CHAT=1 routes the prompt through apply_chat_template EXACTLY like the
# Swift engine's run/probe path (C17 wire string → BPE). This is the
# engine↔oracle parity mode: same ids in, compare the greedy ids out.
CHAT = os.environ.get("QW_CHAT", "0") == "1"
N = int(os.environ.get("QW_N", "8"))   # greedy decode steps

def greedy(model, ids, steps):
    gdn = model.new_gdn_states(1)
    kvs = model.new_kv_caches(1)
    with torch.no_grad():
        logits, _ = model.prefill(ids, gdn, kvs)
        out = [int(logits[0, -1].argmax(-1))]
        for _ in range(steps):
            logits = model.decode_step(torch.tensor([[out[-1]]]), gdn, kvs)
            out.append(int(logits[0, -1].argmax(-1)))
    return out

def run(mode, pack=PACK):
    t0 = time.time()
    m = QW38Oracle()
    load_pack_weights(m, pack, mode=mode, verbose=False)
    tl = time.time() - t0
    if CHAT:
        wire = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(wire, return_tensors="pt").input_ids
        print(f"  chat wire: {ids.shape[1]} ids")
    else:
        ids = tok(PROMPT, return_tensors="pt").input_ids
    g = greedy(m, ids, N)
    print(f"{mode:3s} greedy [{time.time()-t0:.0f}s (load {tl:.0f}s)]: {g}")
    del m
    return g

g4 = run("hq4")
reference = os.environ.get("QW_Q8_REFERENCE")
if os.environ.get("QW_SKIP_Q8", "0") == "1" or not reference:
    print("oracle hq4 greedy ids:", g4)
    print("decoded hq4 tail:", tok.decode(g4, clean_up_tokenization_spaces=False))
    print("Q4g replay only; no Q8 comparison or quality certification.")
    sys.exit(0)
g8 = run("q8", reference)
agree = sum(a == b for a, b in zip(g4, g8))
print(f"agreement: {agree}/{len(g4)} steps")
print("oracle hq4 greedy ids:", g4)
print("decoded hq4 tail:", tok.decode(g4, clean_up_tokenization_spaces=False))
if agree == len(g4):
    print("GATE: PASS — hq4 pack greedy == q8 pack greedy on every step")
else:
    first = next(i for i, (a, b) in enumerate(zip(g4, g8)) if a != b)
    print(f"first mismatch at step {first}: hq4={g4[first]} q8={g8[first]}")
    print(f"GATE: REVIEW — agreement {agree/len(g4):.2%}")
    sys.exit(1)
