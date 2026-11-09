#!/usr/bin/env python3
"""gen_mtp_fixtures.py — C16 MTP golden (shrunk-real oracle, CPU-seconds).

Binds the speculation PROTOCOL end-to-end: streamed draft leg, batched
verify leg, Leviathan accept/resample over an explicit uniform stream,
commit + C20 rollback laws. Swift replays the SAME uniform stream (its
Philox binding is separate — C15 golden) and must reproduce commits +
accept-lens + draft ids EXACTLY.

POSITION LAW (settled end-to-end; cross-checked HF MtpLayer/MtpCache +
vLLM qwen3_next_mtp; full derivation plans/c16-mtp-golden.md):
  s    committed length: tokens at positions 0..s-1. tok = last committed
       token (position s-1); z = hidden that SAMPLED tok (trunk post-norm
       at position s-2 — after prefill: the hidden of prompt row P-1).
  INV  trunk full-attn rows = s-1 (tok's own row pending — re-fed by the
       verify window, materialised at commit; llama.cpp speculative law).
       MTP rows = s-2 (row at position t = pair (e_t, h_{t-1}), t ≥ 1;
       tok's own pair row at position s-1 is ALWAYS this round's step-0
       job — the uniform pending-row law).
  ROUND (N drafts, N=2 here):
    step j (0-based): rope span (s-1)+j, consumes (emb(tok_j), z_j);
       tok_0 = tok, z_0 = z; tok_j (j≥1) = d_j sampled, z_j = MTP's own
       previous output (the chained approximation — trunk h for draft
       positions doesn't exist until verify; vLLM streams identically).
       Writes MTP row index (s-2)+j = position s-1+j. Output logits =
       draft distribution for the token at position s+j → sample d_{j+1}.
    verify: ONE trunk forward over window [tok, d_1..d_N] at positions
       s-1..s+N-1. p_target[i] := softmax(window row i) = target dist for
       the token at position s+i; draft d_{j+1} (at s+j) is judged by
       p_target[j]. Bonus lives at position s+N (dist p_target[N]).
    accept (mtp_spec.accept_round): acc = accepted prefix; reject at d_{
       acc+1} ⇒ t ~ norm(p_target[acc] − p_draft[acc]) occupies position
       s+acc (the rejected draft's slot); all-accept ⇒ bonus from
       p_target[N] at position s+N. Either way s' = s+acc+1, tok'= t,
       z' = hz window row acc (position s+acc-1 = s'-2 — sampled t ✓).
    commit: trunk rows := 0..s+acc-1 (count s+acc = s'-1); GDN replayed
       from pre-verify snapshot over window rows 0..acc; MTP rows kept :=
       rows_pre + acc + 1 (steps 0..acc wrote correct rows: step j's e =
       d_j was accepted for j ≤ acc; step 0's e = tok always) ⇒ count
       s+acc-1+... = (s-2)+(acc+1) = s+acc-1 = s'-2 ✓ UNIFORM pending law
       — no all-accept special case (an earlier draft of this file had
       one; it was wrong — the pending row is tok'∈'s, never bonus').
  MASK CHECK: MTP cache rows live at positions 1..m (rope pos = idx+1);
  a query at position p attends cols ≤ cache_len incl. its own fresh row
  at col cache_len = rope p ⇒ consecutive positions, no gap — matches HF
  MtpCache's mask-offset-by-1 arithmetic exactly.

UNIFORMS: one stream, consumed strictly in order: per round, draft
samples first (n=D step j before its logits are used), then the accept
draws (r_1.., then the resample/bonus draw). mtpA_drawn.bin records the
per-round accept-draw counts; mtpA_uni.bin holds exactly the consumed
values in order. Swift reads: per round — draft draws at the recorded
slots, then accept draws — and must reproduce d ids + acc + commits.

Dump (f64 unless noted): mtpA_ids.bin i32[P], mtpA_zprev.bin f32[H],
mtpA_seedkv_{k,v}.bin f16 [KVH,P-1,HD] (post-prefill MTP seed),
mtpA_rounds.bin i32[R,N], mtpA_committed.bin i32[R],
mtpA_accepted.bin i32[R], mtpA_pd.bin [R,N,V], mtpA_pt.bin [R,N+1,V],
mtpA_uni.bin, mtpA_drawn.bin i32[R], mtpA_mq_{logits.bin f32[V],
greedy.json} one streamed MTP+head logits at the END state (full-chain
math check), mtpA_mtpkv_{k,v}.bin f16 end-state cache + rows in json,
mtpA_l4b.json, mtpA_geom.json.
"""
import json, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qw38_oracle import QW38Oracle, H, KVH, HD
from mtp_spec import accept_round, sample_from

OUT = HERE.parent / "spike" / "out"
OUT.mkdir(parents=True, exist_ok=True)

NL, V, ISZ = 4, 4096, 1024          # shrunk-but-real (1 full-attn at idx 3)
P, N, R = 16, 2, 40

torch.set_grad_enabled(False)
torch.manual_seed(16)
m = QW38Oracle(nl=NL, vocab=V, mtp=True, isize=ISZ).init_small(seed=16)
m.eval()

def w(name, a):
    a = np.ascontiguousarray(a)
    (OUT / name).write_bytes(a.tobytes() if a.dtype == np.float64 else a.view(np.uint8).tobytes())

def probs(lg):                        # [1,1,V] → f64 [V]
    return torch.softmax(lg.double(), dim=-1)[0, 0].numpy()

# ---------------- trunk prefill: consistent state at s = P+1 -------------
ids = torch.randint(0, V, (1, P))
g_cur = m.new_gdn_states(); kv_cur = m.new_kv_caches()
logits, outs = m.prefill(ids, g_cur, kv_cur, return_every=True)
zpost = outs["final_norm"]            # [1,P,H] post-final-norm
assert all(c["k"].shape[2] == P for c in kv_cur), "trunk rows after prefill = P"

mkv = m.new_mtp_kv()
m.mtp_prefill(ids, zpost, mkv)        # MTP pair rows for positions 1..P-1
assert mkv["k"].shape[2] == P - 1, "seed rows = P-1 = s-2"
w("mtpA_seedkv_k.bin", mkv["k"][0].half().numpy())      # [KVH,P-1,HD]
w("mtpA_seedkv_v.bin", mkv["v"][0].half().numpy())

tok = logits[:, -1:].argmax(-1)       # token at position P
z_chain = zpost[:, P - 1:P].clone()   # hidden at P-1 = s-2 that SAMPLED tok
s = P + 1

def trunk_window(toks, gstate, caches, s0):
    """Forward [1,S] at absolute positions s0..; GDN/caches MUTATE."""
    S = toks.shape[1]
    x = m.embed(toks).float()
    gi = ki = 0
    for blk in m.layers:
        if blk.is_full:
            kvc = caches[ki]
            p0 = kvc["k"].shape[2] if kvc["k"] is not None else 0
            cos, sin = m._rope_span(p0, S, toks.device)
            x = x + blk.self_attn(blk.input_layernorm(x), cos, sin, kvc)
            ki += 1
        else:
            x = x + blk.self_attn(blk.input_layernorm(x), gstate[gi]); gi += 1
        x = x + blk.mlp(blk.post_attention_layernorm(x))
    hz = m.norm(x)
    return hz, m.lm_head(hz)

def gdn_replay(toks, gstate, s0):
    """Replay ONLY the linear layers (full-attn rows already correct)."""
    x = m.embed(toks).float(); gi = 0
    for blk in m.layers:
        if not blk.is_full:
            x = x + blk.self_attn(blk.input_layernorm(x), gstate[gi]); gi += 1
        x = x + blk.mlp(blk.post_attention_layernorm(x))

def gdn_snap():
    return [{"conv": g["conv"].clone(), "rec": g["rec"].clone()} for g in g_cur]

rng = np.random.default_rng(16)
uni = rng.random(5 * R + 16)
ui = 0
def nu():
    global ui
    v = float(uni[ui]); ui += 1
    return v

rounds, commits, acc_lens, P_d, P_t, drawn, l4b = [], [], [], [], [], [], {}
gap = 0                       # permanent pair-row holes (all-accept rounds)
for r in range(R):
    rows_pre = mkv["k"].shape[2]
    # INV: rows_pre == s - 2 - gap, where gap counts permanent pair-row
    # holes. One MTP block iterated N times writes exactly N pair rows
    # per round (steps 0..N-1, rope pos s-1+j) while an all-accept round
    # commits N+1 tokens: the LAST accepted draft d_N's own pair row is
    # never written (the round stopped before a step could build on it).
    # The hole is permanent (rows added ≤ commits; +1 lost per all-accept
    # round) and is NEVER back-filled — HF (1 draft/step, bonus commits a
    # second token) and vLLM (spec=N, 1 nextn layer) leave the identical
    # hole: it is the single-layer architecture's approximation, not our
    # bug. Recorded; the engine keeps a plain row counter, no healing.
    assert rows_pre == s - 2 - gap, f"INV-MTP r{r}: rows {rows_pre} vs {s - 2 - gap}"
    assert all(c["k"].shape[2] == s - 1 for c in kv_cur), f"INV-trunk r{r}"
    # ---- draft leg (streamed + sampled — the engine's exact flow) ----
    d_ids, d_probs, zs = [], [], []
    emb_tok, zj = tok, z_chain
    for j in range(N):
        cos, sin = m._rope_span((s - 1) + j, 1, ids.device)
        zz = m.mtp(m.embed(emb_tok).float(), zj, cos, sin, mkv)
        p_ = probs(m.lm_head(zz[:, -1:, :]))
        d = sample_from(p_, nu())
        d_ids.append(int(d)); d_probs.append(p_); zs.append(zz)
        emb_tok = torch.tensor([[d]], dtype=torch.long)
        zj = zz
    # ---- L4b causal-span binds (round 0 only; mask law vs the cache) ----
    # (a) mutating a seeded row (position P-1, idx P-2) must MOVE the
    #     pos-P query's draft dist (cache rows are VISIBLE in decode —
    #     the law the old tril-mask oracle silently broke);
    # (b) replaying step-0 into a PRE-round cache clone must reproduce
    #     p_d[0] bit-exact (own+future rows structurally invisible;
    #     determinism guard).
    if r == 0:
        mk2 = {"k": mkv["k"][:, :, :rows_pre].clone(), "v": mkv["v"][:, :, :rows_pre].clone()}
        mk2["k"][:, :, rows_pre - 1, :] *= 50.0
        zz2 = m.mtp(m.embed(tok).float(), z_chain, *m._rope_span(s - 1, 1, ids.device), mk2)
        l4b["seeded_row_visible"] = bool(not np.array_equal(probs(m.lm_head(zz2[:, -1:, :])), d_probs[0]))
        mk3 = {"k": mkv["k"][:, :, :rows_pre].clone(), "v": mkv["v"][:, :, :rows_pre].clone()}
        zz3 = m.mtp(m.embed(tok).float(), z_chain, *m._rope_span(s - 1, 1, ids.device), mk3)
        l4b["preround_replay_bitexact"] = bool(np.array_equal(probs(m.lm_head(zz3[:, -1:, :])), d_probs[0]))
    # ---- verify leg: window [tok, d_1..d_N] at positions s-1.. ----
    gs = gdn_snap()
    win = torch.cat([tok, torch.tensor([d_ids], dtype=torch.long)], dim=1)   # [1,1+N]
    hz, lg = trunk_window(win, g_cur, kv_cur, s - 1)
    p_t = [probs(lg[:, j:j + 1]) for j in range(1 + N)]
    # ---- accept: Leviathan over p_draft vs p_target (mtp_spec law) ----
    # consumption: r_1 (d_1 test), r_2 (d_2 test) — reject stops early —
    # then ONE commit draw: residual resample (reject) or bonus (all-
    # accept). So n_drawn = acc+2 on reject (the reject's r IS consumed),
    # N+1 on all-accept.
    sl = uni[ui:ui + 3]
    prefix, t_final, n_drawn = accept_round(d_probs, p_t, d_ids, sl)
    acc_expect = len(prefix)
    assert n_drawn == (acc_expect + 2 if acc_expect < N else N + 1), "draws law"
    ui += n_drawn
    drawn.append(n_drawn)
    acc = len(prefix)
    # ---- commit ----
    for c in kv_cur:                       # trunk rows 0..s+acc-1 (t's pending)
        c["k"] = c["k"][:, :, :s + acc, :].clone()
        c["v"] = c["v"][:, :, :s + acc, :].clone()
    for gi2, gsnap in enumerate(gs):
        g_cur[gi2] = gsnap
    gdn_replay(win[:, :acc + 1], g_cur, s - 1)          # positions s-1..s+acc-1
    keep = rows_pre + min(acc + 1, N)   # rows written this round: steps 0..acc
    # (correct iff their e = d_j was accepted; step acc exists only on
    #  all-accept → clamp at N; on reject step acc WAS written and correct)
    mkv["k"] = mkv["k"][:, :, :keep].clone()
    mkv["v"] = mkv["v"][:, :, :keep].clone()
    rounds.append(d_ids); commits.append(int(t_final)); acc_lens.append(acc)
    P_d.append(d_probs); P_t.append(p_t)
    tok = torch.tensor([[int(t_final)]], dtype=torch.long)   # occupies s+acc
    s = s + acc + 1
    # post-commit row law: rows' = s'-2-gap'. Reject: keep=rows_pre+acc+1,
    # rows_pre=s-2-gap ⇒ rows' = s+acc-2-gap = s'-2-gap (gap holds).
    # All-accept: keep=rows_pre+N ⇒ rows' = s+N-2-gap = s'-2-(gap+1):
    # d_N's own pair row was never built by any step → gap' = gap+1
    # PERMANENT (never back-filled; HF 1-draft/step and vLLM spec-N leave
    # the identical hole — single-layer streaming approximation).
    gap += 1 if acc == N else 0
    z_chain = hz[:, acc:acc + 1].clone()                 # hidden that sampled t
    assert mkv["k"].shape[2] == s - 2 - gap, f"post r{r} MTP rows {mkv['k'].shape[2]} vs {s - 2 - gap}"
    assert all(c["k"].shape[2] == s - 1 for c in kv_cur), f"post r{r} trunk rows"
    assert all(torch.isfinite(g["rec"]).all() for g in g_cur), f"GDN finite r{r}"

assert l4b.get("seeded_row_visible") and l4b.get("preround_replay_bitexact"), "L4b"

# end-state single streamed MTP+head logits (full-chain math check vs Swift)
cos, sin = m._rope_span(s - 1 + 1, 1, ids.device)   # next round's step-0 rope pos
zz = m.mtp(m.embed(tok).float(), z_chain, cos, sin,
           {"k": mkv["k"].clone(), "v": mkv["v"].clone()})
mq = m.lm_head(zz[:, -1:, :])[0, 0].float().numpy()
w("mtpA_mq_logits.bin", mq)
(OUT / "mtpA_mq_greedy.json").write_text(json.dumps({"argmax": int(mq.argmax()),
                                                     "rope_pos": s, "mtp_rows": s - 2}))
w("mtpA_mtpkv_k.bin", mkv["k"][0].half().numpy())    # end-state cache
w("mtpA_mtpkv_v.bin", mkv["v"][0].half().numpy())

w("mtpA_ids.bin", ids.numpy().astype(np.int32))
w("mtpA_zprev.bin", zpost[0, P - 1].float().numpy())
w("mtpA_rounds.bin", np.array(rounds, dtype=np.int32))
w("mtpA_committed.bin", np.array(commits, dtype=np.int32))
w("mtpA_accepted.bin", np.array(acc_lens, dtype=np.int32))
w("mtpA_drawn.bin", np.array(drawn, dtype=np.int32))
w("mtpA_pd.bin", np.stack(P_d).astype(np.float64))           # [R,N,V]
w("mtpA_pt.bin", np.stack(P_t).astype(np.float64))           # [R,N+1,V]
w("mtpA_uni.bin", uni[:ui].astype(np.float64))
hist = [int(np.sum(np.array(acc_lens) == k)) for k in range(N + 1)]
(OUT / "mtpA_rounds.json").write_text(json.dumps({
    "R": R, "N": N, "V": V, "P": P, "NL": NL, "ISZ": ISZ, "end_s": s,
    "accept_hist": hist, "mean_committed_per_round": float(np.mean(np.array(acc_lens) + 1)),
    "uni_consumed": int(ui),
    "consumption": "per round: N draft draws (step j before its d), then n_drawn accept draws"}))
(OUT / "mtpA_l4b.json").write_text(json.dumps(l4b))
(OUT / "mtpA_geom.json").write_text(json.dumps({
    "H": H, "KVH": KVH, "HD": HD, "NL": NL, "V": V, "ISZ": ISZ,
    "P": P, "N": N, "R": R, "layers": "4 layers, full-attn at idx 3 (FAI=4)",
    "mtp_rows_law": "pre-round rows = s-2; step j rope pos s-1+j; keep rows_pre+acc+1 = s'-2"},
    indent=1))
print("mtpA golden done:", json.dumps({
    "accept_hist": hist, "end_s": s,
    "committed_per_round": float(np.mean(np.array(acc_lens) + 1)),
    "uni": int(ui), "l4b": l4b}))
