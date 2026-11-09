"""fp32 torch oracle for Qwen3.8-27B (model_type: qwen3_5) — REAL geometry.

Reference implementation written from the HF transformers-5.17 semantics
(verified line-by-line 2026-09-30), used as the parity oracle for every Swift
component. Pure fp32 everywhere (weights stored fp32; GDN state fp32 per the
model law). No third-party kernels — plain torch ops only, MPS- or CPU-runnable.

Semantics captured (see specs/12, specs/08 for the contract):
  GDN: split [q‖k‖v]; β=σ(b); g=−exp(A_log)·softplus(a+dt_bias); causal
       depthwise conv1d(k=4)+silu before the recurrence; repeat_interleave 3:1;
       l2norm q,k (eps 1e-6); q *= 128^-0.5; per-token delta rule;
       RMSNormGated(out)·silu(z); out_proj.
  Attn: q_proj → per-head [Q(256)‖gate(256)] interleave; per-head q_norm/k_norm;
       partial mRoPE (64/256); GQA 24:4; attn·sigmoid(gate); o_proj.
  MLP: silu(gate)·up → down.  Norms: RMSNorm eps 1e-6.
"""
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- geometry (real) ----------------
H = 5120
V = 248320
I = 17408
NL = 64
FAI = 4                      # full-attn interval
QH, KVH, HD = 24, 4, 256
QPROJ = QH * HD * 2          # 12288 packed Q‖gate
PRF = 0.25
ROD = int(HD * PRF)          # 64
ROPE_THETA = 1e7
EPS = 1e-6
MAXPOS = 262144
GQK, GV, GHD = 16, 48, 128
QKV_CH = (2 * GQK + GV) * GHD   # 10240
Z_CH = GV * GHD                 # 6144
CONVK = 4
MTP_LAYERS = 1

def is_full(l: int) -> bool:
    return (l + 1) % FAI == 0


def rms_norm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


# ---------------- rope (partial; text positions ⇒ mrope collapses to 1D) ----
class Rope(nn.Module):
    def __init__(self):
        super().__init__()
        inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROD, 2, dtype=torch.float32) / ROD))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, pos_ids: torch.Tensor):
        """pos_ids [S] float → (cos, sin) each [1, S, ROD] (broadcast over B, heads)."""
        freqs = pos_ids[:, None].float() * self.inv_freq[None, :]  # [S, ROD/2]
        emb = torch.cat([freqs, freqs], dim=-1)                    # [S, ROD]
        return emb.cos()[None], emb.sin()[None]


def apply_partial_rope(x, cos, sin):
    """x [B, heads, S, D]; rotate only first ROD dims, NeoX half-rotate.
    cos/sin [1, S, ROD] broadcast over batch and heads."""
    rot = x[..., :ROD]
    d = ROD // 2
    x1, x2 = rot[..., :d], rot[..., d:]
    rot_half = torch.cat([-x2, x1], dim=-1)
    cos = cos.to(x.dtype); sin = sin.to(x.dtype)
    out = rot * cos[:, None] + rot_half * sin[:, None]
    return torch.cat([out, x[..., ROD:]], dim=-1)


# ---------------- GDN block ----------------
class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(H, QKV_CH, bias=False)
        self.in_proj_z = nn.Linear(H, Z_CH, bias=False)
        self.in_proj_b = nn.Linear(H, GV, bias=False)
        self.in_proj_a = nn.Linear(H, GV, bias=False)
        # OFFICIAL GDN conv is bias-FREE (checkpoint ships conv1d.weight only;
        # the pack audit's 48 conv1d tensors are all .weight). bias=True here
        # left a RANDOM kaiming bias live in every GDN layer — loader put()
        # has no conv1d.bias to copy, the silent-skip law turned it into
        # 48×silu(noise) in the recurrence. Caught by trained-text generation
        # (2026-10-02) — the same blind spot class as the PLUS-ONE miss.
        self.conv1d = nn.Conv1d(QKV_CH, QKV_CH, kernel_size=CONVK, groups=QKV_CH, bias=False)
        self.norm = GatedRMSNorm(GHD)
        self.out_proj = nn.Linear(GV * GHD, H, bias=False)   # core_attn_out = 48·128
        self.register_buffer("A_log", torch.randn(GV) * 0.5, persistent=True)
        self.register_buffer("dt_bias", torch.randn(GV) * 0.5, persistent=True)

    def forward(self, x, state):
        """x [B,S,H]; state dict with 'conv' [B,QKV_CH,CONVK-1], 'rec' [B,GV,GHD,GHD]."""
        B, S, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)          # [B,C,S]
        z = self.in_proj_z(x).reshape(B, S, GV, GHD)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        # causal depthwise conv1d(k=4) + silu, honoring conv_state prefix
        conv_state = state["conv"]                            # [B,C,CONVK-1]
        padded = torch.cat([conv_state, mixed], dim=-1)       # [B,C,S+K-1]
        conv_out = F.conv1d(padded, self.conv1d.weight, None, groups=QKV_CH)
        assert conv_out.shape[-1] == S, (conv_out.shape, S)
        state["conv"] = padded[..., -(CONVK - 1):]
        mixed = F.silu(conv_out).transpose(1, 2)              # [B,S,C]
        q, k, v = torch.split(mixed, [GQK * GHD, GQK * GHD, GV * GHD], dim=-1)
        q = q.reshape(B, S, GQK, GHD)
        k = k.reshape(B, S, GQK, GHD)
        v = v.reshape(B, S, GV, GHD)
        beta = torch.sigmoid(b)                               # [B,S,GV]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        q = q.repeat_interleave(GV // GQK, dim=2)             # → [B,S,48,128]
        k = k.repeat_interleave(GV // GQK, dim=2)
        # recurrence fp32 (model law)
        S_rec = state["rec"].float()                          # [B,GV,GHD(k),GHD(v)]
        q = l2norm(q.float()) * (GHD ** -0.5)
        k = l2norm(k.float())
        out = torch.zeros_like(v, dtype=torch.float32)
        g = g.float(); beta = beta.float()
        for t in range(S):
            S_rec = S_rec * g[:, t].exp()[..., None, None]
            kv_mem = (S_rec * k[:, t].unsqueeze(-1)).sum(dim=-2)      # [B,GV,GHD(v)]
            delta = (v[:, t].float() - kv_mem) * beta[:, t].unsqueeze(-1)
            S_rec = S_rec + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, t] = (S_rec * q[:, t].unsqueeze(-1)).sum(dim=-2)
        state["rec"] = S_rec
        out = self.norm(out, z)                               # RMSNormGated * silu(z)
        out = out.to(self.out_proj.weight.dtype)              # fp32 math → model dtype
        return self.out_proj(out.reshape(B, S, -1))


class GatedRMSNorm(nn.Module):
    """norm(x) * silu(z) with z cast fp32 (HF Qwen3_5RMSNormGated)."""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x, z):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.weight
        return x * F.silu(z.float())


# ---------------- full attention ----------------
class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, QPROJ, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(QH * HD, H, bias=False)
        self.q_norm = HeadRMSNorm(HD)
        self.k_norm = HeadRMSNorm(HD)

    def forward(self, x, cos, sin, kv_cache):
        B, S, _ = x.shape
        # per-head packed Q‖gate: view [B,S,QH,HD*2] → first HD = Q, last HD = gate
        qg = self.q_proj(x).view(B, S, QH, HD * 2)
        q, gate = qg[..., :HD], qg[..., HD:]
        q = self.q_norm(q).transpose(1, 2)                    # [B,QH,S,HD]
        k = self.k_proj(x).view(B, S, KVH, HD)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, KVH, HD).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin)
        # CACHED-LEN LAW (fix 2026-10-01, C16 probe): `cached` is the KV
        # CACHE LENGTH pre-cat. The pre-cat `k.shape[2]-S` was identically 0,
        # so the causal mask let a cached decode query attend ONLY column 0
        # instead of the whole prefix. Oracle-vs-oracle gates cancelled it
        # symmetrically (C08 testCausalLaw bound future-row invisibility in
        # PREFILL only; the mirror law — cached-row VISIBILITY in decode —
        # was never asserted, which is how it shipped through C15).
        cached = 0
        if kv_cache is not None and kv_cache["k"] is not None:
            cached = kv_cache["k"].shape[2]
        # cos/sin conventions (both callers exist): a FULL span [0, cached+S)
        # → slice the tail; or exactly the new span of length S → use as-is.
        if cached and cos.shape[1] == cached + S:
            k = apply_partial_rope(k, cos[:, cached:], sin[:, cached:])
        else:
            k = apply_partial_rope(k, cos, sin)
        if cached:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        if kv_cache is not None:
            kv_cache["k"], kv_cache["v"] = k, v
        # GQA 6:1
        k_r = k.repeat_interleave(QH // KVH, dim=1)
        v_r = v.repeat_interleave(QH // KVH, dim=1)
        # CAUSAL MASK in ABSOLUTE positions (fix 2026-10-01, C16): query i
        # (abs position cached+i) attends column j (abs j) iff j <= cached+i.
        # The old `ones(S, cached+S).tril()` was correct ONLY at cached==0
        # (prefill) and at S==1 (every path C08-C12 ever exercised: prefill
        # cached=0, decode S=1 — both arms degenerate to the same mask).
        # C16 verify forwards a window of 1+N tokens over a long cache —
        # the old mask let query 0 see ONE column instead of the whole
        # cached prefix: silently WRONG, and the C05-F5 batched leg is
        # exactly this shape. Single-token decode outputs are bit-identical
        # old vs new (S==1 arm), which is why nothing downstream caught it.
        logits = q @ k_r.transpose(-1, -2) / math.sqrt(HD)
        col = torch.arange(logits.shape[-1], device=logits.device)[None, :]
        row = torch.arange(S, device=logits.device)[:, None] + cached
        logits = logits.masked_fill(col > row, float("-inf"))
        att = torch.softmax(logits, dim=-1)      # mask before softmax; no nan rows
        o = att @ v_r                            # [B,QH,S,HD]
        o = o.transpose(1, 2).reshape(B, S, QH * HD)
        return self.o_proj(o * torch.sigmoid(gate.reshape(B, S, -1)))


class HeadRMSNorm(nn.Module):
    """OFFICIAL TEXT NORMATIVES (bugfix 2026-10-02, caught by the trained-
    weight generation test): transformers Qwen3_5RMSNorm applies
    norm(x.float()) * (1 + w) — the Llama-style PLUS-ONE law (checkpoint
    ships w≈0: input_layernorm mean −0.03), NOT plain * w. This class is
    used for input_layernorm / post_attention_layernorm / final norm /
    q_norm / k_norm. The GDN's GatedRMSNorm keeps the PLAIN-w law
    (Qwen3_5RMSNormGated: weight * norm, checkpoint mean 0.87 — verified
    against the official classes, not assumed)."""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
                * (1.0 + self.weight))


class MLP(nn.Module):
    def __init__(self, isize=I):
        super().__init__()
        self.gate_proj = nn.Linear(H, isize, bias=False)
        self.up_proj = nn.Linear(H, isize, bias=False)
        self.down_proj = nn.Linear(isize, H, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------- MTP draft layer (C16 wiring golden) ----------------
# Authority: transformers 5.17 modeling_layers.py MtpLayer/MtpModel +
# generation/candidate_generator.py MTPCandidateGenerator (read on disk
class MTPBlock(nn.Module):
    def __init__(self, isize=I):
        super().__init__()
        self.enorm = HeadRMSNorm(H)                  # pre_fc_norm_embedding
        self.hnorm = HeadRMSNorm(H)                  # pre_fc_norm_hidden
        self.eh_proj = nn.Linear(H * 2, H, bias=False)   # mtp.fc
        self.input_layernorm = HeadRMSNorm(H)
        self.self_attn = FullAttention()             # full-attn, own kv
        self.post_attention_layernorm = HeadRMSNorm(H)
        self.mlp = MLP(isize)
        self.post_norm = HeadRMSNorm(H)              # mtp.norm

    def forward(self, tok_emb, prev_hidden, cos, sin, kv):
        """tok_emb [B,S,H] = trunk embedding of the tokens consumed;
        prev_hidden [B,S,H] = chained previous hidden. kv = OWN cache dict.
        Rope span cos/sin must cover the NEW key span (FullAttention
        slices the tail by the cached length itself)."""
        proj = torch.cat([self.enorm(tok_emb), self.hnorm(prev_hidden)], dim=-1)
        x = self.eh_proj(proj)
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.post_norm(x)


class Block(nn.Module):
    def __init__(self, idx, isize=I):
        super().__init__()
        self.is_full = is_full(idx)
        self.input_layernorm = HeadRMSNorm(H)
        self.self_attn = FullAttention() if self.is_full else GatedDeltaNet()
        self.post_attention_layernorm = HeadRMSNorm(H)
        self.mlp = MLP(isize)

    def forward(self, x, cos, sin, gdn_state, kv_cache):
        """cos/sin [1, S_eff, ROD] already sliced for this layer's position span
        (full layers: current span; kv_cache grows so S_eff must equal new K span —
        caller passes per-layer-correct positions)."""
        h = self.input_layernorm(x)
        if self.is_full:
            att = self.self_attn(h, cos, sin, kv_cache)
        else:
            att = self.self_attn(h, gdn_state)
        x = x + att
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class QW38Oracle(nn.Module):
    """Random-init, full-geometry oracle: 64 layers (48 GDN + 16 attn),
    fp32 throughout. Prefill and decode share the same math; decode_step
    reads per-layer positions from the kv cache length."""
    def __init__(self, nl=NL, device="cpu", vocab=V, mtp=False, isize=I):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, H)
        self.layers = nn.ModuleList([Block(i, isize) for i in range(nl)])
        self.norm = HeadRMSNorm(H)
        self.lm_head = nn.Linear(H, vocab, bias=False)
        self.rope = Rope()
        # C16: draft layer (own kv slice; embed/lm_head shared with trunk)
        self.mtp = MTPBlock(isize) if mtp else None
        self.to(device)

    def new_mtp_kv(self, batch=1, device="cpu"):
        return {"k": None, "v": None}

    @torch.no_grad()
    def mtp_prefill(self, ids, trunk_z, mtp_kv):
        """Seed the MTP cache at prefill — the HF prefill arm
        (candidate_generator.py: num_last = P-1; rows = tokens[1..P-1]
        paired with trunk post-norm hidden rows [0..P-2], roped at the
        EMB token's position 1..P-1). ids [B,P], trunk_z [B,P,H]
        (post-final-norm; rows 0..P-2 consumed). Training-pair law: row
        at position t = (e_t, h_{t-1}); position 0 has no pair. One
        causal [1,P-1] pass writes exactly those rows (FullAttention
        slices the new-K tail itself; the fixture binds prefill==stream)."""
        assert self.mtp is not None and self.lm_head is not None
        B, P, _ = trunk_z.shape
        emb = self.embed(ids[:, 1:]).float()               # tokens 1..P-1
        z = trunk_z[:, :P - 1]                              # h_0..h_{P-2}
        cos, sin = self._rope_span(1, P - 1, ids.device)
        return self.mtp(emb, z, cos, sin, mtp_kv)

    @torch.no_grad()
    def mtp_draft(self, tok, z_prev, mtp_kv, n_draft, start_pos):
        """N-step chain through the ONE mtp layer (vLLM spec_step_idx
        pattern; HF itself drafts 1/step with 1 layer). Training-aligned
        pair law (DeepSeek-V3 MTP (e_{t+1}, h_t); HF prefill row at
        position t = (e_t, h_{t-1}) per candidate_generator — verified
        against modeling_layers MtpModel.forward): step j consumes
        (emb(tok_j), z_{j-1}) and ropes its new kv row at the EMB token's
        position start_pos + j. z_prev = trunk post-final-norm hidden at
        position start_pos-1 (the hidden that produced `tok`); chain step
        j+1 consumes the MTP post_norm output z_j. INVARIANT: mtp kv holds
        exactly start_pos-1 rows (positions 1..start_pos-1). CACHE LAW
        (fixture-bound): row C+1 = (emb(a), H_C) with a the trunk-sampled
        token — ALWAYS correct (HF's append-only rationale holds for THAT
        row only); rows C+1+j (j≥2) chain through MTP's own drafts
        (DeepSeek-V3 chaining approximation — vLLM streams identically).
        ⇒ the MTP slice rollback = index rewind to the last ACCEPTED draft
        row (C20 ring); rejected rows are stale-by-construction and get
        overwritten. The fixture binds prefill-rows == streamed-rows for
        the j=1 link exactly and for j≥2 up to chaining (both recorded).
        Returns draft ids [B,N] (greedy, for chaining) + draft logits
        [list of [B,1,V] — the engine samples from these] + chain hiddens
        z_j [list of [B,1,H] — MTP post_norm outputs, streaming bind]."""
        assert self.mtp is not None and self.lm_head is not None
        L = 0 if mtp_kv["k"] is None else mtp_kv["k"].shape[2]
        assert L == start_pos - 1, \
            f"mtp kv rows {L} != start_pos-1 {start_pos - 1} (engine must rewind to the last accepted draft row first — C20)"
        ids, logits, zs = [], [], []
        emb = self.embed(tok).float()
        z = z_prev
        for j in range(n_draft):
            cos, sin = self._rope_span(start_pos + j, 1, tok.device)
            z = self.mtp(emb, z, cos, sin, mtp_kv)
            zs.append(z)
            lg = self.lm_head(z[:, -1:, :])           # shared trunk head
            ids.append(lg.argmax(-1))
            logits.append(lg)
            emb = self.embed(ids[-1]).float()          # greedy chain
        return torch.cat(ids, dim=1), logits, zs

    def new_gdn_states(self, batch=1, device="cpu"):
        return [{"conv": torch.zeros(batch, QKV_CH, CONVK - 1, device=device),
                 "rec": torch.zeros(batch, GV, GHD, GHD, device=device)}
                for l in self.layers if not l.is_full]

    def new_kv_caches(self, batch=1, device="cpu"):
        return [{"k": None, "v": None} for l in self.layers if l.is_full]

    def _rope_span(self, start, length, device):
        pos = torch.arange(start, start + length, device=device, dtype=torch.float32)
        return self.rope(pos)

    @torch.no_grad()
    def prefill(self, ids, gdn_states, kv_caches, return_every=False):
        """Prefill ids [B,S]. Mutates gdn_states/kv_caches. Returns logits +
        optional per-layer activations (parity corpus for Swift components)."""
        outs = {}
        B, S = ids.shape
        mdtype = self.lm_head.weight.dtype
        x = self.embed(ids).float()
        if mdtype != torch.float32:
            x = x.to(mdtype)
        gi = 0  # gdn state index
        ki = 0  # kv cache index
        for i, blk in enumerate(self.layers):
            if blk.is_full:
                kv = kv_caches[ki]
                p0 = kv["k"].shape[2] if kv["k"] is not None else 0
                cos, sin = self._rope_span(p0, S, ids.device)
                h = blk.input_layernorm(x)
                x = x + blk.self_attn(h, cos, sin, kv)
                ki += 1
            else:
                h = blk.input_layernorm(x)
                x = x + blk.self_attn(h, gdn_states[gi])
                gi += 1
            if return_every: outs[f"post_attn_{i:02d}"] = x.clone()
            x = x + blk.mlp(blk.post_attention_layernorm(x))
            if return_every: outs[f"layer_{i:02d}_out"] = x.clone()
        x = self.norm(x)
        outs["final_norm"] = x.clone()
        logits = self.lm_head(x)
        outs["logits"] = logits.clone()
        return logits, outs

    @torch.no_grad()
    def decode_step(self, tok, gdn_states, kv_caches):
        """Single-token decode; kv caches grow. tok [B,1] → logits [B,1,V].
        Position per full layer = current cached length of THAT layer."""
        B = tok.shape[0]
        mdtype = self.lm_head.weight.dtype
        x = self.embed(tok).float()
        if mdtype != torch.float32:
            x = x.to(mdtype)
        gi = 0
        ki = 0
        for blk in self.layers:
            if blk.is_full:
                kv = kv_caches[ki]
                pos = float(kv["k"].shape[2]) if kv["k"] is not None else 0.0
                cos, sin = self._rope_span(int(pos), 1, tok.device)
                h = blk.input_layernorm(x)
                x = x + blk.self_attn(h, cos, sin, kv)
                ki += 1
            else:
                h = blk.input_layernorm(x)
                x = x + blk.self_attn(h, gdn_states[gi])
                gi += 1
            x = x + blk.mlp(blk.post_attention_layernorm(x))
        x = self.norm(x)
        return self.lm_head(x)

    # ---- weight init: small randoms so activations stay sane in fp32 ----
    def init_small(self, seed=20260930):
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None:      # official GDN conv is bias-FREE
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            for l in self.layers:
                if not l.is_full:
                    l.self_attn.A_log.normal_(mean=-2.0, std=0.5)
                    l.self_attn.dt_bias.normal_(std=0.1)
        return self


# ---------------- release-pack weight loading (the release law) ----------------
def _bf16_bits_to_f32(u16):
    import numpy as _np
    return (u16.astype(_np.uint32) << 16).view(_np.float32)


def load_pack_weights(model, qwp_path, nl=None, verbose=True, mode="hq4"):
    """Feed the fp32 oracle from OUR QWP1 pack (the pack is the ONLY container
    the engine reads). HQ4_G32_M5 planes -> inverse-FWHT dequant (standard
    basis); raw planes -> exact bf16->f32 widening. Vision tensors belong to
    C18's gate; every OTHER trunk/mtp tensor MUST map — an unmapped tensor is
    a loader bug and fails loudly (NO-GAP law)."""
    import json as _json, struct as _struct, mmap, numpy as _np, torch as _t, sys as _sys
    from pathlib import Path
    assert mode == "hq4", "HQ4_G32_M5 is the only supported weight quantization"
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantize"))
    from pack import dequant_hq4_g32_m5
    qwp_path = Path(qwp_path)
    if qwp_path.is_dir():
        qwp_path = qwp_path / "model.qwp"
    f = open(qwp_path, "rb")
    assert f.read(4) == b"QWP1"
    n = _struct.unpack("<Q", f.read(8))[0]
    f.seek(-n, 2)
    hdr = _json.loads(f.read(n))
    idx = hdr["tensors"]
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def plane(name):
        e = idx[name]
        shape = tuple(e["shape"]); cnt = int(_np.prod(shape))
        if "raw_off" in e:
            u16 = _np.frombuffer(mm, dtype=_np.uint16, count=cnt, offset=e["raw_off"])
            return _t.from_numpy(_bf16_bits_to_f32(u16).copy()).reshape(shape)
        N, K = shape; g = K // 32
        blk = _np.frombuffer(mm, _np.uint8, N * g * 20, e["hq4_off"]).reshape(N, g, 20)
        return _t.from_numpy(dequant_hq4_g32_m5(blk)).reshape(shape)

    P = "model.language_model."
    mapped = set()
    def put(mod, attr, name):
        if name not in idx: return
        p = plane(name)
        obj = mod
        parts = attr.split(".")
        for pp in parts[:-1]: obj = getattr(obj, pp)
        leaf = parts[-1]
        if leaf in ("A_log", "dt_bias"):
            obj.register_buffer(leaf, p, persistent=True)
        else:
            getattr(obj, leaf).data.copy_(p.reshape(getattr(obj, leaf).shape))
        mapped.add(name)

    L = nl if nl is not None else len(model.layers)
    put(model, "embed.weight", P + "embed_tokens.weight")
    put(model, "norm.weight", P + "norm.weight")
    put(model, "lm_head.weight", "lm_head.weight")
    for i in range(L):
        b = model.layers[i]
        blk = f"{P}layers.{i}."
        put(b, "input_layernorm.weight", blk + "input_layernorm.weight")
        put(b, "post_attention_layernorm.weight", blk + "post_attention_layernorm.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            put(b, f"mlp.{proj}.weight", blk + f"mlp.{proj}.weight")
        if isinstance(b.self_attn, GatedDeltaNet):
            gdn = blk + "linear_attn."
            for pj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                put(b, f"self_attn.{pj}.weight", gdn + f"{pj}.weight")
            put(b, "self_attn.conv1d.weight", gdn + "conv1d.weight")
            if gdn + "conv1d.bias" in idx:
                put(b, "self_attn.conv1d.bias", gdn + "conv1d.bias")
            put(b, "self_attn.norm.weight", gdn + "norm.weight")
            put(b, "self_attn.A_log", gdn + "A_log")
            put(b, "self_attn.dt_bias", gdn + "dt_bias")
        else:
            sa = blk + "self_attn."
            for pj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(b, f"self_attn.{pj}.weight", sa + f"{pj}.weight")
            put(b, "self_attn.q_norm.weight", sa + "q_norm.weight")
            put(b, "self_attn.k_norm.weight", sa + "k_norm.weight")
    if model.mtp is not None:
        put(model.mtp, "enorm.weight", "mtp.pre_fc_norm_embedding.weight")
        put(model.mtp, "hnorm.weight", "mtp.pre_fc_norm_hidden.weight")
        put(model.mtp, "eh_proj.weight", "mtp.fc.weight")
        put(model.mtp, "post_norm.weight", "mtp.norm.weight")
        mb = "mtp.layers.0."
        put(model.mtp, "input_layernorm.weight", mb + "input_layernorm.weight")
        put(model.mtp, "post_attention_layernorm.weight", mb + "post_attention_layernorm.weight")
        for pj in ("gate_proj", "up_proj", "down_proj"):
            put(model.mtp, f"mlp.{pj}.weight", mb + f"mlp.{pj}.weight")
        sa = mb + "self_attn."
        for pj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            put(model.mtp, f"self_attn.{pj}.weight", sa + f"{pj}.weight")
        put(model.mtp, "self_attn.q_norm.weight", sa + "q_norm.weight")
        put(model.mtp, "self_attn.k_norm.weight", sa + "k_norm.weight")
    mm.close(); f.close()
    trunk = [k for k in idx if not k.startswith("model.visual.")]
    def belongs(k):
        if k.startswith("mtp."):
            return model.mtp is not None      # no draft layer → MTP plane is
        if "layers." not in k: return True    # out-of-scope, not a gap
        li = int(k.split("layers.")[1].split(".")[0])
        return li < L
    unmapped = [k for k in trunk if k not in mapped and belongs(k)]
    if verbose:
        print(f"pack-load: {len(mapped)} mapped · {len(unmapped)} unmapped trunk tensors")
    assert not unmapped, f"pack→oracle mapping gaps: {unmapped[:6]}"
    return model
