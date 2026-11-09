#!/usr/bin/env python3
"""gen_vision_fixtures.py — C18 goldens: the vision tower, real weights.

LIVE transformers truth BEFORE any Swift assertion (rule 2). The authority
is the checkpoint's OWN Qwen3_5VisionModel (27 blocks, hidden 1152, 16 heads
head_dim 72, qkv-fused, axial 2D rope full-rotation, gelu_pytorch_tanh MLP,
merger = LN(1152)→[4×1152]→fc1 4608×4608→GELU→fc2 5120×4608) loaded in fp32
from the checkpoint's vision_tower.* tensors (the oracle reference dtype;
the engine races f16 storage — the gate binds 1-ulp binade adjacency on top
of abs tol, C06/C13 law).

Stage goldens per grid (the Swift Metal port must reproduce EACH stage —
the spec names the interpolation as the classic silent-mismatch zone, so
pos-embed interpolation gets its own binding, not just end-to-end):
  s0  patch_embed out        [n_patch, 1152]   (Conv3d ≡ per-patch GEMV over
                                               the C19 slab row layout)
  s1  + interpolated pos-emb [n_patch, 1152]   (the interp bins are the
                                               C19 goldens — same taps)
  s2  rope cos/sin           [n_patch, 72]×2   (axial: freqs=arange(0,36,2)/
                                               36 over h,w coords; recompose
                                               = cat[h,w,h,w] — the FULL
                                               head-dim rotation law)
  s3  block 0 out            [n_patch, 1152]   (the first block exercises
                                               per-FRAME segment attention)
  s4  merger out (pooler)    [n_tokens, 5120]  (THE engine contract: merged
                                               visual tokens for the X₀ slab)
Grids: [1,4,6], [2,4,4], [1,2,2], [2,6,4], [4,4,4], [1,8,8] — per-frame
segments keep softmax O(h·w) bounded (the cu_seqlens law); video grids
exercise the t-repeat of rope coords + per-frame splits.
Inputs: the C19 real slab (mm_patches_c0.bin, grid [1,16,16] folded to the
256-patch stage-0? no — the slab is 256×1536 for [1,16,16]; grids above use
seeded f32 inputs (rng(18) randn·0.02) — the checkpoint weights are REAL.

Everything stored f16 (engine storage law; the generator f32-runs the
authority, then .half()s — same law as C07 rope fixtures). Bin per stage per
grid + vis_index.json (shapes, shas, the tolerance laws, determinism
self-check: s4 re-run byte-compared).

The 2e-2 bf16-tolerance gate in spec-18 is measured at RUNTIME between the
Metal backend and these fp32 refs — the goldens themselves store the fp32
authority rounded to f16 for storage parity checks.
"""
import hashlib, json, os
import numpy as np
import torch
torch.set_grad_enabled(False)

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel, Qwen3_5VisionConfig)
from transformers.vision_utils import (get_vision_position_ids,
    get_vision_interpolation_indices_and_weights, get_vision_attention_seqlens)
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "Scripts", "spike", "out")
# RELEASE LAW (2026-10-02): the original base was packed and REMOVED; the
# canonical vision weight source is now OUR PACK (verbatim-bf16 planes,
# bit-identical to the base — audit maxabs 0.0). QW_PACK points at it;
# QW_CKPT/QW_VIS_PREFIX stay for pre-pack reproduction (checkpoint ships
# vision_tower.*; the patch_embed channels_last_3d law holds on both).
MODEL = os.environ.get("QW_PACK", os.path.expanduser("~/.qwg/models/Qwen3.8-27B-hq4g32"))
PACK = os.path.join(MODEL, "model.qwp") if os.path.isdir(MODEL) else MODEL
ASSETS = os.environ.get("QW_ASSETS", os.path.join(MODEL, "assets")
                        if os.path.isdir(MODEL) else MODEL + ".assets")
CKPT = os.environ.get("QW_CKPT", "")
VISION_PREFIX = os.environ.get("QW_VIS_PREFIX", "model.visual.")
MERGE = 2

def sha(b): return hashlib.sha256(b).hexdigest()

# ── the authority: the checkpoint's own vision tower, fp32 ─────────────────
if CKPT:
    cfg_d = json.load(open(os.path.join(CKPT, "config.json")))
else:
    # config.json is an INHERITED engine asset (pack provenance assets_dir)
    a_dir = json.load(open(os.path.join(ASSETS, "manifest.json")))  # existence gate
    assert "config.json" in a_dir, "config.json not inherited into the engine dir"
    cfg_d = json.load(open(os.path.join(ASSETS, "config.json")))
vcfg = Qwen3_5VisionConfig(**cfg_d["vision_config"])
vsd = {}
if CKPT:
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    shard_tensors = {}
    for k in wm:
        if k.startswith(VISION_PREFIX):
            f = os.path.join(CKPT, wm[k])
            if f not in shard_tensors:
                shard_tensors[f] = load_file(f)
            vsd[k[len(VISION_PREFIX):]] = shard_tensors[f][k].float()   # fp32 oracle
else:
    # PACK MODE: verbatim-bf16 planes → exact fp32 (widening, zero error)
    import struct as _struct, mmap as _mmap, numpy as _np
    pf = open(PACK, "rb")
    assert pf.read(4) == b"QWP1"
    _n = _struct.unpack("<Q", pf.read(8))[0]
    pf.seek(-_n, 2)
    pidx = json.loads(pf.read(_n))["tensors"]
    mm = _mmap.mmap(pf.fileno(), 0, access=_mmap.ACCESS_READ)
    for k, e in pidx.items():
        if k.startswith("model.visual."):
            assert "raw_off" in e, f"vision tensor quantized in pack: {k}"
            cnt = int(_np.prod(e["shape"]))
            u16 = _np.frombuffer(mm, _np.uint16, cnt, e["raw_off"])
            vsd[k[len("model.visual."):]] = torch.from_numpy(
                (u16.astype(_np.uint32) << 16).view(_np.float32).copy()
            ).reshape(e["shape"])
    if "u16" in dir():  # frombuffer export must drop before mmap close (BufferError law)
        del u16
    mm.close(); pf.close()
    assert vsd, f"no model.visual.* tensors in {PACK}"
model = Qwen3_5VisionModel(vcfg)
# THE PATCH-EMBED LAYOUT LAW (measured, not assumed): the checkpoint stores
# patch_embed.proj.weight as [1152, 2, 16, 16, 3] = (o, t, h, w, c) —
# torch channels_last_3d of Conv3d's (o, c, t, h, w) — while the shipped
# model builds the plain contiguous layout. The C19 slab law (row = c-major
# × temporal-dup × raster) matches the plain Conv3d view (-1, 3, 2, 16, 16)
# EXACTLY, so the oracle consumes the permuted-back weight; the Swift port
# consumes the checkpoint bytes in THEIR NATIVE order (row-index =
# c·512 + t·256 + raster — same pairing law as the checkpoint's own W
# indexing (o, t, h, w, c): the dot is pairing-order invariant; both sides
# agree because both pair slab-index with weight-index identically).
pe = "patch_embed.proj.weight"
if vsd[pe].shape == (1152, 2, 16, 16, 3):
    vsd[pe] = vsd[pe].permute(0, 4, 1, 2, 3).contiguous()
missing, unexpected = model.load_state_dict(vsd, strict=False)
assert not missing, f"vision weights missing: {missing[:6]}"
assert not unexpected, f"vision weights unexpected: {unexpected[:6]}"
model = model.float().eval()
print(f"vision tower loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params fp32")

GRIDS = [[1, 4, 6], [2, 4, 4], [1, 2, 2], [1, 16, 16], [1, 14, 26], [3, 6, 10]]
# ↑ the C19 golden grids (overlapping subset) — the interp-bins crosscheck
# below reads the mm_interp_* bins back, so the same tag must exist there.
index = {"dtype_store": "f16", "dtype_run": "f32",
         "tolerance_law": "abs≤2e-2 ∨ binade 1-ulp (f16 storage adjacency, C06/C13 law); interpolation stage STRICT ≤1e-5 (the silent-mismatch zone is gated STRICT, not relaxed)",
         "grids": {}}

for gi, thw in enumerate(GRIDS):
    t_, h_, w_ = thw
    n = t_ * h_ * w_
    gt = torch.tensor([thw], dtype=torch.int64)
    rng = np.random.RandomState(18 + gi)
    x = torch.from_numpy((rng.randn(n, 1536) * 0.02).astype(np.float32))  # patch rows (C19 slab law shape)

    # ── stage 0: patch embed ────────────────────────────────────────────
    s0 = model.patch_embed(x)
    assert s0.shape == (n, 1152)

    # ── stage 1: + interpolated pos-embed (the C19 interp bins, live here) ─
    ii, iw = get_vision_interpolation_indices_and_weights(
        gt, num_grid_per_side=model.num_grid_per_side, mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners, spatial_merge_size=MERGE)
    pos_embeds = (model.pos_embed(ii) * iw[:, :, None]).sum(1)
    s1 = s0 + pos_embeds
    assert s1.shape == (n, 1152)

    # ── stage 2: rope cos/sin (axial full-rotation) ──────────────────────
    position_ids = get_vision_position_ids(gt, MERGE)          # [n,2] h,w
    cos, sin = model.rotary_pos_emb(s1, position_ids)          # the model's own call shape
    assert cos.shape == (n, 72) and sin.shape == (n, 72), cos.shape
    # law assert (the recomposition structure the Swift table mirrors):
    # spatial_dim = 36 → inv_freq len 18 (head_dim//4); freqs = coords ×
    # inv_freq → [n,2,18]; recompose = cat[h(18), w(18)] then duplicate:
    # cos[:, 0:36] == cos[:, 36:72] EXACTLY (cat[hw,hw]), first 18 = h-axis.
    assert torch.equal(cos[:, :36], cos[:, 36:72]) and torch.equal(sin[:, :36], sin[:, 36:72]), \
        "recomposition law changed — the Swift table must mirror cat[hw,hw]"

    # ── stage 3: block 0 (per-frame segment attention) ──────────────────
    cu, maxl = get_vision_attention_seqlens(gt, model.config)
    s3 = model.blocks[0](s1, cu_seqlens=cu, position_embeddings=(cos, sin))
    assert s3.shape == (n, 1152)

    # ── stage 4: merger — THE engine contract (merged tokens into X₀) ────
    # full tower for the final merger out (27 blocks; cheap at these sizes)
    hs = s1
    for blk in model.blocks:
        hs = blk(hs, cu_seqlens=cu, position_embeddings=(cos, sin))
    s5 = model.merger.norm(hs)                            # the strict-LN stage:
    # merger.norm = LayerNorm(1152) NON-affine eps 1e-6 (use_postshuffle_norm
    # = False: LN runs per patch row BEFORE the view-merge to 4608) — the
    # golden the Swift port binds STRICT (1e-5): Swift fp32 accumulation vs
    # torch fp32 must be adjacent, proving the LN law (variance = biased
    # population, mean subtracted BEFORE variance — the fused-rms class of
    # bugs). The merge itself is a pure view; fc1→GELU(erf)→fc2 after it.
    s4 = model.merger(hs)
    assert s4.shape == (n // (MERGE * MERGE), 5120), s4.shape
    # s5x: the STRICT LN binding on the KNOWN input s1 (the 27-block tower's
    # error must not ride into the LN gate; Swift binds LN(s1) ≤1e-5):
    s5x = model.merger.norm(s1.to(torch.float16).to(torch.float32))

    tag = f"{t_}x{h_}x{w_}"
    # THE GOLDEN-INPUT LAW: the input slab ships as a golden too (vis_in_*,
    # fp32) — the Swift reference port READS the same bytes, so "identical
    # inputs" is mechanical; no RNG-replication risk rides on the parity
    # (a numpy-RandomState stream the Swift side cannot regenerate would
    # have made every stage assertion unverifiable).
    stages = {"in": x.numpy().astype("<f4"), "s0": s0, "s1": s1,
              "cos": cos, "sin": sin, "s3": s3, "s5": s5, "s5x": s5x, "s4": s4}
    out_rec = {}
    for name, t_ in stages.items():
        b = (t_.tobytes() if name == "in"
               else t_.numpy().astype(np.float32).tobytes() if name == "s5x"
               else t_.numpy().astype(np.float16).tobytes())
        open(os.path.join(OUT, f"vis_{name}_{tag}.bin"), "wb").write(b)
        out_rec[name] = {"shape": list(t_.shape), "sha": sha(b)}
    out_rec["cu_seqlens"] = cu.tolist()
    out_rec["thw"] = thw
    index["grids"][tag] = out_rec

    # strict gate self-check on the interpolation stage: s1 vs (s0 + the
    # C19-BIN-recomputed pos-embed) must be ≤1e-5 (the bins are read back
    # from the golden files the Swift port consumes — same table, same law)
    ib = np.fromfile(os.path.join(OUT, f"mm_interp_idx_{tag}.bin"), dtype=np.int32) \
        if os.path.exists(os.path.join(OUT, f"mm_interp_idx_{tag}.bin")) else None
    if ib is not None:
        ib = ib.reshape(-1, 4)                                     # (n,4) i32 bins
        wb = np.fromfile(os.path.join(OUT, f"mm_interp_w_{tag}.bin"), dtype=np.float32).reshape(-1, 4)
        table = model.pos_embed.weight.detach().numpy()          # [2304,1152] f32
        pe = np.zeros((len(ib), 1152), np.float32)
        for p in range(len(ib)):
            acc = np.zeros(1152, np.float32)
            for tap in range(4):
                acc = acc + table[ib[p, tap]] * wb[p, tap]
            pe[p] = acc
        d = np.abs(pe - (s1.numpy() - s0.numpy())).max()   # compare against the
        # pos-embed ONLY: s1 = s0 + pos_embeds; comparing pe against s1 would
        # charge the patch-embed output (±3) against a 1e-5 gate — a ref bug,
        # not a bins bug (caught live: bins were right all along).
        out_rec["interp_bins_crosscheck_maxabs"] = float(d)
        assert d < 1e-5, f"interp stage cross-check failed at {tag}: {d}"

# ── determinism self-check (C17 law): recompute the last grid end-to-end ──
thw = GRIDS[-1]; t_, h_, w_ = thw; n = t_ * h_ * w_
rng = np.random.RandomState(18 + len(GRIDS) - 1)
x2 = torch.from_numpy((rng.randn(n, 1536) * 0.02).astype(np.float32))
gt = torch.tensor([thw], dtype=torch.int64)
s0 = model.patch_embed(x2)
ii, iw = get_vision_interpolation_indices_and_weights(gt, num_grid_per_side=model.num_grid_per_side,
    mode=model.interpolation_mode, align_corners=model.interpolation_align_corners, spatial_merge_size=MERGE)
s1 = s0 + (model.pos_embed(ii) * iw[:, :, None]).sum(1)
pos_ids = get_vision_position_ids(gt, MERGE)
cos, sin = model.rotary_pos_emb(s1, pos_ids)              # the model's own call shape
cu, maxl = get_vision_attention_seqlens(gt, model.config)
hs = s1
for blk in model.blocks:
    hs = blk(hs, cu_seqlens=cu, position_embeddings=(cos, sin))
s4b = model.merger(hs).numpy().astype(np.float16).tobytes()
tag = f"{t_}x{h_}x{w_}"
assert sha(s4b) == index["grids"][tag]["s4"]["sha"], "determinism violated (merger)"

json.dump(index, open(os.path.join(OUT, "vis_index.json"), "w"), separators=(",", ":"), sort_keys=True)
print(json.dumps({"grids": list(index["grids"]),
                  "interp_crosscheck_maxabs": {g: index["grids"][g].get("interp_bins_crosscheck_maxabs") for g in index["grids"] if "interp_bins_crosscheck_maxabs" in index["grids"][g]}}))
print("OK: C18 goldens written to", OUT)
