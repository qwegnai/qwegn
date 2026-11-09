#!/usr/bin/env python3
"""gen_mm_fixtures.py — C19 goldens: multimodal ingestion math.

LIVE transformers truth BEFORE any Swift assertion (parity-port rule 2).
Every golden calls the shipped authority, never a re-implementation:

  A  smart_resize        — factor = patch·merge = 32 (the checkpoint's own
                           preprocessor_config; overrides the legacy 28),
                           pixel budget shortest/longest_edge = 65536 /
                           16777216. Table of (h,w) → (H,W) + grid + token
                           count (the placeholder-repeat law, L-A).
  B  patch-slab law      — the processor's pixel_values row layout, PROBED
                           live and asserted at write time: rows are
                           merge-block ordered — block (hb,wb) of the h×w
                           patch grid, inside the block (i,j) ∈ 2×2;
                           patch pixel origin (hb·32+i·16, wb·32+j·16);
                           each row = for c in 0..2: the SAME 16×16 raster
                           twice (temporal_patch_size=2 frame duplication)
                           = 6 × 256 f32 in [-1,1] (mean/std 0.5). Tolerance
                           1e-5 (processor normalizes on-device; 1-ulp
                           associativity noise — recorded, NOT hidden).
  C  vision rope pos     — vision_utils.get_vision_position_ids (2-axis h/w,
                           merge-block patch order, h/w repeated t times).
  D  pos-embed interp    — get_vision_interpolation_indices_and_weights
                           (bilinear, align_corners=True, side=48, 4 taps/
                           patch, partition-of-unity assert). THE LAW: the
                           authority returns int64 indices; every value is
                           provably < 2^31 (side² = 2304 max) — the Swift
                           port binds lossless i32 bins, so the int64-
                           bridge drift class cannot exist.
  E  attention segments  — get_vision_cu_seqlens: per-FRAME segments
                           (t+1 entries, t segments of h·w; NOT the clip).
  F  mRope 3D positions  — Qwen3_5Model.get_rope_index (the authority that
                           ships; called through the real method on a
                           config-only instance): text spans advance by
                           len, image spans by max(h,w)//merge, VIDEO GRIDS
                           SPLIT per frame (timestamp law: t-rows → t grids
                           of t=1), delta law max(pos)+1−len. Bins for 6
                           mixed sequences (the grid-inversion catchers).
  G  illegal grids       — shape contract: h % merge != 0 raises (expected-
                           raise golden, C17 contract style).

Determinism self-check (C17 law): recompute case B + the mixed rope case,
byte-compare against the written bytes; assert processor byte-determinism.
Authority config copy: Scripts/spike/assets/preprocessor_config.json
(re-pulled from the checkpoint at generation time; SOURCES.json updated).

Out files (one role per file — collision law): mm_index.json (table +
hashes), mm_patches_c0.bin (B), mm_pos_<t>x<h>x<w>.bin / mm_interp_idx_*.bin
(I32 bins — fit asserted) / mm_interp_w_*.bin /
mm_cu_*.bin (C-E), mm_rope_pos_<case>.bin [3,S] i32 + mm_rope_delta_<case>.
bin i32 (F).
"""
import hashlib, itertools, json, os, struct, sys
import numpy as np
import torch
torch.set_grad_enabled(False)
from PIL import Image

from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from transformers.vision_utils import (get_vision_position_ids,
    get_vision_interpolation_indices_and_weights, get_vision_cu_seqlens)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "Scripts", "spike", "out")
ASSETS = os.path.join(ROOT, "Scripts", "spike", "assets")
os.makedirs(OUT, exist_ok=True)
# RELEASE LAW: the ENGINE DIRECTORY is authoritative — weights live in
# Qwen3.8-27B.qwp, every non-weight asset in Qwen3.8-27B.qwp.assets/
# (inherited from the official base, manifest-hashed in provenance).
# Host-side math only here; no weights read. (Processor equivalence vs the
# legacy oQ4e-era config: probe maxd 0.0 on 4 shapes — goldens unchanged.)
CKPT = os.environ.get("QW_CKPT", os.path.expanduser("~/.qwg/models/Qwen3.8-27B-hq4g32/assets"))

proc = AutoProcessor.from_pretrained(CKPT)
IP = proc.image_processor
PATCH = IP.patch_size                     # 16
TPATCH = IP.temporal_patch_size           # 2
MERGE = IP.merge_size                     # 2
FACTOR = PATCH * MERGE                    # 32 — checkpoint factor (law A)
MIN_PIX = int(IP.size["shortest_edge"])   # 65536
MAX_PIX = int(IP.size["longest_edge"])    # 16777216
SIDE = 48                                 # int(2304**0.5): pos-embed table side
# the processor's normalization (asserted, not assumed):
assert list(IP.image_mean) == [0.5, 0.5, 0.5] and list(IP.image_std) == [0.5, 0.5, 0.5]
json.dump(IP.to_dict(), open(os.path.join(ASSETS, "preprocessor_config.json"), "w"),
          indent=1, sort_keys=True, default=str)

def sres(h, w):
    return smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIX, max_pixels=MAX_PIX)

def sha(b): return hashlib.sha256(b).hexdigest()

index = {"preprocessor": {"patch": PATCH, "temporal": TPATCH, "merge": MERGE,
                          "factor": FACTOR, "min_pixels": MIN_PIX, "max_pixels": MAX_PIX,
                          "side": SIDE,
                          "mean_std": [0.5, 0.5, 0.5]},
         "resize": [], "slab": {}, "grids": {}, "rope": {}}

# ── A + L-A: smart_resize table + placeholder repeat-count law ────────────
SIZES = [(448, 448), (1024, 768), (768, 1024), (1920, 1080), (1080, 1920),
         (640, 480), (300, 300), (128, 128), (33, 10000), (512, 505),
         (2048, 2048), (448, 1024), (31, 31), (256, 144), (999, 37)]
for (h, w) in SIZES:
    if max(h, w) / min(h, w) >= 200:
        # the authority's aspect law: ratio ≥ 200 must raise (contract golden)
        try:
            sres(h, w)
            raise AssertionError(f"aspect contract broken: {h}x{w} did not raise")
        except AssertionError:
            raise
        except Exception as e:
            index.setdefault("raise_contracts", []).append(
                {"kind": "aspect", "h": h, "w": w, "raises": type(e).__name__})
        continue
    H, W = sres(h, w)
    thw = [1, H // PATCH, W // PATCH]
    n_patch = thw[0] * thw[1] * thw[2]
    assert n_patch % (MERGE * MERGE) == 0            # grid law (merge-exact)
    assert H % FACTOR == 0 and W % FACTOR == 0       # factor law
    assert MIN_PIX <= H * W <= MAX_PIX               # budget law
    index["resize"].append({"h": h, "w": w, "H": int(H), "W": int(W), "thw": thw,
                            "n_tokens": n_patch // (MERGE * MERGE)})  # L-A

# ── B: patch-slab layout — the law PROBED live, asserted at write time ────
rng = np.random.RandomState(7)
img = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
o1 = IP(images=[Image.fromarray(img)], return_tensors="pt")
o2 = IP(images=[Image.fromarray(img)], return_tensors="pt")
pv = o1["pixel_values"]
assert torch.equal(pv, o2["pixel_values"]), "processor not deterministic"
pv = pv.numpy()
grid = o1["image_grid_thw"].numpy().tolist()
assert grid == [[1, 16, 16]], grid                   # 256→256 (factor-exact), t=1
x = img.astype(np.float32) / 255.0
x = (x - 0.5) / 0.5                                   # mean/std law (asserted above)

def patch_origin(pi):
    """merge-block order: blocks raster (hb outer, wb inner), inside (i,j) 2×2."""
    blocks_w = grid[0][2] // MERGE
    hb, rem = divmod(pi, blocks_w * MERGE * MERGE // MERGE)  # patches-per-block-row = blocks_w*4? see lawB
    # explicit: block row = 4·blocks_w patches
    hb, rem = divmod(pi, 4 * blocks_w)
    wb, p = divmod(rem, 4)
    i, j = divmod(p, 2)
    return hb * 32 + i * 16, wb * 32 + j * 16

def law_b_row(tp, wp):
    """row = for c: [256 raster, 256 raster (temporal dup)]"""
    out = np.zeros(1536, np.float32); k = 0
    blk = x[tp:tp + 16, wp:wp + 16, :]
    for c in range(3):
        r = blk[:, :, c].reshape(-1)
        out[k:k + 256] = r; k += 256
        out[k:k + 256] = r; k += 256
    return out

pvb = pv.astype("<f4")
exact = 0; worst = 0.0
for pi in range(pv.shape[0]):
    tp, wp = patch_origin(pi)
    d = float(np.abs(pvb[pi] - law_b_row(tp, wp)).max())
    worst = max(worst, d)
    if d < 1e-5: exact += 1
assert exact == pv.shape[0], f"slab law B broken: {exact}/{pv.shape[0]} exact-ish, worst {worst}"
with open(os.path.join(OUT, "mm_patches_c0.bin"), "wb") as f:
    f.write(pvb.tobytes())
index["slab"] = {"grid": grid[0], "rows": int(pv.shape[0]), "row_f32": 1536,
                 "sha": sha(pvb.tobytes()), "law_tolerances": {"normalization": worst},
                 "law": "row = merge-block order (block raster hb-outer; inside i,j 2x2; "
                        "origin (hb*32+i*16, wb*32+j*16)); row f32 = for c in 3: raster 256 ×2 "
                        "(temporal dup); ±1 range (mean/std 0.5)"}

# ── C/D/E: per-grid bins ──────────────────────────────────────────────────
GRIDS = [[1, 4, 6], [2, 4, 4], [1, 2, 2], [1, 16, 16], [1, 28, 28], [1, 14, 26], [3, 6, 10]]
# Shape contract (measured live): h % merge != 0 OR w % merge != 0 raises
# inside the authority's block reshape. smart_resize's factor=32 law makes
# this unreachable for model-legal grids — the Swift grid builder asserts it
# anyway (defense-in-depth golden). Two shapes bound:
from transformers.vision_utils import get_vision_attention_seqlens  # noqa: F401 (liveness of the module)
for bad in ([4, 10, 7], [3, 5, 4]):
    try:
        get_vision_position_ids(torch.tensor([bad], dtype=torch.int64), torch.tensor([MERGE]))
        raise AssertionError(f"illegal grid {bad} did NOT raise — shape contract changed")
    except AssertionError:
        raise
    except Exception as e:
        index.setdefault("raise_contracts", []).append(
            {"kind": "merge_shape", "thw": bad, "raises": type(e).__name__})

for thw in GRIDS:
    t_, h_, w_ = thw
    gt = torch.tensor([thw], dtype=torch.int64)
    pi = get_vision_position_ids(gt, torch.tensor([MERGE]))              # [n,2]
    ii, iw = get_vision_interpolation_indices_and_weights(
        gt, num_grid_per_side=SIDE, mode="bilinear", align_corners=True,
        spatial_merge_size=MERGE)
    cu = get_vision_cu_seqlens(gt)
    n = t_ * h_ * w_
    assert pi.shape == (n, 2) and ii.shape == (n, 4) and iw.shape == (n, 4)
    assert cu.shape == (t_ + 1,) and int(cu[0]) == 0 and int(cu[-1]) == n  # per-FRAME law
    ws = iw.sum(1)
    assert (ws - 1.0).abs().max().item() < 1e-5, "bilinear weights not a partition of unity"
    idx_np = ii.numpy()
    assert idx_np.dtype == np.int64 and int(idx_np.max()) < 2**31, "index dtype contract changed"
    # int64→i32 bridge law: values provably < 2^31 (side² = 2304 max) so the
    # Swift port reads lossless i32 bins — the int64-bridge drift class is
    # made impossible by assertion, not by hoping.
    tag = f"{t_}x{h_}x{w_}"
    pb = pi.numpy().astype("<i4").tobytes()
    ib = idx_np.astype("<i4").tobytes()       # i32 bins (fit asserted above)
    wb = iw.numpy().astype("<f4").tobytes()
    cb = cu.numpy().astype("<i4").tobytes()
    open(os.path.join(OUT, f"mm_pos_{tag}.bin"), "wb").write(pb)
    open(os.path.join(OUT, f"mm_interp_idx_{tag}.bin"), "wb").write(ib)
    open(os.path.join(OUT, f"mm_interp_w_{tag}.bin"), "wb").write(wb)
    open(os.path.join(OUT, f"mm_cu_{tag}.bin"), "wb").write(cb)
    index["grids"][tag] = {"thw": thw, "n_patch": n, "n_tokens": n // (MERGE * MERGE),
                           "pos_sha": sha(pb),
                           "idx_i32_fit": True, "idx_max": int(idx_np.max()),
                           "iidx_sha": sha(ib), "iw_sha": sha(wb),
                           "cu": cu.tolist()}

# ── F: mRope 3D positions + deltas THROUGH the real authority method ──────
class VC: spatial_merge_size = MERGE
class CFG: vision_config = VC()
m = Qwen3_5Model.__new__(Qwen3_5Model); m.config = CFG()

IMG, VID = 248056, 248057     # <|image_pad|>, <|video_pad|> (vocab-verified below)
vocab = proc.tokenizer.get_vocab()
assert vocab.get("<|image_pad|>") == IMG and vocab.get("<|video_pad|>") == VID

CASES = {
    "text_only": ([10, 11, 12, 13, 14], [0, 0, 0, 0, 0], [], []),
    "one_image": ([10, 11] + [IMG] * 6 + [12, 13], [0, 0] + [1] * 6 + [0, 0],
                  [[1, 4, 6]], []),
    "two_images": ([10] + [IMG] * 4 + [11, 12] + [IMG] * 4 + [13],
                   [0] + [1] * 4 + [0, 0] + [1] * 4 + [0],
                   [[1, 4, 4], [1, 4, 4]], []),
    "one_video": ([10, 20] + [VID] * 4 + [21] + [VID] * 4 + [11],
                  [0, 0] + [2] * 4 + [0] + [2] * 4 + [0],
                  [], [[2, 4, 4]]),            # timestamp-split: 2 spans of 4
    "mixed": ([10, 11] + [IMG] * 6 + [12, 20] + [VID] * 4 + [21, 13] + [VID] * 4 + [14, 15],
              [0, 0] + [1] * 6 + [0, 0] + [2] * 4 + [0, 0] + [2] * 4 + [0, 0],
              [[1, 4, 6]], [[2, 4, 4]]),
    "image_then_video_nolead": ([IMG] * 4 + [10] + [VID] * 4 + [11],
                                [1] * 4 + [0] + [2] * 4 + [0],
                                [[1, 4, 4]], [[1, 4, 4]]),
}
for name, (ids, types, img_g, vid_g) in CASES.items():
    pos, delta = m.get_rope_index(
        torch.tensor([ids], dtype=torch.int64), torch.tensor([types], dtype=torch.int32),
        image_grid_thw=torch.tensor(img_g, dtype=torch.int64) if img_g else None,
        video_grid_thw=torch.tensor(vid_g, dtype=torch.int64) if vid_g else None)
    S = len(ids)
    assert pos.shape == (3, 1, S) and delta.shape == (1, 1)
    P = pos[:, 0, :].numpy().astype("<i4")
    tt = np.array(types)
    for i in range(S):
        if tt[i] == 0:                                   # text axes law
            assert P[0, i] == P[1, i] == P[2, i], f"{name}: text axis split at {i}"
        elif i > 0 and tt[i] == tt[i - 1] and tt[i] != 0:
            pass                                         # span law covered by bytes
    dl = int(delta.item())
    assert dl == int(P.max()) + 1 - S, f"{name}: delta law {dl} vs {int(P.max()) + 1 - S}"
    pb = P.tobytes()
    open(os.path.join(OUT, f"mm_rope_pos_{name}.bin"), "wb").write(pb)
    open(os.path.join(OUT, f"mm_rope_delta_{name}.bin"), "wb").write(np.int32(dl).tobytes())
    index["rope"][name] = {"S": S, "types": types, "img": img_g, "vid": vid_g,
                           "delta": dl, "pos_sha": sha(pb)}

# ── determinism self-check (C17 law): recompute, byte-compare ─────────────
pv3 = IP(images=[Image.fromarray(img)], return_tensors="pt")["pixel_values"].numpy().astype("<f4")
assert sha(pv3.tobytes()) == index["slab"]["sha"], "determinism violated (slab)"
p2, d2 = m.get_rope_index(
    torch.tensor([CASES["mixed"][0]], dtype=torch.int64),
    torch.tensor([CASES["mixed"][1]], dtype=torch.int32),
    image_grid_thw=torch.tensor([[1, 4, 6]]), video_grid_thw=torch.tensor([[2, 4, 4]]))
assert sha(p2[:, 0, :].numpy().astype("<i4").tobytes()) == index["rope"]["mixed"]["pos_sha"], \
    "determinism violated (rope mixed)"

json.dump(index, open(os.path.join(OUT, "mm_index.json"), "w"), separators=(",", ":"), sort_keys=True)
print(json.dumps({"resize": len(index["resize"]), "slab_worst_norm_noise": index["slab"]["law_tolerances"],
                  "grids": list(index["grids"]), "rope": list(index["rope"]),
                  "illegal": index.get("illegal_grid_contract")}))
print("OK: C19 goldens written to", OUT)
