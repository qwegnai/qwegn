# Out-of-core PLE N-gram table scanner, sector-aligned QNGRAM binary builder,
# and standalone auditor for Qwen3.8-Flash-Next (FN-AD-18, Spec 23).

import fcntl
import hashlib
import json
import re
import struct
from pathlib import Path

PLE_NGRAM_MARK = ".ple.ple_embedding.ngram_embedding."
PLE_LOCATION_NAME = "ple_ngram_locations.json"
NGRAM_BIN_NAME = "ngram.bin"
HEAD_DIM = 160
NGRAM_ROW_BYTES = HEAD_DIM * 2

_NG_MAGIC = b"QNGRAM\x00\x00"
# Header version and dtype pairing: v1 is BF16, v2 is scalar group quantised,
# v3 carries an e-means codebook at byte 1024..2048 (VQ2) or 1024..3072 (VQ4).
_NG_VERSION = {"bf16": 1, "q8": 2, "q4": 2, "vq2": 3, "vq4": 3}
_NG_DTYPE = {"bf16": 0, "q8": 1, "q4": 2, "vq2": 3, "vq4": 4}
_NG_QUANT_BY_DTYPE = {v: k for k, v in _NG_DTYPE.items()}
_NG_SECTOR = 4096
_NG_GROUP = 32
_NG_NOTE_LEN = 64
_NG_VQ2_CB_OFF = 1024
_NG_VQ2_CB_BYTES = 1024
_NG_VQ4_CB_OFF = 1024
_NG_VQ4_CB_BYTES = 2048
# The largest finite f16. A Q8 group's scale is `amax / 127` and it is stored
# as f16, so a group above this cannot be represented.
_F16_MAX = 65504.0
# The smallest NORMAL f16 (2^-14). A Q8 group's scale below this is
# subnormal: it is kept, and the group is reconstructed with the scale's
# own f16 rounding. The group peak that follows from it is
# `127 * _F16_MIN_NORMAL`; on the real table 0.129% of the groups fall
# below it, and the cost is measured rather than assumed.
_F16_MIN_NORMAL = 6.103515625e-05
_NG_MIN_PEAK = 127.0 * _F16_MIN_NORMAL
# The BF16 rows-per-sector. The quantised formats pack more rows per sector
# (see `_row_bytes`); this constant is the BF16 one.
_NG_RPS = _NG_SECTOR // NGRAM_ROW_BYTES

_ROUTER_OK = re.compile(
    r"^model\.(?:language_model\.)?layers\.\d+\."
    r"ple\.ple_embedding\.(layer_multipliers|"
    r"ngram_heads_offsets|ngram_heads_vocab_sizes)$"
)


def is_ple_ngram(name: str) -> bool:
    return PLE_NGRAM_MARK in name


def scan_ple_and_router(files: list[Path], shard_names: dict,
                        expected_shapes: dict, source: Path):
    ple_shards = []
    ple_seen = {}
    i64_constants = {}
    for path in files:
        with path.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(hlen))
        for name in shard_names[path]:
            e = head[name]
            if e["dtype"] == "I64" and not is_ple_ngram(name):
                if not _ROUTER_OK.match(name):
                    raise ValueError(
                        f"{name}: I64 tensor outside PLE router allow-list")
                lo, hi = e["data_offsets"]
                i64_constants[name] = {
                    "file": path.name, "key": name, "shape": list(e["shape"]),
                    "dtype": "I64", "bytes": hi - lo, "base_off": 8 + hlen + lo,
                }
                del expected_shapes[name]
                continue
            if not is_ple_ngram(name):
                continue
            dtype, shape = e["dtype"], e["shape"]
            lo, hi = e["data_offsets"]
            if dtype != "BF16":
                raise ValueError(
                    f"{name}: PLE ngram shard dtype {dtype} — expected BF16")
            if len(shape) != 2 or shape[1] != 160:
                raise ValueError(
                    f"{name}: PLE ngram shard shape {shape} — expected [rows, 160]")
            span = hi - lo
            if span % 320 or span // 320 != shape[0]:
                raise ValueError(
                    f"{name}: span {span} B != {shape[0]} * 320")
            idx = int(name.split("shard_")[1].split(".")[0])
            if idx in ple_seen:
                raise ValueError(f"PLE ngram shard_{idx} appears twice")
            ple_seen[idx] = True
            ple_shards.append((idx, {
                "file": path.name, "key": name, "shape": list(shape),
                "dtype": dtype, "row_bytes": 320, "base_off": 8 + hlen + lo,
            }))
            del expected_shapes[name]
    if sorted(ple_seen) != list(range(len(ple_seen))):
        raise ValueError(
            f"PLE ngram shard indices not 0..N-1 ({sorted(ple_seen)[:5]}…)")
    for name, d in i64_constants.items():
        p = source / d["file"]
        n = d["bytes"] // 8
        if d["bytes"] % 8 or d["shape"][0] != n:
            raise ValueError(f"{name}: byte span {d['bytes']} vs shape {d['shape']}")
        with p.open("rb") as f:
            f.seek(d["base_off"])
            raw = f.read(d["bytes"])
        if len(raw) != d["bytes"]:
            raise ValueError(f"{name}: short read of router constants")
        d["values"] = list(struct.unpack(f"<{n}q", raw))
    ple_shards.sort(key=lambda t: t[0])
    rows_base = 0
    ordered_shards = []
    for idx, d in ple_shards:
        d["shard"] = idx
        d["first_row"] = rows_base
        rows_base += d["shape"][0]
        ordered_shards.append(d)
    return ordered_shards, i64_constants


def write_ple_location_file(shards: list[dict], out_path: Path,
                            router_constants=None) -> dict:
    payload = {
        "version": 2 if router_constants else 1,
        "row_bytes": 320,
        "total_rows": sum(d["shape"][0] for d in shards),
        "shards": [{"shard": i, **d} for i, d in enumerate(shards)],
    }
    if router_constants:
        layers = {}
        for name in sorted(router_constants):
            d = router_constants[name]
            lay = int(name.split("layers.")[1].split(".")[0])
            field = name.rsplit(".", 1)[1]
            entry = layers.setdefault(lay, {})
            if field in entry:
                raise ValueError(
                    f"router constant {field} given twice for layer {lay}")
            entry[field] = d["values"]
        for lay, entry in layers.items():
            missing = {"layer_multipliers", "ngram_heads_offsets",
                       "ngram_heads_vocab_sizes"} - set(entry)
            if missing:
                raise ValueError(
                    f"layer {lay} router constants incomplete: {sorted(missing)}")
        payload["router_constants"] = {
            "layers": {str(k): layers[k] for k in sorted(layers)}
        }
    Path(out_path).write_bytes(
        (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode())
    return payload


def _row_bytes(dtype: str, head_dim: int) -> int:
    """The bytes one row of `head_dim` values takes in `dtype`.

    The quantised formats are **size-preserving** with respect to their own
    group law (Law L8): a row is exactly `groups * (payload + scale)` bytes
    and never inflates.
    """
    if head_dim <= 0 or head_dim % _NG_GROUP:
        raise ValueError(
            f"ngram.bin: head dim {head_dim} is not a positive whole number "
            f"of groups")
    groups = head_dim // _NG_GROUP
    if dtype == "bf16":
        return head_dim * 2
    if dtype == "q8":
        return groups * (32 + 2)
    if dtype == "q4":
        return groups * 20
    if dtype == "vq2":
        return groups * 18
    if dtype == "vq4":
        return groups * 10
    raise ValueError(f"ngram.bin: unknown row format {dtype!r}")


def _rows_per_sector(row_bytes: int) -> int:
    """The rows that fit in one sector.

    A row must fit: a table is sector-packed, so a row larger than a sector
    would leave **zero** rows per sector and every row index would divide
    by zero.
    """
    if row_bytes <= 0 or row_bytes > _NG_SECTOR:
        raise ValueError(
            f"ngram.bin: row size {row_bytes} does not fit a sector")
    return _NG_SECTOR // row_bytes


def _bf16_rows_to_f32(raw: bytes, np):
    """BF16 little-endian row bytes -> a float32 array of (rows, HEAD_DIM)."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype("<u4") << np.uint32(16)
    return u32.reshape(-1, HEAD_DIM).view("<f4")


def _encode_q8(rows_f32, np) -> bytes:
    """Quantise rows to Q8_G32: per group of 32, 32 int8 codes and one f16 scale.

    The scale law is **symmetric**: `scale = amax / 127` rounded to f16, the
    codes are `rint(value / scale)` clipped to [-127, 127], and the reader
    multiplies the code by the stored f16 scale. That one rule is the whole
    contract: `value = code * scale`, with no special case for the reader.

    Two degenerate groups are worth naming. A group whose scale is **zero**
    (an all-zero group) stores zero codes and a zero scale. A group whose
    scale is **subnormal** keeps its codes and its subnormal scale: the
    codes were computed against the f32 scale, so the group is reconstructed
    with the scale's own f16 rounding — a 0.05% scale error for a normal
    scale, and, for a subnormal one, an error bounded by the group's own
    peak. On the real table that band is 0.129% of the groups and costs the
    worst affected row 0.5% of its norm, against 27% if such a group were
    stored empty instead.
    """
    n = rows_f32.shape[0]
    groups = HEAD_DIM // _NG_GROUP
    x = rows_f32.reshape(n, groups, _NG_GROUP)
    amax = np.abs(x).max(axis=-1)
    scale32 = (amax / np.float32(127.0)).astype(np.float32)
    if (scale32 > np.float32(_F16_MAX)).any():
        # The scale is stored as f16, so a group whose amax/127 does not fit
        # cannot be represented at all. Say so instead of writing an
        # infinite scale that would decode to garbage.
        raise ValueError("ngram.bin: a row's scale does not fit f16")
    scale16 = scale32.astype(np.float16)
    live = scale16.astype(np.float32) > 0
    safe = np.where(live, scale32, np.float32(1.0)).astype(np.float32)
    codes = np.rint(x / safe[..., None]).clip(-127, 127).astype(np.int8)
    codes = np.where(live[..., None], codes, np.int8(0)).astype(np.int8)
    out = np.empty((n, groups, _NG_GROUP + 2), dtype=np.uint8)
    out[..., :_NG_GROUP] = codes.view(np.uint8).reshape(n, groups, _NG_GROUP)
    out[..., _NG_GROUP:] = scale16.view(np.uint8).reshape(n, groups, 2)
    return out.tobytes()


def _encode_q4(rows_f32, np) -> bytes:
    """Quantise rows to Q4_G32: one Block_HQ4_G32 (20 B) per group of 32.

    The block layout is the repo's, written by `hq4.quant_hq4_blocks` — the
    same function the weight packer uses — but WITHOUT the Walsh-Hadamard
    rotation: a looked-up row is *added* to the residual stream and never
    multiplied by a weight matrix, so there is nothing to rotate it against
    and the rotation would have to be undone on every read.
    """
    from hq4 import quant_hq4_blocks
    return quant_hq4_blocks(rows_f32, rotate=False).tobytes()


def _canonical_vq2_codebook(np):
    """Deterministic 256-point unit-disk fallback when table has < 8*256 live pairs."""
    cb = np.zeros((256, 2), dtype=np.float32)
    rings = [(6, 0.16), (13, 0.32), (20, 0.47), (28, 0.61),
             (38, 0.74), (48, 0.86), (52, 0.94), (50, 0.995)]
    idx = 1
    for count, radius in rings:
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
        cb[idx:idx + count, 0] = radius * np.cos(angles)
        cb[idx:idx + count, 1] = radius * np.sin(angles)
        idx += count
    return cb.astype(np.float16)


def train_vq2_codebook(rows_f32, seed: int = 0, device: str = "cpu"):
    """Fit a [256, 2] f16 unit-disk e-means codebook with codebook[0] = (0, 0)."""
    import numpy as np
    from emeans import EmeansRegime, fit_emeans

    groups = rows_f32.shape[1] // _NG_GROUP
    xg = rows_f32.reshape(-1, groups, _NG_GROUP // 2, 2)
    pair_norms = np.linalg.norm(xg, axis=-1, keepdims=True)
    gmax = pair_norms.max(axis=2, keepdims=True)
    live_mask = (gmax.squeeze(-1).squeeze(-1) > 0)
    if int(live_mask.sum()) * (_NG_GROUP // 2) < 8 * 256:
        return _canonical_vq2_codebook(np)
    unit_pairs = (xg[live_mask] / gmax[live_mask]).reshape(-1, 2)
    cb = fit_emeans(unit_pairs, k=256, regime=EmeansRegime.IN_SAMPLE_STATIC,
                    epochs=2.5, seed=seed, device=device, batch_mode="valley")
    lut = cb.centroids.copy()
    zero_slot = int(np.argmin((lut * lut).sum(axis=1)))
    lut[zero_slot] = lut[0]
    lut[0] = 0.0
    lut16 = lut.astype(np.float16)
    lut16[0] = np.float16(0.0)
    return lut16


def _encode_vq2(rows_f32, codebook_f16, np, device: str = "cpu") -> bytes:
    """Quantise rows to VQ2_G32: 16 uint8 2D-pair codes + 1 f16 scale per 32 lanes."""
    import torch

    n, dim = rows_f32.shape
    groups = dim // _NG_GROUP
    pairs = _NG_GROUP // 2
    if device == "mps" and torch.backends.mps.is_available():
        with torch.no_grad():
            dev = torch.device("mps")
            xt = torch.from_numpy(np.ascontiguousarray(rows_f32, dtype=np.float32)).to(device=dev)
            xg = xt.reshape(n, groups, pairs, 2)
            s0 = torch.linalg.vector_norm(xg, dim=-1).amax(dim=-1)
            if bool((s0 > _F16_MAX).any().item()):
                raise ValueError("ngram.bin: a row's scale does not fit f16")
            cb = torch.from_numpy(np.ascontiguousarray(codebook_f16, dtype="<f2").astype(np.float32)).to(device=dev)
            cb_n2 = (cb * cb).sum(dim=-1)
            scale16 = s0.to(torch.float16)
            xg_flat = xt.reshape(n, groups, _NG_GROUP)
            total_pairs = n * groups * pairs
            bs = 524_288
            idx_buf = torch.empty(total_pairs, device=dev, dtype=torch.int64)
            for _ in range(2):
                s_f32 = scale16.to(torch.float32)
                live = s_f32 > 0
                s_safe = torch.where(live, s_f32, torch.ones_like(s_f32))
                normed = (xg / s_safe[:, :, None, None]).reshape(total_pairs, 2)
                for i in range(0, total_pairs, bs):
                    ub = normed[i : i + bs]
                    idx_buf[i : i + bs] = (cb_n2[None, :] - 2.0 * (ub @ cb.t())).argmin(dim=-1)
                idx = torch.where(live[:, :, None], idx_buf.reshape(n, groups, pairs), 0)
                u = cb[idx].reshape(n, groups, _NG_GROUP)
                denom = (u * u).sum(dim=-1)
                s_ls = torch.where(denom > 0, (xg_flat * u).sum(dim=-1) / denom.clamp(min=1e-30), torch.zeros_like(denom))
                s_ls = s_ls.clamp(min=0.0, max=_F16_MAX)
                scale16 = torch.where(s0 > 0, s_ls.to(torch.float16), torch.zeros_like(scale16))
            s_final_f32 = scale16.to(torch.float32)
            live_final = s_final_f32 > 0
            s_final = torch.where(live_final, s_final_f32, torch.ones_like(s_final_f32))
            normed = (xg / s_final[:, :, None, None]).reshape(total_pairs, 2)
            for i in range(0, total_pairs, bs):
                ub = normed[i : i + bs]
                idx_buf[i : i + bs] = (cb_n2[None, :] - 2.0 * (ub @ cb.t())).argmin(dim=-1)
            codes_t = torch.where(live_final[:, :, None], idx_buf.reshape(n, groups, pairs), 0).to(torch.uint8).cpu().numpy()
            sc_u8 = scale16.cpu().numpy().view(np.uint8).reshape(n, groups, 2)
            out = np.empty((n, groups, pairs + 2), dtype=np.uint8)
            out[..., :pairs] = codes_t
            out[..., pairs:] = sc_u8
            return out.tobytes()

    from emeans import EmeansCodebook, assign_emeans

    xg = rows_f32.reshape(n, groups, pairs, 2)
    s0 = np.linalg.norm(xg, axis=-1).max(axis=-1)
    if (s0 > np.float32(_F16_MAX)).any():
        raise ValueError("ngram.bin: a row's scale does not fit f16")
    lut32 = codebook_f16.astype(np.float32)
    cb_obj = EmeansCodebook(centroids=lut32, biases=None, train_mse=0.0, dead_revivals=0)
    scale16 = s0.astype(np.float16)
    xg_flat = rows_f32.reshape(n, groups, _NG_GROUP)
    for _ in range(2):
        s_safe = np.where(scale16.astype(np.float32) > 0,
                          scale16.astype(np.float32), np.float32(1.0))
        normed = (xg / s_safe[:, :, None, None]).reshape(-1, 2)
        idx = assign_emeans(normed, cb_obj, device=device).reshape(n, groups, pairs)
        idx = np.where((scale16.astype(np.float32) > 0)[:, :, None], idx, 0)
        u = lut32[idx].reshape(n, groups, _NG_GROUP)
        denom = (u * u).sum(axis=-1)
        s_ls = np.where(denom > 0, (xg_flat * u).sum(axis=-1) / np.maximum(denom, 1e-30), 0.0)
        s_ls = np.clip(s_ls, 0.0, _F16_MAX).astype(np.float32)
        scale16 = np.where(s0 > 0, s_ls.astype(np.float16), np.float16(0.0))
    s_final = np.where(scale16.astype(np.float32) > 0,
                       scale16.astype(np.float32), np.float32(1.0))
    codes = assign_emeans((xg / s_final[:, :, None, None]).reshape(-1, 2),
                          cb_obj, device=device).reshape(n, groups, pairs).astype(np.uint8)
    codes = np.where((scale16.astype(np.float32) > 0)[:, :, None], codes, np.uint8(0))
    out = np.empty((n, groups, pairs + 2), dtype=np.uint8)
    out[..., :pairs] = codes
    out[..., pairs:] = scale16.view(np.uint8).reshape(n, groups, 2)
    return out.tobytes()


def decode_vq2_rows(raw_bytes: bytes, codebook_f16, rows: int, head_dim: int = HEAD_DIM):
    """Exact IEEE-754 binary32 VQ2_G32 reader: float32(lut[code]) * float32(scale)."""
    import numpy as np

    groups = head_dim // _NG_GROUP
    pairs = _NG_GROUP // 2
    lut32 = np.asarray(codebook_f16, dtype="<f2").astype(np.float32).reshape(256, 2)
    enc = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rows, groups, pairs + 2)
    codes = enc[:, :, :pairs]
    scales = np.ascontiguousarray(enc[:, :, pairs:]).view("<f2").reshape(rows, groups).astype(np.float32)
    vals = lut32[codes] * scales[:, :, None, None]
    return vals.reshape(rows, head_dim)


def _build_mps_voronoi_ctx(codebook_f16, grid_res: int = 2048, lim: float = 1.125):
    """Precompute an SLC-resident [grid_res, grid_res] uint8 2D Voronoi table on MPS."""
    import numpy as np
    import torch

    dev = torch.device("mps")
    coords = np.linspace(-lim, lim, grid_res, dtype=np.float32)
    gy, gx = np.meshgrid(coords, coords, indexing="ij")
    pts = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], axis=-1)).to(device=dev)
    cbt = torch.from_numpy(np.ascontiguousarray(codebook_f16, dtype="<f2").astype(np.float32)).to(device=dev)
    cb_n2 = (cbt * cbt).sum(dim=-1)
    vor = torch.empty(grid_res * grid_res, device=dev, dtype=torch.uint8)
    with torch.no_grad():
        for i in range(0, grid_res * grid_res, 65536):
            ub = pts[i : i + 65536]
            vor[i : i + 65536] = (cb_n2[None, :] - 2.0 * (ub @ cbt.t())).argmin(dim=-1).to(torch.uint8)
    return {"vor": vor, "cbt": cbt, "grid_res": grid_res, "lim": lim}


def _encode_vq2_voronoi_mps(rows_f32, ctx: dict, np) -> bytes:
    import torch

    n, dim = rows_f32.shape
    groups = dim // _NG_GROUP
    pairs = _NG_GROUP // 2
    g_res = ctx["grid_res"]
    lim = ctx["lim"]
    vor = ctx["vor"]
    cbt = ctx["cbt"]
    scale_g = (g_res - 1) / (2.0 * lim)
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(rows_f32, dtype=np.float32)).to(device=vor.device)
        xg = xt.reshape(n, groups, pairs, 2)
        s0 = torch.linalg.vector_norm(xg, dim=-1).amax(dim=-1)
        if bool((s0 > _F16_MAX).any().item()):
            raise ValueError("ngram.bin: a row's scale does not fit f16")
        sc16 = s0.to(torch.float16)
        xg_flat = xt.reshape(n, groups, _NG_GROUP)
        idx = None
        for it in range(3):
            sf = sc16.to(torch.float32)
            live = sf > 0
            ss = torch.where(live, sf, torch.ones_like(sf))
            u2 = xg / ss[:, :, None, None]
            ix = ((u2[..., 0] + lim) * scale_g).round().clamp(0, g_res - 1).to(torch.int64)
            iy = ((u2[..., 1] + lim) * scale_g).round().clamp(0, g_res - 1).to(torch.int64)
            idx = torch.where(live[:, :, None], vor[iy * g_res + ix].to(torch.int64), 0)
            if it < 2:
                u = cbt[idx].reshape(n, groups, _NG_GROUP)
                den = (u * u).sum(dim=-1)
                sls = torch.where(den > 0, (xg_flat * u).sum(dim=-1) / den.clamp(min=1e-30), torch.zeros_like(den))
                sls = sls.clamp(min=0.0, max=_F16_MAX)
                sc16 = torch.where(s0 > 0, sls.to(torch.float16), torch.zeros_like(sc16))
        out = np.empty((n, groups, pairs + 2), dtype=np.uint8)
        out[..., :pairs] = idx.to(torch.uint8).cpu().numpy()
        out[..., pairs:] = sc16.cpu().numpy().view(np.uint8).reshape(n, groups, 2)
        return out.tobytes()


def _encode_rows(quant: str, bf16_bytes: bytes, codebook_f16=None, device: str = "cpu", voronoi_ctx=None) -> bytes:
    """Re-encode a run of BF16 rows into the requested row format."""
    import numpy as np
    if quant == "vq2_g32":
        quant = "vq2"
    elif quant == "vq4_g32":
        quant = "vq4"
    rows = _bf16_rows_to_f32(bf16_bytes, np)
    if not np.isfinite(rows).all():
        raise ValueError("ngram.bin: a non-finite value in an n-gram row")
    if quant == "q8":
        return _encode_q8(rows, np)
    if quant == "q4":
        return _encode_q4(rows, np)
    if quant == "vq2":
        if codebook_f16 is None:
            raise ValueError("ngram.bin: vq2 encoding requires a trained codebook")
        if voronoi_ctx is not None:
            return _encode_vq2_voronoi_mps(rows, voronoi_ctx, np)
        return _encode_vq2(rows, codebook_f16, np, device=device)
    if quant == "vq4":
        if codebook_f16 is None:
            raise ValueError("ngram.bin: vq4 encoding requires a trained codebook")
        from pack_ngram_vq4 import encode_vq4
        return encode_vq4(rows, codebook_f16, device=device)
    raise ValueError(f"ngram.bin: unknown row format {quant!r}")


def _sample_shard_rows_for_vq2(shards: list[dict], source: Path, max_rows: int = 16_384):
    import numpy as np
    total = sum(d["shape"][0] for d in shards)
    take_per_shard = [max(1, int(round(max_rows * d["shape"][0] / max(total, 1)))) for d in shards]
    chunks = []
    for d, want in zip(shards, take_per_shard):
        rows = d["shape"][0]
        step = max(1, rows // max(want, 1))
        picks = range(0, rows, step)
        with (Path(source) / d["file"]).open("rb") as f:
            for r in list(picks)[:want]:
                f.seek(d["base_off"] + r * NGRAM_ROW_BYTES)
                buf = f.read(NGRAM_ROW_BYTES)
                if len(buf) == NGRAM_ROW_BYTES:
                    chunks.append(_bf16_rows_to_f32(buf, np))
    arr = np.concatenate(chunks, axis=0)
    if not np.isfinite(arr).all():
        raise ValueError("ngram.bin: a non-finite value in an n-gram row")
    return arr


def _validate_vq2_codebook_bytes(cb_bytes: bytes) -> None:
    import numpy as np
    if len(cb_bytes) != _NG_VQ2_CB_BYTES:
        raise ValueError("ngram.bin: v3 codebook slice short")
    cb = np.frombuffer(cb_bytes, dtype="<f2").reshape(256, 2)
    if not np.isfinite(cb).all():
        raise ValueError("ngram.bin: non-finite value in v3 codebook")
    raw_u16 = np.frombuffer(cb_bytes[:4], dtype="<u2")
    if int(raw_u16[0]) != 0 or int(raw_u16[1]) != 0:
        raise ValueError("ngram.bin: v3 codebook[0] must be canonical +0.0")
    if not (np.abs(cb[1:]) > 0).any():
        raise ValueError("ngram.bin: v3 codebook is all zero")


def build_ngram_bin(shards: list[dict], source: Path, out_path: Path,
                    quant: str = "bf16", l1_seed: int = 0,
                    codebook_f16=None, device: str | None = None) -> dict:
    """Write the sector-aligned QNGRAM table (`bf16`, `q8`, `q4`, `vq2`, or `vq4`)."""
    out_path = Path(out_path)
    if quant == "vq2_g32":
        quant = "vq2"
    elif quant == "vq4_g32":
        quant = "vq4"
    if quant not in _NG_DTYPE:
        raise ValueError(f"ngram.bin: unknown row format {quant!r}")
    if device is None:
        if quant in ("vq2", "vq4"):
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "cpu"
    entry_count = sum(d["shape"][0] for d in shards)
    if entry_count <= 0:
        raise ValueError("ngram.bin: empty table")
    if l1_seed > entry_count:
        raise ValueError(f"ngram.bin: l1_seed {l1_seed} > {entry_count}")
    if quant == "vq2" and codebook_f16 is None:
        sample = _sample_shard_rows_for_vq2(shards, source)
        codebook_f16 = train_vq2_codebook(sample, seed=0, device=device)
    elif quant == "vq4" and codebook_f16 is None:
        from pack_ngram_vq4 import train_vq4_codebook
        sample = _sample_shard_rows_for_vq2(shards, source)
        codebook_f16 = train_vq4_codebook(sample, seed=0, device=device)
    if quant == "vq2":
        import numpy as np
        cb_bytes = np.asarray(codebook_f16, dtype="<f2").tobytes()
        _validate_vq2_codebook_bytes(cb_bytes)
        voronoi_ctx = (
            _build_mps_voronoi_ctx(codebook_f16)
            if (device == "mps" and entry_count >= 1_000_000)
            else None
        )
    elif quant == "vq4":
        import numpy as np
        from pack_ngram_vq4 import validate_vq4_codebook_bytes
        cb_bytes = np.asarray(codebook_f16, dtype="<f2").tobytes()
        validate_vq4_codebook_bytes(cb_bytes)
        voronoi_ctx = None
    else:
        cb_bytes = b""
        voronoi_ctx = None
    row_bytes = _row_bytes(quant, HEAD_DIM)
    rps = _rows_per_sector(row_bytes)
    sectors = (entry_count + rps - 1) // rps
    hn = hashlib.sha256()
    sector = bytearray(_NG_SECTOR)
    fill = 0
    written = 0
    chunk_rows = (rps * 2048) if quant in ("vq2", "vq4") else 4096
    import numpy as _np_sec
    with out_path.open("wb") as out:
        try:
            fcntl.fcntl(out.fileno(), fcntl.F_NOCACHE, 1)
        except OSError:
            pass
        out.write(b"\x00" * _NG_SECTOR)
        for d in shards:
            rows = d["shape"][0]
            if d["row_bytes"] != NGRAM_ROW_BYTES:
                raise ValueError(
                    f"ngram.bin: row_bytes {d['row_bytes']} != {NGRAM_ROW_BYTES}")
            with (Path(source) / d["file"]).open("rb") as f:
                f.seek(d["base_off"])
                remaining = rows
                while remaining:
                    want = min(remaining, chunk_rows)
                    buf = f.read(want * NGRAM_ROW_BYTES)
                    if len(buf) != want * NGRAM_ROW_BYTES:
                        raise ValueError(
                            f"ngram.bin: short read in {d['file']}")
                    enc = (buf if quant == "bf16"
                           else _encode_rows(quant, buf, codebook_f16=codebook_f16,
                                             device=device, voronoi_ctx=voronoi_ctx))
                    hn.update(enc)
                    pos = 0
                    if fill == 0 and (len(enc) // row_bytes) >= rps:
                        full_secs = (len(enc) // row_bytes) // rps
                        bulk_bytes = full_secs * rps * row_bytes
                        sec_arr = _np_sec.zeros((full_secs, _NG_SECTOR), dtype=_np_sec.uint8)
                        sec_arr[:, : rps * row_bytes] = _np_sec.frombuffer(
                            enc, dtype=_np_sec.uint8, count=bulk_bytes
                        ).reshape(full_secs, rps * row_bytes)
                        out.write(sec_arr.tobytes())
                        written += full_secs * _NG_SECTOR
                        pos = bulk_bytes
                    while pos < len(enc):
                        take = min((len(enc) - pos) // row_bytes, rps - fill)
                        sector[fill * row_bytes:(fill + take) * row_bytes] = (
                            enc[pos:pos + take * row_bytes]
                        )
                        fill += take
                        pos += take * row_bytes
                        if fill == rps:
                            out.write(bytes(sector))
                            written += _NG_SECTOR
                            fill = 0
                    remaining -= want
        if fill:
            for k in range(fill * row_bytes, _NG_SECTOR):
                sector[k] = 0
            out.write(bytes(sector))
            written += _NG_SECTOR
        if written != sectors * _NG_SECTOR:
            raise ValueError(f"ngram.bin: wrote {written} != {sectors * _NG_SECTOR}")
        note = hn.hexdigest().encode()
        header = bytearray(_NG_SECTOR)
        header[0:8] = _NG_MAGIC
        struct.pack_into("<HHIQHHI", header, 8,
                         _NG_VERSION[quant], 0, _NG_DTYPE[quant], entry_count,
                         HEAD_DIM, row_bytes, l1_seed)
        struct.pack_into("<Q", header, 32, _NG_SECTOR)
        header[44] = _NG_NOTE_LEN
        header[45:45 + _NG_NOTE_LEN] = note
        if cb_bytes:
            header[_NG_VQ2_CB_OFF:_NG_VQ2_CB_OFF + len(cb_bytes)] = cb_bytes
        out.seek(0)
        out.write(bytes(header))
    return {"entry_count": entry_count, "table_sha256": hn.hexdigest(),
            "bytes": out_path.stat().st_size, "version": _NG_VERSION[quant],
            "dtype": _NG_DTYPE[quant], "row_bytes": row_bytes,
            "rows_per_sector": rps}


def audit_ngram_bin(path: Path) -> dict:
    """Check a QNGRAM table against its own header and the producer's note."""
    path = Path(path)
    with path.open("rb") as f:
        try:
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        except OSError:
            pass
        head = f.read(_NG_SECTOR)
        if len(head) != _NG_SECTOR or head[0:8] != _NG_MAGIC:
            raise ValueError("ngram.bin: magic/header missing")
        version, flags, dtype = struct.unpack_from("<HHI", head, 8)
        if version not in (1, 2, 3):
            raise ValueError(f"ngram.bin: unsupported ver={version} dtype={dtype}")
        if dtype not in _NG_QUANT_BY_DTYPE:
            raise ValueError(f"ngram.bin: unsupported ver={version} dtype={dtype}")
        quant = _NG_QUANT_BY_DTYPE[dtype]
        if version == 2 and quant == "bf16":
            raise ValueError("ngram.bin: v2 must carry a quantised dtype")
        if version != _NG_VERSION[quant]:
            raise ValueError(f"ngram.bin: v{version} cannot carry dtype {dtype}")
        if quant == "vq2":
            _validate_vq2_codebook_bytes(head[_NG_VQ2_CB_OFF:_NG_VQ2_CB_OFF + _NG_VQ2_CB_BYTES])
        elif quant == "vq4":
            from pack_ngram_vq4 import validate_vq4_codebook_bytes
            validate_vq4_codebook_bytes(head[_NG_VQ4_CB_OFF:_NG_VQ4_CB_OFF + _NG_VQ4_CB_BYTES])
        entry_count, dim, row_bytes = struct.unpack_from("<QHH", head, 16)
        l1_hot, = struct.unpack_from("<I", head, 28)
        data_offset, = struct.unpack_from("<Q", head, 32)
        note_len = head[44]
        note = bytes(head[45:45 + _NG_NOTE_LEN]) if note_len == _NG_NOTE_LEN else b""
        if row_bytes != _row_bytes(quant, dim):
            raise ValueError(f"ngram.bin: rowBytes {row_bytes} dim {dim}")
        rps = _rows_per_sector(row_bytes)
        if data_offset < _NG_SECTOR or data_offset % _NG_SECTOR:
            raise ValueError("ngram.bin: dataOffset not sector-aligned")
        if not entry_count or l1_hot > entry_count:
            raise ValueError(f"ngram.bin: entryCount {entry_count} l1Hot {l1_hot}")
        whole = (entry_count - 1) // rps
        tail_rows = entry_count - whole * rps
        min_size = data_offset + whole * _NG_SECTOR + tail_rows * row_bytes
        max_size = data_offset + (whole + 1) * _NG_SECTOR
        size = path.stat().st_size
        if not (min_size <= size <= max_size):
            raise ValueError(
                f"ngram.bin: size law: expected [{min_size}, {max_size}], got {size}")
        h = hashlib.sha256()
        f.seek(data_offset)
        sector_rows = rps * row_bytes
        for _ in range(whole):
            blk = f.read(_NG_SECTOR)
            if len(blk) != _NG_SECTOR:
                raise ValueError("ngram.bin: short sector read")
            h.update(blk[:sector_rows])
        last = f.read(tail_rows * row_bytes)
        if len(last) != tail_rows * row_bytes:
            raise ValueError("ngram.bin: short tail read")
        h.update(last)
        if note != h.hexdigest().encode():
            raise ValueError(
                "ngram.bin: producer note (table sha256) does not match row bytes")
    return {"version": version, "flags": flags, "dtype": dtype,
            "entryCount": entry_count, "embeddingDim": dim, "rowBytes": row_bytes,
            "rowsPerSector": rps, "l1HotCount": l1_hot,
            "dataOffset": data_offset, "table_sha256": h.hexdigest(),
            "bytes": size}
