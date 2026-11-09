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
# The row format of a table, and the header version that carries it. v1 is
# BF16 only; v2 adds the two quantised formats. The pairing is strict: a v1
# file is BF16 and a v2 file is quantised.
_NG_VERSION = {"bf16": 1, "q8": 2, "q4": 2}
_NG_DTYPE = {"bf16": 0, "q8": 1, "q4": 2}
_NG_QUANT_BY_DTYPE = {v: k for k, v in _NG_DTYPE.items()}
_NG_SECTOR = 4096
_NG_GROUP = 32
_NG_NOTE_LEN = 64
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


def _encode_rows(quant: str, bf16_bytes: bytes) -> bytes:
    """Re-encode a run of BF16 rows into the requested row format."""
    import numpy as np
    rows = _bf16_rows_to_f32(bf16_bytes, np)
    if not np.isfinite(rows).all():
        raise ValueError("ngram.bin: a non-finite value in an n-gram row")
    if quant == "q8":
        return _encode_q8(rows, np)
    if quant == "q4":
        return _encode_q4(rows, np)
    raise ValueError(f"ngram.bin: unknown row format {quant!r}")


def build_ngram_bin(shards: list[dict], source: Path, out_path: Path,
                    quant: str = "bf16", l1_seed: int = 0) -> dict:
    """Write the sector-aligned QNGRAM table.

    `quant` picks the row format: `bf16` (v1, the source bytes verbatim),
    `q8` (v2, Q8_G32) or `q4` (v2, Q4_G32). `l1_seed` is the header's
    `l1Hot` field — how many hot rows the writer recorded.
    """
    out_path = Path(out_path)
    if quant not in _NG_DTYPE:
        raise ValueError(f"ngram.bin: unknown row format {quant!r}")
    entry_count = sum(d["shape"][0] for d in shards)
    if entry_count <= 0:
        raise ValueError("ngram.bin: empty table")
    if l1_seed > entry_count:
        raise ValueError(f"ngram.bin: l1_seed {l1_seed} > {entry_count}")
    row_bytes = _row_bytes(quant, HEAD_DIM)
    rps = _rows_per_sector(row_bytes)
    sectors = (entry_count + rps - 1) // rps
    hn = hashlib.sha256()
    sector = bytearray(_NG_SECTOR)
    fill = 0
    written = 0
    chunk_rows = 4096
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
                    enc = buf if quant == "bf16" else _encode_rows(quant, buf)
                    hn.update(enc)
                    pos = 0
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
        if version not in (1, 2):
            raise ValueError(f"ngram.bin: unsupported ver={version} dtype={dtype}")
        if dtype not in _NG_QUANT_BY_DTYPE:
            raise ValueError(f"ngram.bin: unsupported ver={version} dtype={dtype}")
        quant = _NG_QUANT_BY_DTYPE[dtype]
        if version == 1 and quant != "bf16":
            raise ValueError(f"ngram.bin: v1 cannot carry dtype {dtype}")
        if version == 2 and quant == "bf16":
            raise ValueError("ngram.bin: v2 must carry a quantised dtype")
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
