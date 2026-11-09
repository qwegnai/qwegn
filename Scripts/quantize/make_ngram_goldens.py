"""Write the QNGRAM row goldens: the tables and the expected rows.

The golden bytes come from the **Python** writer, so the Swift reader and
the writer cannot drift apart without a test going red. Run it with:

    uv run --with numpy --with torch python3 Scripts/quantize/make_ngram_goldens.py
"""

import base64
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from hq4 import dequant_hq4_g32_m5_rot
from pack_ngram import (
    HEAD_DIM,
    _NG_GROUP,
    _NG_SECTOR,
    _bf16_rows_to_f32,
    _encode_q4,
    _encode_q8,
    audit_ngram_bin,
    build_ngram_bin,
)

# The rows the goldens hold. 41 rows cross the sector boundary of every
# format: BF16 packs 12 rows per sector, Q8 packs 24 and Q4 packs 40.
ROWS = 41
# The formats, with the header version and the dtype code each one carries.
FORMATS = [("bf16", 1, 0, 320, 12),
           ("q8", 2, 1, 170, 24),
           ("q4", 2, 2, 100, 40)]
FIXTURES = (Path(__file__).resolve().parents[2]
            / "Tests" / "QWFlashNextTests" / "fixtures")


def source_rows() -> np.ndarray:
    """The rows every format is written from.

    The first rows carry the edge cases the reader has to get right: an
    all-zero row, a zero group inside a live row, a group whose f16 scale
    is subnormal, and a group whose scale is large.
    """
    rng = np.random.default_rng(20261026)
    rows = rng.normal(0.0, 0.02, (ROWS, HEAD_DIM)).astype(np.float32)
    rows[0, :] = 0.0
    rows[1, 0:32] = 0.0
    rows[1, 32:64] = 1e-4
    rows[1, 64:96] = 1e4
    rows[2, 0:32] = 0.0
    return rows


def bf16_bytes(rows: np.ndarray) -> bytes:
    """The source rows as the BF16 bytes the writer reads.

    A BF16 lane is the top 16 bits of an f32, so the bytes are the f32
    patterns shifted right by 16 and rounded to nearest even.
    """
    bits = rows.astype("<f4").view("<u4")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def table_rows(path: Path) -> bytes:
    """The row bytes of a written table, read back out of the file.

    The table is sector-packed, so the rows of one sector are followed by
    padding. This walks the sectors the way the writer laid them out.
    """
    data = path.read_bytes()
    version, flags, dtype = struct.unpack_from("<HHI", data, 8)
    entry_count, dim, row_bytes = struct.unpack_from("<QHH", data, 16)
    data_offset, = struct.unpack_from("<Q", data, 32)
    rows_per_sector = _NG_SECTOR // row_bytes
    out = bytearray()
    for i in range(entry_count):
        off = (data_offset + (i // rows_per_sector) * _NG_SECTOR
               + (i % rows_per_sector) * row_bytes)
        out += data[off:off + row_bytes]
    return bytes(out)


def decode_q8(row_bytes: bytes) -> np.ndarray:
    """The Q8 reader law: the code times the stored f16 scale."""
    groups = HEAD_DIM // _NG_GROUP
    out = np.zeros((ROWS, HEAD_DIM), dtype=np.float32)
    for r in range(ROWS):
        for g in range(groups):
            b = r * 170 + g * 34
            codes = np.frombuffer(row_bytes[b:b + 32], dtype=np.int8)
            scale = np.frombuffer(row_bytes[b + 32:b + 34],
                                  dtype=np.float16).astype(np.float32)[0]
            out[r, g * 32:(g + 1) * 32] = codes.astype(np.float32) * scale
    return out


def decode_q4(row_bytes: bytes) -> np.ndarray:
    """The Q4 reader law: the block's magic-bias form, with no rotation."""
    blocks = np.frombuffer(row_bytes, dtype=np.uint8).reshape(ROWS, 5, 20)
    return dequant_hq4_g32_m5_rot(blocks, f16_round=False)


def main() -> None:
    rows = source_rows()
    src_bf16 = bf16_bytes(rows)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    shard = {"file": "source.bin", "base_off": 0,
             "shape": [ROWS, HEAD_DIM], "row_bytes": 320}
    entries = []
    with tempfile.TemporaryDirectory() as temp:
        src = Path(temp)
        (src / "source.bin").write_bytes(src_bf16)
        for name, version, dtype, row_bytes, rps in FORMATS:
            out = FIXTURES / f"ngram_{name}.bin"
            build_ngram_bin([shard], src, out, quant=name)
            report = audit_ngram_bin(out)
            assert report["version"] == version, report
            assert report["dtype"] == dtype, report
            assert report["rowBytes"] == row_bytes, report
            assert report["rowsPerSector"] == rps, report
            raw = table_rows(out)
            # The writer encodes the BF16 bytes it read, so the source of
            # every format is the BF16 table, not the rows before it.
            source = _bf16_rows_to_f32(src_bf16, np)
            if name == "bf16":
                expected = source
                assert raw == src_bf16, "the BF16 table is not the source bytes"
            elif name == "q8":
                expected = decode_q8(raw)
                assert raw == _encode_q8(source, np), "the Q8 rows are not the encoder's"
            else:
                expected = decode_q4(raw)
                assert raw == _encode_q4(source, np), "the Q4 rows are not the encoder's"
            payload = expected.astype("<f4").tobytes()
            entries.append({
                "name": name,
                "version": version,
                "dtype": dtype,
                "rowBytes": row_bytes,
                "rowsPerSector": rps,
                "file": out.name,
                "expected": base64.b64encode(payload).decode("ascii"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
            print(f"{name}: {out.name} {out.stat().st_size} B "
                  f"rows {ROWS} rowBytes {row_bytes} rps {rps}")
    doc = {
        "headDim": HEAD_DIM,
        "rows": ROWS,
        "note": ("The tables are the Python writer's own output. The expected "
                 "rows are the writer's decode of those bytes."),
        "formats": entries,
    }
    (FIXTURES / "ngram_rows.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURES / 'ngram_rows.json'}")


if __name__ == "__main__":
    main()
