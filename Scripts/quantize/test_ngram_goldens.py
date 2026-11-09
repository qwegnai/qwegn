"""The QNGRAM golden fixtures, against the writer that made them.

The Swift reader is checked against the tables in
`Tests/QWFlashNextTests/fixtures/`. This test is the other half of that
contract: it proves the checked-in tables are the Python writer's own
output and that the golden rows are the writer's own decode of them, so
neither the tables nor the rows can drift away from the writer.
"""

import base64
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hq4 import dequant_hq4_g32_m5_rot
from pack_ngram import (
    HEAD_DIM,
    _NG_GROUP,
    _NG_SECTOR,
    _bf16_rows_to_f32,
    _encode_q8,
    audit_ngram_bin,
    build_ngram_bin,
)

FIXTURES = (Path(__file__).resolve().parents[2]
            / "Tests" / "QWFlashNextTests" / "fixtures")


def table_rows(path: Path) -> bytes:
    """The row bytes of a written table, read back out of the file.

    The table is sector-packed, so the rows of one sector are followed by
    padding. This walks the sectors the way the writer laid them out.
    """
    data = path.read_bytes()
    entry_count, dim, row_bytes = struct.unpack_from("<QHH", data, 16)
    data_offset, = struct.unpack_from("<Q", data, 32)
    rows_per_sector = _NG_SECTOR // row_bytes
    out = bytearray()
    for i in range(entry_count):
        off = (data_offset + (i // rows_per_sector) * _NG_SECTOR
               + (i % rows_per_sector) * row_bytes)
        out += data[off:off + row_bytes]
    return bytes(out)


def decode_q8(row_bytes: bytes, rows: int) -> np.ndarray:
    """The Q8 reader law: the code times the stored f16 scale."""
    groups = HEAD_DIM // _NG_GROUP
    out = np.zeros((rows, HEAD_DIM), dtype=np.float32)
    for r in range(rows):
        for g in range(groups):
            b = r * 170 + g * 34
            codes = np.frombuffer(row_bytes[b:b + 32], dtype=np.int8)
            scale = np.frombuffer(row_bytes[b + 32:b + 34],
                                  dtype=np.float16).astype(np.float32)[0]
            out[r, g * 32:(g + 1) * 32] = codes.astype(np.float32) * scale
    return out


class NgramGoldenFixtureTests(unittest.TestCase):
    """The checked-in tables and rows are the writer's own output."""

    def setUp(self):
        self.doc = json.loads((FIXTURES / "ngram_rows.json").read_text())
        self.rows = self.doc["rows"]
        self.formats = {f["name"]: f for f in self.doc["formats"]}

    def test_the_fixture_names_the_three_formats(self):
        self.assertEqual(self.doc["headDim"], HEAD_DIM)
        self.assertEqual(sorted(self.formats), ["bf16", "q4", "q8"])
        self.assertEqual(self.formats["bf16"]["rowBytes"], 320)
        self.assertEqual(self.formats["q8"]["rowBytes"], 170)
        self.assertEqual(self.formats["q4"]["rowBytes"], 100)

    def test_every_table_is_the_writers_own_output(self):
        # The source is the BF16 table's rows: the writer reads the BF16
        # bytes and encodes them into the format it is asked for.
        source = table_rows(FIXTURES / self.formats["bf16"]["file"])
        self.assertEqual(len(source), self.rows * 320)
        shard = {"file": "source.bin", "base_off": 0,
                 "shape": [self.rows, HEAD_DIM], "row_bytes": 320}
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp)
            (src / "source.bin").write_bytes(source)
            for name in ("bf16", "q8", "q4"):
                out = src / f"{name}.bin"
                build_ngram_bin([shard], src, out, quant=name)
                checked_in = FIXTURES / self.formats[name]["file"]
                self.assertEqual(out.read_bytes(), checked_in.read_bytes(),
                                 f"the {name} table is not the writer's output")

    def test_every_table_passes_the_producers_auditor(self):
        for name in ("bf16", "q8", "q4"):
            report = audit_ngram_bin(FIXTURES / self.formats[name]["file"])
            self.assertEqual(report["version"], self.formats[name]["version"])
            self.assertEqual(report["dtype"], self.formats[name]["dtype"])
            self.assertEqual(report["rowBytes"], self.formats[name]["rowBytes"])
            self.assertEqual(report["rowsPerSector"],
                             self.formats[name]["rowsPerSector"])
            self.assertEqual(report["entryCount"], self.rows)

    def test_the_golden_rows_are_the_writers_own_decode(self):
        source = table_rows(FIXTURES / self.formats["bf16"]["file"])
        rows_f32 = _bf16_rows_to_f32(source, np)
        raw = table_rows(FIXTURES / self.formats["q8"]["file"])
        self.assertEqual(raw, _encode_q8(rows_f32, np))
        expected = {
            "bf16": rows_f32,
            "q8": decode_q8(raw, self.rows),
            "q4": dequant_hq4_g32_m5_rot(
                np.frombuffer(table_rows(FIXTURES / self.formats["q4"]["file"]),
                              dtype=np.uint8).reshape(self.rows, 5, 20),
                f16_round=False),
        }
        for name, want in expected.items():
            payload = base64.b64decode(self.formats[name]["expected"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             self.formats[name]["sha256"])
            got = np.frombuffer(payload, dtype="<f4").reshape(self.rows, HEAD_DIM)
            # The rows are the writer's decode, so they must match bit for
            # bit: the reader has no freedom in the arithmetic.
            self.assertTrue(np.array_equal(got, want.astype("<f4")),
                            f"the {name} golden rows are not the writer's decode")

    def test_the_q4_rows_are_not_rotated(self):
        # The n-gram row is added to the residual stream and never
        # multiplied, so the writer stores it in the original domain.
        blocks = np.frombuffer(table_rows(FIXTURES / self.formats["q4"]["file"]),
                               dtype=np.uint8).reshape(self.rows, 5, 20)
        plain = dequant_hq4_g32_m5_rot(blocks, f16_round=False)
        from hq4 import fwht32_ortho
        rotated = fwht32_ortho(plain)
        self.assertFalse(np.array_equal(plain, rotated))


if __name__ == "__main__":
    unittest.main()
