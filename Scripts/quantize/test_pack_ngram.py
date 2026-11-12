"""The QNGRAM contract: the header, the row formats, and the v2 writer.

The header is a contract between two implementations — the Python writer in
`pack_ngram.py` and the Swift reader (`QWNgramFormat`) — so the tests live
beside the writer, and the golden header bytes are asserted on both sides.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from pack_ngram import (
    HEAD_DIM,
    NGRAM_ROW_BYTES,
    _NG_MAGIC,
    _NG_DTYPE,
    _NG_SECTOR,
    _NG_VERSION,
    _encode_q8,
    _row_bytes,
    _rows_per_sector,
    audit_ngram_bin,
    build_ngram_bin,
)


class NgramHeaderContractTests(unittest.TestCase):
    """The QNGRAM header, as the Swift reader and the Python writer see it."""

    def _header(self, version, dtype, row_bytes, l1_hot=0, entry_count=25):
        import struct
        header = bytearray(_NG_SECTOR)
        header[0:8] = _NG_MAGIC
        struct.pack_into("<HHIQHHI", header, 8, version, 0, dtype,
                         entry_count, HEAD_DIM, row_bytes, l1_hot)
        struct.pack_into("<Q", header, 32, _NG_SECTOR)
        header[44] = 64
        return header

    def test_the_header_fields_land_where_the_reader_looks(self):
        import struct
        header = self._header(1, 0, 320)
        self.assertEqual(header[8], 1)
        self.assertEqual(header[10], 0)
        self.assertEqual(struct.unpack_from("<I", header, 12)[0], 0)
        self.assertEqual(struct.unpack_from("<Q", header, 16)[0], 25)
        self.assertEqual(struct.unpack_from("<H", header, 24)[0], 160)
        self.assertEqual(struct.unpack_from("<H", header, 26)[0], 320)
        self.assertEqual(struct.unpack_from("<I", header, 28)[0], 0)
        self.assertEqual(struct.unpack_from("<Q", header, 32)[0], _NG_SECTOR)

    def test_the_header_bytes_are_the_swift_header_bytes(self):
        # The cross-language contract. `FNNgramFormatTests` asserts these
        # same hex strings against the Swift writer, so the two
        # implementations cannot drift apart without a test going red.
        self.assertEqual(
            bytes(self._header(1, 0, 320)[0:45]).hex(),
            "514e4752414d000001000000000000001900000000000000a00040010000000000100000000000000000000040")
        self.assertEqual(
            bytes(self._header(2, 1, 170)[0:45]).hex(),
            "514e4752414d000002000000010000001900000000000000a000aa000000000000100000000000000000000040")
        self.assertEqual(
            bytes(self._header(2, 2, 100)[0:45]).hex(),
            "514e4752414d000002000000020000001900000000000000a00064000000000000100000000000000000000040")
        self.assertEqual(
            bytes(self._header(3, 3, 90)[0:45]).hex(),
            "514e4752414d000003000000030000001900000000000000a0005a000000000000100000000000000000000040")
        self.assertEqual(
            bytes(self._header(3, 4, 50)[0:45]).hex(),
            "514e4752414d000003000000040000001900000000000000a00032000000000000100000000000000000000040")

    def test_the_row_size_law(self):
        self.assertEqual(_row_bytes("bf16", 160), 320)
        self.assertEqual(_row_bytes("q8", 160), 170)
        self.assertEqual(_row_bytes("q4", 160), 100)
        self.assertEqual(_row_bytes("vq2", 160), 90)
        self.assertEqual(_row_bytes("vq4", 160), 50)
        with self.assertRaises(ValueError):
            _row_bytes("q5", 160)
        with self.assertRaises(ValueError):
            _row_bytes("q8", 100)
        with self.assertRaises(ValueError):
            # A zero head dim gives a zero row size, and a zero row size
            # would make the auditor divide by zero.
            _row_bytes("bf16", 0)

    def test_a_row_that_does_not_fit_a_sector_is_refused(self):
        # A row larger than a sector leaves zero rows per sector, and every
        # row index would divide by zero.
        from pack_ngram import _rows_per_sector
        self.assertEqual(_rows_per_sector(320), 12)
        self.assertEqual(_rows_per_sector(170), 24)
        self.assertEqual(_rows_per_sector(100), 40)
        with self.assertRaises(ValueError):
            _rows_per_sector(4097)
        with self.assertRaises(ValueError):
            _rows_per_sector(0)

    def test_the_version_and_the_dtype_are_paired(self):
        self.assertEqual(_NG_VERSION["bf16"], 1)
        self.assertEqual(_NG_VERSION["q8"], 2)
        self.assertEqual(_NG_VERSION["q4"], 2)
        self.assertEqual(_NG_VERSION["vq2"], 3)
        self.assertEqual(_NG_VERSION["vq4"], 3)
        self.assertEqual(_NG_DTYPE, {"bf16": 0, "q8": 1, "q4": 2, "vq2": 3, "vq4": 4})


class NgramV2WriterTests(unittest.TestCase):
    """The v2 writer: the two quantised row formats, and their audit."""

    def _fake_checkpoint(self, temp, shard_rows, base_shard=1, of=3):
        src = Path(temp) / "source"
        src.mkdir()
        NG_KEY = "model.ple.ple_embedding.ngram_embedding.shard_{}.weight"
        files = {}
        for i, rows in enumerate(shard_rows):
            key = NG_KEY.format(base_shard + i)
            files[key] = torch.randn(rows, 160, dtype=torch.bfloat16)
        save_file(files, src / f"model-{base_shard:05d}-of-{of:05d}.safetensors")
        return src, files

    def _shards_from_source(self, src):
        out = []
        for path in sorted(src.glob("*.safetensors")):
            raw = path.read_bytes()
            hl = int.from_bytes(raw[:8], "little")
            head = json.loads(raw[8:8 + hl])
            for name, e in head.items():
                if "ple.ple_embedding.ngram_embedding." not in name:
                    continue
                idx = int(name.split("shard_")[1].split(".")[0])
                lo = e["data_offsets"][0]
                out.append({"shard": idx, "file": path.name, "key": name,
                            "shape": e["shape"], "dtype": e["dtype"],
                            "row_bytes": NGRAM_ROW_BYTES,
                            "base_off": 8 + hl + lo})
        out.sort(key=lambda d: d["shard"])
        rows = 0
        for d in out:
            d["first_row"] = rows
            rows += d["shape"][0]
        return out

    def _build(self, temp, quant, rows=(27, 5, 40)):
        src, files = self._fake_checkpoint(temp, list(rows))
        shards = self._shards_from_source(src)
        out = Path(temp) / f"ngram-{quant}.bin"
        res = build_ngram_bin(shards, src, out, quant=quant)
        return src, files, shards, out, res

    def test_bf16_is_unchanged_by_the_v2_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, out, res = self._build(temp, "bf16")
            self.assertEqual(res["version"], 1)
            self.assertEqual(res["dtype"], 0)
            self.assertEqual(res["row_bytes"], 320)
            self.assertEqual(res["rows_per_sector"], 12)
            total = 72
            sectors = (total + 12 - 1) // 12
            self.assertEqual(out.stat().st_size, _NG_SECTOR + sectors * _NG_SECTOR)

    def test_q8_writes_the_v2_header_and_a_smaller_table(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, out, res = self._build(temp, "q8")
            self.assertEqual(res["version"], 2)
            self.assertEqual(res["dtype"], 1)
            self.assertEqual(res["row_bytes"], 170)
            self.assertEqual(res["rows_per_sector"], 24)
            total = 72
            sectors = (total + 24 - 1) // 24
            self.assertEqual(out.stat().st_size, _NG_SECTOR + sectors * _NG_SECTOR)
            aud = audit_ngram_bin(out)
            self.assertEqual(aud["version"], 2)
            self.assertEqual(aud["dtype"], 1)
            self.assertEqual(aud["entryCount"], total)
            self.assertEqual(aud["rowBytes"], 170)
            self.assertEqual(aud["rowsPerSector"], 24)
            self.assertEqual(aud["table_sha256"], res["table_sha256"])

    def test_q4_writes_the_v2_header_and_a_smaller_table(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, out, res = self._build(temp, "q4")
            self.assertEqual(res["version"], 2)
            self.assertEqual(res["dtype"], 2)
            self.assertEqual(res["row_bytes"], 100)
            self.assertEqual(res["rows_per_sector"], 40)
            total = 72
            sectors = (total + 40 - 1) // 40
            self.assertEqual(out.stat().st_size, _NG_SECTOR + sectors * _NG_SECTOR)
            aud = audit_ngram_bin(out)
            self.assertEqual(aud["dtype"], 2)
            self.assertEqual(aud["rowBytes"], 100)
            self.assertEqual(aud["rowsPerSector"], 40)

    def test_vq2_writes_the_v3_header_codebook_and_beats_q4_error(self):
        from pack_ngram import _NG_VQ2_CB_BYTES, _NG_VQ2_CB_OFF, decode_vq2_rows
        with tempfile.TemporaryDirectory() as temp:
            _, files, _, out, res = self._build(temp, "vq2", rows=(120, 80, 100))
            self.assertEqual(res["version"], 3)
            self.assertEqual(res["dtype"], 3)
            self.assertEqual(res["row_bytes"], 90)
            self.assertEqual(res["rows_per_sector"], 45)
            aud = audit_ngram_bin(out)
            self.assertEqual(aud["version"], 3)
            self.assertEqual(aud["dtype"], 3)
            self.assertEqual(aud["entryCount"], 300)
            self.assertEqual(aud["table_sha256"], res["table_sha256"])
            raw = out.read_bytes()
            cb = np.frombuffer(raw[_NG_VQ2_CB_OFF:_NG_VQ2_CB_OFF + _NG_VQ2_CB_BYTES], dtype="<f2")
            row0 = decode_vq2_rows(raw[_NG_SECTOR:_NG_SECTOR + 90], cb, 1, 160)[0]
            src0 = files["model.ple.ple_embedding.ngram_embedding.shard_1.weight"][0].float().numpy()
            rel = float(np.linalg.norm(row0 - src0) / np.linalg.norm(src0))
            self.assertLess(rel, 0.11)
            # Corrupting codebook[0] to non-zero must be caught by audit_ngram_bin.
            bad = bytearray(raw)
            bad[_NG_VQ2_CB_OFF] = 1
            out.write_bytes(bytes(bad))
            with self.assertRaisesRegex(ValueError, "codebook\\[0\\]"):
                audit_ngram_bin(out)

    def test_vq4_writes_the_v3_header_codebook_and_roundtrips(self):
        from pack_ngram import _NG_VQ4_CB_BYTES, _NG_VQ4_CB_OFF
        from pack_ngram_vq4 import decode_vq4_rows
        with tempfile.TemporaryDirectory() as temp:
            _, files, _, out, res = self._build(temp, "vq4_g32", rows=(120, 80, 100))
            self.assertEqual(res["version"], 3)
            self.assertEqual(res["dtype"], 4)
            self.assertEqual(res["row_bytes"], 50)
            self.assertEqual(res["rows_per_sector"], 81)
            aud = audit_ngram_bin(out)
            self.assertEqual(aud["version"], 3)
            self.assertEqual(aud["dtype"], 4)
            self.assertEqual(aud["entryCount"], 300)
            self.assertEqual(aud["table_sha256"], res["table_sha256"])
            raw = out.read_bytes()
            cb = np.frombuffer(raw[_NG_VQ4_CB_OFF:_NG_VQ4_CB_OFF + _NG_VQ4_CB_BYTES], dtype="<f2")
            row0 = decode_vq4_rows(raw[_NG_SECTOR:_NG_SECTOR + 50], cb, 1, 160)[0]
            src0 = files["model.ple.ple_embedding.ngram_embedding.shard_1.weight"][0].float().numpy()
            rel = float(np.linalg.norm(row0 - src0) / np.linalg.norm(src0))
            self.assertLess(rel, 0.34)
            bad = bytearray(raw)
            bad[_NG_VQ4_CB_OFF + 6] = 1
            out.write_bytes(bytes(bad))
            with self.assertRaisesRegex(ValueError, "codebook\\[0\\]"):
                audit_ngram_bin(out)

    def test_the_quantised_rows_are_reconstructible(self):
        # The Q8 row decodes back to the source values within the law the
        # writer promises: a symmetric int8 code times its stored f16 scale.
        with tempfile.TemporaryDirectory() as temp:
            src, files, _, out, _ = self._build(temp, "q8")
            key = "model.ple.ple_embedding.ngram_embedding.shard_1.weight"
            src_row = files[key][0].float().numpy()
            data = out.read_bytes()
            row = data[_NG_SECTOR:_NG_SECTOR + 170]
            worst = 0.0
            for g in range(5):
                base = g * 34
                codes = np.frombuffer(row[base:base + 32], dtype=np.int8)
                scale = float(np.frombuffer(row[base + 32:base + 34],
                                           dtype=np.float16)[0])
                got = codes.astype(np.float32) * scale
                want = src_row[g * 32:(g + 1) * 32]
                worst = max(worst, float(np.abs(got - want).max()))
                self.assertTrue(np.all(np.abs(codes) <= 127))
            # The scale is amax/127, so the error is bounded by the scale
            # itself plus the f16 rounding of it.
            amax = float(np.abs(src_row).max())
            self.assertLess(worst, amax / 127.0 * 1.01)

    def test_a_tampered_quantised_table_is_caught(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, out, _ = self._build(temp, "q8")
            data = bytearray(out.read_bytes())
            data[_NG_SECTOR + 100] ^= 0xFF
            out.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "producer note"):
                audit_ngram_bin(out)

    def test_a_table_whose_version_and_dtype_disagree_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, out, _ = self._build(temp, "q8")
            data = bytearray(out.read_bytes())
            # Claim v1 while carrying the Q8 dtype.
            data[8] = 1
            out.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "v1 cannot carry dtype"):
                audit_ngram_bin(out)

    def test_a_subnormal_scale_keeps_its_codes(self):
        # The reader's rule is one line: value = code * scale. A group whose
        # scale is subnormal keeps its codes and its subnormal scale, so the
        # reader needs no special case. Storing such a group empty instead
        # was measured on the real table and is 55x worse on the worst
        # affected row.
        from pack_ngram import _F16_MIN_NORMAL
        row = np.zeros((1, 160), dtype=np.float32)
        peak = 127.0 * _F16_MIN_NORMAL / 2.0     # a subnormal scale
        row[0, 0] = peak
        enc = np.frombuffer(_encode_q8(row, np), dtype=np.uint8).reshape(1, 5, 34)
        codes = enc[:, :, :32].view(np.int8)
        scale = np.ascontiguousarray(enc[:, :, 32:]).view(np.float16).reshape(1, 5)
        self.assertGreater(float(scale[0, 0]), 0.0)
        self.assertLess(float(scale[0, 0]), _F16_MIN_NORMAL)
        self.assertEqual(int(codes[0, 0, 0]), 127)
        # The other groups are all-zero: their scale is zero, so they carry
        # no codes.
        self.assertEqual(float(scale[0, 1]), 0.0)
        self.assertTrue(np.all(codes[0, 1] == 0))

    def test_a_scale_that_underflows_to_zero_stores_an_empty_group(self):
        row = np.zeros((1, 160), dtype=np.float32)
        row[0, 0] = 1e-20                      # its scale underflows f16 to 0
        enc = np.frombuffer(_encode_q8(row, np), dtype=np.uint8).reshape(1, 5, 34)
        codes = enc[:, :, :32].view(np.int8)
        scale = np.ascontiguousarray(enc[:, :, 32:]).view(np.float16).reshape(1, 5)
        self.assertEqual(float(scale[0, 0]), 0.0)
        self.assertTrue(np.all(codes[0, 0] == 0))

    def test_the_reader_rule_is_one_line(self):
        # value = code * scale, for every group, with no special case.
        from pack_ngram import _encode_q8
        rng = np.random.default_rng(5)
        row = rng.normal(0, 0.02, (16, 160)).astype(np.float32)
        enc = np.frombuffer(_encode_q8(row, np), dtype=np.uint8).reshape(16, 5, 34)
        codes = enc[:, :, :32].view(np.int8).astype(np.float32)
        scale = np.ascontiguousarray(enc[:, :, 32:]).view(np.float16).reshape(16, 5).astype(np.float32)
        got = (codes * scale[:, :, None]).reshape(16, 160)
        rel = np.linalg.norm(got - row, axis=-1) / np.linalg.norm(row, axis=-1)
        self.assertLess(float(rel.max()), 0.05)

    def test_the_writer_refuses_an_unknown_row_format(self):
        with tempfile.TemporaryDirectory() as temp:
            src, _ = self._fake_checkpoint(temp, [4])
            shards = self._shards_from_source(src)
            with self.assertRaisesRegex(ValueError, "unknown row format"):
                build_ngram_bin(shards, src, Path(temp) / "x.bin", quant="q5")

    def test_a_scale_that_does_not_fit_f16_is_refused(self):
        # The Q8 scale is stored as f16. A group whose amax/127 overflows it
        # cannot be represented, and an infinite scale would decode to
        # garbage, so the writer refuses the table.
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "source"
            src.mkdir()
            key = "model.ple.ple_embedding.ngram_embedding.shard_1.weight"
            row = torch.zeros(1, 160, dtype=torch.bfloat16)
            row[0, 0] = 1.0e9
            save_file({key: row}, src / "model-00001-of-00001.safetensors")
            shards = self._shards_from_source(src)
            with self.assertRaisesRegex(ValueError, "does not fit f16"):
                build_ngram_bin(shards, src, Path(temp) / "ngram.bin", quant="q8")

    def test_a_non_finite_row_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "source"
            src.mkdir()
            key = "model.ple.ple_embedding.ngram_embedding.shard_1.weight"
            row = torch.zeros(1, 160, dtype=torch.bfloat16)
            row[0, 0] = float("nan")
            save_file({key: row}, src / "model-00001-of-00001.safetensors")
            shards = self._shards_from_source(src)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_ngram_bin(shards, src, Path(temp) / "ngram.bin", quant="q8")


