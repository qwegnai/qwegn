import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from audit_release import audit_release, validate_router_constants
from pack import (
    NGRAM_BIN_NAME,
    NGRAM_ROW_BYTES,
    PLE_LOCATION_NAME,
    _NG_RPS,
    _NG_SECTOR,
    audit_ngram_bin,
    build_ngram_bin,
    dequant_hq4_g32_m5,
    pack_file,
    quant_hq4_g32_m5,
    quant_q4g,
    quantize_model,
    resolve_assets_dir,
    sha256,
)
from verify import audit_pack, compare_planes, read_header, verify_assets


class ModelDirectoryTests(unittest.TestCase):
    def source(self, root):
        source = root / "source"
        source.mkdir()
        for name in ("config.json", "tokenizer.json"):
            (source / name).write_text("{}")
        (source / "chat_template.jinja").write_text("template")
        (source / "LICENSE").write_text("fixture license")
        tensors = {
            "model.language_model.layers.0.mlp.down_proj.weight":
                torch.arange(256, dtype=torch.float32).reshape(2, 128) / 100,
            "model.language_model.embed_tokens.weight": torch.ones(2, 128),
            "lm_head.weight": torch.ones(2, 128),
            "model.visual.block.weight": torch.ones(2, 128),
            "model.language_model.norm.weight": torch.ones(128),
            "model.layers.0.attn.hc_read.down.weight": torch.ones(4, 1, 2560),
            "model.layers.0.mlp.gate.weight": torch.ones(512, 2560),
        }
        save_file(tensors, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
        return source

    def test_directory_hq4_g32_m5_and_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            destination = root / "release"
            quantize_model(source, destination)
            header, counts = audit_pack(destination / "model.qwp")
            self.assertEqual(counts, {"hq4_g32_m5": 2, "bf16": 5})
            self.assertTrue(header["tensors"]["lm_head.weight"].get("hq4_off"))
            self.assertIn("raw_off", header["tensors"]["model.language_model.embed_tokens.weight"])
            self.assertIn("raw_off", header["tensors"]["model.layers.0.attn.hc_read.down.weight"])
            self.assertIn("raw_off", header["tensors"]["model.layers.0.mlp.gate.weight"])
            self.assertEqual(header["provenance"]["quant_recorded"], "hq4_g32_m5")
            self.assertEqual(header["provenance"]["assets_dir"], "assets")
            self.assertEqual(verify_assets(destination, header), 4)
            compared, skipped = compare_planes(destination / "model.qwp",
                                               destination / "model.qwp")
            self.assertGreater(compared, 1600)
            self.assertEqual(skipped, [])
            self.assertFalse(any(p.name.startswith(".release-") for p in root.iterdir()))
            with self.assertRaises(FileExistsError):
                quantize_model(source, destination)
            (destination / "assets" / "LICENSE").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "asset hash/size differs"):
                verify_assets(destination, header)

    def test_flat_root_keeps_the_assets_beside_the_weights(self):
        """A flat root holds config.json and the n-gram table beside
        model.qwp, with no `assets/` subfolder."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            destination = root / "flat"
            quantize_model(source, destination, flat_root=True)

            self.assertTrue((destination / "model.qwp").is_file())
            self.assertTrue((destination / "config.json").is_file())
            self.assertFalse((destination / "assets").exists())

            header, counts = audit_pack(destination / "model.qwp")
            self.assertEqual(counts, {"hq4_g32_m5": 2, "bf16": 5})
            self.assertEqual(header["provenance"]["assets_dir"], ".")

            # The resolver finds the flat assets, and the verifier follows.
            self.assertEqual(resolve_assets_dir(destination), destination)
            self.assertEqual(verify_assets(destination, header), 4)

    def test_resolve_assets_dir_prefers_the_flat_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "legacy"
            (legacy / "assets").mkdir(parents=True)
            (legacy / "assets" / "config.json").write_text("{}")
            self.assertEqual(resolve_assets_dir(legacy), legacy / "assets")

            flat = root / "flat"
            flat.mkdir()
            (flat / "config.json").write_text("{}")
            self.assertEqual(resolve_assets_dir(flat), flat)

            ngram = root / "ngram"
            ngram.mkdir()
            (ngram / NGRAM_BIN_NAME).write_bytes(b"")
            self.assertEqual(resolve_assets_dir(ngram), ngram)

    def test_source_rejection_and_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            destination = root / "release"
            (source / "config.json").write_text('{"quantization_config": {}}')
            with self.assertRaisesRegex(ValueError, "already quantized"):
                quantize_model(source, destination)
            self.assertFalse(destination.exists())
            (source / "config.json").write_text("{}")
            (source / "model.safetensors.index.json").write_text('{"weight_map": {}}')
            with self.assertRaisesRegex(ValueError, "source shards"):
                quantize_model(source, destination)
            (source / "model.safetensors.index.json").unlink()
            save_file({"broken": torch.empty(0, 128)}, source / "model.safetensors")
            with self.assertRaises(ValueError):
                quantize_model(source, destination)
            self.assertFalse(destination.exists())

    def test_hq4_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            quant_q4g(np.full((1, 128), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-empty 2-D"):
            quant_q4g(np.ones((1, 31), dtype=np.float32))

    def test_hq4_roundtrip_outlier_resilience(self):
        rng = np.random.default_rng(42)
        w = rng.normal(0.0, 0.02, (64, 128)).astype(np.float32)
        w[:, 0] = 1.5
        blocks = quant_hq4_g32_m5(w)
        self.assertEqual(blocks.shape, (64, 4, 20))
        dq = dequant_hq4_g32_m5(blocks)
        rel_rmse = np.sqrt(np.mean((w - dq) ** 2)) / np.std(w)
        self.assertLess(rel_rmse, 0.12)

    def test_audit_rejects_q8_and_hidden_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.qwp"
            pack_file({"weight": torch.ones(2, 128)}, path, {})
            header, _ = audit_pack(path)
            for dead in (b"", b"\0" * 256):
                original = path.read_bytes()
                header_size = int.from_bytes(original[4:12], "little")
                payload = original[12:-header_size]
                modified = json.loads(json.dumps(header))
                if not dead:
                    modified["tensors"]["weight"]["q8_off"] = 256
                tail = json.dumps(modified).encode()
                path.write_bytes(b"QWP1" + len(tail).to_bytes(8, "little")
                                 + payload + dead + tail)
                with self.assertRaises(ValueError):
                    audit_pack(path)
                path.write_bytes(original)

    def test_reference_comparison_allows_explicit_head_requantization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = root / "current.qwp"
            reference = root / "reference.qwp"
            weights = torch.arange(256, dtype=torch.float32).reshape(2, 128)
            pack_file({"lm_head.weight": weights}, current, {})
            pack_file([], reference, {}, raw={"lm_head.weight": weights})
            with self.assertRaisesRegex(ValueError, "plane schema differs"):
                compare_planes(current, reference)
            compared, skipped = compare_planes(
                current, reference, allow_requantized=("lm_head.weight",))
            self.assertEqual(compared, 0)
            self.assertEqual(skipped, ["lm_head.weight"])


class ExpertStackPackTests(unittest.TestCase):
    def test_expert_stack_packs_and_reads_back_per_expert(self):
        E, N, K = 3, 8, 128
        g = torch.Generator().manual_seed(11)
        stack = torch.randn(E, N, K, generator=g)
        name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "e.qwp"
            pack_file({name: stack}, p, {}, sidecar_dir=temp)
            header, counts = audit_pack(p)
            self.assertEqual(counts, {"hq4_g32_m5": 1, "bf16": 0})
            entry = header["tensors"][name]
            self.assertEqual(entry["experts"], E)
            self.assertEqual(entry["e_hq4"], N * (K // 32) * 20)
            self.assertEqual(entry["hq4_off"] % 16384, 0)
            raw = p.read_bytes()
            for i in range(E):
                start = entry["hq4_off"] + i * entry["e_hq4"]
                slab = np.frombuffer(
                    raw, np.uint8, entry["e_hq4"], start
                ).reshape(N, K // 32, 20)
                want = np.load(Path(temp) / f"{name}.e{i}.dq4.npy")
                np.testing.assert_array_equal(dequant_hq4_g32_m5(slab), want)

    def test_mystery_rank3_and_odd_shapes_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "x.qwp"
            with self.assertRaisesRegex(ValueError, "refuse to guess"):
                pack_file({"model.visual.block.weight": torch.ones(2, 4, 128)},
                          p, {})
            bad = "model.language_model.layers.0.mlp.experts.down_proj"
            with self.assertRaisesRegex(ValueError, "E>=1, even N"):
                pack_file({bad: torch.ones(2, 3, 128)}, p, {})
            with self.assertRaisesRegex(ValueError, "E>=1, even N"):
                pack_file({bad: torch.ones(2, 4, 30)}, p, {})

    def test_audit_rejects_broken_expert_layout(self):
        E, N, K = 2, 4, 128
        name = "model.language_model.layers.0.mlp.experts.down_proj"
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "e.qwp"
            pack_file({name: torch.ones(E, N, K)}, p, {})
            header, _ = audit_pack(p)
            original = p.read_bytes()
            entry = header["tensors"][name]
            entry["e_hq4"] += 20
            payload = original[12:]
            size = int.from_bytes(original[4:12], "little")
            path = Path(temp) / "bad.qwp"
            tail = json.dumps(header).encode()
            path.write_bytes(b"QWP1" + len(tail).to_bytes(8, "little")
                             + payload[:-size] + tail)
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                audit_pack(path)


NG = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight"


class PleOutOfCoreTests(unittest.TestCase):
    def _mk(self, src: Path, shards):
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "qwen4_exp"}')
        (src / "tokenizer.json").write_text("{}")
        (src / "chat_template.jinja").write_text("")
        for fname, tensors in shards.items():
            save_file({k: t for k, t in tensors}, src / fname)

    def _fake_checkpoint(self, temp, shard_idx=(0, 1), dtype0=torch.bfloat16,
                         shape0=(3, 160)):
        src = Path(temp) / "source"
        ng0 = torch.randn(*shape0, dtype=dtype0)
        ng1 = torch.randn(2, 160, dtype=torch.bfloat16)
        shards = {
            "model-00001-of-00002.safetensors": [
                (NG.format(shard_idx[0]), ng0),
                ("model.language_model.layers.0.o_norm.weight", torch.ones(256)),
            ],
            "model-00002-of-00002.safetensors": [
                ("model.language_model.layers.0.attn.q_proj.weight", torch.randn(8, 256)),
                (NG.format(shard_idx[1]), ng1),
            ],
        }
        self._mk(src, shards)
        return src

    def test_ngram_excluded_and_location_written(self):
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp)
            out = Path(temp) / "rel"
            quantize_model(src, out)
            pack = out / "model.qwp"
            header, counts = audit_pack(pack)
            for i in (0, 1):
                self.assertNotIn(NG.format(i), header["tensors"])
            self.assertEqual(counts, {"hq4_g32_m5": 1, "bf16": 1})
            loc_path = out / "assets" / PLE_LOCATION_NAME
            loc = json.loads(loc_path.read_text())
            self.assertEqual(loc["version"], 1)
            self.assertEqual(loc["row_bytes"], 320)
            self.assertEqual(loc["total_rows"], 5)
            self.assertEqual([s["first_row"] for s in loc["shards"]], [0, 3])
            manifest = json.loads((out / "assets" / "manifest.json").read_text())
            entry = manifest[PLE_LOCATION_NAME]
            self.assertEqual(entry["sha256"], sha256(loc_path))

    def test_reuse_ngram_from_hardlinks_existing_table(self):
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp)
            rel1 = Path(temp) / "rel1"
            rel2 = Path(temp) / "rel2"
            quantize_model(src, rel1)
            quantize_model(src, rel2, reuse_ngram_from=rel1)
            nb1 = rel1 / "assets" / NGRAM_BIN_NAME
            nb2 = rel2 / "assets" / NGRAM_BIN_NAME
            self.assertEqual(nb1.stat().st_ino, nb2.stat().st_ino)
            self.assertEqual(nb1.read_bytes(), nb2.read_bytes())

    def test_reuse_from_an_older_pack_that_knew_no_row_format(self):
        # A pack written before the manifest carried the row format has no
        # version/dtype keys. Reusing its table must not read the missing
        # keys as if they were there.
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp)
            rel1 = Path(temp) / "rel1"
            rel2 = Path(temp) / "rel2"
            quantize_model(src, rel1)
            mf = rel1 / "assets" / "manifest.json"
            manifest = json.loads(mf.read_text())
            for key in ("version", "dtype", "row_bytes"):
                manifest[NGRAM_BIN_NAME].pop(key, None)
            mf.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            quantize_model(src, rel2, reuse_ngram_from=rel1)
            out = json.loads((rel2 / "assets" / "manifest.json").read_text())
            self.assertNotIn("version", out[NGRAM_BIN_NAME])
            self.assertEqual(out[NGRAM_BIN_NAME]["bytes"],
                             (rel1 / "assets" / NGRAM_BIN_NAME).stat().st_size)

    def test_bad_ngram_facts_raise_before_any_write(self):
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp, dtype0=torch.float32)
            with self.assertRaisesRegex(ValueError, "dtype F32"):
                quantize_model(src, Path(temp) / "r1")
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp, shape0=(3, 96))
            with self.assertRaisesRegex(ValueError, r"\[rows, 160\]"):
                quantize_model(src, Path(temp) / "r2")
        with tempfile.TemporaryDirectory() as temp:
            src = self._fake_checkpoint(temp, shard_idx=(0, 2))
            with self.assertRaisesRegex(ValueError, "not 0..N-1"):
                quantize_model(src, Path(temp) / "r3")


class NgramBinTests(unittest.TestCase):
    def _fake_checkpoint(self, temp, shard_rows, base_shard=1, of=3):
        src = Path(temp) / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "qwen4_exp"}')
        (src / "tokenizer.json").write_text("{}")
        (src / "chat_template.jinja").write_text("")
        NG_KEY = "model.ple.ple_embedding.ngram_embedding.shard_{}.weight"
        files = {}
        rows_total = 0
        for i, rows in enumerate(shard_rows):
            key = NG_KEY.format(base_shard + i)
            t = torch.randn(rows, 160, dtype=torch.bfloat16)
            files[key] = t
            rows_total += rows
        core = {"model.language_model.layers.0.o_norm.weight": torch.ones(256)}
        save_file(files, src / f"model-{base_shard:05d}-of-{of:05d}.safetensors")
        save_file(core, src / f"model-{base_shard+1:05d}-of-{of:05d}.safetensors")
        return src, files, rows_total

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

    def test_packing_matches_door_addressing_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            src, files, rows = self._fake_checkpoint(temp, [27, 5, 40])
            shards = self._shards_from_source(src)
            out = Path(temp) / "ngram.bin"
            res = build_ngram_bin(shards, src, out)
            self.assertEqual(res["entry_count"], rows)
            sectors = (rows + _NG_RPS - 1) // _NG_RPS
            self.assertEqual(out.stat().st_size, _NG_SECTOR + sectors * _NG_SECTOR)
            data = out.read_bytes()
            NGK = "model.ple.ple_embedding.ngram_embedding.shard_{}.weight"
            row_of_global = []
            for i, n in ((1, 27), (2, 5), (3, 40)):
                row_of_global += [(NGK.format(i), r) for r in range(n)]
            for g, (key, local) in enumerate(row_of_global):
                off = _NG_SECTOR + (g // _NG_RPS) * _NG_SECTOR + (g % _NG_RPS) * NGRAM_ROW_BYTES
                want = files[key][local].contiguous().view(torch.uint8).numpy().tobytes()
                self.assertEqual(data[off:off + NGRAM_ROW_BYTES], want)
            aud = audit_ngram_bin(out)
            self.assertEqual(aud["entryCount"], rows)
            self.assertEqual(aud["table_sha256"], res["table_sha256"])
            tampered = bytearray(data)
            tampered[_NG_SECTOR + 100] ^= 0xFF
            out.write_bytes(bytes(tampered))
            with self.assertRaisesRegex(ValueError, "producer note"):
                audit_ngram_bin(out)


class ReleaseAuditTests(unittest.TestCase):
    RNG = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight"
    MULT = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    OFFS = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"
    VOC = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
    VOCAB = [7, 5]

    def _build(self, t: Path):
        src = t / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "qwen4_exp"}')
        (src / "tokenizer.json").write_text("{}")
        (src / "vocab.json").write_text("{}")
        (src / "chat_template.jinja").write_text("")
        vocs = self.VOCAB
        offs = [0, vocs[0]]
        muls = torch.tensor([23703573157769, 20109073645365, 8052911324071], dtype=torch.int64)
        save_file({self.RNG.format(0): torch.randn(5, 160, dtype=torch.bfloat16),
                   "model.language_model.layers.0.o_norm.weight": torch.ones(256),
                   "model.language_model.layers.0.attn.q_proj.weight": torch.randn(8, 256),
                   self.MULT: muls,
                   self.OFFS: torch.tensor(offs, dtype=torch.int64),
                   self.VOC: torch.tensor(vocs, dtype=torch.int64)},
                  src / "model-00001-of-00002.safetensors")
        save_file({self.RNG.format(1): torch.randn(3, 160, dtype=torch.bfloat16)},
                  src / "model-00002-of-00002.safetensors")
        rel = t / "rel"
        quantize_model(src, rel)
        return src, rel

    def test_clean_release_and_tamper_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            src, rel = self._build(Path(temp))
            info = audit_release(rel, checkpoint=src)
            self.assertTrue(info["ok"])
            self.assertEqual(info["ngram"]["entryCount"], 8)

            shutil.rmtree(src)
            info_pruned = audit_release(rel, post_pruning=True)
            self.assertTrue(info_pruned["ok"])

            (rel / "model.qwp.sha256").write_text("0" * 64 + "  model.qwp\n")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                audit_release(rel, post_pruning=True)

    def test_location_first_row_shift_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            src, rel = self._build(Path(temp))
            lp = rel / "assets/ple_ngram_locations.json"
            loc = json.loads(lp.read_text())
            loc["shards"][0]["first_row"] += 1
            lp.write_text(json.dumps(loc, indent=1, sort_keys=True))
            mp = rel / "assets/manifest.json"
            man = json.loads(mp.read_text())
            man["ple_ngram_locations.json"] = {"sha256": sha256(lp),
                                               "bytes": lp.stat().st_size}
            mp.write_text(json.dumps(man, indent=1, sort_keys=True))
            new_manifest_hash = sha256(mp)
            pack = rel / "model.qwp"
            data = bytearray(pack.read_bytes())
            header, _ = read_header(pack)
            old_hash = header["provenance"]["assets_manifest_sha256"]
            idx = data.find(old_hash.encode())
            data[idx:idx + 64] = new_manifest_hash.encode()
            pack.write_bytes(bytes(data))
            (rel / "model.qwp.sha256").write_text(f"{sha256(pack)}  model.qwp\n")
            with self.assertRaisesRegex(ValueError, "first_row"):
                audit_release(rel, checkpoint=src)


class RouterLawTests(unittest.TestCase):
    REAL = {"layers": {"1": {
        "layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "ngram_heads_offsets": [0, 20000003, 40000026, 60000059, 80000106,
                                 100000165, 120000228, 140000297, 160000374,
                                 180000455, 200000548, 220000655, 240000802,
                                 260000955, 280001114, 300001275],
        "ngram_heads_vocab_sizes": [20000003, 20000023, 20000033, 20000047,
                                     20000059, 20000063, 20000069, 20000077,
                                     20000081, 20000093, 20000107, 20000147,
                                     20000153, 20000159, 20000161, 20000171]}}}

    def test_production_constants_and_violations(self):
        validate_router_constants(self.REAL)
        import copy
        rc = copy.deepcopy(self.REAL)
        rc["layers"]["1"]["layer_multipliers"][1] = 20109073645364
        with self.assertRaises(ValueError):
            validate_router_constants(rc)


class HybridHQ8PackTests(unittest.TestCase):
    HQ8_NAMES = (
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    )
    DENY_NAME = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"

    def source(self, root):
        source = root / "source"
        source.mkdir()
        for name in ("config.json", "tokenizer.json"):
            (source / name).write_text("{}")
        (source / "chat_template.jinja").write_text("template")
        g = torch.Generator().manual_seed(5)
        tensors = {
            self.HQ8_NAMES[0]: torch.randn(64, 128, generator=g),
            self.HQ8_NAMES[1]: torch.randn(32, 128, generator=g),
            self.DENY_NAME: torch.randn(64, 128, generator=g),
            "model.language_model.layers.0.mlp.experts.gate_up_proj":
                torch.randn(2, 4, 128, generator=g),
            "model.language_model.layers.0.attn_hyper_connection."
            "input_mix_weight_up.weight": torch.randn(8, 128, generator=g),
            "model.language_model.embed_tokens.weight":
                torch.randn(2, 128, generator=g),
            "model.language_model.norm.weight": torch.randn(128, generator=g),
        }
        save_file(tensors, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
        return source

    def test_hybrid_cook_copies_hq4_bytes_and_cooks_hq8_planes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            reference = root / "hq4"
            hybrid = root / "hq8"
            quantize_model(source, reference)
            quantize_model(source, hybrid, hybrid_hq8_from=reference)
            header, counts = audit_pack(hybrid / "model.qwp")
            self.assertEqual(counts["hq8_g32_m5"], 2)
            self.assertEqual(header["provenance"]["quant_recorded"],
                             "hq8_hq4_hybrid_v1")
            for name in self.HQ8_NAMES:
                entry = header["tensors"][name]
                self.assertIn("hq8_off", entry)
                self.assertIn("hq8_s_off", entry)
                self.assertNotIn("hq4_off", entry)
            expert = header["tensors"][
                "model.language_model.layers.0.mlp.experts.gate_up_proj"]
            self.assertIn("hq4_off", expert)
            denied = header["tensors"][self.DENY_NAME]
            self.assertIn("hq4_off", denied)
            self.assertNotIn("hq8_off", denied)
            self.assertEqual(expert["hq4_off"] % 16384, 0)
            compared, skipped = compare_planes(
                hybrid / "model.qwp", reference / "model.qwp",
                allow_requantized=self.HQ8_NAMES)
            self.assertEqual(sorted(skipped), sorted(self.HQ8_NAMES))
            self.assertGreater(compared, 0)

    def test_hybrid_fail_closed_on_unknown_quant_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            reference = root / "hq4"
            quantize_model(source, reference)
            # Remove the reference HQ4 entry for a non-HQ8 candidate by
            # repacking without it: a NEW quant candidate tensor appears.
            late = root / "source2"
            shutil.copytree(source, late)
            tensors = {}
            from safetensors.torch import load_file
            tensors.update(load_file(late / "model.safetensors"))
            tensors["model.language_model.layers.7.attn_hyper_connection."
                    "input_mix_weight_up.weight"] = torch.randn(16, 128)
            (late / "model.safetensors").unlink()
            save_file(tensors, late / "model.safetensors")
            names = sorted(tensors)
            (late / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "model.safetensors" for name in names}}))
            with self.assertRaisesRegex(ValueError, "fail-closed"):
                quantize_model(late, root / "hq8", hybrid_hq8_from=reference)

    def test_hq8_target_deny_list(self):
        from pack import hq8_target
        self.assertTrue(hq8_target(
            "model.language_model.layers.3.self_attn.q_proj.weight"))
        self.assertFalse(hq8_target(
            "model.language_model.layers.3.mlp.experts.gate_up_proj"))
        self.assertFalse(hq8_target(
            "model.language_model.layers.3.attn_hyper_connection."
            "input_mix_weight_up.weight"))
        self.assertFalse(hq8_target("lm_head.weight"))


if __name__ == "__main__":
    unittest.main()
