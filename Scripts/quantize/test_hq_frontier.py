import json
import tempfile
import unittest
from pathlib import Path

from hq_frontier import audit_source, candidate_cost, pareto_frontier


class HQFrontierTests(unittest.TestCase):
    def test_cost_preserves_raw_planes_and_resolves_groups(self):
        header = {"tensors": {
            "lm_head.weight": {"shape": [2, 320], "hq4_off": 256},
            "model.mlp.experts.down_proj": {
                "shape": [2, 4, 256], "experts": 2, "e_hq4": 640, "hq4_off": 16384,
            },
            "model.norm.weight": {"shape": [320], "kind": "bf16", "raw_off": 32768},
        }}
        baseline = candidate_cost(header, 4096, (4, 32), (4, 32))
        mixed = candidate_cost(header, 4096, (8, 128), (4, 64))
        self.assertEqual(baseline["estimated_pack_bytes"], 4096)
        self.assertEqual(mixed["raw_bytes"], 640)
        self.assertEqual(mixed["effective_tensor_formats"], {"hq8_g64": 1, "hq4_g64": 1})
        self.assertEqual(mixed["estimated_pack_bytes"], 4096 + 280 - 128)

    def test_frontier_requires_actual_compatible_measurements(self):
        base = {"teacher_sha256": "teacher", "corpus_sha256": "corpus"}
        rows = [
            dict(base, candidate="small", pack_bytes=100, mean_kl=0.02),
            dict(base, candidate="balanced", pack_bytes=110, mean_kl=0.01),
            dict(base, candidate="dominated", pack_bytes=120, mean_kl=0.02),
        ]
        self.assertEqual([r["candidate"] for r in pareto_frontier(rows)], ["small", "balanced"])
        with self.assertRaisesRegex(ValueError, "different teachers"):
            pareto_frontier([rows[0], dict(rows[1], teacher_sha256="other")])
        with self.assertRaisesRegex(ValueError, "finite"):
            pareto_frontier([dict(rows[0], mean_kl=float("nan"))])
        self.assertEqual(pareto_frontier([]), [])

    def test_source_audit_detects_missing_and_truncated_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"weight": "shard.safetensors"},
            }))
            self.assertEqual(audit_source(root)["status"], "incomplete")
            header = json.dumps({"weight": {
                "dtype": "BF16", "shape": [2], "data_offsets": [0, 4],
            }}).encode()
            shard = root / "shard.safetensors"
            shard.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 2)
            with self.assertRaisesRegex(ValueError, "incomplete/invalid"):
                audit_source(root)
            shard.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
            report = audit_source(root)
            self.assertEqual(report["status"], "payload-extents-verified")
            self.assertFalse(report["checksums_verified"])
            expected = hashlib.sha256(shard.read_bytes()).hexdigest()
            with contextlib.redirect_stdout(io.StringIO()):
                verified = audit_source(root, expected)
                mismatch = audit_source(root, "0" * 64)
            self.assertTrue(verified["checksums_verified"])
            self.assertEqual(verified["status"], "source-checksum-verified")
            self.assertFalse(mismatch["checksums_verified"])
            self.assertEqual(mismatch["status"], "source-checksum-mismatch")
            mismatched = json.dumps({"weight": {
                "dtype": "BF16", "shape": [3], "data_offsets": [0, 4],
            }}).encode()
            shard.write_bytes(len(mismatched).to_bytes(8, "little") + mismatched + b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                audit_source(root)

    def test_source_audit_rejects_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"weight": "../outside.safetensors"},
            }))
            with self.assertRaisesRegex(ValueError, "escapes"):
                audit_source(root)


if __name__ == "__main__":
    unittest.main()
import contextlib
import hashlib
import io
