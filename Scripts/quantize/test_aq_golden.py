"""The load-time writers' goldens must not drift from the packer.

`aq_quantizer_golden.json` is what the incumbent packer writes, and
`aq_repack_golden.json` is MLX's own. If either moves, the GPU writers
would be matching a stale reference — so both are regenerated and
compared here.
"""
import importlib.util
import json
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "Scripts" / "quantize"))
import hq4  # noqa: E402

FIXTURES = ROOT / "Tests" / "QWQuantTests" / "fixtures"


def _generator():
    path = ROOT / "Scripts" / "oracle" / "gen_aq_fixtures.py"
    spec = importlib.util.spec_from_file_location("gen_aq_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AQuantizerGoldenTests(unittest.TestCase):
    def test_the_committed_golden_is_what_the_packer_writes(self):
        doc = json.loads((FIXTURES / "aq_quantizer_golden.json").read_text())
        bf16 = np.frombuffer(bytes.fromhex(doc["bf16"]), dtype="<u2")
        values = (bf16.astype(np.uint32) << 16).view(np.float32)
        blocks = hq4.quant_hq4_g32_m5(values.reshape(doc["rows"], doc["cols"]))
        self.assertEqual(np.ascontiguousarray(blocks).tobytes().hex(), doc["blocks"])

    def test_the_generator_reproduces_the_committed_golden(self):
        generator = _generator()
        built = generator.build_quantizer_golden()
        doc = json.loads((FIXTURES / "aq_quantizer_golden.json").read_text())
        self.assertEqual(built, doc)


class ARepackGoldenTests(unittest.TestCase):
    def setUp(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            self.skipTest("mlx is not installed")

    def test_the_committed_golden_is_mlxs_own(self):
        generator = _generator()
        built = generator.build_repack_golden()
        doc = json.loads((FIXTURES / "aq_repack_golden.json").read_text())
        self.assertEqual(built, doc)


if __name__ == "__main__":
    unittest.main()
