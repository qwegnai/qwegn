import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paired_perf


STATS = """prompt tokens: 10
generated tokens: 4
prefill: 1.00 s (10.0 tok/s, GPU: 1.00 s / 10.0 tok/s)
TTFT (first token): 0.25 s
decode: 0.1 s (40.00 tok/s)
steady-state allocs: 0 (law: 0)
"""


class PairedPerfTests(unittest.TestCase):
    def test_statistics_and_missing_fields(self):
        self.assertEqual(paired_perf.parse_statistics(STATS), {
            "prefill_tps": 10.0, "decode_tps": 40.0, "ttft_seconds": 0.25,
            "prompt_tokens": 10, "generated_tokens": 4, "steady_allocations": 0,
        })
        with self.assertRaisesRegex(ValueError, "incomplete"):
            paired_perf.parse_statistics(STATS.replace("generated tokens:", "missing:"))

    def test_order_and_paired_summary(self):
        self.assertEqual(paired_perf.arm_order(0), ("baseline", "candidate"))
        self.assertEqual(paired_perf.arm_order(1), ("candidate", "baseline"))
        rows = []
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            base = {key: value for key in ("prefill_tps", "decode_tps", "ttft_seconds", "wall_seconds")}
            candidate = {key: value * 2 for key in base}
            rows.append({"baseline": base, "candidate": candidate})
        summary = paired_perf.summarize(rows)
        self.assertEqual(summary["decode_tps"]["paired_ratio_median"], 2.0)
        self.assertEqual(summary["decode_tps"]["paired_ratio_mad"], 0.0)
        rows[0]["baseline"]["decode_tps"] = 0
        with self.assertRaisesRegex(ValueError, "nonpositive baseline"):
            paired_perf.summarize(rows)

    def test_paired_cli_and_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-qwg"
            executable.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                "assert os.environ.get('QW_METALLIB')\n"
                f"sys.stderr.write({STATS!r})\nprint('same output')\n", encoding="utf-8"
            )
            executable.chmod(0o755)
            model, prompt, output, metallib = root / "model", root / "prompt", root / "report.json", root / "kernel.metallib"
            model.write_text("model", encoding="utf-8")
            prompt.write_text("prompt", encoding="utf-8")
            metallib.write_text("metal", encoding="utf-8")
            argv = ["--baseline", str(executable), "--candidate", str(executable),
                    "--baseline-metallib", str(metallib), "--candidate-metallib", str(metallib),
                    "--model", str(model), "--model-id", "test-checkpoint",
                    "--prompt-file", str(prompt), "--output", str(output), "--cooldown-seconds", "60"]
            with contextlib.redirect_stdout(io.StringIO()), patch("paired_perf.time.sleep") as sleep:
                self.assertEqual(paired_perf.main(argv), 0)
                self.assertEqual(sleep.call_count, 12)
                self.assertTrue(all(call.args == (60,) for call in sleep.call_args_list))
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["cooldown_seconds"], 60)
            self.assertEqual(len(report["samples"]), 5)
            self.assertEqual(report["samples"][1]["order"], ["candidate", "baseline"])
            self.assertTrue(report["output_hashes_match"])
            self.assertTrue(report["token_counts_match"])
            self.assertTrue(report["token_counts_stable"])
            self.assertEqual(report["summary"]["decode_tps"]["paired_ratio_median"], 1.0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                paired_perf.main(argv)

            executable.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n", encoding="utf-8")
            failed = root / "failed.json"
            argv[argv.index(str(output))] = str(failed)
            with contextlib.redirect_stderr(io.StringIO()), patch("paired_perf.time.sleep"):
                self.assertEqual(paired_perf.main(argv), 1)
            failure = json.loads(failed.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["warmups"][0]["baseline"]["exit_code"], 7)

    def test_rejects_mismatched_benchmark_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            paired_perf.main(["--baseline", "a", "--candidate", "b", "--model", "c",
                              "--baseline-metallib", "f", "--candidate-metallib", "g",
                              "--model-id", "id", "--prompt-file", "d", "--output", "e",
                              "--options-json", '["--model=other"]'])

    def test_rejects_invalid_cooldown(self):
        for value in ("-1", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                paired_perf.main(["--baseline", "a", "--candidate", "b", "--model", "c",
                                  "--baseline-metallib", "f", "--candidate-metallib", "g",
                                  "--model-id", "id", "--prompt-file", "d", "--output", "e",
                                  "--cooldown-seconds", value])


if __name__ == "__main__":
    unittest.main()
