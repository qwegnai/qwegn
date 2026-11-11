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

STRICT_OPTIONS = [
    "--kv-mode", "q4", "--context-tokens", "4096", "--prefill-chunk-size", "512",
    "--temperature", "0", "--top-p", "1", "--top-k", "1", "--seed", "0",
    "--repetition-penalty", "1", "--mtp-drafts", "0",
    "--mtp-draft-strategy", "neural", "--mtp-relaxed-margin", "0",
    "--mtp-relaxed-min-prob", "0", "--lsa-mode", "off", "--no-thinking", "--no-prefix-cache",
    "--no-profile",
]


class PairedPerfTests(unittest.TestCase):
    def test_strict_options_require_pinned_settings(self):
        self.assertEqual(paired_perf.strict_options(STRICT_OPTIONS)["--kv-mode"], "q4")
        for options in ([], STRICT_OPTIONS + ["--kv-mode=f16"],
                        STRICT_OPTIONS[:-1], ["--kv-mode"],
                        STRICT_OPTIONS + ["--thinking"],
                        STRICT_OPTIONS + ["--enable-thinking=true"],
                        STRICT_OPTIONS + ["--dflash-window=4"],
                        STRICT_OPTIONS + ["--mtp-backend=standard"],
                        STRICT_OPTIONS + ["--max-context=1024"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired_perf.strict_options(options)
        for key, value in (("--temperature", "1"),
                           ("--seed", "-1"), ("--mtp-relaxed-margin", "1.5"),
                           ("--context-tokens", "0"), ("--prefill-chunk-size", "33"),
                           ("--mtp-drafts", "-1")):
            options = STRICT_OPTIONS.copy()
            options[options.index(key) + 1] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                paired_perf.strict_options(options)

    def test_strict_speculative_backend_contract(self):
        options = STRICT_OPTIONS.copy()
        options[options.index("--mtp-drafts") + 1] = "1"
        with self.assertRaisesRegex(ValueError, "require --mtp-backend standard"):
            paired_perf.strict_options(options)
        self.assertEqual(paired_perf.strict_options(
            options + ["--mtp-backend", "standard"])["--mtp-backend"], "standard")
        with self.assertRaisesRegex(ValueError, "require --mtp-backend standard"):
            paired_perf.strict_options(options + ["--mtp-backend", "variational"])

    def test_model_identity_tracks_contents_and_nested_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "assets").mkdir()
            (model / "assets" / "config.json").write_text("{}")
            (model / "model.qwp").write_bytes(b"pack")
            (model / "assets" / "ngram.bin").write_bytes(b"ple")
            first = paired_perf.model_identity(model, hash_files=True)
            self.assertEqual(len(first["files"]), 3)
            self.assertIn("sha256", first["files"]["assets/ngram.bin"])
            self.assertEqual(first, paired_perf.model_identity(model / "model.qwp", hash_files=True))
            (model / "assets" / "ngram.bin").write_bytes(b"new")
            self.assertNotEqual(first, paired_perf.model_identity(model, hash_files=True))
            (model / "assets" / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "config.json"):
                paired_perf.model_identity(model)

    def test_strict_pair_refuses_effective_kv_context_and_parity_mismatch(self):
        expected = paired_perf.strict_options(STRICT_OPTIONS)

        def result(kv="q4", context=4096, config="[config] profile=bypassed kv_mode=q4 lsa_mode=off mtp_drafts=0\n"):
            return {**paired_perf.parse_statistics(STATS), "stdout_sha256": "same",
                    "stderr": f"[memory] kv_mode={kv} max_tokens={context}\n{config}"}

        paired_perf.check_strict_pair({"baseline": result(), "candidate": result()}, expected)
        cases = [
            result("f16"),
            result(context=2048),
            result(config="[config] profile=applied kv_mode=q4 lsa_mode=off mtp_drafts=0\n"),
            result(config="[config] profile=bypassed kv_mode=q4 lsa_mode=auto mtp_drafts=0\n"),
            result(config="[config] profile=bypassed kv_mode=q4 lsa_mode=off mtp_drafts=2\n"),
            result(),
        ]
        cases[-1]["stdout_sha256"] = "different"
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                paired_perf.check_strict_pair({"baseline": result(), "candidate": candidate}, expected)
        for stderr in ("", "[memory] kv_mode=q4\n",
                       "[memory] kv_mode=q4 max_tokens=4096\n" * 2):
            with self.subTest(stderr=stderr), self.assertRaises(ValueError):
                paired_perf.effective_memory(stderr)

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
            strict = root / "strict.json"
            strict_argv = argv.copy()
            strict_argv[strict_argv.index(str(output))] = str(strict)
            with contextlib.redirect_stderr(io.StringIO()), patch("paired_perf.time.sleep"):
                self.assertEqual(paired_perf.main(
                    strict_argv + ["--strict", "--options-json", json.dumps(STRICT_OPTIONS)]), 1)
            self.assertEqual(json.loads(strict.read_text())["status"], "incomparable")
            executable.write_text(
                "#!/usr/bin/env python3\nimport sys\n"
                f"sys.stderr.write({STATS!r})\n"
                "sys.stderr.write('[memory] kv_mode=q4 max_tokens=4096\\n')\n"
                "print('same output')\n", encoding="utf-8"
            )
            strict_pass = root / "strict-pass.json"
            strict_argv[strict_argv.index(str(strict))] = str(strict_pass)
            with contextlib.redirect_stdout(io.StringIO()), patch("paired_perf.time.sleep"):
                self.assertEqual(paired_perf.main(
                    strict_argv + ["--strict", "--options-json", json.dumps(STRICT_OPTIONS)]), 0)
            strict_report = json.loads(strict_pass.read_text())
            self.assertEqual(strict_report["status"], "complete")
            self.assertIn("sha256", strict_report["identity"]["model_files"]["files"]["model"])
            self.assertEqual(strict_report["samples"][0]["candidate"]["effective_memory"]["kv_mode"], "q4")
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
