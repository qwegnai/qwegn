import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import probe_quality


class QualityProbeTests(unittest.TestCase):
    def test_exact_marker_and_nested_shape_assertions(self):
        self.assertIsNone(probe_quality.grade("marker", json.dumps({"note": probe_quality.MARKER})))
        self.assertIsNotNone(probe_quality.grade(
            "marker", json.dumps({"note": probe_quality.MARKER.replace("<|message|>", "<message>")})))
        self.assertIsNone(probe_quality.grade(
            "nested-schema", '{"user":{"name":"Ada","pets":["cat","dog"]}}'))
        for text in ('{"user":{"name":"Ada","pets":[]}}',
                     '{"user":{"name":"Ada","pets":[1,2]}}', "```json\n{}\n```"):
            self.assertIsNotNone(probe_quality.grade("nested-schema", text))

    def test_prompt_only_probe_does_not_request_constraints_or_logprobs(self):
        request = probe_quality.request_for("nested-prompt", "model", 42)
        self.assertNotIn("response_format", request)
        self.assertNotIn("logprobs", request)
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["top_k"], 40)
        self.assertEqual(request["top_p"], 0.9)
        self.assertEqual(request["repetition_penalty"], 1.0)
        self.assertFalse(request["enable_thinking"])

    def test_failed_verdict_is_persisted_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            response = {"text": '{"user":{"name":"Ada","pets":[]}}', "finish_reason": "stop"}
            with patch("probe_quality.send", return_value=response.copy()), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = probe_quality.main([
                    "--endpoint", "http://127.0.0.1:18081/v1/chat/completions",
                    "--model", "model", "--case", "nested-prompt", "--trials", "1",
                    "--output", str(output),
                ])
            self.assertEqual(status, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["passed"], 0)
            self.assertIsNone(report["raw_token_ids"])
            self.assertEqual(report["trials"][0]["request"]["seed"], 2026)


if __name__ == "__main__":
    unittest.main()
