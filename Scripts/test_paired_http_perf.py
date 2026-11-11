import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import paired_http_perf


class ChatHandler(BaseHTTPRequestHandler):
    pieces = ["answer"]
    include_usage = True
    engine_info = {
        "sources": {"noProfile": "explicit"},
        "parameters": {
            "kvMode": "q4", "contextBudget": 4096, "mtpDrafts": 0, "mtpBackend": None,
            "lsaMode": "off", "gdnPrefillMode": "chunked", "prefillScratchEviction": "auto",
        },
    }

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/v1/engine/info":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.engine_info).encode("utf-8"))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in self.pieces:
            data = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
        if self.include_usage:
            data = {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


class SplitChatHandler(ChatHandler):
    pieces = ["an", "swer"]


class MissingUsageHandler(ChatHandler):
    include_usage = False


class PairedHTTPPerfTests(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join()

    def server_url(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    def test_loopback_only_and_endpoint_validation(self):
        self.assertEqual(paired_http_perf.endpoint("http://127.0.0.1:8000/v1/chat"),
                         ("127.0.0.1", 8000, "/v1/chat"))
        for url in ("http://example.com:8000/v1/chat", "http://127.0.0.1/v1/chat",
                    "http://127.0.0.1:8000/v1/chat?key=x", "https://127.0.0.1:8000/v1/chat"):
            with self.assertRaises(ValueError):
                paired_http_perf.endpoint(url)

    def test_paired_concurrency_and_chunk_independent_hash(self):
        baseline = self.server_url(ChatHandler)
        candidate = self.server_url(SplitChatHandler)
        with tempfile.TemporaryDirectory() as directory:
            request, output = Path(directory) / "request.json", Path(directory) / "report.json"
            request.write_text(json.dumps({"model": "test", "messages": [{"role": "user", "content": "private"}]}))
            argv = ["--baseline", baseline, "--candidate", candidate, "--request", str(request),
                    "--output", str(output), "--model-id", "checkpoint", "--concurrency", "2",
                    "--cooldown-seconds", "60"]
            with contextlib.redirect_stdout(io.StringIO()), patch("paired_http_perf.time.sleep") as sleep:
                self.assertEqual(paired_http_perf.main(argv), 0)
                self.assertEqual(sleep.call_count, 12)
                self.assertTrue(all(call.args == (60,) for call in sleep.call_args_list))
            raw = output.read_text()
            report = json.loads(raw)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["cooldown_seconds"], 60)
            self.assertEqual(len(report["samples"]), 5)
            self.assertEqual(report["samples"][1]["order"], ["candidate", "baseline"])
            self.assertTrue(report["token_counts_stable"])
            self.assertTrue(report["text_hashes_stable"])
            self.assertEqual(len(report["samples"][0]["baseline"]["requests"]), 2)
            self.assertGreater(report["summary"]["aggregate_end_to_end_tps"]["baseline_median"], 0)
            self.assertNotIn("private", raw)
            self.assertNotIn("answer", raw)

    def test_missing_usage_is_recorded_as_failure(self):
        target = paired_http_perf.endpoint(self.server_url(MissingUsageHandler))
        result = paired_http_perf.run_batch(target, b'{"stream":true}', 2, 5)
        self.assertIn("error", result)
        self.assertTrue(all("usage object" in request["error"] for request in result["requests"]))

    def test_strict_engine_info_requires_no_profile_and_matching_params(self):
        class ProfiledHandler(ChatHandler):
            engine_info = {**ChatHandler.engine_info, "sources": {}}

        class MismatchedKVHandler(ChatHandler):
            engine_info = {
                **ChatHandler.engine_info,
                "parameters": {**ChatHandler.engine_info["parameters"], "kvMode": "f16"},
            }

        baseline = self.server_url(ChatHandler)
        candidate = self.server_url(SplitChatHandler)
        profiled = self.server_url(ProfiledHandler)
        mismatched = self.server_url(MismatchedKVHandler)
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))
            for bad_cand in (profiled, mismatched):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    paired_http_perf.main(["--baseline", baseline, "--candidate", bad_cand,
                                           "--request", str(request), "--output", str(Path(directory) / "bad.json"),
                                           "--model-id", "checkpoint", "--strict"])
            good_out = Path(directory) / "strict-ok.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(paired_http_perf.main(
                    ["--baseline", baseline, "--candidate", candidate, "--request", str(request),
                     "--output", str(good_out), "--model-id", "checkpoint", "--concurrency", "1", "--strict"]), 0)
            report = json.loads(good_out.read_text())
            self.assertTrue(report["strict"])
            self.assertEqual(report["engine_info"]["baseline"]["sources"]["noProfile"], "explicit")

    def test_rejects_invalid_cooldown(self):
        for value in ("-1", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                paired_http_perf.main(["--baseline", "http://127.0.0.1:8000/v1/chat/completions",
                                       "--candidate", "http://127.0.0.1:8001/v1/chat/completions",
                                       "--request", "a", "--output", "b", "--model-id", "test",
                                       "--cooldown-seconds", value])


if __name__ == "__main__":
    unittest.main()
