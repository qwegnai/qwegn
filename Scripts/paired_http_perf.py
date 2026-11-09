#!/usr/bin/env python3
"""Paired streaming chat benchmarks against two user-managed loopback servers."""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from paired_perf import arm_order, digest, write_report


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("endpoints must be plain HTTP on 127.0.0.1 or [::1], without credentials or query")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid endpoint port: {error}") from error
    if port is None or not parsed.path.startswith("/"):
        raise ValueError("endpoint must include a port and request path")
    return parsed.hostname, port, parsed.path


def request_one(target, payload, timeout, barrier):
    host, port, path = target
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    barrier.wait()
    started = time.monotonic()
    try:
        connection.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return {"error": f"HTTP status {response.status}"}
        first = None
        texts = {key: hashlib.sha256() for key in ("content", "reasoning_content")}
        usage = None
        done = False
        while True:
            line = response.readline()
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"request exceeded {timeout} seconds")
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError("SSE data must be a JSON object")
            if "error" in chunk or chunk.get("type") == "server_error":
                raise ValueError("server returned a streaming error")
            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                raise ValueError("stream choices must be an array")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("stream choice must be an object")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ValueError("stream delta must be an object")
                for key in texts:
                    piece = delta.get(key)
                    if piece:
                        if not isinstance(piece, str):
                            raise ValueError(f"streamed {key} must be a string")
                        if first is None:
                            first = time.monotonic()
                        texts[key].update(piece.encode("utf-8"))
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
        ended = time.monotonic()
        if not done or first is None or not isinstance(usage, dict):
            raise ValueError("stream must end with [DONE], text, and a usage object")
        prompt = usage.get("prompt_tokens")
        generated = usage.get("completion_tokens")
        if (not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0
                or not isinstance(generated, int) or isinstance(generated, bool) or generated <= 0):
            raise ValueError("stream usage must include valid prompt_tokens and completion_tokens")
        return {
            "prompt_tokens": prompt, "generated_tokens": generated,
            "ttft_seconds": first - started, "wall_seconds": ended - started,
            "text_sha256": hashlib.sha256(b"".join(texts[key].digest() for key in texts)).hexdigest(),
        }
    except (OSError, http.client.HTTPException, UnicodeDecodeError, ValueError, TypeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        connection.close()


def run_batch(target, payload, concurrency, timeout):
    barrier = threading.Barrier(concurrency)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_one, target, payload, timeout, barrier) for _ in range(concurrency)]
        requests = [future.result() for future in futures]
    wall = time.monotonic() - started
    result = {"requests": requests, "wall_seconds": wall, "load_average": list(os.getloadavg())}
    if any("error" in request for request in requests):
        result["error"] = "one or more requests failed; inspect requests"
    else:
        result["aggregate_end_to_end_tps"] = sum(r["generated_tokens"] for r in requests) / wall
        result["ttft_median_seconds"] = statistics.median(r["ttft_seconds"] for r in requests)
        result["ttft_max_seconds"] = max(r["ttft_seconds"] for r in requests)
    return result


def summarize(samples):
    summary = {}
    for metric in ("aggregate_end_to_end_tps", "ttft_median_seconds", "ttft_max_seconds", "wall_seconds"):
        pairs = [(sample["baseline"][metric], sample["candidate"][metric]) for sample in samples]
        if any(base <= 0 for base, _ in pairs):
            raise ValueError(f"{metric} has a nonpositive baseline")
        ratios = [candidate / base for base, candidate in pairs]
        middle = statistics.median(ratios)
        summary[metric] = {
            "baseline_median": statistics.median(base for base, _ in pairs),
            "candidate_median": statistics.median(candidate for _, candidate in pairs),
            "paired_ratio_median": middle,
            "paired_ratio_mad": statistics.median(abs(value - middle) for value in ratios),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Loopback chat completions URL")
    parser.add_argument("--candidate", required=True, help="Loopback chat completions URL")
    parser.add_argument("--request", type=Path, required=True, help="Chat request JSON without secrets")
    parser.add_argument("--model-id", required=True, help="Checkpoint identity for both servers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cooldown-seconds", type=float, default=0,
                        help="Idle before each arm, including warmups, to limit thermal bias")
    parser.add_argument("--notes", default="", help="Server builds, configs, hardware and cache state")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be finite and nonnegative")
    if (args.concurrency < 1 or args.samples < 5 or args.warmups < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0 or not args.model_id.strip()):
        parser.error("require positive concurrency and timeout, at least 5 samples and 1 warmup, and a model ID")
    try:
        targets = {"baseline": endpoint(args.baseline), "candidate": endpoint(args.candidate)}
        body = json.loads(args.request.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body.get("model"):
        parser.error("request must be an object with model and messages")
    body["stream"] = True
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    report = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "baseline_url": args.baseline, "candidate_url": args.candidate,
        "model_id": args.model_id, "request_file_sha256": digest(args.request),
        "effective_request_sha256": hashlib.sha256(payload).hexdigest(),
        "concurrency": args.concurrency, "notes": args.notes, "cooldown_seconds": args.cooldown_seconds,
        "warmups": [], "samples": [], "status": "in_progress",
    }
    try:
        with args.output.open("x", encoding="utf-8") as output:
            write_report(output, report)
            for phase, count in (("warmups", args.warmups), ("samples", args.samples)):
                for index in range(count):
                    pair = {"order": list(arm_order(index))}
                    report[phase].append(pair)
                    for arm in pair["order"]:
                        if args.cooldown_seconds:
                            time.sleep(args.cooldown_seconds)
                        pair[arm] = run_batch(targets[arm], payload, args.concurrency, args.timeout)
                        write_report(output, report)
                        if "error" in pair[arm]:
                            report["status"] = "failed"
                            write_report(output, report)
                            print(f"{phase} {index + 1} {arm}: {pair[arm]['error']}", file=sys.stderr)
                            return 1
            if digest(args.request) != report["request_file_sha256"]:
                report["status"] = "failed"
                report["error"] = "request file changed during measurement"
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            report["summary"] = summarize(report["samples"])
            report["token_counts_stable"] = len({
                (r["prompt_tokens"], r["generated_tokens"])
                for pair in report["samples"] for arm in ("baseline", "candidate")
                for r in pair[arm]["requests"]
            }) == 1
            report["text_hashes_stable"] = len({
                r["text_sha256"] for pair in report["samples"]
                for arm in ("baseline", "candidate") for r in pair[arm]["requests"]
            }) == 1
            report["status"] = "complete"
            write_report(output, report)
    except FileExistsError:
        parser.error(f"report already exists: {args.output}")
    print(f"paired HTTP report written: {args.output}")
    print("Inspect per-request counts and hashes; aggregate throughput includes TTFT and queuing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
