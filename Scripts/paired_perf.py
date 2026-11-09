#!/usr/bin/env python3
"""Paired, one-process-at-a-time qwg CLI measurements."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from perf_gate import metrics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def parse_statistics(stderr):
    prefill, decode, allocs = metrics(stderr)
    prompt = re.search(r"prompt tokens:\s*(\d+)", stderr)
    generated = re.search(r"generated tokens:\s*(\d+)", stderr)
    ttft = re.search(r"TTFT \(first token\):\s*([\d.]+)\s+s", stderr)
    if prefill is None or decode is None or prompt is None or generated is None or ttft is None:
        raise ValueError("qwg statistics are incomplete; see the captured stderr")
    return {
        "prefill_tps": prefill,
        "decode_tps": decode,
        "ttft_seconds": float(ttft.group(1)),
        "prompt_tokens": int(prompt.group(1)),
        "generated_tokens": int(generated.group(1)),
        "steady_allocations": allocs,
    }


def arm_order(index):
    return ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")


def summarize(samples):
    result = {}
    for metric in ("prefill_tps", "decode_tps", "ttft_seconds", "wall_seconds"):
        pairs = [(r["baseline"][metric], r["candidate"][metric]) for r in samples]
        if any(base <= 0 for base, _ in pairs):
            raise ValueError(f"{metric} has a nonpositive baseline; cannot compute paired ratios")
        ratios = [candidate / base for base, candidate in pairs]
        middle = statistics.median(ratios)
        result[metric] = {
            "baseline_median": statistics.median(base for base, _ in pairs),
            "candidate_median": statistics.median(candidate for _, candidate in pairs),
            "paired_ratio_median": middle,
            "paired_ratio_mad": statistics.median(abs(value - middle) for value in ratios),
        }
    return result


def run_one(binary, metallib, model, prompt, max_tokens, options, timeout):
    command = [str(binary), "run", "--model", str(model), "--prompt-file", str(prompt),
               "--max-tokens", str(max_tokens), "--no-stream", "--verbose", *options]
    started = time.monotonic()
    try:
        process = subprocess.run(command, capture_output=True, timeout=timeout,
                                 env={**os.environ, "QW_METALLIB": str(metallib)})
    except subprocess.TimeoutExpired as error:
        return {"command": command, "error": f"timed out after {timeout} seconds",
                "stdout_sha256": hashlib.sha256(error.stdout or b"").hexdigest(),
                "stderr": (error.stderr or b"").decode("utf-8", errors="replace")}
    except OSError as error:
        return {"command": command, "error": f"could not start qwg: {error}"}
    result = {
        "command": command,
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "load_average": list(os.getloadavg()),
        "stdout_sha256": hashlib.sha256(process.stdout).hexdigest(),
        "stderr": process.stderr.decode("utf-8", errors="replace"),
    }
    if process.returncode != 0:
        result["error"] = f"qwg exited with status {process.returncode}"
        return result
    try:
        result.update(parse_statistics(result["stderr"]))
    except ValueError as error:
        result["error"] = str(error)
    return result


def write_report(stream, report):
    stream.seek(0)
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.truncate()
    stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline qwg executable")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate qwg executable")
    parser.add_argument("--baseline-metallib", type=Path, required=True)
    parser.add_argument("--candidate-metallib", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Same model path for both arms")
    parser.add_argument("--model-id", required=True, help="User-supplied checkpoint identity or digest")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; never overwritten")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cooldown-seconds", type=float, default=0,
                        help="Idle before each arm, including warmups, to limit thermal bias")
    parser.add_argument("--options-json", default="[]", help="JSON array of shared qwg run arguments")
    parser.add_argument("--notes", default="", help="Hardware, settings, cache and workload provenance")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be finite and nonnegative")
    if args.samples < 5 or args.warmups < 1 or args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require at least 5 samples, 1 warmup, positive max-tokens and timeout")
    try:
        options = json.loads(args.options_json)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --options-json: {error}")
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        parser.error("--options-json must be an array of strings")
    controlled = {"--model", "-m", "--prompt", "--prompt-file", "--max-tokens",
                  "--max-new", "--stream", "--no-stream", "--verbose", "--repl"}
    if any(option.split("=", 1)[0] in controlled for option in options):
        parser.error("--options-json must not override model, prompt, token count, streaming or verbose")
    for path in (args.baseline, args.candidate, args.baseline_metallib,
                 args.candidate_metallib, args.prompt_file):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    if not args.model.exists():
        parser.error(f"model path does not exist: {args.model}")
    if not args.model_id.strip():
        parser.error("--model-id must not be empty")
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    metallibs = {"baseline": args.baseline_metallib.resolve(),
                 "candidate": args.candidate_metallib.resolve()}
    model, prompt = args.model.resolve(), args.prompt_file.resolve()
    identity = {"model": file_identity(model), "prompt_sha256": digest(prompt),
                "binaries": {name: {**file_identity(path), "sha256": digest(path)}
                             for name, path in binaries.items()},
                "metallibs": {name: {**file_identity(path), "sha256": digest(path)}
                              for name, path in metallibs.items()}}
    report = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "load_average": list(os.getloadavg()), "model_id": args.model_id,
        "identity": identity, "options": options, "max_tokens": args.max_tokens,
        "notes": args.notes, "cooldown_seconds": args.cooldown_seconds,
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
                        pair[arm] = run_one(binaries[arm], metallibs[arm], model, prompt, args.max_tokens,
                                            options, args.timeout)
                        write_report(output, report)
                        if "error" in pair[arm]:
                            report["status"] = "failed"
                            write_report(output, report)
                            print(f"{phase} {index + 1} {arm}: {pair[arm]['error']}", file=sys.stderr)
                            return 1
            if file_identity(model) != identity["model"] or digest(prompt) != identity["prompt_sha256"]:
                report["status"] = "failed"
                report["error"] = "model path metadata or prompt changed during measurement"
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            if any(digest(path) != identity["binaries"][name]["sha256"]
                   for name, path in binaries.items()):
                report["status"] = "failed"
                report["error"] = "a benchmark executable changed during measurement"
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            if any(digest(path) != identity["metallibs"][name]["sha256"]
                   for name, path in metallibs.items()):
                report["status"] = "failed"
                report["error"] = "a benchmark Metal library changed during measurement"
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            try:
                report["summary"] = summarize(report["samples"])
            except ValueError as error:
                report["status"] = "failed"
                report["error"] = str(error)
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            report["output_hashes_match"] = all(
                pair["baseline"]["stdout_sha256"] == pair["candidate"]["stdout_sha256"]
                for pair in report["samples"]
            )
            report["token_counts_match"] = all(
                (pair["baseline"]["prompt_tokens"], pair["baseline"]["generated_tokens"]) ==
                (pair["candidate"]["prompt_tokens"], pair["candidate"]["generated_tokens"])
                for pair in report["samples"]
            )
            report["token_counts_stable"] = len({
                (pair[arm]["prompt_tokens"], pair[arm]["generated_tokens"])
                for pair in report["samples"] for arm in ("baseline", "candidate")
            }) == 1
            report["status"] = "complete"
            write_report(output, report)
    except FileExistsError:
        parser.error(f"report already exists: {args.output}")
    print(f"paired report written: {args.output}")
    print("Inspect raw samples, settings and output hashes; no performance or fidelity verdict is implied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
