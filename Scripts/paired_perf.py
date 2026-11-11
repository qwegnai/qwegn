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


def model_identity(path, *, hash_files=False):
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        root = path
        files = [path / "model.qwp"]
        assets = root / "assets"
        if not files[0].is_file() or not (assets / "config.json").is_file():
            raise ValueError("model directory must contain model.qwp and assets/config.json")
    assets = root / "assets"
    if assets.is_dir():
        files.extend(sorted(p for p in assets.rglob("*") if p.is_file()))
    return {
        "root": str(root.resolve()),
        "files": {
            str(p.relative_to(root)): {
                **file_identity(p), **({"sha256": digest(p)} if hash_files else {})
            }
            for p in files
        },
    }


def strict_options(options):
    required = {
        "--kv-mode", "--context-tokens", "--prefill-chunk-size",
        "--temperature", "--top-p", "--top-k", "--seed", "--repetition-penalty",
        "--mtp-drafts", "--mtp-draft-strategy",
        "--mtp-relaxed-margin", "--mtp-relaxed-min-prob", "--lsa-mode",
    }
    conflicting = {
        "--max-context", "--enable-thinking", "--prefix-cache",
        "--dflash-window", "--dflash-top-k",
    }
    if any(option.split("=", 1)[0] in conflicting for option in options):
        raise ValueError("strict experiment must not override pinned settings with aliases or DFlash options")
    values = {}
    for index, option in enumerate(options):
        key, separator, value = option.partition("=")
        if key not in required | {"--mtp-backend"}:
            continue
        if key in values:
            raise ValueError(f"strict experiment specifies {key} more than once")
        if not separator:
            if index + 1 >= len(options) or options[index + 1].startswith("--"):
                raise ValueError(f"strict experiment requires a value for {key}")
            value = options[index + 1]
        if not value:
            raise ValueError(f"strict experiment requires a value for {key}")
        values[key] = value
    missing = required - values.keys()
    if missing:
        raise ValueError("strict experiment requires explicit options: " + ", ".join(sorted(missing)))
    if values["--kv-mode"] not in ("f16", "q8", "q4"):
        raise ValueError("strict experiment requires kv-mode f16, q8 or q4")
    drafts = int(values["--mtp-drafts"])
    if drafts < 0:
        raise ValueError("strict experiment requires nonnegative MTP drafts")
    if drafts == 0 and "--mtp-backend" in values:
        raise ValueError("serial experiments must omit --mtp-backend; the CLI rejects explicit backends with zero drafts")
    if drafts > 0 and values.get("--mtp-backend") != "standard":
        raise ValueError("strict exact experiments with drafts require --mtp-backend standard")
    if float(values["--mtp-relaxed-margin"]) != 0:
        raise ValueError("strict exact experiments require --mtp-relaxed-margin 0")
    if float(values["--temperature"]) != 0 or int(values["--seed"]) < 0:
        raise ValueError("strict exact experiments require greedy sampling and a fixed seed")
    context = int(values["--context-tokens"])
    chunk = int(values["--prefill-chunk-size"])
    if context < 1 or not 32 <= chunk <= 8192 or chunk % 32:
        raise ValueError("strict experiment requires positive context and 32-aligned chunk in 32...8192")
    if sum(option in ("--no-thinking", "--thinking") for option in options) != 1:
        raise ValueError("strict experiment requires exactly one of --thinking or --no-thinking")
    if "--no-prefix-cache" not in options:
        raise ValueError("strict uncached experiments require --no-prefix-cache")
    if "--no-profile" not in options:
        raise ValueError("strict hermetic experiments require --no-profile")
    return values


def effective_memory(stderr):
    rows = re.findall(r"^\[memory\].*$", stderr, re.MULTILINE)
    if len(rows) != 1:
        raise ValueError("strict experiment requires exactly one effective [memory] report")
    values = dict(re.findall(r"(\w+)=([^\s]+)", rows[0]))
    if not {"kv_mode", "max_tokens"}.issubset(values):
        raise ValueError("effective [memory] report is missing kv_mode or max_tokens")
    return {"kv_mode": values["kv_mode"], "max_tokens": int(values["max_tokens"])}


def effective_config(stderr):
    rows = re.findall(r"^\[config\].*$", stderr, re.MULTILINE)
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("strict experiment requires at most one effective [config] report")
    return dict(re.findall(r"(\w+)=([^\s]+)", rows[0]))


def check_strict_pair(pair, expected):
    for arm in ("baseline", "candidate"):
        run = pair[arm]
        actual = effective_memory(run["stderr"])
        if actual["kv_mode"] != expected["--kv-mode"]:
            raise ValueError(f"{arm} effective KV mode differs from the requested experiment")
        if actual["max_tokens"] != int(expected["--context-tokens"]):
            raise ValueError(f"{arm} effective context differs from the requested experiment")
        cfg = effective_config(run["stderr"])
        if cfg is not None:
            if cfg.get("profile") != "bypassed":
                raise ValueError(f"{arm} did not bypass user/model profiles (--no-profile)")
            if cfg.get("kv_mode") != expected["--kv-mode"]:
                raise ValueError(f"{arm} effective [config] kv_mode differs from the requested experiment")
            if cfg.get("lsa_mode") != expected["--lsa-mode"]:
                raise ValueError(f"{arm} effective [config] lsa_mode differs from the requested experiment")
            if int(cfg.get("mtp_drafts", "-1")) != int(expected["--mtp-drafts"]):
                raise ValueError(f"{arm} effective [config] mtp_drafts differs from the requested experiment")
            run["effective_config"] = cfg
        if run["steady_allocations"] != 0:
            raise ValueError(f"{arm} has nonzero or missing steady-state allocation counts")
        run["effective_memory"] = actual
    for key in ("prompt_tokens", "generated_tokens", "stdout_sha256"):
        if pair["baseline"][key] != pair["candidate"][key]:
            raise ValueError(f"strict same-pack experiment differs in {key}")


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
    parser.add_argument("--strict", action="store_true",
                        help="Require pinned settings, hash model/assets, and refuse parity/config mismatches")
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
    expected = None
    if args.strict:
        try:
            expected = strict_options(options)
        except ValueError as error:
            parser.error(str(error))
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
    try:
        model_snapshot = model_identity(model, hash_files=True) if args.strict else None
    except (OSError, ValueError) as error:
        parser.error(str(error))
    identity = {"model": file_identity(model), "model_files": model_snapshot,
                "prompt_sha256": digest(prompt),
                "binaries": {name: {**file_identity(path), "sha256": digest(path)}
                             for name, path in binaries.items()},
                "metallibs": {name: {**file_identity(path), "sha256": digest(path)}
                              for name, path in metallibs.items()}}
    report = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "load_average": list(os.getloadavg()), "model_id": args.model_id,
        "identity": identity, "options": options, "max_tokens": args.max_tokens,
        "notes": args.notes, "cooldown_seconds": args.cooldown_seconds, "strict": args.strict,
        "pinned_settings": expected,
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
                    if expected is not None:
                        try:
                            check_strict_pair(pair, expected)
                        except ValueError as error:
                            report["status"] = "incomparable"
                            report["error"] = str(error)
                            write_report(output, report)
                            print(report["error"], file=sys.stderr)
                            return 1
            if (file_identity(model) != identity["model"]
                    or (args.strict and model_identity(model, hash_files=True) != identity["model_files"])
                    or digest(prompt) != identity["prompt_sha256"]):
                report["status"] = "failed"
                report["error"] = "model files, assets or prompt changed during measurement"
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
            if args.strict and (not report["token_counts_stable"] or len({
                pair[arm]["stdout_sha256"] for pair in report["samples"]
                for arm in ("baseline", "candidate")
            }) != 1):
                report["status"] = "incomparable"
                report["error"] = "strict same-pack outputs or token counts vary between samples"
                write_report(output, report)
                print(report["error"], file=sys.stderr)
                return 1
            report["status"] = "complete"
            write_report(output, report)
    except FileExistsError:
        parser.error(f"report already exists: {args.output}")
    print(f"paired report written: {args.output}")
    print("Inspect raw samples, settings and output hashes; no performance or fidelity verdict is implied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
