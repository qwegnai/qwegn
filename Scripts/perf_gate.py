#!/usr/bin/env python3
"""Performance and parity gate for the engine.

Checks, in order (one GPU process at a time — the repo law):

  1. parity  — the 27B golden tokens and the Flash-Next generated text must
                be byte-identical to the recorded baseline;
  2. perf    — prefill tok/s, decode tok/s and steady-state allocations must
                not regress past the tolerance below the recorded baseline.

Sampling
--------
Every scenario is run `SAMPLES` times and **both** the median and the peak
of the samples are compared against the baseline's. A regression is only
reported when the median *and* the peak are both below the floor.

One sample is not enough, and the reference machine is shared, so a single
sample can be anything: the same binary measured 351-394 tok/s on the
dense prefill and 883-1009 tok/s on the 4K prefill, and one sweep of the
Flash-Next 4K decode read 95.1, 83.8, 84.6 — two different modes in three
samples. A real regression slows *every* sample, so both the median and
the peak fall; contention or a mode shift moves one of them and leaves the
other standing. Requiring both keeps a noisy neighbour from failing the
build, and keeps a real regression from hiding behind one lucky sample.

Usage:
    python3 Scripts/perf_gate.py --save     # record the current results
    python3 Scripts/perf_gate.py            # check against the baseline
    python3 Scripts/perf_gate.py --update-golden   # rewrite the committed golden

The baseline is machine-local and therefore not in the repo:
    $QWG_PERF_BASELINE   (default: .probe/perf_baseline.json)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, ".build", "release", "qwg")
GOLDEN = os.path.join(ROOT, ".build", "release", "golden_pre_p3")
BASELINE = os.environ.get("QWG_PERF_BASELINE",
                          os.path.join(ROOT, ".probe", "perf_baseline.json"))
# The committed byte-parity anchor: the dense engine's greedy tokens and its
# final logit row. It is regenerated only on purpose (--update-golden).
FIXTURE = os.path.join(ROOT, "Tests", "QWEngineTests", "Fixtures", "golden_pre_p3.bin")

DENSE = "Qwen3.8-27B-hq4g32"
FLASH = "Qwen3.8-Flash-Next-hq4g32"
PACKS = {"dense": os.path.expanduser("~/.qwg/models/%s/model.qwp" % DENSE),
         "flash": os.path.expanduser("~/.qwg/models/%s/model.qwp" % FLASH)}

PROMPT = ("Write a Swift struct Matrix with rows, cols, data and a function that "
          "multiplies two matrices. Explain the time complexity.")

# Tolerances: the gate catches a regression, not a noisy sample.
PREFILL_MIN_RATIO = 0.95
DECODE_MIN_RATIO = 0.95

# Samples per scenario, and the settle time between them.
#
# Five, not three: this machine is shared, and a single scenario's samples
# swing by 10-15% between runs (the same Flash-Next code measured 12% apart
# in a back-to-back pair). Three samples make the median a coin flip. The
# rule below still needs the median AND the peak to fall before it calls a
# regression, so more samples make it stricter, not laxer.
SAMPLES = 5
SETTLE_SECONDS = 2.0


def run(cmd, timeout=1800):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def metrics(stderr):
    prefill = decode = None
    allocs = None
    m = re.search(r"prefill[^:]*:\s+[\d.]+ s\s+\(([\d.]+) tok/s", stderr)
    if m:
        prefill = float(m.group(1))
    m = re.search(r"decode:\s+[\d.]+ s\s+\(([\d.]+) tok/s\)", stderr)
    if m:
        decode = float(m.group(1))
    m = re.search(r"steady-state allocs:\s+(\d+)", stderr)
    if m:
        allocs = int(m.group(1))
    return prefill, decode, allocs


def gen(model, tokens, extra=(), prompt=None):
    cmd = [BIN, "run", "--model", model, "--prompt", prompt or PROMPT,
           "--max-tokens", str(tokens), "--temperature", "0",
           "--no-thinking", "--no-stream", "--prefill-chunk-size", "512",
           "--verbose"] + list(extra)
    started = time.time()
    rc, out, err = run(cmd)
    if rc != 0:
        raise SystemExit("run failed (%d) for %s:\n%s" % (rc, model, err[-2000:]))
    prefill, decode, allocs = metrics(err)
    m = re.search(r"prompt tokens:\s+(\d+)", err)
    return {"text": out, "prefill": prefill, "decode": decode, "allocs": allocs,
            "prompt_tokens": int(m.group(1)) if m else None,
            "seconds": round(time.time() - started, 1)}


def median(values):
    """The middle value of a sample set — the one that survives noise."""
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def measure(model, tokens, extra=(), prompt=None, samples=SAMPLES):
    """Run a scenario `samples` times and report the median of each metric."""
    runs = []
    for index in range(samples):
        runs.append(gen(model, tokens, extra, prompt))
        if index + 1 < samples:
            time.sleep(SETTLE_SECONDS)
    prefill_samples = [r["prefill"] for r in runs]
    decode_samples = [r["decode"] for r in runs]
    allocs_samples = [r["allocs"] for r in runs]
    return {"text": runs[0]["text"],
            "prefill": median(prefill_samples),
            "decode": median(decode_samples),
            "prefill_peak": max(prefill_samples),
            "decode_peak": max(decode_samples),
            "allocs": max(a for a in allocs_samples if a is not None),
            "prompt_tokens": runs[0]["prompt_tokens"],
            "prefill_samples": prefill_samples,
            "decode_samples": decode_samples}


def baseline_value(entry, metric):
    """The baseline number to compare against: its median, when it has samples."""
    samples = entry.get("%s_samples" % metric)
    if samples:
        return median(samples)
    return entry.get(metric)


def baseline_peak(entry, metric):
    """The baseline peak: the best sample it ever saw."""
    samples = entry.get("%s_samples" % metric)
    if samples:
        return max(samples)
    return entry.get(metric)


def long_prompt(repeats):
    """A prompt long enough to measure prefill throughput, not launch latency."""
    return " ".join([PROMPT] * repeats)


def golden(model):
    cmd = [GOLDEN, PACKS[model]]
    rc, out, err = run(cmd)
    if rc != 0:
        raise SystemExit("golden capture failed for %s:\n%s" % (model, err[-2000:]))
    return out, err


def main():
    save = "--save" in sys.argv
    for path in (BIN, GOLDEN):
        if not os.path.exists(path):
            raise SystemExit("missing %s — run `make build` first" % path)
    for name, pack in PACKS.items():
        if not os.path.exists(pack):
            raise SystemExit("missing pack %s" % pack)

    results = {}

    # 1. parity — the 27B golden tokens against the committed fixture, and
    # the same prompt through the unified core loop (the module seam).
    if "--update-golden" in sys.argv:
        rc, out, err = run([GOLDEN, PACKS["dense"], "--fixture", FIXTURE])
        if rc != 0:
            raise SystemExit("golden capture failed:\n%s" % err[-2000:])
        print("golden fixture rewritten -> %s" % FIXTURE)
        print("the golden is the byte-parity anchor: commit it only when the change is intended")
        return
    rc, out, err = run([GOLDEN, PACKS["dense"], "--fixture", FIXTURE, "--check", "--module"])
    ok = rc == 0 and "ok tokens" in out
    module_ok = "ok module tokens" in out
    results["golden_tokens"] = ok and module_ok
    print("parity  27B golden tokens: %s" % ("OK" if ok else "MISMATCH\n" + out + err))
    print("parity  27B through the module seam: %s"
          % ("OK" if module_ok else "MISMATCH\n" + out + err))
    if not ok or not module_ok:
        raise SystemExit(1)

    # 2. parity — Flash-Next generated text.
    flash = gen(FLASH, 64)
    results["flash_text"] = flash["text"]

    # 3. perf.
    dense = measure(DENSE, 256, ["--mtp-drafts", "0"])
    dense_mtp = measure(DENSE, 256, ["--mtp-drafts", "3"])
    flash_perf = measure(FLASH, 64)
    results["dense_serial"] = dense
    results["dense_mtp3"] = dense_mtp
    results["flash_perf"] = flash_perf

    # 4. prefill throughput at ~4K tokens, comparable to the published floor.
    long_text = long_prompt(120)
    results["dense_prefill_4k"] = measure(DENSE, 16, ["--mtp-drafts", "3"], prompt=long_text)
    results["flash_prefill_4k"] = measure(FLASH, 16, prompt=long_text)

    if save:
        baseline = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "samples": SAMPLES,
                    "load_average": [round(x, 2) for x in os.getloadavg()],
                    "results": results}
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as fh:
            json.dump(baseline, fh, indent=2)
        print("baseline written -> %s" % BASELINE)
        for key in ("dense_serial", "dense_mtp3", "flash_perf",
                    "dense_prefill_4k", "flash_prefill_4k"):
            r = results[key]
            print("  %-18s prefill %s tok/s  decode %s tok/s  allocs %s  prompt %s tok"
                  % (key, r["prefill"], r["decode"], r["allocs"], r.get("prompt_tokens")))
        return

    if not os.path.exists(BASELINE):
        raise SystemExit("no baseline — run `perf_gate.py --save` first")
    with open(BASELINE) as fh:
        base = json.load(fh)["results"]

    failures = []
    if base["flash_text"] != results["flash_text"]:
        failures.append("Flash-Next output text changed")

    for key, prefill_key, decode_key in (
        ("dense_serial", "prefill", "decode"),
        ("dense_mtp3", "prefill", "decode"),
        ("flash_perf", "prefill", "decode"),
        ("dense_prefill_4k", "prefill", "decode"),
        ("flash_prefill_4k", "prefill", "decode"),
    ):
        now, before = results[key], base[key]
        for metric, floor in (("prefill", PREFILL_MIN_RATIO), ("decode", DECODE_MIN_RATIO)):
            a, b = now.get(metric), baseline_value(before, metric)
            peak = now.get("%s_peak" % metric)
            base_peak = baseline_peak(before, metric)
            if a is None or b is None:
                failures.append("%s.%s: not measured" % (key, metric))
                continue
            # A regression slows every sample: the median and the peak both
            # fall. One of them alone can be a neighbour on the machine.
            if a < b * floor and peak is not None and base_peak is not None \
                    and peak < base_peak * floor:
                failures.append("%s.%s regressed: %.1f < %.1f (%.1f%% of %.1f), peak %.1f < %.1f"
                                % (key, metric, a, b, 100.0 * a / b, b, peak, base_peak))
        if now.get("allocs") not in (0, None):
            failures.append("%s: steady-state allocations = %s (must be 0)"
                            % (key, now.get("allocs")))
        print("%-18s prefill %8s tok/s (base %s, peak %s)   decode %7s tok/s (base %s, peak %s)   allocs %s"
              % (key, now.get("prefill"), baseline_value(before, "prefill"),
                 baseline_peak(before, "prefill"),
                 now.get("decode"), baseline_value(before, "decode"),
                 baseline_peak(before, "decode"), now.get("allocs")))
        print("%-18s samples prefill %s  decode %s"
              % ("", now.get("prefill_samples"), now.get("decode_samples")))

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  - %s" % f)
        raise SystemExit(1)
    print("\nGATE PASSED: parity byte-identical, no perf regression, zero allocations.")


if __name__ == "__main__":
    main()
