#!/usr/bin/env python3
"""Convert the m1 harness's export into the m1_soak_*.jsonl SoakSample
artifact that `qwg bench m1 --check` verifies.

The schema is owned by QWCli.SoakSample (Sources/Apps/QWCli/SoakVerify.swift).
This script moves the measured windows from the export into the artifact and
checks that every window carries the measurements the gate reads. It never
invents a value: a window that does not carry what the gate reads is an error.

The export is the one the m1 harness writes (`m1 --export`). The CLI's
`qwg telemetry --jsonl` export carries the server's metric and request
lines, not soak windows; fed one, this script refuses it rather than
inventing a window.
"""
import argparse
import json
import os
import sys
import time

WINDOW_KIND = "window"
REQUIRED = ("steps", "windowTps", "windowP50Ms", "windowP95Ms",
            "steadyAllocs", "rssKb", "fds")


def read_export(path):
    """Yields (line number, object) for every non-empty line of the export."""
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"gen_soak_sample: cannot read the export: {error}")
    with handle:
        for number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield number, json.loads(raw)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"gen_soak_sample: {path}:{number}: malformed JSON line: {error}")


def is_window(obj):
    """True when the line carries a measured soak window."""
    kind = obj.get("kind")
    if kind is not None:
        return kind == WINDOW_KIND
    return "steps" in obj and "windowTps" in obj


def collect(path):
    """The measured windows of the export, in the order they were written."""
    windows = []
    for number, obj in read_export(path):
        if not isinstance(obj, dict):
            raise SystemExit(
                f"gen_soak_sample: {path}:{number}: every export line is a JSON object")
        if not is_window(obj):
            continue
        missing = [key for key in REQUIRED if obj.get(key) is None]
        if missing:
            raise SystemExit(
                f"gen_soak_sample: {path}:{number}: the window does not carry "
                f"{', '.join(missing)} — a value the gate reads is absent")
        windows.append(obj)
    return windows


def write_artifact(windows, out_path):
    """Writes the windows as the artifact: one JSON object per line."""
    parent = os.path.dirname(os.path.abspath(out_path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"gen_soak_sample: cannot create {parent}: {error}")
    with open(out_path, "w", encoding="utf-8") as handle:
        for window in windows:
            handle.write(json.dumps(window, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Turn the m1 harness's export into the m1_soak_*.jsonl artifact.")
    parser.add_argument("export", help="the m1 harness export to convert (m1 --export)")
    parser.add_argument("--out-dir", default=".", help="where the artifact goes (default: .)")
    parser.add_argument("--out", default=None, help="the artifact path (default: <out-dir>/m1_soak_<stamp>.jsonl)")
    parser.add_argument("--stamp", default=None, help="the stamp in the artifact name (default: now)")
    args = parser.parse_args(argv)

    windows = collect(args.export)
    if not windows:
        raise SystemExit(
            f"gen_soak_sample: {args.export} carries no measured windows — "
            "an artifact with no measurement would read as a passing soak")

    stamp = args.stamp if args.stamp is not None else str(int(time.time()))
    out_path = args.out or os.path.join(args.out_dir, f"m1_soak_{stamp}.jsonl")
    write_artifact(windows, out_path)
    source = windows[0].get("source") or "engine"
    label = windows[0].get("label") or ""
    print(f"gen_soak_sample: {len(windows)} window(s) from {args.export} → {out_path} "
          f"(source: {source}{'; ' + label if label else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
