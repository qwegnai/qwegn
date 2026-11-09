"""Check the kernel bench's winner map: its schema, and that it has entries.

The map's schema must be the one this build writes **and** the one the pin
names. The pin is what makes a schema change a deliberate act: the report,
the engine's own constant and the pin are three separate things, and if a
change moves only one of them the gate goes red. Comparing the report
against the engine's constant alone would be a tautology — the bench
writes that same constant — so it would never catch a change.

    python3 Scripts/check_bench_report.py <report.json>
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "Sources" / "Foundation" / "QWCore" / "QWKernelDispatchTable.swift"
PIN = ROOT / "Scripts" / "bench_kernel_schema.json"


def engine_schema() -> str:
    """The schema the engine's own constant names."""
    found = re.findall(r'currentSchema\s*=\s*"([^"]+)"', ENGINE.read_text())
    assert len(found) == 1, f"the engine's schema constant is not unique: {found}"
    return found[0]


def pinned_schema() -> str:
    """The schema the pin names."""
    return json.loads(PIN.read_text())["schema"]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <bench report.json>", file=sys.stderr)
        return 2
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    engine = engine_schema()
    pinned = pinned_schema()
    problems = []
    if report["schema"] != engine:
        problems.append(f"the map is not this build's: {report['schema']} vs {engine}")
    if report["schema"] != pinned:
        problems.append(f"the pin is stale: {report['schema']} vs {pinned}")
    if len(report["entries"]) == 0:
        problems.append("the winner map is empty")
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print(f"bench: schema {report['schema']}, {len(report['entries'])} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
