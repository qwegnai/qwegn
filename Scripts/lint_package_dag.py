#!/usr/bin/env python3
"""Enforce layered Package.swift target DAG, test isolation, and -warnings-as-errors."""
from pathlib import Path
import re, sys

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {
    "QWCore": 0.0, "QWMetalCounterSupport": 0.0,
    "QWSMCHelper": 0.5,
    "QWHAL": 1.0, "QWQuant": 1.0, "QWToken": 1.0,
    "QWDecision": 1.5,
    "QWAlloc": 2.0, "QWLoad": 2.5, "QWRope": 2.5,
    "QWGEMM": 3.0, "QWKV": 3.0,
    "QWNorm": 4.0, "QWAttn": 4.0, "QWGDN": 4.0, "QWMLP": 4.0, "QWHyperPLE": 4.0, "QWVision": 4.0,
    "QWHead": 5.0, "QWMTP": 6.0, "QWMM": 6.0,
    "QWEngine": 7.0, "QWCli": 7.5, "QWServer": 7.8,
    "qwg": 8.0, "qwg-smc-helper": 8.0, "qwg-bench-kernels": 8.0, "golden_pre_p3": 8.0,
    "m1": 8.0, "QWMenuBarUI": 8.5, "qwg-menu": 9.0,
}
TEST_MAX_LAYER = {
    "QWCoreTests": 0.0, "QWTokenTests": 1.0, "QWDecisionTests": 1.5,
    "QWLoadTests": 2.5, "QWEngineTests": 7.0,
}
BANNED_TARGETS = {"QWState"}


def extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == open_ch:
            depth += 1
        elif text[idx] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1:idx], idx
    return "", start


def parse_package_swift(text: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    prod: dict[str, list[str]] = {}
    for m in re.finditer(r'\bT\(\s*"([^"]+)"(?:\s*,\s*\[([^\]]*)\])?\s*\)', text, re.S):
        deps = re.findall(r'"([^"]+)"', m.group(2) or "")
        prod[m.group(1)] = deps
    for kind in ("target", "executableTarget"):
        for m in re.finditer(rf'\.{kind}\s*\(', text):
            body, _ = extract_balanced(text, m.end() - 1, "(", ")")
            nm = re.search(r'name\s*:\s*"([^"]+)"', body)
            if not nm:
                continue
            name = nm.group(1)
            dm = re.search(r'dependencies\s*:\s*\[', body)
            deps: list[str] = []
            if dm:
                dbody, _ = extract_balanced(body, dm.end() - 1, "[", "]")
                deps = re.findall(r'(?:\.byName\(\s*name:\s*|" )?"([A-Za-z0-9_-]+)"', dbody)
            prod[name] = [d for d in deps if d in LAYERS or d in BANNED_TARGETS]
    tests: dict[str, list[str]] = {}
    for mod, test in re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', text):
        tdeps = [mod]
        for cond, blk in re.findall(r'if\s+([^{]+)\{([^}]+)\}', text):
            if re.search(rf'\bmod\s*==\s*"{re.escape(mod)}"', cond):
                tdeps += re.findall(r'\.byName\(\s*name:\s*"([^"]+)"\s*\)', blk)
        tests[test] = tdeps
    for m in re.finditer(r'\.testTarget\s*\(', text):
        body, _ = extract_balanced(text, m.end() - 1, "(", ")")
        nm = re.search(r'name\s*:\s*"([^"]+)"', body)
        dm = re.search(r'dependencies\s*:\s*\[', body)
        if nm and dm and not nm.group(1).startswith("\\"):
            dbody, _ = extract_balanced(body, dm.end() - 1, "[", "]")
            tests[nm.group(1)] = re.findall(r'"([^"]+)"', dbody)
    return prod, tests


def main() -> int:
    text = (ROOT / "Package.swift").read_text(encoding="utf-8")
    problems: list[str] = []
    if "-warnings-as-errors" not in text:
        problems.append("Package.swift: missing -warnings-as-errors enforcement")
    prod, tests = parse_package_swift(text)
    for bad in sorted(BANNED_TARGETS):
        if bad in prod or re.search(rf'"{bad}"', text):
            problems.append(f"Package.swift: banned production target reference {bad!r}")
    for tgt, deps in sorted(prod.items()):
        if tgt not in LAYERS:
            problems.append(f"Package.swift: unclassified target {tgt!r} in lint_package_dag.py")
            continue
        for dep in deps:
            if dep not in LAYERS:
                problems.append(f"{tgt} -> {dep}: unknown internal target")
            elif LAYERS[dep] >= LAYERS[tgt]:
                problems.append(f"DAG violation: {tgt}(L{LAYERS[tgt]}) -> {dep}(L{LAYERS[dep]})")
    for test, cap in sorted(TEST_MAX_LAYER.items()):
        for dep in tests.get(test, []):
            if dep in LAYERS and LAYERS[dep] > cap:
                problems.append(f"Test isolation violation: {test}(<=L{cap}) -> {dep}(L{LAYERS[dep]})")
    if problems:
        for p in problems:
            print(p)
        print(f"package_dag: {len(problems)} violation(s)")
        return 1
    print(f"package_dag: 0 violation(s) ({len(prod)} targets checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
