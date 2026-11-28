#!/usr/bin/env python3
r"""Kernel reachability gate: the built metallib vs the code that dispatches it.

Direction 1: every entry point that ships in Sources/QWMetal/kernel.metallib
must be named by production Swift (Sources/**.swift, excluding QWMetal's own
string-literal loaders) or by a bench/test harness that compiles it on purpose.
Direction 2: every kernel name Swift asks for must exist in the metallib.
Direction 3: every kernel a module's engine dispatches must be in its
manifest-backed prewarm inventory.

Run Scripts/build_metallib.sh first. Entry points come from the metallib, not
from parsing .metal, so macro-generated families (qwm_mlp_gate_up_m\(count))
are judged on what actually ships.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import gen_kernel_inventory

ROOT = Path(__file__).resolve().parent.parent
METALLIB = ROOT / "Sources" / "QWMetal" / "kernel.metallib"
PREFIXES = ("qwg_", "qwfn_", "qwm_", "qws_", "qwc_", "qwh_", "qwr_")
# Public MCP tool names that share the qwg_ prefix are not Metal kernels.
NON_KERNEL_LITERALS = {
    "qwg_decide",
    "qwg_ask",
    "qwg_guard",
    "qwg_status",
    "qwg_rerank_screen",
    "qwg_compact_select",
    "qwg_episodic_recover",
    "qwg_skill_select",
    "qwg_search_enough",
    "qwg_supervise",
    "qwg_route_turn",
    "qwg_vault_mount",
    "qwg_vault_query",
    "qwg_vault_status",
}
# Names that are library objects rather than dispatch targets.
AIR_NOISE = re.compile(r"(llvm\.|air\.|_Z|MTL_|__tensorops|metal18|cooperative)")


def metallib_names(path: Path | None = None) -> set[str]:
    lib = path or METALLIB
    if not lib.exists():
        sys.exit(f"missing {lib.relative_to(ROOT)} — run Scripts/build_metallib.sh")
    out = subprocess.run(["strings", "-a", str(lib)],
                         capture_output=True, text=True, check=True).stdout
    found = set()
    for line in out.splitlines():
        for tok in re.split(r"[^A-Za-z0-9_]+", line):
            if tok.startswith(PREFIXES) and len(tok) > 6 and not AIR_NOISE.search(tok):
                found.add(tok)
    return found


def basis_problems(rot: Path | None = None, std: Path | None = None) -> list[str]:
    """The rotated and standard libraries must each carry their OWN basis marker."""
    problems: list[str] = []
    rot = rot or METALLIB
    std = std or (ROOT / "Sources" / "QWMetal" / "kernel_std.metallib")
    for lib in (rot, std):
        if not lib.exists():
            problems.append(f"MISSING METALLIB: {lib.relative_to(ROOT)} — run Scripts/build_metallib.sh")
    if problems:
        return problems
    rotated = metallib_names(rot)
    standard = metallib_names(std)
    if "qwg_basis_marker" not in rotated:
        problems.append(f"BASIS MARKER: {rot.name} does not carry "
                        "qwg_basis_marker — the HAL cannot tell which basis it holds")
    if "qwg_basis_marker_standard" in rotated:
        problems.append(f"BASIS MARKER: {rot.name} carries the "
                        "STANDARD marker — it is the wrong build")
    if "qwg_basis_marker_standard" not in standard:
        problems.append(f"BASIS MARKER: {std.name} does not carry "
                        "qwg_basis_marker_standard — it is the wrong build")
    if "qwg_basis_marker" in standard:
        problems.append(f"BASIS MARKER: {std.name} carries the "
                        "ROTATED marker — it is the wrong build")
    markers = {"qwg_basis_marker", "qwg_basis_marker_standard"}
    only_rotated = sorted((rotated - standard) - markers)
    only_standard = sorted((standard - rotated) - markers)
    if only_rotated:
        problems.append(f"BASIS SYMBOLS ({rot.name}): {len(only_rotated)} kernel(s) only in rotated: "
                        f"{only_rotated[:3]}")
    if only_standard:
        problems.append(f"BASIS SYMBOLS ({std.name}): {len(only_standard)} kernel(s) only in standard: "
                        f"{only_standard[:3]}")
    return problems


def m3_problems(full_names: set[str], nax_names: set[str]) -> list[str]:
    """Verify kernel_m3.metallib + kernel_m3_std.metallib markers and non-NAX parity."""
    m3_rot = ROOT / "Sources" / "QWMetal" / "kernel_m3.metallib"
    m3_std = ROOT / "Sources" / "QWMetal" / "kernel_m3_std.metallib"
    problems = basis_problems(m3_rot, m3_std)
    if not m3_rot.exists():
        return problems
    m3_names = metallib_names(m3_rot) - declared_helpers()
    expected_non_nax = full_names - nax_names
    missing = sorted(expected_non_nax - m3_names)
    if missing:
        problems.append(
            f"M3 MISSING NON-NAX KERNELS: {len(missing)} non-NAX kernel(s) absent from {m3_rot.name}: {missing[:5]}")
    return problems


def declared_helpers() -> set[str]:
    """Symbols declared in Metal sources: helpers/structs, never dispatch targets."""
    decl = re.compile(r"(?:static\s+inline|METAL_FUNC|inline|struct|typedef|using)\s+"
                      r"[^\s;]*?\b([a-z][A-Za-z0-9_]*)\s*[({=]")
    names = set()
    for p in (ROOT / "Sources" / "QWMetal").rglob("*"):
        if p.suffix in {".metal", ".metalh"}:
            names.update(decl.findall(p.read_text(encoding="utf-8", errors="ignore")))
    return names


def bench_local_kernels() -> set[str]:
    """Kernels a bench compiles itself — in a bench-local .metal file or in a
    Metal source string inside the bench's Swift. Both are compiled at bench
    time, so they must NOT be expected inside the release metallib."""
    decl = re.compile(r"kernel\s+void\s+([A-Za-z0-9_]+)")
    names = set()
    for base in ("Scripts", "Tests"):
        for p in (ROOT / base).rglob("*.metal"):
            names.update(decl.findall(p.read_text(encoding="utf-8", errors="ignore")))
        for p in (ROOT / base).rglob("*.swift"):
            names.update(decl.findall(p.read_text(encoding="utf-8", errors="ignore")))
    return names


def strip_comments(text: str) -> str:
    """Remove block and line comments so prose never counts as a dispatch."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)//.*$", "", text)


def prod_swift_refs() -> tuple[dict[str, list[str]], set[str]]:
    """(kernel name -> files) for every kernel production Swift asks for, plus
    the interpolated prefixes production Swift can build at runtime."""
    refs: dict[str, list[str]] = {}
    dynamic: set[str] = set()
    lit = re.compile(r"[\"'](q[a-z0-9_]{6,})[\"']")
    interp = re.compile(r"[\"'](q[a-z0-9_]+_[a-z0-9_]*?)\\?\(")
    for path in (ROOT / "Sources").rglob("*.swift"):
        if "QWMetal" in path.parts or "QWBenchKernels" in path.parts or path.name in ("QWKernelInventory.swift", "QWNaxKernelRegistry.swift"):
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        for name in lit.findall(text):
            if name in NON_KERNEL_LITERALS:
                continue
            if name.startswith(PREFIXES):
                refs.setdefault(name, []).append(str(path.relative_to(ROOT)))
        for pre in interp.findall(text):
            dynamic.add(pre)
    return refs, dynamic


def swift_refs() -> tuple[dict[str, list[str]], set[str]]:
    """(kernel name -> files) for every kernel string Swift asks for, plus the
    interpolated prefixes it can build at runtime."""
    refs, dynamic = prod_swift_refs()
    lit = re.compile(r"[\"'](q[a-z0-9_]{6,})[\"']")
    # benches and tests may load bench-local kernels on purpose
    for base in ("Scripts", "Tests"):
        for path in (ROOT / base).rglob("*.swift"):
            text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
            for name in lit.findall(text):
                if name in NON_KERNEL_LITERALS:
                    continue
                if name.startswith(PREFIXES):
                    refs.setdefault(name, []).append(str(path.relative_to(ROOT)))
    return refs, dynamic


def pipeline_literals() -> dict[str, list[str]]:
    """Every kernel name Swift hands `pipeline(...)`, whatever it starts with.

    The `q`-prefixed sweep above cannot see a MISSPELLED name: a typo that
    drops the `q` (or any other letter) is invisible to it, and the mistake
    then surfaces as a runtime "kernel missing from metallib". These are the
    literal call sites, interpolated names excluded — those are built at
    dispatch time and the `dynamic` prefixes already cover them.

    Tests and benches are scanned too: a typo in a test's `pipeline("...")`
    is the same mistake, and it fails just as late."""
    out: dict[str, list[str]] = {}
    call = re.compile(r'pipeline\(\s*"([^"\\]+)"')
    for base in ("Sources", "Tests", "Scripts"):
        for path in (ROOT / base).rglob("*.swift"):
            if "QWMetal" in path.parts:
                continue
            for name in call.findall(path.read_text(encoding="utf-8", errors="ignore")):
                out.setdefault(name, []).append(str(path.relative_to(ROOT)))
    return out


def pipeline_literal_exemptions() -> set[str]:
    """Names a test deliberately asks for that must NOT exist.

    A test that asserts a missing kernel throws names a kernel that is
    absent on purpose. The `pipeline(...)` sweep sees the name and would
    report it as a typo, so those names are exempt — the test says so."""
    out: set[str] = set()
    pat = re.compile(r'XCTAssertThrowsError\([^)]*pipeline\(\s*"([^"\\]+)"', re.S)
    for base in ("Tests", "Scripts"):
        for path in (ROOT / base).rglob("*.swift"):
            out.update(pat.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return out


MODULE_DIR = ROOT / "Sources" / "Runtime" / "QWEngine" / "Module"
GEMM_DIR = ROOT / "Sources" / "Operators" / "QWGEMM"
CHAIN_FILE = MODULE_DIR / "QWMoELayerKernel.swift"


def balanced_body(text: str, start: int) -> str:
    """The text between the brace at `start` and its match."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return ""


def chain_kernels() -> set[str]:
    """The kernels the standard path's sparse chain dispatches.

    The chain names its encoders; each encoder names its pipeline. One
    level of resolution is enough, because the encoders are the chain's
    own boundary."""
    text = CHAIN_FILE.read_text(encoding="utf-8", errors="ignore")
    calls = set(re.findall(r"\bQWGemm\.(encode\w+)\(", text))
    out: set[str] = set()
    for path in sorted(GEMM_DIR.glob("*.swift")):
        body = path.read_text(encoding="utf-8", errors="ignore")
        for call in calls:
            m = re.search(r"func\s+" + re.escape(call) + r"\s*\(", body)
            if not m:
                continue
            brace = body.find("{", m.end())
            out.update(re.findall(r'pipeline\(\s*"([^"]+)"', balanced_body(body, brace)))
    return out


def dispatched_in(dirs: list[str], skip: tuple[str, ...] = ()) -> set[str]:
    """Kernel names the Swift under `dirs` asks for."""
    lit = re.compile(r"[\"'](q[a-z0-9_]{6,})[\"']")
    out: set[str] = set()
    for d in dirs:
        for path in (ROOT / d).rglob("*.swift"):
            if any(s in path.parts for s in skip) or path.name == "QWKernelInventory.swift":
                continue
            text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
            for name in lit.findall(text):
                if name.startswith(PREFIXES):
                    out.add(name)
    return out


ENGINE_DIRS = ["Sources/Runtime/QWEngine"]
OPERATOR_DIRS = ["Sources/Operators/QWAttn", "Sources/Operators/QWGDN",
                 "Sources/Operators/QWGEMM", "Sources/Operators/QWHead",
                 "Sources/Operators/QWKV", "Sources/Operators/QWMLP",
                 "Sources/Subsystems/QWMTP", "Sources/Operators/QWNorm",
                 "Sources/Operators/QWRope", "Sources/Operators/QWHyperPLE",
                 "Sources/Foundation/QWQuant", "Sources/Subsystems/QWVision"]


def door_problems(available: set[str], dynamic: set[str] | None = None) -> list[str]:
    """Doors the unified engine dispatches but DenseModule does not declare."""
    problems: list[str] = []
    if dynamic is None:
        _, dynamic = swift_refs()
    try:
        inventory = gen_kernel_inventory.load_manifest()
    except (OSError, ValueError) as error:
        return [f"INVALID KERNEL INVENTORY: {error}"]
    generated = gen_kernel_inventory.render(inventory)
    if not gen_kernel_inventory.OUTPUT.exists() or gen_kernel_inventory.OUTPUT.read_text(encoding="utf-8") != generated:
        problems.append("STALE KERNEL INVENTORY: run python3 Scripts/gen_kernel_inventory.py")
    all_doors = (set(inventory["common"])
                 | set(inventory["dense_only"])
                 | set(inventory["hyper_only"])
                 | set(inventory["inkling2_only"]))
    for missing in sorted(all_doors - available):
        problems.append(f"MISSING INVENTORY KERNEL: {missing!r} is not in the metallib")
    dispatched = (dispatched_in(ENGINE_DIRS, ("Module", "Core"))
                  | dispatched_in(OPERATOR_DIRS)
                  | chain_kernels())
    for missing in sorted(dispatched - all_doors):
        problems.append(f"UNDECLARED DOOR: DenseModule dispatches {missing!r} "
                        f"but does not declare it")
    all_dispatched = dispatched | dispatched_in(["Sources/Foundation/QWLoad"])
    for extra in sorted(all_doors - all_dispatched):
        if any(extra.startswith(d) for d in dynamic):
            continue
        problems.append(f"STALE INVENTORY DOOR: inventory declares {extra!r} "
                        f"but no engine or operator dispatches it")
    return problems


def nax_registry_problems(available: set[str]) -> list[str]:
    """QWNaxKernelRegistry must match nax_kernels.json and ship in full metallib."""
    problems: list[str] = []
    try:
        nax_names = gen_kernel_inventory.load_nax_manifest()
    except (OSError, ValueError) as error:
        return [f"INVALID NAX MANIFEST: {error}"]
    rendered = gen_kernel_inventory.render_nax_registry(nax_names)
    out_path = gen_kernel_inventory.NAX_OUTPUT
    if not out_path.exists() or out_path.read_text(encoding="utf-8") != rendered:
        problems.append("STALE NAX REGISTRY: run python3 Scripts/gen_kernel_inventory.py")
    for missing in sorted(set(nax_names) - available):
        problems.append(f"MISSING NAX KERNEL: {missing!r} in nax_kernels.json is not in the metallib")
    return problems


def main() -> int:
    names = metallib_names() - declared_helpers()
    prod_refs, prod_dynamic = prod_swift_refs()
    refs, dynamic = swift_refs()
    local = bench_local_kernels()
    problems = []

    for name in sorted(names):
        if name in prod_refs:
            continue
        if any(name.startswith(d) for d in prod_dynamic):
            continue          # built by string interpolation at dispatch time
        problems.append(f"UNREACHABLE KERNEL: {name} ships in the metallib but "
                        f"no production Swift dispatches it")

    for name, files in sorted(refs.items()):
        if name in local:
            continue          # bench-local source, compiled by the bench itself
        if name in names or any(name.startswith(n) for n in names):
            continue
        if any(name.startswith(d) for d in dynamic):
            continue
        problems.append(f"MISSING KERNEL: {files[0]} asks for {name!r}, "
                        f"which is not in the metallib")

    exempt = pipeline_literal_exemptions()
    for name, files in sorted(pipeline_literals().items()):
        if name in local or name in exempt:
            continue
        # A literal at a `pipeline(...)` call site is a COMPLETE kernel name,
        # never an interpolated prefix, so the match must be exact: a name
        # that merely EXTENDS a real kernel is a typo, not a dispatch.
        if name in names:
            continue
        problems.append(f"MISSING KERNEL: {files[0]} asks for {name!r}, "
                        f"which is not in the metallib")

    problems.extend(door_problems(names, dynamic))
    problems.extend(nax_registry_problems(names))
    problems.extend(basis_problems())
    try:
        nax_set = set(gen_kernel_inventory.load_nax_manifest())
    except (OSError, ValueError):
        nax_set = set()
    problems.extend(m3_problems(names, nax_set))

    for line in problems:
        print(line)
    print(f"kernels: {len(names)} in metallib, "
          f"{len(refs)} names dispatched from Swift, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
