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


def basis_problems() -> list[str]:
    """The two libraries must each carry the marker of their OWN basis.

    The basis is a build-time property (specs/02-qwhal.md): there are two
    libraries, one per basis, and the standard one is compiled with
    `-DQW_BASIS_STANDARD`. If that flag stopped reaching the compiler, the
    standard library would be a second copy of the rotated one and a
    standard-basis model would run with rotated activations — wrong numbers
    with no error anywhere. The markers are what the HAL reads to refuse
    that, so the gate checks they are actually there and actually apart.
    """
    problems: list[str] = []
    std = ROOT / "Sources" / "QWMetal" / "kernel_std.metallib"
    if not std.exists():
        return [f"MISSING STANDARD METALLIB: {std.relative_to(ROOT)} "
                f"— run Scripts/build_metallib.sh"]
    rotated = metallib_names()
    standard = metallib_names(std)
    if "qwg_basis_marker" not in rotated:
        problems.append("BASIS MARKER: the rotated metallib does not carry "
                        "qwg_basis_marker — the HAL cannot tell which basis it holds")
    if "qwg_basis_marker_standard" in rotated:
        problems.append("BASIS MARKER: the rotated metallib carries the "
                        "STANDARD marker — it is the wrong build")
    if "qwg_basis_marker_standard" not in standard:
        problems.append("BASIS MARKER: the standard metallib does not carry "
                        "qwg_basis_marker_standard — it is the wrong build")
    if "qwg_basis_marker" in standard:
        problems.append("BASIS MARKER: the standard metallib carries the "
                        "ROTATED marker — it is the wrong build")
    # The two libraries are two builds of the SAME sources, so they must
    # offer the same kernels. A name only one of them has would be a kernel
    # that exists in one basis and not the other — a dispatch that works for
    # one model and refuses the other for no reason the caller can see. The
    # two markers are the one name that must differ: they are how the
    # libraries tell each other apart.
    markers = {"qwg_basis_marker", "qwg_basis_marker_standard"}
    only_rotated = sorted((rotated - standard) - markers)
    only_standard = sorted((standard - rotated) - markers)
    if only_rotated:
        problems.append(f"BASIS SYMBOLS: {len(only_rotated)} kernel(s) only in the rotated "
                        f"metallib: {only_rotated[:3]}")
    if only_standard:
        problems.append(f"BASIS SYMBOLS: {len(only_standard)} kernel(s) only in the standard "
                        f"metallib: {only_standard[:3]}")
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


def swift_refs() -> tuple[dict[str, list[str]], set[str]]:
    """(kernel name -> files) for every kernel string Swift asks for, plus the
    interpolated prefixes it can build at runtime."""
    refs: dict[str, list[str]] = {}
    dynamic: set[str] = set()
    lit = re.compile(r"[\"'](q[a-z0-9_]{6,})[\"']")
    interp = re.compile(r"[\"'](q[a-z0-9_]+_[a-z0-9_]*?)\\?\(")
    for path in (ROOT / "Sources").rglob("*.swift"):
        if "QWMetal" in path.parts or path.name == "QWKernelInventory.swift":
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        for name in lit.findall(text):
            if name.startswith(PREFIXES):
                refs.setdefault(name, []).append(str(path.relative_to(ROOT)))
        for pre in interp.findall(text):
            dynamic.add(pre)
    # benches and tests may load bench-local kernels on purpose
    for base in ("Scripts", "Tests"):
        for path in (ROOT / base).rglob("*.swift"):
            text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
            for name in lit.findall(text):
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


DENSE_ENGINE_DIRS = ["Sources/Runtime/QWEngine"]
FLASH_NEXT_ENGINE_DIRS = ["Sources/Subsystems/QWFlashNext"]
SHARED_OPERATOR_DIRS = ["Sources/Operators/QWAttn", "Sources/Operators/QWGDN",
                        "Sources/Operators/QWGEMM", "Sources/Operators/QWHead",
                        "Sources/Operators/QWKV", "Sources/Operators/QWMLP",
                        "Sources/Operators/QWNorm", "Sources/Operators/QWRope",
                        "Sources/Foundation/QWQuant"]
FLASH_NEXT_OPERATOR_DIRS = ["Sources/Operators/QWGDN", "Sources/Operators/QWGEMM",
                            "Sources/Foundation/QWQuant"]


def door_problems(available: set[str], dynamic: set[str] | None = None) -> list[str]:
    """Doors a module's engine dispatches but its module does not declare."""
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
    common = set(inventory["common"])
    declared_standard = common | set(inventory["standard_only"])
    doors = {
        "DenseModule": declared_standard | set(inventory["dense_only"]),
        "FlashNextModule": common | set(inventory["flash_only"]),
        "StandardModule": declared_standard,
        "Qwen3MoEModule": declared_standard | set(inventory["moe_only"]),
    }
    for missing in sorted(set().union(*doors.values()) - available):
        problems.append(f"MISSING INVENTORY KERNEL: {missing!r} is not in the metallib")
    dense = (dispatched_in(DENSE_ENGINE_DIRS, ("Module", "Core"))
             | dispatched_in(SHARED_OPERATOR_DIRS))
    flash = (dispatched_in(FLASH_NEXT_ENGINE_DIRS)
             | dispatched_in(FLASH_NEXT_OPERATOR_DIRS))
    standard = dense - set(inventory["dense_only"])
    # The sparse family adds the chain's own kernels to the standard set.
    chain = chain_kernels()
    expected = {
        "DenseModule": dense,
        "FlashNextModule": flash,
        "StandardModule": standard,
        "Qwen3MoEModule": standard | chain,
    }
    for name, want in expected.items():
        have = doors[name]
        for missing in sorted(want - have):
            problems.append(f"UNDECLARED DOOR: {name} dispatches {missing!r} "
                            f"but does not declare it")
    all_dispatched = dense | flash | chain | dispatched_in(["Sources/Foundation/QWLoad"])
    for extra in sorted(set().union(*doors.values()) - all_dispatched):
        if any(extra.startswith(d) for d in dynamic):
            continue
        problems.append(f"STALE INVENTORY DOOR: inventory declares {extra!r} "
                        f"but no engine or operator dispatches it")
    # A door the sparse family adds must be one its chain dispatches.
    for extra in sorted(set(inventory["moe_only"]) - chain):
        problems.append(f"UNDECLARED DOOR: Qwen3MoEModule declares {extra!r} "
                        f"but its chain does not dispatch it")
    return problems


def main() -> int:
    names = metallib_names() - declared_helpers()
    refs, dynamic = swift_refs()
    local = bench_local_kernels()
    problems = []

    for name in sorted(names):
        if name in refs:
            continue
        if any(name.startswith(d) for d in dynamic):
            continue          # built by string interpolation at dispatch time
        problems.append(f"UNREACHABLE KERNEL: {name} ships in the metallib but "
                        f"nothing dispatches it")

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
    problems.extend(basis_problems())

    for line in problems:
        print(line)
    print(f"kernels: {len(names)} in metallib, "
          f"{len(refs)} names dispatched from Swift, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
