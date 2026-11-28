#!/usr/bin/env python3
"""Verify compiled metallibs target macOS >= 26.2 and enforce the NAX guard."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METALLIBS = (
    ROOT / "Sources" / "QWMetal" / "kernel.metallib",
    ROOT / "Sources" / "QWMetal" / "kernel_std.metallib",
)
M3_METALLIBS = (
    ROOT / "Sources" / "QWMetal" / "kernel_m3.metallib",
    ROOT / "Sources" / "QWMetal" / "kernel_m3_std.metallib",
)
MIN_TARGET = (26, 2)
M3_MIN_TARGET = (15, 0)
M3_MAX_TARGET_EXCL = (26, 0)
NAX_SYMBOL_RE = re.compile(r"__tensorops|BaseNAXFrag|cooperative_tensor")


def find_metal_tool(name: str) -> list[str]:
    probe = subprocess.run(["xcrun", "-sdk", "macosx", name, "--version"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return ["xcrun", "-sdk", "macosx", name]
    cryptex = sorted(Path("/private/var/run/com.apple.security.cryptexd/mnt").glob(
        f"com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin/{name}"))
    if cryptex:
        if name == "metal":
            sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            return [str(cryptex[-1]), "-isysroot", sdk]
        return [str(cryptex[-1])]
    return ["xcrun", "-sdk", "macosx", name]


def check_metallib_platform(readobj: list[str], path: Path,
                            min_target: tuple[int, int] = MIN_TARGET,
                            max_excl: tuple[int, int] | None = None) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing metallib: {path}")
    r = subprocess.run([*readobj, "--headers", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"metal-readobj failed for {path}:\n{r.stderr[:1000]}")
    maj_m = re.search(r"PlatformMajor:\s*(\d+)", r.stdout)
    min_m = re.search(r"PlatformMinor:\s*(\d+)", r.stdout)
    if not maj_m or not min_m:
        raise RuntimeError(f"could not parse PlatformMajor/Minor from {path}")
    target = (int(maj_m.group(1)), int(min_m.group(1)))
    if target < min_target:
        raise RuntimeError(
            f"{path} targets macOS {target[0]}.{target[1]}, must be >= {min_target[0]}.{min_target[1]}")
    if max_excl is not None and target >= max_excl:
        raise RuntimeError(
            f"{path} targets macOS {target[0]}.{target[1]}, Metal 3.2 library must be < {max_excl[0]}.{max_excl[1]}")
    return f"{path.name}: macOS {target[0]}.{target[1]}"


def check_m3_nax_free(path: Path) -> None:
    r = subprocess.run(["strings", "-a", str(path)], capture_output=True, text=True, check=True)
    hits = [line for line in r.stdout.splitlines() if NAX_SYMBOL_RE.search(line)]
    if hits:
        raise RuntimeError(f"{path.name} contains NAX tensor-op symbols: {hits[:3]}")


def check_negative_compile_guard(metal_cc: list[str]) -> None:
    src = '#include "mlx/backend/metal/kernels/steel/gemm/nax.h"\nkernel void k() {}\n'
    with tempfile.TemporaryDirectory(prefix="qwg-nax-guard-") as td:
        tmp = Path(td)
        m_path = tmp / "guard.metal"
        a_path = tmp / "guard.air"
        m_path.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [*metal_cc, "-std=metal4.0", "-mmacosx-version-min=26.0",
             "-ISources/QWMetal", "-ISources/QWMetal/include", "-c", str(m_path), "-o", str(a_path)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode == 0:
            raise RuntimeError("NAX header compiled under -mmacosx-version-min=26.0 without failing")
        if "require -mmacosx-version-min=26.2" not in r.stderr:
            raise RuntimeError(f"unexpected NAX guard error under 26.0:\n{r.stderr[:1000]}")


def check_mpp_source_guards() -> int:
    metal_dir = ROOT / "Sources" / "QWMetal"
    checked = 0
    missing: list[str] = []
    for ext in ("*.metal", "*.metalh"):
        for path in sorted(metal_dir.rglob(ext)):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "MetalPerformancePrimitives.h" not in text:
                continue
            checked += 1
            if "__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__" not in text or "260200" not in text:
                missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise RuntimeError(
            "NAX/MPP files missing macOS 26.2 (#error < 260200) guard: " + ", ".join(missing))
    return checked


def main() -> int:
    readobj = find_metal_tool("metal-readobj")
    metal_cc = find_metal_tool("metal")
    summaries = [check_metallib_platform(readobj, lib) for lib in METALLIBS]
    for lib in M3_METALLIBS:
        summaries.append(check_metallib_platform(readobj, lib, M3_MIN_TARGET, M3_MAX_TARGET_EXCL))
        check_m3_nax_free(lib)
    check_negative_compile_guard(metal_cc)
    guarded = check_mpp_source_guards()
    print(f"metallib target: PASS ({', '.join(summaries)}; 26.0 guard + {guarded} MPP files verified)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
