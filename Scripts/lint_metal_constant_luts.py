#!/usr/bin/env python3
"""Enforce Law 27: per-PSO transitive constant LUT footprint must stay <= 32 KiB."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METAL_DIR = ROOT / "Sources" / "QWMetal"
INCLUDE_DIR = METAL_DIR / "include"
MAX_CONST_BYTES = 32_768

TYPE_SIZES = {
    "char": 1, "uchar": 1, "int8_t": 1, "uint8_t": 1,
    "short": 2, "ushort": 2, "int16_t": 2, "uint16_t": 2, "half": 2, "bfloat": 2,
    "int": 4, "uint": 4, "int32_t": 4, "uint32_t": 4, "float": 4,
    "char4": 4, "uchar4": 4, "short2": 4, "ushort2": 4, "half2": 4, "bfloat2": 4,
    "long": 8, "ulong": 8, "int64_t": 8, "uint64_t": 8,
    "short4": 8, "ushort4": 8, "int2": 8, "uint2": 8, "float2": 8, "half4": 8, "bfloat4": 8,
    "QWDecodedM": 4, "QWDecodedQ": 4,
    "int4": 16, "uint4": 16, "float4": 16,
}

INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
CONST_ARR_RE = re.compile(
    r'^(?:alignas\([^)]*\)\s*)?(?:static\s+)?constant\s+(?:const\s+)?(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\](?:\s*\[\s*(\d+)\s*\])?\s*=',
    re.MULTILINE,
)

# Mutually exclusive SHVQ per-BITS tables isolated behind `if constexpr (BITS == ...)`.
SHVQ_BIT_EXCLUSIVE_GROUPS = {
    "kShvqLut1Bits": "shvq1",
    "kShvqLut15PackedI8": "shvq15",
    "kShvqLut15DirectBits": "shvq15_direct",
    "kShvqLut2Bits": "shvq2",
    "kShvqLut3Bits": "shvq3",
    "kShvqLut4PackedU8": "shvq4",
    "kShvqLut4DirectBits": "shvq4_direct",
}


def resolve_include(cur: Path, target: str) -> Path | None:
    for base in (cur.parent, METAL_DIR, INCLUDE_DIR):
        cand = (base / target).resolve()
        if cand.is_file():
            return cand
    return None


def file_const_arrays(path: Path, cache: dict[Path, list[tuple[str, int]]]) -> list[tuple[str, int]]:
    if path in cache:
        return cache[path]
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: list[tuple[str, int]] = []
    for m in CONST_ARR_RE.finditer(text):
        ty, name, d1, d2 = m.groups()
        if ty not in TYPE_SIZES:
            raise ValueError(f"Unknown Metal constant type {ty!r} for {name} in {path}")
        elem = TYPE_SIZES[ty]
        count = int(d1) * (int(d2) if d2 else 1)
        out.append((name, elem * count))
    cache[path] = out
    return out


def tu_worst_case_bytes(tu: Path, cache: dict[Path, list[tuple[str, int]]]) -> tuple[int, list[str]]:
    tu_text = tu.read_text(encoding="utf-8", errors="ignore")
    routed_direct = bool(re.search(r"^\s*#define\s+QW_SHVQ_ROUTED_DIRECT_LUT\b", tu_text, re.MULTILINE))
    sub2_direct = "qw_shvq_sub2_routed_lut.metalh" in tu_text or "kShvqLut15DirectBits" in tu_text

    visited: set[Path] = set()
    stack = [tu.resolve()]
    shared_bytes = 0
    shvq_buckets: dict[str, int] = {}
    shared_contrib: list[str] = []

    while stack:
        p = stack.pop()
        if p in visited or not p.is_file():
            continue
        visited.add(p)
        for name, nbytes in file_const_arrays(p, cache):
            bucket = SHVQ_BIT_EXCLUSIVE_GROUPS.get(name)
            if bucket is None:
                shared_bytes += nbytes
                shared_contrib.append(f"{name}:{nbytes}B")
                continue
            if name == "kShvqLut4DirectBits" and not routed_direct and "kShvqLut4DirectBits" not in tu_text:
                continue
            if name == "kShvqLut4PackedU8" and routed_direct and "kShvqLut4PackedU8" not in tu_text:
                continue
            if name == "kShvqLut15DirectBits" and not sub2_direct:
                continue
            if name == "kShvqLut15PackedI8" and sub2_direct and "kShvqLut15PackedI8" not in tu_text:
                continue
            shvq_buckets[bucket] = shvq_buckets.get(bucket, 0) + nbytes
        text = p.read_text(encoding="utf-8", errors="ignore")
        for inc in INCLUDE_RE.findall(text):
            rp = resolve_include(p, inc)
            if rp is not None and rp not in visited:
                stack.append(rp)

    max_shvq_name = ""
    max_shvq_bytes = 0
    for k, v in shvq_buckets.items():
        if v > max_shvq_bytes:
            max_shvq_name = k
            max_shvq_bytes = v
    total = shared_bytes + max_shvq_bytes
    contrib = list(shared_contrib)
    if max_shvq_bytes > 0:
        contrib.append(f"{max_shvq_name}:{max_shvq_bytes}B")
    return total, contrib


def main() -> int:
    cache: dict[Path, list[tuple[str, int]]] = {}
    violations: list[str] = []
    peak = 0
    count = 0
    for tu in sorted(METAL_DIR.rglob("*.metal")):
        count += 1
        total, contrib = tu_worst_case_bytes(tu, cache)
        if total > peak:
            peak = total
        if total > MAX_CONST_BYTES:
            violations.append(
                f"{tu.relative_to(ROOT)}: {total} B > {MAX_CONST_BYTES} B ({', '.join(contrib)})"
            )
    if violations:
        for v in violations:
            print(f"FAIL (Law 27 constant LUT ceiling): {v}", file=sys.stderr)
        return 1
    print(f"metal constant LUTs: PASS ({count} TUs checked, peak {peak} / {MAX_CONST_BYTES} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
