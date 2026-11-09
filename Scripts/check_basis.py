#!/usr/bin/env python3
"""The basis gate: the activation basis must be a BUILD-time property.

The engine's weights are quantised in one of two bases — ROTATED (the
incumbent's own format, where the packer applied the Walsh-Hadamard
transform to the weights' input dimension) or STANDARD (an MLX affine
checkpoint's planes, which carry no rotation). The activations must be
in the same basis as the weights they meet.

The basis is decided when the metallib is COMPILED, not at runtime, and
this gate is why. A Metal function constant resolves through a RUNTIME
global, so a branch on one sits in front of every rotation — and the
rotation's core is a chain of `simd_shuffle_xor` calls, a SIMD-group-wide
operation. The branch puts the shuffle in a DIVERGENT region, and the
compiler can no longer treat the rotation as straight-line code. Measured
on the 8B dense model, that cost 14.5% of prefill throughput (1820.6 ->
1556.3 tok/s) at every context length, while decode — bound by weight
bandwidth, not ALU — did not move.

The gate reads the METAL AIR of both builds, and checks:

  1. The standard build must hold STRICTLY FEWER `simd_shuffle_xor` than
     the rotated one. A compile-time basis deletes the rotation from the
     code; a runtime basis leaves it in a branch that is not taken, and
     the counts would be equal. This is the check that catches the
     regression above.
  2. No function may GAIN a shuffle in the standard build.
  3. A function that keeps its shuffles must be the SAME function in both
     builds. The shuffles that survive are the q4 quantiser's lane
     reductions, which have nothing to do with the basis — so the basis
     must not have touched them.
  4. A function that LOSES a shuffle must carry no 1/sqrt(32) multiply in
     the standard build. The scale belongs to the transform, so a removed
     rotation must leave nothing of itself behind.

Run:  python3 Scripts/check_basis.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METAL_ROOT = ROOT / "Sources" / "QWMetal"
INCLUDES = ["-ISources/QWMetal", "-ISources/QWMetal/include", "-ISources/QWMetal/gdn"]

SHUFFLE = "simd_shuffle_xor"
# 1/sqrt(32), the transform's own scale. It belongs to the ROTATION, so it
# must not survive one — except in the offline HQ4 packer, whose job is to
# produce the rotated format and is the same in both builds.
SCALE = "0x3FC6A09E60000000"
DEFINE = re.compile(r"^define[^\n]*@([A-Za-z0-9_.$]+)\(", re.M)


def normalise(body: list[str]) -> list[str]:
    """The body without metadata, attribute ids and block labels."""
    text = "\n".join(body)
    text = re.sub(r"![-\w.]+ !\d+", "", text)
    text = re.sub(r"![-\w.]+ = [^\n]*", "", text)
    text = re.sub(r"; preds = [^\n]*", "", text)
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"!\d+", "", text)
    text = re.sub(r"%[A-Za-z0-9_.$]+", "%V", text)
    text = re.sub(r"^\s*\d+:\s*", "", text, flags=re.M)
    return [l.strip() for l in text.splitlines() if l.strip()]


def metal_cc() -> list[str]:
    probe = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return ["xcrun", "-sdk", "macosx", "metal"]
    cryptex = sorted(Path("/private/var/run/com.apple.security.cryptexd/mnt").glob(
        "com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin/metal"))
    sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if cryptex:
        return [str(cryptex[-1]), "-isysroot", sdk]
    return ["xcrun", "-sdk", "macosx", "metal"]


def compile_ir(tree: Path, out: Path, extra: list[str]) -> dict[str, list[str]]:
    """Compile every .metal file in `tree` to textual IR; return function -> body."""
    funcs: dict[str, list[str]] = {}
    files = sorted(tree.rglob("*.metal"))
    if not files:
        sys.exit(f"no .metal files under {tree}")
    cc = metal_cc()
    for mtl in files:
        air = out / (mtl.stem + ".ll")
        cmd = [*cc, "-std=metal4.0", "-O2", *extra,
               *INCLUDES, "-S", "-emit-llvm", str(mtl), "-o", str(air)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"metal compile failed for {mtl}:\n{r.stderr[:2000]}")
        src = air.read_text()
        for m in DEFINE.finditer(src):
            name = m.group(1)
            end = src.find("\n}\n", m.start())
            if end < 0:
                continue
            funcs[name] = src[m.start():end].splitlines()
    return funcs


def shuffles(body: list[str]) -> int:
    return sum(1 for line in body if SHUFFLE in line)


def main() -> int:
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="qwg-basis-") as td:
        tmp = Path(td)
        (tmp / "rot").mkdir()
        (tmp / "std").mkdir()
        rotated = compile_ir(METAL_ROOT, tmp / "rot", [])
        standard = compile_ir(METAL_ROOT, tmp / "std", ["-DQW_BASIS_STANDARD=1"])

        common = set(rotated) & set(standard)
        total_rot = sum(shuffles(rotated[n]) for n in common)
        total_std = sum(shuffles(standard[n]) for n in common)

        if total_std >= total_rot:
            problems.append(
                f"THE BASIS IS NOT COMPILE-TIME: the standard build holds {total_std} "
                f"{SHUFFLE}(s) and the rotated one {total_rot}. A runtime basis leaves the "
                f"rotation in a branch that is not taken, so the counts would be equal — and "
                f"that costs 14.5% of prefill.")

        gained = [n for n in sorted(common)
                  if shuffles(standard[n]) > shuffles(rotated[n])]
        if gained:
            problems.append(
                f"THE STANDARD BUILD GAINED A SHUFFLE in {len(gained)} function(s); the basis "
                f"only removes rotations. First: {gained[0]}")

        touched = [n for n in sorted(common)
                   if shuffles(standard[n]) == shuffles(rotated[n]) > 0
                   and normalise(standard[n]) != normalise(rotated[n])]
        if touched:
            problems.append(
                f"THE BASIS TOUCHED A BASIS-INDEPENDENT SHUFFLE in {len(touched)} function(s) "
                f"— the survivors are the q4 quantiser's lane reductions. First: {touched[0]}")

        leftover = [n for n in sorted(common)
                    if shuffles(standard[n]) < shuffles(rotated[n])
                    and SCALE in "\n".join(standard[n])]
        if leftover:
            problems.append(
                f"A REMOVED ROTATION LEFT ITS SCALE BEHIND in {len(leftover)} function(s); the "
                f"1/sqrt(32) multiply belongs to the transform. First: {leftover[0]}")

        kept = sum(1 for n in common if shuffles(standard[n]) > 0)

    for p in problems:
        print("BASIS GATE: " + p)
    if problems:
        print(f"basis: {len(problems)} problem(s)")
        return 1
    print(f"basis: the standard build drops {total_rot - total_std} of {total_rot} "
          f"{SHUFFLE}(s) — the rotations are compile-time")
    print(f"basis: {kept} function(s) keep a basis-independent shuffle, identical in both builds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
