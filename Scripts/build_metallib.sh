#!/usr/bin/env bash
# Build the engine metallib (Sources/QWMetal/*.metal → kernel.metallib).
# QWHAL discovers Sources/QWMetal/kernel.metallib by cwd walk; CI checks
# freshness by rebuilding into a temp and comparing kernel symbol sets.
#
# TWO libraries are built from the same sources, one per activation basis:
#
#   kernel.metallib      ROTATED  — the engine's own weight format, where the
#                                   weights carry the rotation and the
#                                   activations are rotated to match.
#   kernel_std.metallib  STANDARD — an MLX affine checkpoint's planes, which
#                                   carry no rotation, so the activations are
#                                   left alone.
#
# The basis is a BUILD-time property, not a runtime one: the rotation is a
# chain of `simd_shuffle_xor` calls, and a runtime branch in front of them
# puts the shuffle in a divergent region — measured at -14.5% prefill
# throughput on the 8B dense model. A compile-time constant lets the branch
# vanish, so each library's rotation code is exactly what it should be.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-Sources/QWMetal/kernel.metallib}"
STD_OUT="${2:-Sources/QWMetal/kernel_std.metallib}"

if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  METAL_CC=(xcrun -sdk macosx metal)
  METALLIB_LD=(xcrun -sdk macosx metallib)
else
  CRYPTEX_BIN="$(ls -d /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin 2>/dev/null | tail -n 1)"
  SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
  METAL_CC=("$CRYPTEX_BIN/metal" -isysroot "$SDK_PATH")
  METALLIB_LD=("$CRYPTEX_BIN/metallib")
fi

# build_variant <output> [extra compiler flags...]
build_variant() {
  local out="$1"; shift
  local tmp; tmp="$(mktemp -d)"
  local airs=()
  while IFS= read -r -d '' m; do
    a="$tmp/$(basename "${m%.metal}").air"
    "${METAL_CC[@]}" -std=metal4.0 -O2 "$@" \
      -ISources/QWMetal -ISources/QWMetal/include -ISources/QWMetal/gdn \
      -c "$m" -o "$a"
    airs+=("$a")
  done < <(find Sources/QWMetal -name '*.metal' -print0 | sort -z)
  "${METALLIB_LD[@]}" "${airs[@]}" -o "$out"
  rm -rf "$tmp"
}

build_variant "$OUT"
build_variant "$STD_OUT" -DQW_BASIS_STANDARD=1

for d in .build/release .build/debug; do
  if [ -d "$d" ]; then
    cp "$OUT" "$d/kernel.metallib"
    cp "$STD_OUT" "$d/kernel_std.metallib"
  fi
done
echo "built $OUT"
echo "built $STD_OUT"
