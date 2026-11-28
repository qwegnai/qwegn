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
FORCE="${QW_FORCE_METALLIB:-0}"
STRIP_NAX=0
OUT=""
STD_OUT=""
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    FORCE=1
  elif [ "$arg" = "--strip-nax" ]; then
    STRIP_NAX=1
  elif [ -z "$OUT" ]; then
    OUT="$arg"
  elif [ -z "$STD_OUT" ]; then
    STD_OUT="$arg"
  fi
done
if [ "$STRIP_NAX" -eq 1 ]; then
  if [ -z "$OUT" ] || [ -z "$STD_OUT" ]; then
    echo "usage: Scripts/build_metallib.sh [--force] [--strip-nax <out_rot> <out_std>]" >&2
    exit 2
  fi
else
  OUT="${OUT:-Sources/QWMetal/kernel.metallib}"
  STD_OUT="${STD_OUT:-Sources/QWMetal/kernel_std.metallib}"
  M3_OUT="${QW_METALLIB_M3_OUT:-Sources/QWMetal/kernel_m3.metallib}"
  M3_STD_OUT="${QW_METALLIB_M3_STD_OUT:-Sources/QWMetal/kernel_m3_std.metallib}"
fi
METAL_MIN_MACOS="${QW_METAL_MIN_MACOS:-26.2}"
METAL3_MIN_MACOS="${QW_METAL3_MIN_MACOS:-15.0}"
CACHE_DIR="${QW_METALLIB_CACHE_DIR:-.build/metallib_cache}"

if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  METAL_CC=(xcrun -sdk macosx metal)
  METALLIB_LD=(xcrun -sdk macosx metallib)
  METAL_READOBJ=(xcrun -sdk macosx metal-readobj)
else
  CRYPTEX_BIN="$(ls -d /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin 2>/dev/null | tail -n 1)"
  SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
  METAL_CC=("$CRYPTEX_BIN/metal" -isysroot "$SDK_PATH")
  METALLIB_LD=("$CRYPTEX_BIN/metallib")
  METAL_READOBJ=("$CRYPTEX_BIN/metal-readobj")
fi

JOBS="${QW_METAL_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
METAL_TOOLCHAIN_ID="$("${METAL_CC[@]}" --version 2>&1 || true)|$("${METALLIB_LD[@]}" --version 2>&1 || true)|$(xcrun --sdk macosx --show-sdk-build-version 2>/dev/null || true)|min=${METAL_MIN_MACOS}|min3=${METAL3_MIN_MACOS}"
BASE_INPUT_HASH="$(QW_MTL_TC="$METAL_TOOLCHAIN_ID" python3 - <<'PY'
import hashlib, os
from pathlib import Path

h = hashlib.sha256()
h.update(os.environ.get("QW_MTL_TC", "").encode("utf-8"))
h.update(b"\0script\0")
h.update(Path("Scripts/build_metallib.sh").read_bytes())
root = Path("Sources/QWMetal")
for p in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".metallib" and p.name != ".DS_Store"):
    h.update(b"\0file\0")
    h.update(p.as_posix().encode("utf-8"))
    h.update(b"\0")
    h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
PY
)"

verify_metallib_target() {
  local lib="$1" mode="${2:-metal4}" maj="" min=""
  read -r maj min < <("${METAL_READOBJ[@]}" --headers "$lib" | awk '/PlatformMajor:/ {maj=$2} /PlatformMinor:/ {min=$2} END {print maj, min}')
  if [ -z "$maj" ] || [ -z "$min" ] || ! [[ "$maj" =~ ^[0-9]+$ ]] || ! [[ "$min" =~ ^[0-9]+$ ]]; then
    echo "FAIL: could not parse macOS platform target from $lib" >&2
    return 1
  fi
  if [ "$mode" = "metal3" ]; then
    if [ "$maj" -lt 15 ] || [ "$maj" -ge 26 ]; then
      echo "FAIL: $lib targets macOS ${maj}.${min}, expected [15.0, 26.0) for Metal 3.2" >&2
      return 1
    fi
  elif [ "$maj" -lt 26 ] || { [ "$maj" -eq 26 ] && [ "$min" -lt 2 ]; }; then
    echo "FAIL: $lib targets macOS ${maj}.${min}, but Apple MPP NAX matmul2d requires >= 26.2" >&2
    return 1
  fi
  echo "verified $lib target: macOS ${maj}.${min} ($mode)"
}

verify_strip_nax_clean() {
  local lib="$1" hits=""
  hits="$(strings -a "$lib" | grep -E '__tensorops|BaseNAXFrag|cooperative_tensor' || true)"
  if [ -n "$hits" ]; then
    echo "FAIL: stripped metallib $lib still contains tensor-op symbols:" >&2
    printf '%s\n' "$hits" | head -n 10 >&2
    return 1
  fi
  echo "verified $lib has 0 NAX tensor-op symbols"
}

# build_variant <output> <std_flag> <min_macos> <mode> [extra compiler flags...]
build_variant() {
  local out="$1" std_flag="$2" min_macos="$3" mode="$4"
  shift 4
  local vkey
  vkey="$(printf '%s|%s|%s|%s' "$BASE_INPUT_HASH" "$std_flag" "$min_macos" "$*" | shasum -a 256 | awk '{print $1}')"
  local cached_blob="$CACHE_DIR/${vkey}.metallib"
  if [ "$FORCE" != "1" ] && [ -s "$cached_blob" ]; then
    mkdir -p "$(dirname "$out")"
    if [ ! -f "$out" ] || ! cmp -s "$cached_blob" "$out"; then
      cp "$cached_blob" "${out}.tmp.$$"
      mv "${out}.tmp.$$" "$out"
    fi
    touch "$cached_blob" 2>/dev/null || true
    verify_metallib_target "$out" "$mode"
    return 0
  fi

  local tmp; tmp="$(mktemp -d)"
  local airs=() pids=() errs=() sources=()
  local idx=0 m a err fail=0
  while IFS= read -r -d '' m; do
    a="$(printf '%s/%04d_%s.air' "$tmp" "$idx" "$(basename "${m%.metal}")")"
    err="$a.err"
    "${METAL_CC[@]}" "-std=${std_flag}" "-mmacosx-version-min=${min_macos}" -O2 "$@" \
      -ISources/QWMetal -ISources/QWMetal/include -ISources/QWMetal/gdn \
      -c "$m" -o "$a" 2>"$err" &
    pids+=($!)
    errs+=("$err")
    sources+=("$m")
    airs+=("$a")
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$JOBS" ]; then
      if ! wait "${pids[0]}"; then fail=1; fi
      pids=("${pids[@]:1}")
    fi
  done < <(find Sources/QWMetal -name '*.metal' -print0 | sort -z)
  for p in "${pids[@]}"; do
    if ! wait "$p"; then fail=1; fi
  done
  if [ "$fail" -ne 0 ]; then
    for ((i=0; i<${#sources[@]}; i++)); do
      if [ -s "${errs[$i]}" ]; then
        echo "metal compile failed for ${sources[$i]}:" >&2
        cat "${errs[$i]}" >&2
      fi
    done
    rm -rf "$tmp"
    return 1
  fi
  mkdir -p "$(dirname "$out")"
  "${METALLIB_LD[@]}" "${airs[@]}" -o "$out"
  rm -rf "$tmp"
  verify_metallib_target "$out" "$mode"
  mkdir -p "$CACHE_DIR"
  cp "$out" "${cached_blob}.tmp.$$"
  mv "${cached_blob}.tmp.$$" "$cached_blob"
  ls -1t "$CACHE_DIR"/*.metallib 2>/dev/null | tail -n +13 | xargs rm -f -- 2>/dev/null || true
}

if [ "$STRIP_NAX" -eq 1 ]; then
  build_variant "$OUT" "metal4.0" "$METAL_MIN_MACOS" "metal4" -DQW_DISABLE_NAX=1
  build_variant "$STD_OUT" "metal4.0" "$METAL_MIN_MACOS" "metal4" -DQW_BASIS_STANDARD=1 -DQW_DISABLE_NAX=1
  verify_strip_nax_clean "$OUT"
  verify_strip_nax_clean "$STD_OUT"
  echo "built $OUT (strip-nax)"
  echo "built $STD_OUT (strip-nax)"
  exit 0
fi

build_variant "$OUT" "metal4.0" "$METAL_MIN_MACOS" "metal4"
build_variant "$STD_OUT" "metal4.0" "$METAL_MIN_MACOS" "metal4" -DQW_BASIS_STANDARD=1
build_variant "$M3_OUT" "metal3.2" "$METAL3_MIN_MACOS" "metal3" -DQW_DISABLE_NAX=1
build_variant "$M3_STD_OUT" "metal3.2" "$METAL3_MIN_MACOS" "metal3" -DQW_BASIS_STANDARD=1 -DQW_DISABLE_NAX=1
verify_strip_nax_clean "$M3_OUT"
verify_strip_nax_clean "$M3_STD_OUT"

for d in .build/release .build/debug; do
  if [ -d "$d" ]; then
    cp "$OUT" "$d/kernel.metallib"
    cp "$STD_OUT" "$d/kernel_std.metallib"
    cp "$M3_OUT" "$d/kernel_m3.metallib"
    cp "$M3_STD_OUT" "$d/kernel_m3_std.metallib"
  fi
done
echo "built $OUT"
echo "built $STD_OUT"
echo "built $M3_OUT"
echo "built $M3_STD_OUT"
