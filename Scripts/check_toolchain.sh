#!/usr/bin/env bash
# Verifies the host can build QWEngine. Hard-fails on anything the build needs;
# prints one line per check saying what was verified and why it matters.
set -euo pipefail

fail=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n        %s\n' "$1" "$2"; fail=1; }

# 1. macOS 26 or newer: Metal 4 and the NAX GPU kernels need it.
os=$(sw_vers -productVersion 2>/dev/null || echo 0)
if [ "${os%%.*}" -ge 26 ]; then ok "macOS $os (>= 26)"
else bad "macOS $os" "QWEngine needs macOS 26 for Metal 4 (MSL 4.0) and the NAX GPU kernels."; fi

# 2. Apple Silicon: the kernels target the Apple10 GPU family.
arch=$(uname -m)
if [ "$arch" = "arm64" ]; then ok "Apple Silicon ($arch)"
else bad "architecture $arch" "QWEngine targets Apple Silicon (arm64) only."; fi

# 3. Full Xcode selected, not just the Command Line Tools: `xcrun metal` lives in Xcode.
dev=$(xcode-select -p 2>/dev/null || echo "")
case "$dev" in
  *Xcode*.app/Contents/Developer) ok "Xcode selected: $dev" ;;
  *) bad "developer directory: ${dev:-none}" \
        "Install Xcode 26+ and run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" ;;
esac

# 4. Xcode 26+ (ships the Metal 4 compiler used by Scripts/build_metallib.sh).
xv=$(xcodebuild -version 2>/dev/null | awk '/Xcode/{print $2}' || echo 0)
if [ -n "$xv" ] && [ "${xv%%.*}" -ge 26 ]; then ok "Xcode $xv (>= 26)"
else bad "Xcode ${xv:-unknown}" "Metal 4 requires Xcode 26 or newer."; fi

# 5. Metal compiler actually works, not just present.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'kernel void k() {}\n' > "$tmp/k.metal"
if xcrun -sdk macosx metal -std=metal4.0 -c "$tmp/k.metal" -o "$tmp/k.air" 2>"$tmp/err"; then
  ok "metal compiler: -std=metal4.0 compiles a kernel"
else
  bad "metal compiler" "$(head -2 "$tmp/err")"
fi

# 6. Swift 6.2 or newer (swift-tools-version in Package.swift).
sv=$(swift --version 2>/dev/null | sed -n 's/.*Swift version \([0-9.]*\).*/\1/p' | head -1)
if [ -n "$sv" ] && [ "$(printf '%s\n6.2\n' "$sv" | sort -V | head -1)" = "6.2" ]; then ok "Swift $sv (>= 6.2)"
else bad "Swift ${sv:-not found}" "Package.swift requires swift-tools-version 6.2."; fi

# 7. Python 3.11 or newer for the lint and golden scripts.
py=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0)
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then ok "Python $py (>= 3.11)"
else bad "Python ${py}" "The lint and golden scripts need Python 3.11 or newer."; fi

# 8. Optional: Homebrew, used by nothing in the build; reported only.
if command -v brew >/dev/null 2>&1; then ok "Homebrew present (optional)"
else echo "  note  Homebrew not found (optional; nothing in the build needs it)"; fi

if [ "$fail" -ne 0 ]; then
  echo; echo "toolchain check FAILED: fix the items above, then run 'make init' again."
  exit 1
fi
echo "toolchain check: PASS"
