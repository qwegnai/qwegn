#!/usr/bin/env bash
# Generates the synthetic golden fixtures the XCTest suites read, into
# Scripts/spike/out. These generators need no model weights, so every
# contributor can produce them with `make init`.
#
# Generators that need a local model pack or checkpoint are listed in
# WEIGHTED below and are skipped with that reason. Scripts/flashnext/ holds
# the real-checkpoint generators; they are run by hand, not by this script. Their tests report a skip,
# never a failure, when the fixture is absent.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-oracle/bin/python
OUT=Scripts/spike/out
mkdir -p "$OUT"

# Needs a local Qwen checkpoint or pack (~/.qwg/models or the HF cache).
WEIGHTED=(
  gen_vision_fixtures.py
  gen_mm_fixtures.py
  gen_fn_reference_fixtures.py
)

is_weighted() {
  local f=$1 w
  for w in "${WEIGHTED[@]}"; do [ "$w" = "$f" ] && return 0; done
  return 1
}

ok=0; skipped=0; failed=()
for path in Scripts/oracle/gen_*.py; do
  f=$(basename "$path")
  if is_weighted "$f"; then
    echo "skip  $f  (needs a local model checkpoint)"
    skipped=$((skipped + 1))
    continue
  fi
  if "$PY" "$path" >"$OUT/.$f.log" 2>&1; then
    echo "ok    $f"
    ok=$((ok + 1))
  else
    echo "FAIL  $f  (see $OUT/.$f.log)"
    failed+=("$f")
  fi
done

echo "goldens: $ok generated, $skipped skipped (weights), ${#failed[@]} failed"
if [ "${#failed[@]}" -ne 0 ]; then
  printf '  failed: %s\n' "${failed[@]}"
  exit 1
fi
