#!/bin/bash
# c30_c29_sweeps.sh — C30-T5 row-cap sweep + C29-T3 MTP depth sweep.
# One clean GPU window; prints one line per arm.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/.build/release/qwg"
PACK="${PACK:-$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32/model.qwp}"
OUT="${OUT:-/tmp/qw-repro/sweeps-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
PROMPT="$(python3 -c "print(('A tokenizer converts text into identifiers. It handles unicode, normalization, merging rules and vocabulary lookups. Byte level models keep every character representable. ')*10)")"
export QW_FN_WIRE_SKIP=1
pgrep -x qwg >/dev/null && { echo REFUSED; exit 1; }
echo "stamp $(date -u +%FT%TZ)" | tee "$OUT/stamp.txt"
echo mlx_serve $(pgrep -x mlx-serve >/dev/null && echo UP || echo DOWN) >> "$OUT/stamp.txt"

echo "--- C30-T5 row cap sweep (drafts 1)"
for rows in 64 128 256 512; do
  QW_FN_PREFILL_ROWS=$rows timeout 400 "$BIN" run "$PACK" --prompt "$PROMPT" --max-new 4 \
    --mtp-drafts 1 --no-thinking --no-stream --verbose > "$OUT/rows-$rows.log" 2>&1
  echo "rows=$rows :: $(grep -E '^prefill:' "$OUT/rows-$rows.log" | tr '\n' ' ')"
done

echo "--- C29-T3 MTP depth sweep (rows 64)"
for d in 1 2 3; do
  QW_FN_PREFILL_ROWS=64 timeout 400 "$BIN" run "$PACK" --prompt "$PROMPT" --max-new 32 \
    --mtp-drafts $d --no-thinking --no-stream --verbose > "$OUT/depth-$d.log" 2>&1
  echo "drafts=$d :: $(grep -E '^prefill:|^decode:' "$OUT/depth-$d.log" | tr '\n' ' ') :: $(grep -oE 'rounds=[0-9]+ verified=[0-9]+ accepted=[0-9]+' "$OUT/depth-$d.log")"
done
echo "logs: $OUT"
