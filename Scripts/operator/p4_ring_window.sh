#!/bin/bash
# p4_ring_window.sh - FN prefill A/B for the device-side ring copy (P4).
# Two passes per arm: pass 1 warms the page cache, pass 2 is the reading.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PACK="${PACK:-$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32/model.qwp}"
BIN="$REPO/.build/release/qwg"
OUT="${OUT:-/tmp/qw-repro/p4ring-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
PROMPT="A tokenizer converts text into identifiers. It handles unicode, normalization, merging rules and vocabulary lookups. Byte level models keep every character representable. Subword models balance vocabulary size against sequence length. Explain each stage in order, then summarize. A tokenizer converts text into identifiers. It handles unicode, normalization, merging rules and vocabulary lookups. Byte level models keep every character representable. Subword models balance vocabulary size against sequence length."
GREP_RE='fn prefill:|fn chunk drains|phase-probe|prefill:|decode:|error|Error|Killed|trap'
pgrep -x qwg >/dev/null && { echo REFUSED; exit 1; }
{ echo stamp $(date -u +%FT%TZ); echo mlx_serve $(pgrep -x mlx-serve >/dev/null && echo UP || echo DOWN); echo bin $(shasum -a 256 "$BIN" | cut -c1-16); } | tee "$OUT/stamp.txt"
export QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1
for pass in 1 2; do
  for arm in on off; do
    if [ "$arm" = off ]; then export QW_FN_RING_COPY=0; else unset QW_FN_RING_COPY; fi
    timeout 420 "$BIN" run "$PACK" --prompt "$PROMPT" --max-new 8 --mtp-drafts 0 --no-thinking --no-stream --verbose 2>&1 | grep -E "$GREP_RE" > "$OUT/$arm-pass$pass.log"
    echo "pass$pass $arm: $(grep -E '^prefill:' "$OUT/$arm-pass$pass.log" | tail -1) | $(grep -oE 'submissions=[0-9]+ waits=[0-9]+' "$OUT/$arm-pass$pass.log" | tail -1) | chunk-lines=$(grep -c 'fn chunk drains' "$OUT/$arm-pass$pass.log")"
  done
done
unset QW_FN_RING_COPY QW_FN_PHASES QW_FN_WIRE_SKIP
echo logs: $OUT
