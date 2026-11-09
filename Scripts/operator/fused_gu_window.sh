#!/bin/bash
# fused_gu_window.sh — full-model A/B for the fused silu+FWHT gu door (C14).
# ON = default (QW_MLP_FUSED_GU unset), OFF = QW_MLP_FUSED_GU=0. Alternating.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PACK="${PACK:-$HOME/.qwg/models/Qwen3.8-27B-hq4g32/model.qwp}"
BIN="$REPO/.build/release/qwg"
OUT="${OUT:-/tmp/qw-repro/fusedgu-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
PROMPT="Explain the difference between DRAM bandwidth and L2 bandwidth in one paragraph."
GREP_RE='decode:|tok/s|prefill|pack load|error|Error|Killed|trap'

pgrep -x qwg >/dev/null && { echo "REFUSED: a qwg instance is running"; exit 1; }
{
  echo "stamp $(date -u +%FT%TZ)"
  echo "mlx_serve $(pgrep -x mlx-serve >/dev/null && echo UP || echo DOWN)"
  echo "bin $(shasum -a 256 "$BIN" | cut -c1-16)"
} | tee "$OUT/stamp.txt"

for i in 1 2 3; do
  for arm in on off; do
    if [ "$arm" = off ]; then export QW_MLP_FUSED_GU=0; else unset QW_MLP_FUSED_GU; fi
    timeout 300 "$BIN" run "$PACK" --prompt "$PROMPT" --max-new 64 \
      --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 \
      | grep -E "$GREP_RE" > "$OUT/$arm-$i.log"
    printf '%s %s: %s\n' "$i" "$arm" "$(grep -E 'decode:' "$OUT/$arm-$i.log" | tail -1)"
  done
done
unset QW_MLP_FUSED_GU
echo "logs: $OUT"
