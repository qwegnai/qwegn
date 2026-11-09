#!/bin/bash
# rows4_window.sh — full-model A/B for the promoted MLP down door (C14).
# Arms: rows4 ON (default) vs OFF (QW_MLP_DOWN_ROWS4=0), alternating.
# Single-instance law + quiet-GPU attestation per qwegine-run-measure.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PACK="${PACK:-$HOME/.qwg/models/Qwen3.8-27B-hq4g32/model.qwp}"
BIN="$REPO/.build/release/qwg"
OUT="${OUT:-/tmp/qw-repro/rows4-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
PROMPT="Explain the difference between DRAM bandwidth and L2 bandwidth in one paragraph."
GREP_RE='decode:|tok/s|prefill|pack load|wire|start\]|error|Error|Killed|trap'

pgrep -x qwg >/dev/null && { echo "REFUSED: a qwg instance is running"; exit 1; }
{
  echo "stamp $(date -u +%FT%TZ) host=$(hostname -s)"
  echo "available_GB $(vm_stat | awk -v t="$(sysctl -n hw.memsize)" '/page size of/{ps=$8} /Pages wired down/{w=$4} /Anonymous pages/{a=$3} /occupied by compressor/{c=$5} END{gsub(/\./,"",w);gsub(/\./,"",a);gsub(/\./,"",c); print int((t-(w+a+c)*ps)/1073741824)}')"
  echo "mlx_serve $(pgrep -x mlx-serve >/dev/null && echo UP || echo DOWN)"
  echo "pack $(basename "$(dirname "$PACK")")"
  echo "bin $(shasum -a 256 "$BIN" | cut -c1-16)"
} | tee "$OUT/stamp.txt"

for i in 1 2 3; do
  for arm in on off; do
    if [ "$arm" = off ]; then export QW_MLP_DOWN_ROWS4=0; else unset QW_MLP_DOWN_ROWS4; fi
    timeout 300 "$BIN" run "$PACK" --prompt "$PROMPT" --max-new 64 \
      --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 \
      | grep -E "$GREP_RE" > "$OUT/$arm-$i.log"
    printf '%s %s: %s\n' "$i" "$arm" "$(grep -E 'decode:' "$OUT/$arm-$i.log" | tail -1)"
  done
done
unset QW_MLP_DOWN_ROWS4
echo "logs: $OUT"
