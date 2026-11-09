#!/bin/bash
# arm_window.sh - generic paired A/B window (P14).
# Usage: ARM_A=QW_X=1 ARM_B=QW_X=0 PACK=<model.qwp> RUN_ARGS='--max-new 64' \
#        bash Scripts/operator/arm_window.sh <tag>
# Alternates A/B/B/A per rep, stamps the run, and prints one line per arm.
set -u
TAG=${1:?tag}
REPO=$(cd $(dirname $0)/../.. && pwd)
BIN=${BIN:-$REPO/.build/release/qwg}
PACK=${PACK:-$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32/model.qwp}
RUN_ARGS=${RUN_ARGS:---max-new 64 --no-thinking --no-stream}
REPS=${REPS:-3}
OUT=${OUT:-/tmp/qw-repro/$TAG-$(date +%Y%m%d-%H%M%S)}
mkdir -p $OUT
PROMPT=${PROMPT:-$(cat /tmp/qw-prompt.txt 2>/dev/null || echo 'Explain DRAM versus L2 bandwidth in one paragraph.')}
GREP_RE=${GREP_RE:-'prefill:|decode:|fn chunk drains|phase-probe|error|Error|Killed|trap'}
pgrep -x qwg >/dev/null && { echo REFUSED; exit 1; }
echo stamp $(date -u +%FT%TZ) > $OUT/stamp.txt
echo mlx_serve $(pgrep -x mlx-serve >/dev/null && echo UP || echo DOWN) >> $OUT/stamp.txt
echo bin $(shasum -a 256 $BIN | cut -c1-16) >> $OUT/stamp.txt
cat $OUT/stamp.txt
run_arm() {
  local arm=$1 rep=$2 envval=${3:-}
  if [ -n "${envval:-}" ]; then export "$envval"; fi
  timeout 600 $BIN run $PACK --prompt "$PROMPT" $RUN_ARGS --verbose 2>&1 | grep -E "$GREP_RE" > "$OUT/$arm-$rep.log"
  printf '%s %s: %s\n' $rep $arm "$(grep -E 'prefill:|decode:' "$OUT/$arm-$rep.log" | paste -sd' ' -)" 
  if [ -n "${envval:-}" ]; then unset "${envval%%=*}"; fi
}
for r in $(seq 1 $REPS); do
  run_arm A $r ${ARM_A:-}
  run_arm B $r ${ARM_B:-}
done
echo logs: $OUT
