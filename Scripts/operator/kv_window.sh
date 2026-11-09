#!/bin/bash
# kv_window.sh — FN KV-mode + prefix-cache operator window (driver mirrors
# int8_window.sh's guarded protocol via cadence_lib.sh). Run from the
# worktree root during a GPU-quiet window; single engine instance at a time
# (sequential-only law).mlx-serve UP -> server-controlled driver protocol;
# DOWN -> operator-reserved attested quiet (same fallback as race9).
#   STAGE=kvmode   QW_FN_KV_MODE=f16/q8/q4 on the HQ4 pack, interleave x2.
#   STAGE=prefix   serve process: same long prompt twice, QW_FN_PREFIX_CACHE=1
#                  + QW_FN_KV_MODE=q8; run 2 must log "[prefix] HIT".
# Zero fabricated numbers: raw tool output under out/<TAG>/.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source Scripts/operator/cadence_lib.sh

REF="$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32"
TAG="kv-win-$(date +%Y%m%d-%H%M%S)"
OUT="out/$TAG"; mkdir -p "$OUT"
STAGE="${1:-all}"   # kvmode | prefix | all

swift build -c release > "$OUT/build.log" 2>&1 || { echo "RELEASE BUILD FAILED"; exit 1; }
PROMPTF="$OUT/prompt.txt"
cadence_longprompt "$PROMPTF"
echo "[kv-window] tag=$TAG prompt_bytes=$(wc -c < "$PROMPTF" | tr -d ' ')"

if [ "$STAGE" = all ] || [ "$STAGE" = kvmode ]; then
  live=$(curl -s --max-time 3 localhost:11234/v1/models 2>/dev/null || true)
  if [ -n "$live" ]; then
    MODEL=$(printf '%s' "$live" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit)
    export MODEL QW_WORKLOAD_DRAIN=1
    echo "[kvmode] server-controlled guard, MODEL=$MODEL" >> "$OUT/race.log"
  else
    export QW_PHASE4_WORKLOAD_CONTROL=operator-reserved
    unset QW_WORKLOAD_DRAIN
    export MODEL="operator-attested-quiet-window"
    cadence_stamp "$OUT/race.log"
    { echo "[kvmode] ATTESTED QUIET WINDOW $(date -Iseconds) — mlx-serve DOWN."; } >> "$OUT/race.log"
  fi
  for i in 1 2; do
    for mode in f16 q8 q4; do
      cadence_run_pass "$TAG-m$mode-$i" "kvmode arm=$mode iter=$i" "$REF" \
        "QW_FN_KV_MODE=$mode" "$PROMPTF" 128 \
        || echo "RUN FAILED kvmode $mode-$i" >> "$OUT/race.log"
    done
  done
  # Suspicion-law arms (9782b4e): re-settle the two PROVISIONAL verdicts
  # (HC M4 and HQ8 dense) in the SAME uninstrumented window.
  for i in 1 2; do
    for arm in hcm0 hcm1; do
      case $arm in hcm0) V=0;; hcm1) V=1;; esac
      cadence_run_pass "$TAG-hc$V-$i" "hc arm=hc_m$V iter=$i" "$REF" \
        "QW_FN_HC_M4=$V" "$PROMPTF" 128 \
        || echo "RUN FAILED hc$V-$i" >> "$OUT/race.log"
    done
    for arm in w8off w8on; do
      case $arm in w8off) PACK="$REF";  W8E="" ;; w8on) PACK="$HOME/.qwg/models/Qwen3.8-Flash-Next-hq8hyb"; W8E="QW_FN_DENSE_W8=1";; esac
      cadence_run_pass "$TAG-$arm-$i" \
        "w8 arm=$arm iter=$i" "$PACK" "$W8E" "$PROMPTF" 128 \
        || echo "RUN FAILED $arm-$i" >> "$OUT/race.log"
    done
  done
  echo "DONE kvmode $TAG — extract with:"
  grep -E "^=== |decode:|prefill:|wire|start]" "$OUT/race.log" | tail -40
fi

if [ "$STAGE" = all ] || [ "$STAGE" = prefix ]; then
  echo "[prefix] starting serve process (QW_FN_KV_MODE=q8 QW_FN_PREFIX_CACHE=1)"
  QW_FN_KV_MODE=q8 QW_FN_PREFIX_CACHE=1 QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1 \
    .build/release/qwg serve "$REF/model.qwp" --port 12790 --host 127.0.0.1 --model-name fn \
    > "$OUT/serve.log" 2>&1 &
  SERVE_PID=$!
  echo "$SERVE_PID" > "$OUT/serve.pid"
  up=0
  for _ in $(seq 1 120); do
    curl -s --max-time 2 localhost:12790/v1/models >/dev/null 2>&1 && { up=1; break; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "SERVE FAILED — see $OUT/serve.log"; kill "$SERVE_PID" 2>/dev/null; exit 1; }
  for n in 1 2; do
    t0=$(date +%s)
    body=$(python3 -c '
import json, sys
prompt = open(sys.argv[1]).read()
print(json.dumps({
    "model": "fn", "prompt": prompt,
    "max_tokens": 128, "temperature": 0, "stream": False}))' "$PROMPTF")
    code=$(printf '%s' "$body" | curl -s --max-time 900 -o "$OUT/gen$n.json" \
      -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
      --data-binary @- localhost:12790/v1/completions)
    t1=$(date +%s)
    { echo "=== prefix gen=$n http=$code wall=$((t1 - t0))s"; \
      python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    t = d.get("usage", {})
    print("completion_tokens:", t.get("completion_tokens"))
except Exception as e:
    print("parse error:", e)' "$OUT/gen$n.json"; } >> "$OUT/race.log"
  done
  kill "$SERVE_PID" 2>/dev/null; sleep 3
  echo "DONE prefix $TAG — check run2 for '[prefix] HIT' in:"
  grep -E "prefix] HIT|prefix] MISS|RESTORE FAILED|=== prefix" "$OUT/serve.log" "$OUT/race.log" | tail -10
fi