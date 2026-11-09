#!/bin/bash
# int8_window.sh — INT8 program operator-window driver (exp1 cook, exp3
# acceptance, exp2 engine-level race, exp9 two-arm quiet-window race). Run
# from the worktree root during a GPU-quiet window with mlx-serve IDLE
# (loaded is fine; busy is not). Zero fabricated numbers: every stage logs
# raw tool output under out/<TAG>/.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

MASTER="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
REF="$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32"
HYB="$HOME/.qwg/models/Qwen3.8-Flash-Next-hq8hyb"
PY="${PY:-$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python}"
TAG="int8-win-$(date +%Y%m%d-%H%M%S)"
OUT="out/$TAG"; mkdir -p "$OUT"
STAGE="${1:-all}"   # cook | accept | race9 | all

stamp() { { echo "== stamp $(date -Iseconds)"; uptime; memory_pressure | grep "free percentage";
  pgrep -fl "qwg run" | grep -v "$$" && echo "WARN: concurrent qwg" || echo "no concurrent qwg";
  if pgrep -f "mlx-serve --serve" >/dev/null; then echo "mlx-serve: UP (port 11234)"; else
    echo "mlx-serve: DOWN — quiet-window attested by server absence"; fi; } >> "$1"; }

[ -s "$HYB/model.qwp" ] || { [ "$STAGE" = accept ] && { echo "REFUSING: $HYB missing — run stage cook first"; exit 1; }; }

if [ -z "${QW_SKIP_COOK:-}" ] && { [ "$STAGE" = all ] || [ "$STAGE" = cook ]; }; then
  # Available = total - wired - anonymous - compressor (Activity-Monitor law).
  # Apple Silicon page size is 16 KiB — NEVER hardcode 4096 (4x under-read
  # blocked a cook at real 112 GB free, operator catch 2026-10-17).
  FREE_GB=$(vm_stat | awk -v t="$(sysctl -n hw.memsize)" \
    '/page size of/{ps=$8} /Pages wired down/{w=$4} /Anonymous pages/{a=$3} /occupied by compressor/{c=$5} \
     END{gsub(/\./,"",w);gsub(/\./,"",a);gsub(/\./,"",c); print int((t-(w+a+c)*ps)/1073741824)}')
  SWAP_FREE_MB=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$9); print int($9)}')
  { [ "$FREE_GB" -ge 8 ] && [ "$SWAP_FREE_MB" -ge 1024 ]; } \
    || { echo "REFUSING cook: ${FREE_GB} GB available / ${SWAP_FREE_MB} MB swap free (need 8 GB + 1 GB)"; exit 1; }
  echo "[cook] $(date) free=${FREE_GB}GB -> $OUT/cook.log"
  "$PY" Scripts/quantize/pack.py --model "$MASTER" \
    --hybrid-hq8-from "$REF" --reuse-ngram-from "$REF" --out "$HYB" > "$OUT/cook.log" 2>&1 \
    || { echo "COOK FAILED — see $OUT/cook.log"; exit 1; }
  # Mode fall-through guard: pack.py exits 0 on sanity-only mode (bit us once).
  [ -s "$HYB/model.qwp" ] || { echo "COOK PRODUCED NO PACK — see $OUT/cook.log"; exit 1; }
  ALLOW=(); while IFS= read -r tname || [ -n "$tname" ]; do ALLOW+=(--allow-requantized "$tname"); done < <(
    "$PY" -c "import sys; sys.path.insert(0, 'Scripts/quantize')
from verify import read_header
t, _ = read_header('$HYB/model.qwp')
print('\n'.join(k for k, v in t['tensors'].items() if 'hq8_off' in v))")
  echo "[verify] skipping ${#ALLOW[@]} / 2 requantized HQ8 tensors"
  "$PY" Scripts/quantize/verify.py "$HYB" --reference "$REF" "${ALLOW[@]}" > "$OUT/verify.log" 2>&1 \
    || { echo "VERIFY FAILED — see $OUT/verify.log"; exit 1; }
  echo "[cook+verify OK] artifacts: $(du -sh "$HYB" | cut -f1)"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = accept ]; then
  swift build -c release > "$OUT/build.log" 2>&1 || { echo "RELEASE BUILD FAILED"; exit 1; }
  PROMPT="Explain the difference between TCP and UDP networking protocols in bullet points:"
  stamp "$OUT/accept.log"
  LIVE=$(curl -s --max-time 3 localhost:11234/v1/models 2>/dev/null | \
    python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
  export MODEL="${LIVE:-ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}" QW_WORKLOAD_DRAIN=1
  echo "[accept] MODEL=$MODEL" >> "$OUT/accept.log"
  GREP='decode:|tok/s|prefill|wire|start]|error|Error|Killed|trap'
  QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1 Scripts/phase4/run_flashnext_experiment.sh "$TAG-f0" \
    env GREP_RE="$GREP" bash -c 'P=$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32
    echo "=== f0-control"; .build/release/qwg run "$P/model.qwp" --prompt "Explain the difference between TCP and UDP networking protocols in bullet points:" --max-new 128 --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 | grep -E "$GREP_RE"' \
    >> "$OUT/accept.log" 2>&1 || echo "F0 GATE FAILED — inspect before candidate arms" >> "$OUT/accept.log"
  for i in 1 2 3; do
    for arm in ctrl cand; do
      case $arm in
        ctrl) PACKPATH='$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32'; W8ENV="";;
        cand) PACKPATH='$HOME/.qwg/models/Qwen3.8-Flash-Next-hq8hyb'; W8ENV="QW_FN_DENSE_W8=1";;
      esac
      QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1 Scripts/phase4/run_flashnext_experiment.sh "$TAG-$arm-$i" \
        env GREP_RE="$GREP" bash -c "P=$PACKPATH
    echo \"=== arm=$arm iter=$i\"; env $W8ENV .build/release/qwg run \"\$P/model.qwp\" --prompt \"$PROMPT\" --max-new 128 --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 | grep -E \"\$GREP_RE\"" \
        >> "$OUT/accept.log" 2>&1 || echo "RUN FAILED arm=$arm iter=$i" >> "$OUT/accept.log"
    done
  done
  echo "DONE $TAG — paste ledger rows from:"
  grep -E "arm=|f0|decode:|prefill|tok/s|wire|start]" "$OUT/accept.log" | tail -40
fi

if [ "$STAGE" = race9 ]; then
  # exp9 — one quiet window, three verdicts:
    #   (1) HQ8 dense A/B  (exp8 needed a quiet-window verdict; pack REF vs HYB
    #       with QW_FN_DENSE_W8=1) — phase A of the action plan.
    #   (2) HC M-batch re-race (commit-message verdict "neutral prefill, -8%
    #       verify-row decode" had no artifacts; QW_FN_HC_M4=0 vs 1 on REF).
    #   (3) Cold vs warm prefill split (exp8 addendum: the 25 s short-prompt
    #       prefill is pack cold first-touch, not GPU). Each experiment runs
    #       the SAME qwg command twice back-to-back: pass-1 cold, pass-2 warm
    #       (page cache warm from pass-1 in-process footprint).
    # Every experiment exports QW_FN_PHASES=1 so door-level span attribution
    # is captured regardless of the cadence verdict. QW_FN_WIRE_SKIP=1 matches
    # the exp8 protocol (bench-determinism switch, commit 9640d41).
    swift build -c release > "$OUT/build.log" 2>&1 || { echo "RELEASE BUILD FAILED"; exit 1; }
    PROMPT="Explain the difference between TCP and UDP networking protocols in bullet points:"
    stamp "$OUT/race9.log"
    LIVE=$(curl -s --max-time 3 localhost:11234/v1/models 2>/dev/null | \
        python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
      if [ -n "$LIVE" ]; then
        export MODEL="$LIVE" QW_WORKLOAD_DRAIN=1
        echo "[race9] MODEL=$MODEL (server-controlled guard protocol)" >> "$OUT/race9.log"
      else
        # mlx-serve control plane unreachable: a server-DOWN quiet window is
        # STRONGER than a drained one (no co-tenant can become resident), but
        # the guard cannot verify it — fall back to operator-reserved mode
        # (workload_guard races this as a manually attested quiet window).
        export QW_PHASE4_WORKLOAD_CONTROL=operator-reserved
            unset QW_WORKLOAD_DRAIN
            export MODEL="operator-attested-quiet-window"  # driver asserts MODEL exists; unused in this mode
        STAMP=$(date -Iseconds); NOGPU=$(pgrep -f "mlx-serve --serve" | wc -l | tr -d ' ')
        { echo "[race9] ATTESTED QUIET WINDOW $STAMP — mlx-serve DOWN (pgrep mlx-serve: $NOGPU procs)."; \
          echo "[race9] Timing is operator-attested; no residency monitor runs (guard limitation, not a contamination signal)."; } >> "$OUT/race9.log"
      fi
      echo "[race9] MODEL=${MODEL:-none} mode=${QW_PHASE4_WORKLOAD_CONTROL:-server-controlled}" >> "$OUT/race9.log"
    LOG_DIR="${QW_PHASE4_LOG_DIR:-/tmp/qw-repro}"
    GREP='decode:|tok/s|prefill|wire|start]|error|Error|Killed|trap'
      RM='=== |decode:|prefill:|tok/s|wire|start]|error|Error|Killed|trap'
      # Sequential-only law (operator, 2026-10-17): one engine instance holds
      # ~85 GB; two resident packs jetsam. Refuse to start an experiment while
      # another engine process exists, and validate each pass produced output
      # (exp9 lost 3/12 experiments to silent zero-output runs) — one retry.
      ensure_single_engine() {
        N=$(ps -axo comm | grep -c '[q]wg$')
        [ "$N" -le 1 ] || { echo "REFUSING: $N engine instances resident (max 1 — 85 GB each)"; return 1; }
      }
      run_pass_checked() { # $1 tag $2 pass $3 packpath $4 envstr
        local TAGPASS="$1" PASS="$2" PACKPATH="$3" W8ENV="$4" LOGP="$LOG_DIR/flashnext-$TAGPASS.log"
        for try in 1 2; do
          ensure_single_engine || return 1
          QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1 Scripts/phase4/run_flashnext_experiment.sh "$TAGPASS" \
            env GREP_RE="$RM" bash -c "P=$PACKPATH
        echo '=== $PASS'
        env $W8ENV .build/release/qwg run \"\$P/model.qwp\" --prompt \"$PROMPT\" --max-new 128 --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 | grep -E \"\$GREP_RE\"" \
            >> "$OUT/race9.log" 2>&1
          if grep -q "prefill:" "$LOG_DIR/flashnext-$TAGPASS.log" 2>/dev/null; then return 0; fi
          echo "PASS EMPTY ($TAGPASS try=$try) — flagged INVALID, $( [ $try = 1 ] && echo retrying || echo giving up )" >> "$OUT/race9.log"
        done
        return 1
      }
    for i in 1 2 3; do
      for arm in hcm0 hcm1; do
        case $arm in hcm0) V=0;; hcm1) V=1;; esac
        run_pass_checked "$TAG-hc$V-$i"     "raceA arm=hc_m$V iter=$i COLD" "$REF" "QW_FN_HC_M4=$V" \
          || echo "RUN FAILED raceA hc_m$V-$i COLD" >> "$OUT/race9.log"
        run_pass_checked "$TAG-hc$V-$i-warm" "raceA arm=hc_m$V iter=$i WARM" "$REF" "QW_FN_HC_M4=$V" \
          || echo "RUN FAILED raceA hc_m$V-$i WARM" >> "$OUT/race9.log"
      done
    done
    for i in 1 2 3; do
      for arm in w8off w8on; do
        case $arm in
          w8off) PACKPATH="$REF";  W8ENV="" ;;
          w8on)  PACKPATH="$HYB";  W8ENV="QW_FN_DENSE_W8=1" ;;
        esac
        # Interleave off/on within each ROW so drift lands on both arms.
        run_pass_checked "$TAG-$arm-$i"      "raceB arm=$arm iter=$i COLD" "$PACKPATH" "$W8ENV" \
          || echo "RUN FAILED raceB $arm-$i COLD" >> "$OUT/race9.log"
        run_pass_checked "$TAG-$arm-$i-warm" "raceB arm=$arm iter=$i WARM" "$PACKPATH" "$W8ENV" \
          || echo "RUN FAILED raceB $arm-$i WARM" >> "$OUT/race9.log"
      done
    done
    echo "DONE race9 $TAG — extract with:"
    grep -E "^=== |decode:|prefill:|tok/s|wire|start]" "$OUT/race9.log" | tail -80
  fi
