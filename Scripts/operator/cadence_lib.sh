#!/bin/bash
# cadence_lib.sh — shared helpers for Scripts/operator/*_window.sh drivers:
# evidence stamps (with mlx-serve UP/DOWN state), the single-engine-instance
# gate (sequential-only law, 2026-10-17: one engine pack ≈ 85 GB), guarded
# per-pass runner with output validation + one retry, and the pinned long
# prompt used by KV/prefix races. Sourced, not executed.

# Long prompt for KV-mode and prefix-cache arms: ~2K tokens of fixed text.
cadence_longprompt() { # $1 = target file
  python3 - "$1" <<'PYEOF'
import sys
line = ("Explain the difference between TCP and UDP networking protocols in "
        "bullet points, covering reliability, ordering, flow control, "
        "congestion behaviour and typical use cases, and then compare each "
        "property with QUIC fallback semantics. ")
with open(sys.argv[1], "w") as f:
    f.write(line * 120)   # ~1,900-2,100 tokens depending on the tokenizer
PYEOF
}

cadence_stamp() { # $1 = log file
  { echo "== stamp $(date -Iseconds)"; uptime; memory_pressure | grep "free percentage";
    pgrep -fl "qwg run" | grep -v "$$" && echo "WARN: concurrent qwg" || echo "no concurrent qwg";
    if pgrep -f "mlx-serve --serve" >/dev/null; then echo "mlx-serve: UP (port 11234)"; else
      echo "mlx-serve: DOWN — quiet-window attested by server absence"; fi; } >> "$1"
}

# Refuse to start an experiment while another engine process exists.
cadence_ensure_single_engine() {
  local n
  n=$(ps -axo comm | grep -c '[q]wg$')
  [ "$n" -le 1 ] || { echo "REFUSING: $n engine instances resident (max 1 — ~85 GB each)"; return 1; }
}

# One guarded qwg pass: tag, label, pack path, env string; validates the
# per-experiment log produced engine output, retries once, flags empty runs.
cadence_run_pass() { # $1 tag  $2 label  $3 packpath  $4 envstr  $5 promptfile  $6 maxnew
  local TAGPASS="$1" PASS="$2" PACKPATH="$3" W8ENV="$4" PROMPTF="$5" MAXNEW="$6"
  local LOG_DIR="${QW_PHASE4_LOG_DIR:-/tmp/qw-repro}"
  local LOGP="$LOG_DIR/flashnext-$TAGPASS.log"
  local RM='=== |decode:|prefill:|tok/s|wire|start]|error|Error|Killed|trap|prefix'
  local try
  for try in 1 2; do
    cadence_ensure_single_engine || return 1
    QW_FN_PHASES=1 QW_FN_WIRE_SKIP=1 Scripts/phase4/run_flashnext_experiment.sh "$TAGPASS" \
      env GREP_RE="$RM" bash -c "P=$PACKPATH
        echo '=== $PASS'
        env $W8ENV .build/release/qwg run \"\$P/model.qwp\" --prompt \"\$(cat $PROMPTF)\" --max-new $MAXNEW --no-thinking --no-stream --no-ngram-draft --verbose 2>&1 | grep -E \"\$GREP_RE\"" \
      >> "$OUT/race.log" 2>&1
    if grep -q "prefill:" "$LOGP" 2>/dev/null; then return 0; fi
    echo "PASS EMPTY ($TAGPASS try=$try) — flagged INVALID, $( [ "$try" = 1 ] && echo retrying || echo giving up )" >> "$OUT/race.log"
  done
  return 1
}