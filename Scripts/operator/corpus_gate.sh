#!/bin/bash
# corpus_gate.sh — record or compare greedy outputs for a precision-changing
# change (split-KV attention, KV quant, etc.). Byte parity is unavailable by
# design there; these are the three signals specs/29 L5 requires.
#
#   ./corpus_gate.sh record <tag>     # write /tmp/qw-gate/<tag>.jsonl
#   ./corpus_gate.sh compare <a> <b>  # token-level diff + acceptance delta
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/.build/release/qwg"
PACK="${PACK:-$HOME/.qwg/models/Qwen3.8-Flash-Next-hq4g32/model.qwp}"
OUT="${GATE_DIR:-/tmp/qw-gate}"
MAXNEW="${MAXNEW:-24}"
mkdir -p "$OUT"
pgrep -x qwg >/dev/null && { echo "REFUSED: qwg already running"; exit 1; }

prompts=(
  "Explain in one sentence why a KV cache makes decoding faster."
  "List three differences between prefill and decode in a transformer."
  "Write a haiku about memory bandwidth."
  "What is the capital of France, and why does that matter for embeddings?"
  "Summarise the trade-off between context length and throughput."
)
# LONG=1 adds >1K-token prompts so KV-split paths (active at kvLen >= 1024)
# are actually exercised by the gate.
if [ "${LONG:-0}" = "1" ]; then
  long=$(python3 -c "print(('A tokenizer converts text into identifiers. It handles unicode, normalization, merging rules and vocabulary lookups. Byte level models keep every character representable. ')*50)")
  prompts+=("$long Summarise the text above in one sentence." "$long What is the topic?")
fi

case "${1:-}" in
record)
  tag="${2:?tag required}"
  : > "$OUT/$tag.jsonl"
  for i in "${!prompts[@]}"; do
    QW_FN_WIRE_SKIP=1 timeout 400 "$BIN" run "$PACK" --prompt "${prompts[$i]}" \
      --max-new "$MAXNEW" --mtp-drafts 1 --no-thinking --no-stream --verbose \
      > "$OUT/$tag-$i.log" 2>&1
    python3 - "$OUT/$tag.jsonl" "$i" "$OUT/$tag-$i.log" <<'PY'
import json, re, sys
path, idx, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = open(log).read()
text = t.strip().splitlines()[-1] if t.strip() else ""
acc = re.search(r'rounds=\d+ verified=\d+ accepted=(\d+)', t)
dec = re.search(r'^decode:\s+([0-9.]+) s\s+\(([0-9.]+) tok/s\)', t, re.M)
row = {"i": idx, "accepted": int(acc.group(1)) if acc else -1,
       "tok_s": float(dec.group(2)) if dec else -1.0, "text": text}
open(path, "a").write(json.dumps(row) + "\n")
PY
  done
  echo "recorded $(wc -l < "$OUT/$tag.jsonl") rows -> $OUT/$tag.jsonl"
  ;;
compare)
  a="${2:?first tag}"; b="${3:?second tag}"
  python3 - "$OUT/$a.jsonl" "$OUT/$b.jsonl" <<'PY'
import json, sys
A = [json.loads(l) for l in open(sys.argv[1])]
B = [json.loads(l) for l in open(sys.argv[2])]
bad = 0
for x, y in zip(A, B):
    same = x["text"] == y["text"]
    bad += 0 if same else 1
    print(f"p{x['i']}: text {'SAME' if same else 'DIFF'} | accepted {x['accepted']}->{y['accepted']}"
          f" | {x['tok_s']:.2f}->{y['tok_s']:.2f} tok/s")
print(f"\nsummary: {len(A) - bad}/{len(A)} prompts identical,"
      f" accepted {sum(r['accepted'] for r in A)} -> {sum(r['accepted'] for r in B)}")
sys.exit(0 if bad == 0 else 2)
PY
  ;;
*)
  echo "usage: $0 record <tag> | compare <a> <b>"; exit 64 ;;
esac
