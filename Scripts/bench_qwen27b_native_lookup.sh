#!/usr/bin/env bash
# bench_qwen27b_native_lookup.sh
# Empirical benchmark evaluating Native Training-Free Lookup (Context N-gram Drafter Tier-0)
# integrated into Qwen3.8-27B speculative execution on Apple Silicon (M5 Max).
#
# Usage:
#   ./Scripts/bench_qwen27b_native_lookup.sh [MODEL_PACK]
#
set -euo pipefail

MODEL_PACK="${1:-$HOME/.qwg/models/Qwen3.8-27B-hq4g32/model.qwp}"
BIN=".build/release/qwg"

if [ ! -f "$BIN" ]; then
    echo "==> Building release binary ($BIN)..."
    swift build -c release --product qwg
fi

if [ ! -f "$MODEL_PACK" ]; then
    echo "Error: Model pack not found at $MODEL_PACK"
    exit 1
fi

echo "================================================================================"
echo "    QWEN3.8-27B NATIVE LOOKUP SPECULATIVE CASCADE BENCHMARK (APPLE SILICON)   "
echo "================================================================================"
echo "Model: $MODEL_PACK"
echo "Date:  $(date)"
echo ""

PROMPTS=(
  "Write a Swift function that implements a circular buffer with enqueue and dequeue operations."
  "Repeat after me word for word: The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox"
  "Explain the difference between TCP and UDP networking protocols in bullet points:"
)

MAX_NEW=48

for idx in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$idx]}"
    echo "--------------------------------------------------------------------------------"
    echo "TEST SCENARIO $((idx+1)): \"${P:0:60}...\""
    echo "--------------------------------------------------------------------------------"

    echo ""
    echo ">>> [1/3] Baseline Serial Decode (No Speculation):"
    "$BIN" run "$MODEL_PACK" --prompt "$P" --max-new "$MAX_NEW" --no-thinking --verbose --mtp-drafts 0 --no-stream 2>&1 | grep -E "(tok gen|decode:|prefill:)"

    echo ""
    echo ">>> [2/3] Speculative MTP Baseline (Tier-1 MTP Only, No N-gram Lookup):"
    "$BIN" run "$MODEL_PACK" --prompt "$P" --max-new "$MAX_NEW" --no-thinking --verbose --mtp-drafts 2 --no-ngram-draft --no-stream 2>&1 | grep -E "(tok gen|decode:|prefill:|speculative cascade:|speculative rounds:)"

    echo ""
    echo ">>> [3/3] Native Lookup Cascade (Tier-0 N-gram + Tier-1 MTP Fallback):"
    "$BIN" run "$MODEL_PACK" --prompt "$P" --max-new "$MAX_NEW" --no-thinking --verbose --mtp-drafts 2 --ngram-draft --no-stream 2>&1 | grep -E "(tok gen|decode:|prefill:|speculative cascade:|speculative rounds:)"
    echo ""
done

echo "================================================================================"
echo "Benchmark completed successfully."
echo "================================================================================"
