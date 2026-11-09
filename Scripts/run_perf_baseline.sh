#!/usr/bin/env bash
# run_perf_baseline.sh — one-by-one model perf sweep with co-runner watch.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/qw_perf_$(date +%H%M%S)
mkdir -p "$OUT"
CO=2816   # mlx-serve co-runner

cor() { ps -o rss= -p $CO 2>/dev/null | awk '{printf "%.1f", $1/1048576}'; }
mem() { memory_pressure 2>/dev/null | grep free | grep -o '[0-9.]*%' | tr -d '%'; }

for M in "Qwen3.8-27B-hq4g32" "Qwen3.8-Flash-Next-hq4g32"; do
  TAG="${M//./_}"
  echo "===== $M start $(date +%H:%M:%S) corunner=$(cor)GB freemem=$(mem)%"
  ./Scripts/bench_qwen27b_native_lookup.sh "$HOME/.qwg/models/$M/model.qwp" > "$OUT/$TAG.log" 2>&1
  RC=$?
  echo "===== $M done  $(date +%H:%M:%S) exit=$RC corunner=$(cor)GB freemem=$(mem)%"
done
echo "OUT=$OUT"
