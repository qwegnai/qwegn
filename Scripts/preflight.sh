#!/usr/bin/env bash
# Deterministic session/worktree/UMA/quiet-GPU preflight summary (Code Mode).
set -euo pipefail

REQUIRE_QUIET_GPU=0
if [ "${1:-}" = "--require-quiet-gpu" ]; then
  REQUIRE_QUIET_GPU=1
elif [ -n "${1:-}" ]; then
  echo "Usage: $0 [--require-quiet-gpu]" >&2
  exit 64
fi

PWD_NOW="$(pwd)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY_COUNT="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
WORKTREE_COUNT="$(git worktree list 2>/dev/null | wc -l | tr -d ' ')"

if [[ "$PWD_NOW" == *"/.worktrees/"* ]]; then
  CHECKOUT_KIND="worktree"
else
  CHECKOUT_KIND="primary (WARN: create .worktrees/<task> per Law 2)"
fi

STRAY_LINKS=()
if [[ "$PWD_NOW" == *"/.worktrees/"* ]]; then
  for p in .venv-oracle Scripts/spike/out out/vision_decision_oracle; do
    if [ -L "$p" ]; then
      STRAY_LINKS+=("$p")
    fi
  done
fi

PAGE_SIZE="$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)"
UMA_FREE_GIB="$(vm_stat 2>/dev/null | awk -v ps="$PAGE_SIZE" '
  /Pages free:/ {f=$NF}
  /Pages speculative:/ {s=$NF}
  /Pages inactive:/ {i=$NF}
  /Pages purgeable:/ {p=$NF}
  END {
    gsub(/\./, "", f); gsub(/\./, "", s); gsub(/\./, "", i); gsub(/\./, "", p);
    printf "%.1f", (f + s + i + p) * ps / (1024 * 1024 * 1024)
  }' || echo "unknown")"

CORUNNERS="$(pgrep -fl '(^|/)(qwg|swift-test|llmprobe|mlx-serve|ollama)(\s|$)|Qwegn\.app' 2>/dev/null | grep -v 'preflight\.sh' || true)"
PORT_8080="$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $1 ":" $2}' | tr '\n' ' ' || true)"

echo "preflight:"
echo "  cwd:        $PWD_NOW [$CHECKOUT_KIND]"
echo "  git:        branch=$BRANCH head=$HEAD_SHA dirty=$DIRTY_COUNT worktrees=$WORKTREE_COUNT"
if [ ${#STRAY_LINKS[@]} -eq 0 ]; then
  echo "  symlinks:   clean"
else
  echo "  symlinks:   WARN (${STRAY_LINKS[*]})"
fi
echo "  uma_free:   ${UMA_FREE_GIB} GiB (free+spec+inactive+purgeable)"
if [ -z "$CORUNNERS" ]; then
  echo "  gpu_idle:   yes (0 active co-runners)"
else
  echo "  gpu_idle:   NO"
  echo "$CORUNNERS" | sed 's/^/    /'
fi
echo "  port_8080:  ${PORT_8080:-free}"

if [ "$REQUIRE_QUIET_GPU" -eq 1 ] && [ -n "$CORUNNERS" ]; then
  echo "REFUSING: active GPU co-runners detected" >&2
  exit 2
fi
