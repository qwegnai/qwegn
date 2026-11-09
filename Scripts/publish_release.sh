#!/usr/bin/env bash
# publish_release.sh — cut the public release from the current master tip.
#
# The public repo (github.com/qwegnai/qwegn) carries one commit per release:
# a curated snapshot of master, never the internal history, never a force push.
# This script is the whole procedure; run it from anywhere, once per release.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUB="${QWEGN_PUB:-$HOME/Developer/ai/qwegine-publish}"
REMOTE_URL="git@github.com:qwegnai/qwegn.git"

VER=$(grep -oE 'current: String = "[0-9.]+"' "$SRC/Sources/Foundation/QWCore/QWVersion.swift" | grep -oE '[0-9.]+')
[ -n "$VER" ] || { echo "FATAL: no version parsed"; exit 1; }
echo "== releasing v$VER from $(git -C "$SRC" rev-parse --short master)"

# 1. The full gate on the source tree, on a quiet machine (the perf law applies).
( cd "$SRC" && git status --porcelain | grep -q . ) && { echo "FATAL: source tree dirty"; exit 1; }
( cd "$SRC" && make ci )

# 2. Export master's tracked tree only (no untracked cruft, no private docs)
#    and drop the two internal scratch directories excluded from public.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git -C "$SRC" archive master | tar -x -C "$STAGE"
rm -rf "$STAGE/Scripts/spike/nax-int8" "$STAGE/Scripts/oracle/golden"

# 3. Public tree: never rewrite published history — add a commit on top.
cd "$PUB"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE_URL"
git fetch origin
git checkout -q main
git reset -q --hard origin/main
rsync -a --delete --exclude .git "$STAGE/" "$PUB/"
# Belt for the gate: personal paths and private docs must not exist here.
if grep -rq "$HOME" --include="*.swift" --include="*.md" --include="*.json" . 2>/dev/null; then
  echo "FATAL: personal paths in public tree"; exit 1; fi
[ -e docs/open-source-standards.md ] && { echo "FATAL: private standards doc staged"; exit 1; }

git add -A
git diff --cached --quiet && { echo "Nothing to release (tree identical to origin/main)"; exit 0; }
git commit -q -m "Qwegn v$VER"
git tag -a "v$VER" -m "v$VER"

# 4. Publish. The tag triggers .github/workflows/release.yml, which builds the
#    downloadable artifact and attaches it to a GitHub Release.
git push origin main "v$VER"
echo "== released: https://github.com/qwegnai/qwegn/releases/tag/v$VER (artifact building)"
