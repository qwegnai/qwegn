#!/usr/bin/env bash
# CI: build + codegen freshness + test + CLI smoke. Perf benches run separately
# (BENCH_PROTOCOL.md: one authoritative bench at a time, no co-runners).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== build"
swift build 2>&1 | tail -3
echo "== style gate (Swift + Metal + Python)"
python3 Scripts/lint_style.py Sources Tests Scripts || { echo "FAIL: style gate"; exit 1; }
echo "== kernel reachability (metallib vs dispatch sites)"
python3 Scripts/test_kernel_inventory.py || { echo "FAIL: kernel inventory tests"; exit 1; }
./Scripts/build_metallib.sh > /tmp/qwegine_metallib.log 2>&1 || { tail -20 /tmp/qwegine_metallib.log; echo "FAIL: metallib build"; exit 1; }
python3 Scripts/lint_kernels.py || { echo "FAIL: kernel reachability gate"; exit 1; }
python3 Scripts/check_basis.py || { echo "FAIL: basis gate"; exit 1; }
echo "== geometry defaults (a QWModelGeometry parameter carries no default)"
python3 Scripts/lint_geometry_defaults.py Sources || { echo "FAIL: geometry-default gate"; exit 1; }
echo "== paired performance harness tests (no model or GPU)"
python3 -m unittest discover -s Scripts -p 'test_paired*perf.py' || { echo "FAIL: paired performance harness"; exit 1; }
echo "== test"
# THE EXIT-CODE LAW (C22): `swift test | tail` leaks TAIL's exit code (0)
# through pipefail when the upstream swift exits nonzero via SIGPIPE-style
# early close — a red suite could pass CI. Capture PIPESTATUS[0] explicitly.
set +e
swift test --no-parallel > /tmp/qwegine_ci_test.log 2>&1
tstatus=$?
set -e
tail -5 /tmp/qwegine_ci_test.log
if [ "$tstatus" -ne 0 ]; then echo "FAIL: swift test exit $tstatus"; exit "$tstatus"; fi
echo "== cli smoke"
# THE SMOKE LAW (C22): exercise a REAL verb on the fixture pack — a bare
# `qwg` root exits 64 by ArgumentParser contract (no default subcommand),
# so the old skeleton-print smoke is meaningless.
./.build/debug/qwg inspect Tests/QWQuantTests/fixtures/selfpack.qwp | grep -q '"geometryGuard" : "ok"' || { echo "FAIL: inspect smoke"; exit 1; }
./.build/debug/qwg inspect Tests/QWQuantTests/fixtures/selfpack.qwp --dequant w_test --out /tmp/ci_dq.npy | grep -q wrote || { echo "FAIL: dequant smoke"; exit 1; }
echo "== phase 2: the bench's winner map"
# The bench must produce a winner map the ENGINE can read. The map's schema
# is checked against the engine's constant AND against the pin, so a schema
# change that moves only one of them goes red. The bench runs in RELEASE:
# the debug build takes minutes, the release build takes 11.7 s.
BENCH_OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$BENCH_OUT_DIR"' EXIT
swift build -c release --product qwg-bench-kernels 2>&1 | tail -1
./.build/release/qwg-bench-kernels --out "$BENCH_OUT_DIR" >/dev/null
python3 Scripts/check_bench_report.py "$BENCH_OUT_DIR/bench_kernels_report.json" \
  || { echo "FAIL: the winner map"; exit 1; }
echo "ci ok"
