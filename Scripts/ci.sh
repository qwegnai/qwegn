#!/usr/bin/env bash
# CI: build + codegen freshness + test + CLI smoke. Perf benches run separately
# (BENCH_PROTOCOL.md: one authoritative bench at a time, no co-runners).
set -euo pipefail
cd "$(dirname "$0")/.."
CI_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwegine_ci.XXXXXX")"
trap 'rm -rf "$CI_TMP_DIR"' EXIT
echo "== build"
set +e
swift build > "$CI_TMP_DIR/build.log" 2>&1
bstatus=$?
set -e
if [ "$bstatus" -ne 0 ]; then
  echo "FAIL: swift build exit $bstatus"
  cat "$CI_TMP_DIR/build.log"
  exit "$bstatus"
fi
tail -3 "$CI_TMP_DIR/build.log"
echo "== style gate (Swift + Metal + Python)"
python3 Scripts/lint_style.py Sources Tests Scripts || { echo "FAIL: style gate"; exit 1; }
echo "== skill hygiene & SDLC script gates"
python3 Scripts/lint_skills.py || { echo "FAIL: skill hygiene gate"; exit 1; }
python3 Scripts/test_sdlc_scripts.py || { echo "FAIL: SDLC script tests"; exit 1; }
echo "== kernel reachability (metallib vs dispatch sites)"
python3 Scripts/test_kernel_inventory.py || { echo "FAIL: kernel inventory tests"; exit 1; }
./Scripts/build_metallib.sh > "$CI_TMP_DIR/metallib.log" 2>&1 || { cat "$CI_TMP_DIR/metallib.log"; echo "FAIL: metallib build"; exit 1; }
python3 Scripts/check_metallib_target.py || { echo "FAIL: metallib target gate"; exit 1; }
python3 Scripts/lint_kernels.py || { echo "FAIL: kernel reachability gate"; exit 1; }
python3 Scripts/check_basis.py || { echo "FAIL: basis gate"; exit 1; }
python3 Scripts/lint_metal_constant_luts.py || { echo "FAIL: metal constant LUT gate"; exit 1; }
echo "== geometry defaults, capability seams & package DAG gates"
python3 Scripts/lint_geometry_defaults.py Sources || { echo "FAIL: geometry-default gate"; exit 1; }
python3 Scripts/lint_capability_seams.py || { echo "FAIL: capability-seam gate"; exit 1; }
python3 Scripts/lint_package_dag.py || { echo "FAIL: package DAG gate"; exit 1; }
echo "== paired performance harness tests (no model or GPU)"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "skipping paired performance harness tests on GitHub Actions CI"
else
  python3 -m unittest discover -s Scripts -p 'test_paired*perf.py' || { echo "FAIL: paired performance harness"; exit 1; }
fi
echo "== test"
# THE EXIT-CODE LAW (C22): `swift test | tail` leaks TAIL's exit code (0)
# through pipefail when the upstream swift exits nonzero via SIGPIPE-style
# early close — a red suite could pass CI. Capture exit status explicitly.
set +e
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "running host/CLI/server/quant/token/UI suites on GitHub Actions (skipping bare-metal Metal & golden suites)"
  swift test --no-parallel \
    --filter QWCoreTests \
    --filter QWQuantTests \
    --filter QWTokenTests \
    --filter QWDecisionTests \
    --filter QWCliTests \
    --filter QWServerTests \
    --filter QWMenuBarTests \
    > "$CI_TMP_DIR/test.log" 2>&1
  tstatus=$?
else
  swift test --no-parallel > "$CI_TMP_DIR/test.log" 2>&1
  tstatus=$?
fi
set -e
if [ "$tstatus" -ne 0 ]; then
  echo "FAIL: swift test exit $tstatus"
  cat "$CI_TMP_DIR/test.log"
  exit "$tstatus"
fi
tail -5 "$CI_TMP_DIR/test.log"
echo "== cli smoke"
# THE SMOKE LAW (C22): exercise a REAL verb on the fixture pack — a bare
# `qwg` root exits 64 by ArgumentParser contract (no default subcommand),
# so the old skeleton-print smoke is meaningless.
./.build/debug/qwg inspect Tests/QWQuantTests/fixtures/selfpack.qwp | grep -q '"geometryGuard" : "ok"' || { echo "FAIL: inspect smoke"; exit 1; }
./.build/debug/qwg inspect Tests/QWQuantTests/fixtures/selfpack.qwp --dequant w_test --out "$CI_TMP_DIR/ci_dq.npy" | grep -q wrote || { echo "FAIL: dequant smoke"; exit 1; }
echo "== phase 2: the bench's winner map"
# The bench must produce a winner map the ENGINE can read. The map's schema
# is checked against the engine's constant AND against the pin, so a schema
# change that moves only one of them goes red. The bench runs in RELEASE:
# the debug build takes minutes, the release build takes 11.7 s.
set +e
swift build -c release --product qwg-bench-kernels > "$CI_TMP_DIR/bench_build.log" 2>&1
bbstatus=$?
set -e
if [ "$bbstatus" -ne 0 ]; then
  echo "FAIL: swift build qwg-bench-kernels exit $bbstatus"
  cat "$CI_TMP_DIR/bench_build.log"
  exit "$bbstatus"
fi
tail -1 "$CI_TMP_DIR/bench_build.log"
python3 Scripts/check_swifterror_abi.py ./.build/release/qwg-bench-kernels || { echo "FAIL: swifterror ABI gate"; exit 1; }
set +e
swift build -c release --product qwg > "$CI_TMP_DIR/qwg_release.log" 2>&1
qbstatus=$?
set -e
if [ "$qbstatus" -ne 0 ]; then
  echo "FAIL: swift build qwg release exit $qbstatus"
  cat "$CI_TMP_DIR/qwg_release.log"
  exit "$qbstatus"
fi
tail -1 "$CI_TMP_DIR/qwg_release.log"
python3 Scripts/check_swifterror_abi.py ./.build/release/qwg || { echo "FAIL: swifterror ABI gate (qwg)"; exit 1; }
set +e
swift build -c release --product qwg-menu > "$CI_TMP_DIR/qwg_menu_release.log" 2>&1
mbstatus=$?
set -e
if [ "$mbstatus" -ne 0 ]; then
  echo "FAIL: swift build qwg-menu release exit $mbstatus"
  cat "$CI_TMP_DIR/qwg_menu_release.log"
  exit "$mbstatus"
fi
tail -1 "$CI_TMP_DIR/qwg_menu_release.log"
python3 Scripts/check_swifterror_abi.py ./.build/release/qwg-menu || { echo "FAIL: swifterror ABI gate (qwg-menu)"; exit 1; }
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "skipping qwg-bench-kernels execution on GitHub Actions paravirtual GPU"
else
  BENCH_OUT_DIR="$CI_TMP_DIR/bench_out"
  mkdir -p "$BENCH_OUT_DIR"
  ./.build/release/qwg-bench-kernels --samples 2 --out "$BENCH_OUT_DIR" >/dev/null
  python3 Scripts/check_bench_report.py "$BENCH_OUT_DIR/bench_kernels_report.json" \
    || { echo "FAIL: the winner map"; exit 1; }
fi
echo "ci ok"
