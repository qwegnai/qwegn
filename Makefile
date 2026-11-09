# QWEngine (qwg) — build, test, and golden-generation recipes.
#
# Start here:   make init     (checks the toolchain, creates the Python venv,
#                              generates the test goldens)
#               make ci       (build + lint + tests + CLI smoke)
#
# Design laws:
#   * Each recipe is a thin wrapper over a script CI can also run standalone.
#   * Everything regenerable (.build, goldens, the venv, kernel.metallib) is
#     git-ignored; `make clean` removes it.
#   * Tests read only generated artifacts that `make goldens` produces, never
#     files from outside the repository.

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

VENV        := .venv-oracle
PYTHON      ?= python3.13
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STAMP       := $(VENV)/.requirements.stamp
BUILD       ?= debug
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------------ help --

help: ## Show the available recipes
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- init --

init: ## Check the toolchain, create the Python venv, and generate the goldens
	@echo "== 1/4 toolchain: Xcode 26+ with Metal 4, Swift 6.2+, Python 3.11+, macOS 26+"
	@./Scripts/check_toolchain.sh
	@echo "== 2/4 Python venv: $(VENV) (pinned by requirements.txt)"
	@command -v $(PYTHON) >/dev/null || { echo "FAIL: $(PYTHON) not found (brew install python@3.13)"; exit 1; }
	@test -x $(PY) || $(PYTHON) -m venv $(VENV)
	@if [ ! -f $(STAMP) ] || [ requirements.txt -nt $(STAMP) ]; then \
	  $(PIP) install -q --upgrade pip && \
	  $(PIP) install -q --index-url $(TORCH_INDEX) torch==$$(sed -n 's/^torch==//p' requirements.txt) && \
	  $(PIP) install -q -r requirements.txt && \
	  touch $(STAMP) || { echo "FAIL: pip install did not complete; fix it and re-run make init"; rm -f $(STAMP); exit 1; }; \
	fi
	@echo "== 3/4 metallib: compiling Sources/QWMetal kernels"
	@./Scripts/build_metallib.sh >/dev/null
	@echo "== 4/4 goldens: generating the synthetic test fixtures (no model weights)"
	@$(MAKE) --no-print-directory goldens
	@echo "init: done. Next: make ci"

goldens: ## Regenerate the synthetic golden fixtures the tests read (needs the venv)
	@test -x $(PY) || { echo "FAIL: run 'make init' first (no $(VENV))"; exit 1; }
	@./Scripts/generate_goldens.sh

# ---------------------------------------------------------------- build --

build: metallib ## Build the qwg CLI and the menu bar app (release)
	@swift build -c release --product qwg
	@swift build -c release --product qwg-menu
	@echo "built: .build/release/qwg, .build/release/qwg-menu"

metallib: ## Compile Sources/QWMetal/*.metal into kernel.metallib
	@./Scripts/build_metallib.sh

# ----------------------------------------------------------------- test --

test: ## Run the XCTest suite (serial: parallel GPU bundles are unsafe)
	@swift test --no-parallel

lint: ## Style, geometry, and kernel-reachability gates
	@python3 Scripts/lint_style.py Sources Tests Scripts
	@python3 Scripts/lint_geometry_defaults.py Sources
	@python3 Scripts/test_kernel_inventory.py
	@./Scripts/build_metallib.sh >/dev/null
	@python3 Scripts/lint_kernels.py

ci: ## The full pipeline: build, lint, test, CLI smoke
	@./Scripts/ci.sh

perf: ## Performance gates (needs the model packs in ~/.qwg/models; prints why if missing)
	@python3 Scripts/perf_gate.py

m1: ## Soak gate: writes the measured soak artifact bench-m1 verifies (CPU reference soak without a pack)
	@set -o pipefail; swift build -c release --target m1 2>&1 | tail -1
	@./.build/release/m1 --harness --steps $${STEPS:-10000} --window $${WINDOW:-1000} --export .build/m1_export.jsonl
	@python3 Scripts/gen_soak_sample.py .build/m1_export.jsonl --out-dir $${SOAK_DIR:-.}

bench-m1: ## Merge gate: re-verify the newest soak artifact and the ratchet (no GPU)
	@set -o pipefail; swift build 2>&1 | tail -1
	@./.build/$(BUILD)/qwg bench m1 --check --soak-dir $${SOAK_DIR:-.}

# -------------------------------------------------------------- release --

release: ci ## Release gate: ci must pass on a clean tree
	@git diff --quiet && git diff --cached --quiet || { echo "FAIL: uncommitted changes"; exit 1; }
	@echo "release gate: PASS"

# ---------------------------------------------------------------- clean --

clean: ## Remove regenerable build outputs (keeps the venv)
	@rm -rf .build Sources/QWMetal/kernel.metallib Scripts/spike/out
	@echo "clean: done (run 'make init' to regenerate)"
