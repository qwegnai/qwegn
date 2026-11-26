---
name: qwegine-release
description: "Execute an owner-requested Qwegn version bump, local build/codesign, GitHub release verification, and cleanup."
disable-model-invocation: true
---

# Qwegn Release Preparation, Publish & Verification (`/qwegine-release`)

Run this workflow **only** when the owner explicitly requests publishing a release (`AGENTS.md` Verification footer). Normal feature merges to `master` record changes under `## [Unreleased]` without bumping versions.

## Procedure

1. **Pre-Flight Cleanliness & Symlink Removal:**
   - Remove any temporary worktree symlinks (`rm -f .venv-oracle Scripts/spike/out out/vision_decision_oracle`) and ensure `git status --porcelain` is clean (`clean_src` requirement).
   - Refresh primary-checkout oracle fixtures not covered by `Scripts/generate_goldens.sh` (e.g., `.venv-oracle/bin/python Scripts/oracle/vision_decision_oracle.py`) before tagging.
2. **Synchronize Version & Zero-Mention Release Notes:**
   - Pick patch/minor/major per accumulated changes since the last tag. In one release-preparation commit, update: `Sources/Foundation/QWCore/QWVersion.swift`, `CHANGELOG.md` (rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`; add `### Cumulative Highlights Since Previous Public Release (vA.B.C -> vX.Y.Z)` if multiple versions accumulated), `README.md` (badge, table, BibTeX), `CITATION.cff`, `assets/qwegn_engine_bento.svg` (top pill + footer), and `docs/tuning.md`.
   - Confirm zero bare `@[A-Za-z0-9_.-]+` tokens exist in `CHANGELOG.md` (write `at 4K`, `at 1M`, `Observable` — never `@4K` or `@MainActor`).
3. **Run Full Gate, Build & `CFBundle` Codesign Verification:**
   - Park `Qwegn.app` / `qwg serve` (`pgrep -fl "qwg|Qwegn"`), then run `make ci && make test && make build` (`make test` prints exact XCTest counts, whereas `Scripts/ci.sh` tails 5 lines).
   - Verify `.build/release/qwg --version`, `.build/release/Qwegn.app/Contents/MacOS/qwg --version`, and nested `Contents/Resources/*.bundle` normalization (`Scripts/build_app.sh` / `Scripts/build_pkg.sh`) via:
     ```bash
     /usr/bin/codesign --verify --deep --strict .build/release/Qwegn.app
     ```
4. **Publish & Verify GitHub-Compiled Release Artifacts (`.github/workflows/release.yml`):**
   - Run `./Scripts/publish_release.sh` (only pass `--skip-ci` after `--dry-run` on an already-gated commit if a GPU co-runner cannot pause), then watch both workflows:
     ```bash
     gh run watch <run-id> --repo qwegnai/qwegn --exit-status
     ```
   - Because `release.yml` overwrites assets with `--clobber`, download `qwegn-vX.Y.Z-macos-arm64.tar.gz`, `Qwegn-X.Y.Z-macos-arm64.pkg`, and `SHA256SUMS.txt` into `/tmp/relcheck` after `release.yml` completes and verify:
     1. `shasum -a 256 -c SHA256SUMS.txt` passes.
     2. `xcrun -sdk macosx metal-readobj --headers` confirms `PlatformMajor: 26`, `PlatformMinor: 2`+ on `kernel.metallib`, `kernel_std.metallib`, and `Qwegn.app/Contents/Resources/kernel.metallib`, matching GitHub's `metalfe-` string.
     3. `pkgutil --expand-full` (`find pkg_expanded -name "kernel*.metallib"`) matches `.tar.gz` SHA-256 byte-for-byte.
     4. Extracted `qwg --version` == `X.Y.Z` and passes a quiet-GPU NAX probe + `serve` smoke via `QW_METALLIB=<extracted>/kernel.metallib`.
5. **Cleanup (Never Delete `out/llmprobe/*.json`):**
   - Delete only your own `.worktrees/<task>` + feature branch, clean `/tmp/relcheck` while preserving `out/llmprobe/*.json`, and optionally prune superseded upstream BF16 caches (`hf cache rm model/<org>/<repo> --dry-run` then `--yes`).

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `codesign` fails with `bundle format unrecognized` on `qwegine_QWServer.bundle` | Re-run `./Scripts/build_app.sh` so `normalize_resource_bundle` converts the flat bundle to `Contents/Info.plist` + `Contents/Resources/`. |
| GitHub Actions `macos-26` job times out with `steps: []` (`not acquired by Runner`) | Rerun `release` solo first via `gh run rerun <id> --repo qwegnai/qwegn`, then rerun `ci` after `release` completes (`Law 26`). |
| `release.yml` fails *after* public tag push | Never force-push public tags: fix in a worktree, bump patch `X.Y.(Z+1)`, publish the new tag, and mark the broken release `--prerelease`. |
| `release.yml` succeeds but `ci.yml` fails on `main` (or `master` moved ahead) | Fix in clean worktree, pass local `make ci`, apply only that diff via `git -C <master> show --format= <FIX_SHA> | git -C ~/Developer/ai/qwegine-publish apply`, verify `0` `$HOME` leaks, push `main` only. |
| Bare `@` token triggers unintended GitHub contributor mention | Strip/reword bare `@<token>` in `CHANGELOG.md` before tagging (`rg -n '@[A-Za-z0-9_.-]+' CHANGELOG.md`). |
