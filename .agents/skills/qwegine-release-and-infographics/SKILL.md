---
name: qwegine-release-and-infographics
description: >-
  Use when publishing Qwegn benchmark reports, running full llmprobe server
  qualifications, generating or updating SVG/PNG infographics in assets/ or
  ~/.qwg/models/*/media/, syncing model cards, or executing a Qwegn version
  bump and release cleanup.
---

# Qwegn Benchmarks, Infographics & Release Runbook

## 1. Presentation SVG & Retina PNG Infographics

Every benchmark or architecture infographic in `assets/qwegn_*.svg` and `~/.qwg/models/*/media/` follows five invariants:

1. **Header & Machine Card:** Place the **Qwegn inference engine logo badge in the top-right corner** and include the exact **inference machine specification card** (`Apple M5 Max · Mac17,6 · 18-Core CPU · 40-Core Apple10 NAX GPU · 128 GB Unified Memory · 585 GB/s DRAM · 950 GB/s SLC · macOS 26+`).
2. **Evergreen Current-State Metrics Only:** Infographics present **current canonical model and engine performance** — record `before -> after` deltas, legacy pack baselines, and speedup multipliers against retired fallbacks in `CHANGELOG.md` and `docs/tuning.md` only, never inside `assets/qwegn_*.svg` or `~/.qwg/models/*/media/`.
3. **SVG 1.1 Text Halos (No `paint-order="stroke"`):** macOS CoreSVG / `NSImage` ignores SVG2 `paint-order="stroke"` and draws dark strokes *over* text fills. Render chart value halos as two stacked `<text>` elements: background `<text fill="none" stroke="#0b1120" stroke-width="3.5" stroke-linejoin="round">` + foreground `<text fill="#...">`.
4. **Mandatory Visual Verification (`2×` Retina PNG):** Rasterize every touched `.svg` at `2×` scale via `swift -e 'import AppKit ...'` (`NSImage` + `NSBitmapImageRep` with `pixelsWide = width * 2`, `rep.size = image.size`; do not rely on Python `import AppKit`, which is absent from Homebrew/Xcode Python), inspect the rendered `/tmp/*.png` with `read_file`, and delete the temporary `.png` before committing — verifying zero pill `<rect width>` clipping, zero label collisions, and clean axis clearance.
5. **Model-Pack Media & Manifest-Safe Card Sync:** Copy updated per-model `.svg` and `2×` `.png` files into `~/.qwg/models/<pack>/media/`, update `~/.qwg/models/<pack>/README.md`, and sweep all repo docs mentioning the model (`README.md`, `CHANGELOG.md`, `docs/*.md`). Keep `~/.qwg/models/<pack>/assets/*` untouched so upstream `assets/manifest.json` SHA-256 digests stay valid.

## 2. Full `llmprobe` Server Qualification (`qwg serve`)

1. **Quiet-GPU Execution & Preserved Reports:** On an idle GPU (`0%/0%`), run `llmprobe --full` plus `--eval-suite code --reasoning off` (`100` tasks) and write JSON/HTML outputs directly into `out/llmprobe/`.
2. **Zero-Error & Sustained-Load Audit:** Verify `0` HTTP/transport errors across `conformance`, `capability`, `agentic`, `fidelity`, and `bench`, steady `20m+` `loadDrift` (no UMA residency paging dip), deduplicated constrained tool-delimiter IDs (`"special": false` tokenizers), and clean parallel `<tool_call>` splitting.

## 3. Worktree `make ci`, Version Bump & Cleanup Checklist

1. **Worktree Goldens without Dirty Symlinks:** `make goldens` reads `.venv-oracle/bin/python`. Because `.gitignore` ignores `.venv-oracle/` (directory slash), if you symlink `ln -s <root>/.venv-oracle .venv-oracle` inside `.worktrees/<task>` to run `make goldens && make ci`, `rm -f .venv-oracle` before `git commit`.
2. **Canonical Version Bump Locations:** Keep `Sources/Foundation/QWCore/QWVersion.swift`, `CHANGELOG.md`, `README.md` (badge + headline table), `CITATION.cff`, `assets/qwegn_engine_bento.svg` (top pill + footer), and `docs/tuning.md` (ledger row) synchronized.
3. **`1M`-Context Qualification on `80+ GiB` Packs:** Validate `1M` (`1,048,576`) UMA admission & residency (`planned <= available`, `0` steady-state allocations) with a deterministic short-prompt run paired with `16K / 128K / 300K` 3-needle passkey sweeps; avoid multi-hour full `1M`-token prefills unless explicitly requested.
4. **Merge, Release Binary & Artifact Hygiene:** After `make ci` passes (`0` warnings, `0` style/kernel violations), merge into `master`, run `make build` on `master`, verify `./.build/release/qwg --version`, delete `.worktrees/<task>` + feature branch, and clean temporary `/tmp` / `scratch` logs while **preserving `out/llmprobe/*.json`**.
5. **Post-Merge HF Source Cache Cleanup (When Requested / After Pack Qualification):** Once code and pack qualification have landed on `master`, remove the upstream BF16 cache repo with `hf cache rm model/<org>/<repo> --dry-run` followed by `hf cache rm model/<org>/<repo> --yes`, and verify `~/.qwg/models/<pack>/model.qwp` remains intact.
