---
name: qwegine-release-and-infographics
description: >-
  Use when publishing Qwegn benchmark reports, generating or updating SVG/PNG
  infographics in assets/ or ~/.qwg/models/*/media/, syncing model cards, or
  executing a Qwegn version bump and release cleanup.
---

# Qwegn Benchmarks, Infographics & Release Runbook

## 1. Presentation SVG & Retina PNG Infographics

Every benchmark or architecture infographic in `assets/qwegn_*.svg` and `~/.qwg/models/*/media/` follows four invariants:

1. **Header & Machine Card:** Place the **Qwegn inference engine logo badge in the top-right corner** and include the exact **inference machine specification card** (`Apple M5 Max · Mac17,6 · 18-Core CPU · 40-Core Apple10 NAX GPU · 128 GB Unified Memory · 585 GB/s DRAM · 950 GB/s SLC · macOS 26+`).
2. **SVG 1.1 Text Halos (No `paint-order="stroke"`):** macOS CoreSVG / `NSImage` ignores SVG2 `paint-order="stroke"` and draws dark strokes *over* text fills. Render chart value halos as two stacked `<text>` elements: background `<text fill="none" stroke="#0b1120" stroke-width="3.5" stroke-linejoin="round">` + foreground `<text fill="#...">`.
3. **Mandatory Visual Verification (`2×` Retina PNG):** Rasterize every touched `.svg` at `2×` scale via `swift -e 'import AppKit ...'` (`NSImage` + `NSBitmapImageRep` with `pixelsWide = width * 2`, `rep.size = image.size`; do not rely on Python `import AppKit`, which is absent from Homebrew/Xcode Python), inspect the rendered `/tmp/*.png` with `read_file`, and delete the temporary `.png` before committing — verifying zero pill `<rect width>` clipping, zero label collisions, and clean axis clearance.
4. **Model-Pack Media Sync:** Copy updated per-model `.svg` and `2×` `.png` files into `~/.qwg/models/<pack>/media/` and update matching tables in `~/.qwg/models/<pack>/README.md`.

## 2. Worktree `make ci`, Version Bump & Cleanup Checklist

1. **Worktree Goldens without Dirty Symlinks:** `make goldens` reads `.venv-oracle/bin/python`. Because `.gitignore` ignores `.venv-oracle/` (directory slash), if you symlink `ln -s <root>/.venv-oracle .venv-oracle` inside `.worktrees/<task>` to run `make goldens && make ci`, `rm -f .venv-oracle` before `git commit`.
2. **Canonical Version Bump Locations:** Keep `Sources/Foundation/QWCore/QWVersion.swift`, `CHANGELOG.md`, `README.md` (badge + headline table), `CITATION.cff`, `assets/qwegn_engine_bento.svg` (top pill + footer), and `docs/tuning.md` (ledger row) synchronized.
3. **Merge, Release Binary & Artifact Hygiene:** After `make ci` passes (`0` warnings, `0` style/kernel violations), merge into `master`, run `make build` on `master`, verify `./.build/release/qwg --version`, delete `.worktrees/<task>` + feature branch, and clean temporary `/tmp` / `scratch` logs while **preserving `out/llmprobe/*.json`**.
