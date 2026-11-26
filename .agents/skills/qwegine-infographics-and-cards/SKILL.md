---
name: qwegine-infographics-and-cards
description: >-
  Use when generating or updating SVG/PNG benchmark or architecture infographics
  in assets/ or ~/.qwg/models/*/media/, rasterizing Retina previews for visual
  checks, verifying QWP1 container headers, or syncing Hugging Face model cards.
---

# Qwegn Infographics, Pack Headers & Model Card Sync

Follow these five steps whenever creating or updating benchmark/architecture SVGs or publishing Hugging Face model cards and media.

## Procedure

1. **Apply Evergreen Infographic Layout & Machine Card:**
   - Place the **Qwegn inference engine logo badge in the top-right corner** and include the exact **inference machine specification card** (`Apple M5 Max · Mac17,6 · 18-Core CPU · 40-Core Apple10 NAX GPU · 128 GB Unified Memory · 585 GB/s DRAM · 950 GB/s SLC · macOS 26.2+`).
   - Present **current canonical model and engine performance only** inside `assets/qwegn_*.svg` and `~/.qwg/models/*/media/`; record `before -> after` deltas, legacy pack baselines, and speedup multipliers in `CHANGELOG.md` and `docs/tuning.md`.
2. **Author CoreSVG 1.1–Compatible SVG Halos & Text:**
   - Render chart value halos as two stacked `<text>` elements (`fill="none" stroke="#0b1120" stroke-width="3.5" stroke-linejoin="round"` behind `fill="#..."`), never SVG2 `paint-order="stroke"`.
   - Keep bullet rows as plain single-spaced text (`• Label: value`) without `x="..."` overrides on nested `<tspan>` elements inside translated `<g>` groups.
3. **Rasterize `2×` Retina PNG & Visually Inspect (`read_file`):**
   - Render every modified `.svg` via Swift AppKit (`NSImage` + `NSBitmapImageRep` with `pixelsWide = width * 2`, `rep.size = image.size`), inspect every `/tmp/*.png` with `read_file` for pill `<rect width>` clipping, label collisions, or axis overlap, and remove temporary PNGs before committing:
     ```bash
     swift -e 'import AppKit; let img = NSImage(contentsOfFile: CommandLine.arguments[1])!; let w = Int(img.size.width), h = Int(img.size.height); let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w*2, pixelsHigh: h*2, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!; rep.size = img.size; NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep); img.draw(in: NSRect(origin: .zero, size: img.size)); NSGraphicsContext.restoreGraphicsState(); try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))' assets/qwegn_engine_bento.svg /tmp/preview.png
     ```
4. **Verify QWP1 Container Headers From Ground Truth:**
   - Audit tensor formats and container magic directly against `Sources/`, QWP1 footer JSON (`QWP1` magic, 8-byte LE footer size at offset `4`, JSON at `length - size`), or `Scripts/quantize/audit_release.py` (never copy invented tags or stale `/usr/local/bin/qwg inspect` output):
     ```bash
     .venv-oracle/bin/python Scripts/quantize/audit_release.py --release ~/.qwg/models/<pack>
     ```
5. **Sync Multi-Variant Hugging Face Repos Sequentially:**
   - Create repos with `hf repos create <org>/<repo> --type model --public --exist-ok` and upload sequentially with `HF_HUB_DISABLE_PROGRESS_BARS=1 hf upload ... > out/hf_upload.log 2>&1` so HF Xet deduplicates shared chunks.
   - Generate/upload matching `model.qlsa` via `qwg lsa-convert` when applicable, copy `media/*.{svg,png}` and `LICENSE`, sync `~/.qwg/models/<pack>/README.md` + repo docs (`README.md`, `CHANGELOG.md`, `docs/*.md`) without touching `~/.qwg/models/<pack>/assets/*`, and verify `hf models info <org>/<repo>` lists `model.qwp`, sidecars, `README.md`, `LICENSE`, `media/`, and `assets/`.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| Dark stroke paints *over* SVG text in Finder/Safari/PNG | Replace SVG2 `paint-order="stroke"` with two stacked `<text>` elements (outline first, fill second). |
| Misaligned bullet columns in macOS CoreSVG | Remove `x="..."` overrides on nested `<tspan>` inside `<g transform=...>` and use plain `• Label: value` single-spaced lines. |
| `python3 -c "import AppKit"` fails (`ModuleNotFoundError`) | Use the `swift -e 'import AppKit ...'` rasterizer command in Step 3 instead of Homebrew/Xcode Python. |
| `qwg inspect` reports `"q4g"` on mixed-plane `.qwp` packs | Run `.venv-oracle/bin/python Scripts/quantize/audit_release.py --release <dir>` or read the QWP1 footer JSON directly. |
| `hf repo create` rejected by newer `hf` CLI | Use plural `hf repos create <org>/<repo> --type model --public --exist-ok`. |
