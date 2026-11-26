---
name: qwegine-menubar-and-webui
description: >-
  Use when designing, modifying, or visually validating Qwegn.app (QWMenuBar:
  CupertinoTelemetryView, TelemetryCards, SettingsView, LogViewerWindow,
  QGProcessorLogoView), the offline QWServer WebUI (/ui, controlbar.html,
  Chat, #modal-memory-pack, QWDashboardViewModel), or menu bar / dashboard parity
  and honesty tests.
---

# Qwegn Menu Bar (`Qwegn.app`) & Offline WebUI (`/ui`) Runbook

Follow these four steps whenever changing `QWMenuBar`, `/ui` HTML/CSS/JS assets, or `QWKnobRegistry` UI surfaces (`ADR-022`: `6` `/ui` tabs max, `0` new `440pt` cards, `0` new knobs without weekly human use + explicit ADR).

## Procedure

1. **Enforce Control-Panel Hierarchy & Lifecycle Boundaries:**
   - Render live telemetry and active-model traits (`quantization`, `KV`, `ctx`, `Spec`, `MemPack`) only — omit constant host specs, duplicate `modelAlias`, and internal dispatch tags from the `440pt` popover (`width: 420` + `.padding(10)` + `spacing: 8` inside `.frame(width: 440).clipped()`, `.fixedSize(horizontal: true, vertical: false).layoutPriority(1)` on button clusters, `MarqueeText` / `.hero-model-marquee` guarded by `title.dataset.sig`).
   - Show unavailable metrics as `—` / `— tok/s` with `card.reason` tooltips; auto-collapse `THROUGHPUT TRACE` when both series have zero non-`nil` samples; keep server lifecycle (`Restart`/`⌘R`, `Stop`/`Start`) in header and model lifecycle (`Switch`, `Unload`/`Load`) inside `RESIDENT MODEL`.
2. **Apply macOS 26 Glass, Transition & Co-Bundled Binary Rules:**
   - Back dropdowns inside `GlassEffectContainer` with `.buttonStyle(.plain)` + `NSViewRepresentable` `MenuAnchorBridge` (`NSMenu.popUp`), keep `busy: false` on secondary header buttons (`Restart`), and bundle/codesign release `qwg` inside `Qwegn.app/Contents/MacOS/qwg` (`Scripts/build_app.sh`).
3. **Pass Telemetry Honesty, Emblem & CLI–GUI–WebUI Parity Tests:**
   - Register every card ID in `MenuBarCardCatalogue.ids` and `QWDashboardBuilder.specs`; keep WebUI files `<= 800` lines, zero CDN URLs, zero `"Idle"`/`"Stopped"` fallbacks, and all 4 `QWG` emblems (`QGProcessorLogoView.swift`, `gen_app_icon.swift`, `logo.svg`, `controlbar.html`) synced (`x2="106"` left bus pins).
   - Sync `Sources/Foundation/QWCore/QWKnobRegistry.swift` with `Scripts/gen_knob_cli.py --stdout`, `Sources/Apps/qwg/Generated/KnobOptions.swift`, `SettingsView`, and `/ui` `settings.js`, then run:
     ```bash
     swift test --no-parallel --filter QWMenuBarTests --filter QWServerTests
     ```
4. **Run Mandatory `2×` Retina Offscreen & Live `/ui` Verification:**
   - Capture `2×` Retina `NSHostingView` snapshots via `QWG_SNAPSHOT_DIR=/tmp/qwg_snaps swift test --no-parallel --filter QWMenuBarDecisionTests` (or scratch AppKit renderer with `telemetry.isDaemonTransitioning = false`), inspect every PNG via `read_file`, and delete temp PNGs before commit.
   - Validate `http://127.0.0.1:<port>/ui` across all 6 tabs at `1440×900` and `900×800` (`0` console/HTTP errors, `scrollWidth <= clientWidth`), and verify `#modal-memory-pack` (`missing`, `wrong_model`, `stale`, `ready`) via rendered DOM badges (in `.qwcache` tests, advance `modificationDate >= 2s`, change byte count, and seed `createdAtEpochSec` as `Int64`).

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| SwiftUI `Menu` collapses invisible inside `GlassEffectContainer` | Replace with `.buttonStyle(.plain)` + `MenuAnchorBridge` (`NSMenu.popUp` + `objc_setAssociatedObject`). |
| macOS TCC blocks headless `screencapture` on `Qwegn.app` | Use `QWG_SNAPSHOT_DIR=/tmp/qwg_snaps` `NSHostingView` rasterization and inspect via `read_file`. |
| Chrome DevTools MCP fails with `EPERM` on `DevToolsActivePort` | Run a scratch native `WKWebView` Swift harness against loopback `/ui`, inspect saved PNGs via `read_file`, note the block in final report, and delete scratch files. |
| `computeDelta` / `#modal-memory-pack` staleness test flakes | Change file byte count, advance `modificationDate >= 2s`, and write `createdAtEpochSec` as `Int64` (not ISO-8601). |
| `QWCliGuiWebParityTests` fails after adding/editing a knob | Regenerate `Sources/Apps/qwg/Generated/KnobOptions.swift` via `Scripts/gen_knob_cli.py` and sync `SettingsView` + `js/settings.js`. |
