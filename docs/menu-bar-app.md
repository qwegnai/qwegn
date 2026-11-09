# Menu Bar App (`qwg-menu`)

`qwg-menu` places a live status and telemetry item in the macOS menu bar and manages the `qwg serve` lifecycle:

```bash
.build/release/qwg-menu
```

- **Status Indicator**: Gray (`offline`), Amber (`idle / unloaded`), Cyan (`loading weights`), Green (`ready`), Blue (`prefilling`), Purple (`decoding`), Red (`error`).
- **Live Telemetry**: Displays real-time decode `tok/s` in the menu bar and a detailed popover with prefill speed, decode throughput, TTFT, unified memory footprint, prefix-cache hit rate, and active context window.
- **Settings (`⌘,`)**: Configure active model pack, on-demand or persistent **Vision Tower (`loadVision`)** residency without KV cache reset, quantization/KV mode, context window, speculative decoding (`MTP` / `Variational` / `DFlash` / `MoE-Spec`), **Codebase Memory Pack (`.qwg/manifest.json` mount/unmount with automatic `kvMode` enforcement lock)**, **Performance (` `QWEngine` ` tuning: `maxResident` vs `safe` memory policy, OS reserve GiB, resident experts/layer, dynamic expert LRU cache, event-pipelined cold staging, GPU prefill span map)**, server port, and launch-at-login.
- **Shortcuts**: `⌘L` opens the live server log viewer; `⌘R` restarts the inference engine; `⌘Q` cleanly stops the managed server and exits.

Per-model LSA and Codebase Memory Pack (`QWMemoryPack`) settings are documented in [lsa.md](lsa.md#3-menu-bar-app-qwg-menu-per-model-settings) and [lsa-codebase-memory-whitepaper.md](lsa-codebase-memory-whitepaper.md).
