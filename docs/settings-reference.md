# Settings Reference & Operational Recipes

Qwegn shares **one unified configuration system** ([`QWKnobRegistry`](../Sources/Foundation/QWCore/QWKnobRegistry.swift)) across the `qwg` CLI, the `Qwegn.app` macOS Menu Bar app, the OpenAI/Anthropic/Ollama HTTP server (`qwg serve`), and per-model profiles (`~/.qwg/profiles/<model>.json`).

This guide starts with **what works out of the box** and **copy-paste recipes for common workflows**, followed by the complete reference table of every setting, Menu Bar tab, CLI flag, and environment variable.

---

## 1. Zero-Config Defaults (For Everyday Users)

You do **not** need to tune flags to get peak speed and accuracy. Out of the box, Qwegn inspects each model's `assets/config.json` and your Mac's live unified memory (`vm_statistics64`) to choose safe, high-throughput defaults:

- **`--kv-mode q4` (4-bit KV Cache):** Cuts KV cache memory by **`4×`** compared to `f16` while preserving core reasoning (`10/10` `llmprobe`) and needle recall (`3/3` up to `1M` tokens), and matches `qwg index` `.qwcache` repository memory packs with zero configuration friction.
- **`--lsa-mode auto` (Adaptive Sparse Attention):**
  - **Short prompts (`<= 2,564` tokens):** Automatically bypasses sparse indexing and runs **bit-exact dense attention** (`0%` overhead).
  - **GQA / Hybrid models (`Qwen3.6-35B-A3B`, `Qwen3.8-27B`, `SenseNova-Loop-9B`):** Automatically loads `<model>/model.qlsa` if present (or synthesizes Weight-SVD projections at load), avoiding the `13.9×` dense prefill cliff at long context.
  - **QSA models (`Qwen3.8-Flash-Next`):** Uses native flat QSA up to `262K` tokens and automatically engages **`e-means-4096` coarse clustering** above `262K` (`+39.8%` prefill / `+55.3%` decode at `1M`).
  - **NoPE MLA models (`GLM-5.3-Flash`):** Runs zero-overhead flat indexing below `41K` tokens and automatically engages **`e-means` coarse pruning** at `>= 41K` tokens (`743.0 / 44.98 tok/s` at `242.5K` — `+3.0%` prefill / `+2.2%` decode vs `exactFlat`, `3/3` needle recall, `1M` UMA ready).
- **Smart Unified-Memory Residency (`--memory-policy maxResident`, `--no-load-vision`, `--no-stream-ngrams`):**
  - Vision towers stay unloaded on text turns (saving **`1.60 GiB` RSS**) and load automatically on demand when an image is supplied.
  - External Per-Layer Embedding tables (`ngram.bin`: `27.1 GiB` `VQ2_G32` or `15.1 GiB` `VQ4_G32`) wire into Metal GPU RAM when headroom allows, and **automatically fall back to NVMe SSD streaming** (`64 MiB` L1 cache, saving **`15.1–27.3 GiB` GPU RAM**) when long contexts or lower-RAM Macs require room.
- **Drafter Auto-Detection (`DFlash 2` & `DSpine`):** Passing `--dflash-window` or choosing a drafter pack (`HQ*`, `AQ*`, or sub-2-bit `SHVQ{1,15}*`) in `Qwegn.app` inspects the drafter's `config.json` to select `dspine` (`B = 16`, up to `15` drafts) vs `dflash2` (`B = 8`, up to `7` drafts) automatically.

### Precedence & Scope

When multiple configuration sources exist, Qwegn resolves them from most specific to broadest:

```text
Per-Request API Body  >  CLI Flags  >  QW_* / QWG_* Env Vars  >  ~/.qwg/profiles/<model>.json  >  ~/.qwg/settings.json  >  Model config.json + Engine Defaults
```

> **Tip:** Pass `--no-profile` on `qwg run`, `qwg repl`, or `qwg serve` to ignore saved user profiles (`~/.qwg/settings.json` and `~/.qwg/profiles/*.json`) for clean, reproducible benchmarks.

Each setting has one of four **scopes**:
- **`request`** — Applies immediately to the next turn or HTTP request (no reload).
- **`session`** — Applies in place to the live session or vision residency without resetting KV state.
- **`load`** — Sizes GPU arenas or weight residency; changing it in `Qwegn.app` shows an orange reload badge (`↻`).
- **`process`** — Network bind (`--host`, `--port`), lazy startup (`--lazy`), and log sink (`--log-level`).

---

## 2. Copy-Paste Recipes by Operational Scenario

### Scenario 1 — Zero-Config Local Chat & Coding Server (Novice Quick Start)

Launch the macOS Menu Bar app (`Qwegn.app`) or start an OpenAI/Anthropic/Ollama-compatible server on `http://127.0.0.1:8080`:

```bash
# Start the HTTP server with adaptive Q4 KV + Auto LSA at 64K context
qwg serve Qwen3.6-35B-A3B-hq5g64-hq4g64 -c 65536

# Or chat directly in your terminal
qwg repl Qwen3.6-35B-A3B-hq5g64-hq4g64
```

### Scenario 2 — Instant Whole-Repository Coding Agent (`qwg index` + `.qwcache`)

Pre-index a codebase once into `<repo>/.qwg/` (`32K` resident Cortex + `< 0.5 ms` off-context Vault recall across hundreds of files), then serve or chat with `< 250 ms` time-to-first-token:

```bash
# 1. Build the repository memory pack once (defaults to q4 KV + auto LSA)
qwg index /path/to/your/repo --model Qwen3.6-35B-A3B-hq5g64-hq4g64

# 2. Serve with the memory pack mounted (live git edits inject via <|memory_delta|>)
qwg serve Qwen3.6-35B-A3B-hq5g64-hq4g64 -c 131072 --repo /path/to/your/repo
```

### Scenario 3 — Peak Lossless Single-Stream Coding (`Qwen3.8-27B` + `DSpine-b16`, `4.19×` Serial)

Pair the dense `Qwen3.8-27B` target model with its 5-layer `DSpine` block drafter (`B = 16`, `15` speculative drafts verified in one GPU pass — `100%` bit-exact to greedy serial decode at up to `134.5 tok/s`):

```bash
qwg serve Qwen3.8-27B-hq4g32 \
  -c 65536 \
  --dflash-draft-pack ~/.qwg/models/Qwen3.8-27B-DSpine-b16-hq4g32 \
  --dflash-window 16
```

### Scenario 4 — Ultra-Long Context Analysis (`262K` to `1M` Tokens)

Qwegn scales cleanly to **1,048,576 tokens** on a single Mac via automatic YaRN RoPE scaling, `q4` KV compression, and subquadratic LSA / `e-means`:

```bash
# Qwen3.6-35B-A3B at 1M context (2,317 tok/s prefill, 105.6 tok/s decode, ~27.3 GiB Metal)
qwg serve Qwen3.6-35B-A3B-hq5g64-hq4g64 -c 1048576

# Qwen3.8-Flash-Next at 1M context (auto-engages e-means-4096; stream PLE from SSD to free 27.3 GiB RAM)
qwg serve Qwen3.8-Flash-Next-hq8g32-hq4g64 -c 1048576 --stream-ngrams

# Qwen3.8-27B at 524K context using e-means-4096 macro-blocks for maximum prefill throughput
qwg serve Qwen3.8-27B-hq4g32 -c 524288 --lsa-cluster-block 4096
```

### Scenario 5 — Low-Memory / Desktop Co-Existence Mode (`-28.5 GiB` RAM Saved)

When running Xcode, Docker, browsers, or IDEs alongside `Qwen3.8-Flash-Next` (or on `64–96 GB` Macs), use `safe` memory policy (never swaps other apps), stream the PLE table from NVMe (`-27.30 GiB`), omit eager vision (`-1.60 GiB`), and evict prefill scratch after prompt ingestion:

```bash
qwg serve Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  -c 65536 \
  --memory-policy safe \
  --stream-ngrams \
  --no-load-vision \
  --evict-prefill-scratch always
```

### Scenario 6 — `321B` Flagship Reasoning (`GLM-5.3-Flash`)

On a `128 GB` Mac with co-runners closed (`~108 GiB` free unified memory), the `100.6 GiB` mixed `SHVQ` pack loads **100% resident** (`288/288` experts per layer in RAM, zero SSD expert misses, `1,006 tok/s` prefill, `46.0–53.6 tok/s` decode, `81/100` HumanEval+MBPP) and uses `e-means` MLA indexing automatically up to `242.5K+` (`1M` UMA-capable):

```bash
# 100% UMA-resident 321B reasoning with NextN MTP-1 speculation
qwg serve GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  -c 202752 \
  --mtp-backend standard --mtp-drafts 1 \
  --thinking --reasoning-effort medium

# Optional: run a >128 GiB MoE pack alongside desktop apps with zero swap via cold-expert NVMe streaming
qwg serve GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  --memory-policy safe --os-reserve 24 --moe-spec-budget 4
```

### Scenario 7 — Multimodal Vision Analysis (CLI, REPL & Server)

```bash
# One-shot image analysis (loads the ViT vision encoder automatically)
qwg run Qwen3.8-27B-hq4g32 --image ./diagram.png --prompt "Summarize the architecture in this diagram."

# Interactive multi-turn vision in REPL (use `/image <path>` inside the session)
qwg repl Qwen3.6-35B-A3B-hq5g64-hq4g64 --load-vision
```

### Scenario 8 — Hermetic A/B Benchmarking & Deterministic Evaluation

When measuring kernel or model changes (see `Scripts/paired_perf.py --strict`), bypass saved profiles, disable stochastic sampling, and fix the seed:

```bash
qwg run Qwen3.6-35B-A3B-hq5g64-hq4g64 \
  --no-profile \
  --kv-mode q4 -c 32768 --prefill-chunk-size 4096 \
  --lsa-mode auto --no-load-vision \
  --temperature 0 --seed 42 --no-thinking \
  --mtp-drafts 0 --max-tokens 128 --verbose \
  --prompt "Write a branchless radix sort in Swift."
```

---

## 3. Recommended Presets by Flagship Model (`128 GB` Apple M5 Max)

| Model Pack | Resident Metal Footprint | Context & KV | Sparse Attention (`--lsa-mode auto`) | Recommended Speculation | Typical Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen3.6-35B-A3B`** (`HQ5/HQ4`) | `~20.1 GiB` | `q4` · `32K–1M` | `model.qlsa` · `Ward-512` (`--lsa-cluster-block 512`) | `variational` (`3` drafts) or `standard` | `2,317–6,389` pre · `105–233 tok/s` dec |
| **`Qwen3.8-Flash-Next`** (`HQ8/HQ4` + `VQ2` PLE) | `98.9 GiB` (`71.6 GiB` w/ `--stream-ngrams`) | `q4` · `32K–1M` | Flat QSA `<= 262K` · auto `e-means-4096` `> 262K` | `variational` (`3` drafts) or `standard` | `1,888–2,696` pre · `64–145 tok/s` dec |
| **`Qwen3.8-27B`** (`HQ4_G32`) | `~16.8 GiB` (`+0.6 GiB` DSpine) | `q4` · `32K–524K` | `model.qlsa` (`4096` max prefill, `512` max `>=262K` dec) | `dspine` (`--dflash-window 16`, `15` drafts) | `831–914` pre · `100.5–134.5 tok/s` dec |
| **`GLM-5.3-Flash`** (`321B`, `SHVQ4/2/3`) | `~101.8 GiB` (`100%` resident) | `q4` · `16K–242.5K+` (`1M` UMA) | `< 41K` flat bypass · `>= 41K` `e-means` coarse | `standard` (`1` draft, NextN MTP) | **`825–1,006` pre · `46.0–53.6 tok/s` dec** |

---

## 4. Complete Knob Reference (`QWKnobRegistry` — 49 Controls)

Every control below can be set in `~/.qwg/profiles/<model>.json` or via `/v1/engine/info` / `/v1/settings`; CLI and Menu Bar columns show the exact user-facing control on each surface.

### 4.1 Sampling, Chat & Reasoning (`request` scope)

| Knob ID | CLI Flag (`run` / `repl` / `serve`) | Menu Bar Tab | Default | Valid Values | When to Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature` | `--temperature` (`0.0` default in `run`/`repl`) | Sampling | `0.7` | `0.0...2.0` | Set `0` for deterministic coding, math, and benchmarks; `0.6–0.8` for creative chat. |
| `topP` | `--top-p` | Sampling | `0.9` | `0.0...1.0` | Nucleus probability cutoff when `temperature > 0`. |
| `topK` | `--top-k` | Sampling | `40` | `0...2048` | Top-K candidate cutoff (`0` disables cutoff). |
| `repetitionPenalty` | `--repetition-penalty` | Sampling | `1.05` | `1.0...2.0` | Set `1.0` for strict parity / code generation; `1.05–1.10` to curb chat loops. |
| `maxTokens` | `--max-tokens` | Sampling | `512` | `1...1048576` | Maximum generated tokens per response (`<think>` + final answer). |
| `seed` | `--seed` | Sampling | `-1` | `-1...Int32.max` | `-1` chooses a fresh random seed; `>= 0` makes sampling reproducible. |
| `enableThinking` | `--thinking` / `--no-thinking` (`--enable-thinking`) | Sampling | `true` | `true`, `false` | Toggle `<think>...</think>` chain-of-thought reasoning blocks. |
| `reasoningEffort` | `--reasoning-effort` | Sampling | `"xhigh"` | `low`, `medium`, `xhigh` | Controls reasoning budget injected into templates supporting effort tiers. |
| `systemPrompt` | `--system-prompt`, `--system` | Sampling | `""` | Free-form text | Default system instruction prepended when a client omits a `system` message. |

### 4.2 Context, KV Cache & Prefix Caching (`load` / `session` scope)

| Knob ID | CLI Flag | Menu Bar Tab | Default | Valid Values | When to Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `contextTokens` | `--context-tokens`, `--max-context` (`run`/`repl`/`serve`/`plan`) | Model | `4096` (or model native in CLI) | `1...1048576` | Raise to `65536`–`1048576` for large repos or documents; YaRN scales automatically beyond native window. |
| `kvMode` | `--kv-mode` (`run`/`repl`/`serve`/`plan`) | Model | `"q4"` | `q4`, `q8`, `f16` | Keep **`q4`** (`4×` RAM savings, `.qwcache` compatible). Use `q8` or `f16` only for quantization sensitivity ablations. |
| `prefixCacheEnabled` | `--prefix-cache` / `--no-prefix-cache` (`run`) | *(Always active in `serve` when limit > 0)* | `false` (`run`) | `true`, `false` | Persist single-turn `qwg run` prompt prefixes to `--cache-dir`. |
| `warmCacheLimitGiB` | `--warm-cache-limit` (`repl`/`serve`) | KV Cache | `4.0` GiB | `0...512` GiB | Warm RAM multi-turn prefix cache budget (automatically clamped to `<= 25%` of post-load UMA headroom; `0` disables). |
| `ssdCacheLimitGiB` | `--ssd-cache-limit` (`repl`/`serve`) | KV Cache | `10.0` GiB | `0...4096` GiB | Cold NVMe SSD tier for evicted prefix KV snapshots (`0` disables disk persistence). |
| `cacheDir` | `--cache-dir` (`run`/`repl`/`serve`) | KV Cache | `~/.qwg/kv-cache` | Path | Directory holding persisted SSD prefix snapshots. |
| `cacheEntryLimit` | *(Profile / API)* | *(Internal)* | `32` | `1...4096` | Maximum saved prefix entries retained in the index. |
| `ropeScalingFactor` | *(Profile / API / `QW_ROPE_YARN_FACTOR`)* | *(Auto-derived)* | `1.0` | `1.0...8.0` | Leave untouched; Qwegn derives YaRN scaling automatically whenever `contextTokens > max_position_embeddings`. |

### 4.3 Latent Sparse Attention (`LSA` & `e-means` — `load` scope)

| Knob ID | CLI Flag (`run` / `repl` / `serve`) | Menu Bar Tab | Default | Valid Values | When to Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lsaMode` | `--lsa-mode` | Performance | `"auto"` | `off`, `auto`, `sidecar`, `svd`, `qsa-cluster` | Keep **`auto`**. Use `off` only to force dense GQA or `exactFlat` MLA baselines; use `qsa-cluster` to force `e-means-4096` on QSA at `<= 262K`. |
| `lsaBudget` | `--lsa-budget` | Performance | `2048` | `512...8192` | Active middle-token attention budget `k` (plus `4` sink + `512` local sliding window tokens). Raise to `4096` for ultra-dense multi-hop retrieval. |
| `lsaClusterBlock` | `--lsa-cluster-block` | Performance | `512` | `256...4096` | `512` selects Ward-64 clustering (fastest decode on narrow-KV MoE); **`4096` selects capacity-balanced `e-means`** (fastest prefill on dense 27B and `>262K` QSA). |
| `lsaSidecarPath` | `--lsa-sidecar-path` | Performance | `""` | `.qlsa` file path | Optional override when storing `model.qlsa` outside the model pack directory. |

### 4.4 Speculative Decoding (`MTP`, `Variational`, `DFlash 2`, `DSpine`, `MoE-Spec` — `load` scope)

| Knob ID | CLI Flag (`run` / `repl` / `serve`) | Menu Bar Tab | Default | Valid Values | When to Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `speculativeEnabled` | *(Implied by `--mtp-drafts > 0`)* | Speculative | `true` | `true`, `false` | Master switch in Menu Bar / profiles; on CLI pass `--mtp-drafts 0` for serial decode. |
| `mtpBackend` | `--mtp-backend` | Speculative | `"variational"` | `standard`, `variational`, `dflash2`, `dspine` | `standard` = lossless exact verify; `variational` = higher acceptance via log-margin; `dspine` (`B=16`) / `dflash2` (`B=8`) = external dense block drafters. |
| `mtpDrafts` | `--mtp-drafts` | Speculative | `1` (`3` Server, `7` DFlash2, `15` DSpine) | `0...15` | `0` = serial decode; `1` for NextN MLA; `3` for native MTP; `7` for DFlash 2; `15` for DSpine (`B=16`). |
| `mtpDraftStrategy` | `--mtp-draft-strategy` | Speculative | `"hybrid"` | `hybrid`, `neural`, `ngram` | `hybrid` combines neural MTP heads with zero-cost prompt N-gram lookup for code/repo edits. |
| `mtpRelaxedMargin` | `--mtp-relaxed-margin` | Speculative | `1.5` | `0.0...10.0` | Variational log-margin `δ` in nats (`0` = strict exact argmax; `1.5–2.0` boosts chat/code throughput). |
| `mtpRelaxedMinProb` | `--mtp-relaxed-min-prob` | Speculative | `0.2` | `0.0...1.0` | Fidelity floor `τ`: rejects relaxed drafts when target verifier probability `< τ`, preventing repetition drift. |
| `mtpQuadraticSeeding` | `--mtp-quadratic-seeding` / `--no-...` | Speculative | `true` | `true`, `false` | Seeds MTP KV state in one batched pass on all-accept rounds. Keep `true`. |
| `mtpAdaptiveDepth` | `--mtp-adaptive-depth` / `--no-...` | Speculative | `false` | `true`, `false` | Dynamically truncates draft depth when drafter entropy exceeds `mtpEntropyThreshold`. |
| `mtpEntropyThreshold` | `--mtp-entropy-threshold` | Speculative | `0.7` | `0.0...1.0` | Confidence cutoff used when `mtpAdaptiveDepth` is enabled. |
| `dflashDraftPackPath` | `--dflash-draft-pack` | Speculative | `""` (auto-discovered) | Pack directory | Path to an installed `DFlash 2` or `DSpine` drafter pack in `~/.qwg/models`. |
| `dflashSelectorTopK` | `--dflash-top-k` | Speculative | `16` | `1...16` | Candidate selector Top-K inside DFlash 2 / DSpine bridge layers (`16` recommended). |
| `moeSpecBudget` | `--moe-spec-budget` | Performance (`MoE`) | `0` | `0...8` | `0` = lossless verification of all routed experts; `1...8` caps unique cold SSD expert reads per verify layer when streaming oversized MoE packs. |

### 4.5 Unified Memory Residency, MoE Streaming & Server Process

| Knob ID | CLI Flag | Menu Bar Tab | Default | Valid Values | When to Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `memoryPolicy` | `--memory-policy` (`run`/`repl`/`serve`) | Performance | `"maxResident"` | `maxResident`, `safe` | `maxResident` wires maximum weights into physical RAM minus OS reserve (fastest); **`safe`** sizes strictly to currently free/reclaimable memory minus `6 GiB` headroom (**zero swap**). |
| `osReserveGiB` | `--os-reserve` (`run`/`repl`/`serve`) | Performance | `8.0` GiB | `0...256` GiB | RAM reserved for macOS and co-running desktop apps (`16–24 GiB` recommended when multitasking). |
| `loadVision` | `--load-vision` / `--no-load-vision` | Model | `false` | `true`, `false` | Keep `false` (`-1.60 GiB` RSS saved); toggles live in `Qwegn.app` without resetting KV cache! |
| `streamNgrams` | `--stream-ngrams` / `--no-stream-ngrams` | Performance | `false` | `true`, `false` | Enable on `Flash-Next` (`ngram.bin`) to free **`~27.3 GiB` GPU RAM** via NVMe + `64 MiB` L1 streaming. Auto-enabled if context exceeds UMA headroom. |
| `evictPrefillScratch` | `--evict-prefill-scratch` (`run`/`repl`/`serve`) | Performance | `"auto"` | `auto`, `always`, `never` | `auto` retains prefill scratch on Macs `> 36.5 GB` RAM unless wired engine arena `>= 75%` of physical RAM (where eviction prevents UMA paging); set `always` to reclaim `1–2 GiB` immediately after prompt prefill. |
| `prefillChunkRows` | `--prefill-chunk-size` (`run`/`repl`/`serve`) | Performance | `512` (`4096` MoE on `>36.5 GB`) | `32...8192` | `4096` maximizes MoE prefill throughput; lower to `512–1024` to reduce peak activation scratch RAM. |
| `maxConcurrentSessions` | `--max-concurrent-sessions` (`serve`) | Performance | `8` | `1...16` | Number of isolated warm KV cache slots protected against mutual eviction when running parallel coding subagents. |
| `residentExpertsPerLayer`| `--resident-experts-per-layer` (`run`/`repl`/`serve`) | Performance (`MoE`) | `0` (auto) | `0...4096` | `0` sizes hot resident experts automatically from `memoryPolicy`; set `> 0` to pin an exact hot-expert count per MoE layer. |
| `expertCache` | *(Env `QW_EXPERT_CACHE`)* | Performance (`MoE`) | `true` | `true`, `false` | Cross-token GPU LRU cache for streamed cold experts. Keep `true`. |
| `coldEvents` | *(Env `QW_COLD_EVENTS`)* | Performance (`MoE`) | `true` | `true`, `false` | `MTLSharedEvent` pipelining for NVMe cold-expert reads during decode. Keep `true`. |
| `coldEventOverlap` | *(Env `QW_COLD_EVENT_OVERLAP`)* | Performance (`MoE`) | `true` | `true`, `false` | Overlaps GPU execution of resident experts with NVMe staging of cold experts. Keep `true`. |
| `prefillSpanMap` | *(Env `QW_PREFILL_SPAN_MAP`)* | Performance (`MoE`) | `true` | `true`, `false` | Coalesces per-chunk MoE expert spans on GPU during prefill. Keep `true`. |
| `gdnPrefill` | `--gdn-prefill` (`run`/`serve`) | *(CLI Expert)* | `"chunked"` | `chunked`, `regres`, `naive` | Leave on `"chunked"` (WY parallel chunked recurrent solver). |
| `splitKTokensPerSplit` | *(Profile / API)* | *(Internal)* | `256` | `1...8192` | Internal split-K partition granularity. |
| `lazyLoad` | `--lazy` / `--no-lazy` (`serve`) | General | `true` | `true`, `false` | Defers weight loading until the first HTTP request arrives (`--no-lazy` pre-warms immediately at startup). |
| `logLevel` | `--log-level` (`serve`) | Logging | `"info"` | `debug`, `info`, `warn`, `error`, `none` | Server diagnostic verbosity written to `~/.qwg/logs/qwg.log`. |

---

## 5. Subcommand CLI Flags & Interactive REPL Commands

### Key Non-Registry Flags (`qwg run`, `qwg repl`, `qwg serve`, `qwg index`)

| Flag | Commands | Description |
| :--- | :--- | :--- |
| `-m, --model <name\|alias\|path>` | `run`, `repl`, `serve`, `index` | Select a model pack in `~/.qwg/models`, a configured alias, or a direct directory path. |
| `--dflash-window <2...16>` | `run`, `repl`, `serve` | Shorthand for block drafters: inspects the drafter pack (`dspine` vs `dflash2`) and sets `--mtp-drafts (B - 1)` (`16` = `DSpine-b16`). |
| `--repo <path>` / `--no-memory-pack` | `run`, `repl`, `serve` | Mount `<path>/.qwg/manifest.json` codebase memory pack, or disable auto-discovery in the current working directory. |
| `--no-profile` | `run`, `repl`, `serve` | Ignore `~/.qwg/settings.json` and `~/.qwg/profiles/*.json` (hermetic benchmark mode). |
| `--image <path>` / `--video <path>` | `run`, `repl` | Attach multimodal image or video input to the prompt. |
| `--show-thinking` / `--hide-thinking` | `run`, `repl` | Display or suppress `<think>` reasoning text on terminal stdout while keeping internal reasoning active. |
| `--verbose` | `run`, `repl` | Print load time, prefill `tok/s`, TTFT, decode `tok/s`, prefix-cache hit tier, and arena footprint after generation. |
| `--host <ip>` / `--port <n>` | `serve` | Network bind address and port (default `127.0.0.1:8080`). |

### Interactive `qwg repl` Slash Commands

Inside `qwg repl`, type `/help` at any prompt:
- `/memory` · `/memory mount <repo>` · `/memory unmount` — Inspect, hot-mount, or unmount a `.qwcache` repository memory pack.
- `/cache` — Inspect warm RAM and cold SSD prefix cache utilization.
- `/image <path>` — Attach an image to the next turn.
- `/thinking [on|off]` · `/effort <low|medium|xhigh>` · `/tokens <n>` · `/system <text>` — Adjust generation settings live.
- `/stats` — Toggle per-turn prefill/decode/TTFT telemetry readout.
- `/clear` — Reset conversation history while keeping the model resident in GPU memory.

---

## 6. Environment Variables (`QW_*` & `QWG_*`)

### 6.1 Operator & CI Overrides

| Variable | Equivalent CLI / Setting | Purpose |
| :--- | :--- | :--- |
| `QWG_NO_PROFILE=1` | `--no-profile` | Bypass saved user/model profiles across `qwg` processes. |
| `QWG_CONFIG=<path>` | `--config <path>` | Load an alternate `settings.json` file in `qwg serve`. |
| `QWG_MODELS_DIR=<dir>` | `~/.qwg/models` | Override the root directory scanned for installed model packs. |
| `QW_MEMORY_POLICY=safe\|maxResident` | `--memory-policy` | Override unified-memory residency policy via environment. |
| `QW_OS_RESERVE_GIB=<num>` | `--os-reserve` | Override OS RAM reserve in GiB. |
| `QW_RESIDENT_EXPERTS_PER_LAYER=<n>` | `--resident-experts-per-layer` | Pin resident hot experts per MoE layer. |
| `QW_MOE_SPEC_EXPERT_BUDGET=<0..8\|auto>` | `--moe-spec-budget` | Cap cold SSD expert reads during MoE speculative verification (`auto` adapts to acceptance rate). |
| `QW_STREAM_NGRAMS=1` | `--stream-ngrams` | Force NVMe streaming for external PLE `ngram.bin` tables. |
| `QWG_LOAD_VISION=1` | `--load-vision` | Eagerly load the vision encoder at startup. |
| `QW_MLA_EMEANS=0\|1` | `--lsa-mode off\|auto` | Explicit override for MLA `e-means` coarse pruning (`auto` enables it automatically at `>= 32K` capacity with `< 41K` flat bypass). |

### 6.2 Internal Kernel Defaults & Engineering Trace Probes

- **Always-On Validated Kernel Paths** (`QW_MLA_ATTN_MMA`, `QW_MLA_ATTN_NAX`, `QW_MLA_IDX_SPLIT`, `QW_MLA_ATTN_SPLIT`, `QW_KDA_GATES_MMA`, `QW_MHC_SPLIT`, `QWG_PLE_GPU_GATHER`, `QWG_KEEPALIVE`, `QW_EXPERT_CACHE`, `QW_COLD_EVENTS`, `QW_COLD_EVENT_OVERLAP`, `QW_PREFILL_SPAN_MAP`): **Already enabled by default.** Set to `0` only when running kernel ablation diagnostics.
- **Developer Trace Probes** (`QW_FLASH_TRACE=1`, `QW_DECODE_PROFILE=1`, `QW_PREFILL_PROFILE=1`, `QW_VERIFY_TIMING=1`, `QW_LSA_TIMING=1`, `QW_PHASES=1`): Emit per-buffer Metal GPU timestamps and stage breakdowns to `stderr`. Leave unset during normal use and final `paired_perf.py` runs, as trace serialization perturbs host submission timing.
