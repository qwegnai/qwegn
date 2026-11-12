# Changelog

All notable changes to Qwegn (`qwg`, `qwg-menu`) are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/). Until 1.0, a minor bump
may change CLI flags, pack formats or profile files.

The single source of the version is `QWVersion.current` in
`Sources/Foundation/QWCore/QWVersion.swift`; release tags are `vMAJOR.MINOR.PATCH`.

## [Unreleased]

## [0.17.1] - 2026-11-12

### Fixed
- **Non-special XML delimiter token deduplication in `QWToolCallConstraint` (`QWServer`).**
  - Deduplicated and range-clamped `QWToolCallConstraint.allowedTokens()` so models whose `tokenizer.json` marks XML tool delimiters (`</tool_call>`, `<tool_call>`) with `"special": false` (such as `GLM-5.3-Flash` and `GLM-4.7`) never emit duplicate token IDs in `.closingTag` that trigger `samplingFailed` HTTP 500s under `tool_choice: "required"` or named tool forcing.
- **Parallel `<tool_call>` boundary matching with embedded literal tags in `QWToolCallParser` (`QWServer`).**
  - Replaced backward-only `</tool_call>` scanning on the final segment with forward `matchingToolCallClose(in:)` so adjacent parallel tool calls (`<tool_call>...</tool_call><tool_call>...</tool_call>`) never merge across boundaries while still preserving literal `<tool_call>...</tool_call>` substrings inside tool-call argument bodies.
- **High-water (`>= 75%` physical RAM) automatic `PrefillScratch` eviction across `DenseModule` & `HybridMoEModule` (`QWEngine`).**
  - Wired `evictPrefillScratch()` into `DenseModule.encodePrefill` and extended `QWPrefillScratchEvictionPolicy.auto` (`shouldEvict(totalRAMBytes:wiredEngineBytes:)`) to evict transient prefill scratch whenever the wired engine arena consumes `>= 75%` of physical RAM (such as `~101 GiB` models on `128 GB` Macs), eliminating sustained-load Metal residency paging dips (`19.2 tok/s → 43.3 tok/s`, `+5.4%` steady over `21m`).

### Verified
- **Full `llmprobe v0.6.16` `--full` (`1,489,839` tokens) & 100-Task `code` Qualification on `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64`:**
  - **`184 / 184` (`100%`) API conformance** (`9/9` core endpoints, `0` errors), **`37 / 37` (`100%`, `strong`) capability**, **`81 / 100` (`81.0%`) HumanEval+MBPP** (`46/50` `92.0%` HumanEval, `35/50` `70.0%` MBPP), **`66 / 88` (`75.0%`) agentic execution**, **`99.74%` engine fidelity** (`36/36` correctness, `3/3` greedy determinism, `36/36` logprob argmax consistency), **`53.6 tok/s` predictable / `45.6 tok/s` novel** NextN MTP-1 (`1.98 tok/step`), **`825.3 tok/s` `9.4K` prefill** (`780 tok/s` @`65.6K`), **`96×` (`1,728 → 18 ms`) warm prefix cache**, and **steady `41.1 → 43.3 tok/s` (`+5.4%`) 21-minute sustained thermal load**; updated all GLM infographics and documentation.

## [0.17.0] - 2026-11-12

### Changed
- **Generalized NoPE MLA MMA/NAX weight & paged-KV acceleration across `SHVQ{1,15,2,3,4}`, `AQ/HQ{2..8}`, and `q8`/`q4` KV (`QWAttn`, `QWMetal`).**
  - Generalized `qwg_mla_absorb_q(_mma[_sub2])`, `qwg_mla_proj_uv(_mma[_sub2])`, `qwg_mla_sparse_attn_mma`, and Apple10 NAX `qwg_mla_sparse_attn_nax(_split)` via `qw_mla_dequant.metalh` (`ShvqMlaTgLut` threadgroup SRAM staging + AGX-isolated `22 KiB` / `18 KiB <= 32 KiB` constant LUT PSOs + inline `q8`/`q4` paged latent dequantization) and vectorized `qwg_mla_absorb_q[_sub2]` (`128` quads × `4` row-slices), eliminating scalar fallbacks on non-incumbent `kv_b_proj` weights and quantized KV modes across prefill, serial decode, and MTP verify (`mla.sparse_attn` `26,035 → 487 ms` `53.5×`, `mla.absorb_q` `1,115 → 53 ms` `21.1×`, `mla.proj_uv` `924 → 75 ms` `12.3×`, decode MLA `20.44 → 1.72 ms/tok` `11.9×`).
- **Generalized `SHVQ{1,15,2,3,4}` direct-half & threadgroup SRAM LUT kernels (`QWMetal`, `QWGEMM`).**
  - Enabled `QW_SHVQ_ROUTED_DIRECT_LUT` (`kShvqLut4DirectBits` / `kShvqLut15DirectBits`, exact `32.000 KiB <= 32 KiB` per specialized PSO) across dense, SwiGLU MLP, GDN, HyperConnection, batched GEMV, and NAX GEMM kernels, and staged `ShvqTgLut<1|2|3>` (`2–4 KiB`) in `threadgroup` SRAM for dense `SHVQ1/2/3` NAX GEMMs and multi-row GEMVs while keeping grouped MoE NAX on `2–4 KiB` resident AGX constant LUTs (`2.23–2.24×` faster cold-streaming `prefill_nax_f32` `SHVQ2/3`).
- **`PrefillScratch` & `MoEScratch` UMA tightening (`-2.65 GiB`), 1-draft `QWMTPEmbeddingTable` gating, and subquadratic `> 131K` `4,096`-row prefill chunking (`QWEngine`, `QWServer`).**
  - Removed 11 dead `PrefillScratch` allocations and unused `gdnY`/dense-attn scratch on `gdnVectorDecay` / `usesNoPEMLA`, tightened grouped-NAX `MoEScratch` (`r1Rows <= 8`, `.f16` `dnAcc`, `prefillBytesPerToken` `1.60 MB → 556 KB` on `GLM-5.3-Flash`, `760 KB → 500 KB` on `Qwen3.8-27B`), and unified `decodeAndMTPScratchBytes` + `mlaFixedScratchBytes` across `QWEngineDriver` and `DriverResidency` so out-of-the-box `--no-profile` CLI runs keep `288/288` experts resident (`fixed_scratch` `6.96 → 4.31 GiB` serial, `7.11 → 4.48 GiB` MTP-1; `-1.02 GiB` on `Qwen3.6-35B-A3B` MTP-1).
  - Gated automatic `QWMTPEmbeddingTable` (`1.48 GiB` `MAP_SHARED` `MTLResidencySet`) to `mtpDrafts > 1` (plus `POSIX_MADV_WILLNEED` when active), eliminating VNODE page-fault stalls (`p95 896 → 39 ms`) on single-draft NextN/MTP-1 speculation where only CPU `embed_tokens` is read.
  - Exempted subquadratic sparse attention paths (`usesNoPEMLA` and active `QWLSAOperator`) from the `> 131K` `256/128`-row quadratic chunk clamp (preserving full `4,096`-row NAX tiles up to `1M` context with 4-layer command-buffer slicing above `131K`).

### Verified
- **Cross-Flagship 4-Pack Qualification (`AGENTS.md` Law 9):**
  - Paired, interleaved `--strict` A/B (`0%/0%` quiet GPU) lifts `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`q4` KV) prefill **`145.8 → 1,005.9 tok/s` (`6.92×`, TTFT `35.64 → 5.17 s`, `1,061.3 tok/s` GPU @4K, `851.8–869.2 tok/s` `qwg serve`)**, serial decode **`23.81 → 44.64 tok/s` (`1.88×` CLI; `46.0` med / `46.4` peak `qwg serve`)**, exact NextN MTP-1 **`23.44 → 47.59 tok/s` (`2.03×` paired CLI, `51.19` standalone, `50.7` peak `qwg serve`, `10 ms` `193.4×` warm cache)**, and `@242.5K` `3/3`-needle long context to **`743.0 / 44.98 tok/s` (`e-means`, `6.29×` prefill & `1.69×` decode vs `@160K` `v0.14.1`)**, with `0.0%` regression (`0.994×–1.008×`), `5/5` bit-exact `SHA-256` determinism, `3/3` needle fidelity, and `9/10` `llmprobe` `code-10` across `GLM-5.3-Flash`, `Qwen3.6-35B-A3B`, `Qwen3.8-Flash-Next`, and `Qwen3.8-27B` (+ `DSpine-b16`).

## [0.16.0] - 2026-11-12

### Added
- **Sub-2-Bit Spherical Harmonic Vector Quantization (`SHVQ15`, `SHVQ1`) & Multi-Format `DFlash 2` / `DSpine` Speculative Drafters (`QWQuant`, `QWMetal`, `QWEngine`, `Scripts/quantize/`).**
  - Added `SHVQ15` (1.5-bit 8D 2048-centroid antipodal spherical $e$-means VQ with 0-ULP dyadic scale `int8 * 3 * 2^-7`, 1.75 bpw at G64, 1.625 bpw at G128) and `SHVQ1` (1.0-bit 8D 128-centroid antipodal spherical $e$-means VQ, 1.25 bpw at G64, 1.125 bpw at G128).
  - Shipped Metal 4 and NAX dequantization headers (`qw_dequant_shvq_sub2.metalh`, `qw_shvq_sub2_tables.metalh`, `qw_shvq_sub2_routed_lut.metalh`), Swift table bindings (`QWSHVQSub2Tables.swift`), and offline quantization pipelines (`shvq_sub2.py`, `shvq_metal_enc.mm`).
  - Isolated AGX `<= 32 KiB` constant-cache PSOs via `_sub2` kernel variants (`qwg_mla_absorb_q_sub2`, `qwg_dspine_refine_scores_sub2`, and SwiGLU / MoE / GDN dequantizers) to prevent spilling to slow device memory, maintaining peak throughput across all existing formats.
  - Enabled multi-format speculative drafters in `QWDFlashDrafter+Quant.swift` and `pack_dflash.py` (`HQ*`, `AQ*`, `SHVQ{1,15,2,3,4}`): speculative drafters achieve a mathematically verified 0.0% target quality loss (bit-exact greedy token stream parity) while slashing drafter UMA by -42.0% (`DSpine-b16-shvq15g64` 925.22 MiB vs 1,595.53 MiB) and speeding up serial decode by 1.229× at $B=8$.
- **4D PLE N-gram Vector Quantization (`QNGRAM v3 VQ4_G32`) (`QWNgramFormat.swift`, `qwple.metal`, `pack_ngram_vq4.py`, `FNGPUPLEEncoders.swift`).**
  - Added 4D spherical $e$-means vector quantization (`VQ4_G32`, 50 B/row, 2.50 bpw) for HyperPLE n-gram tables.
  - Slashes `Qwen3.8-Flash-Next` n-gram table footprint from 27.13 GiB to 15.07 GiB (-12.06 GiB UMA saved) while preserving 0.9557 mean cosine similarity and accelerating 480×512 prefill gather by +9.0% (0.4672 ms vs 0.5088 ms).

### Verified
- **Zero-Regression Flagship Qualification (`AGENTS.md` Law 9):**
  - Bit-exact token stream parity, 0.0% UMA degradation, and 9/10 `llmprobe` code-10 scores across all 4 flagship model packs: `Qwen3.6-35B-A3B-hq4g32` (1.0028× vs master), `Qwen3.8-27B-hq4g32` (1.0040× vs master), `Qwen3.8-Flash-Next-hq8g32-hq4g64` (1.0235× vs master), and `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (0.9889× vs master).

## [0.15.1] - 2026-11-11

### Fixed
- **GitHub Actions CI paravirtual GPU & host environment test gating and `QWCliTests` Swift Testing macro fixes (`Scripts/ci.sh`, `Tests/`).**
  - Fixed Swift Testing `#expect(Bool(true))` warning in `PullProgressTests.swift` and `#require` macro expansion overload ambiguity in `KnobBindingTests.swift` across Xcode 26 minor toolchains.
  - Gated bare-metal Apple Silicon GPU (`isMetal4TensorCapable` NAX cooperative-tensor, 1024-thread GDN recurrent, and 49K-context multi-TG LSA selector) and host environment (`>= 110 GiB` physical RAM, local `~/.qwg/models`, uncontaminated process `ru_maxrss`, and bare-metal timing) tests with `XCTSkipUnless`, and skipped executing `qwg-bench-kernels` in `Scripts/ci.sh` Phase 2 when `GITHUB_ACTIONS=true` while preserving full test and benchmark execution on real Apple Silicon hardware.

## [0.15.0] - 2026-11-11

### Added
- **CLI residency & MoE speculation handles (`--memory-policy`, `--resident-experts-per-layer`, `--moe-spec-budget`), `HQ8_G32_M5` Weight-SVD dequantization, `Qwegn.app` `DSpine` / `streamNgrams` / `moeSpecBudget` controls, and tracked [`docs/settings-reference.md`](docs/settings-reference.md) (`QWCore`, `QWQuant`, `QWEngine`, `QWServer`, `qwg`, `QWMenuBar`, `docs/`).**
  - Promoted `memoryPolicy` (`--memory-policy maxResident|safe`) and `residentExpertsPerLayer` (`--resident-experts-per-layer`) to first-class CLI options via `QWKnobRegistry` (`normalSurfaces`) and canonical case-insensitive enum transforms in `Scripts/gen_knob_cli.py`, and registered `moeSpecBudget` (`--moe-spec-budget 0...8`, `QWEngineTuning.moeSpecBudget`, `QW_MOE_SPEC_EXPERT_BUDGET`) across `qwg run`, `qwg repl`, and `qwg serve`.
  - Added inverse-FWHT `QWQuant.Pack.dequantHQ8` (`HQ8_G32_M5`) to `QWLSASidecar.synthesizeFromPack` so every `.qwp` pack format supports automatic in-memory Weight-SVD under `--lsa-mode auto`.
  - Extracted `SettingsView+Speculative.swift` in `QWMenuBar` (`Qwegn.app`), adding `DSpine` (`B = 16`, up to `15` drafts) with automatic drafter `config.json` pack syncing, plus `Stream PLE N-gram Table from SSD` (`streamNgrams`, `-27.3 GiB` GPU RAM) and `MoE-Spec Verify Budget` (`moeSpecBudget`) in `SettingsView+Performance.swift`.
  - Added tracked open-source [`docs/settings-reference.md`](docs/settings-reference.md) covering zero-config defaults, 8 copy-paste operational scenarios, flagship presets, all `49` `QWKnobRegistry` controls, CLI/REPL flags, and environment variables.

### Changed
- **Adaptive out-of-the-box defaults (`--kv-mode q4`, `--lsa-mode auto`), `>262K` QSA `e-means-4096` auto-escalation, `<41K`-bypassed MLA `e-means` coarse pruning, and `config.json`-driven drafter auto-detection (`QWKnobRegistry`, `QWLSAConfig`, `QWMLAAttnOperator`, `FNGPUEngine`, `QWDFlashConfig`, `QWServer`).**
  - Upgraded default `kvMode` from `"f16"` to `"q4"` (`4×` KV RAM savings, direct `.qwcache` compatibility) and default `lsaMode` from `"off"` to `"auto"` across `QWKnobRegistry`, `qwg`, `qwg plan`, and `QWServer`.
  - Under `lsaMode = "auto"`, `FNGPUEngine` and `QWLSAConfig.resolveBackend` retain native flat QSA up to `262,144` tokens and automatically enable `e-means-4096` coarse clustering above `262K`; `QWMLAAttnOperator` automatically enables `e-means` coarse pruning (`<41K` zero-Stage-1 flat bypass, `>=41K` `e-means`, `--lsa-mode off` / `QW_MLA_EMEANS=0` `exactFlat`); and `QWDFlashConfig` discovers and classifies `dspine` (`maxDrafts = 15`) vs `dflash2` (`maxDrafts = 7`) directly from drafter `assets/config.json` with zero folder-name heuristics.

## [0.14.1] - 2026-11-11

### Added
- **CLI & Menu Bar `--lsa-cluster-block 4096` (`e-means` macro-clustering) knob exposure and `64K → 1M` long-context LSA vs `e-means` benchmarks, whitepaper (§2.6 / §3.6), and presentation infographics (`QWCore`, `qwg`, `QWMenuBar`, `docs/`, `assets/`).**
  - Widened `QWKnobRegistry.lsaClusterBlock` (`--lsa-cluster-block`) from `256...1024` to `256...4096` (`4096` activates `QWCoarsePoolIndex` capacity-balanced spherical `e-means` macro-clustering) and added `4,096 (e-means)` to `qwg-menu` (`SettingsView+Performance.swift`).
  - Registered 44 quiet-GPU (`0% / 0%`) empirical runs from `16K` / `64K` to `1,039,928` (`1M`) tokens across `Qwen3.6-35B-A3B-hq4g32` (`2,317.0 / 105.56 tok/s` @1M `Ward-512`, `136.84 tok/s` @64K `e-means`), `Qwen3.8-Flash-Next-hq8g32-hq4g64` (`1,887.6 / 63.74 tok/s` @1M `e-means` — `+39.8%` prefill / `+55.3%` decode vs Native QSA), `Qwen3.8-27B-hq4g32` (`830.6–914.3 tok/s` prefill `+3.2%–9.0%` `e-means`), and `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`26.63 tok/s` @160K `+19.8%` `e-means`, `1M` `102.3 GiB` UMA admission) with **100% (`3/3`) 3-needle passkey recall** on every sparse run, plus 6 dual-panel Prefill Profile & Decode SVG infographics (`assets/qwegn_lsa_*.svg`) in `docs/benchmarks.md`, `docs/lsa.md`, and `docs/lsa-codebase-memory-whitepaper.md`.

## [0.14.0] - 2026-11-11

### Added
- **Spherical Hadamard Vector Quantization (`SHVQ{2,3,4}_G{32,64,128,256}`), `QNGRAM v3 VQ2_G32` 2-bit PLE tables (`50.9 → 27.1 GiB`), and zero-overspill Metal SHVQ quantizer (`QWQuant`, `QWGEMM`, `QWMLP`, `QWFlashNext`, `Scripts/quantize/shvq.py`, `pack_ngram.py`, `emeans.py`).**
  - Implemented 12 `SHVQ{2,3,4}_G{32,64,128,256}` e-means vector quantization formats (`arXiv:2611.09501`) across dense GEMV, greedy `lm_head` argmax, batched Apple10 NAX GEMM, `f32` BM/GEMV (`bm32` exact 32-row small-prefill tiles), and grouped/routed MoE kernels (`ShvqTgLut<2|3>` threadgroup SRAM codebook staging + `SHVQ4` 32 KiB direct half-LUT lookup), shrinking `GLM-5.3-Flash` (`321B`) from `134.37 GiB` (`3.59 bpp` cold-streamed) to **`100.63 GiB` (`2.69 bpp` `SHVQ4_G32 + SHVQ2_G64 + SHVQ3_G64`, `-33.74 GiB` / `-25.1%`)** so all `321B` weights reside **100% in 128 GB UMA** (`36.2 tok/s` decode, `44.0 tok/s` speculative, `0` SSD expert misses).
  - Added `QNGRAM v3 VQ2_G32` 2-bit e-means PLE N-gram table quantization (`170 → 90 bytes/row`, `50.86 → 27.13 GiB`, **`-23.73 GiB` / `-46.7%`**) with resident Metal GPU gather (`FNGPUPLETable` / `qwg_fn_ple_vq2_gather`) alongside SSD sector streaming, shrinking `Qwen3.8-Flash-Next-hq8g32-hq4g64` + PLE total disk footprint from `122.15 GiB` to **`98.42 GiB`** and enabling GPU-resident PLE alongside the `71.29 GiB` `HQ8+HQ4` pack (**`95.1 tok/s` decode `+13.9%` vs `83.5` `v0.13.1`, `145.4 tok/s` predictable `4.92 tok/step`, `2,528 tok/s` @10.4K prefill `+11.5%`**) with `10/10` (`100%`) `llmprobe` core reasoning and `9/9` structured JSON fidelity.
  - Added zero-overspill `146 MiB` Metal SHVQ encoder (`shvq_metal_enc.mm`) and fast `VQ2` N-gram packer in `Scripts/quantize/`.
- **Universal `--load-vision` / `--no-load-vision` and `--stream-ngrams` / `--no-stream-ngrams` CLI & runtime residency toggles (`QWKnobRegistry`, `qwg`, `QWEngineDriver`, `FNGPUEngine`, `FNGPUPLETable`), plus `--no-profile` hermetic benchmark mode (`paired_perf.py --strict`, `qwg-bench-kernels --only-shvq`).**
  - Exposed `--load-vision` / `--no-load-vision` (`-1.60 GiB` CPU RSS when vision planes stay unmapped) and `--stream-ngrams` / `--no-stream-ngrams` (`-27.30 GiB` Metal GPU RAM when VQ2 PLE streams from SSD with a `64 MiB` L1 cache) across `qwg run`, `qwg repl`, `qwg serve`, and `qwg bench`, saving **`-28.48 GiB` combined RAM** (`108.77 → 80.29 GiB`) on `Qwen3.8-Flash-Next` with zero throughput penalty (`110.5 tok/s` warm decode).
  - Added `--no-profile` across CLI commands and `Scripts/paired_perf.py --strict`, `QW_FLASH_TRACE=1` submission-interval diagnostics, and cache-hot + `256 MiB` ring-cycled cold-streaming `SHVQSuites` in `qwg-bench-kernels`.
- **Capacity-balanced e-means coarse pool index (`QWCoarsePoolIndex`), slow-EMA KV calibration, and semantic `QWMemoryVault` centroids (`QWAttn`, `QWState`, `QWEngine`).**
  - Added deterministic single-TG `qwg_emeans_collect`, on-device exact flat Top-K overflow fallback (`candState.overflow`), zero-Stage-1 flat bypass below `< 41K` tokens, and opt-in `QW_MLA_EMEANS=1` (keeping MLA `exactFlat` by default).

### Changed
- **Unified on-device indirect grouped-NAX MoE dispatch, dynamic 128 GB UMA admission, MoE `4096`-row prefill auto-tuning, and early Flash-Next decode submissions (`QWGEMM`, `QWHal`, `QWMLP`, `QWResidencyAdmission`, `PrefixCache`, `FNGPUEngine`).**
  - Unified `encodeMoEGateUpNax` and `encodeMoENaxSplitF32` onto on-device `MTLDispatchThreadgroupsIndirectArguments` written by `qwg_moe_group_pairs_f32` (`0` empty threadgroups/slices) across an explicit `QWHal.rotateEncoder` boundary for AGX indirect-grid coherency, lifting `GLM-5.3-Flash` multi-row verify determinism to `5/5` bit-exact SHA-256 and improving server throughput across all flagship packs (`+21.2%` decode & `+7.0%` prefill on `Qwen3.8-27B + DSpine-b16`, `+11.4%` 10K prefill on `Qwen3.6-35B-A3B`, `+4.9%` decode on `Qwen3.8-Flash-Next`).
  - Unified live `vm_statistics64` working-set admission (`QWResidencyAdmission`) across CLI load guards, `DriverConfig` MoE residency, and `qwg serve` `PrefixCache` warm-tier clamping (`<= 25%` of post-load headroom, `0 B` under pressure).
  - Restored early layer cuts `[0, 1, 8, layerCount]` for GPU-PLE Flash-Next decode/verify stacks (`+2.5–2.9%` paired decode throughput with `15/15` full-arena SHA-256 checkpoint identity) and extended exact `bm32` Apple10 prefill tiles across `HQ4`/`HQ8`/`SHVQ4` fused epilogues `0..3`.

### Fixed
- **Flash-Next `[memory]` weight accounting and `--stream-ngrams` GPU VQ2 residency gating (`FNGPUEngine`, `FNGPUPLEEncoders`).**
  - Fixed `FNGPUEngine` `[memory]` startup telemetry to report full `.qwp` weight residency (`71.29 GiB` instead of `3.33 GiB`) and wired `options.streamNgrams` into `FNGPUPLETable.resolveBacking` so `--stream-ngrams` strictly bypasses Metal GPU wiring on `VQ2_G32` PLE tables.

## [0.13.1] - 2026-11-09

### Fixed
- **macOS `.pkg` installer welcome/conclusion HTML character encoding (`Scripts/build_pkg.sh`).**
  - Replaced raw UTF-8 em-dashes (`—`, which macOS `Installer.app` misdecoded as `â€”` under Latin-1 HTML defaults) with strict 7-bit ASCII colons and explicit `<meta http-equiv="Content-Type" content="text/html; charset=utf-8">` / `<meta charset="utf-8">` headers, plus a build-time and `QWCliTests` non-ASCII guard.
- **GitHub Actions `macos-26` release & CI runner compatibility (`.github/workflows/release.yml`, `Scripts/build_pkg.sh`, `Scripts/build_app.sh`, `QWHal.swift`).**
  - Added `permissions: contents: write` and Xcode 26+ / `MetalToolchain` setup to `release.yml`, split single-valued `swift build -c release --product qwg` and `--product qwg-menu` invocations so SwiftPM builds both binaries, and switched `QWDeviceCaps.probe` to `MTLGPUFamily(rawValue: 1011)` so `QWHAL` compiles cleanly against both Xcode 26.6 (`macos-26` runner) and Xcode 27+ SDKs.

### Changed
- **Removed bloated `## News` section from `README.md`** (release history remains in `CHANGELOG.md`).

## [0.13.0] - 2026-11-09

### Added
- **macOS `.pkg` installer (`Qwegn-<version>-macos-arm64.pkg`), `Qwegn.app` bundle, `/usr/local/share/qwegn/uninstall.sh`, and lazy Python environment (`qwg quantize` / `qwg init --python`).**
  - Added `Scripts/build_app.sh` and `Scripts/gen_app_icon.swift` to package `qwg-menu` into a native `/Applications/Qwegn.app` bundle (`Info.plist`, `AppIcon.icns`, `qwegine_QWServer.bundle`, and bundled `kernel.metallib` / `kernel_std.metallib`), and updated `QWHal.defaultMetallibPath` and `QWHttpServer` static UI lookup to resolve bundled Metal libraries and web UI resources from `/Applications/Qwegn.app` and `/usr/local/share/qwegn/`.
  - Added `Scripts/build_pkg.sh` (`make pkg`, package identifier `ai.qwegn.pkg`) and `Scripts/uninstall.sh`, installing `/usr/local/bin/qwg`, `/Applications/Qwegn.app`, and `/usr/local/share/qwegn/` (`uninstall.sh`, quantizer Python modules, `requirements.txt`, `LICENSE`, `README.md`), with `postinstall` and Installer conclusion screens noting `/usr/local/share/qwegn/uninstall.sh` (`pkgutil --forget ai.qwegn.pkg` while preserving `~/.qwg`).
  - Added `QWCli.ensurePythonEnvironment` (`~/Library/Application Support/Qwegn/venv/`) and `qwg init --python` so `qwg quantize` lazily creates its Python virtual environment on first use (`"Creating Python environment…"`) without embedding absolute-path venvs inside the `.pkg`.
  - Updated `.github/workflows/release.yml` to build and attach both `Qwegn-<version>-macos-arm64.pkg` (main installer) and `qwegn-<tag>-macos-arm64.tar.gz` (portable archive), with optional Apple Developer ID signing and notarization via GitHub secrets.

## [0.12.0] - 2026-11-09

### Added
- **DSpine block-causal speculative decoding (`tencent/Qwen3.8-27B-DSpine-b16`, `arXiv:2609.36173`), custom fused Metal kernels (`qwdspine.metal`), and `M <= 16` (`D <= 15`) native block verification (`QWDFlashDrafter`, `QWVerifyWindow`, `QWGDNReplayCheckpoint`, `QWHead`, `QWMTPTDraftHead`, `QWMTPBatchedTrunk`, `QWMTPBatchedMLP`).**
  - Implemented six custom Metal 4 kernels in `Sources/QWMetal/mtp/qwdspine.metal` (`qwg_dspine_rms_norm_concat5`, `qwg_dspine_add_norm_split`, `qwg_dspine_conv1d_silu`, `qwg_dspine_qk_norm_rope_kv_write`, `qwg_dspine_gqa_block_attn`, `qwg_dspine_swiglu_down_res`) plus low-rank feature bridge (`fc1`/`fc2`, rank `1024`), `256`-dim KV cache (`4` KV heads, `head_dim=64`), causal depthwise `Conv1d(k=3) + SiLU`, and learned positional embeddings (`block_pos_emb[16, 4096]`) in `QWDFlashDrafter+DSpine.swift` (`Scripts/quantize/pack_dflash.py --arch dspine`).
  - Widened target-model verification and draft-block buffers (`QWGDNReplayCheckpoint.maxRows = 16`, `QWVerifyWindow`, `QWHead`, `QWMTPBatchedTrunk`, `QWMTPBatchedMLP`, `QWMTPTDraftHead`, `QWDFlashDrafter.maxBlockSize = 16`, `QWMTPBackend`) from `M <= 8` (`D <= 7`) to **`M <= 16` (`D <= 15`)** via balanced two-half `(c0, c1)` row-paired `HQ4`/`AQ4` dispatches, running `Qwen3.8-27B-DSpine-b16-hq4g32` (`1.63 GB`) at its full trained block horizon $B \in [2, 16]$ alongside `--mtp-backend dspine` across `qwg run`, `qwg repl`, and `qwg serve`.
  - Verified paired, interleaved M5 Max A/B sweeps across all block sizes $B \in [6, 16]$ with **100% bit-exact SHA-256 greedy parity**: **`134.47 tok/s` peak (`4.19×` vs `32.09 tok/s` serial, $\tau = 7.47$ tok/round, `92.4%` accept at $B=8$)**, **$\tau = 12.70$ tokens/round (`83.3%` accept, `118.53 tok/s` at $B=16$)**, **`93.86 tok/s` (`2.86×` vs `32.85 tok/s` serial on algorithm code at $B=6$)**, **`83.29 tok/s` (`2.54×` vs `32.83 tok/s` serial on technical prose at $B=6$)**, and **`100.47 tok/s` (`3.09×` serial) geometric mean** at default $B=6$ (`D=5`).

## [0.11.3] - 2026-11-09

### Fixed
- **Prefix-invariant multi-turn LSA Memory across live disk edits, autonomous tool loops (`assistant -> tool`), and `179K+`-token large-volume contexts (`QWMemoryVault`, `QWMemoryPack`, `QWChatTurnPreparer`, `QWEngineLifecycleActor`, `QwgReplAndServe`).**
  - Eliminated mid-prefix cache invalidation (`9.5%` hit / `311K`-token re-prefill after `patch` tool calls) by freezing historical Vault snapshots (`historicalSectionCache`) on completed turns and removing mutable `<|memory_delta|>` token splicing at offset `32,704` ahead of the conversation history.
  - Added causal Off-Context Vault recall on autonomous `assistant -> tool` steps (`extractAssistantToolStepContext`) while preserving monotonic `tool`-turn Vault attachments across subsequent `user` turns, and capped `buildCortexArchitectureMap` (`<= 24,000` chars, `-54.7%` map token overhead: `22,979 -> 10,415` tokens) with non-test file prioritization and a `20%` per-file Cortex cap.
  - Capped `QWMemoryVault.recallVaultContext` query scanning to `4,096` UTF-8 bytes (`prefix(2048) + "\n" + suffix(2048)`) and switched `encodeWireIncrementallyWithSystemPrefix` to contiguous `.utf8` byte views, accelerating `179,173`-token follow-up turn TTFT by **`31.0×` (`14,006.4 ms -> 452.6 ms`, `99.96%` Warm-RAM hit, `395,874.9 tok/s` effective prefill)** and preserving **`99.87%` prefix hits (`159.3 ms` TTFT, `307,018.2 tok/s`)** immediately after live disk edits.
  - Updated `docs/lsa-codebase-memory-whitepaper.md` and SVG infographics to present the final Two-Tier LSA Cortex + Off-Context Vault architecture and `179.2K`-token large-volume verification cleanly and concisely.

## [0.11.2] - 2026-11-08

### Changed
- **Power-of-2 Context Limit slider (`64` to `1,048,576` tokens) + live per-model memory footprint prediction & over-budget alert (`ContextLimitSliderRow`, `QWSettingsCoherence`, `QWEngineDriver.estimateContextMemory`, `QWContextMemoryEstimate`).**
  - Replaced the static menu picker (`"256k native"`, `"512k YaRN"`, `"1M YaRN"`) in `qwg-menu` (`SettingsView.swift`, `SettingsCoherenceViews.swift`) with a power-of-2 stepped slider spanning `64` to `1,048,576` (`1M`) tokens (`QWSettingsCoherence.contextSliderSteps`) and label-free token readouts (`QWSettingsCoherence.formatContextTokens`), matching the engine's config-sourced `max_position_embeddings` YaRN gate.
  - Added `QWEngineDriver.estimateContextMemory` and `QWServerSettings.estimateContextMemory` (`QWContextMemoryEstimate`) to compute resident model weights, KV cache (`f16` / `q8` / `q4`), and activation/GDN scratch against the host physical memory budget (`QWCapacityBudget`), surfacing a live memory readout, inline over-budget warning, and native confirmation alert (`"Clamp to <maxFitting>"` / `"Keep Anyway"`) whenever the selected context exceeds available memory.

## [0.11.1] - 2026-11-08

### Changed
- **Domain-agnostic structural-depth file discovery, universal multi-language/schema/data concept extraction, compound-identifier Vault recall filtering, and complete 7-experiment Two-Tier `.qwcache` Whitepaper verification (`QWMemoryPack`, `QWMemoryVault`, `qwg.swift`, `docs/lsa-codebase-memory-whitepaper.md`).**
  - Replaced hardcoded folder-specific tiering (`specs/`, `docs/`, `Sources/`, `Tests/`, `Scripts/`) in `QWMemoryPack.discoverFiles` with **universal directory nesting depth ($\text{depth}(f) \ge 0$) and temporal stability ordering**, and expanded `QWMemoryVault.extractSymbols` to extract structural anchors across arbitrary programming languages (C/C++, Rust, Go, Python, JS/TS, Swift, Java/Kotlin, SQL, Protobuf/GraphQL/HCL), structured data formats (JSON, YAML, TOML, INI, CSV/TSV column headers), and documents (Markdown, AsciiDoc, LaTeX `\section`/`\subsection`).
  - Refined `QWMemoryVault.recallVaultContext` so single-symbol triggers require compound identifiers (`camelCase` / `PascalCase` / `snake_case`, `count >= 5`), eliminating false-positive Vault attachments on generic words (`build`, `test`, `lint`, `init`) in Cortex queries while maintaining `100%` (`4/4`) recall across file paths, stems (`QWDashboardBuilder`, `QWSlotPinnedLRUStore`, `QwgIndex`), and compound symbols (`DashboardCardSpec`, `isSharedBaseKey`).
  - Wired `QWMemoryVault.recallVaultContext` and `QWMemoryVault.applyElasticBudgetGuard` into `qwg run --repo <path>`, expanded `memoryReserve` to include `(precomputedDelta?.tokens.count ?? 0) + 8192` so live `<|memory_delta|>` overlays are never clipped by the one-shot arena cap, and honored `--no-prefix-cache` with `--repo <path>` for exact paired interleaved A/B cold-prefill benchmarking.
  - Re-ran all 7 whitepaper experiments (`§4.1`–`§4.7`) on the Two-Tier Cortex (`32,742` tokens, `264.61 MiB` `35B-A3B` / `464.54 MiB` `Flash-Next`) + Off-Context Vault (`287` files) implementation and updated all SVG infographics (`assets/qwegn_memory_pack_architecture.svg`, `assets/qwegn_memory_pack_experiments.svg`, `assets/qwegn_engine_bento.svg`) and `docs/lsa-codebase-memory-whitepaper.md`: **`54.4×` faster TTFT (`6.53 s → 0.12 s`, `290,645.4 tok/s` prefill) on `Qwen3.6-35B-A3B`**, **`74.3×` faster TTFT (`17.84 s → 0.24 s`, `138,390.6 tok/s` prefill) on `Qwen3.8-Flash-Next`**, **`0.40 s` live `<|memory_delta|>` TTFT**, and **`150.5–163.9 tok/s` multi-turn decode**.

## [0.11.0] - 2026-11-08

### Added
- **Two-Tier Hierarchical LSA Cortex (`32,768` tokens) + Sub-Millisecond Off-Context Vault Recall (`QWMemoryVault`) + Elastic Context Budget Guard (`QWMemoryPack`, `QwgIndex`, `QwgReplAndServe`, `QWChatTurnPreparer`, `QWEngineLifecycleActor`).**
  - Replaced monolithic $1\text{M}$-token linear repository prefix packing with a Two-Tier architecture (`qwg index` defaults to `--max-tokens 32768`; `--full-prefix` retains monolithic mode):
    1. **Tier 1 — Resident Cortex `.qwcache` (`32,745` tokens, `264.61 MiB` on `Qwen3.6-35B-A3B` `q4` / `464.57 MiB` on `Qwen3.8-Flash-Next` `q4` — `23.6×–23.7×` smaller than a `1.01M`-token monolithic pack):** Embeds `[QWG REPOSITORY CORTEX MAP]` indexing all `293` repository files and their extracted top-level AST declarations alongside verbatim Tier-0 architecture/build files (`96.9%` / `1,015,831` tokens of the 1M context window left free for coding agents).
    2. **Tier 2 — Off-Context LSA Vault (`QWMemoryVault.recallVaultContext`, `< 0.5 ms`):** Indexes all remaining `284` repository files in `.qwg/manifest.json` (`inCortex: false`, `symbols: [String]`) and pages matching files on-demand into `<|memory_vault_recall|>` appended deterministically to Turn $k$'s user message (`user(q_k + vault(q_k))`), preserving **100% `[warm_ram]` prefix cache identity** across all prior conversation turns ($1 \dots k-1$).
    3. **Elastic Context Budget Guard (`QWMemoryVault.applyElasticBudgetGuard`):** Reserves `agentPromptTokens + maxNewTokens + safetyMargin` first before attaching a mounted `.qwcache`, automatically bypassing oversized legacy `.qwcache` artifacts so coding agents never encounter `contextDoesNotFit`.
  - Fixed `qwg serve --repo <path>` startup ordering to mount and enforce `kvMode` before opening the HTTP listener, eliminating the redundant `f16` $\to$ unload $\to$ `q4` double-load (`2.81 s` total boot + mount vs `73.0 s` on monolithic 1M packs).
  - Verified live 4-turn multi-turn sessions over `qwegine` in both `qwg repl` (`0.10 s` / `0.15 s` Cortex TTFT, `1.30 s` / `0.64 s` Vault recall TTFT, `147.5–156.0 tok/s` decode — **`+53%` faster decode** than 1M monolithic prefix) and `qwg serve` (`102.7 ms` / `143.4 ms` Cortex TTFT, `1,321.2 ms` / `643.8 ms` Vault recall TTFT, `116.2–163.7 tok/s` decode, `390.2–532.7 MB` warm RAM, `0.0 MB` SSD spill).

## [0.10.4] - 2026-11-08

### Changed
- **8-row speculative verification ring & dynamic wide Prompt-Lookup Decoding (`QWFourSlotRollbackRing`, `QWContextCopyDrafter`, `QWGDNF32`, `QWGemmF32`, `FNGPUEngine`, `FNGPUMTP`, `FNGPUSession`).**
  - Widened the speculative rollback ring in `QWFourSlotRollbackRing`, `FNGPUWeights` (`convRing`, `stateRing`, `histRing`), and `FNPrefixCache` from `4` slots (`maxVerifyRows = 4`, `ringMask = 3`) to `8` slots (`maxVerifyRows = 8`, `ringMask = 7`), and added packed 8-slot ring indexing (`0x8000 | ((ringParity & 7) << 8) | rows`) to `qwg_gdn_conv_prep_f32` and `qwg_gdn_recurrent_rows_f32` (`QWGDNF32.encodeConvPrepF32` / `encodeRecurrentRowsF32`).
  - Enabled dynamic context-copy scaling in `QWContextCopyDrafter.proposeDrafts` (`draftCapacity = 7`, `defaultMTPDrafts = 3`), expanding suffix and periodic matches up to `7` speculative drafts (`8.0 tok/step` max) while keeping neural MTP drafts at `1..3` (`2..4` verify rows) on novel text.
  - Chunked `QWGemm.encodeHQ4Gemv` and `encodeHQ4F32V4Argmax` for `rows = 5..8` into two 4-row (`M <= 4`) fused dispatches (`qwg_gemv_*_f32_r4`), and skipped redundant chained single-row MTP passes (`encodeChain`) when context-copy or periodic drafting is active (`!session.mtpActive || session.periodicDrafts`), lifting `Qwen3.8-Flash-Next` Predictable (Copy) throughput from `129.3 → 156.3 tok/s` (`3.37 → 5.49 tok/step`, `1.87×` speedup) on the native `hq8g32-hq4g64` pack and `126.1 → 158.8 tok/s` (`3.31 → 5.49 tok/step`, `1.74×` speedup) on the MLX `mixed-4-8bit` pack with `100%` Engine Fidelity (`36/36` correctness, `3/3` determinism, `36/36` logprob consistency).

## [0.10.3] - 2026-11-08

### Changed
- **Unified engine harness generality, module deepening, and dead/duplicate code elimination (`QWEngineBootstrap`, `QWSafeTensorsContainer`, `QWLSASidecar`, `QWKVCompressedLayout`, `QWSlotPinnedLRUStore`, `QWSpeculationBackoff`, `QWEngineLifecycleActor`, `DenseModule`, `StandardModule`).**
  - Extracted `QWEngineBootstrap.makePlan` and `QWEngineBootstrap.admit` in `QWCore`, eliminating duplicated plan construction, capacity log formatting, single-instance lock acquisition, and host memory validation across `QWEngineDriver` and `FNGPUEngine`.
  - Consolidated `.bf16`/`.f16`/`.f32` and parallel MLX affine `u32` (`.scales`/`.biases`) float reading into `QWSafeTensorsContainer.readFloats` and `isMLXCheckpoint`, removing duplicated reader implementations from `FNPackLoader` and `DriverWeightLoading`.
  - Generalized `QWLSASidecar` Weight-SVD synthesis (`synthesizeFromSafeTensors`, `synthesizeFromModel`) across `.qwp` and `.safetensors` checkpoints and parallelized per-layer SVD computation via `DispatchQueue.concurrentPerform`.
  - Generalized `QWKVCompressedLayout` (`quantizeQ8`/`dequantizeQ8`/`quantizeQ4`/`dequantizeQ4`) to arbitrary `(kvHeads, headDim)` geometries and added $O(1)$ root-hash pre-filtering (`StoredEntry.rootHash`) to `QWEnginePrefixStore.lookup` and `FNPrefixWarmStore.longestMatch`.
  - Added deterministic worker thread shutdown (`shutdown()` / `deinit`) to `ColdEventPipeline`, event-driven `CheckedContinuation` in-flight drain (`waitForInFlightRequests`) to `QWEngineLifecycleActor`, `Config.updating`/`makeEngineBuildOptions` helpers, and `DenseModule: QWInferenceEngine` conformance while removing 5 copy-pasted `make(plan:engine:)` overrides in `QWStandardModule` and `.qwen38_27b` fallback defaults.
  - Hardened `QWEngineDriver.residencyBudgetBytes` and `QWCapacityBudget.validateMemoryForEngineLoad` so `.maxResident` and `.safe` enforce their `16 GiB` / `24 GiB` default OS reserves, cap cold-streaming MoE arenas against live reclaimable RAM minus safety headroom (`4 GiB` / `6 GiB`), and clamp default KV auto-sizing on cold-streaming MoEs (`GLM-5.3-Flash`) to `32,768` tokens so wired Metal arenas cannot starve macOS.

## [0.10.2] - 2026-11-08

### Fixed
- **Multi-turn chat retention, pinned `.qwcache` protection, and delta LSA snapshots across `qwg repl` and `qwg serve` (`QWSlotPinnedLRUStore`, `PrefixCache`, `QWLSAOperator`, `FNPrefixCache`, `QwgReplAndServe`).**
  - Prevented `QWSlotPinnedLRUStore.makeRoom` and `evictOldest` from ever evicting pinned `isSharedBase` `.qwcache` memory packs under slot pressure.
  - Charged only incremental COW KV pages (`actualPayloadCharge = max(0, value.memoryBytes - sharedKVBytes)`) against `store.byteLimit` when a turn extends a mounted `.qwcache` base, and suppressed opportunistic multi-GB cold-SSD write-throughs (`> 256 MiB` or when `hasSharedBase` is active).
  - Updated `QWSlotPinnedLRUStore.resolveSlot` and `prepareSlotForStore` plus `PrefixCache` `autoRetain` (`firstVis - 4`) so multi-turn conversations advance `slotBaseKey` past `<think>` stripping on every turn (`O(1)` incremental prefill per turn: `0.17 s` / `0.27 s` / `0.29 s` TTFT across Turns 1–3 on a `1,003,803`-token mounted `.qwcache`).
  - Added `QLSADLT1` pinned-base delta snapshots and GPU residency watermark tracking (`residentPos512`) to `QWLSAOperator`, reducing per-turn LSA snapshot capture/restore above a 1M-token `.qwcache` from `467.4 MiB` to `~6.4–111 KB`.
  - Updated `qwg repl` header banner to display `qwg repl (<model-pack-name>)` instead of the internal engine/descriptor family name.
- **Deterministic live-decode-rate telemetry test (`QWEngineTelemetryRegistry.recordDecodeToken(at:)`).**
  The speculative-burst gate drove its synthetic timeline with `Thread.sleep`, which macOS inflates 3-5× under scheduler load, making `make test`/`make ci` fail reproducibly on an otherwise quiet machine; the registry now accepts injected timestamps (test seam — production call sites unchanged) and the case verifies the FIFO coalescing/eviction arithmetic in ~20 ms with no wall-clock dependency.

## [0.10.1] - 2026-11-08

### Fixed
- **Ignore-rule filtering (`.gitignore`, `.git/info/exclude`, `.qwgignore`) and envelope slicing in `qwg index` (`QWMemoryIgnoreMatcher`, `QWMemoryPack`).**
  Added full gitignore-pattern matching across repository root and nested `.gitignore`, `.git/info/exclude`, and `.qwgignore` files during `QWMemoryPack` discovery, and fixed token-envelope slicing when truncating indexed blocks to `--max-tokens`.

### Removed
- **Pruned 125 dead and superseded Metal kernels (`846` $\to$ `721` in `kernel.metallib` / `kernel_std.metallib`) and hardened `Scripts/lint_kernels.py`.**
  Removed 27 unreferenced Metal kernels (`qwg_gdn_norm`, `qwg_xgroupsum_f16`, `qwg_gemv_q4g_dx_h16`,
  `qwg_gemv_q4g_dx_h16_gdn_inproj`, `qwg_gemm_q4g_nax_f32_bm128`, 20 `qwg_gemm_aq*_nax_f32_bm128`
  variants, `qwg_bf16_gemv`, and `qwg_bf16_gemv_bm`), 98 superseded kernels and their dead Swift
  operator wrappers (`qwg_gemv_q4g_f32_r1`, 20 `qwg_gemv_aq*_f32_r1`, `qwg_moe_gateup_q4g_f32`,
  20 `qwg_moe_gateup_aq*_f32`, 20 `qwg_moe_gateup_group_aq*_f32`, `qwg_moe_down_group_f32`,
  20 `qwg_moe_down_group_aq*_f32`, `qwg_moe_gather_f16`, `qwg_moe_unsort_f32`, `qwg_moe_nax_tiles`,
  `qwg_moe_gather_pair_split`, `qwg_moe_unsort_rows_f32`, `qwg_gdn_prep`, `qwg_gdn_prep_f32`,
  `qwg_gdn_conv_f32`, `qwg_gemm_q4g_h16_dual`, `qwg_gemm_q4g_h16_bn128`, `qwg_gemm_q4g_h16_bk128`,
  `qwg_tf_row_matmul`, `qwg_gemv_q4g_dx_h16_dual`, `qwg_gemv_q4g_v4_pscale`, and
  `qwg_residual_add_step`), plus unused `psoGroupSum`/`groupSums` properties on `QWEngineDriver` and
  `QWMTPBatchedMLP`. Updated `Scripts/lint_kernels.py` to strip comments, exclude
  `QWKernelInventory.swift` from `swift_refs()`, and flag `STALE INVENTORY DOOR` entries so the
  prewarm manifest or prose comments can never mask dead kernels again.

## [0.10.0] - 2026-11-08

### Added
- **Unified Codebase & Knowledge Memory Packs (`QWMemoryPack`, `qwg index`, `<repo>/.qwg/*.qwcache`).**
  Added `QWLSAOperator` indexer state capture/restore (`QLSA0001` snapshot serialization into
  `QWCacheDiskManager` `.qwcache` files), standalone `.qwcache` build/mount APIs on `QWEngine` and
  `FNGPUEngine` (`isSharedBase: true`), `QWMemoryPack` 3-tier stability discovery with 512-token
  LSA Ward superblock alignment and live working-tree `<|memory_delta|>` overlays, `qwg index` CLI
  subcommand (with `--max-tokens` 3-tier budget cap and automatic per-model profile `kvMode` selection),
  interactive `qwg repl` (`/memory`, `/memory mount`, `/memory unmount`) and `qwg run`
  (`--repo`, `--no-memory-pack`) mounting, automatic repository discovery from coding-agent system
  prompts in `QWServer` (`/v1/memory/mount`, `/v1/memory/unmount`), automatic `kvMode`
  enforcement with a Settings UI lock in `qwg-menu`, and whitepaper/infographics
  (`docs/lsa-codebase-memory-whitepaper.md`, `assets/qwegn_memory_pack_architecture.svg`,
  `assets/qwegn_memory_pack_experiments.svg`, `55.2×–66.0×` faster 27.6K repository TTFT).

## [0.9.7] - 2026-11-08

### Added
- **Menu Bar Model-tab vision tower switcher (`loadVision`, live in-place residency toggle without KV cache reset).**
  Added **Load Vision Tower (Multimodal Image Input)** (`loadVision: Bool`, default `false`) to the
  Menu Bar Settings **Model** tab (`SettingsView`, `SettingsView+Actions`), `QWKnobRegistry`,
  `QWServerSettings`, `QWServerConfigurationResolver`, and `QWSettingsCoherence.liveKeys`. Toggling
  `loadVision` and clicking **Save & Apply** dynamically loads or unloads the vision tower in place on
  the running server (`POST /v1/settings` $\to$ `QWEngineLifecycleActor.setVisionLoaded(_:)`) without
  restarting the engine or resetting the GPU KV cache, and preserves KV prefix cache lookup for the
  pure-text prefix (`0..<firstVisualOffset`) before image spans across both `FNGPUEngine` and
  `QWEngineDriver`.
- **Vision head support (`qwen4_exp` / `Qwen3.8-Flash-Next`) with prompt-scoped on-demand residency (`QWVisionOnDemandBackend`).**
  Enabled `.vision` capability on `FlashNextModule`, threaded `QWVisualSpan` embeddings through
  `FNGPUEngine` (`FNGPUPrefillStack` and `FNGPUDecodeStack`) with `hc = 4` hyper-connection stream
  expansion, added smart-resize image resampling in `QWImageDecoder`, and implemented
  `QWVisionOnDemandBackend` across both `.qwp` and `.safetensors` checkpoints so the ~1.98 GB vision
  tower starts unloaded (`0 B` resident RAM), loads on demand when a prompt contains an image, and
  unloads automatically when the next prompt has no image.

### Fixed
- **Accelerated vision encoding 31× (`79.9s` $\to$ `2.57s`), added encoded image LRU cache (`0 ms` repeat turns), full visual-span KV prefix caching, compact in-place-KV long-context retention, and sibling-quantization hot-swap protection (`QWVisionRef`, `QWImageDecoder`, `QWVisualSpan`, `FNGPUEngine`, `FNPrefixCache`, `QWEngineDriver`, `QWEngineLifecycleActor`).**
  Replaced scalar `Double` GEMM, attention, LayerNorm, `foldPosEmbed`, and `qwGeluTanh` in `QWVisionRef` with Apple `Accelerate` hardware AMX `cblas_sgemm$NEWLAPACK` and `vDSP`/`vForce` vector kernels over pre-allocated workspace buffers (`79,883.9 ms` $\to$ `2,574.7 ms` on `44×80` patches / `3,520` patches across 27 ViT blocks); added an in-memory `EncodedImageLRUCache` in `QWImageDecoder` so repeat turns with the same image skip re-decoding and ViT re-encoding (`0.0 ms`); salted KV prefix cache token histories across `QWVisualSpan`s (`QWVisualSpan.saltedCacheHistory` / `saltedCacheTokens`) so `FNGPUEngine` and `QWEngineDriver` retain and hit full KV prefixes past image turns; added compact in-place-KV warm prefix retention in `FNPrefixCache` (`hdr.hasMTP |= 2`) when full 390k+ token KV snapshots exceed `warm.byteLimit`; and guarded `handleChatCompletion` against unloading an active model when an incoming tool request names a sibling quantization of the same base model family (`canonicalModelFamilyStem`).
- **Fixed unintended model hot-swap, `79.90 GiB` OOM admission failure, vision attention latency, and `QWEngineDriver` (`Qwen3.6-35B-A3B` / `Qwen3.8-27B`) multimodal prefill (`QWModelRoot`, `QWServerSettings`, `QWEngineLifecycleActor`, `QWCapacityPlan`, `QWVisionRef`, `QWImageDecoder`, `QWEngineDriver`).**
  Preserved version dots in `QWModelRoot.canonicalModelName(for:)` so `Qwen3.8-...` and `Qwen3.6-...` model directories are not truncated at `.8` / `.6` into `"Qwen3"` (which missed the saved `q8` profile and inflated 1M-context KV admission to `79.90 GiB` `f16`), made `selectModelIfNeeded` and `resolveModelReference` match the active model/alias case-insensitively, serialized model unloads behind in-flight requests with a 3s Metal/IOKit memory reclamation settle window, parallelized the 16 attention heads in `QWVisionRef.block` with row-local softmax buffers, capped default live vision smart-resize to `1,048,576` pixels (`1024×1024`, overridable via `QW_VISION_MAX_PIXELS`), and wired `request.visualSpans` into `QWEngineDriver` (`DriverPrefill`, `DriverDecode`, `EncodeKernels`) so all vision-capable models across both `FlashNextModule` (`Qwen3.8-Flash-Next`) and `DenseModule` (`Qwen3.6-35B-A3B`, `Qwen3.8-27B`) interleave visual embeddings during prefill.

### Changed
- **Unified `QWEngine` runtime architecture across all model families.** Consolidated the leaf
  operators (`QWMoEOperator`, `QWContextCopyDrafter`), three-tier prefix cache (`QWSlotPinnedLRUStore`,
  `QWResidentKVWatermarkTracker`, `QWPrefixColdStore`, `QWPrefixCacheCoordinator`), 4-seam compiled
  layer stack (`QWCompiledLayerStack`, `QWFourSlotRollbackRing`, `QWLayerStackSchedule`, `QWPLEOperator`,
  `QWHCOperator`, `QWQSAOperator`), continuous batch scheduler (`QWBatchScheduler`), and top-level
  `QWEngine` handle across `Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `Qwen3.5-MoE`, `Qwen3`, `Qwen2`,
  `Llama`, and `Mistral`.
- **Deepened weight-loading, CLI configuration, wire-protocol, and chat-turn preparation modules.**
  Extracted `QWCheckpointTensorLoader` and `QWDeviceEmbedTable` into `QWLoad`, unified CLI engine
  option resolution in `QWResolvedCliEngineConfig`, consolidated HTTP wire model-alias and SSE/NDJSON
  stream framing in `QWHttpServer+Streams`, and extracted `QWChatTurnPreparer` for pure, testable
  prompt/schema/constraint preparation.

## [0.9.6] - 2026-11-07

### Changed
- **Menu Bar & Web UI model card redesign and dropdown model selector.** Un-congested the
  `loadedWeightsCard` in `qwg-menu` (`TelemetryCards`, `TelemetryObservable`) so the full model
  name and complete quantization + Metal family details (`HQ8_G32 (native, GPU) • Metal 4`) are
  displayed on dedicated rows without truncation, added an interactive dropdown model selector
  menu to switch models directly from the Menu Bar popover and Web UI (`controlbar.html`), and
  surfaced `KV` and `MTP` pills alongside `Cores`, `GDN`, and `Context`.

## [0.9.5] - 2026-11-07

### Added
- **Configurable concurrent sessions (`maxConcurrentSessions`, `1...16`, default `8`).** Added
  `maxConcurrentSessions` across `QWKnobRegistry`, `qwg serve --max-concurrent-sessions`
  (`QWG_MAX_CONCURRENT_SESSIONS`), `QWCacheConfig`, `QWEngineBuildOptions`, `QWServerSettings`,
  `QWServerConfigurationResolver`, and the macOS Menu Bar **Performance** and **Cache** tabs
  (`SettingsView`, `SettingsView+Performance`).
- **Shared `system + tools` base prefix checkpointing (`retainPrefixLength`, `isSharedBase`).**
  `QWEngineLifecycleActor` now maintains a multi-session LRU token prefix cache
  (`cachedTokenPrefixes`) and detects the leading `<|im_start|>system` / `developer` block boundary,
  allowing `FNGPUEngine` and `FNPrefixCache` to checkpoint and pin the shared `system + tools`
  prefix in `FNPrefixWarmStore` so branching subagents reuse both CPU tokenization and GPU KV
  state on Turn 1.

### Changed
- **Continuation-first multi-session prefix cache slot resolution.** `FNPrefixWarmStore.resolveSlot`
  and `QWEnginePrefixStore.resolveSlot` now prioritize direct continuation of an active slot's
  `slotLastEntry`, single-slot rollback, and free slot allocation (`0..<maxSlots`) before falling
  back to `rootHash` (first 128 tokens) matching, preventing concurrent subagents that share a
  system prompt from colliding on Slot 0.
- **Concurrent KV-cached batch execution in `FNGPUBatchScheduler`.** Enabled 2..4 concurrent
  requests that fit in a 4-way KV partition to execute in lockstep batch mode with per-slot MTP
  `attnState1` partitioning, per-slot Tier-0 residency tracking (`residentKVHistoryBySlot`), and
  full `FNPrefixCache` lookup, restore, and post-prefill persistence.

## [0.9.4] - 2026-11-07

### Changed
- **Unified Qwegn processor emblem across Menu Bar and Web UI.** The macOS menu bar status item
  (`QWMenuBarApp`), menu bar control panel (`QGProcessorLogoView`), Web UI header and active-model
  card (`controlbar.html`), and `logo.svg` now render the same canonical `QG` processor microchip
  vector mark.

### Fixed
- **Web UI Model Catalogue brain icon.** Replaced the broken `brain` Material Symbols ligature
  (which rendered as literal `BRAIN` text on model cards) with an inline human-brain vector glyph
  matching the macOS Settings Model tab (`controlbar.html`).

## [0.9.3] - 2026-11-07

### Changed
- **Speculative decoding decoupled from sampling.** The verify path no longer inherits sampler
  side effects, so draft acceptance and token drawing evolve independently
  (`FNGPUEngine`, `FNGPUSession`, `QWEngineDriverConformance`).
- **Per-slot warm cache protection.** Warm-cache entries are pinned per scheduler slot, so a
  concurrent stream cannot evict another slot's prefix (`FNPrefixWarmStore`, `FNPrefixCache`).
- **Telemetry metrics fixed.** Server metrics, logger attribution, and menu-bar telemetry now
  report measured values only (`QWServerMetrics`, `QWServerLogger`, `TelemetryObservable`).

## [0.9.1] - 2026-11-07

### Added

- **Token-level constrained decoding (`QWTokenConstraint`, `QWStructuredOutputConstraint`, `QWToolCallConstraint`).**
  Byte-trie JSON Schema (`json_schema` / `json_object`) enforcement and multi-format tool-call
  schema constraint (Qwen XML `<function=NAME>` / `<parameter=NAME>`, GLM XML
  `NAME<arg_key>NAME</arg_key>`, and JSON `{"name": "NAME"}`), integrated with prefix caching and
  exact target-model speculative decoding across both `DenseModule` (`QWEngineDriver`) and
  `FlashNextModule` (`FNGPUEngine`).
- **Mixed `HQ8_G32 + HQ4_G64` `Qwen3.8-Flash-Next` pack (`Qwen3.8-Flash-Next-hq8g32-hq4g64`).**
  Upgrades the 3.4% shared trunk to `HQ8_G32` and packs the 512 routed MoE experts in `HQ4_G64`,
  reducing logit KL divergence vs BF16 by `2.62×` (`3.56×` on nested JSON) while shrinking
  `model.qwp` by `5.16 GiB` (`71.30 GiB`, `4.76 bits/param`) and reaching `100%` conformance,
  `100%` capability (`6/6` nested JSON), and `100%` logit fidelity (`meanTopProb = 0.9658`).
- Controlled loopback JSON-quality replays with persisted exact verdicts and output
  hashes, native HQ header-based byte forecasting with reference-compatible measured
  Pareto comparisons, and full-vocabulary aligned teacher-logit KL scoring. Size-only
  results explicitly carry no inferred quality ranking.

## [0.9.0] - 2026-11-06

### Added

- **GLM-5.3-Flash (`glm5_next`) end-to-end support.** Runs the 321.3B-parameter (18B active)
  hybrid MoE model on a single 128 GB Mac from a 134.37 GiB mixed `HQ4_G32_M5 + HQ3_G64`
  pack (3.59 bits/param): 45 layers (3 dense + 42 MoE with 288 experts, 8 routed + 1 shared),
  34 KDA vector-decay linear-attention layers, 11 NoPE MLA layers (`kv_lora_rank = 512`,
  `q_lora_rank = 1536`, 64:1 MQA matrix absorption) with Key-Pool(4) 32-head sparse indexer
  (`topK = 2048`), 4-stream Sinkhorn-20 mHC, NextN Layer 45 MTP speculative head, and
  ephemeral 24-block vision tower.
- **Generalized Hadamard-rotated quantization (`HQ{2,3,4,6,8}_G{32,64,128,256}`).** 20
  rotated `(bits, group)` combinations with MSE-optimal `clip_shrink` search, `F8_E4M3`
  block-scale dequantization, and independent `expertDownBits` / `expertDownGroupSize`
  support in QWP packs and Metal GEMV/GEMM/MoE kernels.
- **Dynamic expert LRU residency & event-pipelined NVMe/GPU cold-expert streaming.**
  - Static router-signal hot/cold expert permutation (`e_score_correction_bias` + router L2 norm).
  - On-GPU dynamic per-layer expert LRU cache (`QW_EXPERT_CACHE=1`), promoting cold-miss
    experts staged from SSD into the resident bank (`~60 → ~15.5` cold loads/token at `K=216`).
  - Event-pipelined 1–4 row decode (`ColdEventPipeline` with `MTLSharedEvent`, self-tuned
    dependent-FMA delay command buffer hiding the `~115 µs` event wake latency, and
    `qwg_moe_resident_ids` overlapping GPU resident-expert compute with NVMe SSD staging).
  - GPU prefill span map (`qwg_moe_span_ids`, enabled by default), overlapping resident MoE
    execution with cold-expert staging (`0.40 s → 0.10 s` wall-GPU gap at 16K).
- **Selectable engine tuning & memory policies (`QWEngineTuning`).** `maxResident` (default,
  `858–968 tok/s` prefill and `31.3–31.6 tok/s` serial decode on GLM-5.3-Flash at 16K) and
  `safe` memory policies, plus per-model `osReserveGiB`, `residentExpertsPerLayer`,
  `expertCache`, `coldEvents`, `coldEventOverlap`, and `prefillSpanMap` controls in
  `~/.qwg/profiles/`, server settings, environment variables (`QW_*`), and the new
  `qwg-menu` **Performance** tab.
- **Batched MTP verification across all 20 HQ/AQ formats & MoE-Spec.** Multi-row (`M = 2..4`)
  `qwg_aq_slice_dot2` and `qwg_gemv_aq_rows4_m_f16` kernels, routed MoE expert union
  deduplication across verify tokens, expert-byte optimal draft depth (`arXiv:2610.34440`),
  and `MoE-Spec` expert budgeting (`arXiv:2602.16052`).
- **Stage profilers.** `QW_PREFILL_PROFILE=1` and `QW_DECODE_PROFILE=1` report per-chunk and
  per-step GPU sub-stage timings (mHC, KDA/GDN, MLA, MoE resident/cold staging, and head).
- **GLM-5.3-Flash infographics & model card.** Added `assets/qwegn_model_glm53_flash.svg`,
  `assets/qwegn_lsa_scaling_glm53_flash.svg`, and the Hugging Face pack card.

### Changed

- **NoPE MLA & KDA Metal 4 kernels.** Added Apple10 NAX tensor-MMA sparse latent attention
  (`qwg_mla_sparse_attn_nax`, `4–5×` faster prefill), 16-way split-K NAX decode attention
  (`qwg_mla_sparse_attn_nax_split`, `2.23 → 0.44 ms/tok`), `simdgroup_matrix` MMA absorbed Q
  projection (`qwg_mla_absorb_q_mma`, `340 → 50 ms` per 4K chunk) and UV projection
  (`qwg_mla_proj_uv_mma`, `130 → 58 ms`), pool-parallel MMA indexer scoring with split top-K
  select (`8.4 → 0.92 ms/tok`), dedicated `qwg_mla_gate_weights_proj` (`2.17 → 0.26 ms/tok`),
  MMA KDA low-rank gate expansion (`qwg_kda_gates_mma`, `218 → 60 ms`), and pre-exponentiated
  `exp(g)` KDA decay storage.
- **4-stream mHC kernels.** Split decode/verify router-norm across threadgroups with stream
  Gram-matrix (`pre^T G pre`) reductions and 4 row groups (`2.94 → 1.38 ms/tok`), and added
  4-row prefill router-norm (`qwg_mhc_router_norm_rows_f16`, `-35%` router time).
- **Flash-Next MoE verify.** Uses independently dispatched rows at verification sizes to avoid
  cross-token union overhead when all experts are resident.

### Fixed

- Deterministic MLA indexer top-K candidate collection via per-thread contiguous chunks and
  SIMD prefix sums.
- MLA Key-Pool(4) indexer state is captured in warm RAM and cold SSD prefix snapshots and
  stashed/restored when switching between concurrent sessions.
- Streamed MoE paths preserve router-order reduction before adding the shared expert.
- GPU prefill span map promotes cold misses in route first-appearance order, matching the
  host-mapped residency set bit-for-bit.
- Shared-memory barrier before `sgScratch` reuse in MLA sparse attention and latent prep.

## [0.8.0] - 2026-11-04

### Added

- **Latent Sparse Attention (LSA).** A model-agnostic sparse path for full-attention
  layers (`QWLSAOperator`, `--lsa-mode auto|off|qsa-cluster`) that scales to 1M-token
  contexts without retraining. On Qwen3.6-35B-A3B at 524K it gives 13.1× prefill and
  4.7× decode over dense attention, with 3/3 needle recall from 64K to 1M.
- Parallel LSA block selector that returns the same selection as the serial one.
- Long-context MTP acceptance backoff: from 128K, decode drops to serial steps for a
  cooldown window when speculative acceptance falls.
- Variational MTP speculative decoding, with general MoE MTP support.
- Looped mHC model support (SenseNova-Loop-9B) and per-model profiles in `~/.qwg/profiles/`.
- HQ4/HQ6/HQ8 packs at G32/G64/G128/G256, and mixed dense/expert QWP packs.
- Recorded cool-off periods for paired benchmark measurements.
- Per-model LSA scaling slides and Qwegn-branded infographics in `assets/`.

### Changed

- Flash-Next keeps its trained native QSA selector under `--lsa-mode auto`; the LSA
  cluster selector replaces it only with an explicit `--lsa-mode qsa-cluster`.
- Row-paired and 4-row FWHT GEMVs for HQ4/AQ4 verify, lm_head, MLP and the DFlash
  drafter; row-parallel residual RMSNorm for decode and MTP verify.
- MoE: one expert per simdgroup router for decode, NAX prefill router GEMM with fused
  shared-expert gate, grouped expert GEMMs sliced into ~2560-threadgroup dispatches.
- Flash-Next: balanced 8K prefill chunking, faster 32K–256K QSA selection, parallel
  prefix restore and snapshots, PLE staging off the GPU critical path.
- Dense NAX prefill uses BM64 tiles and SLC-sized row chunks.

### Fixed

- Flash-Next periodic drafts are verified by exact argmax under Variational MTP.
- Multimodal prompts are sized by the loaded model's hidden width.
- Vision merger projection dimension for Qwen3.6-35B-A3B and other hidden sizes.
- Flash-Next PLE race.

## [0.7.0] - 2026-11-01

Baseline release before this changelog was started.

[Unreleased]: https://github.com/qwegnai/qwegine/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/qwegnai/qwegine/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/qwegnai/qwegine/compare/v0.9.7...v0.10.0
[0.9.7]: https://github.com/qwegnai/qwegine/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/qwegnai/qwegine/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/qwegnai/qwegine/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/qwegnai/qwegine/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/qwegnai/qwegine/compare/v0.9.1...v0.9.3
[0.9.1]: https://github.com/qwegnai/qwegine/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/qwegnai/qwegine/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/qwegnai/qwegine/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/qwegnai/qwegine/releases/tag/v0.7.0
