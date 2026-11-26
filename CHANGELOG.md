# Changelog

All notable changes to Qwegn (`qwg`, `qwg-menu`) are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/). Until 1.0, a minor bump
may change CLI flags, pack formats or profile files.

The single source of the version is `QWVersion.current` in
`Sources/Foundation/QWCore/QWVersion.swift`; release tags are `vMAJOR.MINOR.PATCH`.
Merges to `master` record changes under `## [Unreleased]` without touching
`QWVersion.current` or any release artifact; the version is bumped only in the
release-preparation commit, which renames `[Unreleased]` to the new
`[X.Y.Z] - YYYY-MM-DD` heading.

## [0.43.1] - 2026-11-26

Adds native `/v1/embeddings` and `/v1/rerank`, pure-Swift `QWCognitiveRails` with 9 Cognitive/Vault MCP tools, COW `QWSessionTree` (`/v1/sessions/*`), unified `QWResidencyPool` UMA admission, client request traceability, and modernized offline `/ui`, alongside Inkling-2 prefix-cache / thinking-off / non-null fixes and community PR [#1](https://github.com/qwegnai/qwegn/pull/1).

### Performance Improvements

| Workload (Apple M5 Max, 128 GB UMA, `q4` KV) | Before | v0.43.1 | Impact |
| :--- | :--- | :--- | :--- |
| `Inkling-2-Mini` exact-repeat prefix cache (`1,539–1,540` tok) | `622 ms` (`0.9x`, `0/1540` cached) | **`49 ms`** (`10x` vs `490 ms` cold, `1539/1539` cached) | **`12.7x` faster warm TTFT** (`finalNormRow` bypass) |
| `Inkling-2-Mini` `llmprobe-code100` (`--reasoning off`) | `14m 27s` (`55/100`, `40` capped, `72,961` tok) | **`2m 3s`** (`75/100`, `1` capped, `11,729` tok) | **`7.0x` faster, `+20` pts** (thinking-off prefill + `dspark`) |
| `Inkling-2-Mini` conformance & EvalPlus (`HE` / `MBPP`) | `288/291` (`99%`), EvalPlus `ERR(1)` | **`292/292` (`100%`, `0` warn)** · **`76.8%` / `84.4%`** | **Full parity** (`73.8%+` HE+ · `71.4%+` MBPP+) |
| `Qwen3.8-Flash-Next` full eval (`--stream-ngrams`, `40K`) | Unbenchmarked on `bench` suite | **`291/291` (`100%`)** · **`65x`** cache (`845 -> 13 ms`) | **`93.9%` HE · `91.5%` MBPP · `86.4%` agentic · `93.3%` GSM8K** |

### Added
- **Dense Embeddings & Rerank (`/v1/embeddings`, `/v1/rerank`)**: Generalized `QWDecisionPackEngine` across all 4 `HeadKind` modes, `LayerNorm`, `GELU`/`GELU-tanh` (gated to `.decisionEncoder`), bidirectional encoder masks, and L2-normalized Matryoshka embeddings.
- **Cognitive Rails, Episodic Memory & MCP Suite**: Added `QWCognitiveRails`, `reasoning.effort = "auto"`, `0o600` BM25 `QWEpisodicTranscriptStore`, `/v1/cognitive/*`, and 9 Cognitive + Vault tools across `POST /mcp` and `qwg mcp`.
- **Session Trees (`/v1/sessions/*`)**: Added `QWSessionTree` with O(1) COW `fork`, Radix pin refcounts, non-leaf `prune` guard + `cascade`, `0o600` JSONL replay, and `QWResponsesStore.sessionNodeId` chaining.
- **Client Request Traceability**: Added `QWClientIdentity` / `QWRequestTraceContext` header & User-Agent fingerprinting (`client=`, `req=`, `sess=`, non-loopback `peer=`, `499` disconnect logging) across server routes, `/ui`, and `Qwegn.app`.

### Changed
- **Unified Multi-Slot UMA & Image Preprocessing**: Extracted `QWResidencyPool` actor for multi-slot UMA admission and unified EXIF-oriented `vImage` RGB888 decoding in `QWMM+OrientedImage.swift`.
- **Offline `/ui` Web Standards**: Upgraded `/ui` with ARIA `tablist` roving keyboard navigation, Forced-Colors active underline, shared `dialog:modal` scroll containment, `closedby` / `commandfor` invokers, and CSS container queries.
- **Contributors & Release Hygiene**: Added `CONTRIBUTORS.md` and adopted release-only version bumps.

### Fixed
- **Inactive LSA Selector Prewarm (`#1`)**: Skipped `QWLSAOperator` selector pipeline prewarm when no LSA operator is active (e.g. `--lsa-mode off`), preventing Metal compiler failures on macOS 26.6.2 (contributed by [ViktorBystrov72](https://github.com/ViktorBystrov72) in [#1](https://github.com/qwegnai/qwegn/pull/1)).
- **Inkling-2 Exact Prefix-Cache Hits**: Cached post-`finalNorm` logits state (`finalNormRow`) to skip full 48-visit prefill on exact prompt repeats and honored `QWCacheConfig.entryLimit` (`32`).
- **Inkling-2 Thinking-Off & Non-Null Chat Content**: Prefilled `<|message_model|><|content_text|>` with `<|end_message|>` stop when thinking is disabled, and serialized non-streaming `content` as `""` instead of JSON `null` when `tool_calls` is absent.
- **Model Metadata & Prefill Progress Bar**: Published per-model `context_length`, `max_output_tokens`, modalities, `capabilities`, and `reasoning` metadata on `/v1/models` and `/api/show`, and kept `Qwegn.app` / `/ui` prefill progress bars fixed-height across idle and active states.
- **AArch64 `swifterror` ABI in Cognitive Shadow Log**: Replaced `FileHandle` + `defer { try? h.close() }` in `QWHttpServer.appendShadowRouteLog` with non-throwing POSIX `open`/`write`/`close`, eliminating `[UNRESTORED_X21_TAIL]` ABI violations on release builds.

## [0.42.0] - 2026-11-25

Brings SOTA Metal 4 decision-model throughput (up to `9.8×` faster long-context prefill and `15,829 tok/s` peak on `d1-2B-hq8`), full unmodified SDK & wire-protocol compliance across `/v1/systemone`, `/v1/decisions`, RFC 7578 multipart image uploads, standalone `qwg serve <decision-pack> --no-lazy` boot, and unified configuration/UI integration across CLI, `Qwegn.app`, and `/ui`.

### Added
- **SOTA Metal 4 Decision Kernels, Scratch Arena & Coalesced Batching (`QWEngine`, `QWMetal`)**:
  - Added Apple10 NAX cooperative-tensor BF16/F16 prefill GEMM (`qwg_gemm_bf16_nax_coop_fast`), tiled NAX topology FlashAttention kernels (`qwg_attn_topology_prefill_nax_d64` and `qwg_attn_topology_prefill_nax_d128`), and GPU residual cast (`qwg_cast_f16_to_f32`) backed by reusable `QWDecisionScratchArena` buffers and coalesced `/v1/decisions/batch` & `/v1/systemone/batch` GPU evaluation.
- **Full Wire & SDK Compliance, Standalone Boot & Decision Knobs (`QWServer`, `QWCli`)**:
  - Added full `/v1/systemone`, OpenAI `/v1/decisions`, RFC 7578 `multipart/form-data` image upload support, `/v1/models` decision capability metadata, UTF-8 `char_span` (`charSpan`) offsets, `always_emit_box` (`alwaysEmitBox`) best-effort coordinate emission, `X-Qwegn-Decision-Route` CORS-exposed routing headers (fail-open below `0.70` confidence, supporting `auto` and `high` `reasoningEffort`), and standalone `--no-lazy` server boot on decision packs.
  - Exposed all 6 decision configuration knobs uniformly across CLI (`--decision-model`, `--decision-compute-unit`, `--decision-load-vision`, `--decision-score-readout`, `--decision-truncation`, `--decision-always-emit-box`), `QWG_DECISION_*` environment variables, `/v1/settings`, `Qwegn.app`, and WebUI Settings.
- **Coherent Decision Telemetry & UI Across `Qwegn.app` and `/ui`**:
  - Extended `QWDashboardFacts` (`decisionSidecarVision`, `decisionSidecarArchetype`, `decisionSidecarPromptTokens`), upgraded the `QWMenuBar` `Decision Sidecar` popover card and `SettingsView`, added Monitor `#decision-telemetry-card`, integrated live Chat `Sidecar` status + `X-Qwegn-Decision-Route` badge + `Decisions →` handoff, added `Models` Decision capability badges + `Load Sidecar` action, and refactored `?tab=decisions` (`js/decisions-views.js`, `js/decisions.js`, `css/decisions.css`) with presets, wire-dialect switcher, threshold/`always_emit_box` controls, live SDK snippets, and sidecar loading.

### Fixed
- **Exact Decision UMA Estimation Under Vision Toggles (`QWDecisionWeightLoader`)**:
  - Updated `QWDecisionWeightLoader.estimateResidentBytes(root:descriptor:loadVision:)` (`QWDecisionPackEngine+Weights.swift`) to include resident vision buffer overhead when `loadVision && hasVision`, keeping UMA estimation error `<= 3.16%` across `d1-2B-hq8` and `d1-2B-bf16` with vision on and off (`vis1`/`vis0`).

### Verified
- **Paired Interleaved A/B (`v0.41.2` vs `v0.42.0`, Apple M5 Max, quiet GPU, 2 alternating rounds)**:
  - `d1-2B-hq8`: `41`-tok `17.63/17.62 -> 14.93/14.74 ms` (`min 13.81 ms`), `1035`-tok `201.81/201.64 -> 68.87/68.77 ms` (`15,829 tok/s`), `4059`-tok `1952.93/1948.68 -> 306.30/303.30 ms` (`13,781 tok/s`, `6.4×`), `8091`-tok `7243.65/7208.51 -> 743.40/733.66 ms` (`11,264 tok/s`, `9.8×`), and `/v1/decisions/batch N=8` `129.24/129.72 -> 38.39/37.86 ms` (`4.73 ms/item`, `3.4×`).
  - `d1-2B-bf16`: `41`-tok `51.91/51.88 -> 19.35/19.18 ms`, `1035`-tok `362.69/361.94 -> 93.20/87.06 ms` (`12,396 tok/s`, `4.1×`), `4059`-tok `2446.22/2461.90 -> 388.62/387.20 ms` (`10,732 tok/s`, `6.3×`), `8091`-tok `8041.58/8144.37 -> 871.93/872.90 ms` (`9,446 tok/s`, `9.3×`), and `/v1/decisions/batch N=8` `403.51/404.26 -> 43.36/43.71 ms` (`5.42 ms/item`, `9.3×`).
  - Dual-slot co-residency (`Inkling-2-Mini-QWP-Fast` + `d1-2B-hq8`): solo decode unchanged (`72.01/72.11 -> 72.28/71.96 tok/s`), concurrent decode retention `91.3%` while queued sidecar latency drops `22.89/23.64 -> 19.39/19.44 ms`.
  - UMA sizing accuracy: `d1-2B-hq8` `3,036.59 MiB` (est `2,940.61 MiB`, `3.16%`) / `vis0` `2,143.14 MiB` (est `2,170.18 MiB`, `1.26%`); `d1-2B-bf16` `4,259.42 MiB` (est `4,163.86 MiB`, `2.24%`) / `vis0` `3,365.92 MiB` (est `3,393.43 MiB`, `0.82%`).
- **Zero-Shim Ecosystem Compliance & Benchmarks**:
  - Unmodified `jeval check` PASS (`Kev` `/v1/systemone` and `OpenAI-Decisions` `/v1/decisions`), unmodified `typesafe-sdk` PASS (`models.list`, `system_one` text, multipart image), unmodified `s1kit test` `21/23` (`0` errors, `p50 26.0–27.1 ms`), Laya golden `10/10`, `/mcp` OK, `JevBench` (`231` questions) `77.06%` (`0.77056277`, `0` failed) on both `bf16` and `hq8`, and `ImageJevBench-Box` (`50` items) `50/50` valid (`Mean IoU 0.8032 / 0.8033`, Box Capability `80.59 / 80.93`, ranking floor PASS).

### Screenshots
Live co-resident telemetry (`Inkling-2-Mini-QWP-Fast` + `d1-2B-hq8` Decision Sidecar in unified memory), interactive `Decisions` workbench, and `Chat` auto-routing across `Qwegn.app` and `/ui`:

<table>
  <tr>
    <td width="64%"><img src="https://raw.githubusercontent.com/qwegnai/qwegn/main/assets/qwegn_dashboard.png" alt="Qwegn /ui Monitor with co-resident Decision Sidecar and UMA memory" width="100%"></td>
    <td width="36%" align="center"><img src="https://raw.githubusercontent.com/qwegnai/qwegn/main/assets/qwegn_menubar.png" alt="Qwegn.app control panel with co-resident LLM and Decision Sidecar" width="100%"></td>
  </tr>
  <tr>
    <td width="55%"><img src="https://raw.githubusercontent.com/qwegnai/qwegn/main/assets/qwegn_decisions.png" alt="Qwegn /ui Decisions workbench" width="100%"></td>
    <td width="45%"><img src="https://raw.githubusercontent.com/qwegnai/qwegn/main/assets/qwegn_chat_decision.png" alt="Qwegn /ui Chat with Decision Sidecar route badge" width="100%"></td>
  </tr>
</table>

## [0.41.2] - 2026-11-25

> **Supersedes `v0.41.1` for macOS 26 release binaries.** Fixes Xcode 26.6 AArch64 `swifterror` (`x21`) tail-call violations in `QWDecisionWeightLoader` and decision-sidecar path estimation / small-host admission on CI runners.

What's changed since the previous GitHub release (`v0.36.0`): brings native `ERNIE-5.5-Flash-Preview` support (`MXFP4` packing, `chatv2p3`, dialect-agnostic XML tool calls, capability-driven `DSpark`, `FlashIndexer`, and two UMA-resident Hugging Face packs), qualifies `LiquidAI/d1-2B` (`lfm3_vl`) on a generalized Metal 4 System-1 `QWDecision` sidecar engine (`/v1/decisions`, `/mcp`, `/ui?tab=decisions`, `Qwegn.app`), ships native `AVFoundation` video + unified 3D `MRoPE` + `ScreenCaptureKit` vision (`--video`, `--screen`, `0 B` idle RAM), accelerates structured JSON/tool-call generation (`1.27×–3.18×`) via lossless constrained speculation and hybrid 256-token paged Radix KV (`1.73×–2.43×` 8-subagent waves, up to `85.41×` lower foreground TTFT), and rolls out capability-wide Metal 4 NAX & `SHVQ` optimizations across `ERNIE-5.5`, `GLM-5.3-Flash`, `Qwen3.8-27B`, and `Qwen3.6-35B-A3B`.

### Fixed
- **Xcode 26.6 AArch64 `swifterror` (`x21`) ABI & Decision Sidecar CI Hardening**:
  - Hoisted `buf.contents()` outside throwing `withBytes` / `withRawPointer` / `withUnsafeBytes` closures under `withExtendedLifetime(...)` across `QWDecisionWeightLoader` (`QWDecisionPackEngine+Weights.swift`), `DriverWeightLoading+Ernie55.swift`, and `QWDSparkDrafter.swift`, eliminating all 6 `[UNSAVED_X21_TAIL]` `swifterror` violations caught by `Scripts/check_swifterror_abi.py` on `macos-26` Xcode 26.6 release builds (`qwg` and `qwg-menu`).
  - Guarded `QWEngineLifecycleActor.estimateDecisionSidecarBytes(at:)` (`QWEngineLifecycleActor+Decisions.swift`) against non-existent paths and single files so it never enumerates parent directories, aligned sub-1 GiB synthetic fixture handling in `QWResidencyAdmission.admitDecisionSidecar` (`QWResidencyAdmission.swift`) on `<= 8 GiB` runners while keeping strict `>= 1 GiB` and explicit `osReserveGiB` enforcement, and added regression unit tests in `QWResidencyAdmissionTests` and `QWDecisionServerTests`.

### Cumulative Highlights Since Previous Public Release (`v0.36.0` -> `v0.41.2`)
- **New Models & Packs (`v0.37.0`–`v0.41.2`)**: Onboarded `baidu/ERNIE-5.5-Flash-Preview` (`ernie5_5_moe`, `280B`/`14B`) via OCP `MXFP4` (`uint8` E2M1 + `F8_E8M0`) dequantization, native `chatv2p3` & parallel `<ernie:tool_call>`/`<invoke>` envelopes, bundled `dspark.qwp` + `flashindexer.qwp`, and Track C / Track A Hugging Face packs; onboarded `LiquidAI/d1-2B` (`lfm3_vl`, `bf16` & `hq8`) into native Metal 4 `QWDecisionPackEngine`.
- **New Engine Features (`v0.38.0`–`v0.41.0`)**: Generalized System-1 Decision Engine (`QWDecision`, 5-axis `QWDecisionModelConfig`, 12 Metal 4 shaders, `noul`/`choice`/`score`/`read`/`estimate`/`box` readouts, dual-slot sidecar co-residency, `/v1/decisions`, `/v1/systemone`, `/mcp`, `/ui?tab=decisions`, `Qwegn.app` sidecar controls); Native `AVFoundation` Video + unified 3D `MRoPE` across non-MLA/MLA VLMs + loopback `ScreenCaptureKit` vision (`--video`, `--screen`, `video_url`, `0 B` idle RAM under `--no-load-vision`); capability-driven `DSpark` runtime (`dspark.qwp`, `--dspark-confidence-schedule`, constrained draft seeding) and `FlashIndexer` sparse MLA residency.
- **Performance & Multi-Agent Scheduling (`v0.39.0`, `v0.41.0`)**: Lossless grammar-accelerated speculative decoding (`1.27×–3.18×` JSON/tool speedup across `Qwen3.6-35B`, `Qwen3.8-27B + DSpine-b16`, `Qwen3.8-Flash-Next`, and `Inkling-2-Mini`, `100%` SHA-256 greedy parity); hybrid-aware `256`-token paged Radix KV cache + `B=1..8` pooled slots + `512`-token slice-interleaved prefill (`1.73×–2.43×` 8-subagent wave speedup, `2.67×–85.41×` lower foreground TTFT, `-26%` to `-39%` deduplicated Warm-RAM); and capability-wide Metal 4 NAX & `SHVQ` optimizations (`ERNIE-5.5` Track C **`1,544.9 tok/s` prefill at 4K**, **`110.6 tok/s` `DSpark`** (`7.11 tok/step`), `89.48 tok/s` CLI JSON, `291/291` API, `78/100` `Code-100`; Track A **`1,051.3 tok/s`** at 4K (`+19.5%`), **`1,000.0 tok/s`** at 16K (`+22.9%`), `81–83/100` `Code-100`; `GLM-5.3-Flash` `+11.8%` prefill to `1,039.8 tok/s`; `Qwen3.8-27B` `+16.9%` to `1,002.0 tok/s`; `Qwen3.6-35B-A3B` `+2.9%` to `6,764.2 tok/s`).
- **Reliability, Seams & Server Hardening (`v0.37.0`–`v0.41.2`)**: Generalized `mxfp4` packing across `pack.py` & adapters, replaced family-specific chat/tool conditionals with dialect-driven `QWToolCallEnvelope` capabilities and enforced `Sources/Apps/` Law 15 seam linting, fixed iterative heap-stack `QWToolCallConstraint` trie walk for multi-KiB tokens, enforced deterministic `tool_choice: "required"`, isolated Inkling-2 `<|content_thinking|>` markers inside structured JSON, fixed deterministic mixed-QWP `DriverConfig.weightFormatChoice` selection, fixed `qwg serve --no-profile` to honor CLI `--mtp-*` overrides with auto `DSpark` defaults, and resolved `QWDecisionWeightLoader` Xcode 26.6 `swifterror` ABI clobbers + sidecar path estimation/admission on CI hosts.

## [0.41.1] - 2026-11-25

### Added
- **Qualified `LiquidAI/d1-2B` Native Metal 4 Decision Packs & `QWDecisionPackEngine`**:
  - Onboarded and qualified real `LiquidAI/d1-2B` (`a6bce774e6dde338dcc3036ceb5d91f78bf4c19e`, `lfm3_vl`) into QWP1 packs `~/.qwg/models/d1-2B-bf16` (`3.57 GiB`, `424 bf16` tensors) and `~/.qwg/models/d1-2B-hq8` (`2.38 GiB`, `282 bf16` + `142 hq8_g32` tensors), plus `audit_release.py` support for pure `bf16` and `decision_*` `quant_recorded` tags.
  - Added native Metal 4 `QWDecisionPackEngine` runtime (`LFM3`/`d1-2B` GQA + `slidingWindow` + gated `ShortConv` + arbitrary-width RMSNorm + zero-tail RoPE), `Lfm3VlVisionModel` (`RMSNorm`, gated conv, QK-norm, half-split Axial RoPE, `SwiGLU`), `packedSet` batching, and CLI/server decision loader wiring (`qwg decide`, `qwg inspect`).
- **`ERNIE-5.5-Flash-Preview` Hugging Face Packs & Infographic (`assets/qwegn_model_ernie55_flash_preview.svg`)**:
  - Published Track C (`qwegnai/ERNIE-5.5-Flash-Preview-hq4g32m5-hq8head-shvq15g64-shvq3g64`, `77.91 GiB` disk / `76.37 GiB` GPU, `2.36 bpw`) and Track A (`qwegnai/ERNIE-5.5-Flash-Preview-hq8g32-shvq15g64-shvq3g64`, `81.57 GiB` disk / `80.02 GiB` GPU, `2.52 bpw`), each bundling `dspark.qwp` (`4.39 GiB`, `86` tensors), `flashindexer.qwp` (`3.00 MiB`, `12` `BF16` tensors), and `assets/model.qlsa` (`4.05 MiB`) alongside `docs/models.md` and `README.md` catalog links.

### Fixed
- **LFM3/d1 Parity, ChatML Framing & Typed Readouts**:
  - Aligned `zeroTailPartialRotation` in `QWModelDescriptor+LFM.swift` and `QWDecisionPackEngine+Forward.swift`, expanded multimodal prompt token accounting (`QWDecisionRunner`), ChatML framing & pointer masking, discrete `score_median`, currency/scale `parseReadNumber`, and 2-space order-preserving `QWJson` rendering across `/v1/decisions`, `/v1/systemone`, and `/mcp`.
  - Qualified on Apple M5 Max (`--no-profile`, quiet GPU) vs CPU PyTorch `D1Model.system_one`: `100%` token-count parity (`49–439` tokens), `max |Δp| <= 0.0085` across text paradigms & 5-question `packedSet`, vision localization `IoU = 0.9967`/`1.0` vs PyTorch (`0.9748–0.9868` vs ground truth), `3×` bit-exact CLI determinism, `60.34 ms` (`noul`) / `65.70 ms` (`packedSet` 5Q) warm `p50`, `-745.4 MB` RSS under `--no-load-vision` (exit `64` refusal with `--image`), ADR-009 `32/32` (`100%` Top-1; `hq8` `ΔECE = 0.00065`, `ΔBrier = 0.00087`; `hq4` rejected at `ΔECE = 0.03172` and deleted), and `9/9 PASS` live dual-slot `QWServer` QA (`Inkling-2-Mini-QWP-Fast` + `d1-2B-bf16`).

## [0.41.0] - 2026-11-25

### Added
- **Capability-Driven `DSpark` Runtime (`dspark.qwp`) & `FlashIndexer` Sparse MLA Support**:
  - Generalized capability-driven `DSpark` (`QWModelCapabilities` `dsparkConfig`, Markov/Tree drafting, `FlashIndexer` `QW_MLA_FLASHINDEXER` residency, prior-commit overlap guard, hybrid suffix extension past `7` drafts, and confidence-adaptive early exit via `--dspark-confidence-schedule` / `QWSettingsCoherence`) across `QWMTP`, `QWEngineDriver`, `QWCli`, `QWServer`, and `Qwegn.app`.
  - Wired `DSpark` into constrained speculation (`proposeCanonicalOrFilteredDrafts`, async `asyncSpecAllowed` slot plans), fixed `qwg serve --no-profile` to honor CLI `--mtp-drafts`/`--mtp-backend` overrides and auto-enable `DSpark` defaults (`mtpDrafts = min(7, maxDrafts)`, `dsparkConfidenceSchedule = true`), and added `Scripts/quantize/{ernie55_adapter,pack_dspark}.py` for `dspark.qwp`, `flashindexer.qwp`, BF16 routers, and selective dense repacking with PyTorch parity oracles.

### Changed
- **Capability-Wide Metal 4 NAX & `SHVQ` Kernel Optimizations + Deterministic QWP1 Format Selection**:
  - Eliminated `SHVQ15` `> 32 KiB` AGX constant-cache spills via isolated sub-2-bit PSOs, fixed `SHVQ15` subnormals, added branchless 6-byte `frag8` unpacking, staged `SHVQ3` threadgroup SRAM LUTs across grouped-MoE NAX, fused MoE & shared `SwiGLU` NAX epilogues, and coalesced `T=2..8` + sub-2-bit `CompactQWP` dispatch.
  - Added isolated `8 KiB` per-mode / `BM=16` / `ALIGNED_M` NAX PSOs for Heavily Compressed Attention (`HCA wo_a`), shrunk `MLA-DSA` shared-KV NAX scratch to `16 KiB` across `h32`/`h64`, extended `qwg_router_nax_f32` (`E % 64 == 0 && rows >= 64`) with a `plainF16` router bridge, and added fused-merge `8 KiB` pair-staged `mHC` row kernels.
  - Fixed intermittent `weight loader/plan mismatch at allocation 3` on mixed-format QWP1 packs in `DriverConfig.weightFormatChoice` via deterministic lexicographic minimum-name selection excluding auxiliary `lm_head.weight`/`embed_tokens`/vision tensors.

### Verified
- **Cross-Family Paired Interleaved A/B on Apple M5 Max (`48/48` runs `rc=0`, `0` steady-state allocs, `0` non-ERNIE regressions)**:
  - **`ERNIE-5.5-Flash-Preview` Track C** (`76.37 GiB`, `2.36 bpw`, `BASE = N/A` since `master v0.40.0` predates mixed-QWP `headFormat` support): `P4K` (`4,087` tok) **`1,544.9 tok/s`** GPU (`1,542.8–1,544.9`, `2.65 s`, `43.28` serial decode), `16K` **`1,259.9 tok/s`** (`3/3` verbatim needles across `d0` & `d7n_cs`), `128`-tok `DSpark d=7 neural --dspark-confidence-schedule`: Prose `44.42 -> 59.95 / 62.65` (`--moe-spec-budget 36`), Code `44.61 -> 63.92`, JSON `44.78 -> 74.83 / 89.48 tok/s` (`5.77 tok/rnd`, `90.5%` accept); active-DSpark `llmprobe`: **`110.6 tok/s`** (`7.11 tok/step`, `3.47x`), **`291/291 (100%)`** conformance, **`36/36 (100%)`** fidelity, **`60.9 tok/s`** at `64K`, `38x` prefix cache, `75/100` (`--reasoning off`) / `78/100` (`--reasoning low`) `Code-100`; Law 28 `S1–S9` **`9/9 PASS`** (`0` errors).
  - **`ERNIE-5.5-Flash-Preview` Track A** (`80.02 GiB`, `2.52 bpw`, paired `BASE v0.40.0` vs `CAND`): `P4K` **`879.6 -> 1,051.3 tok/s` (`+19.5%`)**, `16K` **`813.7 -> 1,000.0 tok/s` (`+22.9%`, `3/3` needles)**, `100%` `BASE == CAND` SHA-256 identity across all rounds, JSON `128`-tok `32.15 -> 62.78 tok/s` (`+95.3%`), and **`81/100` `Code-100`** (`HumanEval 47/50`, `MBPP 34/50`).
  - **Non-ERNIE Fleet**: `GLM-5.3-Flash` `930.1 -> 1,039.8 tok/s` (`+11.8%` prefill, `+1.7%` decode); `Qwen3.6-35B-A3B` `6,571.9 -> 6,764.2 tok/s` (`+2.9%`, `4/4` bit-exact, `--no-load-vision --image` `ExitCode(2)`, `QWServer --no-lazy` `+1,013.66 MiB` vision RSS); `Qwen3.8-27B` `857.2 -> 1,002.0 tok/s` (`+16.9%` prefill, `4/4` bit-exact); `Inkling-2-Mini-QWP-Fast` `2,791.0 -> 2,794.2 tok/s` (`15 ms` step latency, `4/4` bit-exact); `Qwen3.8-Flash-Next` `2,176.5` vs `2,173.8 tok/s` (`18 ms` step latency, `-27.13 GiB` `--stream-ngrams`, `CAND r1==r2` bit-exact; Law 11 note on `qwg_router_nax_f32` BF16-split FP order).

## [0.40.0] - 2026-11-25

### Added
- **Native `AVFoundation` Video Pipeline, Unified 3D `MRoPE` & Privacy-Gated `ScreenCaptureKit` Vision (`specs/09-native-video-and-screen-vision.md`)**:
  - Completed `--video <path>` (`--video-fps`, `--video-max-frames`, `--screen`) in `qwg run` and `qwg repl` (`/video`, `/screen`), honoring `--load-vision` / `--no-load-vision` residency toggles (`ExitCode(2)` refusal when disabled, `0 B` idle vision RAM when unloaded).
  - Added hardware `VideoToolbox` / `AVAssetImageGenerator` frame extraction (`QWVideoDecoder`) with bin-midpoint timestamp sampling, ISO-BMFF/EBML/RIFF magic-byte verification, `0o600` ephemeral temp files, IP-pinned SSRF-guarded remote fetch (`64 MiB` / `300 s` / `64`-frame caps), bounded LRU caching, `qwVideoSmartResize` + 0-ULP `vDSP` temporal `.conv3d` patch slabs (`qwBuildVideoPatchSlab`), vocabulary-driven media tag resolution (`QWMultimodalTagResolver`), and `[Frame 1..K]` fallback when `videoTokenId < 0` (`Inkling-2`).
  - Generalized 3D `MRoPE` (`QWRopeTables.buildMRoPECosSin` for `.qwenInterleaved` `[11,11,10]` and `.bailingVideoRope` `[W,H,T]`, `customTable` in `QWRopeApply`, `mropeTurnTableBuffer` + `mropeDelta` across `DriverPrefill`, `QWHybridLayerKernels`, and `QWMTP*`) to non-MLA `GQA`/`QSA`/`LSA`/`MBSA` VLMs alongside `MLA`, fixed mixed image+video token-order interleaving (`QWX0Plan.visionSpanIsVideo`), and detected QWP-packed `model.visual.*` tensors in `QWInkling2Weights`.
  - Added multi-protocol video parts (`video_url`, `video`, `input_video`) across `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/messages/count_tokens`, offline `/ui?tab=chat` video upload + screen capture (`js/chat-media.js`), loopback-only `POST /v1/vision/screen-capture` (`QWScreenCaptureHelper` with `CGPreflightScreenCaptureAccess()` preflight), and `Qwegn.app` `Capture Screen to Chat…`.
  - Verified on Apple M5 Max (`100%` SHA-256 text parity vs `v0.39.0`, `0` steady-state allocations, `3/3` needles at `14.6K` tokens) and live multi-turn `QWServer` video + `T=0.6` `json_schema` extraction across `Qwen3.6-35B-A3B` (`0.43 s`), `SenseNova-Loop-9B` (`0.64 s`), `Ling-3.2-air` (`0.85 s`), and `Inkling-2-Mini-QWP-Fast` (`3.30 s`).

## [0.39.0] - 2026-11-25

### Added
- **Lossless Grammar-Accelerated Speculative Decoding & Canonical Draft Seeding (`specs/08-agentic-speculation-and-radix-scheduling.md`)**:
  - Enabled zero-ULP constrained speculative verification across `QWEngineDriver` (`MTP`, `DFlash 2`, `DSpine`, `PLD`/`.ngram`), `QWHyperStreamRuntime`, and `QWInkling2Runtime` (`DSpark`/`PLD`) for `QWStructuredOutputConstraint` (`json_schema` / `json_object`), `QWToolCallConstraint` (JSON `<tool_call>` and XML `<invoke>` envelopes), `QWThinkingBudgetConstraint`, and `QWPrefixHealingConstraint` (`QW_CONSTRAINED_SPEC`, default `1`).
  - Added response-local style-learning canonical structural draft seeding (`proposeCanonicalDrafts` in `QWJSONConstraint+Speculation.swift` and `QWToolCallConstraint.swift`) with deterministic single-allowed-token fast-forwarding (`allowedTokens().count == 1`), masked-support variational renormalization (`mtpRelaxedMargin`, `mtpRelaxedMinProb`), Gumbel/target stochastic support parity (`QWSampler+Distribution.swift`), and single-wait `Commit(r-1) -> ConstrainedDraft(r) -> Verify(r)` async pipelining.
  - Paired interleaved A/B on Apple M5 Max (`100%` SHA-256 greedy parity): `Qwen3.6-35B-A3B-hq4g32` (`MTP`) `json_schema` **`140.7 -> 184.1 tok/s` (`1.31×`)** / `tool_call` **`139.6 -> 209.7 tok/s` (`1.50×`)**; `Qwen3.8-27B-hq4g32 + DSpine-b16` `json_schema` **`31.5 -> 78.3 tok/s` (`2.49×`)** / `tool_call` **`31.2 -> 99.1 tok/s` (`3.18×`, `5.38 tok/rnd`)**; `Qwen3.8-Flash-Next` (`MTP`) `json_schema` **`56.2 -> 89.8 tok/s` (`1.60×`)** / `tool_call` **`55.9 -> 90.4 tok/s` (`1.62×`)**; `Inkling-2-Mini-QWP-Fast` (`DSpark`) `json_schema` **`65.2 -> 82.9 tok/s` (`1.27×`, `4.21 tok/rnd`)**.
- **Hybrid-Aware 256-Token Paged Radix-Tree KV Cache, `B=1..8` Pooled Slots & Slice-Interleaved Prefill**:
  - Added `QWRadixPrefixTree` (`256`-token FNV-1a `PageEdge` lookup, subtree pruning, and shared-branch detection, `QW_RADIX_KV` default `1`), `QWPrefixPageLedger` (reference-counted physical `QWKVCache` page byte deduplication across Warm-RAM entries), zero-copy `QWKVCache.sliceSharedPrefix` for pure-attention models, and `4,096`-token periodic + dynamic wave-leader (`isBranchCheckpoint`) recurrent checkpoints for hybrid `GDN`/`KDA`/`PLE`/`MLA` models.
  - Scaled pooled session slots to `8` on Macs `>= 64 GiB` (`1..16` via `--max-concurrent-sessions`) with prewarmed `m = 2...8` batched head PSOs (`qwg_head_{f16,q4g,q4g_argmax,q4g_rp}_m2..8`) and `gpuTurnArbitrator` step-boundary yielding in `forwardBatchStep`.
  - Added scheduler-level `.prefilling(cursor)` × `.decoding` slice interleaving (`QW_INTERLEAVED_PREFILL_SLICE`, default `512` tokens when decode slots or arriving requests are active; full adaptive `prefillChunkRows` when solo) across `QWEngineDriver` and `QWHyperStreamRuntime`.
  - Paired interleaved A/B on Apple M5 Max: 8-subagent shared-prefix wave (`2,877` prompt tokens) speeds up **`5.94s -> 3.43s` (`1.73×`, TTFT `2,550 -> 729 ms`)** on `Qwen3.6-35B-A3B`, **`34.31s -> 14.09s` (`2.43×`, TTFT `15,576 -> 3,418 ms`)** on `Qwen3.8-27B`, and **`18.05s -> 8.51s` (`2.12×`, TTFT `8,037 -> 1,759 ms`)** on `SenseNova-Loop-9B` while retaining all `8` subagent slots (`-26.1%` to `-39.1%` deduplicated Warm-RAM); interactive foreground chat TTFT under concurrent `9,605`-token background prefill drops **`1,711.1 -> 639.8 ms` (`2.67×`)** on `Qwen3.6-35B-A3B`, **`10,522.5 -> 123.2 ms` (`85.41×`)** on `Qwen3.8-27B`, and **`5,489.6 -> 75.1 ms` (`73.05×`)** on `SenseNova-Loop-9B` with `100%` SHA-256 bit-exact parity.

## [0.38.0] - 2026-11-25

### Added
- **Generalized System-1 Decision Engine (`QWDecision`, `ADR-001`–`ADR-010`)**:
  - Added capability-driven 5-axis `QWDecisionModelConfig` (`SequenceFraming`, `AttentionTopology`, `HeadKind`, `QWCalibrationEngine`, backbone/vision capabilities) across 18 config-decoded families and synthesized generative verbalizers (`specs/07-decision-models-and-apis.md`), bijective UTF-8 `QWTokenizer+Offsets`, and zero-Metal `L1.5 QWDecision` decoding across all 4 paradigms (`noul`/`predicate`, `choice`/`multiple` with `abstain_below`, `score` with `"mean"`/`"median"`/`"argmax"`, `read` with `extractText`/`extractNumber`/`extractDate`/JSON pointer `path`, `estimate`, and 4D Halton IoU `box`/`boxes`).
  - Added dual-slot sidecar co-residency (`QWEngineLifecycleActor+Decisions`, `QWResidencyAdmission.admitDecisionSidecar`), step-boundary GPU preemption (`QWGPUTurnArbitrator`), `QWEngineDecisionAdapter` (Metal 4), and `QWCoreAIDecisionAdapter` (`#if canImport(CoreAI)` macOS 27+ `.aimodel` ANE bundles).
  - Added 12 Metal 4 decision shaders (`Sources/QWMetal/Decision/`), 5 operators, `QWDecisionMetalPipeline`, sparse BF16 verbalizer head (`U <= 256`), `Scripts/quantize/{lfm_adapter,decision_adapter}.py`, and PyTorch parity oracles (`Scripts/oracle/{decision_oracle,vision_decision_oracle}.py`).
  - Added multimodal vision decision support (`LFM3-VL` hybrid `VisionConv2d` + 2D Axial-RoPE GQA and `SigLIP2` NaFlex/fixed-patch in `QWLFMVision`, `QWVisionCheckpoint`, `QWVisionConfig`, `QWMM`) honoring `--load-vision` / `--no-load-vision` residency toggles and `box`/`boxes` visual localization.
- **Multi-Dialect Decision APIs, CLI, `/ui?tab=decisions` & `Qwegn.app` Parity**:
  - Exposed REST endpoints (`/v1/decisions`, `/v1/decisions/batch`, `/v1/systemone`, `/v1/systemone/batch`, `/v1/laya`, `/v1/decisions/model/load`, `/v1/decisions/model/unload`), stateless streamable-HTTP `/mcp` (`qwg_decide`, `laya_ask`, `laya_guard`, `laya_status`), opt-in `reasoning.effort: "auto"` sidecar complexity routing, and CLI commands (`qwg decide`, `qwg calibrate`, `qwg mcp`).
  - Added offline WebUI `/ui?tab=decisions` workbench (`decisions.css`, `decisions.js`, `settings.js`), `QWDashboardViewModel` single-source telemetry fields, and `Qwegn.app` (`QWMenuBar`) `DECISION SIDECAR` popover card + Settings dual-slot controls.

## [0.37.1] - 2026-11-24

### Changed
- **Generalize MXFP4 Source Packing (`quant_method: "mxfp4"`)**: Added `validate_source_quant_config` and `unpacked_safetensors_shape` in `Scripts/quantize/hq_blocks.py`, switched generic `Scripts/quantize/pack.py` (with unpacked MXFP4 `expected_shapes` + cross-shard `_scale_inv` fallback) and `Scripts/quantize/eval_hq_kld.py` to `dequant_with_scale_inv`, unified FP8/MXFP4 config validation across `ernie55_adapter.py`, `bailing_adapter.py`, and `maxwell_adapter.py`, and fixed the trailing assistant prompt double-newline in `Scripts/quantize/ernie55_chat_template.py`.
- **Template Capabilities & Dialect-Agnostic XML Tool-Call Envelopes**: Replaced `QWToolCallParser+Ernie.swift` with `QWToolCallParser+XMLInvoke.swift` (`QWToolCallEnvelope`), replaced `isErnieChatTemplate` with capability properties (`toolCallEnvelope`, `thinkOpenTag`, `thinkCloseTag`, `usesStandardToolCallHint`) on `QWChatTemplate`, generalized `QWChatStreamParser`, `QWToolCallParser`, `QWChatTurnPreparer`, `QWToolCallConstraint`, `QWThinkingBudgetConstraint`, `QWChatPolicyResolver`, and `QWEngineLifecycleActor+Chat` so custom namespaced envelopes (`<ns:tool_call>`) and parallel XML `<invoke name="...">` inside `<tool_call>` work without model-specific conditionals while preserving JSON string-literal payloads.
- **Full `Sources/Apps/` Capability-Seam Linting**: Extended `Scripts/lint_capability_seams.py` to enforce Law 15 seam isolation across `Sources/Apps/` in addition to `Sources/Operators/`, `Sources/Subsystems/`, and `Sources/Runtime/`.

## [0.37.0] - 2026-11-24

### Added
- **Native `baidu/ERNIE-5.5-Flash-Preview` MXFP4 Packing, `chatv2p3` Wire Protocol & `<ernie:tool_call>` Support**:
  - Added OCP MXFP4 blockwise dequantization (`dequant_mxfp4_blockwise` + `dequant_with_scale_inv`, `uint8` E2M1 nibbles with `F8_E8M0` group-32 scales and fail-closed NaN rejection) in `Scripts/quantize/hq_blocks.py`, wired into `ShardReader.get()` (`Scripts/quantize/ernie55_adapter.py`) alongside `Scripts/quantize/ernie55_chat_template.py` for checkpoints shipping `encoding_ernie.py` without `chat_template.jinja`.
  - Quantized `baidu/ERNIE-5.5-Flash-Preview` (`de3ffd3e8d55ec0f68ca51cbcec52d2a6490cf3b`, skipping optional `model.mtp.*` DSpark draft and `flashindexer/` tensors per Base parity) into `ERNIE-5.5-Flash-Preview-hq8g32-shvq15g64-shvq3g64` (`82 GiB`, `2.52 bpw`, `model.qlsa`, `pack.py --audit-release` `ok: true`).
  - Implemented native Swift `chatv2p3` rendering (`QWChatTemplate+Ernie.swift`, byte-for-byte parity with `encoding_ernie.py` sorted JSON spacing), `<ernie:tool_call>` / parallel `<invoke>` / `<parameter>` streaming and non-streaming parsing (`QWToolCallParser+Ernie.swift`), `</ernie:think>` closing in `QWThinkingBudgetConstraint` / `ChatStreamCapture` / `QWTokenizer+Codec`, and ERNIE constrained tool-call tag transitions in `QWToolCallConstraint`.
  - Qualified on Apple M5 Max (`128 GB` UMA, `0` steady-state allocations): `p512` `883.6–890.5 tok/s` prefill / `33.34–34.17 tok/s` decode (`f16`/`q8`/`q4`, `81.31–81.34 GiB` RAM, `100%` SHA-256 repeatability); `p4k` **`1,179.9–1,197.9 tok/s`** (`f16`), `1,083.8–1,144.7` (`q8`), `1,064.8–1,069.2` (`q4`) prefill and `33.01–33.60 tok/s` decode (`85.71–85.74 GiB`); `16K` (`15,918` prompt tokens, `q4` KV) **`913.5 tok/s`** prefill / **`32.13 tok/s`** decode / **`3/3` verbatim passkey recall**; `QWServer` `S1–S9` `9/9`, `Scripts/probe_quality.py` `6/6`, `llmprobe --full` (`--reasoning low`): **`292/292 (100%)`** conformance, **`100%` (`strong`)** capability, `72/88 (81.8%)` agentic, `36/36 (100%)` correctness & `3/3 (100%)` determinism; and `llmprobe --full --eval-suite code` (`--reasoning off`): **`83/100` Code-100** (`HumanEval 47/50`, `MBPP 36/50`), **`100%` fidelity**, `1,008.9 tok/s` prefill, `33.8 tok/s` decode (`33.5` at `513` → `32.3` at `65.6K`), `77.2×` prefix cache (`18 ms`), and `0` server errors.

### Fixed
- **Iterative `QWToolCallConstraint` Trie Walk**: Replaced recursive trie traversal in `QWToolCallConstraint.allowedTokens()` with an explicit heap stack so tokenizers containing multi-kilobyte ByteLevel BPE whitespace tokens (e.g., ERNIE-5.5's `4,096`-space token) never overflow 512 KiB Dispatch QoS thread stacks.
- **Deterministic `tool_choice: "required"` & Forced-Tool Enforcement**: Prevented raw `<ernie:tool_call>` token injection in ERNIE system instructions (`QWChatTurnPreparer`) and enforced mandatory tool-call initiation (`requireToolCall` / `startsInThinking`) in `QWToolCallConstraint`.
- **Constrained-JSON Marker Isolation (`allowInk2Markers`)**: Gated Inkling-2 `<|content_thinking|>` channel switching in `QWChatStreamParser` on `QWEngineLifecycleActor.isInkling2Template()` across all HTTP handlers (`ChatCompletions`, `Responses`, `Anthropic`, `OllamaChat`, `OllamaGenerate`) so literal delimiter strings inside structured JSON remain raw data.

## [0.36.0] - 2026-11-24

### Added
- **Capability-Driven Architecture Seams (`QWModelCapabilities`) & Linter Gate (`Scripts/lint_capability_seams.py`)**: Replaced `descriptor.family` / `model_type` branches across `Sources/Operators/`, `Sources/Subsystems/`, `Sources/Runtime/`, and `Sources/Apps/` with typed geometry capability structs (`KDAConfig`, `MLAConfig`, `MoERoutingConfig`, `SwiGLUClampConfig`, `VisionTowerConfig`, `MTPConfig`, `HCAConfig`, `BlockSparseGQAConfig`, `MRoPELayout`) codified in `AGENTS.md` Law 15 and `.agents/skills/qwegine-model-onboarding/SKILL.md`.
- **Generalized Group-Limited MoE, SwiGLU Clamp, Full-Rank KDA, Direct-Q MLA & VideoRoPE ViT**: Added `n_group` / `topk_group` group-limited top-`k` Metal routing up to `1,024` experts, per-layer `SwiGLUClampConfig` (`expert_swiglu_limit_list` / `share_expert_swiglu_limit_list`), full-rank KDA projection (`no_kda_lora` / `use_kda_lora` / `kda_lower_bound`), Direct-Q decoupled-RoPE MLA (`q_lora_rank: null`, interleaved RoPE, `norm_qkv`), and `2`-layer GELU `linear_proj` + `3D VideoRoPE` `[8, 12, 12]` Metal ViT encoding.
- **Native `bailing_moe_v3_vl` & `Ling-3.2-air-hq8g32-shvq3g64` Support**: Added runtime/packer support (`bailing_moe_v3`, `bailing_moe_v3_vl`, `bailing_moe_v3_text`) in `Scripts/quantize/bailing_adapter.py` and `QWEngineDriver`. Qualified `Ling-3.2-air-hq8g32-shvq3g64` (`124B` total / `5.5B` active, `52.23 GiB` `3.41 bpw`, `35` KDA + `7` MLA + `512` experts + NextN `mtp=1` + `27L` ViT) on Apple M5 Max: **`2,847 tok/s`** at `4.1K` / **`2,141.4 tok/s`** at `10.4K` prefill, **`100.0 tok/s`** sustained `QWServer` decode (**`110.5 tok/s`** predictable spec `1.98 tok/step`, **`103.7 tok/s`** CLI `mtp=1`), **`14.5×` warm prefix cache (`37 ms`)**, **`100%` capability (`strong`)**, **`98.04%` fidelity (`36/36`)**, **`285/293` API conformance**, and **`3/3` needle recall at `16.9K`** with `0` steady-state allocations.
- **Dedicated Model Catalog (`docs/models.md`) & `Ling-3.2-air` Infographic**: Added `docs/models.md` housing detailed model cards and SVG infographics (`assets/qwegn_model_ling32_air.svg`).

### Changed
- **Streamlined Root `README.md`**: Refactored `README.md` to focus on installation, quick start, compact architecture/benchmark overview tables, multi-protocol server endpoints, CLI reference, and documentation links, moving per-model SVG cards into `docs/models.md`.

## [0.35.0] - 2026-11-23

### Added & Improved
- **Full-Slice Q4 Multi-Head LSA (`isMultiHeadDirect`), Exact Softmax Correction & Decode Loop Guard**: Replaced symmetric 64-dim centroid indexing with direct per-head Q4 pooled scoring across `qwlsa_mh.metal` / `qwlsa_mh_select.metal` (`isMultiHeadDirect`, paired with the `6.0 MiB` `model.qlsa` `QLSA` v1 SVD sidecar on `Inkling-2-Mini-QWP-Fast`), added two-stage `p16 -> p4` top-`k` selection with exact full-denominator `all_sum` softmax tail correction and a 64-token dense tail, added capacity-scaled adaptive LSA key budgeting (`98K -> 4096`, `196K..786K -> 8192`, `>= 786K -> 16384` for `k=2048`), and added a deterministic 256-token / `period <= 96` / `3x` 8-gram decode loop guard across sampler/runtime paths (`QWLSA*`, `QWInkling2*`). Achieves **`3/3 (12/12, 100%)` multi-hop needle recall across `8K–1M` tokens** on both `Inkling-2-Mini-QWP-Fast` (`2,338.6 / 43.40` at 256K, `2,298.4 / 42.32` at 512K, **`2,016.5 / 33.92 tok/s` at 1M** = **`3.74×` prefill / `4.51×` decode** vs dense `539.3 / 7.52`, `29 ms` p50, `21.77 GiB` RSS, `0` allocs) and `Inkling-2-Mini-MLX` (`2,235.8 / 35.17` at 256K, `2,338.0 / 33.73` at 512K, **`1,935.1 / 28.15 tok/s` at 1M** = **`3.59× / 3.87×`**, `29.91 GiB` RSS, `0` allocs).
- **100% CLI–GUI (`QWMenuBar`)–WebUI (`/ui`) Parameter Parity**: Exposed `gdnPrefill` and `lsaCrossLoopReuse` (`[restart]` badge) + `none` log level in `QWMenuBar` (`QWKnobRegistry`, `SettingsView+Performance.swift`, `SettingsView+Sections.swift`), published `lsaCrossLoopReuse` in `QWServerConfigKnobs`, modularized `/ui` Settings into `Sources/Apps/QWServer/Resources/js/settings.js` exposing all CLI knobs and typed `QWKnobValue` `engineTuning` controls with XSS escaping and JSON round-trip validation, and added `QWCliGuiWebParityTests`.

## [0.34.1] - 2026-11-23

> **Supersedes `v0.34.0` for macOS 26 release binaries.** Fixes an upstream Xcode 26.6 LLVM AArch64 codegen crash (`SIGSEGV` in `_swift_errorRetain` during model/prefix loading) and adds an automated binary ABI verification gate.

### Fixed
- **Xcode 26.6 AArch64 `swifterror` (`x21`) Tail-Call Miscompilation Fix**: Hoisted `MTLBuffer.contents()`, `region.pointer`, and `cpuPointer` lookups outside throwing/rethrowing `withRawPointer`, `withBytes`, and `withUnsafeBytes` closures ending in `memcpy` / `memmove` (with `withExtendedLifetime` guards where buffers are not referenced later in scope) across `QWAlloc` (`QWArena`), `QWAttn` (`QWHCAAttnOperator`, `QWMLAAttnOperator`), `QWKV` (`QWKVCache`), `QWLoad` (`QWCheckpointTensorLoader`), `QWEngine` (`PrefixCache`, `QWEngineSharedPrefix`, `QWEngineSessionState`, `ColdExpertStreamer`, `DriverInitPhases+Ernie55`, `DriverWeightLoading*`, `QWInkling2*`), and `QWBenchKernels`, eliminating all 9 unsaved-`x21` sibling tail-call miscompilations in `qwg` and `qwg-menu`.
- **Automated `swifterror` ABI Release & CI Gate (`Scripts/check_swifterror_abi.py`)**: Added CFG-based arm64 Mach-O disassembly verification (`objdump` + `otool`) enforcing zero unsaved or unrestored `x21` sibling tail-calls across CI (`Scripts/ci.sh` on `qwg-bench-kernels`) and `.github/workflows/release.yml` (`qwg`, `qwg-menu`, `qwg-smc-helper`).

### Cumulative Highlights Since Previous Public Release (`v0.28.2` -> `v0.34.1`)
- **New Features**: Native Metal 4 `Inkling-2-Mini` (`inkling2_loop` 48-visit looped MoE + 23-layer ViT + `DSpark` Markov/Tree MTP + `QWP1` packs), configurable `--mtp-drafts 0...15` across CLI/Menu Bar/WebUI, Wave-3 `--moe-spec-budget 0...64` speculative MoE pruning, `--lsa-cross-loop-reuse` with `1M` (`1,048,576`-token) 64-bit LSA indexing, privileged `qwg-smc-helper` LaunchDaemon non-root fan control, WebUI smart free-scroll streaming + collapsed reasoning quips, and full `attention_bias` / `qkv_bias` / hybrid `sliding_window` support.
- **Performance & Optimizations**: Continuous `B=2..4` wavefront batching (`+55%` aggregate `QWServer` decode on `Inkling-2`), `T=2..8` indirect compact QWP MoE dispatch (`+11.97%` `GLM-5.3-Flash`, `+5.70%` `Qwen3.6-35B-A3B`) + single-TG MoE bucketing, 4-SG shared-KV Sink+RelBias decode + generalized `T<=16` union MoE pruning (`+12.67%` `Inkling-2-Mini-QWP-Fast` `code256`), FA4-inspired NAX MBSA selector & Split-K QSA decode kernels (`+10.70%` `Qwen3.8-Flash-Next`), and shared-prefix `B=2...8` verify kernels.
- **Fixes & Hardening**: Single-window AppKit Settings entry + non-Latin ANSI hotkeys (`QWMenuBarShortcutRouter`), zero-jitter SF Mono menu bar telemetry columns, `-mmacosx-version-min=26.2` NAX compile guard + `Scripts/check_metallib_target.py`, and single descriptor-driven `QWEngineDriver` / `QWHyperStreamRuntime` engine unification.

## [0.34.0] - 2026-11-23

What's changed since the previous GitHub release (`v0.28.2`): introduces full native Metal 4 support and quantized `QWP1` packs for `Inkling-2-Mini` (`inkling2_loop` 48-visit looped MoE + 23L ViT + `DSpark` Markov/Tree MTP), expands `--mtp-drafts 0...15`, adds Wave-3 generalized MoE speculative pruning (`--moe-spec-budget`), `1M` 64-bit LSA indexing (`--lsa-cross-loop-reuse`), indirect `T=2..8` compact QWP dispatch, and `B=2..4` wavefront batching, ships privileged `qwg-smc-helper` LaunchDaemon fan control, WebUI free-scroll streaming + quips, single-window AppKit settings/hotkeys + fixed-advance zero-jitter Menu Bar telemetry, FA4-inspired MBSA/QSA/Split-KV Metal 4 attention optimizations, attention bias + hybrid SWA page-ring compaction, and completes the single descriptor-driven `QWEngineDriver` / `DenseModule` / `QWHyperStreamRuntime` engine unification.

### v0.34.0 — Wave-3 Generalized MoE Speculation, 1M 64-Bit LSA, Indirect Compact QWP & Continuous Wavefront Batching (`perf/ink2-wave3`)
- **Shared-KV Attention & Speculative MoE Pruning**: Added 4-SG shared-KV Sink+RelBias `T<=4` decode (`QWAttnSinkRelApply`, `qwattn_sink_rel_decode_batch.metal`) and generalized `T<=16` union MoE speculative pruning (`--moe-spec-budget 0...64`, default `0` lossless; `qw_router_union.metalh`, `qwrouter.metal`, CLI/`QWServer`/`QWMenuBar` knob plumbing), accelerating `Inkling-2-Mini-QWP-Fast` `code256` **`76.49 -> 86.18 tok/s` (`+12.67%`)**, `Cand5` `2K` **`55.62 -> 59.37 tok/s` (`+7.04%`)**, `MLX` `2K` **`47.65 -> 50.23 tok/s` (`+5.30%`)**, and `Qwen3.6-35B-A3B` `2K` **`175.50 -> 183.86 tok/s` (`+5.49%`)** with `3/3` `16K` needles and `89/89 (100%)` `llmprobe` conformance / `9/10` `code10`.
- **Fused Single-TG MoE Bucketing & Indirect Compact QWP**: Replaced split MoE group+bucket passes with single-TG `qwg_moe_group_build_buckets_f32` (`QWGemmMoE+Group`, `qwgemm_moe_group_f32.metal`), added offline `AQ8` KL oracle (`Scripts/spike/eval_ink2_quant_kl.py`: `mean_kl=0.001159`, `100.0%` top-1), trimmed serial router barriers, overlapped async command buffers + cooldown fusion (`Inkling-2-Mini-QWP-Fast` `2K` `59.85 -> 60.34 tok/s`), and wired `T=2..8` indirect compact QWP dispatch in `QWMoEOperator` + `QWGemmMoE+CompactQWP` — lifting **`GLM-5.3-Flash` (`DFlash2`) `39.20 -> 44.08 tok/s` (`+11.97%`)** and **`Qwen3.6-35B-A3B` `166.23 -> 175.12 tok/s` (`+5.70%`)** with `0-ULP` SHA-256 identity across all 7 flagship packs and `0` steady-state allocations.
- **1M 64-Bit LSA Indexing & Cross-Loop Reuse**: Upgraded `QWLSAOperator` and `QWLSAConfig` to 64-bit `1M` (`1,048,576`-token) indexing, added `DSpark <= 16K` KV capping and `1M` `q4` admission (`Inkling-2` `19.72 GiB` planned / `20.69 GiB` RSS / `78.01 tok/s`, `0` allocs), and introduced `--lsa-cross-loop-reuse` (`Inkling-2` `32K` **`56.63 -> 61.83 tok/s` (`+9.18%`)**, `SenseNova-Loop-9B` `32K` **`48.65 -> 50.32 tok/s` (`+3.41%`)**, `3/3` `128K` multi-hop needles at `71.85 tok/s`).
- **Continuous `B=2..4` Server Wavefront Batching**: Added pre-allocated `Ink2SlotState` and row-offset wavefront execution (`QWInkling2Forward+Wavefront`, `QWInkling2Runtime+Wavefront`, `QWInkling2Session`, `QWBatchScheduler`, `DriverBatchDecode`), scaling `QWServer` `Inkling-2` aggregate decode **`65.60 -> 77.36 tok/s` (`B=3`, `eq_all=True`)** and **`64.97 -> 100.56 tok/s` (`B=4`, `1.55×`)**, passing `Law 28` `S1..S9` across all 6 flagship server configurations (`0` server errors).

### Cumulative Changes Since GitHub Release v0.28.2 (`v0.29.0` – `v0.33.0`)

#### v0.33.0 — Native `Inkling-2-Mini` (`inkling2_loop`), Generalized NAX/MoE/Attention & `--mtp-drafts 0...15`
- Shipped native Metal 4 runtime (`QWInkling2*`), `ignore_merges` BPE support (`QWTokenizer+Codec`), 23-layer vision tower (`QWInkling2Vision`), Auto-DSpark Markov + tree verification + `B=16` PLD + variational speculation, and trailing-64 `QWSampler.loopRepetitionTokens` scoping.
- Generalized Apple10 NAX FP/BF16/NVFP4/MXFP8/SHVQ matmul & fused MoE SwiGLU (`QWGemmFP*`, `qwgemm_fp_moe_nax_swiglu.metal`), un-spilled `T<=16` compact SHVQ/HQ/AQ MoE (`QWGemmMoE+CompactQWP`), NAX Sink+RelBias FlashAttention & position-invariant Split-KV GQA decode (`QWAttnSinkRelApply`, `QWAttnSinkRel+Sparse`, `qwlsa_bf16.metal`), and batched `M=1..16` BF16 head argmax (`QWHead+BF16`).
- Expanded `--mtp-drafts` upper bound to `0...15` across CLI (`qwg`), `Qwegn.app` Menu Bar (`SettingsView+Actions`), `/ui` WebUI (`GET /v1/models/discovered` `maxSpeculativeDrafts` + `actions.js`), and speculative drivers while keeping `Qwen3.8-Flash-Next` at `3` and `DFlash 2` at `7`.
- Qualified native `QWP1` packs vs `thinkingmachines/Inkling-2-Mini`: **`Inkling-2-Mini-shvq4g32-shvq2g64-shvq3g64`** (`QWP-Fast`, `10.20 GiB`, `3,637.9 tok/s` at `8K` prefill, `72.68–99.89 tok/s` decode, `21/25` `code25`) and **`Inkling-2-Mini-shvq4g32-nvfp4`** (`QWP-Cand5`, `15.15 GiB`, `0.013170` mean KL / `97.92%` Top-1, `79.90 tok/s` `2K var5`, `13/13` HE), plus `15.20×` MLX prefill (`3,324.0 tok/s`) & `4.42×`/`2.45×` decode.

#### v0.32.0 – v0.32.2 — Fan Helper Daemon, WebUI Free-Scroll Streaming & Menu Bar Window/Hotkey Fixes
- **`v0.32.2` (WebUI Chat Streaming UX)**: Replaced forced bottom-pinning during SSE streaming with smart tail-following (`↓ Latest` pill, in-place DOM patching), made reasoning disclosures collapsed by default across streaming and Smoke Probe, and added rotating dimmed italic reasoning status quips.
- **`v0.32.1` (Menu Bar Single Settings Window & Hotkeys)**: Replaced SwiftUI `Settings` wrapper with a pure AppKit entrypoint in `QWMenuBarApp` (fixing duplicate blank settings windows on `⌘,`) and added `QWMenuBarShortcutRouter` for Caps Lock, Fn/NumPad, and non-Latin ANSI keyCode routing (`⌘,`, `⌘W`, `Esc`).
- **`v0.32.0` (`qwg-smc-helper` Privileged Fan Daemon)**: Added shared 80-byte `QWSMCHelper` + `/var/run/qwg-smc.sock` root LaunchDaemon (`LOCAL_PEERCRED` auth, `[1000, maxRPM]` clamp, `15 s` dead-man watchdog), `qwg fan-helper status|install|uninstall`, Menu Bar **Thermal & Cooling** controls, and `.pkg`/`.app`/uninstall packaging.

#### v0.31.0 – v0.31.1 — FA4-Inspired Metal 4 Attention & Zero-Jitter Menu Bar Telemetry
- **`v0.31.1`**: Switched `NSStatusItem` and popover numeric displays to fixed-advance SF Mono column widths (`QWMenuBarStatusStyle`, `QWMenuBarStatusFormatter`), eliminating horizontal icon/popover jitter.
- **`v0.31.0`**: Added 128-thread `BaseNAXFrag::mma_k32` MBSA selector, LPT prefill + 32-lane MBSA decode (`qwattn_mbsa_nax.metal`), wave-aligned `QWAttnDecode.splits`, `qwattn_decode_general` `sPk`/`sPv` hoisting, and 5 Split-K NAX QSA decode/verify kernels (`qwattn_sparse_nax.metal`), lifting `Qwen3.8-Flash-Next` decode **`+10.70%`**, `Maxwell-V1-Flash` decode **`+9.09%`** / prefill **`+2.67%`**, and `Qwen3.6-35B-A3B` / `Apex-35B-V1.0` decode **`+2.23%`** / **`+2.01%`**.

#### v0.29.0 – v0.30.0 — Single Geometry-Driven Engine Unification, Biases & Hybrid SWA Compaction
- **`v0.30.0`**: Unlocked `attention_bias`/`qkv_bias`/optional `o_proj.bias` (`P1`), added per-layer hybrid `sliding_window` with host-recycled physical KV page rings & compacted `.qwcache` (`P2`), and added 42 shared-prefix `B=2...8` general verify kernels (`qwattn_decode_general_shared.metal`, `_hd128` 8 KiB TG staging, `gqa <= 16`) (`P3`).
- **`v0.29.0`**: Consolidated `QWModelDescriptor.decode` (`geometryDigestV2`), replaced `StandardModule` + 10 redundant subclasses with geometry-derived `QWModelCapabilities`, promoted `QWAttn/QSA` & `QWHyperPLE`, absorbed `QWFlashNext` -> `QWHyperStreamRuntime` into `QWEngineDriver` (`0` family-name branches), and delivered first-class `.qwp` + MLX `.safetensors` parity.

## [0.33.0] - 2026-11-22

### Added & Optimized
- **Native `Inkling-2-Mini` (`inkling2_loop`) Looped-MoE Runtime, Generalized NAX/MoE/Attention Kernels, QWP1 Packs & `--mtp-drafts 0...15` (`feat/inkling2-loop`)**:
  - Shipped native Metal 4 runtime (`QWInkling2*`), `ignore_merges` BPE support (`QWTokenizer+Codec`), native vision tower (`QWInkling2Vision`), Auto-DSpark Markov + tree verify + `B=16` PLD + variational speculation, and trailing-64 `QWSampler.loopRepetitionTokens` scoping for `repetition_penalty`.
  - Generalized Apple10 NAX FP/BF16/NVFP4/MXFP8/SHVQ matmul & fused MoE SwiGLU (`QWGemmFP*`, `qwgemm_fp_moe_nax_swiglu.metal`), un-spilled `T<=16` compact SHVQ/HQ/AQ MoE (`QWGemmMoE+CompactQWP`), NAX Sink+RelBias FlashAttention & position-invariant Split-KV GQA decode (`QWAttnSinkRelApply`, `QWAttnSinkRel+Sparse`, `qwlsa_bf16.metal`), and batched `M=1..16` BF16 head argmax (`QWHead+BF16`).
  - Expanded `--mtp-drafts` upper bound to `0...15` across CLI (`qwg`), `Qwegn.app` Menu Bar (`SettingsView+Actions`), `/ui` WebUI (`GET /v1/models/discovered` `maxSpeculativeDrafts` + `actions.js`), and speculative drivers while keeping `Qwen3.8-Flash-Next` at `3` and `DFlash 2` at `7`.
  - Evaluated native QWP1 recipes vs unquantized `thinkingmachines/Inkling-2-Mini`: **`Inkling-2-Mini-QWP-Cand5`** (`cand5_hybrid_lossless_moe`, `15.15 GiB`, `0.013170` mean KL / `97.92%` Top-1, `79.90 tok/s` `2K var5`) and **`Inkling-2-Mini-QWP-Fast`** (`cand6_shvq4_shvq2d3`, `10.20 GiB`, `3637.9 tok/s` `8K` peak prefill, `72.68–99.89 tok/s` decode, `21/25` `llmprobe code25`), plus `15.20×` paired `2K` MLX prefill (`218.90 -> 3324.00 tok/s`) and `4.42×` `d=0` / `2.45×` `d=3` decode (`46.64`/`54.66 tok/s`) with `0` regressions across `Qwen3.8-27B`, `Qwen3.6-35B-A3B`, `SenseNova-Loop-9B`, and `GLM-5.3-Flash`.

## [0.32.2] - 2026-11-22

### Changed
- **WebUI Chat Free-Scroll Streaming, Collapsed-by-Default Reasoning & Amused Quips (`chat-presets.js`, `chat.js`, `markdown.js`, `controlbar.html`, `panels.css`, `chat.css`, `QWChatStreamingUXTests`)**:
  - Replaced forced bottom-pinning during SSE streaming with smart tail-following (`isTranscriptNearBottom`, `handleChatTranscriptWheel`, `handleChatTranscriptScroll`, `#chat-scroll-bottom-btn` floating `↓ Latest` pill) and in-place streaming turn DOM patching (`updateStreamingChatTurn`) so users can scroll up freely at any point while responses stream.
  - Made reasoning disclosures (`<details class="md-thinking">`) collapsed by default during streaming and Smoke Probe output (`markdown.js`), preserving explicit user open/close toggles (`handleChatThinkingToggle`) across chunks and completion.
  - Added rotating dimmed italic reasoning status quips (`CHAT_REASONING_QUIPS`, `.md-thinking-quip`, `@keyframes qwg-quip-pulse`) shown beside active `Reasoning (N chars)` summaries and automatically cleared once answer generation begins.

## [0.32.1] - 2026-11-22

### Fixed
- **Single Settings Window & Menu Bar Hotkey Routing (`⌘,`, `⌘W`, `Esc`)**: Replaced the SwiftUI `App` + `Settings { EmptyView() }` wrapper in `QWMenuBarApp` with a pure AppKit entrypoint retaining `AppDelegate`, eliminating the duplicate blank `"Qwegn Settings"` window triggered on `⌘,`. Added `QWMenuBarShortcutRouter` for Caps Lock, Fn/NumPad, and non-Latin ANSI `keyCode` fallback routing without hijacking Dvorak/AZERTY ASCII shortcuts or `TextField` editing (`⌘R`/`⌘D`/`⌘L` suppressed), repeat-key filtering, `NSApp.mainMenu` App/Edit/Window shortcuts (`NSMenuItemValidation`), popover activation (`NSApp.activate`), and `⌘W`/`Esc` dismissal.

## [0.32.0] - 2026-11-22

### Added
- **Privileged `qwg-smc-helper` LaunchDaemon & Non-Root Fan Control (`QWSMCHelper`, `qwg-smc-helper`, `QWFanController`, `qwg fan-helper`, `SettingsView+FanHelper`)**:
  - Extracted the 80-byte IOKit AppleSMC client into shared `QWSMCHelper` (`L0.5`, zero Metal/HTTP/ML deps) paired with minimal root daemon `qwg-smc-helper` (`L8.0`) listening on `/var/run/qwg-smc.sock` (`0660 root:staff`, `LOCAL_PEERCRED` peer UID verification, hardware `[minRPM, maxRPM]` clamping with `1000 RPM` safety floor, explicit `OK`/`ERR` replies, `15 s` dead-man auto-restore watchdog, and `SIGTERM`/`SIGINT` cleanup).
  - Upgraded `QWFanController` to verify `OK`/`ERR` replies over `/var/run/qwg-smc.sock` with a `2 s` `PING` cache so live helper activation works immediately without restarting `qwg`.
  - Added `qwg fan-helper status|install|uninstall` CLI subcommands, one-click `Enable Fan Control…` / `Remove` status controls inside **Settings → Performance → Thermal & Cooling** (`SMAppService.daemon` for Developer ID builds with admin authorization fallback for local/ad-hoc builds), and integrated bundle/pkg/uninstall packaging across `Scripts/build_app.sh`, `Scripts/build_pkg.sh`, `Scripts/uninstall.sh`, `Makefile`, and `.github/workflows/release.yml`.

## [0.31.1] - 2026-11-21

### Fixed
- **Menu Bar & Popover Zero-Jitter Alignment (`QWMenuBarStatusStyle`, `QWMenuBarStatusFormatter`, `QWMenuBarApp`, `TelemetryCardPrimitives`, `TelemetryCards`, `SettingsMenuBarAppearanceView`)**:
  - Switched `NSStatusItem` typography from proportional-punctuation `monospacedDigitSystemFont` to fixed-advance SF Mono (`QWMenuBarStatusStyle.font` with `.kern: 0` attributed string and state/title change deduplication) so every glyph, space, punctuation mark, and unit shares an identical advance width (`6.800 pt` at 11pt).
  - Locked `QWMenuBarStatusFormatter` fields to fixed column widths (`decodeFieldWidth = 5` for 3-digit `XXX.X`, `prefillFieldWidth = 6` for 4-digit `XXXX.X` matching `"↑from cache"` / `"↑  from cache"`, `memoryFieldWidth = 6` for `XXX.XX`, `activeContextFieldWidth = 5`, `temperatureDigitsWidth = 3` + `"°C"`) and retained fixed-width `—` placeholders for unavailable readings so menu bar icons never jump horizontally.
  - Column-locked popover hero rate units (`"999.9"` Decode / `"9999.9"` Prefill templates + constant `"From cache"` row height + `minWidth: 28` mode badge), right-aligned bounded numeric `inlineMetricCell` values, padded `prefillProgressBarRow` percentages (`%3d%%`), and added `.minimumScaleFactor(0.82)` on the Settings menu bar preview pill.

## [0.31.0] - 2026-11-21

What's changed since `v0.30.0`: delivers FA4-inspired Metal 4 attention optimizations (`perf/attn-fa4-w2`) across MBSA, Split-K QSA decode/verify, and wave-aligned Split-KV general decode — lifting `Qwen3.8-Flash-Next` decode **`+10.70%`**, `Maxwell-V1-Flash` decode **`+9.09%`** / prefill **`+2.67%`**, and `Qwen3.6-35B-A3B` / `Apex-35B-V1.0` decode **`+2.23%`** / **`+2.01%`** with zero regressions across all 8 flagship packs.

### Added & Optimized
- **MBSA NAX Selector, Compact 32-Lane Decode & LPT Prefill (`1fa9a21d`: `qwattn_mbsa_nax.metal`, `qwattn_mbsa.metal`, `QWMBSAAttnOperator`, `QWMBSAAttnOperatorTests`)**:
  - Upgraded `qwg_mbsa_select_blocks_tiled32` to 128-thread `BaseNAXFrag::mma_k32` + in-register row max (`#error` `< 260200` guard), added Longest-Processing-Time (LPT) tile scheduling and leader-lane `ktab`/`vtab` broadcast in NAX prefill, shrunk NAX decode to 32-lane `ELEMS=4` (`8 KiB` TG SRAM with dynamic `8`/`16 KiB` allocation), hoisted `sAbsKey`/`sPk`/`sPv` with 8-lane `mbsa_fwht32_h4`, and added wave-aligned `QWMBSAAttnOperator.targetSplits`.
- **Wave-Aligned `QWAttnDecode` Split-KV & `qwattn_decode_general` Hoisting (`05073baa`, `0a3f5ffd`: `QWAttnDecodeApply`, `qwattn_decode_general.metal`, `QWHybridLayerKernels`, `QWMTPBatchedTrunk`, `QWMTPDraftEncoder`, `QWEchoDraftEncoder`)**:
  - Added wave-aligned `QWAttnDecode.splits(forKeys:kvHeads:gpuCores:tokensPerSplit:)` preserving the `>= 8` short-context floor (`keys <= 512`) and synced across main, hybrid, and MTP/Echo draft callers; hoisted TG `sPk`/`sPv` multipliers, added `if (Mb > m && m > -1e29f)` online-softmax rescaling, and fused single-exp combine scaling in `qwattn_decode_general.metal`.
- **Split-K NAX QSA Decode & Verify Kernels (`d925b5d8`: `qwattn_sparse_nax.metal`, `QWKernelInventory`, `Scripts/kernel_inventory.json`, `QWQSACacheState`, `QWQSAEncoders`, `FNGPULayer`, `FNGPULayer+Batch`, `FNMetalSparseAttnTests`)**:
  - Added 5 Metal 4 Split-K NAX QSA kernels (`qwg_attn_sparse_gated_nax_split_{f32,f16,q8,q4}` and `qwg_attn_sparse_gated_nax_combine`, keeping prefill on exact `master` `xchg[2][256]` SRAM) so `T <= 8` decode and verify execute on Split-K NAX `matmul2d` instead of low-occupancy scalar `gqa3`.

### Rejected & Rolled Back
- **Noise-Floor & Regressive Attention Prototypes (Wave 1 `perf/attn-fa4-opt` + Wave 2 `qwattn_paged_nax.metal`, Laws 9 & 18)**:
  - Rolled back Wave 1 `xchg[2][2][512]` sparse prefill double-buffering (`-0.65%` on `Qwen3.8-Flash-Next`), `QWMLAAttnOperator` `recommendedSplits` (decode slowdown + sporadic `sample[1]` divergence), noise-floor HCA/LSA tweaks, and Wave 2 `qwattn_paged_nax.metal` (`0.9992x` flat on `Qwen3.6-35B-A3B`, `-1.63%` prefill dip on `Qwen3.8-27B`).

### Verified
- **Unit, Metal, Style & CI Gates (`v0.31.0`)**: `make lint` (`0` style/geometry/DAG/kernel violations across `28` targets, `1,234` kernels in metallib / `317` dispatched from Swift), `Scripts/check_metallib_target.py` (`PASS`, `kernel.metallib: macOS 26.2`, `kernel_std.metallib: macOS 26.2`, `26.0` guard enforced), `Scripts/check_basis.py` (`805/1,193` `simd_shuffle_xor` dropped, `65` identical), `swift build -c release --product qwg -Xswiftc -warnings-as-errors` (`0` warnings), `make ci` (`ci ok`, `102` `qwg-bench-kernels` entries, `14/14` Python harness tests `OK`), and `make test` (`1,732` executed: `1,680` XCTest across `23` bundles + `52` Swift Testing in `1` suite, `25` skipped, `0` failed).
- **Paired Interleaved 5-Pair A/B vs `master` (`f0f8b3d8`, All 8 Flagship Packs, `0` steady allocs, `6/6` candidate determinism)**:
  - `Maxwell-V1-Flash-HQ8-SHVQ3D4`: prefill `4,515.2->4,635.9` (**`+2.67%`**, `1.0267x` MAD `0.0078`), decode `47.64->51.85` (**`+9.09%`**, `1.0909x` MAD `0.0041`), TTFT `1.35->1.32s`, RSS `17.61 GiB` (`4f66b8b3` `6/6`).
  - `Qwen3.8-Flash-Next-hq8g32-hq4g64`: prefill `2,862.3->2,840.4` (`0.9932x` MAD `0.0050`), decode `49.91->55.25` (**`+10.70%`**, `1.1070x` MAD `0.0122`), TTFT `2.13->2.15s`, RSS `10.79 GiB` (`a5d7bd95` `6/6`).
  - `Qwen3.6-35B-A3B-hq4g32`: prefill `6,529.3->6,522.5` (`0.9967x` MAD `0.0034`), decode `123.08->125.53` (**`+2.23%`**, `1.0223x` MAD `0.0031`), RSS `21.28 GiB` (`2e7e71a9` `6/6`).
  - `Apex-35B-V1.0-hq4g32`: `0`-ULP `aaf1eb81` (`6/6`), prefill `6,667.3->6,528.0` (`0.9957x`), decode `122.63->127.18` (**`+2.01%`**, `1.0201x` MAD `0.0206`), RSS `21.28 GiB`.
  - `Qwen3.8-27B-hq4g32` (`--strict`): `0`-ULP `a4b3e8bf` (`6/6`), prefill `707.4->841.7` (`1.0102x`), decode `28.52->28.85` (`1.0055x`), RSS `17.03–17.09 GiB`.
  - `SenseNova-Loop-9B-hq4g32` (`--strict`): `0`-ULP `a05559f1` (`6/6`), prefill `1,833.1->1,863.1` (`0.9822x`), decode `47.59->47.33` (`0.9921x`), RSS `6.13–6.14 GiB`.
  - `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`--strict`): `0`-ULP `f5b823a7` (`6/6`), prefill `627.5->609.0` (`1.0102x`), decode `37.01->37.15` (`0.9981x`), RSS `101.08 GiB`.
  - `Sage-S2.1-Flash-hq8g32-shvq2g64-shvq3g64` (`--strict`): `0`-ULP `6d193415` (`6/6`), prefill `795.4->800.6` (`0.9984x`), decode `31.60->32.08` (`1.0097x`), RSS `105.32 GiB`.
- **Law 11 Written Rulings**: Expected 1-ULP deterministic reduction-order hash changes vs `f0f8b3d8` on the 3 sped-up decode paths (`Maxwell-V1-Flash` `cc4e5721->4f66b8b3` from Split-K `32->20`, 32-lane `ELEMS=4` `simd_sum`, and NAX `tiled32` selector; `Qwen3.8-Flash-Next` `2b5414d5->a5d7bd95` from upgrading decode `T <= 8` from scalar FP32 `gqa3` to Metal 4 Split-K NAX FP16 `matmul2d`; `Qwen3.6-35B-A3B` `8e5fd521->2e7e71a9` from wave-aligned `splits` across Split-KV reduction); all 3 are `6/6` deterministic across paired runs and pass 16K–21K needle recall, `llmprobe`, and Law 28 QA.
- **Residency Toggles, 16K–21K Needles, `llmprobe` & Live `QWServer` Law 28 QA**: `Flash-Next` `--stream-ngrams` `-27.13 GiB` weights (`98.42->71.29 GiB`, `SHA-256` `135069c1da7c32ba` exact); `Qwen3.6-35B-A3B` `--no-load-vision` vs `--load-vision` (`71aaa7aeda95aa6b` exact); `4/4` 16K–21K needle exact recall (`QWEGN-FA4W2-83917` on `Maxwell` 17.2K, `Flash-Next` 21.0K, `35B-A3B` 21.0K, `Apex-35B` 21.0K); `4/4` representative packs `llmprobe` (`89/89` `100%` conformance + `9/10` `code10`: `5/5` HE + `4/5` MBPP, `0` errors) and `36/36` Law 28 live `QWServer` QA PASS (`S1`–`S9` × `Maxwell`, `Flash-Next`, `35B-A3B`, `27B`; `0` server errors).

## [0.30.0] - 2026-11-21

What's changed since `v0.29.0`: completes post-unification geometry & kernel generalization (`feat/general-kernels-and-swa`) — adding `attention_bias`/`qkv_bias`/optional `o_proj.bias` (`P1`), per-layer hybrid `sliding_window` with host-recycled physical page rings and compacted `PrefixCache`/`.qwcache` (`P2`), and general shared-prefix batched verification `B=2...8` with `_hd128` compact TG staging and `gqa <= 16` (`P3`), while rolling back unpaged general NAX prefill (`P4`) after paired A/B testing showed no speedup over `qwattn_paged_nax` (Laws 9 & 18).

### Added & Generalized
- **Attention QKV & Optional Output Biases (`P1`: `QWRopeApply`, `qwrope_general.metal`, `QWNorm`, `qwnorm.metal`, `QWModelDescriptor`, `QWHybridLayerKernels`, `QWMTPBatchedTrunk`, `MTPVerification`, `e5b86ee8`)**:
  - Unlocked `attention_bias`, `qkv_bias`, and optional `o_proj.bias` across prefill, decode, `B=2...4` batched decode, and MTP paths with fail-closed loader/allocation validation and `geometryDigestV2` preserved across all 12 `Family` cases.
  - Fused `qkv_bias` addition inside `qwrope_general.metal` and added `qwg_add_bias_f16` in `qwnorm.metal` for optional `o_proj.bias`.
- **Per-Layer Hybrid `sliding_window` & Host-Recycled SWA KV Ring (`P2`: `QWKVCache`, `QWKVLayerPair`, `QWAdmission`, `QWScratchArenaBuilder`, `PrefixCache`, `QWCacheDiskManager+IO`, `9ee96ecb`, `c590f9d5`)**:
  - Added per-layer `layerSlidingWindows` (`"sliding_attention"`, `max_window_layers`, `sliding_window_pattern`) across `f16`/`q8`/`q4` with `recycleExpiredWindowPages` physical page-ring reuse in `QWKVCache`, SWA admission/scratch budgeting, and compacted live-window `PrefixCache`/`.qwcache` serialization.
- **General Shared-Prefix Verify `B=2...8`, Compact `_hd128` TG Staging & `gqa <= 16` (`P3`: `qwattn_decode_general_shared.metal`, `qwattn_decode_general.metal`, `qwattn_decode.metal`, `QWAttnDecodeApply`, `d8d399ab`)**:
  - Added 42 general shared-prefix verify kernels (`B=2...8` × `f16`/`q8`/`q4` × `STAGE_DIM=128/256`, optional `slidingWindow`), added `_hd128` (`8 KiB` TG staging) decode/shared variants, raised general GQA ceiling from `<= 8` to `<= 16` (`keysPerPhase = min(16, 2 * gqa)`), and restored frozen `qwattn_decode.metal` to pure 27B.

### Rejected & Rolled Back
- **General Short-Prompt Unpaged NAX Prefill (`P4`: `1887b4f3` rolled back in `049e3f09`, Laws 9 & 18)**:
  - Paired interleaved A/B evaluation across 24 short-prompt runs (`Suite 1A` `--kv-mode f16` strict + `Suite 1B` `--kv-mode q4` on `Maxwell-V1-Flash`, `Qwen3.6-35B-A3B`, `SenseNova-Loop-9B`, `Qwen3.8-27B` at `t32`/`t64`/`t115`) showed `-1.0%` to `+0.97%` GPU prefill delta (noise floor / `-0.9%` on `Maxwell` `t115`), `0 ms` TTFT improvement, and `q4` hash drift vs `master`. Cleanly rolled back `qwattn_nax_general.metal` and `useUnpagedNAX` routing, restoring `100%` short-prompt strict `--kv-mode q4` parity and keeping `qwattn_nax.metal`, `QWAttnApply.swift`, and `EncodeKernels.swift` unchanged from `v0.29.0`.

### Verified
- **Unit, Metal, Style & CI Gates (`v0.30.0`)**: `make lint` (`0` style/geometry/DAG/kernel violations across `28` targets, `1,229` kernels in metallib / `312` dispatched from Swift), `Scripts/check_metallib_target.py` (`PASS`, `kernel.metallib: macOS 26.2`, `kernel_std.metallib: macOS 26.2`, `26.0` guard enforced), `Scripts/check_basis.py` (`794/1,151` `simd_shuffle_xor` dropped, `61` identical), `swift build --build-tests -Xswiftc -warnings-as-errors` (`0` warnings), `make ci` (`ci ok`, `101` `qwg-bench-kernels` entries, `14/14` Python harness tests `OK`), and `swift test --no-parallel` (`1,726` executed: `1,674` XCTest across `23` bundles + `52` Swift Testing, `25` skipped, `0` failed).
- **Paired Interleaved A/B vs `master` (`29afed25`), Residency Toggles & 16K Needles (All 8 Flagship Packs, `0` steady allocs)**:
  - Short-prompt strict `--kv-mode q4` (`post-P4-revert`, `2/2` `SHA-256` identical): `Maxwell-V1-Flash` (`702.0->696.3` wall / `719.1->717.8` GPU pre, `100.16->99.96` dec), `Qwen3.6-35B-A3B` (`826.4->833.7` wall / `863.2->864.5` GPU pre, `147.54->147.11` dec).
  - Speculative (`7/7` paired): `Qwen3.6-35B-A3B` `MTP-3` (`4,346.4->4,325.9` / `177.07->177.98`, exact), `Apex-35B-V1.0` `MTP-3` (`4,411.6->4,412.6` / `161.59->162.29`, exact), `Maxwell-V1-Flash` `ngram-4` (`3,639.2->3,646.3` / `86.04->86.26`, exact), `Qwen3.8-27B` `DFlash2` (`990.6->976.8` / `86.67->86.70`, exact) & `DSpine-b16` (`1,035.4->1,034.5` / `68.96->69.65`, exact), `Qwen3.8-Flash-Next` `DFlash2` (`1,917.4->1,933.3` / `86.24->87.41`, exact), `GLM-5.3-Flash` `DFlash2` (`538.6->536.6` / `27.67->28.92`, shared FP near-tie hash `2ddbce83d649`).
  - Serial strict (`9/9` across `8/8` flagship packs, `0` regressions): `SenseNova-Loop-9B` (`2,020.8->2,022.0` / `52.37->52.12`, exact), `Maxwell-V1-Flash` (`3,671.9->3,658.9` / `88.80->88.80`, exact), `Qwen3.8-27B` (`1,045.0->1,039.1` / `31.68->31.71`, exact), `Qwen3.6-35B-A3B` (`4,358.2->4,363.3` / `140.81->139.29`, exact), `Apex-35B-V1.0` (`4,416.9->4,414.0` / `142.59->140.84`, exact), `Qwen3.8-Flash-Next` resident (`1,899.3->1,894.7` / `54.87->54.97`, exact) & streamed (`1,937.3->1,936.4` / `56.50->56.51`, exact), `GLM-5.3-Flash` (`687.7->720.8` / `42.74->43.72`, exact), `Sage-S2.1-Flash` (`573.9->571.8` / `32.45->32.51`, non-strict exact after known FP near-tie `{a0399f5425da, c491fd3d4205}`).
  - Residency & 16K needles: `Flash-Next` `--stream-ngrams` `-27.13 GiB` (`98.42->71.29 GiB`, `SHA-256` exact); live `QWServer --no-lazy` `--no-load-vision` vs `--load-vision` (`SenseNova` `+0.849 GiB` RSS, `27B` `+1.043 GiB`, `35B-A3B` `+0.990 GiB`, `Apex` `+0.997 GiB`); `8/8` 16K needles recovered `QWEGN-UNIFY-74921` (`base==cand` `SHA-256` exact).
- **Live `QWServer` Law 28 QA & `llmprobe` (`v0.30.0`)**: `54/54` QA scenarios (`S1`–`S9` × `6` flagship families: `Flash-Next`+DF2, `27B`+DF2, `SenseNova-Loop-9B`, `Maxwell-V1-Flash`, `35B-A3B`, `GLM-5.3`+DF2) PASS (`0` server errors); `llmprobe --full` on `Flash-Next` (`291/291` `100%` conf, `100%` cap `strong`, `74/88` `84.1%` ag, `100%` fid, `17/20` `code20`) + `7/7` packs `89/89` `100%` conf (`9/10` `code10` on `27B`/`Maxwell`/`35B-A3B`/`GLM-5.3`/`Apex`, `8/10` on `SenseNova`/`Sage`, `0` errors).

## [0.29.0] - 2026-11-21

What's changed since `v0.28.2`: completes the single geometry-driven engine unification — retiring `QWFlashNext`, `FlashNextModule`, `FNGPUEngine`, `QWEngineCore`, `StandardModule`, and 10 redundant family module subclasses so all `.qwp` and standard HF/MLX `.safetensors` models execute behind a single `QWEngineDriver` / `DenseModule` / `QWModelCapabilities` architecture.

### Architecture & Unification
- **Unified Trait-Based Config Decoder, Sliding-Window GQA & Geometry-Derived Capabilities (`QWModelDescriptor+Decode`, `QWModelCapabilities`, `DenseModule`, `qwattn_{decode,paged}_general.metal`, `QWServerSettings`, `07657e45`, `9dcb8071`, `b1450412`)**:
  - Consolidated `QWStandardConfigDecoder` into trait-based `QWModelDescriptor.decode` (`geometryDigestV2` pinned across all 12 `Family` cases); unlocked `tie_word_embeddings`, `rope_scaling` (`yarn`/`linear`/`dynamic`), `shared_expert_intermediate_size`, `first_k_dense_replace`/`mlp_only_layers`/`decoder_sparse_step`, `sliding_window` (`UInt32` mask across `f16`/`q8`/`q4`), and trait-based `hasQKNorm`.
  - Replaced `StandardModule` and 10 redundant family subclasses with geometry-derived `QWModelCapabilities.resolve(descriptor:weightFormat:hasVisionWeights:)` (`usesMultiStreamHC = (descriptor.hcCount ?? 0) > 1`), deleted dead `QWEngineCore`/`QWEngineSessionPool`/`LoopToyModule`, and removed family-name branching in `QWServer`.
- **Native `.qwp` & Standard MLX `.safetensors` First-Class Parity (`DriverWeightKernels`, `DriverWeightLoading`, `QWMTPDraftEncoder`, `QWCheckpointTensorLoader`, `c8103090`, `b5683f49`, `7ff0ef90`)**:
  - Added `QWEngineDriver.resolveHeadSpec` supporting tied embeddings for both per-tensor quantized MLX affine (`AQ{b}_G{g}` via `QWAQRepacker.layout`) and BF16/F16 `.safetensors` heads; fixed `WeightLoader.isMLXCheckpoint` for BF16 `embed_tokens`, added missing `memoryBarrier` before `greedySelector.encode` when `q4View == nil` in `QWMTPDraftEncoder`, and lifted the stale `DriverConfig.familyDeclineReason` trap on `.nativeAffine` MTP.
  - Replaced legacy `is27BShape` / `is27BKVShape` predicates with geometry predicates (`matchesHead256GQA6Kernel` / `matchesKVStride1024Kernel`), generalized `sharedPrefix` `B=2...8` general GQA (`combine_batched_general`), hardened `stackedExpertBanks`, and cached `hasVisionWeights`.
- **Layer-4 Operator Promotion & `QWFlashNext` Absorption into `QWEngineDriver` (`QWHyperPLE`, `QWAttn/QSA`, `QWHyperStreamRuntime`, `QWKernelInventory`, `eaeb7d83`, `80aeec06`, `4511a6d4`, `1cf03e03`)**:
  - Promoted `QWQSAEncoders` + `QWQSACacheState` into `QWAttn/QSA` and created Layer-4.0 `QWHyperPLE` (`28`-target acyclic DAG, `0` violations).
  - Absorbed `FNGPUEngine` -> `QWHyperStreamRuntime` (`FNPrefixHeader.version = 2`) directly into `QWEngineDriver` (`QWDriverStandardState` allocates only hyper-stream runtime when `(descriptor.hcCount ?? 0) > 1`), deleted `Sources/Subsystems/QWFlashNext/` and `FlashNextModule`, added `QWQSALayerKernels`, and unified `QWKernelInventory` (`common` + `dense` + `flash`, `330` doors).

### Verified
- **Unit, Metal, Style & CI Gates (`8f7753ee` + final `v0.29.0`)**: `make lint` (`0` style/geometry/DAG/kernel violations across `28` targets, `1,183` kernels / `308` dispatches), `check_metallib_target.py` (`macOS 26.2`, `26.0` guard), `check_basis.py` (`788/1,141` `simd_shuffle_xor` dropped, `60` identical), `swift build --build-tests -Xswiftc -warnings-as-errors` (`0` warnings), `make ci` (`ci ok`), and `swift test --no-parallel` (`1,704` executed: `1,652` XCTest across `23` bundles + `52` Swift Testing, `25` skipped, `0` failed).
- **Paired Interleaved A/B vs `master` (`7c37478a`), Residency Toggles & 16K Needles (All 8 Flagship Packs, `0` steady allocs)**:
  - Serial strict (`8/8` packs, `0` regressions): `SenseNova-Loop-9B` (`2,023.1->2,022.4` / `52.62->52.51`, `SHA-256` exact), `Maxwell-V1-Flash` (`3,652.1->3,660.5` / `89.86->89.49`, exact), `Qwen3.8-27B` (`944.1->952.4` / `31.34->31.73`, exact), `Qwen3.6-35B-A3B` (`4,376.7->4,362.8` / `142.25->142.06`, exact), `Apex-35B-V1.0` (`4,444.1->4,437.9` / `143.39->143.38`, exact), `Qwen3.8-Flash-Next` resident (`1,901.8->1,886.3` / `55.07->55.06`, exact) & streamed (`1,941.2->1,941.5` / `56.62->56.79`, exact), `GLM-5.3-Flash` (`722.6->722.9` / `45.94->45.98`, exact), `Sage-S2.1-Flash` (`714.8->712.7` / `33.99->34.07`, `4/4` `task-745` standalone exact `a0399f5425da` + `5/6` paired w/ pre-existing FP near-tie).
  - Speculative (`5/5` paired): `Apex` `MTP-3` (`158.78->161.38`, exact), `27B` `DFlash2` (`75.19->76.70`, exact), `27B` `DSpine` (`72.37->73.56`, `5/6` exact `3df4955007d1` ⊆ `master` `{3df4,1a4c}`), `Flash-Next` `DFlash2` (`87.61->87.85`, exact), `GLM-5.3` `DFlash2` (`30.10->32.70`, pre-existing multi-hash tie sharing `3` exact hashes).
  - Residency toggles & 16K needles: `Flash-Next` `--stream-ngrams` `-27.13 GiB` (`98.42->71.29 GiB`, `SHA-256` exact); live `QWServer` `--no-load-vision` vs `--load-vision` (`SenseNova` `+0.850 GiB` RSS / `+870 MB` footprint, `27B` `+1.012 GiB`, `35B-A3B` `+0.993 GiB`, `Apex` `+0.988 GiB`); `8/8` 16K needles recovered `QWEGN-UNIFY-74921` (`base==cand` `SHA-256` exact).
- **Live `QWServer` Law 28 QA & `llmprobe` (`v0.29.0`)**: `45/45` QA scenarios (`S1`–`S9` × 5 flagship families: `Flash-Next`+DF2, `27B`+DF2, `SenseNova-Loop-9B`, `GLM-5.3`+DF2, `35B-A3B`) PASS (`0` server errors); `llmprobe --full` on `Flash-Next` (`291/291` `100%` conf, `100%` cap `strong`, `74/88` `84.1%` ag, `100%` fid, `17/20` `code20`) + `8/8` packs `100%` conformance (`89/89` quick, `8–9/10` `code10`, `0` errors).

## [0.28.2] - 2026-11-20

What's changed since `v0.28.1`: fixes the GitHub Actions basis-gate failure in LSA sparse paged attention kernels by enforcing `METAL_FUNC` (`inline __attribute__((__always_inline__))`) on basis-dependent Metal helpers.

### Fixed
- **Compile-Time Basis Inlining in LSA Sparse Kernels (`qw_attn_sparse_prefill_i8.metalh`, `qwattn_paged_sparse.metal`, `qw_attn_sparse_decode.metalh`, `e16adf03`)**:
  - Replaced `static inline` with `METAL_FUNC` on `lsa_sparse_nax_epilogue`, `lsa_prefill_paged_nax_sparse_body`, and `lsa_decode_sparse_core`.
  - Prevents Metal's LLVM frontend from emitting an out-of-line `lsa_sparse_nax_epilogue` call whose dead `uint sn` parameter under `-DQW_BASIS_STANDARD=1` triggered dead-argument elimination IR divergence across all 6 `qwg_attn_prefill_paged_nax_sparse_pair*` kernels (`Scripts/check_basis.py`), while eliminating the out-of-line `Otile` pointer call boundary in AIR.

### Verified
- **Unit, Basis & CI Gates**: `python3 Scripts/check_basis.py` (`788` of `1,141` `simd_shuffle_xor` instructions dropped in standard build, `60` basis-independent functions identical), `Scripts/check_metallib_target.py` (`macOS 26.2` on both metallibs, `26.0` guard enforced), `make lint` (`0` style/geometry/DAG/kernel violations, `1,183` kernels / `308` dispatches), `swift build --build-tests -Xswiftc -warnings-as-errors` (`0` warnings/errors), `swift test -Xswiftc -warnings-as-errors --no-parallel` (`1,711` executed: `1,659` XCTest across `23` bundles + `52` Swift Testing in `1` suite, `25` skipped, `0` failures), and full `make ci` (`ci ok`, `14/14` Python harness tests, `101` `qwg-bench-kernels` entries).

## [0.28.1] - 2026-11-20

What's changed since `v0.28.0`: ships end-to-end Latent Sparse Attention (LSA) prefill/selector kernel optimizations, out-of-place batched GDN speculative verification with GPU suffix-stability detection, and DFlash 2 / DSpine sliding-window prefill capture skipping across dense and hybrid MoE architectures and all KV modes (`f16`/`q8`/`q4`).

### Performance
- **DFlash 2 / DSpine Sliding-Window Prefill Capture Skipping (`QWDFlashConfig`, `FNGPUDFlash`, `QWLayerKernel`, `QWDenseLayerKernels`, `MTPSeeding`, `DriverPrefill`, `DenseModule`, `471f2f53`)**:
  - Skips dead drafter hidden-state captures and context KV updates outside the drafter sliding window during prefill chunks (`-16.3 ms/chunk` non-trunk overhead on `Qwen3.8-27B`, `+1.06%` at `32K q4` with `100%` SHA-256 identity).
- **Out-of-Place Batched GDN Verify & GPU `unstableSuffixMask` (`qwgdn_decode.metal`, `QWGDNDecode`, `QWGDNReplayCheckpoint`, `QWVerifyWindow`, `MTPVerification`, `291c1d6d`)**:
  - Eliminated the `~6.2 MB` synchronous CPU `rollbackTapeIsStable()` scan via a GPU `-ffast-math`-safe `unstableSuffixMask` and replaced unconditional `144 MiB` GDN `encodeBackup` blits with out-of-place state verification (`+2.78%` med / `+1.85%` max decode at `2K q4`, `+1.79%` med / `+2.06%` max at `16K q4`).
- **Parallelized Ward Rescan & Vectorized LSA Index Projection (`qwlsa.metal`, `15c4ff69`)**:
  - Parallelized `qwg_lsa_ward_cluster_512` full rescan across all 128 threads (`1.973 -> 0.344 ms` / `5.74×`, `-27.9 ms/2K chunk`) with bit-exact cluster hashes and vectorized `qwg_lsa_index_prep` with contiguous `half4` loads (`2.24×–3.86×` speedup).
- **INT8 NAX Q×K Path & Warp-Broadcast Lookup for Sparse Paged Prefill (`qw_attn_sparse_prefill_i8.metalh`, `qwattn_paged_sparse.metal`, `c33a0a10`)**:
  - Generalized dense `i8_body` (`kvblk` aliasing `kvblk_i8 + kScale` with zero extra TG SRAM and zero-barrier `simd_shuffle` page lookup) to sparse paged prefill across `hd256` and `hd128` (`q8`/`q4`), yielding `1.72×–1.76×` sparse prefill kernel speedups (`2.535/2.515/2.506 -> 1.440/1.448/1.458 ms`) and `< 2e-3` parity vs the scalar general kernel.
- **Vectorized LSA Block Selector Dot Products & SRAM Score Cache (`qwlsa_select.metal`, `1d051005`)**:
  - Added `half4` IEEE-754-identical `lsa_dot_q_half`, removed redundant device fences in `lsa_select_blocks_body`, and added `tgScores[2304]` SRAM score caching (`1.21×–1.30×` `decode_select` and `1.47×–2.24×` `prefill_select_512` across `64K–256K` with bit-exact hashes).

### Verified
- **Unit & Lint Gates**: `swift build -c release -Xswiftc -warnings-as-errors --product qwg` (`0` warnings/errors), `Scripts/check_metallib_target.py` (`macOS 26.2` on rotated & standard metallibs), `make lint` (`0` style/geometry/DAG/kernel violations, `1,183` kernels / `308` dispatches), and `swift test -Xswiftc -warnings-as-errors --no-parallel` (`3,318` executed across `23` bundles, `50` skipped, `0` failed).
- **Paired A/B, Parity, Vision Residency & Law 28 `QWServer` QA**:
  - Paired 3-round A/B vs `v0.28.0` (`44d07982`, `0` steady allocs): `Qwen3.8-27B` `DFlash2 w=8` prefill `+0.50%..+9.52%` across `2K..32K` (`32K q4`: `759.7 -> 783.6 tok/s`, `+3.15%`, decode `40.96 -> 41.48 tok/s`; `16K q4` cool-GPU: `752.4 -> 818.7 tok/s`, `+8.81%`, decode `40.92 -> 48.19 tok/s`), `DSpine-b16` `16K q4` (`+4.19%` prefill / `+3.08%` dec med), `Qwen3.6-35B-A3B` HybridMoE (`+4.27%` at `2K q4`, `+11.86%` prefill / `+3.72%` dec med at `16K q4`).
  - `10/10` Candidate-Serial ≡ Candidate-DFlash2/DSpine greedy SHA-256 identity (`2K/16K/32K × f16/q8/q4` + `DSpine 16K/q4`); `--no-load-vision` vs `--load-vision` saved `1.01 GiB` (`27B`) / `0.99 GiB` (`35B-MoE`) RSS; `18/18` Law 28 `QWServer` QA scenarios passed (`9/9` on `Qwen3.8-27B`+DFlash2 and `Qwen3.6-35B-A3B`, `0` errors). `llmprobe` was not run (`llmprobe` binary not installed on host).

## [0.28.0] - 2026-11-20

What's changed since `v0.27.4`: ships Wave 5 speculative pipeline and hot-path allocation fixes plus Wave 6 HAL Stage 1 lock-free residency, Native Metal FP16 ViT + FlashAttention (`4.09×–4.12×` faster vision encode, `-45%` active vision weights), `1.66×–1.90×` zero-allocation pure-ASCII/BPE tokenizer acceleration, deterministic JSON key-order preservation for server prefix caching, and dead-code/kernel hygiene.

### Added & Performance
- **Native Metal FP16 ViT, `O(N)` FlashAttention & `vDSP`/`vImage` Patch Slab (`QWVision`, `QWVisionMetalBackend`, `QWMM`, `QWServer`, `Sources/QWMetal`)**:
  - Replaced CPU/AMX-only vision encoding with basis-independent FP16 Metal ViT kernels + `O(N)` FlashAttention and `vDSP`/`vImage` patch slab preprocessing wired into `QWVisionOnDemandBackend` (`--load-vision` / `--no-load-vision` residency toggles honored).
  - Cut vision encode latency `4.09×` on `SenseNova-Loop-9B-hq4g32` (`3,304.7 -> 807.8 ms` vision encode; `7.19 -> 4.61 s` paired CLI wall; `996.7 MB` / `950.6 MiB` vs `1.82 GB` resident vision weights with `100%` SHA-256 output parity) and `4.12×` on `Qwen3.8-27B-hq4g32` (`3,348.6 -> 812.2 ms`; `9.45 -> 6.94 s` paired wall; `1.01 GB` / `961.6 MiB` vs `1.84 GB`, `0.8856` text similarity). `--no-load-vision --image` cleanly exits `2`.
- **`QWTokenizer` Pure-ASCII Skip, Zero-Alloc BPE & Bounded Word Cache (`QWToken`)**:
  - Added ASCII fast-path scanning, zero-allocation BPE scratch buffers, and a bounded word cache in `QWTokenizer`, delivering `1.88×–1.90×` speedup at `2K` (`0.69 -> 0.36 ms`) and `1.66×–1.67×` at `32K` (`10.54–10.62 -> 6.35–6.36 ms`) with `4/4` bit-exact token-ID identity across `SenseNova-Loop-9B` and `Qwen3.8-27B`.
- **HAL Lock-Free Residency, Unretained CBs & Arena Hazard Policy (Stage 1: `QWHAL`, `QWAlloc`)**:
  - Added `QWResidencyRegistry` lock-free snapshot adoption, unretained command buffer handling, and `.safeDefault` arena hazard policy, covered by `QWWave6HALResidencyTests` and `QWWave6ArenaHazardTests`.
- **Wave 5 Speculative Sync & Hot-Path Scratch Reuse (`QWMTP`, `QWHead`, `QWEngine`, `QWFlashNext`)**:
  - Eliminated `DFlash`/`NextN` sync bubbles, redundant GDN rollback scans/copies, and 3-CB decode splits, and introduced `O(1)` N-gram LRU eviction and zero-allocation constraint, sampler, and recycler scratch buffers.

### Fixed
- **Ordered JSON Root Key Preservation Across HTTP Decoding (`QWCore`, `QWServer`, `QWServerRequestDecoder`)**:
  - Added `QWJson.orderedRootUserInfoKey` and routed HTTP JSON request decoding through ordered root decoding across endpoints so prefix-cache prompts remain deterministic regardless of client key serialization order (`a313027a`).
- **Fused-3D Cold-Expert Slicing & Bounds Safety (`01f09cf0`)**:
  - Fixed fused-3D cold-expert plane slicing offsets and bounds checks during pack loading.

### Removed & Cleaned
- **Engine & Kernel Dead-Code Purge (`033a6a5d`, `f3076c25`)**:
  - Removed unused/MLX/bench-only Metal kernels (`1,183` live production kernels / `308` dispatches) and dead/test-only Swift symbols, deduplicated admission/HAL/greedy helpers, moved `m1` soak helpers out of production sources, and tightened `Scripts/lint_kernels.py` reachability enforcement.

### Verified
- **Unit & CI Gates**: `Scripts/ci.sh` (`ci ok`, `check_metallib_target.py` `macOS 26.2`, `14/14` paired_perf Python tests, `103` `qwg-bench-kernels` entries), `swift test --no-parallel` (`1,708` executed: `1,656` XCTest + `52` Swift Testing, `25` skipped, `0` failed, `0` compiler warnings under `-Xswiftc -warnings-as-errors`).
- **Paired Quiet-GPU A/B, Law 28 QA & `llmprobe`**: `10/10` Stage 2 paired A/B checks (`0` steady allocs, bit-exact text SHA-256 across all text runs, `-27.13 GiB` `--stream-ngrams`), `36/36` Law 28 `QWServer` QA scenarios (`S1–S9` across `SenseNova-Loop-9B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B + DFlash2`, `Qwen3.8-Flash-Next`, `0` server errors), and `8/8` model `llmprobe --full` + `Code-100` runs (`17` reports archived in `out/llmprobe/`).

## [0.27.4] - 2026-11-19

What's changed since `v0.27.3`: fixed silent Metal 4 NAX `matmul2d` cooperative-tensor prefill corruption in macOS 26 release builds by pinning `-mmacosx-version-min=26.2`, added compile-time and CI metallib deployment-target guards, and landed Wave 1–4 security, memory-safety, kernel, concurrency, and architecture hardening across the runtime and server.

### Fixed
- **Metal 4 NAX Deployment Target (`-mmacosx-version-min=26.2`) & Compile-Time/CI Target Guards (`Scripts/build_metallib.sh`, `Scripts/check_basis.py`, `Scripts/check_toolchain.sh`, `Scripts/check_metallib_target.py`, `Scripts/ci.sh`, `.github/workflows/release.yml`, `nax.h`, `qwattn_sparse_nax.metal`)**:
  - Fixed silent NAX `matmul2d` cooperative-tensor prefill corruption in GitHub Actions `macos-26` (Xcode 26.6) release metallibs: compiling `.metal` sources with `-std=metal4.0` without `-mmacosx-version-min` defaulted to `macOS 26.0` (`__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ == 260000`), triggering Apple `MetalPerformancePrimitives` (`MPPTensorOpsAvailability.h`) pre-26.2 cooperative-tensor fallback codegen without `operand_index` and corrupting prefill across NAX models (`SenseNova-Loop-9B`, `Qwen3.8-27B`, `GLM-5.3-Flash`, etc.).
  - Pinned `-mmacosx-version-min=26.2` across `build_metallib.sh`, `check_basis.py`, and `check_toolchain.sh` (which also keeps `macOS 27.x`-built metallibs loadable on `macOS 26.2+`), added `< 260200` compile-time `#error` guards in `nax.h` and `qwattn_sparse_nax.metal`, and added `Platform >= 26.2` + negative-guard verification in `build_metallib.sh`, `Scripts/ci.sh`, and `.github/workflows/release.yml`.
- **SSE Keepalive Stop/Send Synchronization (`QWSSESession`, `QWStreamKeepaliveGate`)**:
  - Synchronized keepalive emission with gate shutdown via `QWStreamKeepaliveGate.runIfActive` so an in-flight keepalive tick can never emit `: keepalive` after `stopKeepaliveIfRunning()` returns.
- **DFlash 2 / DSpine Prefill Context Seeding (`DriverPrefill`, `MTPVerification`)**:
  - Restored DFlash 2 / DSpine prefill context seeding and hardened the `hasValidMTPCache` guard.
- **Wave 1–4 Security, Memory-Safety, GPU Kernel & Architecture Hardening**:
  - **Security & Foundation (`QWServer`, `QWToken`, `QWMenuBar`, `QWAlloc`, `QWPosixIO`, `QWCore`)**: Hardened HTTP/CSRF/SSRF/image and `#modal-memory-pack` validation, grammar arena bounds, `QWJson`, POSIX FD/pointer lifetimes, `FNRope` consolidation into `QWRopeCPU`, and extracted `QWBatchScheduler`, `QWSpeculativeTokenResolver`, and `QWTwoTierPrefixCache` control seams.
  - **GPU Operators, Cold Streaming & Concurrency (`QWMetal`, `QWAttn`, `QWGEMM`, `QWKV`, `QWMTP`, `QWEngine`, `QWFlashNext`)**: Hardened attention, KV cache, RoPE, RMSNorm, GDN, MoE/BF16 router dtype, cold expert/event/drain RAII (`ColdEventPipeline`, `ColdExpertStreamer`), speculative/`DFlash` residency & `num_experts == 0` normalization, non-blocking telemetry & SSE/EOF backpressure, and unified GEMM/MoE/NAX `QuantTraits`, `QWScratchArenaBuilder`, `QWSessionAdmissionManager`, `QWCoarseSelectionPipeline`, and `QWSSESession` abstractions.

## [0.27.3] - 2026-11-19

What's changed since `v0.27.2`: tightened `QWFlashNext` repetition tracking, expanded stochastic and repetition-penalized speculative decoding coverage, hardened `.naive` prefill and async-commit fallback synchronization, and capped unbounded fallback thinking budgets at large `max_tokens`.

### Fixed
- **`QWFlashNext` Repetition Tracking & Fast-Accept Hygiene (`FNGPUSession`, `FNGPUDFlash`, `FNGPUEngine+Seam`, `FNGPUBatchScheduler+Drive`)**:
  - Recorded `firstToken` in `drawFirstToken` when `repetitionPenalty != 1.0`, moved `recordSeenToken` in speculative seam/batch loops from pre-step tokens to newly sampled `next` / `nextTok` tokens, and relaxed fast-accept checks to `!session.seenTokens.contains(draftTok)` while enabling `allowPLD` and `isUnconstrainedNow` `tokenRecycler` parity in `FNGPUDFlash`.
- **Stochastic, Penalty, Constraint & Uncached-Streaming Speculation (`MTPVerification+Rounds`, `MTPVerification+Hybrid`, `Stochastic`, `QWEngineDriverConformance`)**:
  - Enabled `.ngram`, `PLD`, `tokenRecycler`, and `DSpine` / `DFlash 2` greedy-with-penalty speculation across stochastic and `isUnconstrainedNow` constraint sessions, fixed deterministic target sampling in `prepareRound` (`proposals.isEmpty && owner != nil`), updated `tokenRecycler` on stochastic verification commits, and enabled speculative streaming in `generateExperimentalStochastic` for `cache_prompt: false` / `--no-prefix-cache` SSE requests — lifting `Qwen3.8-27B` `DSpine b=16` (`rep=1.05`) decode `+3.31%` (`55.82 -> 57.71 tok/s`, acceptance `43.5% -> 47.1%`).
- **Prefill/Decode Synchronization, Batch Retention & Thinking-Budget Safety Cap (`DriverPrefill`, `DriverDecode`, `DriverBatchDecode`, `QWServerSchema`)**:
  - Skipped unused `40 MB` `mtpPromptEmbeddings` copies when `dflashDrafter != nil`, isolated `.naive` prefill intermediate tokens from `activeConstraint` / session history until the final token, drained `session.lastCommitCB` before touching slabs on serial fallback in `DriverDecode.forwardStep`, retained concurrent stochastic prompts in `DriverBatchDecode`, and guarded explicit `enable_thinking: true` without `reasoning_effort` at `max_tokens > 32` with the `reserved` budget cap.

## [0.27.2] - 2026-11-19

What's changed since `v0.27.1`: fixed persistent multi-turn generation failures, thinking-budget overflows, degenerate repetition loops under WebUI/API sampling overrides (`repetition_penalty > 1.0`, `temperature > 0`), and stochastic speculative speed regressions across dense, looped, MoE, and Flash-Next architectures.

### Fixed
- **Multi-Turn Prefix Cache & Stochastic Token History Hygiene (`PrefixCache`, `DriverPrefill`, `DriverDecode`, `DriverBatchDecode`, `EngineSession`, `QWEngineSessionState`, `FNGPUEngine+Seam`, `FNGPUSession`)**:
  - Restored live prefill head-logit writes (`runsHead == true`) and multi-checkpoint prefix retention on prompt boundaries so Turn 2+ prefix cache hits resume cleanly from the exact KV/recurrent frontier across `QWEngineDriver` and `QWFlashNext`.
  - Fixed prefill/decode `tokenHistory` and `tokenRecycler` corruption during stochastic runs where greedy `argmax` tokens were previously recorded alongside (or instead of) sampled tokens, corrupting constrained decoding, `.ngram` drafting, and repetition penalties on subsequent turns.
  - Scoped `repetition_penalty` strictly to the trailing 64 generated tokens (`promptTokenCount`, `EngineSession.activeRepetitionHistory`, `FNGPUSessionData.recentGeneratedTokens`) so repeated tokens in long system/tool prompts or earlier conversation turns no longer suppress valid output tokens (including `</think>` or `</ernie:think>`).
- **Consecutive-Cycle `loopRepetitionTokens` Predicate (`QWSampler+Distribution`, `FNGPUSession`)**:
  - Replaced blanket full-history / non-consecutive bigram repetition checks—which penalized valid recurring digits (`102` → `10`), math operators, and stop tokens under strong penalties (`repetition_penalty = 1.5`)—with a loop-targeted detector matching consecutive periodic cycles (`4×` period-1, `3×` period-2, `2×` period-`3..16`) or dense bursts (`≥6×` within the trailing 16 generated tokens).
- **Graduated Thinking Budgets & Stochastic Speculative Decoding (`QWServerSchema`, `QWEngineDriverConformance`, `MTPVerification+Rounds`, `MTPVerification+Hybrid`, `FNGPUDFlash`, `chat-presets.js`)**:
  - Wired `reasoning.effort` (`low` → `256`, `medium` → `1024`, `high` → `4096`, `xhigh` → `8192` reasoning tokens) into `resolvedThinkingPolicy` so reasoning traces reliably close within budget across all effort levels.
  - Enabled speculative decoding (`DSpine`, `DFlash 2`, `.ngram`, and speculatable constraints such as `QWThinkingBudgetConstraint` and `QWToolCallConstraint`) with adaptive `DFlash`/`MTP` drafting under stochastic sampling, disabled unsafe greedy-only fast-accept shortcuts when `repetition_penalty != 1.0`, and cleared stale `maxTokens` session overrides when switching WebUI chat presets.

## [0.27.1] - 2026-11-18

What's changed since `v0.27.0`: quantized, qualified, and published **`Apex-35B-V1.0`** (`qwegnai/Apex-35B-V1.0-hq4g32`, `22.18 GiB` `HQ4_G32_M5` + `963.6 KiB` `QLSA0001` sidecar, dynamic `<think>` + AMX vision) from short context up to `1M` (`1,048,576`) tokens with **`6,416 tok/s`** prefill, **`213.8 tok/s`** speculative decode, **`9.79×` prefill / `3.16×` decode** LSA speedup over dense attention at `256K`, **`5,491 / 100.9 tok/s` at `1M`** (`3/3` needles across `64K–1M`), **`295/295 (100%)`** API conformance, **`100%`** engine fidelity, and **`0` steady-state allocations**.

### Added
- **Native Model Pack — `qwegnai/Apex-35B-V1.0-hq4g32` (`HQ4_G32_M5` + `model.qlsa`, `22.18 GiB`)**: Packed `Apexintelligence-AI/Apex-35B-V1.0` (`qwen3_5_moe`, `35.95B` total / `3B` active, `40L`: `10` `16Q/2KV` GQA + `30` Gated-DeltaNet, `256` experts `8+1`, native MTP head, `27`-layer vision tower, `5.30 bpw` across `401` `HQ4_G32_M5` + `644` `BF16` tensors) with data-free SVD `model.qlsa` (`QLSA0001`, `index_dim=64`, budget `2,048`), pullable via `qwg pull qwegnai/Apex-35B-V1.0-hq4g32`.
- **Short-Context, Speculative, Multimodal Vision & `llmprobe --full` Qualification (`1.64M` tok)**: **`6,415.7 tok/s`** at `4K` (`6,475.7` GPU) / **`6,261.8`** at `16K` CLI & **`5,810.2`** at `10.4K` serve prefill; **`213.8 tok/s`** predictable spec (`178.0` serve median, `174.93` CLI `d=3` Var-MTP, `148.54` serial); **`89 ms`** cold TTFT / **`28 ms` (`11.5×`)** warm prefix cache; **`+2.9%`** steady load drift; **`295/295 (100%)`** API conformance, **`100%`** fidelity (`36/36`, `0.968` top-1 prob, `3/3` byte-exact), **`91.9%` (`34/37`)** capability, **`64/88`** agentic, **`72/100`** `code-100` (**`82%` HumanEval**); and CLI/HTTP AMX vision (`3,218.8 / 134.57 tok/s`, `--no-load-vision` RAM guard).
- **`64K → 1M` Latent Sparse Attention (`Ward-512` & `e-means-4096`) Scaling & Infographics**: Verified **`3/3` needle recall across all `13/13` runs (`64K`–`1M`)** with **`0` steady allocs**, **`9.79×` prefill (`442.37s → 45.17s`) & `3.16×` decode (`42.52 → 134.22 tok/s`)** vs Dense at `256K`, and **`5,491.2 / 100.90 tok/s` at `1M`** (`189.40 s` TTFT, `28.96 GiB` `Ward-512` / `28.78 GiB` `e-means-4096` RSS); added `assets/qwegn_model_apex_35b_v1.svg`, `assets/qwegn_lsa_scaling_apex_35b_v1.svg`, and updated `README.md`, `docs/benchmarks.md`, `docs/lsa.md`, `docs/tuning.md`, and `assets/qwegn_engine_bento.svg`.

## [0.27.0] - 2026-11-18

What's changed since `v0.26.0`: onboarded `SenseNova-Loop-9B` Final (`qwegnai/SenseNova-Loop-9B-hq4g32`, `524K` native YaRN-2×, `F32` `mHC`/`GDN`, `SHVQ4` `.qlsa`, `7.38 GiB`) and landed six generalized looped/hybrid engine optimization slices — fused paired-tile NAX `gate_up + SwiGLU` prefill, weight-free `.ngram` (`d=5`) speculation, batched LSA sparse verify (`q4`/`q8`/`f16`), `B = 2..4` concurrent `mHC` + short-context `LSA` server batching, `0`-ULP `C = 2` SIMD Sinkhorn `mHC`, and compact GDN `ab32` / `_hd128` sparse NAX / virtual-layer `DFlash 2` capture — reaching **`2,045 tok/s`** prefill, **`165.7 tok/s`** speculative decode, and **`153.0 tok/s` (`3.02×`)** 4-stream aggregate HTTP decode with **`0` steady-state allocations** and **`0` cross-model regressions** across all 7 flagship packs.

### Added
- **Native `SenseNova-Loop-9B` Final (`qwegnai/SenseNova-Loop-9B-hq4g32`, `524K` YaRN-2×, `F32` `mHC`/`GDN`, `SHVQ4` `.qlsa`)**: Resolved outer envelope `sensenova_loop` ahead of `text_config.model_type = "qwen3_5_text"` (`QWModelDescriptor`), parsed `yarnRopeConfig` (`factor: 2.0` from `262,144` → `524,288` native / `1M` YaRN admission, extending `geometryDigestV3`), preserved `F32` raw `.mhc_attn.`/`.mhc_ffn.`/`.A_log`/`.dt_bias`/`loop_gate` planes (`Scripts/quantize/pack.py`), added `f32`+`shvq1` `.qlsa` SVD support (`Scripts/convert_lsa_sidecar.py`), and published the `7.38 GiB` `HQ4_G32` pack (`4.66 GiB` GPU trunk weights, **`4.87 GiB`** `--no-load-vision` / `6.58 GiB` `--load-vision` server RSS): **`2,044.9 tok/s`** at `4K` / **`1,943.5`** at `16K` / **`1,763.2`** at `64K` prefill, **`165.7 tok/s`** serve spec (`5.49 tok/step`) / **`98.07`** CLI `code_edit`, **`152.96 tok/s`** 4-stream HTTP (**`3.02×`**), **`18 ms`** warm cache (**`46.1×`**), **`211/211 (100%)`** API conformance, **`98.89%`** logit fidelity (`0.0236` nats KLD vs BF16), **`92%` (`23/25`)** HumanEval (`36/50` `Code-50`), **`10/12`** `Core-12` thinking, **`3/3`** needles at `16K`/`64K`, and **`49.29 / 49.35 tok/s`** decode at `524K / 1M` (`14.27 / 22.42 GiB`).
- **Fused Paired-Tile NAX `gate_up + SwiGLU` Prefill Kernels (`qw_gemm_nax_swiglu.metalh`, `qwgemm_swiglu_nax.metal`, `qwgemm_aq_swiglu_nax.metal`, `qwgemm_shvq_swiglu_nax.metal`, `QWGemmAQ`, `DriverWeightKernels`, `QWDenseLayerKernels`)**: Added `0`-ULP paired `(32 gate, 32 up)` Apple10 cooperative-tensor NAX GEMMs across `HQ4_G32`, `AQ*`, and `SHVQ*` (`<= 32 KiB` per-PSO constant-cache LUT isolation) fusing `gate_up + qwg_swiglu_mul + fwht32_inplace` in threadgroup SRAM (`tgGU[4096]`), cutting `guRaw16` traffic (`PrefillScratch`, `DriverInitPhases`) and lifting looped MLP prefill across dense and hybrid models.
- **Looped Virtual-Layer `DFlash 2` Target Capture (`QWDFlashConfig`, `QWDFlashDrafter`, `QWDenseLayerKernels`, `MTPVerification`)**: Added `virtualLayerCapture` / `usesVirtualCapture(for:)` to map virtual looped execution slots (`0..<56`) distinctly without pass-2 target feature overwrite.

### Changed
- **Weight-Free `.ngram` (`d=5`) Speculation, Batched LSA Sparse Verify & `AQ` `0`-ULP IEEE-754 Pinning (`QWEngineDriver+Init`, `QWInferenceParameters`, `MTPVerification+Hybrid`, `QWMTPBatchedTrunk`, `QWAttnDecodeApply`, `qw_attn_sparse_decode.metalh`, `qwattn_decode_general.metal`, `qw_dequant*.metalh`)**: Unlocked zero-VRAM `.ngram` (`d=5`, `QWContextCopyDrafter` + `QWTokenRecycler`) when `mtp_num_hidden_layers == 0` (`d=0 == d=5` `7/7` bit-exact `SHA-256`, **`+88.3%`** `code_edit` `52.07 -> 98.07 tok/s`), lowered `mtpSparseThreshold` to `lsaConfig.denseBypassThreshold` (`8,192`) with single-launch batched `f16`/`q8`/`q4` LSA sparse verify (`+6.52%` at `64K` on `SenseNova-Loop-9B-Preview` `MTP-3`, `94.00 tok/s` at `512`), and pinned `AQ` `M=1..8` IEEE-754 order with `#pragma clang fp contract(off) reassociate(off)` (**`+6.3%`** at `16K` decode on `Qwen3.6-35B-A3B` → `160.97 tok/s`).
- **Concurrent `DriverBatchDecode` (`B = 2..4`) with `mHC` & Short-Context `LSA` (`DriverBatchDecode`, `DriverBatchDecode+LSA`, `DriverSessions`, `QWDenseLayerKernels`, `QWHybridLayerKernels`)**: Enabled multi-session `B = 2..4` batch decode when `mhcConfig` or `lsaOperator` is active (`eligibleBatchSlotIndices`, `prepareLSASessionsForBatchStep`, safe `stepSingleSlot` fallback for `> denseBypassThreshold` or mixed sessions), lifting 4-stream HTTP `/v1/chat/completions` **`3.02×`** (`50.72` → **`152.96 tok/s`**, `4/4` bit-exact serial parity).
- **`C = 2` `0`-ULP SIMD Sinkhorn `mHC`, Compact GDN `ab32`, Parallel `.qwp` Embed Gather & `_hd128` Sparse NAX (`qw_mhc_common.metalh`, `qwmhc*.metal`, `qwgdn_conv.metal`, `qwlsa.metal`, `qwattn_paged_sparse.metal`, `DriverPrefill`, `EncodeKernels+HeadAndEmbed`)**: Added `0`-ULP `qwg_mhc_controls2_simd` (`-15/16` redundant `mhcControls` stores), stored GDN gates at compact `2 * QW_GDN_V_HEADS` stride (`-100 MiB` scratch), parallelized `.qwp` `copyEmbedRows` (`0.07s` → `0.03s` at `64K`) with sliver-free `nextBalancedChunkCount` chunking, added LSA `qMode` dead-query pruning, and specialized `STAGE_HDIM=128` `_hd128` sparse NAX (`16 KiB` TG SRAM, `2×` occupancy).
- **Documentation, Specs, Infographics & Model Cards**: Synced `README.md`, `docs/benchmarks.md`, `docs/tuning.md` (Law 9 7-flagship qualification), `specs/02-model-families-and-descriptor-v3.md`, `assets/qwegn_engine_bento.svg`, `assets/qwegn_model_sensenova_loop_9b.svg`, and all `21` SVGs + `2×` Retina PNGs to `v0.27.0`, and published `qwegnai/SenseNova-Loop-9B-hq4g32` on Hugging Face.

## [0.26.0] - 2026-11-17

What's changed since `v0.25.0`: onboarded the 12th architecture family `Maxwell-V1-Flash` (`maxwell_moe` / `MaxwellMoeForCausalLM`) with Apple10 NAX & split-K Maxwell Block-Sparse Attention (`QWMBSAAttnOperator`), fused `qkvIdxProj` and MoE pre-norm/mHC-to-FWHT32 bridges, server `recommendedPrefillChunkSize` auto-selection, and a zero-bubble async `Commit(r-1) -> Draft(r) -> Verify(r)` speculative pipeline reaching **`6,462 tok/s`** prefill and **`382.8 tok/s`** speculative decode on `Qwen3.6-35B-A3B`.

### Added
- **12th Architecture Family — `Maxwell-V1-Flash` (`maxwell_moe` / `MaxwellMoeForCausalLM`) & Mixed `HQ8_G32 + SHVQ3/4_G64` Pack**: Added full runtime, tokenizer (`<|maxwell_start|>`/`<|maxwell_end|>`), and packer support for 45-layer all-MBSA MoE (`1,024` prefix + `1,024` local + `16×128` remote blocks, `32Q / 4KV` `head_dim=128` `qk_layernorm`, 4-stream Sinkhorn-20 mHC, 3 dense + 42×256 sigmoid top-12 MoE w/ F32 bias, dedicated mHC-MTP head), quantized to **`17.18 GiB`** (`3.84 bpw`, `33,992` tensors, `0.00010` head KL / `0.0055` E2E KLD / `100.0%` top-1 agreement vs `68.59 GiB` BF16), delivering **`5,323.9 tok/s`** at `2K–4K` prefill, **`100.4` CLI / `102.3` serve / `210.5 tok/s` serve spec**, **`15 ms` warm cache (`25.8×`)**, **`139/139` (`100%`)** conformance, `92%` (`23/25`) HumanEval, `100%` (`12/12`) finished MMLU-Pro, and `3/3` needle recall at `16K` and `64K` (`17.10–19.28 GiB` `q4` RAM, `0` allocs). Added `assets/qwegn_model_maxwell_v1_flash.svg` & `assets/qwegn_lsa_scaling_maxwell_v1_flash.svg`.
- **NAX Sparse Prefill & Split-K Decode for `QWMBSAAttnOperator` (`qwattn_mbsa.metal`, `qwattn_mbsa_nax.metal`)**: Replaced scalar MBSA scoring with tiled-32 causal block selection, Apple10 NAX cooperative-tensor sparse prefill (`4.16×` at `5.4K` `q4` -> **`4,663.5 tok/s`**, `2.80×` `f16` -> `4,489.8 tok/s`), and split-K decode (`5.24×` at `5.4K` -> `33.75 tok/s`).

### Changed
- **Zero-Bubble Async `Commit(r-1) -> Draft(r) -> Verify(r)` MTP Pipeline & GPU Argmax (`MTPVerification+Rounds`, `MTPVerification+Pipeline`, `QWHal.touchKeepAlive`, `qwmtp_argmax.metal`, `QWMTPEmbeddingTable`)**: Overlapped prior-round `pendingCommitCB` execution with next-round drafting and GPU embedding on one `waitUntilCompleted()` boundary per round, plus fused `qwg_mtp_pre_fc_norm`, 2D batched target argmax (`qwmtp_argmax.metal`), cache-line-disjoint BF16 GPU `encodeRows` + batched `encodeBatchedMTPPairSeed`, and safe `QWContextCopyDrafter` skip — lifting `Qwen3.6-35B-A3B` to **`6,462.1`** at `2K` / **`6,316.4`** at `6.5K` prefill, **`324.33 tok/s`** CLI `d=5` (`+6.87%`), **`173.60 tok/s`** CLI `d=1` (`+9.02%`), **`182.0 tok/s`** serve decode / **`382.8 tok/s`** serve spec (`82%` `Code-50`, `83.3%` `Core-30`, `80%` `MMLU-Pro`), plus CLI gains on `GLM-5.3-Flash` (`51.38` / `54.7` spec, `+5.24%`), `ERNIE-5.5` (`45.58` / `52.7` serve, `+4.37%`), `SenseNova-Loop-9B` (`91.42` / `149.2` spec, `+3.82%`), and `Qwen3.8-27B+DSpine` (`69.17`, `+1.56%`) with `100%` bit-exact `SHA-256` parity and `0` steady allocs.
- **MBSA `qkvIdxProj` Fusion, MoE Bridge & `QWServerSettings` Chunking (`QWWeightAllocationPlan`, `DriverDecode`, `QWServerConfigurationResolver`)**: Fused MBSA `qProj+kIdx+vProj` (`attnProjTotalN`), fused pre-MoE RMSNorm and mHC router-finalize -> FWHT32 bridge, excluded MBSA from dead LSA key synthesis (`-0.48 GiB`), and applied `recommendedPrefillChunkSize` even without a saved profile JSON.
- **Documentation, Specs & `v0.26.0` Infographics**: Added §7 to `specs/04-hal-memory-and-metal-ceilings.md`, refreshed `assets/qwegn_engine_bento.svg` and all `21` SVGs + `2×` Retina PNGs to `v0.26.0`.

## [0.25.0] - 2026-11-17

What's changed since `v0.24.1`: full OpenAI `/v1/responses` and legacy `/v1/completions` support, Anthropic `/v1/messages/count_tokens`, vLLM-compatible remote tokenizer endpoints, generalized `B = 2..4` continuous multi-session batching across all dense and hybrid MoE families (`219–942 ms` 4-stream worst TTFT vs `2,034–8,320 ms` in `v0.24.1`), universal `16K+` KV page-pressure & RoPE boundary stability (`8/8` flagship configurations passing vs `1/8` in `v0.24.1`), non-thinking/JSON reasoning parity, and GitHub Actions `macos-26` CI gating.

### Added
- **OpenAI `/v1/responses` & Stateful Background Execution (`QWHttpServer+Responses`, `QWServerSchema+OpenAIResponses`, `QWResponsesStore`)**:
  - Added `POST /v1/responses`, `GET /v1/responses/:id`, and `DELETE /v1/responses/:id` with multi-turn `previous_response_id` state chaining, `background: true` asynchronous job execution (`QWResponsesStore` LRU cache), preserved streamed item IDs, tool calling, structured JSON schemas, and disconnect cancellation.
- **OpenAI Legacy `/v1/completions`, Anthropic `/v1/messages/count_tokens` & Remote Tokenizer APIs (`QWHttpServer+Completions`, `QWEngineLifecycleActor+Completions`, `QWHttpServer+Anthropic`)**:
  - Added `POST /v1/completions` with prompt `echo: true`, token `logprobs` (clamped at finite `-9999.0`), streaming SSE, and stop sequences for log-likelihood harnesses.
  - Added `POST /v1/messages/count_tokens` for exact pre-flight Anthropic message, system, and tool-schema token accounting with structured HTTP `400` error mapping.
  - Added vLLM-compatible `GET /tokenizer_info`, `POST /tokenize`, and `POST /detokenize` (plus `/v1/*` aliases backed by `QWTokenizer.piece(_:)` and `QWChatTemplate.rawSource`), enabling zero-workaround evaluation with `lm-evaluation-harness` (`local-completions`) and `inspect_ai` (`responses_api=true`) directly against `.qwp` model packs.
- **Generalized Continuous Multi-Session Batching (`B = 2..4`) Across Standard `QWEngineDriver` (`QWBatchScheduler`, `DriverBatchDecode`, `QWDenseLayerKernels`, `QWHybridLayerKernels`)**:
  - Generalized continuous multi-slot decode batching beyond `QWFlashNext` to all `DenseModule` and `HybridMoEModule` architectures (`Qwen3.8-27B`, `Qwen3.6-35B-A3B`, `SenseNova-Loop-9B-Preview`, `GLM-5.3-Flash`, `ERNIE-5.5-Flash-Base`) with per-slot `ropeRows` capacity alignment, ungated MoE shared-gate initialization, `fgbProj` prewarming, per-request speculation opt-out, and KV-pressure polling backoff — reducing 4-stream concurrent worst TTFT from `2,034–8,320 ms` (`v0.24.1`) to **`219–942 ms`**.
- **Composable Thinking-Budget Constraint (`QWThinkingBudgetConstraint`, `QWChatTemplate.supportsThinkingTags`)**:
  - Added `QWThinkingBudgetConstraint` to bound reasoning token budgets and inject closing `</think>` / `</ernie:think>` tags before `max_tokens` exhaustion while composing cleanly with `QWJSONConstraint` and `QWToolCallConstraint`.

### Changed
- **Low-Latency Streaming & Pre-Stream HTTP `400` Validation (`QWHttpServer`)**:
  - Enabled `TCP_NODELAY` (`tcpOptions.noDelay = true`) on HTTP listeners to eliminate Nagle stalls on single-token SSE chunks, unified client-disconnect cancellation across `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`, `/api/chat`, and `/api/generate` (`QWConnectionCancellationHub`), and moved malformed JSON, parameter-bound, schema, and context-window errors to pre-stream HTTP `400` responses (`sendJsonResponse`).
- **Multi-Slot `DFlash 2` / `DSpine` Residency & Batched `F16` Head Tiling (`QWDFlashDrafter`, `QWMLAAttnOperator+DecoupledRoPE`, `qwhead_f16_batched.metal`)**:
  - Parameterized `QWDFlashDrafter(..., sessionSlotCount:)` to match `DriverResidency.gdnSlotCount()` (`4` slots on `>= 32 GiB` UMA, `2` on `16 GiB`), added decoupled RoPE attention encode paths in `QWMLAAttnOperator`, and updated `qwhead_f16_batched.metal` to use `device` buffer bindings with row-paired 128-thread tiling within the AGX `32 KiB` constant-cache ceiling.

### Fixed
- **Non-Thinking Template Isolation & `/v1/responses` Structured-Output Parity (`QWChatTurnPreparer`, `QWServerSchema`, `QWHttpServer+*`)**:
  - Guarded thinking tags, closing-tag injection, and reasoning constraints behind `QWChatTemplate.supportsThinkingTags` (`<think>` / `<ernie:think>`) across OpenAI Chat, Responses, and Anthropic handlers so base/non-thinking templates (`ERNIE-5.5-Flash-Base`) never enter phantom reasoning states.
  - Prevented non-empty `reasoning.effort` on `/v1/responses` from synthesizing `enable_thinking: true` over JSON/structured-output requests, enforced `thinkingBudget` caps on initial structured-output turns (`resolvedThinkingPolicy`), synced `QWChatStreamParser.setStartsInReasoning(_:allowReasoningTags:)`, and preserved direct JSON brace slicing when `!shouldThink`.
- **Concurrent Batch Recurrent State Hygiene (`FNGPUSession`, `FNGPUBatchScheduler`)**:
  - Unconditionally cleared GDN recurrent states, convolution rings, PLE histories, and parities in `FNGPUSession.wipeSessionState(overwriteTokens:)` and `FNGPUBatchScheduler.cleanupExtraSlotsBeforeSingleMode()` even when `used == 0` / `highWaterPos == 0` (plus `slotStates[syncSlot].highWaterPos` sync after `syncSlotFromEngine`), eliminating inter-request recurrent state leaks under concurrent HTTP bursts.
- **Universal `16K+` KV Page-Pressure Eviction, RoPE Boundary Step-Down & `.qwcache`-Safe Prefix Eviction (`QWKVCache`, `PrefixCache`, `FNPrefixCache`, `FNGPUEngine`)**:
  - Added universal `QWKVCache` page-pressure eviction, greedy/speculative RoPE boundary step-down, and `+1024` KV token headroom (`alignedCap` in `[memory]`), while marking normal per-turn prefix cache entries `isSharedBase: false` so LRU headroom reclaim frees KV/RAM pages without evicting pinned `.qwcache` shared repository bases (`isSharedBase: true`) — fixing `7/8` `16K`-context failures seen in `v0.24.1`.
- **GitHub Actions `macos-26` CI Fix & Bare-Metal GPU Test Gating (`QWTelemetryHonestyTests`, `QWEngineLifecycleTests`, `QWResponsesEndpointTests`, `Makefile`, `Tests/*`)**:
  - Fixed the `v0.24.1` GitHub Actions `macos-26` CI failure in `QWTelemetryHonestyTests.testTheDecodeRateComesFromTheInterTokenLatencies` by replacing wall-clock `Thread.sleep` intervals with deterministic `recordDecodeToken(at:)` timestamps, synchronized `QWEngineLifecycleTests` loading-phase checks via `LoadGate` (`DispatchSemaphore` + `CheckedContinuation`), widened background job polling in `QWResponsesEndpointTests`, aligned `Makefile` `make test` with `Scripts/ci.sh` under `GITHUB_ACTIONS=true`, gated all `v0.25.0` Metal GPU tests on `GITHUB_ACTIONS != "true" && QWDeviceCaps.current?.familyApple9 == true`, and scoped live `~/.qwg/models` pack-scan tests to registered `QWModelFamilyRegistry` model types.
- **M5 Max (`128 GB` UMA) Fleet & Zero-Workaround Qualification (`v0.24.1 -> v0.25.0`)**:
  - **Zero-workaround external suites (`Qwen3.8-Flash-Next` default args, thinking on):** `llmprobe --full --no-bench` **`292/292 (100%)`** (`chat 98/98`, `responses 94/94`, `messages 90/90`, `count-tokens 3/3`, `completions 3/3`, `0` stampede disagreements), `promptfoo` **`8/8 (100%)`**, `inspect_evals/gpqa_diamond` (`responses_api=true`) **`0.800 ± 0.1333`** (`40/40`); `ERNIE-5.5-Flash-Base` `lm_eval` `piqa --limit 50` **`acc = 0.8800` / `acc_norm = 0.9000`** (`100/100`).
  - **8-configuration fleet gate vs `v0.24.1`:** **`89/89 (100%)`** API conformance across all `7/7` instruct packs (up from `61/61` in `v0.24.1`), **`8/8` `16K` context pass** with `0` server errors (vs `1/8` in `v0.24.1`), **`219–942 ms`** 4-stream worst TTFT (vs `2,034–8,320 ms` on `6/8` in `v0.24.1`), paired CLI A/B non-regression (`0.9978x–1.0033x`, `0` steady-state allocations), **`9/9`** bit-exact `SHA-256` greedy determinism, **`8/8 (24/24 needles)`** `16K` 3-needle recall, and live `--load-vision` (`+1.725–1.738 GiB` server / `exit=2` CLI refusal) & `--stream-ngrams` (`+27.13 GiB` `Edge0-145B`) residency verification.

## [0.24.1] - 2026-11-17

### Added
- **WebUI `/ui` Speculative Controls & Drafter Inventory (`QWHttpServer`, `actions.js`, `state.js`)**:
  - Extended `GET /v1/models/discovered` (`QWServerSettings.DiscoveredDrafter` / `scanDrafterSummaries`) with compatible `drafters` summaries (`name`, `packPath`, `backend`, `maxDrafts`, `compatibleModels`).
  - Added `MTP Draft Tokens` (`mtpDrafts` + block-size hint), `Draft Strategy` (`mtpDraftStrategy`), `Drafter Model`, `Drafter Pack Path` (`dflashDraftPackPath`), and `Selector Top-K` (`dflashSelectorTopK`) to `/ui?tab=settings` (`handleWebMTPBackendChange`, `handleWebDrafterSelectChange`) with per-backend `dflash2` / `dspine` availability gating, post-save server reload sync, and even `.settings-grid` alignment (`panels.css`).

### Fixed
- **Stale Cross-Model Drafter Profile Healing & Per-Backend Gating (`QWServerSettings+Discovery`, `QWSettingsManager`, `QWHttpServer`)**:
  - Added `QWServerSettings.supportsExternalDrafter(backend:descriptor:discoveredModel:drafters:configuredPackPath:)` to gate `DFlash 2` vs `DSpine` independently against geometry-compatible installed drafters while preserving strict fail-fast errors in `validateDFlashConfiguration()` and `validateBeforeLaunch()`.
  - Added `reconcileStaleDrafterSelection(for:drafters:)` and wired it into `QWSettingsManager.effectiveSettings` and `POST /v1/settings` model/backend switches to heal deleted or cross-model `dflashDraftPackPath` / unsupported `dspine` selections.
- **`Qwegn.app` 7-Tab Settings UI Alignment (`SettingsView*`)**:
  - Fixed `Performance` horizontal overflow/sidebar bleed (`Cooling Policy` `Auto` | `Pre-Spin` | `Max Blast`, caption wrapping, `.clipped()` detail pane, `195pt` `formRow` label column), unnested helper/lock captions from `formRow` (`Model Alias:`, `KV Cache Mode:`, `Random Seed:`), left-aligned `.menu` `Picker` controls (`alignment: .leading`), allowed 2-line wrapping in `settingsToggleRow`, disabled external backends with no installed compatible pack (`.selectionDisabled`), and hid `Adaptive Draft Depth (Entropy Stop)` when an external drafter is selected; verified across all 7 tabs at `2×` Retina (`740×640` and `660×560+`).

## [0.24.0] - 2026-11-17

### Added
- **Generalized `DFlash 2` / `DSpine` Support (`QWMTP` `L6.0`, `QWFlashNext`, `.qwcache`, Auto-Discovery & Packer)**:
  - Generalized `QWDFlashDrafter` / `QWDFlashConfig` beyond fixed `Qwen3.8-27B` geometry via `QWMTP` `L6.0` Metal kernels (`conv_group_size`, general head counts, dynamic target layer capture, and `vocab` bounds) and wired `FNGPUDFlash` end-to-end inside `QWFlashNext` (`Qwen3.8-Flash-Next-hq8g32-hq4g64` + `incoai/Qwen3.8-Flash-Next-DFlash2`).
  - Added `.qwcache` standalone repository prefix-cache support for `--mtp-backend dflash2` / `dspine` in `qwg index`, `QWMemoryIndexRequest`, `FNPrefixCache`, and `FNPrefixCache+Save` (`0.12 s` warm repo TTFT, `99,412` token hit, plus `--mtp-drafts 0` serial mount reuse of DFlash-enabled `.qwcache`).
  - Added `pack_dflash.py --mlp-quant` and generalized `derive_dflash_model_id` for packing mixed-precision `DFlash 2` packs (`Qwen3.8-Flash-Next-DFlash2-shvq4g32` `521.51 MiB`, `hq4g32` `548.86 MiB`, and `shvq4g32-shvq3g64` `477.69 MiB`).
- **Sparse-MoE `DFlash 2` Defaults & Adaptive Window Selector (`3110a1db`, `47c43a64`)**:
  - Defaulted `FNGPUDFlash.resolveDraftVocab(fullVocab:)` to `98,304` when `fullVocab > 98,304` (`0` disables cap), and ranked compatible drafter packs (`QWDFlashConfig.drafterQuantPreferenceRank` & `QWServerSettings.drafterMatchScore`) to prefer pure `SHVQ4_G32` (`-0.80 GiB` UMA vs `HQ4_G32_M5`, `-0.79 GiB` planned vs native MTP-3).
  - Extended `QWDFlashAdaptivePolicy(isSparseMoE:)` (`FNGPUSession` + `QWEngineSession`/`DriverSessions` gated on `geometry.numExperts > 1`) with `b2CostMs = 25.5`, sparse-MoE bucket priors `(30.5, 39.5, 46.5, 98.0)`, round-1 `min(3, cap)`, and monotonic step-up (`2 -> 3 -> 5 -> 7 -> cap` on `consecutiveFullAccepts >= 2 && reach >= 0.52`), eliminating the 80-expert `M=8` cliff (`p4k +13.2%` vs `master` MTP-3 at `76.25`, `p8k +12.8%` vs `47c43a64` at `65.92` vs `66.79` `master` MTP-3, `novel +13.0%` vs `47c43a64` at `84.40` vs `83.86` `master` MTP-3, `code +6.7%` vs `master` MTP-3 at `90.22`, `16K` 3-needle `+71.0%` vs `master` MTP-3 at `82.34 tok/s`, bit-exact `SHA-256`) while leaving `isSparseMoE == false` (`Qwen3.8-27B`) invariant.

### Fixed
- **`FlashNextModule` Concurrent Batch State Synchronization (`3c1cdf2e`)**:
  - Guarded `FlashNextModule` shared turn/cache state (`_onCacheEvent`, `_prefixCache*`, `_naxLogMessage`, `_lastResult`, `_lastSpec*`) under `stateLock.withLock` and moved `batchScheduler` callback wiring into `FlashNextModule.init(engine:)`, resolving a `_CFRelease` `EXC_BREAKPOINT` race on `QWGenerationResult` destruction under concurrent HTTP bursts (`llmprobe --full` `191/191` conformance, `91.9%–100%` capability, `75/88` agentic, `100%` fidelity, `83/100` Code-100, `175.1` spec / `99.1` bench / `65.9` novel / `94.4 tok/s` at `65K` context scaling).
- **8-Configuration `v0.24.0` M5 Max Qualification**:
  - Verified paired `--strict` `d=0`, speculative A/B, vision/PLE residency toggles, `16K` 3-needle determinism, `61/61` (`100%`) API conformance, and `code-10` across all 8 fleet configurations (`Qwen3.8-Flash-Next + DFlash2-shvq4g32`, `Qwen3.8-27B` `DFlash2` & `DSpine-b16`, `Qwen3.6-35B-A3B`, `SenseNova-Loop-9B-Preview`, `GLM-5.3-Flash`, `ERNIE-5.5-Flash-Base`, and `Edge0-145B`).

## [0.23.0] - 2026-11-16

### Added
- **AppleSMC Thermal Telemetry & Cooling Policy Control (`QWServer`, `/ui`, `QWMenuBar`, Prometheus)**:
  - Added bare-metal IOKit `AppleSMC` telemetry and fan-policy controller (`QWSMCClient.swift`: `QWSMCClient`, `QWSMCSnapshot`, `QWCoolingPolicy`, `QWFanController`), reading GPU/CPU/memory die sensors, fan tachometer RPM (`F0Ac`/`F1Ac`, `F*Mn`/`F*Mx`), and system power (`PSTR`), with privilege-safe `auto` | `proactive` | `max` cooling modes (`atexit` auto-restore, minimum-RPM clamp, optional `/var/run/qwg-smc.sock` helper socket, and explicit unprivileged fallback status).
  - Extended `QWSystemMetricsSampler`, `QWDashboardBuilder`, `QWDashboardViewModel`, `QWTelemetrySnapshotMaps`, `QWPrometheusExposition`, and `QWMenuBarStatusFormatter` (`.gpuTemperature`) with `hardware.gpu_temp_celsius` (`GPU Temp`), `hardware.fan_rpm` (`Fans`), `hardware.system_power_watts` (`System Power`), and `hardware.thermal_state` (`Thermal State`), plus `coolingPolicy` / `fanControlStatus` dashboard facts and `QWEngineTelemetryRegistry` inference-phase hooks.
  - Exposed live `coolingPolicy` (`auto` | `proactive` | `max`) across `QWServerSettings`, `QWSettingsCoherence`, `QWHttpServer`, the `/ui` `Settings` (`Speculative, LSA & Cooling`) and `Monitor` (`Silicon, Memory & Pipeline`) views (`actions.js`), and the macOS `QWMenuBar` popover (`TelemetryCards.swift`) and `Performance` settings tab (`SettingsView+Performance.swift`), backed by `QWMenuBarUITests`, `DashboardRendererParityTests`, `MenuBarRendererParityTests`, `QWSettingsCoherenceTests`, and `QWTelemetryHonestyTests`.
- **Live `.qwcache` Memory Pack Inspection, Streaming SSE Indexing, CLI Lock Delegation & WebUI Modal (`QWEngine`, `QWFlashNext`, `QWServer`, `qwg index`, `/ui`)**:
  - Updated `QWMemoryManifest.matchingCache(..., allowFallbackToAnyModel: false)` to filter model identity (`geometryDigest`, `modelName`, `QWMemoryPack.safeSlug`) before `kvMode` and refuse silent cross-model fallback by default; added `QWMemoryPack.evaluateGitStaleness(manifestSHA:in:)`, per-file tokenization progress (`FileProgressUpdate`), and chunked Metal prefill progress (`onPrefillProgress`) across `QWInferenceEngine`, `QWUnifiedEngine`, `DenseModule`, `FlashNextModule`, `QWEngineDriver`, and `FNGPUEngine`.
  - Added `POST /v1/memory/inspect` (`/api/memory/inspect`) and `POST /v1/memory/index` (`/api/memory/index`, SSE stream + JSON) on `QWHttpServer` / `QWEngineLifecycleActor` (`QWServerSchema+MemoryPack.swift`, `QWEngineLifecycleActor+MemoryIndex.swift`), reporting `ready` / `stale` / `wrong_model` / `missing` status, artifact availability, git commit delta (`commitsBehind`, `gitDiverged`), working-tree modifications, and phase/file/token progress.
  - Enabled automatic CLI `qwg index` delegation (`QwgIndex.swift`) to the running local `QWServer` when `QWEngineInstanceLock` is held, streaming live progress to `stderr` and writing `.qwg/manifest.json` + `.qwcache` without stopping the daemon.
  - Added the `/ui` `#modal-memory-pack` workflow (`controlbar.html`, `css/panels.css`, `js/{state,chat-presets,actions}.js`) across Chat and Settings to inspect workspaces before mounting, surface `missing`, `wrong_model` (`Switch & Mount` + `Index for <active> & Mount`), `stale` (`Update & Mount` + `Mount Existing with Live Delta`), and `ready` states with `KV Mode`, `LSA Mode`, `Scope`, and `Include Tests` controls plus `requestAnimationFrame`-batched SSE progress.

### Changed
- **`GLM-5.3-Flash` & `ERNIE-5.5-Flash-Base` HCA/MLA NAX, Multi-Row GEMV & Sparse-MoE Hybrid MTP Wave (`10acd205`)**:
  - Shipped Apple10 HCA NAX prefill with `24 MiB` SLC-chunked grouped `wo_a` (`-36.8%` GPU span), NoPE MLA `h32` kernels, multi-row verify GEMV (`M = 2..8`), disjoint prefill scratch aliasing (`-410 MiB` `ERNIE-5.5`, `-1.63 GiB` `GLM-5.3` `c=4096`), `1 GiB` `MAP_SHARED` embedding reserve, KDA conditioning (`>= 1e-3`), and sparse-MoE `.hybrid` `neuralCap = 1` (`ERNIE-5.5-Flash-Base`: **`+15.6%`** 4K prefill `1,136.5 tok/s`, **`+25.6%`** `d=5` code decode `64.19 tok/s`, `1,014.7 tok/s` serve prefill; `GLM-5.3-Flash`: **`+24.4%`** `d=5` code `47.89 tok/s`, `68.5 tok/s` serve spec).
- **`Qwen3.8-27B` `DSpine` / `DFlash 2` Drafter NAX GEMMs, `FC-232` `AQ4` Specialization, Adaptive Window, Cliff-Aware Hybrid PLD & `>90 GB` UMA `2048` Prefill Chunk Scaling (`5659668c`, `74e42ef2`)**:
  - Dispatched small-tile NAX GEMMs and single-pass `lm_head` in `DSpine`/`DFlash 2`, coalesced GDN ring replay blits, and switched stream UTF-8 slicing to `O(delta)`.
  - Added compile-time `function_constant(232)` (`QW_IS_AQ4`) specialization across batched head sampling (`qwhead_q4g_batched.metal`) and MTP MLP kernels (`qw_dequant.metalh`, `qwmtp_mlp.metal`), moved batched head activations from `constant` to `device const half*` to avoid `> 32 KiB` AGX constant-cache spills, and verified `0`-ULP `HQ4`/`AQ4` parity for `M in [2, 3, 4, 7, 8, 9, 12, 16]`.
  - Retained high adaptive draft windows (`alpha = 0.18` on `accepted >= 2`, immediate promotion to `min(7, cap)` on full accept in `QWDFlashAdaptivePolicy`), extended suffix matches with `tokenRecycler` (`minPldTokens = min(7, count)`, `MTPVerification+Rounds`), and clamped `8...11`-token PLD drafts to `7` (`M = 8` single-pass verify) to avoid the `M = 9..11` two-pass GEMV cliff.
  - Added `QWServerSettings.recommendedPrefillChunkSize(for:physicalMemory:)` auto-scaling dense models to `2048`-row prefill chunks on `> 90 GB` UMA (`Qwen3.8-27B`: `DSpine` code **`+5.33%`** `77.00 tok/s`, `147.97 tok/s` refactor, `144.1 tok/s` **`+17.5%`** serve spec, `1,043.1 tok/s` cold 4K peak, **`2.23M–2.33M tok/s`** `29.3 ms` warm `65.5K` HTTP prefix cache; `DFlash 2`: `87.72 tok/s` CLI / `154.2 tok/s` **`+37.4%`** serve spec; `SenseNova-Loop-9B`: **`+7.8%`** serve prefill `1,758.6 tok/s` / **`+10.6%`** decode `89.6 tok/s`).

### Fixed
- **Standalone `FlashNext` Telemetry & Stage Labels (`8f633e68`)**: Reported standalone `FlashNext` weight telemetry and non-mHC stage labels accurately across server and menu bar diagnostics.
- **`llmprobe` `v0.23.0` 8-Configuration Fleet Qualification**: All `7/7` instructed flagship configurations scored `61/61` (`100%`, `0` MUST failures) conformance and matched or improved `code-10` (`9/10` on core instruct packs, `8/10` `SenseNova-Loop-9B`, `4/10` uninstructed `ERNIE-5.5-Flash-Base`).

## [0.22.2] - 2026-11-15

### Added
- **End-to-End Chat-to-Engine Request Metadata Handling (`QWServerSchema`, `QWHttpServer`, `QWEngineLifecycleActor`, `QWEngine`, `QWFlashNext`, `/ui?tab=chat`)**:
  - Supported OpenAI, Anthropic (`/v1/messages` bridge), and local-engine (`vLLM` / `SGLang` / `llama.cpp`) request metadata fields (`metadata`, `user`, `session_id`, `prompt_cache_key`, `cache_salt`, `cache_prompt`, `store`) plus `X-Request-ID` / `X-Session-ID` HTTP headers with scalar coercion, validation (max `16` keys, `<= 64` key chars, `<= 512` value chars), and CORS exposure/echo headers.
  - Wired `cache_prompt: false` bypass and `prompt_cache_key` / `cache_salt` namespace isolation uniformly across wire-token prefix caching (`QWEngineLifecycleActor+TokenPrefix`), `QWEngineDriver` (`prefixPolicyKey` + `QWEnginePrefixStore`), and `FNGPUEngine` (`saltedCacheHistory` + `FNPrefixCache`), while preserving unsalted pinned `.qwcache` shared-base reuse when `use_memory_pack != false`.
  - Added per-request `.qwcache` controls (`metadata.workspace_path` discovery priority over `@repo` / system prompt, `metadata.use_memory_pack: "false"` per-turn bypass), `metadata.exact_speculation: "true"` lossless target verification, and `store: false` Request Ledger suppression alongside sanitized metadata attributes in `QWRequestRecord.config`.
  - Added WebUI Chat metadata controls (`#chat-param-cache-prompt`, `#chat-param-cache-key`, `#chat-param-store`, `#chat-param-exact-spec`, `#chat-param-metadata`), custom `key=val` / JSON metadata parser, preset `exactSpeculation` defaults, and request ID telemetry (`req: <id>`).

## [0.22.1] - 2026-11-15

### Added
- **WebUI Chat `.qwcache` (`QWMemoryPack`) Integration (`QWDashboardFacts`, `controlbar.html`, `js/chat-presets.js`, `js/actions.js`)**:
  - Enriched `QWDashboardFacts` (`mountedMemoryPackRoot`, `mountedMemoryPackFiles`, `mountedMemoryPackTokens`, `mountedMemoryPackDelta`, `mountedMemoryPackKV`) from `QWEngineLifecycleActor.getMountedMemoryPackInfo()` so `/ui` surfaces live `.qwcache` telemetry purely through `/v1/dashboard` and `/v1/dashboard/stream`.
  - Added `#chat-mempack-badge`, `#chat-workspace-path`, `Mount .qwcache` / `Unmount` actions (`POST /v1/memory/mount`, `POST /v1/memory/unmount`), live system-prompt path auto-detection (`detectWorkspacePathInSystemPrompt`) with 1-click mounting, and automatic `<workspace_context>` + `<memory_pack_harness>` injection in outgoing `/v1/chat/completions` payloads so the engine's auto-discovery loads the session's `.qwcache` Resident Cortex & Off-Context Vault without redundant tool calls.
  - Added `.qwg` Memory Pack mount/unmount controls and `memory-pack` `kvMode` lock indicator to `/ui?tab=settings`.
- **Researched System Role Presets & Preset Harnessing (`js/chat-presets.js`, `js/chat.js`)**:
  - Upgraded built-in presets (`Coding Agent`, `Deep Reasoning`, `Concise Technical`, `JSON / Structured Extractor`, and `Server Default`) using community-vetted prompt engineering patterns paired with automatic **Preset Harness Profiles** (`applyPresetHarnessToSession`, `buildHarnessedSystemPrompt`) tuning sampling, thinking mode, reasoning effort, `<think>` / `<tool_execution_harness>` directives, native `response_format` JSON constrained decoding, a sample schema loader (`loadSampleExtractionSchema`), and assistant turn `JSON Valid` / `JSON Invalid` + `Format JSON` controls.

## [0.22.0] - 2026-11-15

### Added
- **Dedicated `/ui` Chat & Local Agent Workbench (`controlbar.html`, `css/chat.css`, `js/{chat,state,dashboard,actions}.js`)**: Promoted the WebUI (`?tab=chat`) with a top-level **Chat** workspace alongside `Monitor`, `Models`, `Live Logs`, and `Settings`:
  - **Session Sidebar & Offline Persistence**: Browser `localStorage`-persisted multi-turn sessions (`qwg.webui.chats.v1`) and custom system prompt presets (`qwg.webui.presets.v1`) with collapsible sidebar, `+ New Chat`, instant search/filter, inline rename, delete, 2-step `Clear All`, and `Export JSON` / `Export MD`.
  - **System Prompt Conditioning & `/v1/settings` Sync**: Built-in role presets (`Server Default`, `Coding Agent`, `Deep Reasoning`, `Concise Technical`, `JSON / Structured Extractor`), inline custom preset create/delete, per-session editable system prompt textarea, **Save as Server Default** (`POST /v1/settings`), and expanded `Sampling & Reasoning` controls (`Repetition Penalty`, `Seed`, `Thinking Mode`, `Reasoning Effort`, `Default System Prompt`) in the `Settings` tab.
  - **Streaming, Thinking Traces, Agent Tools & Multimodal Vision**: Live SSE streaming from `/v1/chat/completions` with collapsible `<think>` / `reasoning_content` trace, `AbortController` `Stop` cancellation, per-session overrides drawer (`temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `seed`, `enable_thinking`, `reasoning_effort`, `stop`, `response_format` `text`/`json_object`/`json_schema`, OpenAI `tools` JSON), interactive `role: "tool"` result submission cards, drag/drop/paste `data:image/*;base64` attachments with vision capability status, Edit & Fork, Regenerate, Continue prefill (`continue_final_message: true`), Raw/Rendered toggle, and per-code-block `Copy` buttons (`js/markdown.js`).
  - **Smoke Probe Handoff**: Retained Quick Inference Smoke Probe in `Monitor` with an **Open in Chat →** action (`openSmokeProbeInChat()`) transferring the prompt, `max_tokens`, and any generated response into a Chat session.
- **WebUI Parity & Chat Workbench Verification (`DashboardRendererParityTests`, `QWTelemetryHonestyTests`)**: Registered `chat.css` and `chat.js` across static-asset serving, offline/honesty parity scans, markup assertions, and `JSContext` behavioral tests (`buildChatCompletionPayload`, `parseChatSSELine`, `mergeToolCallChunks`, `splitInlineThinkTags`, code-block copy markdown).

## [0.21.6] - 2026-11-15

### Fixed
- **Pragmatic GitHub Actions CI Gate (`ci.yml`, `Scripts/ci.sh`, `QWTokenTests`, `QWChatTemplateGoldenTests`, `paired_perf.py`)**: Omitted `.venv-oracle` + PyTorch golden fixture generation (`make goldens`) on GitHub Actions CI, gated `Scripts/ci.sh` under `GITHUB_ACTIONS=true` to execute only non-Metal/non-golden host, CLI, server, quant, token, and UI unit test suites (`QWCoreTests`, `QWQuantTests`, `QWTokenTests`, `QWCliTests`, `QWServerTests`, `QWMenuBarTests`) while skipping performance harnesses and bare-metal GPU kernel suites, and isolated `paired_perf.time` / `paired_http_perf.time` via `SimpleNamespace` so `unittest.mock.patch("paired_perf.time.sleep")` never intercepts stdlib `subprocess.Popen._wait` backoff sleeps.

## [0.21.5] - 2026-11-15

### Fixed
- **Menu Bar Popover Concentric Outer Margin & Inter-Card Spacing (`CupertinoTelemetryView`, `TelemetryCards`)**: Placed `.frame(width: 420)` strictly before `.padding(10)` inside `.frame(width: 440).clipped()` with uniform `8pt` inter-card and dock spacing (`GlassEffectContainer(spacing: 8)`, `VStack(spacing: 8)`, `footerDock` `HStack(spacing: 8)` without extra `.padding(.top, 2)`), eliminating the doubled `28pt` left/right outer border (`56pt` wasted width) caused by nested `.padding(.horizontal, 14)` inside `.frame(width: 412)` and restoring `+36pt` of usable card width (`384pt` → `420pt`) with flush right-edge `headerSection` alignment (`.padding(.leading, 2)`).

### Added
- **Popover Outer Margin Regression Gates (`MenuBarRendererParityTests`, `QWMenuBarUITests`)**: Enforced modifier ordering (`.frame(width: 420)` strictly preceding `.padding(10)` plus `.clipped()`) in `MenuBarRendererParityTests`, and added `QWMenuBarUITests.testCupertinoTelemetryViewUsesConcentric10PtOuterMargin()` scanning a `2×` offscreen raster (`880px` wide) to verify symmetric `8.5..13.0pt` left/right card gutters.

## [0.21.4] - 2026-11-15

### Fixed
- **Menu Bar Popover Horizontal Overflow & Long Model Name Marquee (`CupertinoTelemetryView`, `TelemetryCards`, `TelemetryCardPrimitives`, `MarqueeText`)**: Clamped `CupertinoTelemetryView`'s inner stack to `412pt` (`440pt` outer container with `.clipped()`) and preserved right-hand action clusters (`Restart`/`Stop`, `Switch`/`Unload`) via `.fixedSize(horizontal: true, vertical: false).layoutPriority(1)`, preventing long quantization badges (`HQ8_G32+ExpSHVQ15_G64/SHVQ3_G64 (native, GPU)`) and model names (`ERNIE-5.5-Flash-Base-hq8g32-shvq15g64-shvq3g64`) from expanding and clipping left/right popover edges; added reusable `MarqueeText` (`MarqueeSlotLayout` + `MarqueeMath` + `ViewThatFits`) so overflowing model names and quantization pills scroll continuously inside their fixed slot with `1.4 s` initial hold, seamless wrap, edge fade masks, `@Environment(\.accessibilityReduceMotion)` fallback, and zero animation resets across `1 Hz` telemetry ticks while staying static when text fits.
- **WebUI `/ui` Hero Title Containment & Marquee Parity (`css/base.css`, `js/dashboard.js`, `js/actions.js`)**: Constrained `.hero-top` left metadata with `min-width: 0`, added `.hero-model-marquee` / `@keyframes qwg-hero-marquee` (`prefers-reduced-motion` safe) with bounded `.hero-quant-pill`, and guarded `#hero-model-title` DOM reconstruction via `dataset.sig` so long model IDs scroll cleanly without horizontal page overflow or `1 Hz` SSE animation restarts.

### Added
- **Popover Width & Marquee Regression Tests (`QWMenuBarUITests`, `MenuBarRendererParityTests`, `DashboardRendererParityTests`)**: Added `NSHostingView` `440.0 ± 0.5 pt` layout-width enforcement under overflowing ERNIE-5.5 model/quantization fixtures, deterministic `MarqueeMath` hold/scroll/wrap tests, and source/parity assertions across SwiftUI and WebUI assets.

## [0.21.3] - 2026-11-14

### Changed
- **Modular WebUI Asset Layout (`Package.swift`, `QWServer/Resources`, `QWHttpServer+Streams`, `QWHttpServer+Routes`)**: Split the monolithic `controlbar.html` (`3,303` lines) into semantic static files — `controlbar.html` (`407` lines, zero inline `<style>`/`<script>`), `css/base.css` (`618` lines), `css/panels.css` (`476` lines), `js/state.js` (`241` lines), `js/dashboard.js` (`700` lines), `js/markdown.js` (`429` lines), and `js/actions.js` (`476` lines) — copied with directory structure via `Package.swift` `.copy(...)` and served by `QWHttpServer` with `Cache-Control: no-cache`, explicit CSS/JS/SVG/JSON MIME types, directory exclusion (`!isDir`), and crash-free active-bundle discovery.

### Added
- **Offline XSS-Safe Smoke Probe Markdown Renderer (`js/markdown.js`, `css/panels.css`, `controlbar.html`)**: Rendered Quick Inference Smoke Probe responses (including `<think>` / `reasoning_content` collapsible sections, `#`–`######` headings, horizontal rules, GFM tables preserving backtick pipes, nested/task lists, blockquotes, language-labeled code blocks + monospace ASCII trees, inline code angle brackets, snake_case-safe underscores, `Raw`/`Rendered` toggle, `Copy`, and unrestricted `#probe-max-tokens`) with zero external dependencies and strict HTML/URL sanitization, backed by `JavaScriptCore` XSS and live `QWHttpServer` MIME/traversal unit tests (`DashboardRendererParityTests`, `QWTelemetryHonestyTests`).

### Fixed
- **Cross-Phase Sparkline Line Continuity (`QWSparklineGeometry`, `js/state.js`, `MenuBarRendererParityTests`, `DashboardRendererParityTests`)**: Prevented active `1 Hz` decode windows from splitting sub-second prefill measurements into isolated dots (and vice versa) across both WebUI `/ui` and `Qwegn.app` `DualSparklineView` by ignoring `nil` slots while the companion phase is actively running, using monotone horizontal-tangent cubic Bézier segments in `/ui`, keeping single-turn horizontal reference guides visible while decode runs, and still breaking runs cleanly on true unmeasured dual-nil gaps while preserving raw `nil` values in `QWEngineTelemetryRegistry` and hover tooltips.
- **macOS `qwegine_QWServer.bundle` Code-Signing Normalization (`Scripts/build_app.sh`, `Scripts/build_pkg.sh`, `.github/workflows/release.yml`)**: Normalized flat SwiftPM resource bundles emitted by Xcode 26 `swift build` into canonical macOS `Contents/Info.plist` + `Contents/Resources/` `CFBundle` structure before running `codesign`, preventing `bundle format unrecognized, invalid, or unsuitable` errors when signing nested `css/` and `js/` WebUI subdirectories in the GitHub Actions `release` workflow.

## [0.21.2] - 2026-11-14

### Changed
- **README Live Telemetry Previews (`README.md`, `assets/qwegn_dashboard.png`, `assets/qwegn_menubar.png`)**: Replaced the static top banner with a live `2880×1800` `/ui` web dashboard screenshot and added a `932×1014` `Qwegn.app` menu bar control panel preview (`Edge0-145B-hq8g32-hq4g64`).
- **Engineering Laws & Git Hygiene (`AGENTS.md`, `.gitignore`)**: Recorded Swift Testing `#expect(Bool(true))` / `#require` variable-shadowing rules (Law 25), bare-metal vs GitHub Actions `Apple Paravirtual device` CI gating (`GITHUB_ACTIONS=true`, Law 26), and ignored root `bench_kernels_report.*` CLI outputs.

## [0.21.1] - 2026-11-14

### Fixed
- **Live Prefill Progress Streaming (`QWEngineCore`, `DriverPrefill`, `FNGPUEngine`, `QWServerMetrics`, `QWHttpServer+Streams`, `controlbar.html`)**: Wired `onPrefillProgress` across all model runners (standard drivers and `FlashNextModule` / `FNGPUSession`) into `QWEngineTelemetryRegistry.updatePrefillProgress(completedTokens:totalTokens:)`, accelerated `/v1/metrics/stream` and `/v1/dashboard/stream` SSE cadence to `150 ms` while actively generating, and updated `/ui` `#prefill-banner` to track incremental progress (`0%`→`100%`) and hold a brief `100%` `"Prefill Complete"` banner before hiding on transition into decode.
- **1-Second Scrolling Throughput Trace (`QWServerMetrics`, `QWTelemetrySnapshotMaps`, `QWDashboardReadings`, `QWDashboardBuilder`, `TelemetryObservable`, `TelemetryObservable+Streams`)**: Emitted 1-second (`1 Hz`) aligned `prefillRateSeries` and `decodeRateSeries` samples (up to `60` slots, frozen when `.idle`) from `QWEngineTelemetryRegistry` during active `.prefill` and `.decode` without client-side synthetic history, preserved `QWDashboardCard.series` across `nil`-value transitions so Turn 1's prefill trace stays visible during Turn 1 decode, clamped active `Qwegn.app` polling to `<= 1.0 s` on `RunLoop.main` `.common` mode, and guarded against out-of-order HTTP `/v1/dashboard` snapshots overwriting newer SSE frames.

### Added
- **Menu Bar Prefill Progress & Readable Token Formatting (`TelemetryCards`, `TelemetryCardPrimitives`, `QWTokenCountFormatter`, `QWDashboardBuilder`, `controlbar.html`)**: Added a compact `PREFILL PROGRESS` bar inside `Qwegn.app`'s `inferenceTelemetryCard` during `.prefill`, and formatted `tok`-unit counters consistently across WebUI and `QWMenuBar` with ASCII comma separators below `1,000,000` (e.g., `62,707`) and compact decimal `M` notation at `>= 1,000,000` (`1.00M`–`99.99M`, `125.4M` at `>= 99,995,000`) while keeping numeric `QWDashboardCard.number` raw.

## [0.21.0] - 2026-11-14

### Added
- **Zero-Overload 11-Family Model Registry & Config-Sourced Geometry**: Config-sourced 11-family model registry and `geometryDigestV3` replacing overloaded model-type handling, backed by normative `specs/` architecture, family, quant, HAL, security, and UI parity specifications (`specs/06-menubar-and-webui-parity.md`).
- **GitHub Actions Virtualized GPU CI Parity**: Gated bare-metal Apple Silicon GPU execution (`qwg-bench-kernels`) when `GITHUB_ACTIONS=true` while preserving full `make lint`, Swift/Python tests, and build verification.

### Changed
- **Layered DAG, File Decomposition & `@Observable` Menu Bar**: Enforced the 28-target layered `Package.swift` DAG with `-Xswiftc -warnings-as-errors`, decomposed oversized `Sources/` files below 650 LOC, removed dead/`Tests/`-only production symbols, and migrated `TelemetryObservable` and menu bar views to Swift `Observation` (`@Observable`).

### Fixed
- **Security, Runtime Lifetimes & Metal Coherency**: Hardened HTTP framing, loopback/auth defaults, image decoding, path/PID safety, `pread`, zero-copy buffer lifetimes, lock-free PSO snapshots, stream cancellation, `QWAttn` intra-threadgroup `device` coherency (`qwlsa.metal`, `qwlsa_select.metal`, `qwemeans_coarse.metal`, `qwattn_mla_indexer.metal`, `qwattn_mla_indexer_split.metal`, `qw_mla_indexer.metalh`), ungated shared-expert `psc.moe.gate` `1.0` reset in `DriverPrefill.swift`, and `--no-stream-ngrams` / `--stream-ngrams` residency toggle compliance.

## [0.20.5] - 2026-11-14

### Changed
- **Release Infographics & Model Card Sync (`assets/qwegn_*.svg`, `README.md`, `docs/benchmarks.md`, `docs/tuning.md`)**: Updated all 19 repository and Hugging Face `qwegnai/*` infographics for `v0.20.5` with latest `Apple M5 Max (Mac17,6)` measurements (`6,419 tok/s` `Qwen3.6-35B-A3B`, `3,035 tok/s` at `8K` `Qwen3.8-Flash-Next`, `2,572 tok/s` `Edge0-145B`, `1,035 tok/s` `Qwen3.8-27B`, `1,014 tok/s` `GLM-5.3-Flash`, `989 tok/s` `ERNIE-5.5-Flash-Base`), added standalone `qwegn_model_ernie55_flash.svg` and `qwegn_lsa_scaling_ernie55_flash.svg` cards, and documented `qwg pull qwegnai/<pack>` pre-quantized packs.

## [0.20.4] - 2026-11-14

### Fixed
- **Throughput Trace Rendering & Live Projection (`QWSparklineGeometry`, `DualSparklineView`, `QWServerMetrics`, `controlbar.html`)**: Replaced `GeometryReader`-based sparkline paths with an immediate-mode `Canvas` backed by shared `QWSparklineGeometry` dual-band normalization so single-turn traces (`len == 1`), `nil`-gapped runs, and high-prefill / moderate-decode ratios render clearly in `Qwegn.app` and `/ui`; projected the active in-flight decode rate onto `snapshot()` series without mutating turn history, and added `enableAutoLifecycle` on `TelemetryObservable.init`.

## [0.20.3] - 2026-11-14

### Fixed
- **Menu Bar Popover Header (`TelemetryCards`, `TelemetryObservable+MenuBar`)**: Removed the duplicate inline prefill/decode `tok/s` label next to `Loaded` in the `Qwegn.app` control panel header (keeping speeds exclusively on the dedicated `Decode` and `Prefill` tiles and optional macOS menu bar item) and retired the unused `liveMenuBarRateText` helper.

## [0.20.2] - 2026-11-14

### Added
- **Customizable Menu Bar Metrics (`QWMenuBarStatusFormatter`, `SettingsMenuBarAppearanceView`)**: Added a live menu bar preview pill, 4 selectable metrics (`Decode speed`, `Prefill speed`, `Engine memory`, `Context size`), and `"Only show speeds during active generation"` in **Settings → General**, migrating legacy `qwg.ui.showMenuBarThroughput`.

### Changed
- **Menu Bar Control Panel Composition (`CupertinoTelemetryView`, `TelemetryCards`)**: Simplified `RESIDENT MODEL` to show only model-relevant traits (`quantization`, `KV`, `ctx`, `Spec`, optional `MemPack`) with a native `NSMenu` `[Switch ⌄]` picker and `[⏏ Unload]` / `[↓ Load]`; moved server `[↻ Restart]` (`⌘R`) and `[⏻ Stop]` / `[▶ Start]` into the popover header; unified the 2-column `Decode` / `Prefill` card, 3-column `TTFT` / `Context` / `Eng GPU` strip, auto-collapsing `THROUGHPUT TRACE` sparkline, `Engine RAM` (`arena_reserved_bytes` / `resident_bytes`) + capability-gated cache row, and symmetric 4-button footer (`Web UI ⌘D`, `Logs ⌘L`, `Settings ⌘,`, `Quit ⌘Q`).
- **Daemon Restart Shortcut (`QWMenuBarApp`)**: Bound `⌘R` and the header `Restart` control to `restartServerProcess()`, removing the redundant `"Restart model"` button from the resident model tile.

## [0.20.1] - 2026-11-14

### Added
- **App Bundle Co-Packaging**: `Scripts/build_app.sh` now embeds and codesigns the release `qwg` engine binary inside `Qwegn.app/Contents/MacOS/qwg`, and `TelemetryObservable.findQWGBinary()` prefers the sibling executable in `Bundle.main` before falling back to `PATH` or `/usr/local/bin`.
- **Layered `QWG` Microprocessor Emblem**: Updated `QGProcessorLogoView`, `gen_app_icon.swift`, `logo.svg`, and the inline WebUI SVG marks with a layered `Q` / `W` / `G` glyph design.
- **WebUI Console Enhancements**: Added a collapsible **Runtime Parameters** drawer, live **Smoke Probe** panel, server log viewer, model selector/detail cards, and settings controls in the offline `/ui` control bar (`controlbar.html`).

### Changed
- **Menu Bar Popover & Settings**: Refined the `440pt` menu bar popover geometry (`CupertinoTelemetryView`, `TelemetryCards`), standardized numeric spacing, added `steppedSlider` controls across Settings tabs (`SettingsView*`), moved lazy vs eager loading into `Model`, and organized shortcuts (`⌘D` Web UI, `⌘,` Settings, `⌘L` Logs, `⌘R` Restart, `⌘Q` Quit).
- **Log Console (`LogViewerWindow` / `QWLogLineParser`)**: Added structured log-line parsing with severity/category badges, level filtering, search, and line-wrap toggle.

### Fixed
- **SVG Bus Pin Geometry**: Fixed the 4th left bus pin coordinate (`x2="340"` -> `x2="108"`) across all four SVG/Swift/HTML vector sources so the pin no longer cuts across the die wall and glyph strokes.

## [0.20.0] - 2026-11-14

### Added
- **FlashNext / `Edge0-145B` `3K` Prefill Kernel Wave 1+2 (`QWMetal`, `QWFlashNext`, `QWAttn`):**
  - **MoE tail tile compaction & coalesced Q4G NAX (`5ddd33e6`, `9ecbf93b`):** Added `qw_moe_tile.metalh` and compacted `tileR[t]` scheduling in `qwg_moe_group_pairs_f32` across `HQ4`, `AQ`, and `SHVQ` batched NAX kernels (`full 64`-row + compact `16/32`-row tail blocks) plus `qwg_moe_nax_gemm_q4g_t` inner-loop coalescing.
  - **`0`-ULP PLE & HC fusions (`f39d2655`, `9ecbf93b`):** Merged `keyValueBank` `W+H` projection, added bit-exact `qwg_ple_fused_gate_norm`, specialized `qwg_hc_inject_norm_core<false>` (skipping unneeded WHT32 on plain HC), and enabled in-place `qwg_ple_conv` residual accumulation.
  - **Fused GDN `inProjBA32` & strided conv prep (`509cf13c`, `9ecbf93b`):** Concatenated `in_proj_b` + `in_proj_a` into `ba32` for one `[T, 2*VH, H]` GEMV and fast `h / 3u` head indexing when `VHEADS == kh * 3u`.
  - **QSA `simdgroup_half8x8` scorers, full-`T` `chunkMax[516]`, & `gqa8` Sparse NAX (`7bb0567b`, `9ecbf93b`):** Added `BM64/256` `simdgroup_half8x8` flat and `e-means` coarse QSA scorers (`qwqsa.metal`, `qwemeans_coarse_scorers.metal`), full-`T` `trackChunkMax` radix top-k beyond 16 chunks, and `qwg_attn_sparse_gated_nax_gqa8` (`gqa <= 8` halving `xchg` barriers and exponent passes on `24Q/2KV`).
  - **Sliver-free outer prefill chunking (`9ecbf93b`):** Locked `FNGPUSession.prefillStepChunk` once per prompt and deferred partial outer-slice remainders (`FNGPUPrefillStack.deferredSubchunkRows`) with `0` steady-state allocations.
- **Refuted & Rolled-Back Hypotheses (`apple10-roofline-findings.md`):**
  - Rolled back `GDN L=64` UT/WY chunked `f32` prefill (`0.629x–0.858x` vs SRAM-resident `qwg_gdn_recurrent_seq_tg_impl`), 4-TG partial-buffer `qwg_hc_mix_h32` + `qwg_hc_inj_finalize`, and cross-token `reuseExpertsAcrossTokens` (`0.55x–0.69x`).

### Fixed
- **64-bit `>16 GiB` PLE gather offsets (`93893f4a`, `qwple.metal`, `FNGPUDecodeStack.swift`, `FNGPUPrefillStack.swift`):**
  - Widened `targetFloatOffsets` in `qwg_ple_vq2_gather` and `qwg_ple_vq4_gather` to `device const ulong*` and introduced `FNGPUDecodeStack.pleTargetWordOffset(regionBase:actBase:tensorOffset:) -> UInt64`, fixing `UInt32` float-index `SIGTRAP` overflow above `16 GiB` inside `activationsSub` (verified on `Edge0-145B` `1M` `--kv-mode q8 --no-stream-ngrams` at `102.99 GiB` planned).

### Verified
- **Paired `512–256K` Prefill Sweep & Chunk Scaling (`v0.19.1` → `v0.20.0`, `chunk=4096`):**
  - `Qwen3.8-Flash-Next`: `2,072.9 tok/s` (at `512`), `2,941.3` (at `2K`), **`3,034.8`** (at `8K`), `2,724.7` (at `32K`), `2,385.1` (at `65K`, `1.2917x`), `1,909.0` (at `131K`, `1.0271x`), `1,839.7` (at `256K`, `1.0447x`), `0` allocs; chunk scaling `512 -> 2048 -> 4096`: `2,046.6 -> 2,888.3 -> 3,027.6` (at `8K`), `1,851.8 -> 2,640.4 -> 2,763.4` (at `32K`).
  - `Edge0-145B`: `1,945.8 tok/s` (at `512`), `2,543.4` (at `2K`), **`2,568.4`** (at `8K`), `2,291.8` (at `32K`, `1.0496x`), `2,132.3` (at `65K`), `2,051.7` (at `131K`, `1.0427x`), `1,899.3` (at `256K`, `1.0265x`), `0` allocs; chunk scaling `512 -> 2048 -> 4096`: `1,952.5 -> 2,467.9 -> 2,474.4` (at `8K`), `1,710.3 -> 2,297.7 -> 2,351.7` (at `32K`).
- **`Edge0-145B` Beyond-`256K` (`300K`–`1M`) Qualification & Law 9 6-Flagship Gate:**
  - `320K`/`512K`/`1M` (`q4 --stream-ngrams`: `61.84` / `63.99` / `69.73 GiB`) and `1M` `q8 --no-stream-ngrams` (`102.99 / 107.52 GiB`, `0` allocs), Law 12 refusal guards (`1M f16`, `1M q4 --os-reserve 65`, `2M` cap: `rc=1`, `0` SIGTRAPs), and **`3/3` needle recall** at `16K` (`2,507.8 tok/s`), `128K` (`1,918.7 tok/s`), and `300K` (`1,885.5 tok/s` `1.25×` YaRN & `1,882.6 tok/s` raw RoPE).
  - 6-Flagship Law 9 gate: `Edge0-145B` (`2,572.0` prefill `1.0131x`, **`67.59 tok/s` MTP-3 `1.1024x`**), `Qwen3.8-Flash-Next` (`2,947.3` prefill `1.0108x`, **`79.91 tok/s` MTP-3 `1.0210x`**), `ERNIE-5.5-Flash-Base` (`0.997x–1.011x`), `GLM-5.3-Flash` (`1.020x–1.062x`), `Qwen3.6-35B-A3B` (`1.003x–1.030x`), `Qwen3.8-27B+DSpine` (`1.001x–1.008x`); `9/10` `llmprobe` `code-10` on all 5 instruct flagships (`3/10` on `ERNIE-5.5-Flash-Base`), `3/3` `16K` needles & `0` steady allocs across all 6.

## [0.19.1] - 2026-11-13

### Added
- **Apple10 `32`-token small-prefill tiles for `HQ4_G32` and `AQ4_G32`/`AQ4_G64` (`QWGemm.encodeHQ4F32BM`, `qwgemm_f32.metal`, `qwgemm_aq_f32.metal`, `Scripts/kernel_inventory.json`):**
  - Added `qwg_gemm_q4g_f32_bm32`, `qwg_gemm_aq4_g32_f32_bm32`, and `qwg_gemm_aq4_g64_f32_bm32` (`32`-token, `128`-thread tiles, dispatching at `rows == 32` or `33..63` when `K >= 2N`, `epi <= 3`) for symmetry with `AQ8_G32` and `SHVQ4`, reachable across `QWMoELayerKernel`, `QWMoEOperator+Latent.encodeDenseFallback`, and `QWMoEOperator+GroupLatent.encodeResidentGroupLatent`.
- **Cross-model Apple10 roofline attribution & runbook (`apple10-roofline-findings.md`):**
  - Codified empirical Apple10 kernel limits and refuted hypotheses (`W4A8` INT8 NAX `2.65×–2.85×` slower than FP16 NAX; single-pass `B=9..16` DSpine verify `0.6726×–0.9724×`; `wideTG` `32`-row MoE combine `0.9874×`) in `.agents/skills/qwegine-quant-and-kernels/references/apple10-roofline-findings.md`.

### Verified
- **`QWSmallPrefillTests`, `paired_perf.py --strict`, and 6-Flagship Parity (`v0.19.0` → `v0.19.1`):**
  - `QWSmallPrefillTests`: `0`-ULP IEEE-754 bit-exact parity (`3/3`) across `(N, K, M, epi)` combinations; isolated `6`-pair GPU probe achieves **`1.22×–1.52×`** (`(N=2560, K=6144, rows=32)` up to **`1.5167×`**, `MAD <= 0.0021`).
  - `paired_perf.py --strict` across all 6 flagship packs: `Qwen3.6-35B-A3B` (`32`-tok) prefill **`1.0128×`** (`759.7 → 772.7 tok/s`, `MAD 0.0037`, `5/5` pairs `> 1.0×`), `Edge0-145B` (`32`-tok) prefill `1.0054×` (`298.1 → 300.0 tok/s`, `MAD 0.0034`), `ERNIE-5.5-Flash-Base` (`1.000×` inside noise), `Qwen3.8-27B` (+ `DSpine-b16`), `Qwen3.8-Flash-Next`, and `GLM-5.3-Flash` (`SHA-256` bit-exact, `0` steady decode allocations, `3/3` needle passkey recall, `9/10` `llmprobe` `code-10` on instruct flagships).

## [0.19.0] - 2026-11-13

### Added
- **Native `Edge0-145B` (`145B` Hybrid GDN+QSA MoE + `VQ2_G32` PLE, `56.00 GiB` core + `27.13 GiB` PLE = `83.13 GiB` weights / `90.94 GiB` resident / `63.87 GiB` `--stream-ngrams`) support (`pack.py`, `FNGPUBatchScheduler`):**
  - Added NVFP4 (`f8_e4m3` block scales + `f32` tensor-wide scales) two-level dequantization in `Scripts/quantize/pack.py` so NVFP4 MoE checkpoints pack losslessly into `HQ8_G32 + HQ4_G64` within `< 90 GB` disk.
  - Added `FNBatchAdmissionTracker` in `FNGPUBatchScheduler` for clean 4-stream batch admission coordination.

### Changed
- **Wave A Cross-Architecture Kernel & Runtime Accelerations (`QWMetal`, `QWFlashNext`):**
  - `qwg_fn_moe_router_topk_f32`: fused `E=256/512` 1-threadgroup SIMD bitonic router eliminating 2 dispatches and 2 barriers per MoE layer.
  - `qwg_fn_ple_norm_add_f32`: `N_PER_TG=4` coalesced fusion reducing PLE dispatch overhead.
  - `FNGPUStack`: 6-cut pipelined prefill submission for `>= 40` layers plus `FNGPUEngine` prefill scratch release.

### Verified
- **6-Flagship `paired_perf.py --strict` & `llmprobe` Qualification (`v0.18.0` → `v0.19.0`):**
  - `Edge0-145B`: prefill `+4.24%–4.85%` (`1,976.6–1,995.6 tok/s`), MTP-3 decode `+11.75%` (`53.86 → 60.22 tok/s`), 16K prefill `+16.1%` (`1,975.7 tok/s`), `3/3` 16K needles (`SHA-256` bit-exact); `Qwen3.8-Flash-Next`: MTP-3 `--stream-ngrams` decode `+4.67%` (`73.70 → 76.23 tok/s`); `0` regressions & `100%` `SHA-256` parity across `ERNIE-5.5-Flash-Base`, `GLM-5.3-Flash`, `Qwen3.6-35B-A3B`, and `Qwen3.8-27B` (+ `DSpine-b16`).
  - `Edge0-145B` `llmprobe`: `99.3%` (`140/141`) conformance, `100%` capability, `36/36` fidelity, `9/10` `Code-10` & `Core-10`, `80.9 tok/s` novel / `116.4 tok/s` (`1.76×`) speculative / `102.6 tok/s` 4-stream batching / `119.8×` (`8 ms`) warm prefix cache.

## [0.18.0] - 2026-11-13

### Added
- **Native `ernie5_5_moe` architecture support (`ERNIE-5.5-Flash-Base`, `280B` total / `14B` active + `0.36B` ECHO, `81.61 GiB` `HQ8_G32 + SHVQ15_G64 + SHVQ3_G64`, `2.52 bpw`) across all runtime layers (`QWCore` → `QWEngine` + `ernie55_adapter.py`):**
  - `QWHCAAttnOperator` & `qwattn_hca.metal` (`<= 32 KiB` constants): `37` Heavily Compressed Attention layers (`128:1` `compress_kv`, `128`-tok SWA ring, YaRN `theta=160,000`, `o_groups=8`, `o_lora_rank=1024`, VHA, `>262K` Weight-SVD `QWLSASidecar` `e-means`, and `.qwcache` repo memory).
  - Decoupled-RoPE `QWMLAAttnOperator+DecoupledRoPE` & `qwattn_mla_dsa.metal`: `6` backbone `MLA-DSA` layers (`64H`, `index_topk=2048` Walsh-Hadamard `H_128`, `128`-tok SWA tail, `attn_sink`, VHA, `>= 41K` `64H` `QWCoarsePoolIndex` `e-means`).
  - `LatentBottleneck` `QWMoEOperator` (`512` routed experts, `moe_split_feature_routing` dual-split sigmoid `top-10` + `1` shared expert, `SituAndMul` `beta=4.0, linear_beta=25.0`), additive-eps 4-stream `mHC` with `hc_head`, `MultiMax` `lm_head` (`qwg_multimax_f32`), and shared-Layer-43 `QWEchoDraftEncoder` (`ECHO`).

### Changed
- **Phase 2 M5 NAX & SHVQ Prefill Accelerations (`QWMetal`, `QWGEMM`, `QWAttn`):**
  - Added Apple10 `32`-head `qwg_hca_attn_nax`, staged `ShvqTgLut` SRAM codebooks inside Grouped-NAX `qwg_moe_nax_gemm_shvq*` and `qwg_moe_nax_split_shvq*_f32`, gated `QWMoEOperator` grouped-NAX on density (`groupedNaxMinRows = 26` at `E=512, K=10`, `minNax = 16`), and added `BM=16/32` NAX tiles to `gemmBatchedH16`.
  - Lifts paired `ERNIE-5.5-Flash-Base` Phase 2 prefill across every regime (`t11_zh` `84.8 → 89.2` `1.052×`, `t29_code` `178.1 → 211.4` `1.185×`, `t512` `756.0 → 825.3` `1.092×`, `t19k` `788.6 → 855.9` `1.130×`, `873.3–898.9` at 18.8K).

### Fixed
- **Latent-attention MTP draft-head argmax scratch sizing (`QWEngine`):** Sized `partialsCarveBytes` via `QWEngineDriver.draftHeadArgmaxPartialsBytes(vocab:)` (`55297ddd`) so fused draft-head argmax on latent/compressed-attention models (`GLM-5.3-Flash` `--mtp-drafts 1`) has full partials capacity.
- **Basis-gate compile-time scale purity (`QWMetal`):** Folded `0.5f * 0.17677669529663689f` into `0.08838834764831844f` (`1/sqrt(128)`) in `qw_mla_fwht128_ortho` (`fe5cb42d`) so `Scripts/check_basis.py` passes cleanly (`505/755` shuffles dropped, `0` leftover `1/sqrt(32)` constants).

### Verified
- **`ERNIE-5.5-Flash-Base` (`81.61 GiB`):** `mean_kl = 0.1757` nats/tok (`79.69%` top-1 vs `BF16`); `ECHO` `d=1` decode `31.39 → 40.22 tok/s` (`+28.1%`, peak `42.57`); `.qwcache` `78.59×` (`22.79 → 0.29 s`); `3/3` needles at `18.8K`, `149.5K` (`603.6 / 37.93`), and `300K` (`302.6 / 16.28`); `1M` `87.59 GiB` planned / `91.31 GiB` RSS UMA residency; `4/4` vision×ngram toggles `SHA-256` exact (`0` allocs).
- **Cross-Flagship `--strict` Gate (`v0.17.1` → `v0.18.0`):** `0.984×–1.001×`, `5/5 SHA-256` exact, `9/10` `llmprobe` `code-10`, and `0` steady allocs across `GLM-5.3-Flash` (serial & MTP-1), `Qwen3.6-35B-A3B`, `Qwen3.8-Flash-Next` (`-27.07 GiB` `--stream-ngrams`), and `Qwen3.8-27B` (+ `DSpine-b16` `1.0014×`).

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
  - **`184 / 184` (`100%`) API conformance** (`9/9` core endpoints, `0` errors), **`37 / 37` (`100%`, `strong`) capability**, **`81 / 100` (`81.0%`) HumanEval+MBPP** (`46/50` `92.0%` HumanEval, `35/50` `70.0%` MBPP), **`66 / 88` (`75.0%`) agentic execution**, **`99.74%` engine fidelity** (`36/36` correctness, `3/3` greedy determinism, `36/36` logprob argmax consistency), **`53.6 tok/s` predictable / `45.6 tok/s` novel** NextN MTP-1 (`1.98 tok/step`), **`825.3 tok/s` `9.4K` prefill** (`780 tok/s` at `65.6K`), **`96×` (`1,728 → 18 ms`) warm prefix cache**, and **steady `41.1 → 43.3 tok/s` (`+5.4%`) 21-minute sustained thermal load**; updated all GLM infographics and documentation.

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
  - Paired, interleaved `--strict` A/B (`0%/0%` quiet GPU) lifts `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`q4` KV) prefill **`145.8 → 1,005.9 tok/s` (`6.92×`, TTFT `35.64 → 5.17 s`, `1,061.3 tok/s` GPU at 4K, `851.8–869.2 tok/s` `qwg serve`)**, serial decode **`23.81 → 44.64 tok/s` (`1.88×` CLI; `46.0` med / `46.4` peak `qwg serve`)**, exact NextN MTP-1 **`23.44 → 47.59 tok/s` (`2.03×` paired CLI, `51.19` standalone, `50.7` peak `qwg serve`, `10 ms` `193.4×` warm cache)**, and at `242.5K` `3/3`-needle long context to **`743.0 / 44.98 tok/s` (`e-means`, `6.29×` prefill & `1.69×` decode vs at `160K` `v0.14.1`)**, with `0.0%` regression (`0.994×–1.008×`), `5/5` bit-exact `SHA-256` determinism, `3/3` needle fidelity, and `9/10` `llmprobe` `code-10` across `GLM-5.3-Flash`, `Qwen3.6-35B-A3B`, `Qwen3.8-Flash-Next`, and `Qwen3.8-27B` (+ `DSpine-b16`).

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
  - Registered 44 quiet-GPU (`0% / 0%`) empirical runs from `16K` / `64K` to `1,039,928` (`1M`) tokens across `Qwen3.6-35B-A3B-hq4g32` (`2,317.0 / 105.56 tok/s` at 1M `Ward-512`, `136.84 tok/s` at 64K `e-means`), `Qwen3.8-Flash-Next-hq8g32-hq4g64` (`1,887.6 / 63.74 tok/s` at 1M `e-means` — `+39.8%` prefill / `+55.3%` decode vs Native QSA), `Qwen3.8-27B-hq4g32` (`830.6–914.3 tok/s` prefill `+3.2%–9.0%` `e-means`), and `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`26.63 tok/s` at 160K `+19.8%` `e-means`, `1M` `102.3 GiB` UMA admission) with **100% (`3/3`) 3-needle passkey recall** on every sparse run, plus 6 dual-panel Prefill Profile & Decode SVG infographics (`assets/qwegn_lsa_*.svg`) in `docs/benchmarks.md`, `docs/lsa.md`, and `docs/lsa-codebase-memory-whitepaper.md`.

## [0.14.0] - 2026-11-11

### Added
- **Spherical Hadamard Vector Quantization (`SHVQ{2,3,4}_G{32,64,128,256}`), `QNGRAM v3 VQ2_G32` 2-bit PLE tables (`50.9 → 27.1 GiB`), and zero-overspill Metal SHVQ quantizer (`QWQuant`, `QWGEMM`, `QWMLP`, `QWFlashNext`, `Scripts/quantize/shvq.py`, `pack_ngram.py`, `emeans.py`).**
  - Implemented 12 `SHVQ{2,3,4}_G{32,64,128,256}` e-means vector quantization formats (`arXiv:2611.09501`) across dense GEMV, greedy `lm_head` argmax, batched Apple10 NAX GEMM, `f32` BM/GEMV (`bm32` exact 32-row small-prefill tiles), and grouped/routed MoE kernels (`ShvqTgLut<2|3>` threadgroup SRAM codebook staging + `SHVQ4` 32 KiB direct half-LUT lookup), shrinking `GLM-5.3-Flash` (`321B`) from `134.37 GiB` (`3.59 bpp` cold-streamed) to **`100.63 GiB` (`2.69 bpp` `SHVQ4_G32 + SHVQ2_G64 + SHVQ3_G64`, `-33.74 GiB` / `-25.1%`)** so all `321B` weights reside **100% in 128 GB UMA** (`36.2 tok/s` decode, `44.0 tok/s` speculative, `0` SSD expert misses).
  - Added `QNGRAM v3 VQ2_G32` 2-bit e-means PLE N-gram table quantization (`170 → 90 bytes/row`, `50.86 → 27.13 GiB`, **`-23.73 GiB` / `-46.7%`**) with resident Metal GPU gather (`FNGPUPLETable` / `qwg_fn_ple_vq2_gather`) alongside SSD sector streaming, shrinking `Qwen3.8-Flash-Next-hq8g32-hq4g64` + PLE total disk footprint from `122.15 GiB` to **`98.42 GiB`** and enabling GPU-resident PLE alongside the `71.29 GiB` `HQ8+HQ4` pack (**`95.1 tok/s` decode `+13.9%` vs `83.5` `v0.13.1`, `145.4 tok/s` predictable `4.92 tok/step`, `2,528 tok/s` at 10.4K prefill `+11.5%`**) with `10/10` (`100%`) `llmprobe` core reasoning and `9/9` structured JSON fidelity.
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
