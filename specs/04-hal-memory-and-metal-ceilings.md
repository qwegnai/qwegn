# 04 — HAL Concurrency, Memory Discipline & Apple Silicon Ceilings

> Normative specification for `QWHal`, `ColdExpertStreamer`, UMA memory budgeting, and AGX hardware limits.

## 1. Lock-Free Post-Freeze PSO Cache (`QWHal`)
`QWHal` manages Metal `MTLComputePipelineState` (PSO) lookup with zero lock contention on the hot decode/prefill critical path (`Sources/Foundation/QWHAL/QWHal.swift`, verified by `QWHalDoorArenaTests.testFrozenPipelineHitsTakeNoLock`):
- Mutable pipeline compilation and table registration (`MutablePSOState`) are guarded by `OSAllocatedUnfairLock`.
- Publishing/freezing snapshots builds an immutable `FrozenPSOSnapshot` and swaps `frozenSnapshotPtr` (`Synchronization.Atomic<UnsafeMutableRawPointer?>`) with release/acquire ordering.
- Prior snapshots remain retained in `retiredSnapshots` until `QWHal.deinit` releases the active atomic pointer, guaranteeing safe lock-free reader dereference throughout process lifetime.

## 2. AGX 32 KiB Constant-Cache Ceiling
- Apple Silicon AGX spills `constant` address-space tables to device DRAM whenever a single PSO references `> 32 KiB` of constant arrays.
- Separate codebook families (e.g. `qw_shvq_tables.metalh`, `qw_shvq_sub2_tables.metalh`, `qw_shvq_routed_lut.metalh`, `qw_shvq_sub2_routed_lut.metalh`) MUST stay isolated in family-specific `.metalh` headers and PSOs so each PSO stays `<= 32 KiB` constant LUT footprint.

## 3. Unified Memory Discipline & Zero Steady-State Allocations
- **Host budget (`128 GiB` UMA):** `QWResidencyAdmission` / `QWAdmission` verifies `planned <= available` before loading weights or expanding context/KV arenas; refusals exit non-zero before allocation.
- **Zero steady-state allocations:** After warmup/pre-allocation (`QWArena`), decode and prefill steady state executes with `0` heap or Metal buffer allocations per step across dense, MoE, GDN, HCA, and MLA/QSA paths.
- **Dual-slot sidecar admission & GPU preemption (`QWResidencyAdmission.admitDecisionSidecar`, `QWGPUTurnArbitrator`):** When a `decisionModelPackPath` sidecar co-resides alongside a primary model, `admitDecisionSidecar` verifies `primaryWired + sidecarEstimate + warmCache <= availableBytes` (clamping warm caches first, refusing nonzero before allocation). Shared `QWGPUTurnArbitrator` pauses primary Metal prefill/decode chunks cleanly between command-buffer boundaries when a Metal decision sidecar call awaits GPU access; CoreAI ANE (`.aimodel`) sidecars bypass GPU arbitration completely.
- **Sequential GPU ownership:** Only one heavyweight GPU inference process executes at a time; benchmarks (`Scripts/paired_perf.py`) run strictly sequentially with quiet-GPU preflight checks.

## 4. Safe `ColdExpertStreamer` Synchronization
- Cold-expert NVMe streaming (`Sources/Runtime/QWEngine/Driver/ColdExpertStreamer.swift`) eliminates unchecked `!` unwraps across `syncGPUSlotMap`, `verifyPromoted`, and `prepareSpanDynamic`, validating slot bounds and staging buffers explicitly before GPU promotion.
- `QWHal` command queue allocates `maxCommandBufferCount: 256` to accommodate ~2 CBs per MoE layer during cold-expert decode without command-buffer starvation.

## 5. Bare-Metal vs Virtualized CI Gating
- `QWDeviceCaps.isMetal4TensorCapable` (`MTLDevice.supportsFamily(.apple10)`) separates real Apple Silicon Metal 4 hardware from virtualized CI GPUs (`Apple Paravirtual device`, 16 GiB RAM, no `~/.qwg/models`).
- Production code and Metal kernels never weaken or branch around virtualized runner limits; instead, bare-metal Metal 4 tests (NAX cooperative tensors, 1024-thread GDN recurrent kernels, 32 KiB-TG / multi-TG atomic e-means selectors, full hybrid/speculative GPU driver E2E) gate in `Tests/` via `XCTSkipUnless(hal.caps.isMetal4TensorCapable, ...)`, and host-dependent checks (`ru_maxrss`, `~/.qwg/models`, Debug CPU timing budgets) gate via `GITHUB_ACTIONS` `XCTSkipIf`/`XCTSkipUnless`.

## 6. Multi-Session Continuous Batching, Recurrent Reset & Drafter Slot Sizing
- **Unified batch scheduler (`QWBatchScheduler`, `QWBatchAdmissionTracker`):** The single `QWEngineDriver` (`DenseModule`, routing standard execution via `DriverBatchDecode` and `hcCount > 1` HyperStream execution via `QWHyperStreamRuntime` / `FNGPUBatchScheduler`) supports continuous multi-slot batching (`maxSlots = 4`) alongside universal KV page-pressure eviction and RoPE boundary step-down across `>= 16K` contexts.
- **Unconditional recurrent state hygiene (`FNGPUSession`, `FNGPUBatchScheduler`):** Exiting batched waves back to single-slot mode (`cleanupExtraSlotsBeforeSingleMode`) and resetting slot-0 (`wipeSessionState(overwriteTokens:)`) unconditionally clear GDN recurrent states, convolution rings, PLE histories, and parities even when `highWaterPos == 0` or `used == 0`, preventing inter-request state leakage across concurrent HTTP bursts. After `syncSlotFromEngine(syncSlot)`, `slotStates[syncSlot].highWaterPos` is synchronized to `engine.highWaterPos`.
- **Drafter slot alignment & prefix cache pinning (`QWDFlashDrafter`, `PrefixCache`, `FNPrefixCache`):** `QWDFlashDrafter(..., sessionSlotCount:)` matches `DriverResidency.gdnSlotCount()` (`4` on `>= 32 GiB` UMA, `2` on `16 GiB`). Normal per-turn prefix caching stores entries with `isSharedBase: false` so LRU headroom reclaim works without evicting mounted `.qwcache` shared bases (`isSharedBase: true`).

## 7. Async Speculative Commit Retention & Cache-Line-Disjoint MTP Embeds
- **Async `Commit(r-1) -> Draft(r) -> Verify(r)` pipeline (`MTPVerification`, `QWHal.touchKeepAlive`):** Speculative verification retains ephemeral staging and input buffers used by asynchronous `pendingCommitCB` command buffers in `QWHal` via `touchKeepAlive(_:)` until GPU completion, overlapping prior-round `Commit` execution with next-round drafting and GPU MTP embedding inside a single `waitUntilCompleted()` boundary per speculative round.
- **Cache-line-disjoint `QWMTPEmbeddingTable.tokens` sub-buffers:** Across multi-draft `encodeRows` and `encodeBatchedMTPPairSeed` invocations, CPU-written `tokenIDs` slots and GPU-read embedding outputs are partitioned on 128-byte cache-line boundaries inside `QWMTPEmbeddingTable`, eliminating false sharing during pipelined BF16 GPU MTP seed construction.

## 8. Sliding-Window Split-KV Span & Law 11 Reduction Ruling (ADR-023)
- **128-key sliding-window Split-KV span (`QWAttnSinkRel.splitSpan`, `QWAttnSinkRel.splits`):** When `slidingWindow > 0`, `QWAttnSinkRel.splitSpan(ringCap:slidingWindow:maxSplits:)` partitions active ring keys into `128`-key fixed blocks (`256`-key blocks for global attention), and `QWAttnSinkRel.splits(...)` enables Split-KV whenever `activeKeys > 128` for sliding windows (`> 256` for global windows), doubling threadgroup split occupancy across bounded sliding windows (e.g. `slidingWindow == 1024`) while keeping `T=1` and `T=2..16` batch partition boundaries identical.
- **Explicit `Law 11` reduction ruling (`ADR-023`):** Halving the sliding-window Split-KV block span changes cross-split FP32 reduction tree partitioning relative to the prior `256`-key baseline (producing an expected argmax tie-break token-stream SHA shift on sliding-window packs compared to pre-ADR `master`), while preserving exact FP64 CPU reference parity (`QWAttnSinkRel` + `QWAttnSharedKVBatchParityTests` `0-ULP` `T=1 == T=2..16`) and `100%` bit-exact speculative self-consistency (`d=0 == dspark == dflash2`) with `+15.2%` decode throughput on `ctx2k` DFlash2 (`+5.1%` at `ctx8k`).

## 9. Unified Apple Silicon M1..M6 (`Apple7`..`Apple11`) Capability & Simulation Model
- **Single unified runtime across all generations (`QWDeviceCaps`, `QWHal`):**
  - `familyApple10 || familyApple11` (`isMetal4TensorCapable`): M5/M6 Metal 4 NAX tensor-op dispatch (`kernel.metallib` / `kernel_std.metallib`, `-std=metal4.0 -mmacosx-version-min=26.2`).
  - `familyApple9` (`hasDynamicCaching`, `supportsMetal4Library`): M3/M4 Metal 4 library without NAX, dispatching `simdgroup_half8x8` (`_sg` / `_mma`) paths and enforcing NAX pipeline poison guards (`QWNaxKernelRegistry`).
  - `familyApple7 || familyApple8` (`supportsSimdgroupMMA`, `!supportsMetal4Library`): M1/M2 load stripped Metal 3.2 libraries (`kernel_m3.metallib` / `kernel_m3_std.metallib`, `-std=metal3.2 -mmacosx-version-min=15.0 -DQW_DISABLE_NAX=1`, `0` NAX tensor-op symbols) automatically via `QWWeightBasis.metallibFileName(metal4: false)`.
  - `QWGemm.naxRowChunk(rows:k:gpuCores:)` scales activation-panel SLC budgets across core tiers (`24 MiB` at `>=25c`, `12 MiB` at `13..24c`, `6 MiB` at `<13c`), and grouped MoE `simdgroup_half8x8` kernels (`qwg_moe_sg_*`) use a `10 KiB` `QWSGMoETileMem` with direct-register `thread_elements()` stores while dense `BM=64` GEMMs retain coalesced `16 KiB` `QWSGTileMem`.
- **Host simulation overrides (`QWDeviceCaps.probe`):**
  - `QW_SIMULATE_GPU_FAMILY` (`apple7|m1`, `apple8|m2`, `apple9|m3|m4`, `apple10|m5`, `apple11|m6`), `QW_SIMULATE_GPU_CORES`, and `QW_SIMULATE_WORKING_SET_GB` (clamped `<= device.recommendedMaxWorkingSetSize`) allow deterministic bare-metal validation of every Apple Silicon tier on an M5 host.
