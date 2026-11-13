---
name: qwegine-quant-and-kernels
description: >-
  Use when optimizing or benchmarking Qwegn Metal kernels (GEMM, GEMV, NAX, MoE,
  GDN, MLA/HCA/QSA, small-prefill tiles), researching or packing quantization
  formats and codebook LUTs (HQ*, AQ*, SHVQ*, VQ*, QNGRAM), tuning speculative
  drafters and batched verification (DFlash 2, DSpine, MTP, ECHO), or auditing
  operator scalar fallbacks (MLA/LSA/MoE/KV modes) and UMA scratch residency.
---

# Qwegn Metal Kernel, Roofline & Quantization Runbook

Follow these eight gates in order when introducing or optimizing a quantization format, LUT-backed Metal kernel, or operator fast path:

1. **Gate 0 — Call-Graph Reachability, Quality Oracle & Single-Slice Probe Before Engine Wiring:**
   - Consult [apple10-roofline-findings.md](./references/apple10-roofline-findings.md) first to avoid already-refuted M5 Max kernel shapes (`W4A8` INT8 NAX, `M > 8` single-pass verify, outer-`m` pre-unpacked GEMV, wide-TG disjoint-weight MoE combine).
   - Trace exact call-graph reachability across all flagship paths (`DenseModule`, `QWMoELayerKernel`, `QWMoEOperator` / `Latent-MoE`, `FlashNextModule`, `NoPE MLA` / `HCA`) and calibrate benchmark prompt token counts to hit the exact dispatch predicate (e.g., `rows == 32` for `_bm32`).
   - **Lossy quant / pruning:** Benchmark candidate reconstruction MSE, cosine similarity, and logit KL in `Scripts/quantize/` against existing Qwegn formats at matched `bpw`. Reject any PTQ format that degrades target quality at equal `bpw`. Route sub-2-bit PTQ formats (`< 2.0 bpw` without QAT) to exact-verified speculative drafters (`DFlash 2` / `DSpine`) or cold banks that pass full `llmprobe` + needle gates.
   - **Throughput kernels:** Measure an isolated GPU-timestamp probe (`>= 6` alternating pairs, `min-of-2`, `>= 8` reps/CB) before multi-file engine integration, and implement **one kernel slice per diff** so any end-to-end regression rolls back cleanly.
2. **AGX `<= 32 KiB` Constant-Cache Isolation & SRAM Staging:**
   Sum the byte sizes of all `constant` arrays referenced by each Metal PSO (`xcrun metal` / `.metalh` includes) and keep every PSO `<= 32 KiB`: compile-time format-specialized PSOs may bind one `<= 32.0 KiB` direct half LUT (`QW_SHVQ_ROUTED_DIRECT_LUT`), runtime-`bits` multi-format PSOs must split codebooks across `_sub2` / family PSOs (`template <bool SUB2>`), and `2–4 KiB` sub-4-bit LUTs in divergent dense/GEMV/MLA loops should stage into `threadgroup` SRAM (`ShvqTgLut`).
3. **Code-Generated Tables, Plane Sizing & Residency Toggles:**
   - Keep binary table assets (`Scripts/quantize/*.bin`), Swift tables (`QWQuant`), Metal headers (`Sources/QWMetal/include/`), and doc prose (`docs/quantization.md` dyadic formulas, asset names, script paths) generated/synced from a single Python emitter.
   - Size mixed raw + quantized packs exclusively through `PlaneLayout.resolve(e)` (`e.rawOff != nil` checked before quantized plane layouts, rounded to `16 KiB` alignment).
   - Honor `--load-vision` / `--no-load-vision`, `--stream-ngrams` / `--no-stream-ngrams`, and `--no-profile`.
4. **Operator Fast-Path Sweep & High-UMA (`>= 75%`) Residency:**
   - Audit `QWGemm`, `QWAttn` (`QWMLAAttnOperator`, `QWLSAOperator`, HCA), `QWGDN`, and `QWMTP` dispatch predicates so non-incumbent weight containers (`HQ*`/`AQ*`/`SHVQ*`) and quantized KV modes (`q8`/`q4`) stay on MMA/NAX paths rather than falling back to scalar kernels.
   - Preserve full `4096`-row MoE NAX tile occupancy across CLI and `QWServer` (exempting subquadratic MLA/LSA/HCA paths from quadratic `>131K` chunk clamps with command-buffer slicing), auto-evict `PrefillScratch` across both `DenseModule` and `HybridMoEModule` when `wiredEngineBytes >= 75%` of physical RAM, and exclude unused `MAP_SHARED` tables (`QWMTPEmbeddingTable` when `mtpDrafts <= 1`) from `MTLResidencySet`.
5. **Kernel Inventory Source of Truth & Zero-Warning Swift Closures:**
   Edit `Scripts/kernel_inventory.json` first and regenerate `Sources/Runtime/QWEngine/Module/QWKernelInventory.swift` via `python3 Scripts/gen_kernel_inventory.py` whenever adding or rolling back kernels (`gen_kernel_inventory.py` overwrites `QWKernelInventory.swift` from JSON; verify `python3 Scripts/lint_kernels.py` -> `0 problem(s)`). Write explicit closure buffer types (`{ (raw: UnsafeRawBufferPointer) in`) on all `Data` / `MTLBuffer` pointer closures.
6. **Basis Gate, Grouped-NAX SRAM LUTs & Argmax Scratch Invariants:**
   - **Compile-time basis (`Scripts/check_basis.py`):** Gate weight-basis `fwht32` rotations at build time (`#if !QW_BASIS_STANDARD`) so rotations never sit behind runtime branches. For basis-independent `N != 32` transforms inside kernels that drop a `fwht32`, fold normalization into a single constant (e.g. `0.08838834764831844f` for `1/sqrt(128)`) — never leave a `1/sqrt(32)` (`0x3FC6A09E60000000`) factor in standard-basis AIR.
   - **Grouped-NAX SRAM LUTs & small-`M` regime alignment:** Wire `DECL_TG` / `PREP_TG` (`HAS_TG = true, tgLut.ptr16()`) into both dense NAX (`qwg_gemm_shvq*`) and Grouped-NAX MoE kernels (`qwg_moe_nax_gemm_shvq*`, `qwg_moe_nax_split_shvq*_f32`). Scale `groupedNaxMinRows` with `E / K` density for high-expert-count (`E >= 256`) MoE models, align precast dense/shared paths (`T >= 16`), and dispatch `BM = 16 / 32` NAX specializations for short prefills (`M <= 32`).
   - **Latent-attention MTP argmax partials:** When `usesLatentOrCompressedAttention == true`, size fused draft-head argmax scratch for full `vocab` via `QWEngineDriver.draftHeadArgmaxPartialsBytes(vocab:)` rather than collapsing to `4096 B`.
7. **Quiet-GPU Bit-Exact & Paravirtual CI-Safe Unit Tests:**
   - Add `0`-ULP IEEE-754 `bitPattern` CPU/GPU dequant and GEMV/NAX parity tests (preserving reduction tree order and canonical `+ 0.0f` sign) without loading real model packs. Guard every NAX cooperative-tensor dispatch with `try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, ...)` while keeping standard Metal parity assertions active for GitHub Actions `macos-26` runners.
   - Confirm `0` active GPU co-runners before diagnosing tie-sensitive GPU parity tests (such as `QWLSASelectParityTests`), which can flake under `99%` external GPU contention.
8. **Paired Flagship & Speculative Parity Qualification:**
   - On an idle GPU (`0%/0%` utilization, `0` co-runners), run `Scripts/paired_perf.py --strict` against fresh `master` release across every supported flagship pack, `qwg-bench-kernels`, `llmprobe`, and 16K+ needle passkey: pass `--options-json` as a raw JSON array string (`'["--kv-mode","q4","--no-profile"]'` or `"$(cat opts.json)"`, not a file path), always include `"--no-profile"`, and omit `"--mtp-backend"` whenever `"--mtp-drafts", "0"` is set.
   - For speculative drafter formats, verify **100% bit-exact greedy token SHA-256 identity** against serial target decode across multi-prompt sweeps, and preserve `out/llmprobe/*.json`.
