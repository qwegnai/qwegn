---
name: qwegine-quant-and-kernels
description: >-
  Use when researching, adding, packing, or benchmarking quantization formats
  (HQ*, AQ*, SHVQ*, VQ*, QNGRAM), codebook LUTs, Metal GEMV/NAX dequantization
  kernels, or multi-format speculative drafters (DFlash 2 / DSpine) in Qwegn.
---

# Qwegn Quantization & Metal LUT Kernel Runbook

Follow these six gates in order when introducing or modifying a quantization format or LUT-backed Metal kernel:

1. **Gate 0 — Offline Quality Oracle Before `Sources/`:**
   Benchmark candidate reconstruction MSE, cosine similarity, and logit KL in `Scripts/quantize/` against existing Qwegn formats at matched `bpw`. Reject any PTQ format that degrades target quality at equal `bpw`. Route sub-2-bit PTQ formats (`< 2.0 bpw` without QAT) to exact-verified speculative drafters (`DFlash 2` / `DSpine`) or cold banks that pass full `llmprobe` + needle gates.
2. **AGX `<= 32 KiB` Constant-Cache Isolation:**
   Sum the byte sizes of all `constant` arrays referenced by each Metal PSO (`xcrun metal` / `.metalh` includes). Keep every PSO `<= 32 KiB`. When adding new LUT codebooks (`shvq_*_tables.bin`, `VQ*`), emit dedicated `.metalh` headers and `_sub2` / family-specific PSOs (`template <bool SUB2>`) instead of unioning >32 KiB of tables into existing kernels.
3. **Code-Generated Tables, Plane Sizing & Residency Toggles:**
   - Keep binary table assets (`Scripts/quantize/*.bin`), Swift tables (`QWQuant`), Metal headers (`Sources/QWMetal/include/`), and doc prose (`docs/quantization.md` dyadic formulas, asset names, script paths) generated/synced from a single Python emitter.
   - Size mixed raw + quantized packs exclusively through `PlaneLayout.resolve(e)` (`e.rawOff != nil` checked before quantized plane layouts, rounded to `16 KiB` alignment).
   - Honor `--load-vision` / `--no-load-vision`, `--stream-ngrams` / `--no-stream-ngrams`, and `--no-profile`.
4. **Kernel Inventory & Zero-Warning Swift Closures:**
   Register every new kernel entry point in both `QWKernelInventory.swift` and `Scripts/kernel_inventory.json` (`python3 Scripts/lint_kernels.py` -> `0 problem(s)`). Write explicit closure buffer types (`{ (raw: UnsafeRawBufferPointer) in`) on all `Data` / `MTLBuffer` pointer closures.
5. **Paravirtual CI-Safe Unit Tests:**
   Add bit-exact CPU/GPU dequant and GEMV/NAX parity tests without loading real model packs. Guard every NAX cooperative-tensor dispatch with `try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, ...)` while keeping standard Metal parity assertions active for GitHub Actions `macos-26` runners.
6. **Paired Flagship & Speculative Parity Qualification:**
   On an idle GPU (`0%/0%`), run `Scripts/paired_perf.py --strict` against fresh `master` release across all 4 flagship packs, `qwg-bench-kernels`, `llmprobe`, and 16K needle passkey. For speculative drafter formats, verify **100% bit-exact greedy token SHA-256 identity** against serial target decode across multi-prompt sweeps.
