---
name: qwegine-quant-and-kernels
description: >-
  Use when optimizing or benchmarking Qwegn Metal kernels (GEMM, GEMV, NAX, MoE,
  GDN, MLA/HCA/QSA, small-prefill tiles), researching or packing quantization
  formats and codebook LUTs (HQ*, AQ*, SHVQ*, VQ*, QNGRAM), tuning speculative
  drafters (DFlash 2, DSpine, MTP, ECHO), or auditing UMA scratch residency.
---

# Qwegn Metal Kernel, Roofline & Quantization Runbook

Follow these eight gates in order when introducing or optimizing a quantization format, LUT-backed Metal kernel, or operator fast path. Consult [apple10-roofline-findings.md](./references/apple10-roofline-findings.md) and [kernel-invariants-and-bench-flags.md](./references/kernel-invariants-and-bench-flags.md) before writing code.

## Procedure

1. **Gate 0 — Roofline Audit, Quality Oracle & Single-Slice Probe:**
   - Check [apple10-roofline-findings.md](./references/apple10-roofline-findings.md) to avoid refuted M5 Max shapes (`W4A8` INT8 NAX, `M > 8` single-pass verify, outer-`m` pre-unpacked GEMV, `8 KiB` `sparse_nax` double-buffering, `16 -> 9` MLA splits, rank-64/SVD compressed LSA).
   - Profile live bottlenecks first (`QW_PREFILL_PROFILE=1 QW_DECODE_PROFILE=1`) and trace call-graph reachability across `DenseModule`, `QWMoELayerKernel`, `QWMoEOperator`, `FlashNextModule`, and `NoPE MLA` / `HCA`.
   - **Lossy quant / pruning:** Benchmark MSE, cosine similarity, and logit KL in `Scripts/quantize/` at matched `bpw`; route `< 2.0 bpw` PTQ without QAT only to exact-verified drafters or cold banks passing `llmprobe` + needle gates.
   - **Throughput kernels:** Run an isolated GPU-timestamp probe (`>= 6` alternating pairs, `min-of-2`, `>= 8` reps/CB) — or `.build/release/qwg-bench-kernels --only-core-scaling --samples 6 --repeats 8` for TG-saturation and dispatch-tax ceilings — and implement **one kernel slice per diff**.
2. **AGX `<= 32 KiB` Constant-Cache Isolation & SRAM Staging (`Law 27`):**
   - Keep every Metal PSO's `constant` arrays `<= 32 KiB`: isolate format-specialized PSOs (`QW_SHVQ_ROUTED_DIRECT_LUT`), split runtime-`bits` codebooks via `_sub2` / `template <bool SUB2>`, and stage `2–4 KiB` sub-4-bit LUTs into `threadgroup` SRAM (`ShvqTgLut`).
3. **Code-Generated Tables, Plane Sizing & Residency Toggles:**
   - Keep `Scripts/quantize/*.bin`, `QWQuant`, `Sources/QWMetal/include/`, and `docs/quantization.md` synced from one Python emitter; size mixed raw/quant packs via `PlaneLayout.resolve(e)` (`rawOff != nil` checked first, `16 KiB` aligned); honor `--load-vision`/`--no-load-vision`, `--stream-ngrams`/`--no-stream-ngrams`, and `--no-profile`.
4. **Operator Fast-Path Sweep & High-UMA (`>= 75%`) Residency:**
   - Keep `HQ*`/`AQ*`/`SHVQ*` and `q8`/`q4` KV on MMA/NAX paths (`QWGemm`, `QWAttn`, `QWGDN`, `QWMTP`) and apply the residency/aliasing/verify invariants in [kernel-invariants-and-bench-flags.md](./references/kernel-invariants-and-bench-flags.md).
5. **Kernel Inventory Source of Truth & Zero-Warning Closures:**
   - Edit `Scripts/kernel_inventory.json` first, regenerate `QWKernelInventory.swift`, and pass both inventory and reachability gates:
     ```bash
     python3 Scripts/gen_kernel_inventory.py && python3 Scripts/lint_kernels.py
     ```
   - Use explicit closure buffer types (`{ (raw: UnsafeRawBufferPointer) in`) on `Data` / `MTLBuffer` closures (`Law 25`).
6. **Basis Gate, Fast-Math Order, KDA & Split-KV Invariants:**
   - Gate weight-basis `fwht32` rotations at compile time (`#if !QW_BASIS_STANDARD`, verified via `python3 Scripts/check_basis.py`), fold `1/sqrt(128)` (`0.08838834764831844f`) when dropping `fwht32`, preserve `-std=metal4.0 -O2` FP32 `fma` order, and follow KDA/Split-KV/argmax contracts in [kernel-invariants-and-bench-flags.md](./references/kernel-invariants-and-bench-flags.md).
7. **Quiet-GPU Bit-Exact & Paravirtual CI-Safe Unit Tests:**
   - Require `0`-ULP `bitPattern` parity tests without loading real packs, guard NAX dispatches with `XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, ...)`, pass `-mmacosx-version-min=26.2` on every `metal -std=metal4.0` build + `#error` `< 260200` guard (`Law 29`), and verify idle GPU + metallib target:
     ```bash
     ./Scripts/preflight.sh --require-quiet-gpu && ./Scripts/build_metallib.sh && python3 Scripts/check_metallib_target.py
     ```
8. **Paired Flagship & Speculative Parity Qualification:**
   - Run `python3 Scripts/paired_perf.py --strict` (see [kernel-invariants-and-bench-flags.md](./references/kernel-invariants-and-bench-flags.md) for required flags) against fresh `master` release across flagship packs, `qwg-bench-kernels`, `llmprobe`, and `16K+` passkey needles (`100%` SHA-256 greedy parity, `0` steady-state allocations; preserve `out/llmprobe/*.json`).

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| AGX `constant` cache spill degrades kernel >2× | Split LUTs into format-specific PSOs (`<= 32 KiB` each) or stage `2–4 KiB` LUTs into `threadgroup` SRAM (`ShvqTgLut`). |
| Corrupt NAX `matmul2d` prefill output with 0 warnings | Pass `-mmacosx-version-min=26.2` and run `python3 Scripts/check_metallib_target.py` (`Platform >= 26.2`). |
| `QW_METALLIB_PATH` ignored during release vs local bisect | Set `QW_METALLIB=<path>/kernel.metallib` (`QWHal.swift` reads `QW_METALLIB`) and run the 2×2 matrix in [kernel-invariants-and-bench-flags.md](./references/kernel-invariants-and-bench-flags.md). |
| `Scripts/paired_perf.py --strict` rejects flags or fails `d=0` | Omit `"--mtp-backend"` when `"--mtp-drafts", "0"`, use `--mtp-draft-strategy hybrid|neural|ngram`, and remove existing `--output` path. |
| `python3 Scripts/check_basis.py` catches `1/sqrt(32)` in std AIR | Fold exact `1/sqrt(N)` (`0.08838834764831844f` for `128`) directly when `#if QW_BASIS_STANDARD` drops `fwht32`. |
| `qwg serve` refuses 2nd instance during `B=2..8` HTTP A/B | `QWPaths.swift` enforces a single-engine PID lock; alternate base/candidate servers sequentially per round (`start -> warm -> timed batch -> SIGTERM`) reusing `paired_http_perf.py` helpers. |
| `qwg serve` rejects `--no-prefix-cache` or `--warm-cache-limit 4096` | Pass `--warm-cache-limit 0 --ssd-cache-limit 0` (`warmCacheLimitGiB` is in GiB `0.0...512.0`, not tokens) and `--lsa-mode off` (not `none`). |
