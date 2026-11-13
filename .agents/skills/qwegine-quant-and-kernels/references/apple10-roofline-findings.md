# Apple10 (M5 Max, 40-Core GPU) Measured Kernel Findings & Prerequisites

Consult before proposing or re-implementing GEMM, GEMV, MoE, NAX, or speculative-verify optimizations on Apple10 GPUs.

## 1. Measured Negative Results (Do Not Re-Attempt Without New Hardware Mechanism)

| Candidate Optimization | Measured M5 Max Result | Root Cause |
| :--- | :--- | :--- |
| **Online `W4A8` `NAXTile<int8_t>` GEMM** (`HQ4` unpack -> `int8_t` + dynamic `simd_max` activation quant) | **`2.65x–2.85x` slower** than FP16 `encodeNaxF32` at `N=K=2560, M=64..1024` (`60–64 TFLOPS` FP16 baseline) | Per-block `HQ4` nibble-to-`int8_t` unpack, per-tile activation quantization, and scale/bias correction consume more vector ALU cycles than INT8 tensor pipes save. |
| **Single-pass `B=9..16` speculative verify** (`QWMTPBatchedTrunk` / `QWMTPBatchedMLP` / `QWHead`) | `v1` **`0.7617x`**, `v2` **`0.6726x`** (`48.65 -> 32.17 tok/s`), `QWHead`-only **`0.9724x`** (`48.13 -> 46.87 tok/s`) vs `master` `2x8` split | `M=16` panels (`17408x16x2 = 544 KiB`) thrash L1 activation cache and spill VGPRs; even `M`-way ILP `QWHead` loses `2.76%` occupancy vs two `M=8` dispatches. Keep batched GEMV verify split at `M <= 8`. |
| **Pre-unpacked outer-`m` GEMV loop** (`half2 wv[R][16]` across `m=0..M-1`) | **`0.6726x`** decode on `Qwen3.8-27B` + `DSpine-b16` | Holding `wv[2][16]` (`32` VGPRs) alongside `acc[2][16]` (`32` VGPRs) spills registers and serializes FMA chains. Keep `master`'s streaming `w=0..3` inner unpack (`4` `half2` regs live) with `m`-parallel accumulators. |
| **Wide-TG (`32` rows/TG) MoE `down_shared_combine`** | **`0.9874x`** decode on `Qwen3.6-35B-A3B` MTP-3 | Output rows read disjoint weight rows while input `x` is already L1-hot (`0` DRAM weight savings), and halving threadgroups (`128 -> 64` at `hidden=2048`) drops occupancy across `40` GPU cores. |
| **Cross-token MoE `reuseExpertsAcrossTokens` on `FlashNext` / widening `> 4` rows** | **`0.55x–0.69x`** speed ratio; non-bit-exact `down/combine` | Token-union packing/scatter overhead outweighs expert weight reuse at top-10/512 sparsity, and the packed 3-bit token field caps deduplication at `7`. |

## 2. Proven Positive Tile Pattern

* **Apple10 `_bm32` (`32`-token, `128`-thread) `f32` GEMM tiles (`HQ4_G32`, `AQ4_G{32,64}`, `AQ8_G32`, `SHVQ4_G{32,64}`):** Dispatch `_bm32` when `rows == 32 || (rows > 32 && rows < 64 && K / 2 >= N)` and `epi <= 3`. Eliminates `50%` idle row lanes vs `64`-row `_bm` (`256` threads), yielding **`1.22x–1.52x`** isolated operator speedups (`N=2048..2560, K=512..6144`) and **`1.0112x`** `32`-token CLI prefill speedup on `Qwen3.6-35B-A3B-hq4g32` with `0`-ULP bit-exactness.

## 3. Structural Prerequisites for Deferred Ideas

* **Tree-Structured Speculative Verification:** Requires (1) parent-bitmask causal attention in paged verify (`verifyTokensLocked` currently assigns linear `pos + qi + 1` causal prefixes where sibling branches attend to one another) and (2) parent-slot-indexed GDN recurrent state and PLE conv history rings (currently row-ordered linear rollback tapes).
* **Adaptive Probability-Mass (`tau`) Top-$k$ Expert Pruning:** Do not substitute cross-row `--moe-spec-budget` (`budget=8` drops `Qwen3.6-35B-A3B` `llmprobe` `code-10` from `9/10` to `1/10`). Any per-token `tau` cutoff requires router probability-mass trace analysis, an offline logit/KL oracle, full `llmprobe`, and `3`-needle passkey gates before merge.
