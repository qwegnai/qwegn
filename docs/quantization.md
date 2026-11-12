# Quantization Formats & Quality Tooling

## Supported formats

By default `qwg quantize` writes `HQ4_G32_M5`. It applies an orthonormal Walsh–Hadamard rotation to each 32-weight group, then stores 4-bit codes plus a scale and bias (5 effective bits per weight) in a layout the Metal kernels read directly. The engine also supports non-incumbent rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` and standard-basis affine `AQ{2,3,4,6,8}_G{32,64,128,256}` packs, with row-paired batched verify (`M >= 4`), 4-row fused-FWHT gate/up, and batched `lm_head` kernels active across both incumbent `HQ4_G32` and non-incumbent `HQ4_G32` / `AQ4_G32` containers.

The full matrix — Hadamard-rotated `HQ{2,3,4,6,8}_G{32,64,128,256}`, affine `AQ{2,3,4,6,8}_G{32,64,128,256}`, Spherical e-means `SHVQ{1,15,2,3,4}_G{32,64,128,256}`, `QNGRAM v3` (`VQ2_G32` and `VQ4_G32`), `HQ4_G32_M5`, and `BF16` — supports mixed-format packs and external `DFlash 2` / `DSpine` speculative drafters (`Scripts/quantize/pack_dflash.py --quant`), such as:

- `SHVQ4_G32` dense/MLA/MTP trunk + `SHVQ2_G64` routed `gate_up` (`2.25 bpw`) + `SHVQ3_G64` routed `down_proj` (`3.25 bpw`) (`2.69 bits/param`, **`100.63 GiB`** on `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` — `-33.74 GiB` / `-25.1%` vs `HQ4_G32_M5 + HQ3_G64`, fitting `321B` parameters 100% resident in 128 GB UMA at **`1,006 tok/s` prefill, `46.0–53.6 tok/s` decode, `81/100` HumanEval+MBPP (`92%` HE), and `99.74%` `llmprobe` fidelity**).
- `HQ8_G32` dense trunk + `HQ4_G64` (`4.5 bits/weight`) routed experts (`4.75 bits/param`, `71.29 GiB` `.qwp` — `5.16 GiB` smaller with `2.62×` lower KLD vs BF16 than uniform `HQ4_G32`) paired with **`QNGRAM v3 VQ2_G32`** (`90 B/row`, **`27.13 GiB`** vs `50.86 GiB` v2 `q8g32`, `-46.7%`) or **`QNGRAM v3 VQ4_G32`** 4D e-means PLE N-gram embeddings (`50 B/row`, **`15.07 GiB`**, saving **`-12.06 GiB`** additional UMA vs `VQ2_G32` at `0.9557` cosine similarity) on `Qwen3.8-Flash-Next-hq8g32-hq4g64`.
- **`SHVQ15` (`1.5`-bit) and `SHVQ1` (`1.0`-bit) antipodal sub-2-bit spherical VQ:** `SHVQ15_G64` (`14 B/grp`, `1.75 bpw`; `1.625 bpw` at `G128`) packs `1` hemisphere sign bit + `11`-bit (`2048`-centroid) 8D spherical e-means index (`3` bytes per two 8D octets) with a 0-ULP dyadic `stepMap15` (`int8 * 3 * 2^-7`); `SHVQ1_G64` (`10 B/grp`, `1.25 bpw`; `1.125 bpw` at `G128`) packs `1` hemisphere sign bit + `7`-bit (`128`-centroid, `2 KiB` LUT) 8D spherical e-means index (`1` byte per 8D octet). On `Qwen3.8-27B` `DSpine-b16` speculative drafters (`pack_dflash.py --quant shvq15_g64`), `SHVQ15_G64` shrinks the drafter pack by `-42.0%` (`1,595.53 -> 925.22 MiB`, `-0.74 GiB` resident RAM) while delivering `1.229×` serial decode speedup at `B=8` with `100%` bit-exact target token-stream identity (`0.0%` target quality loss).

## SHVQ lookup, AGX constant-cache isolation, and small-prefill tiles

The Spherical Hadamard Vector Quantization family (`SHVQ{1,15,2,3,4}_G{32,64,128,256}`)
is supported across dense GEMV, head argmax, batched NAX GEMM, f32 BM/GEMV,
grouped/routed MoE, NoPE MLA absorption, and `DFlash 2` / `DSpine` drafter kernels:

- **AGX `<= 32 KiB` constant-cache PSO isolation (`_sub2` split):** Apple AGX GPUs
  spill `constant` address-space tables to slow device memory when a single Metal
  `MTLComputePipelineState` references `> 32 KiB` of constant arrays. Qwegn splits
  sub-2-bit codebooks (`qw_shvq_sub2_tables.metalh`: `2 KiB` `SHVQ1` + `16 KiB`
  `SHVQ15` = `18 KiB`) from `>= 2`-bit codebooks (`qw_shvq_tables.metalh`: `22 KiB`
  `SHVQ{2,3,4}`) across `qwg_shvq_read_elem_t<bool SUB2>`, `qwg_mla_absorb_q_sub2` /
  `qwg_mla_proj_uv_sub2`, and `qwg_dspine_refine_scores_sub2`. Every PSO stays
  `<= 22 KiB`, preserving `GLM-5.3-Flash` `SHVQ4/2/3` decode throughput (`1.00×`)
  while keeping `SHVQ1_G64` (`0.0367 ms`, `0.712×` `AQ2_G64`) and `SHVQ15_G64`
  (`0.0480 ms`, `0.765×` `AQ2_G64`) `1.31×–1.41×` faster than `AQ2_G64` on
  256 MiB-flushed cold-streaming `[12288×4096]` decode GEMV.
- **SHVQ4 direct half codebook + `bm32` small prefill:** Routed f32 gate/up,
  down/combine, and grouped split-NAX MoE (`qwg_moe_nax_split_f32_shvq4_g*`) read
  the generated 32-KiB half-value codebook (`kShvq4HalfLutBits`) directly instead
  of reconstructing half values from the 16-KiB packed-u8 table on each lookup.
  For `SHVQ4_G32` and `SHVQ4_G64` on Apple GPU family 10+, `QWGemm.encodeHQ4F32BM`
  also dispatches exact 32-token/128-thread `bm32` tiles at 32 rows, or at 33--63
  rows when `K >= 2N` (with `epi == 2` restricted to 32 rows), matching the
  64-token reference float bit patterns across epilogues `0..3`.
- **SHVQ1/2/3 threadgroup SRAM codebook staging (`ShvqTgLut<1|2|3>`):** Because random
  `constant` address-space lookups serialize across simdgroups on AGX
  when indices diverge, dense decode/GEMV (`_dx_h16_t4`, `_v4`, `_v4_f32`), greedy
  head argmax, f32 GEMV (`_f32_v4`, `_f32_v4_am`, `_f32_v4_am2`), and routed MoE
  f32 gate/up + down/combine kernels cooperatively stage the `2 KiB` (`SHVQ1`),
  `4 KiB` (`SHVQ2`), or `2 KiB` (`SHVQ3`) codebook into `threadgroup` SRAM once per
  threadgroup.
- **Pack-lane separation (`GLM-5.3-Flash` & sub-2-bit PTQ):** `HQ8_G32` dense/MLA/MTP
  trunk + `SHVQ4_G64` routed MoE experts is the fidelity-matched pack lane (retaining
  cold-expert NVMe streaming). `SHVQ4_G32` trunk + `SHVQ2_G64`/`SHVQ3_G64` routed
  experts (`100.63 GiB`) fits fully resident in 128 GB UMA as an explicit
  approximate-pack lane; sub-2-bit `SHVQ15` (`1.625–1.75 bpw`) and `SHVQ1`
  (`1.125–1.25 bpw`) target lossless speculative drafters (`DFlash 2` / `DSpine`)
  and cold expert banks rather than uncalibrated dense trunks. Never compare across
  checkpoint bit-widths as same-model performance without teacher-KL and task
  qualification.

`emit_metal_and_swift_tables` in `Scripts/quantize/shvq.py` (and `emit_sub2_metal_and_swift_tables` in `Scripts/quantize/shvq_sub2.py`) emit the canonical Metal and Swift tables from saved binary assets (`shvq_tables.bin` for `SHVQ{2,3,4}` and `shvq_sub2_tables.bin` for `SHVQ{1,15}`), leaving existing `SHVQ{2,3,4}` assets bit-identical. See [tuning.md](tuning.md) for controlled performance evidence and exact engine-state qualification.

## Sub-2-bit PTQ gate & LittleBit (`arXiv:2506.13771`) qualification

Before shipping sub-2-bit containers in `Sources/`, Qwegn ran a controlled Gate 0
spike comparing 8D antipodal Hadamard spherical VQ (`SHVQ15`, `SHVQ1`) against
PTQ `LittleBit` (*Ultra Low-Bit Quantization via Latent Factorization*, NeurIPS 2025,
`arXiv:2506.13771` — multi-scale `S_row * U * V^T * S_col` binary latent factorization
refined via Joint-ITQ) on a 16-layer `[4096×12288]` SwiGLU cascade + `32K`-vocabulary
head (`KL(teacher || candidate)` in nats/token) and on `2,097,152` real `Qwen3.8-Flash-Next`
PLE rows:

| Format / Method | Effective bpw | 16-Layer Cascade Mean KL (nats/tok) | P95 KL | Top-1 Agreement | Engine Disposition |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `SHVQ4_G32` | `4.50` | `0.00052` | `0.00205` | `100.00%` | Shipped (dense/MLA trunk & MoE) |
| `SHVQ3_G64` | `3.25` | `0.00267` | `0.00931` | `100.00%` | Shipped (routed MoE `down_proj`) |
| `SHVQ2_G64` | `2.25` | `0.01694` | `0.05843` | `99.22%` | Shipped (routed MoE `gate_up`) |
| **`SHVQ15_G64`** (`1s + 11b` 8D) | **`1.75`** | **`0.04454`** | **`0.16518`** | **`99.22%`** | **Shipped (`DSpine`/`DFlash` & cold banks)** |
| **`SHVQ15_G128`** (`1s + 11b` 8D) | **`1.625`** | **`0.04690`** | **`0.17149`** | **`98.44%`** | **Shipped (`pack.py` / `pack_dflash.py` & cold banks)** |
| `SHVQ1_G64` (`1s + 7b` 8D) | `1.25` | `0.14199` | `0.59468` | `97.66%` | Shipped (lossless drafters & ultra-cold banks) |
| `LittleBit-PTQ` (`Dual-SVID + Joint-ITQ`) | `1.625` | `0.11306` | `0.45860` | `97.66%` | **Rejected** (`2.41×` worse KL + 2-pass latency) |
| `LittleBit-PTQ` (`Dual-SVID + Joint-ITQ`) | `1.00` | `0.24183` | `0.98929` | `97.66%` | **Rejected** (requires multi-billion-token QAT) |
| `LittleBit-PTQ` (`Dual-SVID + Joint-ITQ`) | `0.55` | `0.41920` | `1.76073` | `97.66%` | **Rejected** (collapses without QAT) |

Why `SHVQ15` beats PTQ `LittleBit` on Apple Silicon M5 Max:
1. **Full-rank geometry vs rank bottleneck:** Hadamard rotation (`FWHT_32`) Gaussianizes heavy-tailed LLM weights onto the hypersphere $S^7$, where `2048` antipodal 8D centroids (`4096` effective directions) preserve full matrix rank (`min(N, K)`). Without multi-billion-token QAT (`L2 -> L1` curriculum + teacher KD used in the LittleBit paper), binary low-rank `U * V^T` factors (`rank r << min(N, K)`) truncate the singular-value tail of pre-trained frontier weights (`2.41×` higher cascade KL at identical `1.625 bpw`).
2. **Single-pass UMA streaming vs two-pass barrier:** `SHVQ15_G64` (`0.0480 ms`) and `SHVQ1_G64` (`0.0367 ms`) stream weights once in a single fused Metal kernel with zero intermediate global activation buffers, whereas `y = S_row * (U * (V^T * (S_col * x)))` requires two sequential kernel launches (`K -> r -> N`) separated by a global synchronization barrier.
3. **PLE N-gram `VQ4_G32` (`2.50 bpw`) vs `VQ8_G32` (`1.50 bpw`):** On `Qwen3.8-Flash-Next` `ngram.bin` (`320,001,536` rows × `160D`), `QNGRAM v3 VQ4_G32` (`256` 4D centroids, `50 B/row`, `81` rows per `4096 B` sector = **`15.07 GiB`** vs `27.13 GiB` `VQ2_G32`, `-12.06 GiB` UMA) preserves **`0.9557` mean cosine similarity** (`rel_rmse = 0.2943`) and **`0.917×` (`+9.0%` faster)** `480×512` GPU gather latency (`0.4672 ms` vs `0.5088 ms`), whereas 8D `VQ8_G32` (`30 B/row`, `8.98 GiB`) dropped to `0.8262` cosine (`rel_rmse = 0.5640`) and was rejected.

## Native HQ quality investigation

`Scripts/probe_quality.py` replays the marker-enum and nested-array JSON tests
against a user-managed loopback server. It records request settings, output hashes,
finish reasons, and exact shape failures; failed checks exit nonzero. The prompt-only
test does not request `response_format`, and no probe requests logprobs (which would
disable Flash-Next speculation and change the execution being tested).

```bash
python3 Scripts/probe_quality.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen3.8-Flash-Next-hq8g32-hq4g64 --case marker \
  --output /tmp/marker-replay.json

PYTHONPATH=Scripts:Scripts/quantize .venv-oracle/bin/python \
  Scripts/quantize/hq_frontier.py \
  --pack ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --output /tmp/hq-size-frontier.json
```

The HQ tool forecasts trunk/expert bit-width and group-size combinations from
actual tensor headers, preserving BF16 exceptions. Estimates preserve the original
padding/header budget; they are not measured final pack sizes, KL scores, or a
quality recommendation. External PLE tables are unchanged and excluded from
`model.qwp` byte counts. `--source` checks a downloaded safetensors index and
payload extents and reports missing shards. Add `--verify-source-hash` to stream all
indexed shards in the packer's order and compare the aggregate checksum with the
incumbent pack's provenance. This explicitly reports checksum mismatches (exit 2);
the default extent-only audit does not certify checksums.
`--measurements` accepts measured bytes/KL with matching teacher and scoring-corpus
identities to calculate a two-objective Pareto frontier; it never selects a winner.

`Scripts/quantize/score_teacher_logits.py` scores verified full-vocabulary `.npy`
logit captures with matching integer token-prefix matrices (right padding is `-1`).
It reports `KL(teacher || candidate)` in nats/token, mean/p95/max, top-1 agreement,
and optional target-token NLL delta. It uses float64 CPU accumulation and rejects
diverged contexts, non-finite logits, or invalid targets. This is a capture scorer,
not a teacher forward pass: source completeness and oracle correctness must be
verified separately. Use `--help` for capture paths and optional candidate pack-byte
accounting. All three tools refuse to overwrite existing reports.
