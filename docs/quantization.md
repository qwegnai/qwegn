# Quantization Formats & Quality Tooling

## Supported formats

By default `qwg quantize` writes `HQ4_G32_M5`. It applies an orthonormal Walsh–Hadamard rotation to each 32-weight group, then stores 4-bit codes plus a scale and bias (5 effective bits per weight) in a layout the Metal kernels read directly. The engine also supports non-incumbent rotated `HQ{2,3,4,6,8}_G{32,64,128,256}` and standard-basis affine `AQ{2,3,4,6,8}_G{32,64,128,256}` packs, with row-paired batched verify (`M >= 4`), 4-row fused-FWHT gate/up, and batched `lm_head` kernels active across both incumbent `HQ4_G32` and non-incumbent `HQ4_G32` / `AQ4_G32` containers.

The full matrix — Hadamard-rotated `HQ{2,3,4,6,8}_G{32,64,128,256}`, affine `AQ{2,3,4,6,8}_G{32,64,128,256}`, Spherical e-means `SHVQ{2,3,4}_G{32,64,128,256}`, `QNGRAM v3 VQ2_G32`, `HQ4_G32_M5`, and `BF16` — supports mixed-format packs such as:

- `SHVQ4_G32` dense/MLA/MTP trunk + `SHVQ2_G64` routed `gate_up` (`2.25 bpw`) + `SHVQ3_G64` routed `down_proj` (`3.25 bpw`) (`2.69 bits/param`, **`100.63 GiB`** on `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` — `-33.74 GiB` / `-25.1%` vs `HQ4_G32_M5 + HQ3_G64`, fitting `321B` parameters 100% resident in 128 GB UMA).
- `HQ8_G32` dense trunk + `HQ4_G64` (`4.5 bits/weight`) routed experts (`4.75 bits/param`, `71.29 GiB` `.qwp` — `5.16 GiB` smaller with `2.62×` lower KLD vs BF16 than uniform `HQ4_G32`) paired with **`QNGRAM v3 VQ2_G32`** 2-bit e-means PLE N-gram embeddings (`90 B/row`, **`27.13 GiB`** vs `50.86 GiB` v2 `q8g32`, `-46.7%`) on `Qwen3.8-Flash-Next-hq8g32-hq4g64`.

## SHVQ lookup, SRAM staging, and small-prefill tiles

The Spherical Hadamard Vector Quantization family (`SHVQ{2,3,4}_G{32,64,128,256}`)
is supported across dense GEMV, head argmax, batched NAX GEMM, f32 BM/GEMV, and
grouped/routed MoE kernels:

- **SHVQ4 direct half codebook + `bm32` small prefill:** Routed f32 gate/up,
  down/combine, and grouped split-NAX MoE (`qwg_moe_nax_split_f32_shvq4_g*`) read
  the generated 32-KiB half-value codebook (`kShvq4HalfLutBits`) directly instead
  of reconstructing half values from the 16-KiB packed-u8 table on each lookup.
  For `SHVQ4_G32` and `SHVQ4_G64` on Apple GPU family 10+, `QWGemm.encodeHQ4F32BM`
  also dispatches exact 32-token/128-thread `bm32` tiles at 32 rows, or at 33--63
  rows when `K >= 2N` (with `epi == 2` restricted to 32 rows), matching the
  64-token reference float bit patterns across epilogues `0..3`.
- **SHVQ2/3 threadgroup SRAM codebook staging (`ShvqTgLut<2|3>`):** Because random
  256-entry `constant` address-space lookups serialize across simdgroups on AGX
  when indices diverge, dense decode/GEMV (`_dx_h16_t4`, `_v4`, `_v4_f32`), greedy
  head argmax, f32 GEMV (`_f32_v4`, `_f32_v4_am`, `_f32_v4_am2`), and routed MoE
  f32 gate/up + down/combine kernels cooperatively stage the `4 KiB` (`SHVQ2`) or
  `2 KiB` (`SHVQ3`) codebook into `threadgroup` SRAM once per threadgroup.
- **Pack-lane separation (`GLM-5.3-Flash`):** `HQ8_G32` dense/MLA/MTP trunk +
  `SHVQ4_G64` routed MoE experts is the fidelity-matched pack lane (still exceeding
  128 GB UMA and retaining cold-expert NVMe streaming). `HQ8_G32` trunk +
  `SHVQ2_G64` routed experts (`~85 GiB`) fits fully resident in 128 GB UMA as an
  explicit approximate-pack lane; never compare `SHVQ2` against `SHVQ4` or `HQ4`
  as same-model performance without separate teacher-KL and task qualification.

`emit_metal_and_swift_tables` in `Scripts/quantize/shvq.py` emits the canonical
Metal and Swift tables from saved assets. The table test checks all coordinates,
step-conversion identity and reproducible generated files. See [tuning.md](tuning.md)
for controlled performance evidence and exact engine-state qualification.

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
