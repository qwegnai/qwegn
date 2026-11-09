# Benchmarks

All numbers measured with `llmprobe v0.6.16` on an Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s DRAM, macOS 26) using `--kv-mode q4 --temperature 0`. Speculative runs use Variational MTP-3 (margin 2.5, hybrid drafting; DFlash 2 for the drafter row; exact NextN MTP-1 for `GLM-5.3-Flash`). Standard decode is serial (`--mtp-drafts 0`). Code eval is 50 HumanEval + 50 MBPP with `--reasoning off`.

Variational MTP relaxes only MTP-sourced draft rows, and only when the verifier gives the draft `p ≥ 0.2` (`--mtp-relaxed-min-prob`); context-copy drafts always verify exactly. On Flash-Next code eval this lifts margin-2.5 runs from 73–74 / 100 (14–15 runaway repetition loops) to 81–85 over 4 runs (0–1 loops), above exact MTP (79). With MTP-only relaxation, novel-text speculative decode measured 94.1 tok/s (three runs) versus 82.6 tok/s on `mlx-serve`; predictable-text decode measured 137.3 tok/s versus 159.2 tok/s. Relaxed greedy output is not bit-reproducible across runs because MTP chain depth adapts to measured round timings, so single eval runs vary by ±2–3 points.

## Qwegn Native Model Indicators

| Model pack (`~/.qwg/models`) | Architecture | Prefill (10.4K / 16K / 64K) | Cold TTFT | Warm cache TTFT | Standard decode | Speculative decode | 1M Prefill / Decode (LSA) | Needle (16K–1M) | 
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| `GLM-5.3-Flash` (`HQ4_G32_M5 + HQ3_G64`) | 45L MoE (`321B`/`18B`, 288 exp, 11 MLA + 34 KDA, 4× mHC) | **858–968 tok/s** @16K (`maxResident`) | — | MLA Key-Pool(4) snapshot | **31.6 tok/s** (`44.8` GPU proxy) | **34.8 tok/s** (NextN MTP-1 · `31.1` MoE-Spec) | 1M arch ready (`1,024 B/tok/L` KV) | **100% (conformance & needle)** |
| `Qwen3.6-35B-A3B-hq4g32` | 40L MoE (256 exp, 8 act) | **6,005 tok/s** (6,389 peak) | **45 ms** | 5 ms (55×) | **148.3 tok/s** | **233.4 tok/s** | **2,294 / 102.8 tok/s** | **3 / 3 (100%)** |
| `Qwen3.8-Flash-Next-hq8g32-hq4g64` | 48L MoE (512 exp, 10 act, `HQ8_G32 + HQ4_G64`) | 2,460–2,688 tok/s (2,696 peak) | 113–122 ms | 7–10 ms (59×–99×) | 72.3 tok/s | 106.7–113.7 tok/s (`142.4` pred) | 1,353 / 45.4 tok/s | **3 / 3 (100%)** |
| `Qwen3.8-27B-hq4g32` | 64L dense hybrid | 876 tok/s @64K (834 @10K) | 113 ms | 6 ms (307×) | 31.2–32.8 tok/s | 84.4 tok/s (MTP-3) | 805 / 43.1 tok/s | **3 / 3 (100%)** |
| `Qwen3.8-27B-hq4g32` + `DSpine-b16` | 64L dense + 5L block-causal drafter (`B=2..16`) | 832–876 tok/s | 94 ms | 7 ms | 32.1–32.9 tok/s | **100.5 tok/s geomean** (`134.5` peak, `4.19×`) | — | **3 / 3 (100%)** |
| `Qwen3.8-27B-hq4g32` + `DFlash2` | 64L dense + 5L diffusion drafter | 832 tok/s (869 peak) | 94 ms | 7 ms (266×) | 31.2 tok/s | 83.8 tok/s | — | **3 / 3 (100%)** |
| `SenseNova-Loop-9B-Preview` | 56L looped (32L, 2× loop, mHC) | 1,749 tok/s @64K | 64 ms | 7 ms (164×) | 48.9 tok/s | 93.9 tok/s | 1,476 / 46.5 tok/s | **3 / 3 (100%)** | 

All models execute with **0 heap and 0 Metal buffer allocations** in steady state across 64K–1M tokens.

## 321B Cold-Expert NVMe/GPU Streaming, NoPE MLA & Engine Tuning (`GLM-5.3-Flash`)

`GLM-5.3-Flash` (`321.32 B` parameters, `134.37 GiB` on disk in `HQ4_G32_M5 + HQ3_G64` at `3.59 bits/param`) exceeds a 128 GB Mac's unified memory by `6.4 GiB` before OS, KV, and scratch buffers. Qwegn executes it with zero OS swap through five coordinated subsystems:

| Subsystem / Optimization | Measured Gain / Metric |
| :--- | :--- |
| **`maxResident` memory policy (`16 GiB` OS reserve, `242/288` experts/layer = `84%` resident)** | `27.3 → 31.3–31.6 tok/s` serial decode (`+15%`), `858–968 tok/s` @16K prefill, `0` swap |
| **Dynamic per-layer expert LRU cache (`QW_EXPERT_CACHE=1`, GPU `expert_remap[42][288]`)** | Cuts cold SSD reads per token by **4×** (`60 → 15.5 misses/tok` at `K=216`) |
| **Event-pipelined cold staging (`ColdEventPipeline` + self-tuned FMA delay CB)** | Hides `~115 µs` `MTLSharedEvent` IOKit wake latency; overlaps GPU resident MoE with NVMe `pread` |
| **GPU prefill span map (`qwg_moe_span_ids`, `QW_PREFILL_SPAN_MAP=1`)** | Cuts 16K prefill host-GPU gap from `0.40 s` to `0.10 s` |
| **Apple10 NAX NoPE MLA & KDA kernels (`qwg_mla_sparse_attn_nax`, `_decode_nax_split`, `_absorb_q_mma`, `qwg_kda_gates_mma`)** | Prefill MLA `4–5×` faster; absorb-Q `340 → 50 ms`; KDA gates `218 → 60 ms`; decode MLA `2.23 → 0.44 ms/tok`; split indexer `8.4 → 0.92 ms/tok` |
| **64-TG Gram-matrix split 4-stream Sinkhorn-20 mHC (`qwg_mhc_router_split_f16`)** | Decode mHC `2.94 → 1.38 ms/tok` (`-53%`); prefill row-parallel mHC `-35%` |
| **NextN MTP-1 + 2-row `HQ3_G64`/`HQ4_G32_M5` verify (`qwg_aq_slice_dot2`) + MoE union deduplication** | **`34.8 tok/s`** speculative decode (`31.1 tok/s` with `--moe-spec-budget`), `75–90%` draft acceptance, `1.35–1.42×` verify cost, 100% bit-exact greedy parity |

## Dense speculative-verify optimizations (Qwen3.8-27B & general HQ4/AQ4)

The dense and looped models were re-measured after general verify-path changes, each gated by
geometry, row count or weight container (never by model name), with a 90–120 s idle period before every server run.
Serial decode is bandwidth-bound (GEMVs at 527–576 GB/s) and unchanged.

| Change | MTP-3 | DFlash 2 | MTP-5 |
| :--- | :---: | :---: | :---: |
| Row-parallel residual RMSNorm for verify rows | +2.2% | +3.2% | — |
| Row-paired batched HQ4/AQ4 verify GEMVs (M ≥ 4) | +6.3% | +6.6% | +20.0% |
| Row-paired DFlash drafter projections | — | +0.1% | — |
| Row-paired batched HQ4/AQ4 `lm_head` (M ≥ 4) | +0.4% | +0.4% | +1.2% |

Ratios are medians of five alternating cooled pairs (45–60 s cooldown). Except for the norm change,
every change is bit-identical across incumbent `HQ4_G32_M5` and non-incumbent `HQ4_G32`/`AQ4_G32` containers.
The norm change makes verify use the same numerics as serial decode, so speculative output follows serial output for longer.
Other models stayed neutral or improved: Flash-Next and serial decode on every model are bit-identical,
and extending row-paired verify + `lm_head` to non-incumbent `HQ4_G32` improved `SenseNova-Loop-9B-Preview` decode to 93.9 tok/s (102.4 pred @512).
Deeper drafts help 27B: variational hybrid MTP-5 is about 8% faster than MTP-3.

## Codebase Memory Pack (`QWMemoryPack`, `.qwg/*.qwcache`) Paired A/B Benchmarks

Measured on the `qwegine` repository (`294` tracked files: `7` Resident Cortex files + `287` Off-Context Vault files + 100% Architecture & Symbol Map = `32,742` pinned Cortex tokens, `32,790` total prompt tokens, 2 alternating rounds `A1 → B1 → A2 → B2` on Apple M5 Max 128 GB, `--temperature 0`). Full methodology and ReAct agent / REPL / live `<|memory_delta|>` scenarios are in [lsa-codebase-memory-whitepaper.md](lsa-codebase-memory-whitepaper.md).

| Model Pack | KV / LSA Mode | `.qwcache` Size (Build Time) | Cold 32.8K TTFT (R1 / R2) | Mounted `.qwcache` TTFT (R1 / R2) | TTFT Speedup | Effective Prefill (`tok/s`) | Live `<memory_delta>` TTFT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.6-35B-A3B-hq4g32` | `q4` / `sidecar` | `264.61 MiB` (`8.6 s`) | `6.55 s` / `6.53 s` | **`0.12 s` / `0.12 s`** | **`54.4×`** | `5,020` → **`290,645 tok/s` (`57.9×`)** | **`0.40 s`** (`86,715 tok/s`) |
| `Qwen3.8-Flash-Next-hq8g32-hq4g64` | `q4` / `sidecar` | `464.54 MiB` (`16.7 s`) | `17.84 s` / `19.83 s` | **`0.24 s` / `0.24 s`** | **`74.3×`** | `1,838` → **`138,391 tok/s` (`75.3×`)** | **`0.60 s`** (`46,620 tok/s`) |

### Two-Tier LSA Cortex (`32.7K` tok) + Off-Context Vault (`293` Files) Multi-Turn Benchmarks (`QWMemoryVault`)

Comparing the **Monolithic 1M Linear Pack** (`1,008,064–1,011,738` tokens) against **Two-Tier LSA Cortex + Off-Context Vault** (`32,745` Cortex tokens across `9` Tier-0 files + `284` Off-Context Vault files + 100% Symbol Map) across 4-turn `qwg repl` and `qwg serve` sessions on `Qwen3.6-35B-A3B-hq4g32` (`kv=q4`, `lsa=sidecar`):

| Metric / Multi-Turn Gate | Monolithic 1M Linear Pack | **Two-Tier LSA Cortex + Vault (`QWMemoryVault`)** | Measured Gain |
| :--- | :---: | :---: | :---: |
| **`.qwcache` Size (`35B-A3B` `q4` / `Flash-Next` `q4`)** | `6,282.9 MB` / `10,958.7 MiB` (`73.0 s` mount) | **`264.61 MiB` / `464.57 MiB` (`2.81 s` boot+mount)** | **`23.7×` smaller (`26.0×` faster mount)** |
| **Free Context Window for Coding Agents (`1M` Ceiling)** | `36,838` tok (`3.6%`, fails at `170K` agent prompt) | **`1,015,831` tok (`96.9%`) + Elastic Budget Guard** | **`27.6×` more free agent context (`0` overflows)** |
| **Turns 1–2 (Resident Cortex: `Makefile`, `docs/lsa.md`)** | `0.08–0.35 s` TTFT · `91.9–101.6 tok/s` decode | **`0.10–0.15 s` TTFT (`repl`) · `102.7–143.4 ms` (`serve`) · `153.8–154.2 tok/s` decode** | **`+53%` faster decode (`96.8–99.2%` MTP accept)** |
| **Turns 3–4 (Off-Context Vault Recall: `QWSlotPinnedLRUStore`, `QwgIndex`)** | N/A (monolithic linear prefix) | **`< 0.5 ms` Vault recall · `100%` `[warm_ram]` hit on prior turns (`40,550/44,054` tok) · `147.5–163.7 tok/s` decode** | **`11.8×` lower warm RAM (`390–533 MB`, `0` SSD spill)** |

## Reproducing these numbers

For thermal-controlled A/B measurements, both `Scripts/paired_perf.py` and
`Scripts/paired_http_perf.py` accept `--cooldown-seconds 60`. They alternate
baseline/candidate order and record the idle interval in each JSON report.

```bash
make perf   # automated performance regression checks against Scripts/bench/baselines.json
```
