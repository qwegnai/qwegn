# Engine Tuning & Memory Policy

> For the complete user-facing knob catalogue, Menu Bar guide, and copy-paste operational recipes across workloads, see **[settings-reference.md](settings-reference.md)**.

## Unified-memory residency

| Flag / env | Default | Description |
| :--- | :--- | :--- |
| `--memory-policy` / `QW_MEMORY_POLICY` | `maxResident` | Unified-memory residency policy (`maxResident` sizes to physical RAM minus OS reserve; `safe` sizes to free/reclaimable memory minus `6 GiB` headroom, zero swap). |
| `--os-reserve` / `QW_OS_RESERVE_GIB` | `8.0` GiB | RAM reserved for macOS and desktop apps. |
| `--resident-experts-per-layer` / `QW_RESIDENT_EXPERTS_PER_LAYER` | `0` *(auto)* | Pin hot resident experts per MoE layer (`0` sizes automatically from `--memory-policy`). |
| `QW_EXPERT_CACHE` | `1` *(on)* | Dynamic per-layer GPU expert LRU cache. |
| `QW_COLD_EVENT_OVERLAP` | `1` *(on)* | Event-pipelined NVMe/GPU cold-expert overlap (`ColdEventPipeline`). |
| `QW_PREFILL_SPAN_MAP` | `1` *(on)* | GPU prefill span map (`qwg_moe_span_ids`). |
| `--evict-prefill-scratch` | `auto` | `auto` evicts on Macs `<= 36.5 GB` RAM or when wired engine arena `>= 75%` of physical RAM (`run`, `repl`, `serve`); `always` or `never`. |
| `--warm-cache-limit` / `--ssd-cache-limit` | `4` / `10` GiB | Prefix-cache budgets (`serve`). Warm RAM is dynamically clamped to `<= 25%` of post-load `QWResidencyAdmission` headroom. |
| `--prefill-chunk-size` | `512` | Chunk size (`32...8192`). The general driver caps chunks at 4096 rows; Flash-Next uses balanced chunks of up to 8192 rows. |
| `--load-vision` / `--no-load-vision` | `false` | Eagerly wire or omit multimodal ViT planes at startup (`-1.60 GiB` CPU RSS when omitted). |
| `--stream-ngrams` / `--no-stream-ngrams` | `false` | Stream external PLE N-gram tables (`ngram.bin`: `27.13 GiB` `VQ2_G32` or `15.07 GiB` `VQ4_G32`) from NVMe (`64 MiB` L1 cache) vs wiring into Metal RAM when admissible. |
| `--kv-mode` | `q4` | Default 4-bit KV cache (`4×` smaller than `f16`, `.qwcache` compatible); `q8` and `f16` remain selectable. |
| `--lsa-mode` | `auto` | Adaptive sparse attention: exact dense `<= k+516`, `.qlsa`/Weight-SVD on GQA, flat QSA `<= 262K` + `e-means-4096` `> 262K` on QSA, and `< 41K` flat bypass + `>= 41K` `e-means` coarse pruning on MLA (`--lsa-mode off` forces dense / `exactFlat`). |
| `--no-profile` | `false` | Disables implicit per-model profile overrides (`run`, `repl`, `serve`) for hermetic A/B comparisons. |

Defaults come from `QWKnobRegistry`. Per-model profiles and machine size can override
them unless `--no-profile` is passed; for example, MoE models on Macs over 36.5 GB use
4096-row prefill chunks. `QWResidencyAdmission` unifies live `vm_statistics64`
admission (`effectiveAvailableBytes(includeReclaimable:)`) across CLI load guards,
`DriverConfig` MoE expert residency, and `qwg serve` `PrefixCache` warm-tier clamping.

## Controlled same-pack comparisons

`Scripts/paired_perf.py --strict` requires explicit KV mode, context, prefill chunk
size, sampling/seed, MTP settings, LSA mode, thinking, and disabled prefix caching
in `--options-json`. With positive drafts, explicitly use the standard MTP backend;
with zero drafts, omit `--mtp-backend` (the CLI rejects an explicit backend).
Use zero relaxed margin. Different pack names can select different saved profiles: identical
partial CLI options do not guarantee comparable settings.

Strict runs hash the model and its assets (including PLE) before and after the
experiment, check effective KV/context from each process, and exit nonzero with
`status: incomparable` on parity, allocation, or configuration mismatches. Hashing
large packs adds I/O outside the timed samples. Non-strict reports remain raw
measurements, not a performance or fidelity verdict. Neither mode proves that the
GPU is idle: record quiet-machine and memory admission evidence separately.
Output hashes cover decoded stdout, not token IDs or intermediate state; they are
a necessary consistency check, not a replacement for bit-exact engine golden gates.

### Benchmark-controls ledger

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| Paired CLI harness | Strict comparisons are opt-in; preserve existing raw measurements. Refuse mismatched effective KV/context or unstable outputs. | 13 Python tests, `make lint`, `make test` and `make ci` pass; live identical-release A/A passes. No engine speedup is claimed. |

## Flash-Next submission diagnostics

`QW_FLASH_TRACE=1` writes one `[flash-execution]` JSON record to stderr for each
decode/verify step and prefill chunk. It records actual host/GPU PLE placement,
mapped PLE buffer bytes, active rows, position, MTP presence, layer ranges and
command-buffer timestamps without changing encoders, barriers or submission cuts.
Mapped bytes are not a physical-residency measurement or a full memory budget.

Times are milliseconds relative to stack entry, using the same monotonic clock
as Metal command-buffer timestamps. For each buffer, `submitMS` is sampled just
before ending its encoder and committing: `gpuStartMS - submitMS` includes that
host submission work as well as queue latency. `submitMS - encodeStartMS` includes
explicit host PLE waits; subtract `pleWaitMS` when aggregating CPU encoding.
`plePrepareMS` covers initial row-ID preparation or launching host staging, not
the whole asynchronous read. Prefill's event wait is included in GPU buffer time.
`nextPLEPrepareMS` covers pipelined preparation of the next prefill chunk.

CPU encoding, GPU busy time and completion waits overlap; never add them as
independent stage costs. Retain individual intervals to compute the GPU span,
idle gaps, and time to first submission. `stackToGPUCompleteMS` excludes token
readback and trace serialization. Initialization can emit a cold verify record
at position zero even for serial generation; exclude it from steady decode.
Missing/invalid GPU timestamps fail the diagnostic explicitly.

Trace logging perturbs execution. Use it for attribution and route checks only;
disable it for final alternating A/B measurements and keep golden/state gates.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| Flash-Next trace | Keep diagnostics opt-in and retain raw, overlapping CPU/GPU intervals. | 15 targeted tests and full lint/serial tests pass; live serial and exact-MTP trace-on/off output checks pass. |

## Flash-Next decode submission policy

Decode and multi-row verification use the existing early layer cuts
`[0, 1, 8, layerCount]` for stacks larger than eight layers, independent of host/GPU
PLE placement. Small stacks retain one command buffer. GPU PLE gather stays in
the first buffer with its existing buffer barrier; host PLE waits remain attached
only to the first consuming chunk. No arithmetic or kernel dispatch changes.

The same-pack, Q4-KV release comparison on the 128-GiB M5 Max used five alternating
pairs, two trials per arm, greedy decoding and disabled prefix caching:

| Workload | Single-buffer / early-submit decode medians | Paired throughput ratio / MAD |
| :--- | :--- | :--- |
| 63-token prompt, 128 serial tokens | 55.055 / 56.680 tok/s | 1.028950 / 0.001300 |
| 4597-token prompt, 128 serial tokens | 49.220 / 50.700 tok/s | 1.029072 / 0.000906 |
| 63-token prompt, 128 exact-MTP tokens (one neural draft) | 74.685 / 76.605 tok/s | 1.025367 / 0.001555 |

These are measured decode gains for these workloads, not native-context or
general quality claims. Prefill changes were within paired dispersion. The
larger prompt is a repeated diagnostic, not a held-out task corpus. Process-load
wall times were noisy and are not an optimization claim.

The synthetic VQ2 test checks consumer output bit patterns across buffer cuts.
A separate sequential real-pack gate compared token IDs, session history/pending
tokens, acceptance counters and full GDN/activation/KV arena SHA-256 values at
15 checkpoints across serial and speculative runs, including rejected drafts.
All matched. No tolerance changed and no unit test loads a real pack.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| Flash-Next decode cuts | Retain early submissions for GPU PLE; do not change prefill or arithmetic. | 17 targeted tests, full `make lint`, `make test`, `make ci` and release build pass with zero compiler warnings; 15 full-state checkpoints match. Independent read-only review: `VERDICT: PASS`. |

### Operator-attribution follow-up

A standalone release-linked probe of this same mixed-format pack found
SHVQ4-G64 routed expert banks, but HQ8-G32 HC, shared-expert, GDN, attention
projection and PLE banks. Do not optimize a generic SHVQ HC/dense route and
assume it improves this pack: qualify the actual bank and dispatcher first.

With synthetic inputs and repeated hot operators, the PLE-bearing layer's
two-row MoE measured 151.68 us (MAD 0.21), including isolated routed gate/up
64.39 us (0.31) and down/combine 48.93 us (0.08). Its GDN input projection
measured 61.77 us (0.14), HC mix 22.13 us (0.05), and PLE key/value projection
46.88 us (0.11). These are attribution diagnostics, not additive production
stage percentages or performance-gain claims. The first one-row whole-MoE
samples had high dispersion and are not a stable baseline.

The probe also exposed a small-prefill route cliff: this layer's 32-row GDN
input projection took 474.65 us versus 109.22 us at 64 rows. The dispatcher
changes from f32 batched multiplication to f16-input NAX at that boundary;
simply lowering the threshold changes arithmetic and is not an exact fix.
Prioritize routed-expert unpack/tiling experiments and exact small-prefill
kernels, then require same-pack end-to-end A/B and full-state parity again.
This probe does not cover attention selection, recurrence, cold experts,
native-context memory capacity, or task quality.

### Captured routes and rejected expert reuse

`QW_FLASH_ROUTE_TRACE=1` emits `[flash-routes]` JSON after decode/verify completes.
Each record identifies phase, position, active rows, layer, expert count and
row-major route IDs. Only PLE-bearing layers and the last layer are captured:
other layers share scratch, whose previous route IDs have already been overwritten.
This is partial route coverage, not an all-layer routing profile. Trace output is
diagnostic-only; disable it for performance runs.

The real-pack operator gate captured 676 route records from layers 1 and 47
(zero-based), with one/two active rows, while all 15 token/session/full-arena
checkpoints matched the untraced release exactly. Replaying six distinct
two-row route patterns with deterministic synthetic activations rejected the
existing expert-reuse switch: gate/up was slower, and down/combine differed in
647--1345 output float bit patterns per captured case. Four-row repeated,
disjoint and pruned-route diagnostics also rejected broad enablement. Keep
Flash-Next's `reuseExpertsAcrossTokens: false`; no approximation was enabled.
The final six-pair replay measured baseline/reuse speed ratios of 0.554--0.634
for captured gate/up patterns and 0.670--0.691 for down/combine.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| Flash-Next route capture | Keep partial route diagnostics opt-in; reject blanket expert reuse. | 17 targeted tests, full `make lint`, `make test`, `make ci` and release build pass without compiler warnings; 15 full-state checkpoints match. Six alternating pairs per case preserve both raw trials and min-of-two samples. |

### Exact SHVQ4 routed unpack

Routed f32 gate/up and down/combine now look up canonical half values directly,
instead of unpacking four u8 codebook coordinates, converting to f32, scaling
and rounding to half on every lookup. The generated table is 32 KiB rather
than 16 KiB. It changes no pack bytes, signs, group scales, reduction order,
routing, scratch or command-buffer schedule. Dense/HC/NAX paths and SHVQ2/3
retain their previous lookup; this is not a dense-format conversion.

An exhaustive GPU comparison of every 16-bit code/sign payload matched all
262,144 half components. On actual SHVQ4-G64 banks, 24 paired operator
comparisons retained both raw trials for six alternating pairs. All float
output words matched, including pruned routes. Captured two-row gate/up
baseline/candidate ratios were 1.2079--1.2195 (MAD 0.0010--0.0049);
down/combine ratios were 1.3659--1.3868 (MAD 0.0005--0.0037).
These cache-hot operator results use synthetic activations and are not
additive end-to-end stage percentages.

The final release's same-pack Q4-KV comparison on the 128-GiB M5 Max used five
alternating pairs and two trials per arm, greedy decoding, fixed contexts,
GPU PLE, disabled tracing/prefix caching and unchanged effective settings:

| Workload | Packed / direct lookup decode medians | Paired throughput ratio / MAD |
| :--- | :--- | :--- |
| 63-token prompt, 128 serial tokens | 56.900 / 59.170 tok/s | 1.041065 / 0.001276 |
| 4597-token prompt, 128 serial tokens | 50.710 / 52.490 tok/s | 1.034822 / 0.003324 |
| 63-token prompt, 128 exact-MTP tokens (one neural draft) | 76.670 / 80.530 tok/s | 1.050867 / 0.000190 |

These are min-of-two decode summaries, with all trials retained. Short-prompt
prefill and TTFT were the same speed; no prefill gain is claimed. The larger
prompt was repeated once with two warmup pairs because both arms had substantial
throughput drift. The table uses that repeat; the initial decode ratio was
1.034696 (MAD 0.000261). Across all raw trials, larger-prompt prefill ratios were
0.986469/0.983581 with MAD 0.041307/0.038398. An identical-baseline A/A control
measured prefill ratio 1.000041 with MAD 0.027161: the A/B prefill deltas are
inside observed noise, not evidence of a gain or a separable regression.
Before/after executable, metallib and pack/asset hashes were stable; idle/idle and reclaimable-memory
admission passed. Fifteen real-engine token/session/full GDN/activation/KV
checkpoints matched the prior release exactly, including speculative rejection.
No tolerance or model-quality trade-off was introduced.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| SHVQ4 routed unpack | Use the exact direct half lookup only in routed f32 kernels; keep dense precision and expert reuse unchanged. | Five quantizer/table tests, five targeted SHVQ GPU tests, release build, full `make lint`, `make test` and `make ci` pass without compiler warnings. Exhaustive code/sign parity, 24 actual-bank operator comparisons and 15 full-state checkpoints are bit-exact; decode gains exceed paired dispersion. |

### Exact 32-row prefill tile

The initial generic HQ8-G32 f32 BM qualification selected a 32-token/128-thread
tile for exactly 32 rows, no epilogue, and Apple GPU family 10 support.
The geometry-qualified extension below additionally covers selected 33--63-row
projections. Other formats, incumbent containers, epilogues and hardware retain
the existing 64-token/256-thread path. The measured M5 Max reports family 10, not family 11;
qualification checked the live capability and the public encoder, rather than
assuming the discovery kernel was selected.

The tile removes unused token work while retaining f32 inputs and the same
float8x8 accumulation order. It does not lower the f16 NAX threshold or change
weights, reductions, KV mode or scheduling. The original 64-row weight-address
hoisting/load block is preserved: moving it into a generalized staging loop
caused a measurable regression during qualification and was rejected.

Actual-bank, cache-hot operator comparisons used six alternating pairs,
two raw trials per arm and min-of-two samples:

| 32-row projection | Original / selected tile | Paired speed ratio / MAD |
| :--- | :--- | :--- |
| GDN QKV+Z | 467.83 / 248.97 us | 1.87890 / 0.00694 |
| GDN output | 277.01 / 187.16 us | 1.48027 / 0.00049 |
| HC down | 258.21 / 249.81 us | 1.03331 / 0.00065 |
| HC up | 38.84 / 23.15 us | 1.67679 / 0.00147 |

All 12 output comparisons, including protected 48/63-row cases, were bit-exact.
Protected-case ratios ranged from 0.99769 to 1.00098, inside paired dispersion.
The discovery tile made HC up slower at 48/63 rows; it is deliberately not
selected there, despite isolated GDN gains at those sizes.

The same SHVQ/VQ2 pack on the 128-GiB M5 Max used Q4 KV, context 4096,
chunk 512, greedy sampling, GPU PLE and disabled tracing. Five alternating
engine pairs followed a warmup pair, with two trials per arm:

| Engine workload | Original / candidate median | Paired speed ratio / MAD |
| :--- | :--- | :--- |
| 32-token prefill | 110.268 / 94.129 ms | 1.171709 / 0.000593 |
| 63-token prefill, protected | 127.782 / 127.786 ms | 1.000510 / 0.001894 |
| 127 decode steps after 32 tokens | 2.14284 / 2.14650 s | 0.998787 / 0.002065 |
| 127 decode steps after 63 tokens | 2.14961 / 2.15333 s | 0.999683 / 0.000913 |

The 32-token result is 290.203 to 339.958 tok/s, a paired throughput gain
of 17.17% (14.64% lower median latency). Protected prefill and decode are the same
speed within noise. These synthetic valid-token prompts measure the engine's
prefill/head call, including any first-use pipeline lookup, but exclude model
loading, prefill preparation, chat rendering and tokenization. This is not a
general long-prompt, CLI TTFT or task-quality result, nor a new identical-arm
A/A control. No MTP performance gain is claimed for this tile.

Twenty-five real-model token/session/full GDN/activation/KV checkpoints matched
at 31/32/33/63-row and exact-MTP boundaries. Both weight bases and epilogues
also passed synthetic exact-bit tests without loading model packs in XCTest.
Executable, metallib and pack/asset hashes were stable; every timing trial
passed idle/idle and reclaimable-memory admission. An immediate next-process
admission initially refused delayed Metal memory reclamation; those incomplete
runs were not used. The final harness verifies the pack after each process and
requires fresh admission for the next, without weakening guards or sleep polling.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| HQ8-G32 initial small prefill | Initially select the exact 32-token tile only for 32 rows, epi0 and Apple10; preserve the original 64-row load path. | Six targeted GPU tests, full `make lint`, `make test`, `make ci` and release build pass without compiler warnings. Twelve actual-bank comparisons and 25 full-state checkpoints are exact; five engine pairs separate the narrow prefill gain from noise. |

Raw trials, refused admissions, rejected variants, frozen binaries/libraries,
input identities and full-state snapshots remain with the session experiment
artifacts. Non-M5 performance, native-context capacity, a blanket 33--63-row
selector, `make perf` and held-out task evaluations were not qualified.

### Geometry-qualified 33--63-row prefill

The same kernel is additionally selected at 33--63 rows when `K >= 2N`,
where `N` is the output width and `K` the reduction width. The existing
exactly-32-row selection, generic HQ8-G32 and Apple10 gates remain. This
qualification used epi0; the fused-epilogue extension is documented below.
Two 32-token tiles preserve every output's accumulation order. Expansion
shapes, including the previously regressing HC-up shape, and square matrices
retain the original 64-token kernel. No model-specific dimensions, precision,
weights, kernels, KV behavior or scheduling are changed by this extension.

All five matching actual shape classes in the pack were tested at every
integer row count from 33 through 63. Discovery and final public-dispatch
runs each made 155 comparisons, using six alternating pairs, two trials/arm,
min-of-two samples and two warmup rounds. All float words matched exactly.
Speed-ratio ranges over those 31 row counts were:

| Actual matrix shape, N x K | Final public-dispatch speed ratio range |
| :--- | :--- |
| 48 x 2560 | 1.05472--1.09898 |
| 320 x 10240 | 1.03942--1.07954 |
| 512 x 2560 | 1.05822--1.10493 |
| 640 x 2560 | 1.28430--1.30553 |
| 2560 x 6144 | 1.31949--1.33521 |

These are cache-hot geometry probes, not additive stage percentages or a claim
that every representative tensor uses this encoder in the live engine.
For example, HC down/up use epi3/epi2 below 64 rows and were unchanged in
this qualification.

Against the prior 32-row-only release, the same pack and engine settings used
five alternating A/B pairs after a warmup pair, two raw trials/arm. A separate
identical-baseline A/A control used three alternating pairs after a warmup pair:

| Actual prompt tokens | Baseline / candidate prefill | A/B speed ratio / MAD | A/A speed ratio / MAD |
| :--- | :--- | :--- | :--- |
| 33 | 113.744 / 108.852 ms | 1.045127 / 0.001932 | 1.002997 / 0.000018 |
| 48 | 122.407 / 117.422 ms | 1.041910 / 0.002540 | 1.000520 / 0.000960 |
| 63 | 128.476 / 123.625 ms | 1.040033 / 0.001531 | 1.000231 / 0.000985 |
| 32, protected | 94.965 / 94.670 ms | 1.003118 / 0.000866 | 1.001459 / 0.001042 |
| 64, protected | 82.167 / 82.032 ms | 1.001529 / 0.001183 | 1.001093 / 0.000294 |

The qualified prompts gain 4.00--4.51% paired prefill throughput, exceeding
paired dispersion and A/A noise. Their arm-median throughput changes are
290.126 to 303.165, 392.136 to 408.780, and 490.363 to 509.606 tok/s.
Protected 32/64-row prefill is the same speed; no additional gain is claimed.
Decode paired medians were 0.08--0.16% slower, while identical-arm decode
median ratios spanned 0.999828--1.003315 and raw latency ranges per workload
were 0.355--0.788%. This experiment does not resolve a decode change.

Timing has the same synthetic-prompt and engine-call boundaries described
above; these are not CLI TTFT, long-context or MTP-throughput claims.
Thirty real-model full-state checkpoints matched at serial 32/33/48/63/64
and exact-MTP63 boundaries. Synthetic tests cover both bases, aspect-ratio
boundaries, expansion/square fallbacks, partial tiles and rows 31/32/33/48/63/64/65.
Every executed model trial passed idle/idle and memory admission; binaries,
identical metallibs and pack/assets remained hash-stable. One non-idle
admission stopped A/B before a trial started. Its completed raw slots were
retained and the remaining slots resumed only after fresh admission and
matching artifact identities; no completed samples were rerun or discarded.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| HQ8-G32 reduction-heavy small prefill | Extend bm32 to 33--63 rows only when K >= 2N, retaining epi0/Apple10/format gates and all expansion fallbacks. | Six targeted GPU tests, full `make lint`, `make test`, `make ci` and release build pass without compiler warnings. All 155 actual-bank comparisons and 30 full-state checkpoints are exact; five A/B pairs exceed the separate three-pair A/A noise floor. |

Raw discovery/production sweeps, A/B including its refusal/resume, A/A, full
snapshots, hashes and gate logs remain in the session artifacts. No independent
review verdict, `make perf`, native-context/capacity, held-out task-quality or
non-M5 performance qualification is claimed for this selector.

### Fused-epilogue small-prefill selection

The same row/aspect policy now also accepts epilogues 1--3: scaled SiLU,
sigmoid, and scaled SiLU plus H_32. Format and Apple10 gates are unchanged.
This only changes kernel selection; the existing bm32 implements the same
epilogue arithmetic and output-lane mapping. HC down (epi3, scale4) is eligible
at 32--63 rows; HC up (epi2) is eligible only at 32 rows. HC up still uses the
original tile at 33--63 rows, where discovery found about a 4.7% slowdown.
No new kernel, reduction order, precision, weight bytes or scheduling change
was introduced.

Discovery and final public-dispatch sweeps each tested 460 actual-bank cases:
every integer row count 32--63 on the five reduction-heavy shape classes,
epilogues 1/2/3 (H_32 only for aligned output widths), plus HC up at
32/33/48/63. Six alternating measured pairs followed two warmup rounds,
with two raw trials/arm and min-of-two samples. All float words matched.
Public-dispatch HC-up32 sigmoid measured a 1.67937 speed ratio; protected
HC-up33/48/63 remained within operator dispersion. Reduction-heavy epilogue
ratios ranged from 1.03151 to 1.51962.

Real-engine qualification used the same pack, Q4 KV, context4096, chunk512,
greedy sampling and GPU PLE as above. Five alternating A/B pairs and a separate
three-pair identical-baseline A/A control each followed one warmup pair,
with two trials/arm:

| Actual prompt tokens | Paired A/B prefill speed ratio / MAD | Paired A/A prefill speed ratio / MAD |
| :--- | :--- | :--- |
| 32 | 1.028490 / 0.006443 | 0.996378 / 0.002378 |
| 48 | 1.020704 / 0.001773 | 0.999423 / 0.001110 |
| 63 | 1.018473 / 0.001692 | 0.998994 / 0.001667 |
| 31, protected | 0.997798 / 0.002172 | 0.995887 / 0.006909 |
| 64, protected | 1.000350 / 0.000900 | 0.994689 / 0.000906 |

Paired prefill gains are 2.85%, 2.07% and 1.85%. Maximum absolute raw A/A
prefill-ratio deviations for those prompts were 1.066%, 1.593% and 1.546%:
the gains exceed observed variation, but the 63-token margin is narrow.
These are not additive to earlier percentages or general prompt-length claims.
The benchmark retains the synthetic-prompt and engine-call timing boundaries
described above, not CLI TTFT or task-quality qualification.

The split/resumed batch exhibited a common time-dependent shift: decode
moved from roughly 2.14 to 2.21 seconds in both arms, while protected64
prefill moved from roughly 82.3 to 84.9 ms. Its cause was not established.
Unpaired arm medians therefore misrepresent the paired effect; use the table,
not ratios of separately pooled medians. Protected prefill and decode changes
were unresolved within the A/A variation; no decode gain is claimed.
All successful samples, including drift/outliers, were retained.

Thirty-five real-model token/session/full GDN/activation/KV checkpoints matched
exactly at serial31/32/33/48/63/64 and exact-MTP32 boundaries. Synthetic tests
cover both bases and all epilogues, using scale4 for the HC-down epilogue.
Every executed model trial passed idle/idle and memory admission; pack/assets,
release executables and identical metallibs were hash-stable. Refused non-idle
admissions stopped runs before any sample executed. Completed slots were
preserved and resumed only with matching identities and fresh admission;
the selector was not enabled while qualification was incomplete.

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| HQ8-G32 fused-epilogue small prefill | Permit epi0--3 with the existing 32-row or reduction-heavy33--63-row policy; keep HC-up expansion fallback above32. | Six targeted GPU tests, full `make lint`, `make test`, `make ci` and release build pass without compiler warnings. 460 actual-bank comparisons and 35 full-state checkpoints are exact; five A/B pairs exceed the separate three-pair A/A variation. |

Raw sweeps, all resumed A/B and A/A trial slots, refusals, full-state snapshots,
hashes and gates remain in session artifacts. No independent review verdict,
`make perf`, native-context/capacity, held-out task-quality or non-M5 performance
qualification is claimed.

### Dense SHVQ: storage versus throughput

For this 67.706-GiB pack, the 597 two-dimensional HQ8-G32 planes total
4.538 GiB, including the head and MTP. Replacing these payloads with SHVQ4-G64
would save 2.395 GiB (3.54% of the pack); SHVQ3-G64 would save 2.900 GiB
(4.28%). Excluding the head, savings are 2.044/2.474 GiB. These are payload
projections, not a rebuilt pack or measured resident-memory reduction; PLE
ngram assets, KV, scratch, raw BF16 and expert banks are unchanged.

A separate shape-matched experiment requantized actual HQ8 GDN input/output
and HC down/up planes without changing the source pack. Six alternating
pairs, two trials per arm and eight operator repetitions per buffer compared
the production dense projection encoders. Selected HQ8/SHVQ4 median times:

| Dense primitive | Rows | HQ8 / SHVQ4 microseconds | Paired speed ratio / MAD |
| :--- | ---: | :--- | :--- |
| GDN input QKV, 10240 x 2560 | 512 | 432.83 / 498.72 | 0.8672 / 0.0009 |
| GDN output, 2560 x 6144 | 1 | 20.29 / 18.74 | 1.0822 / 0.0015 |
| GDN output | 2 | 22.46 / 27.21 | 0.8263 / 0.0040 |
| GDN output | 512 | 272.47 / 325.03 | 0.8382 / 0.0003 |
| HC down, 320 x 10240 | 2 | 15.10 / 33.87 | 0.4457 / 0.0003 |
| HC up, 10240 x 320 | 2 | 10.15 / 24.51 | 0.4154 / 0.0022 |

Ratios above one favor SHVQ. These repeated hot-operator measurements use
synthetic activations, not a production workload. HC rows here measure plain
projection primitives, not the fused HC norm/injection/mix path; QKV here is
not the combined QKV+Z bank. The first one/two-row QKV comparisons had high
dispersion and are excluded from recommendations. Both raw trials and selected
min-of-two samples are retained; each run passed idle/idle and reclaimable-memory
admission with unchanged input/executable/metallib hashes. No end-to-end gain
is inferred.

SHVQ3 was generally less slow than SHVQ4, but still regressed most measured
multi-row/prefill primitives. Both formats lost on all four 512-row shapes.
Relative weight RMSE against the existing HQ8 planes was 8.1--8.6% for SHVQ4
and 15.7--16.3% for SHVQ3. Synthetic output RMSE was comparable, but neither
weight nor output error is task-quality qualification; requantizing HQ8 is
also not equivalent to quantizing the original BF16 checkpoint.

Recommendation: retain HQ8 dense/HC defaults. Dense SHVQ is a potential
memory-capacity trade-off, not a general speed optimization with the current
kernels. If capacity actually requires it, qualify individual banks from the
original checkpoint with teacher-forced/task evaluations and serial/exact-MTP/
prefill measurements. The isolated one-row GDN-output benefit is insufficient
to justify a default change that slows multi-row verification and prefill.

## Cold-expert streaming

For models larger than unified memory (e.g. the 321B `GLM-5.3-Flash` pack at
`134.37 GiB` on a 128 GB Mac), Qwegn streams cold experts from NVMe with zero OS
swap through five coordinated subsystems: the `maxResident` memory policy
(`242/288` experts/layer resident), the dynamic per-layer expert LRU cache,
event-pipelined cold staging (`ColdEventPipeline` hiding the `~115 µs`
`MTLSharedEvent` wake latency), GPU prefill span maps, and NextN MTP with
2-row batched verify. Measured gains per subsystem are tabulated in
[benchmarks.md](benchmarks.md#321b-uma-resident-shvq-1006-gib--cold-expert-nvme-streaming-1344-gib-glm-53-flash).

## Speculative decoding knobs

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--mtp-backend` | `variational` | `standard`, `variational` (Voronoi log-margin), `dflash2` (`B=2..8`), or `dspine` (`B=2..16`). |
| `--mtp-draft-strategy` | `hybrid` | `hybrid` (MTP + prompt lookup), `neural` or `ngram`. |
| `--mtp-drafts` | `1` | Drafts per verify step (`0` = serial; up to `15` with `dspine`). |
| `--mtp-relaxed-margin` | `1.5` | Variational log-margin δ (`2.0–2.5` works well for code and chat). |
| `--mtp-relaxed-min-prob` | `0.2` | Variational fidelity floor τ: a draft accepted through the margin must also hold `p ≥ τ` under the verifier, which stops flat rows from waving through degenerate repeats (`0` disables it). Only MTP-sourced rows are relaxed; context-copy drafts verify exactly. |
| `--dflash-draft-pack` / `--dflash-window` | — | DFlash 2 / DSpine drafter pack (`HQ*`, `AQ*`, or `SHVQ{1,15,2,3,4}` via `pack_dflash.py --quant`) and window B in `2...16` (auto-detects `dspine` vs `dflash2` from drafter `config.json`). |
| `--moe-spec-budget` | `0` *(off)* | MoE-Spec routed expert budget per verify layer (`0...8`, or `QW_MOE_SPEC_EXPERT_BUDGET=<B\|auto>`), capping cold expert SSD reads during speculative verification. |

## Architecture & Bottleneck Remediation Ledger

| Component | Decision | Evidence |
| :--- | :--- | :--- |
| `QWResidencyAdmission` + `PrefixCache` | Unify live `vm_statistics64` UMA admission across CLI load guards, `DriverConfig`, and `qwg serve`; clamp `PrefixCache` warm RAM to `<= 25%` of post-load headroom (`0` bytes under pressure). | 14 targeted residency/cache tests and `make lint` pass without loading real model packs. |
| `QWCoarsePoolIndex` (`e-means` attention) | Use deterministic single-TG `qwg_emeans_collect` (ascending cluster tie-break), required on-device exact flat Top-K overflow fallback (`candState.overflow` / `numMiddle > candCap`), zero-Stage-1 flat bypass below `<41K` tokens (`probeClusters == 0`), and keep MLA `exactFlat` by default (`QW_MLA_EMEANS=1` opt-in). | 68 `QWAttnTests` (including 6-run equal-score tie determinism, forced `candCap = 128` 16/16 needle recovery, and `<41K` bypass parity) and 79 Flash-Next tests pass. |
| Hermetic `--no-profile` + `SHVQSuites` | Require `--no-profile` in `paired_perf.py --strict` so saved model profiles cannot silently diverge arms; extend `qwg-bench-kernels` (`--only-shvq`) with cache-hot and 256 MiB ring-cycled cold-streaming SHVQ suites. | Python strict-harness unit tests, 46 CLI/Serve Swift tests, and `qwg-bench-kernels` release build + winner-map check pass with `0` warnings. |
| `MoETilePlan` indirect NAX + SHVQ LUT/tiles | Dispatch grouped-NAX MoE (`encodeMoEGateUpNax`, `encodeMoENaxSplitF32`) exclusively via on-device `MTLDispatchThreadgroupsIndirectArguments` written by `qwg_moe_group_pairs_f32` (`0` empty threadgroups/slices), separated by `QWHal.rotateEncoder` so AGX indirect-grid fetch sees flushed tile counts; stage `SHVQ2/3` LUTs into `threadgroup` SRAM (`ShvqTgLut<2|3>`) and enable `SHVQ4` direct half lookup in split-NAX plus Apple10 `bm32` small-prefill tiles. | `make lint` (`995` kernels, `0` violations), `QWSHVQGemmTests` (`6/6`), `QWSmallPrefillTests` (`2/2` bit-exact across shapes/rows/epilogues), `FNMetalMoE*` tests, and `QWEngineTests` pass with `0` warnings; live real-pack A/B requires a quiet GPU (`0%/0%`). |
| `--load-vision` / `--stream-ngrams` + `v0.14.0` Option A consolidation | Wire universal vision & PLE residency toggles across CLI, `QWEngineDriver`, `FNGPUEngine`, and `FNGPUPLETable.resolveBacking`; ship `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`100.63 GiB` 100% resident) and `Qwen3.8-Flash-Next-hq8g32-hq4g64` + `QNGRAM v3 VQ2_G32` (`71.29 + 27.13 GiB`, `-28.48 GiB` RAM toggle savings). | Paired CLI A/B, 19-case UMA memory matrix, `audit_release.py` (`ok: true`), `llmprobe v0.6.17` server benchmarks + `core` reasoning evals (`10/10` Flash-Next, `8/10` 27B+DSpine, `0` wrong on 35B & GLM SHVQ), and full `make ci` pass. |
| `--lsa-cluster-block 4096` (`e-means`) + `64K–1M` LSA/`e-means` qualification (`v0.14.1`) | Expose `--lsa-cluster-block 4096` in `QWKnobRegistry`, `qwg`, and `qwg-menu`; keep descriptor-driven defaults (`512` `Ward` on `QWLSAOperator`, `4096` `e-means` on `qsa-cluster`, `exactFlat` `<41K` + `QW_MLA_EMEANS=1` opt-in on MLA). | 44 quiet-GPU (`0%/0%`) 3-needle runs (`16K`–`1M`) across all 4 packs (`100%` `3/3` sparse recall; `+39.8%` pre / `+55.3%` dec `@1M` on `Flash-Next` `e-means`, `+19.8%` dec `@160K` on `GLM-5.3-Flash`, `+3.2%–9.0%` pre on `Qwen3.8-27B`, `2,317 / 105.6 tok/s` `@1M` on `35B-A3B` `Ward-512`), `15/15` knob tests, and full `make ci` pass. |
| Unified adaptive defaults, CLI/Menu Bar handles & settings reference (`v0.15.0`) | Default `kvMode = "q4"` and `lsaMode = "auto"`; auto-escalate QSA `>262K` to `e-means-4096` and MLA `>=41K` to `e-means` (`<41K` flat bypass, `off` -> `exactFlat`); add `HQ8_G32_M5` SVD dequant, `config.json` drafter detection (`dspine` `B=16` vs `dflash2` `B=8`), CLI `--memory-policy` / `--resident-experts-per-layer` / `--moe-spec-budget`, Menu Bar `DSpine` / `streamNgrams` / `moeSpecBudget`, and tracked `docs/settings-reference.md`. | `make lint`, zero-warning `qwg` & `qwg-menu` builds, `KnobBindingTests` (`15/15`), `QWServerTests`, `QWAttnTests`, `make goldens`, and full `make ci` pass. |
| GitHub Actions paravirtual GPU & host environment CI gating (`v0.15.1`) | Resolve Swift Testing `#expect(Bool(true))` and `#require` macro ambiguities in `QWCliTests`; gate bare-metal Apple Silicon GPU (`isMetal4TensorCapable` NAX, 1024-thread GDN recurrent, 49K-context LSA selector) and host environment (`>= 110 GiB` RAM, `~/.qwg/models`, process `ru_maxrss`, bare-metal timing) tests via `XCTSkipUnless`, and skip `qwg-bench-kernels` execution in `Scripts/ci.sh` Phase 2 on `GITHUB_ACTIONS=true` while running full tests on bare-metal machines. | `make lint` (`0` violations), 25 modified test suites (`0` warnings under `-Xswiftc -warnings-as-errors`, `0` failures on Apple M5 Max), and `./.build/release/qwg --version` (`v0.15.1`). |
| Sub-2-bit antipodal `SHVQ15` / `SHVQ1`, `QNGRAM v3 VQ4_G32` PLE & multi-format `DSpine`/`DFlash 2` (`v0.16.0`) | Ship `SHVQ15` (`1.5`-bit 8D `2048`-centroid antipodal spherical VQ, `1.75/1.625 bpw`) and `SHVQ1` (`1.0`-bit 8D `128`-centroid antipodal VQ, `1.25/1.125 bpw`) across Python/Swift/Metal with AGX `<= 32 KiB` constant-cache PSO isolation (`_sub2` split); add `QNGRAM v3 VQ4_G32` (`50 B/row` = `2.50 bpw`, `15.07 GiB` vs `27.13 GiB` `VQ2_G32`, `-12.06 GiB` UMA) honoring `--stream-ngrams`; enable multi-format `DFlash 2` / `DSpine` drafters (`pack_dflash.py --quant`, `PlaneLayout.resolve` sizing); reject PTQ `LittleBit` (`2.41×` worse KL) and `VQ8_G32` (`0.826` cos) from `Sources/`. | `0.0%` regression across all 4 flagship packs (`paired_perf --strict` `0.989×–1.023×`, `SHA-256` bit-exact, identical residency toggles, `9/10` `llmprobe` code-10); cold-streaming `SHVQ1_G64` `0.0367 ms` (`0.712×` `AQ2`) & `SHVQ15_G64` `0.0480 ms` (`0.765×` `AQ2`); `DSpine-b16-shvq15g64` `-42.0%` size (`925.22 MiB`), `1.229×` serial at `B=8`, `35/35` bit-exact `SHA-256`, `9/10` `llmprobe`; `VQ4_G32` `0.9557` cos & `0.917×` `480×512` GPU gather; `14/14` Python tests + full `make ci` pass. |
| Generalized NoPE MLA weight & paged-KV acceleration (`SHVQ{1,15,2,3,4}` + `AQ/HQ{2..8}` + `q8`/`q4` KV) (`v0.17.0`) | Generalize `qwg_mla_absorb_q(_mma[_sub2])`, `qwg_mla_proj_uv(_mma[_sub2])`, `qwg_mla_sparse_attn_mma`, and Apple10 NAX `qwg_mla_sparse_attn_nax(_split)` via `qw_mla_dequant.metalh` (`ShvqMlaTgLut` SRAM staging + `22 KiB`/`18 KiB <= 32 KiB` packed constant LUT isolation + `q8`/`q4` paged latent dequant), removing scalar fallbacks on non-incumbent `kv_b_proj` and quantized KV modes across prefill, decode, and MTP verify. | `make lint` (`0` violations), `QWAttnTests` (`71/71`, including `QWMLAGeneralizedTests` across `SHVQ{1,15,2,3,4}`, `AQ{3,4,8}`, `q8`/`q4` KV, NAX/split-K, and operator E2E), `0` warnings; on all-resident `GLM-5.3-Flash` (`SHVQ`, `q4` KV, `4,096`-tok chunk), GPU `mla.sparse_attn` drops `26,035 -> 487 ms` (`53.5×`), `mla.absorb_q` `1,115 -> 53 ms` (`21.1×`), `mla.proj_uv` `924 -> 75 ms` (`12.3×`), and decode MLA drops `20.44 -> 1.72 ms/tok` (`11.9×`). |
| `PrefillScratch` + `MoEScratch` UMA tightening, `QWMTPEmbeddingTable` gating & subquadratic long-prefill chunking (`v0.17.0`) | Remove 11 dead `PrefillScratch` fields and unused `gdnY`/dense-attn tensors on `gdnVectorDecay` / `usesNoPEMLA`; size grouped-NAX `MoEScratch` to `r1Rows <= 8` + `.f16` `dnAcc` (`prefillBytesPerToken` `1.60 MB -> 556 KB` on `GLM-5.3-Flash`, `760 KB -> 500 KB` on `27B`); share `decodeAndMTPScratchBytes` + `mlaFixedScratchBytes` across `QWEngineDriver` and `DriverResidency` (eliminating the double-counted `4 GiB` reserve); gate automatic `QWMTPEmbeddingTable` (`1.48 GiB` `MAP_SHARED` `MTLResidencySet`) to `mtpDrafts > 1` so `mtpDrafts = 1` never triggers VNODE page-fault stalls (`p95 896 -> 39 ms`); exempt `usesNoPEMLA` and `QWLSAOperator` from the `> 131K` `256/128`-row quadratic chunk clamp with 4-layer command-buffer slicing. | `make lint` (`0` violations), `QWEngineTests` + `QWServerTests` (`339` tests, `0` failures, `0` warnings); out-of-the-box (`--no-profile`) `GLM-5.3-Flash` (`5,195` tok, `q4` KV) keeps `288/288` experts resident (`cold: staged=0`, `fixed_scratch` `6.96 -> 4.31 GiB`), lifting `--strict` prefill `145.8 -> 1,005.9 tok/s` (`6.92×`, `1,061.3 tok/s` GPU @4K) and `@242.5K` `3/3`-needle prefill/decode to `743.0 / 44.98 tok/s` (`e-means`). |
| Generalized `SHVQ{1,15,2,3,4}` direct-half & SRAM LUT kernels + vectorized `qwg_mla_absorb_q` (`v0.17.0`) | Enable `QW_SHVQ_ROUTED_DIRECT_LUT` (`kShvqLut4DirectBits` / `kShvqLut15DirectBits` `32.0 KiB <= 32 KiB` per specialized PSO) across dense/MLP/GDN/HC/NAX `SHVQ4` & `SHVQ15` kernels, stage `ShvqTgLut<1\|2\|3>` in `threadgroup` SRAM for dense `SHVQ1/2/3` NAX GEMMs and multi-row GEMVs while keeping grouped MoE NAX on resident `2–4 KiB` AGX constant LUTs, keep runtime-`bits` MLA PSOs at `22 KiB`/`18 KiB <= 32 KiB`, and vectorize `qwg_mla_absorb_q[_sub2]` (`128` quads × `4` row-slices). | `make lint` (`0` violations), `QWGEMMTests` + `QWAttnTests` pass with `0` warnings; cold-flushed `qwg-bench-kernels --only-shvq` `prefill_nax_f32` `SHVQ2/3` speeds up `2.23–2.24×`, and `--strict` paired `GLM-5.3-Flash` (`SHVQ`, `q4` KV) reaches `1,005.9 tok/s` prefill (`6.92×`), `44.64 tok/s` CLI / `46.0 tok/s` `serve` serial decode (`1.88×`), and `47.59 tok/s` paired (`51.19` CLI / `50.7` `serve` peak) NextN MTP-1 decode (`2.03×`), with `0.0%` regression and `9/10` `llmprobe` code-10 across all 4 flagships. |
| Generalized `QWToolCallConstraint`, parallel `QWToolCallParser` & `>= 75%` UMA prefill-scratch eviction (`v0.17.1`) | Deduplicate and range-clamp `QWToolCallConstraint.allowedTokens()` when XML delimiter tokens (`</tool_call>`) have `"special": false` in `tokenizer.json`; fix `QWToolCallParser.matchingToolCallClose` so adjacent parallel `<tool_call>...</tool_call>` blocks never merge while preserving literal embedded `<tool_call>` tags inside arguments; wire `evictPrefillScratch()` into `DenseModule` and auto-trigger `.auto` eviction when `wiredEngineBytes >= 75%` of physical RAM. | `make lint` (`0` violations), `make ci` unit suites (`0` warnings), and uncontested `1,489,839`-token `llmprobe --full` + `100`-task `code` on `GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64` (`184/184` API conformance, `37/37` capability, `81/100` HumanEval+MBPP `92%` HE, `66/88` agentic, `99.74%` fidelity, `53.6` pred / `45.6` novel `1.98 tok/step`, `96×` `18 ms` warm prefix cache, and steady `+5.4%` `41.1 -> 43.3 tok/s` 21m thermal stability). |
| Native `ERNIE-5.5-Flash` (`HCA` + `MLA-DSA` + `Latent-MoE`) + M5 `HCA` NAX, Grouped-NAX `ShvqTgLut`, Small-`M` NAX Prefill & Draft Partials Fix (`v0.18.0`) | Support `ernie5_5_moe` (`37` HCA + `6` decoupled-RoPE `MLA-DSA` + `42` `512`-expert `Latent-MoE` `SituAndMul` + 4-stream `mHC` + `MultiMax` + shared-Layer-43 `ECHO`); stage `ShvqTgLut` SRAM codebooks inside `qwg_moe_nax_gemm_shvq*` & `qwg_moe_nax_split_shvq*_f32`; gate `QWMoEOperator` grouped-NAX on density (`groupedNaxMinRows = 26` at `E=512, K=10`) and `minNax = 16` for dense F32 projections; add `BM=16, BN=128` / `BM=32, BN=64` NAX tiles in `gemmBatchedH16` (`mode = (M >> 29u) & 3u`); add Apple10 `32`-head `qwg_hca_attn_nax`; size latent-attention `partialsCarveBytes` via `draftHeadArgmaxPartialsBytes(vocab:)` (`55297ddd`). | `make lint` (`0` violations), `QWStandardGeometryLawTests` (`13/13`), `make goldens`, `swift build -Xswiftc -warnings-as-errors`, and `make ci` pass with `0` warnings; paired ERNIE-5.5 Phase 2 A/B (`--prefill-chunk-size 4096`) lifts prefill (`t11_zh` `84.8 -> 89.2 tok/s` `1.052×`, `t29_code` `178.1 -> 211.4` **`1.185×`**, `t512` `756.0 -> 825.3` **`1.092×`**, `t19k` `788.6 -> 855.9` **`1.130×`**, peak `898.9`); `18.8K`/`149.5K`/`300K` `3/3` needles + `1M` `87.59 GiB` UMA admission; `.qwcache` **`78.59×`** (`0.29 s`); `paired_perf --strict` across all 4 instruct flagships (`0.984×–1.001×`, `5/5 SHA-256` exact, `9/10` `llmprobe` `code-10`, `0` allocs). |
