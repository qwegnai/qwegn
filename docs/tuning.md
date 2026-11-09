# Engine Tuning & Memory Policy

## Unified-memory residency

| Flag / env | Default | Description |
| :--- | :--- | :--- |
| `--memory-policy` / `QW_MEMORY_POLICY` | `maxResident` | Unified-memory residency policy (`maxResident` reserves `16 GiB` for OS/KV; `safe` reserves `24 GiB`). Override via `QW_OS_RESERVE_GIB` or `QW_RESIDENT_EXPERTS`. |
| `QW_EXPERT_CACHE` | `1` *(on)* | Dynamic per-layer GPU expert LRU cache. |
| `QW_COLD_EVENT_OVERLAP` | `1` *(on)* | Event-pipelined NVMe/GPU cold-expert overlap (`ColdEventPipeline`). |
| `QW_PREFILL_SPAN_MAP` | `1` *(on)* | GPU prefill span map (`qwg_moe_span_ids`). |
| `--evict-prefill-scratch` | `auto` | `auto` retains scratch on Macs over 36.5 GB; `always` or `never`. |
| `--warm-cache-limit` / `--ssd-cache-limit` | `4` / `10` GiB | Prefix-cache budgets (`serve`). |
| `--prefill-chunk-size` | `512` | Chunk size (`32...8192`). The general driver caps chunks at 4096 rows; Flash-Next uses balanced chunks of up to 8192 rows. |

Defaults come from `QWKnobRegistry`. Per-model profiles and machine size can override
them; for example, MoE models on Macs over 36.5 GB use 4096-row prefill chunks.

## Cold-expert streaming

For models larger than unified memory (e.g. the 321B `GLM-5.3-Flash` pack at
`134.37 GiB` on a 128 GB Mac), Qwegn streams cold experts from NVMe with zero OS
swap through five coordinated subsystems: the `maxResident` memory policy
(`242/288` experts/layer resident), the dynamic per-layer expert LRU cache,
event-pipelined cold staging (`ColdEventPipeline` hiding the `~115 µs`
`MTLSharedEvent` wake latency), GPU prefill span maps, and NextN MTP with
2-row batched verify. Measured gains per subsystem are tabulated in
[benchmarks.md](benchmarks.md#321b-cold-expert-nvmegpu-streaming-nope-mla--engine-tuning-glm-53-flash).

## Speculative decoding knobs

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--mtp-backend` | `variational` | `standard`, `variational` (Voronoi log-margin) or `dflash2`. |
| `--mtp-draft-strategy` | `hybrid` | `hybrid` (MTP + prompt lookup), `neural` or `ngram`. |
| `--mtp-drafts` | `1` | Drafts per verify step (`0` = serial). The cap comes from the model descriptor: 3 for Flash-Next, 5 otherwise. |
| `--mtp-relaxed-margin` | `1.5` | Variational log-margin δ (`2.0–2.5` works well for code and chat). |
| `--mtp-relaxed-min-prob` | `0.2` | Variational fidelity floor τ: a draft accepted through the margin must also hold `p ≥ τ` under the verifier, which stops flat rows from waving through degenerate repeats (`0` disables it). Only MTP-sourced rows are relaxed; context-copy drafts verify exactly. |
| `--dflash-draft-pack` / `--dflash-window` | — | DFlash 2 drafter pack and window B in `2...8`. |
| `--moe-spec-budget` | `0` *(off)* | MoE-Spec routed expert budget per verify layer (`1...8`, or `QW_MOE_SPEC_BUDGET=<B>`), capping cold expert SSD reads during speculative verification. |
