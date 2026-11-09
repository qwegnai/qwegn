<p align="center">
  <img src="assets/qwegn_engine_bento.svg" alt="Qwegn" width="820">
</p>

<h1 align="center">Qwegn</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Swift-6.2-orange.svg" alt="Swift 6.2">
  <img src="https://img.shields.io/badge/platform-macOS%2026%20(Apple%20Silicon)-lightgrey.svg" alt="macOS 26, Apple Silicon">
  <a href="https://github.com/qwegnai/qwegn/actions/workflows/ci.yml"><img src="https://github.com/qwegnai/qwegn/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Release-v0.12.0-green.svg" alt="Release v0.12.0"></a>
</p>

**Qwegn** (`qwg`) is a from-scratch LLM/VLM inference engine for Apple Silicon,
written in Swift 6.2 and Metal 4 — no MLX, no llama.cpp, no PyTorch at runtime.
Custom cooperative-tensor kernels, direct Metal command encoding, and zero heap
or Metal-buffer allocations in steady-state decode, from dense 9B models up to a
321B MoE streamed from NVMe on a 128 GB Mac.

Requires macOS 26+ on Apple Silicon (M-series; M5 recommended for NAX cooperative
tensors), Xcode 26+ and Python 3.13 for the build. One-line dev loop: `make init && make ci`.

## News

- **2026-11-09 — v0.12.0:** Native **DSpine** block-causal speculative decoding (`Qwen3.8-27B-DSpine-b16-hq4g32`, `1.63 GB`, `--mtp-backend dspine`) with 6 custom fused Metal 4 kernels (`qwdspine.metal`: fused 5-layer RMSNorm+Concat, low-rank `5×5120 → 1024 → 4096` feature bridge, `256`-dim KV cache, causal depthwise `Conv1d(k=3) + SiLU`, and learned positional embeddings) and widened **`M <= 16` (`D <= 15`)** row-paired target verification: **`134.5 tok/s` peak (`4.19×` vs `32.1 tok/s` serial at $B=8$)**, **up to `12.70` accepted tokens/round (`83.3%` accept at $B=16$)**, and **`100.5 tok/s` (`3.09×` serial) 3-workload geometric mean** at default $B=6$ (`D=5`) with 100% bit-exact SHA-256 greedy parity.
- **2026-11-09 — v0.11.3:** Prefix-invariant multi-turn LSA Memory across live disk edits (`99.87%` Warm-RAM hit, `159.3 ms` TTFT, `307.0K tok/s` prefill after file modification), autonomous `assistant -> tool` Vault paging, bounded `10.4K`-token Cortex concept map (`-54.7%` overhead), and `179.2K`-token large-volume validation (`2,041.0 tok/s` sustained GPU chunked prefill on `+130.2K` new tokens; **`99.96%` Warm-RAM hit, `452.6 ms` TTFT, `395,874.9 tok/s` effective prefill at `179,173` tokens**).
- **2026-11-08 — v0.11.1:** Verified full 7-experiment Two-Tier LSA Cortex (`32,742` tokens, `264.61 MiB` `35B-A3B` / `464.54 MiB` `Flash-Next` — **`23.7×` smaller `.qwcache`**) + Off-Context Vault (`287` files, `< 0.5 ms` compound-identifier recall) whitepaper benchmark suite: **`54.4×` faster TTFT (`6.53 s → 0.12 s`, `290.6K tok/s`) on `Qwen3.6-35B-A3B`**, **`74.3×` faster TTFT (`17.84 s → 0.24 s`, `138.4K tok/s`) on `Qwen3.8-Flash-Next`**, **`100%` (`4/4`) Cortex + Vault recall**, **`0.40 s` live `<|memory_delta|>` TTFT**, and **`150.5–163.9 tok/s` (`+53%`) multi-turn decode** with `96.9%` (`1,015,834` tokens) of the 1M context window free.
- **2026-11-08 — v0.11.0:** Two-Tier Hierarchical LSA Cortex (`32.7K` tokens, `265 MiB` `35B-A3B` / `465 MiB` `Flash-Next` — **`23.7×` smaller `.qwcache`**) + Sub-Millisecond Off-Context Vault Recall (`QWMemoryVault`, `< 0.5 ms`) + Elastic Context Budget Guard: indexes 100% of repository files (`293` files + AST symbol map) while leaving **`96.9%` (`>1,015,000` tokens) of the 1M context window free** for coding agents, preserving **100% `[warm_ram]` prefix hits** across multi-turn `qwg repl` and `qwg serve` sessions, and boosting multi-turn decode throughput by **`+53%` (`147.5–163.7 tok/s`)**.
- **2026-11-08 — v0.10.4:** 8-row speculative verification ring & dynamic wide Prompt-Lookup Decoding (`draftCapacity = 7`, `8`-slot GDN/PLE rollback ring, 4-row chunked `encodeHQ4Gemv`/`encodeHQ4F32V4Argmax`, chain-pass suppression on copy rounds), accelerating `Qwen3.8-Flash-Next` Predictable (Copy) throughput from `129.3 → 156.3 tok/s` (`3.37 → 5.49 tok/step`, `1.87×`) on native HQ and `126.1 → 158.8 tok/s` (`3.31 → 5.49 tok/step`, `1.74×`) on MLX packs at `100%` Engine Fidelity.
- **2026-11-08 — v0.10.3:** Unified engine harness generality & deep-module refactoring: shared `QWEngineBootstrap` admission, unified `QWSafeTensorsContainer.readFloats` (with parallel MLX affine dequantization), parallel multi-layer `.qwp`/`.safetensors` `QWLSASidecar` Weight-SVD synthesis, generalized `QWKVCompressedLayout`, $O(1)$ root-hash prefix cache filtering, and deterministic `ColdEventPipeline` / `QWEngineLifecycleActor` lifecycle draining.
- **2026-11-08 — v0.10.2:** Multi-turn `.qwcache` memory-pack retention across `qwg repl` and `qwg serve`: pinned shared-base protection, incremental COW KV page accounting, `QLSADLT1` delta LSA snapshots (`467.4 MiB` $\to$ `~6.4 KB` per turn), and pre-`<think>` checkpoint advancement (`0.17–0.29 s` TTFT across multi-turn 1M-token sessions).
- **2026-11-08 — v0.10.1:** Pruned 125 dead/superseded Metal kernels (`846` $\to$ `721` in `kernel.metallib`), hardened `Scripts/lint_kernels.py` against stale inventory/comment masking, and added `.gitignore`/`.git/info/exclude`/`.qwgignore` filtering in `qwg index`.
- **2026-11-08 — v0.10.0:** Unified Codebase & Knowledge Memory Packs (`qwg index`, `<repo>/.qwg/*.qwcache`) delivering `55.2×–66.0×` faster repository TTFT (`0.08–0.17 s`), live `<|memory_delta|>` working-tree overlays without base rebuilds, and automatic repository discovery & `kvMode` enforcement across `qwg repl`, `qwg run`, and `qwg serve`.
- **2026-11-08 — v0.9.7:** Vision head support (`qwen4_exp` / `Qwen3.8-Flash-Next` and `Qwen3.6-35B-A3B`) with live in-place residency toggle (`loadVision`), `31.0×` faster AMX `cblas_sgemm` ViT encoding (`79.9s` $\to$ `2.57s`), and `0 ms` repeat-turn encoded image LRU cache. Full history in [CHANGELOG.md](CHANGELOG.md).

## Quick Start

```bash
make init     # verify toolchain, create .venv-oracle, generate unit-test goldens
make build    # .build/release/qwg and .build/release/qwg-menu
make ci       # metallibs + linters + tests
```

Pull, quantize and run a model:

```bash
.build/release/qwg pull Qwen/Qwen3.8-27B --folder ~/.qwg/hf
.build/release/qwg quantize ~/.qwg/hf/Qwen3.8-27B --out ~/.qwg/models/Qwen3.8-27B-q4g --quant q4g
.build/release/qwg run ~/.qwg/models/Qwen3.8-27B-q4g \
  --prompt "Explain Query-Aware Sparse Attention and Gated DeltaNet in three bullet points." \
  --max-tokens 256
```

`qwg plan <pack> --context-tokens 262144 --kv-mode q4` reports GPU capabilities and
whether a context budget fits in unified memory — a zero-generation way to verify
the toolchain.

> Model packs are large quantized checkpoints. Until official Hugging Face packs
> are published, build one from a local Hugging Face download with `qwg pull` +
> `qwg quantize` (above). <!-- TODO(link-hf-org) -->

## Highlights

- **Codebase & Knowledge Memory Packs (`QWMemoryPack` + `QWMemoryVault`)** — precompute a repository offline (`qwg index`) into a **Two-Tier Hierarchical LSA Memory Pack** (`<repo>/.qwg/*.qwcache` + `.qwg/manifest.json`): **`32.7K`-token Resident Cortex (`265 MiB` `35B` / `465 MiB` `Flash-Next`, `23.7×` smaller `.qwcache`)** + **`< 0.5 ms` Off-Context Vault Recall** across all `294` files, leaving **`96.9%` (`1,015,834` tokens) of the 1M context window free** for coding agents while delivering **`54.4×–74.3×` faster TTFT (`0.12–0.24 s`)**, **`150.5–163.9 tok/s` multi-turn decode (`+53%`)**, and zero-rebuild `<|memory_delta|>` live git edits (`0.40 s` TTFT). [Manual](docs/lsa.md#4-unified-codebase--knowledge-memory-packs-qwg-index--repoqwgqwcache) · [Whitepaper](docs/lsa-codebase-memory-whitepaper.md)
- **Model-agnostic runtime** — hybrid, looped and MoE architectures driven by the model descriptor (`QWModelDescriptor`), never by model name. [Supported models](#supported-models)
- **321B MoE from NVMe, zero swap** — 31.6 tok/s serial / 34.8 tok/s speculative on a 128 GB Mac. [Cold-expert streaming](docs/tuning.md#cold-expert-streaming)
- **1M-token context** — Latent Sparse Attention: 2,294 tok/s prefill, 102.8 tok/s decode at 1M, 3/3 needle recall, training-free. [LSA manual](docs/lsa.md)
- **Speculative decoding & mixed quantization** — MTP / Variational / DSpine (`B=2..16`, up to `4.19×` / `12.7 tok/round`) / DFlash 2 / MoE-Spec across all 20 Hadamard `HQ*` and affine `AQ*` formats. [Quantization guide](docs/quantization.md)

## Benchmarks (headline)

| Model | Prefill (peak) | Decode (spec) | 1M-ctx prefill / decode | Repo `.qwcache` TTFT (32.7K Cortex) |
| :--- | :---: | :---: | :---: | :---: |
| Qwen3.6-35B-A3B | 6,389 tok/s | 233.4 tok/s | 2,294 / 102.8 tok/s | **0.12 s** (`54.4×` vs 6.53 s cold) |
| Qwen3.8-Flash-Next (HQ8+HQ4) | 2,696 tok/s | 142.4 tok/s | 1,353 / 45.4 tok/s | **0.24 s** (`74.3×` vs 17.84 s cold) |
| Qwen3.8-27B (+ DSpine-b16) | 876 tok/s | **134.5 tok/s** (`4.19×`) | 805 / 43.1 tok/s | 6 ms warm cache (307×) |
| GLM-5.3-Flash (321B, NVMe-streamed) | 968 tok/s @16K | 34.8 tok/s | arch ready | MLA Key-Pool(4) snapshot |

All measured on Apple M5 Max 128 GB (`llmprobe v0.6.16`, `--temperature 0`).
Full tables, paired A/B repository memory benchmarks, and thermal protocol: [docs/benchmarks.md](docs/benchmarks.md).

## Supported models

| | |
| :---: | :---: |
| <img src="assets/qwegn_model_glm53_flash.svg" alt="GLM-5.3-Flash" width="720"> | <img src="assets/qwegn_model_qwen36_35b_a3b.svg" alt="Qwen3.6-35B-A3B" width="720"> |
| **GLM-5.3-Flash** · 45L MoE (`321B` total / `18B` active, 288 experts, 8 + 1 active), 11 NoPE MLA (512D latent KV + Key-Pool(4)) + 34 KDA, 4-stream Sinkhorn mHC, NextN MTP, vision (`HQ4_G32_M5 + HQ3_G64`) | **Qwen3.6-35B-A3B** · 40L MoE (256 experts, 8 + 1 active), 10 QSA + 30 GDN, MoE MTP-3, vision |
| <img src="assets/qwegn_model_flash_next.svg" alt="Qwen3.8-Flash-Next" width="720"> | <img src="assets/qwegn_model_qwen38_27b.svg" alt="Qwen3.8-27B" width="720"> |
| **Qwen3.8-Flash-Next** · 48L MoE (512 experts, 10 + 1 active), 12 QSA + 36 GDN, 4-branch HC, PLE, MTP, vision (`HQ8_G32 + HQ4_G64`) | **Qwen3.8-27B** (+ DSpine-b16 & DFlash 2) · 64L dense hybrid, 5120 hidden, 24Q / 4KV GQA, MTP-3/5, 5-layer DSpine (`134.5 tok/s` / `4.19×`) & DFlash 2 drafters, vision |
| <img src="assets/qwegn_model_sensenova_loop_9b.svg" alt="SenseNova-Loop-9B" width="720"> | <img src="assets/qwegn_memory_pack_experiments.svg" alt="QWMemoryPack Live Benchmarks" width="720"> |
| **SenseNova-Loop-9B-Preview** · 32 physical → 56 effective layers (2× loop), 2-stream Sinkhorn mHC, MTP-3 | **QWMemoryPack (`.qwcache`)** · Two-tier COW repository memory (`55.2×–66.0×` TTFT speedup, `0.04 ms` per-token LSA selection, live `<memory_delta>`) |

Standard `llama`, `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5_text` and `mistral` checkpoints also run on the unified runtime. Their model configs are adopted automatically, and per-model profiles are stored in `~/.qwg/profiles/`.

## Codebase Memory Packs (`qwg index`)

Index any git repository into `<repo>/.qwg/` (`manifest.json` + `.qwcache`) to give `qwg repl`, `qwg run`, and `qwg serve` instant whole-codebase visibility without cold prefill or exploratory grep loops:

```bash
# 1. Build repository memory pack offline (auto-selects kvMode from ~/.qwg/profiles/<model>.json)
.build/release/qwg index /path/to/repo --model ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64

# 2. Interactive REPL (auto-mounts .qwg/manifest.json in cwd; slash commands: /memory, /memory mount|unmount)
cd /path/to/repo && .build/release/qwg repl --model Qwen3.8-Flash-Next-hq8g32-hq4g64

# 3. Single-shot CLI query against mounted repository memory (pass --no-memory-pack to bypass)
.build/release/qwg run ~/.qwg/models/Qwen3.6-35B-A3B-hq4g32 --repo /path/to/repo --prompt "Audit AGENTS.md Law 20."
```

Full architecture, 3-tier stability ordering, live `<|memory_delta|>` overlays, and benchmarks: [docs/lsa.md](docs/lsa.md#4-unified-codebase--knowledge-memory-packs-qwg-index--repoqwgqwcache) and [docs/lsa-codebase-memory-whitepaper.md](docs/lsa-codebase-memory-whitepaper.md).

## Server

OpenAI-, Anthropic-, and Ollama-compatible HTTP server (auto-discovers `<repo>/.qwg/manifest.json` from coding-agent workspace prompts):

```bash
.build/release/qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --host 127.0.0.1 --port 8080 --kv-mode q4 --context-tokens 262144 --no-lazy
```

| Protocol | Endpoints | Features |
| :--- | :--- | :--- |
| **OpenAI API** | `POST /v1/chat/completions` · `GET /v1/models`<br>`POST /v1/models/load` · `POST /v1/models/unload` | Streaming SSE (`stream: true`), multimodal `image_url`, constrained JSON & tool calling, `reasoning_content`, runtime model switching |
| **Anthropic API** | `POST /v1/messages` | Streaming SSE events, system prompts, thinking blocks, tool use |
| **Ollama API** | `POST /api/chat` · `POST /api/generate` · `GET /api/tags`<br>`GET /api/ps` · `POST /api/show` · `POST /api/load` · `POST /api/unload` | NDJSON streaming, model status and residency management |
| **Memory & Telemetry** | `POST /v1/memory/mount` · `POST /v1/memory/unmount`<br>`GET /v1/engine/info` · `GET /ui` · `GET /v1/dashboard` · `GET /metrics` | Live `.qwcache` repo memory mount/unmount & `kvMode` lock, built-in web UI (`http://127.0.0.1:8080/ui`), live token/s and cache telemetry |

Example OpenAI request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-hq8g32-hq4g64",
    "messages": [{"role": "user", "content": "Write a fast radix top-k selection summary."}],
    "temperature": 0.0,
    "max_tokens": 256,
    "stream": true
  }'
```

## CLI reference

| Command | Description |
| :--- | :--- |
| `qwg plan` | Reports Apple Silicon GPU capabilities and verifies whether a context budget fits in unified memory. |
| `qwg pull` | Downloads a Hugging Face checkpoint (`org/model`) into a local directory using parallel HTTP range requests. |
| `qwg quantize` | Converts a BF16 safetensors checkpoint into a page-aligned quantized `.qwp` pack. |
| `qwg inspect` | Validates a pack's header, provenance, quantization descriptors, and tensor geometry (`--dequant` exports `.npy`). |
| `qwg index` | Builds an offline 3-tier LSA codebase memory pack (`<repo>/.qwg/manifest.json` + `.qwcache`). |
| `qwg run` | Executes single-turn generation from a prompt string or file (`--repo <path>` mounts `.qwcache`). |
| `qwg repl` | Launches an interactive multi-turn terminal chat session with automatic `.qwg/` memory & prefix caching. |
| `qwg serve` | Runs the multi-protocol HTTP inference server (OpenAI, Anthropic, Ollama, Memory API, and Web UI). |
| `qwg bench` | Runs performance regression gates against `Scripts/bench/baselines.json`. |

Core flags for `run` / `repl` / `serve` (defaults come from `QWKnobRegistry`; per-model profiles and `.qwg/manifest.json` can override them):

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--context-tokens` | `4096` | Context capacity, up to `1048576` (1M tokens) with YaRN auto-scaling and LSA. |
| `--kv-mode` | `f16` | KV cache format: `q4`, `q8` or `f16` (automatically enforced when a `.qwg/` memory pack is mounted). |
| `--lsa-mode` | `off` | Latent Sparse Attention mode (`auto` recommended). [Full LSA flag matrix](docs/lsa.md#lsa-usage-manual) |
| `--repo` / `--no-memory-pack` | `cwd` | Mount `<repo>/.qwg/manifest.json` in `run`/`repl`, or bypass automatic repository memory mounting. |
| `--temperature` / `--top-p` / `--top-k` | `0.7` / `0.9` / `40` | Sampling (`0` selects greedy). `--enable-thinking` (`true`) toggles `<think>` blocks. |

## Testing & Quality Gates

```bash
make lint     # file size, comment style, geometry defaults, basis, Metal kernel reachability gates
make test     # XCTest unit and Metal parity suites (serial: GPU bundles are not parallel-safe)
make perf     # performance regression gates against Scripts/bench/baselines.json
```

## Documentation

- [docs/lsa.md](docs/lsa.md) — Latent Sparse Attention, NoPE MLA, 1M-context scaling, and `qwg index` usage manual
- [docs/lsa-codebase-memory-whitepaper.md](docs/lsa-codebase-memory-whitepaper.md) — Whitepaper & live experiments: `QWMemoryPack` vs. RAG and Memorilla (`arXiv:2610.42147`)
- [docs/benchmarks.md](docs/benchmarks.md) — full benchmark tables and thermal-controlled (`Scripts/paired_perf.py`) reproduction protocol
- [docs/quantization.md](docs/quantization.md) — quant formats, mixed packs, quality tooling
- [docs/tuning.md](docs/tuning.md) — memory policy, cold-expert streaming, speculation knobs
- [docs/menu-bar-app.md](docs/menu-bar-app.md) — `qwg-menu` status, telemetry, vision residency & memory pack settings

## Repository Layout

- `Sources/Foundation` — Core tensor types, quantization formats (`QWQuant`), arena allocator (`QWAlloc`), and Metal HAL (`QWHAL`)
- `Sources/Operators` — Quantized matrix multiply (`QWGEMM`), normalization (`QWNorm`), RoPE (`QWRope`), sparse/dense attention (`QWAttn`), and Gated DeltaNet (`QWGDN`)
- `Sources/Subsystems` — Model family pipelines (`QWFlashNext`, `QWMLP`, `QWHead`, `QWMTP`, `QWVision`, `QWState`)
- `Sources/Runtime` — Engine driver (`QWEngine`), `QWMemoryPack`, continuous batching, prefix caching, and speculative decoding (`DFlash` / `MTP`)
- `Sources/QWMetal` — Metal 4 compute kernels (`.metal`) and shared headers (`.metalh`)
- `Sources/Apps` — CLI (`qwg`), HTTP server (`QWServer`), macOS menu bar app (`QWMenuBar`), and kernel microbenchmarks (`qwg-bench-kernels`)
- `Scripts` / `Tests` — Metal build scripts, style/kernel/basis linters, golden generators, and XCTest unit/GPU parity suites

## Contributing & License

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under Apache License 2.0 ([LICENSE](LICENSE)).

## Citation

```bibtex
@software{qwegn,
  title  = {Qwegn: a from-scratch LLM/VLM inference engine for Apple Silicon},
  author = {{QwegnAI}},
  year   = {2026},
  url    = {https://github.com/qwegnai/qwegn}
}
```
