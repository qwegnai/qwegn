# Latent Sparse Attention (LSA), Capacity-Balanced `e-means`, NoPE MLA & 1M Context Scaling

Qwegn implements a general, model-agnostic **Latent Sparse Attention (LSA)** subsystem (`QWLSAOperator`) and **Capacity-Balanced Spherical `e-means` Macro-Clustering** (`QWCoarsePoolIndex`, `arXiv:2611.09501`) that scale full-attention, QSA, and NoPE MLA layers up to **1,048,576 (1M) tokens** on a single Apple Silicon Mac without model retraining.

<p align="center">
  <img src="../assets/qwegn_lsa_emeans_overview.svg" alt="Qwegn 64K to 1M Long-Context Scaling Overview: Without LSA vs LSA (Ward-512) vs Capacity-Balanced e-means" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_lsa_scaling_qwen36_35b_a3b.svg" alt="Qwegn LSA & e-means vs Dense scaling, 64K to 1M tokens: Qwen3.6-35B-A3B" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_lsa_scaling_flash_next.svg" alt="Qwegn Native QSA vs LSA Ward-512 vs e-means-4096 scaling, 64K to 1M tokens: Qwen3.8-Flash-Next" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_lsa_scaling_qwen38_27b.svg" alt="Qwegn LSA & e-means vs Dense scaling, 64K to 1M tokens: Qwen3.8-27B" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_lsa_scaling_glm53_flash.svg" alt="Qwegn NoPE MLA exactFlat vs e-means scaling, 16K to 1M tokens: GLM-5.3-Flash 321B SHVQ" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_lsa_scaling_sensenova_loop_9b.svg" alt="Qwegn LSA vs Dense scaling, 64K to 1M tokens: SenseNova-Loop-9B" width="100%">
</p>

## How Qwegn LSA & Capacity-Balanced `e-means` Work

1. **Low-Rank RoPE-Free Index Projections (`qwg_lsa_index_prep`):** Projects post-RMSNorm unrotated Q/K slices (`rotaryDims..<headDim` on partial-RoPE models, or full `headDim` on standard GQA models) into a compact $d_{\text{idx}} = 64$ latent index space using either an offline `.qlsa` sidecar or zero-config in-memory **Weight-SVD** synthesized at load time from each layer's quantized `wq`/`wk` weights.
2. **4-Token Key Pooling & Dynamic Threadgroup Ward Clustering (`qwg_lsa_pool_keys` + `qwg_lsa_ward_cluster_512`):** Every 4 tokens are mean-pooled and L2-normalized. In standard `Ward-512` mode (`--lsa-cluster-block 512`), every completed 512-token superblock (128 pooled blocks) is partitioned on-GPU into **64 variable-size semantic buckets** (hard cap $\le 16$ tokens / 4 blocks per bucket) via greedy Ward's minimum-variance hierarchical clustering (`16.5 B/token/layer` total cache footprint).
3. **Capacity-Balanced Spherical `e-means` Macro-Clustering (`QWCoarsePoolIndex`, `--lsa-cluster-block 4096` / `QW_MLA_EMEANS=1`, `arXiv:2611.09501`):** For high-dimensional indexers (`Qwen3.8-Flash-Next` $d_{\text{idx}}=128$ QSA, `GLM-5.3-Flash` 64-head 128D MLA, and `Qwen3.8-27B` 16L GQA prefill), every completed **4,096-token macro-block** (`1,024` 4-token pool blocks) is partitioned into **64 capacity-balanced spherical macro-clusters** (`16,384` centroids at `1M` — **`8×` fewer than `Ward-512` and `16×` fewer than `262,144` raw pool blocks**). Stage 1 scores macro-centroids and deterministically expands top clusters via single-TG `qwg_emeans_collect` (`<= 14,336` candidates), falling back on-device to exact flat Top-$K$ whenever `candState.overflow` triggers and bypassing Stage 1 entirely below `41K` tokens.
4. **4-Stage Parallel Radix Block Selector (`qwlsa_select.metal`):** Scores centroids across parallel threadgroups (`_score` $\to$ `_hist2` $\to$ `_expand` $\to$ `_finalize`), expands winning buckets, and selects the top-$k$ 4-token blocks ($k = 2048$ tokens default) while always retaining **sink block 0** (tokens `0..3`) and the **512-token local sliding window**. Decode block selection takes **0.039 ms** at 64K and **0.062 ms** at 1M (**125× faster** than single-threadgroup selection).
5. **M5 NAX Sparse Paged Prefill & Split-K Sparse Decode (`qwattn_paged_sparse.metal`):** Attends only over the selected 4-token blocks in the `f16`/`q8`/`q4` paged KV pool. Below `denseBypassThreshold` ($4 + k + 512 = 2564$ tokens), the engine automatically bypasses LSA for bit-exact dense attention.
6. **Long-Context Speculation Backoff (`QWSpeculationBackoff`):** At prompts $\ge 131\text{K}$ tokens, if rolling 8-round draft acceptance drops below `0.35`, decode automatically switches to serial mode with exponential probe cooldown (`64 → 512` tokens), avoiding wasted multi-row verify passes.

## Measured Scaling Across 3 Regimes (64K → 1M Tokens, 3-Needle Haystack)

All runs measured on an idle Apple M5 Max (`Mac17,6`, 128 GB Unified Memory, 40-core Apple10 NAX GPU, 585 GB/s DRAM, `--kv-mode q4 --temperature 0 --no-profile`, 3 hidden passkeys at 10%, 50%, and 90% document depth):

| Model | Context | Without LSA (Prefill / Decode) | LSA `Ward-512` (Prefill / Decode) | **LSA + `e-means-4096` (Prefill / Decode)** | Needle (No LSA / Ward / `e-means`) | Headline Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.6-35B-A3B-hq4g32`** | 64K | `2,538 / 99.2` | **`6,141.7`** / `126.2` | `5,716.5` / **`136.8`** | `0/3` · **`3/3`** · **`3/3`** | **`2.42×` prefill (`Ward`) · `+8.4%` decode (`e-means`)** |
| *(40L MoE: 10 GQA + 30 GDN)* | 160K | `1,128.6 / 67.2` | **`4,913.8 / 131.1`** | `4,283.0 / 122.5` | `2/3` · **`3/3`** · **`3/3`** | **`4.35×` prefill · `1.95×` decode vs Dense** |
| | 262K | `581 / 50.1` | **`3,805.2 / 118.3`** | `3,526.3 / 102.1` | `0/3` · **`3/3`** · **`3/3`** | **`6.55×` prefill · `2.36×` decode vs Dense** |
| | 524K | `231 / 23.2` | **`3,207.1 / 103.9`** | `3,126.1 / 93.4` | `0/3` · **`3/3`** · **`3/3`** | **`13.88×` prefill · `4.48×` decode vs Dense** |
| | **1M** | — | **`2,317.0 / 105.6`** (`448.8 s`) | `2,315.1 / 88.5` (`30.53 GiB`) | — · **`3/3`** · **`3/3`** | **`105.6 tok/s` @1M (`Ward`) · `-180 MiB` (`e-means`)** |
| **`Qwen3.8-Flash-Next-hq8g32-hq4g64`** | 64K | **`2,279.9 / 72.7`** *(QSA)* | `2,155.4 / 56.7` | `2,183.8 / 72.5` | **`3/3` · `3/3` · `3/3`** | `+27.9%` decode (`e-means` vs `Ward-512`) |
| *(48L MoE: 12 QSA + 36 GDN,* | 160K | **`2,157.6`** / `66.8` *(QSA)* | `1,741.2 / 40.4` | `2,041.0` / **`70.9`** | **`3/3` · `3/3` · `3/3`** | **`+6.2%` decode vs Native QSA (`+75.5%` vs `Ward`)** |
| *`+ VQ2_G32` PLE)* | 262K | `2,051.8` / **`68.6`** *(QSA)* | `1,522.1 / 32.4` | **`2,057.6`** / `63.5` | **`3/3` · `3/3` · `3/3`** | **`1.35×` prefill · `1.96×` decode vs `Ward-512`** |
| | 524K† | `1,721.9 / 54.5` *(QSA)* | `976.2 / 18.7` | **`1,969.6 / 68.1`** (`263.0 s`) | **`3/3` · `3/3` · `3/3`** | **`+14.4%` pre / `+25.0%` dec vs QSA (`3.64×` vs `Ward`)** |
| | **1M†** | `1,350.3 / 41.1` (`770.1 s`) | — | **`1,887.6 / 63.7`** (`550.9 s`) | **`3/3`** · — · **`3/3`** | **`+39.8%` prefill (`-3.65m` TTFT) · `+55.3%` decode** |
| **`Qwen3.8-27B-hq4g32`** | 64K | `672 / 42.1` | `884.8 / 54.0` | **`914.3 / 56.2`** (`70.0 s`) | `3/3` · **`3/3`** · **`3/3`** | **`+3.3%` pre / `+4.1%` dec vs `Ward` (`1.36×` vs Dense)** |
| *(64L Dense Hybrid)* | 160K | `467.4 / 25.9` | `831.9` / **`47.1`** | **`907.0`** / `40.7` (`176.4 s`) | `3/3` · **`3/3`** · **`3/3`** | **`+9.0%` prefill vs `Ward` (`1.94×` vs Dense)** |
| | 262K | `351.0 / 25.3` | `830.2` / **`43.1`** | **`896.5`** / `42.1` (`287.7 s`) | `3/3` · **`3/3`** · **`3/3`** | **`+8.0%` prefill vs `Ward` (`2.55×` vs Dense)** |
| | 524K | — | `831.2` / **`41.1`** | **`891.0`** / `36.1` (`581.3 s`) | — · **`3/3`** · **`3/3`** | **`+7.2%` prefill (`-41.8 s` TTFT) vs `Ward-512`** |
| | **1M** | — | `804.6` / **`36.5`** (`1,292.5 s`) | **`830.6`** / `35.6` (`1,252.0 s`) | — · **`3/3`** · **`3/3`** | **`+3.2%` prefill (`-40.5 s` TTFT) · `-270 MiB` RSS** |
| **`GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64`** | 15K | **`834.2 / 46.54`** *(`exactFlat`)* | — | **`834.2 / 46.54`** *(`<41K` bypass)* | **`3/3`** · — · **`3/3`** | Zero Stage-1 overhead below `41K` tokens |
| *(321B SHVQ `100.63 GiB`,* | 60K | `706.7 / 32.38` *(`exactFlat`)* | — | **`706.7 / 32.38`** (`84.99 s`) | **`3/3`** · — · **`3/3`** | Generalized `q4`/`SHVQ` NAX sparse MLA (`4,096` chunk) |
| *11 NoPE MLA + 34 KDA)* | 242.5K | `721.4 / 44.00` *(`exactFlat`)* | — | **`743.0 / 44.98`** (`326.3 s`) | **`3/3`** · — · **`3/3`** | **`+3.0%` prefill · `+2.2%` decode (`QW_MLA_EMEANS=1`)** |
| | **1M** | `102.3 GiB` plan / `107.2 GiB` RSS | — | **100% UMA resident (`3,696 B/tok` KV)** | `0` SSD misses · `0` swap | `321B` weights + `1M` KV fit in 128 GB Mac |
| **`SenseNova-Loop-9B-Preview`** | 64K | `1,207 / 65.0` | **`1,749 / 75.2`** | — | `0/3` · **`3/3`** · — | **`1.45×` prefill · `1.16×` decode** |
| *(32→56L 2× Loop + mHC)* | 262K | `568 / 38.6` | **`1,606 / 70.4`** | — | `1/3` · **`3/3`** · — | **`2.83×` prefill · `1.82×` decode** |
| | **1M** | — | **`1,476 / 46.5`** (`702.9 s`) | — | — · **`3/3`** · — | RSS: `24.93 GiB` · `0` allocs |

*†`Qwen3.8-Flash-Next` at `524K` and `1M` uses `--stream-ngrams` (`-27.30 GiB` Metal GPU RAM saved).*

> **Why LSA & `e-means` improve long-context Needle Recall (`0/3 → 3/3`):** Under YaRN RoPE scaling at 64K–1M tokens, dense softmax sums over hundreds of thousands of distractors, diluting needle attention weights. By restricting softmax to sink tokens (`0..3`), the top-$k$ highest-similarity 4-token blocks ($k=2048$), and the 512-token local window, both `Ward-512` and capacity-balanced `e-means-4096` eliminate distractor mass and recover **100% (`3/3`) passkey retrieval** across every tested model up to 1M tokens.

## LSA Usage Manual

### 1. Zero-Config CLI Usage (`run`, `repl`, `serve`)

LSA works out of the box on any model pack with `--lsa-mode auto` (uses `<pack>/model.qlsa` if present, otherwise synthesizes Weight-SVD projections on load; pass `--lsa-cluster-block 4096` to switch from `Ward-512` to capacity-balanced `e-means` macro-clustering, or `--lsa-mode qsa-cluster --lsa-cluster-block 4096` on `Qwen3.8-Flash-Next` at `>= 524K` tokens):

```bash
# 1. Qwen3.6-35B-A3B at 1M context with LSA Ward-512 (default 512 superblock: 2,317 / 105.6 tok/s)
.build/release/qwg serve ~/.qwg/models/Qwen3.6-35B-A3B-hq4g32 \
  --context-tokens 1048576 --kv-mode q4 --lsa-mode auto --lsa-budget 2048 --lsa-cluster-block 512 --no-lazy

# 2. Qwen3.8-Flash-Next at 1M context with capacity-balanced e-means-4096 (1,888 / 63.7 tok/s, +55.3% decode)
.build/release/qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --context-tokens 1048576 --kv-mode q4 --lsa-mode qsa-cluster --lsa-cluster-block 4096 --stream-ngrams --no-lazy
```

| LSA Mode (`--lsa-mode`) | Behavior | Best For |
| :--- | :--- | :--- |
| `auto` *(recommended)* | Uses `model.qlsa` sidecar if detected; falls back to in-memory Weight-SVD on dense/hybrid/looped models, or keeps native QSA on `Flash-Next`. | All models (`--lsa-cluster-block 512` default, `4096` for `e-means` prefill) |
| `svd` | Forces training-free in-memory Weight-SVD projection synthesis from quantized `wq`/`wk` weights at model load. | Any `.qwp` pack without a sidecar |
| `sidecar` | Requires a precomputed `model.qlsa` (`QLSA0001`) sidecar file (auto-detected in `<pack>/model.qlsa` or via `--lsa-sidecar-path`). | Instant load & calibrated/distilled projections |
| `qsa-cluster` | Replaces flat QSA block scoring on `Qwen3.8-Flash-Next` with hierarchical macro-clustered selection (`--lsa-cluster-block 4096` `e-means` default). | `Qwen3.8-Flash-Next` at `524K–1M` tokens (`+39.8%` prefill, `+55.3%` decode) |
| `off` | Disables LSA (runs standard dense attention or native QSA). | Short-context baselines |

### 2. Offline `.qlsa` Sidecar Utility (`Scripts/convert_lsa_sidecar.py`)

To precompute a `model.qlsa` (`QLSA0001`) sidecar for instant load and automatic detection by `qwg` and `qwg-menu`:

```bash
# 1. Training-free Weight-SVD sidecar (fastest, works directly on any .qwp pack or HF directory)
python3 Scripts/convert_lsa_sidecar.py \
  --pack ~/.qwg/models/Qwen3.6-35B-A3B-hq4g32 \
  --method svd \
  --index-dim 64

# 2. Calibration-set PCA or teacher-student KL distillation (optional)
python3 Scripts/convert_lsa_sidecar.py \
  --pack ~/.qwg/models/Qwen3.8-27B-hq4g32 \
  --method pca \
  --calib-prompts calib.jsonl \
  --index-dim 64
```

The engine automatically detects `<model_pack>/model.qlsa` (and verifies its `geometry_digest_v2` against `config.json`).

### 3. Menu Bar App (`qwg-menu`) Per-Model Settings

Open **Settings (`⌘,`)** in `qwg-menu` under **Memory & Context → Latent Sparse Attention (LSA)**:
- Select **LSA Mode** (`Auto`, `Weight-SVD`, `Sidecar (.qlsa)`, `QSA Cluster`, or `Off`), **Retrieval Budget** (`512–8192` tokens), and **Cluster Superblock** (`256 / 512 / 1024 / 4,096 (e-means)`).
- View live `.qlsa` sidecar detection status or click **"Build `.qlsa` (SVD)"** to generate `<model>/model.qlsa` in one click.
- All LSA settings are persisted **per model** in `~/.qwg/profiles/<model>.json`.

### 4. Unified Codebase & Knowledge Memory Packs (`qwg index` + `<repo>/.qwg/*.qwcache`)

Building on **Latent Sparse Attention** (`arXiv:2610.42147`), Qwegn supports precomputing an entire code repository or documentation tree into a persistent, pinned **Tier-0 Base Memory Pack** (`<root>/.qwg/manifest.json` + `<root>/.qwg/<pack>_<model>_<kv>.qwcache`) that sits underneath the per-session LRU prefix cache (`~/.qwg/cache/*.qwcache`). Full architectural analysis, comparisons against Vector RAG and Stanford SNAP Memorilla (`arXiv:2610.21375`), and experimental verification are detailed in the [**LSA Codebase Memory Whitepaper**](lsa-codebase-memory-whitepaper.md).

<p align="center">
  <img src="../assets/qwegn_memory_pack_architecture.svg" alt="Qwegn Memory Pack Architecture and Paradigm Comparison" width="100%">
</p>
<p align="center">
  <img src="../assets/qwegn_memory_pack_experiments.svg" alt="Qwegn Memory Pack Live Experimental Benchmarks" width="100%">
</p>

#### Measured Repository Memory Pack Performance (`27,472` Indexed Tokens)

Paired interleaved A/B measurements on the `qwegine` repository (`12` Tier-0/1 files, `27,472` cached repository tokens + `116` user prompt tokens = `27,588` total tokens, Apple M5 Max 128 GB, `--temperature 0`):

| Model Pack | `.qwcache` Artifact | Cold 27.6K TTFT / Prefill | Mounted `.qwcache` TTFT / Effective Prefill | Live `<memory_delta>` TTFT | Decode Speed | Grounded Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.6-35B-A3B-hq4g32`** (`kv=q4`) | `232.95 MiB` (`6.5 s` build) | `4.42 s` / `6,239.3 tok/s` | **`0.08 s` (`55.2×`)** / **`372,137 tok/s` (`59.6×`)** | **`0.42 s`** (`65,806 tok/s`) | `135.9 tok/s` | **`4/4` (100%)** vs `0/4` cold |
| **`Qwen3.8-Flash-Next-hq8g32-hq4g64`** (`kv=q8`) | `574.26 MiB` (`15.0 s` build) | `11.22 s` / `2,460.6 tok/s` | **`0.17 s` (`66.0×`)** / **`162,036 tok/s` (`65.9×`)** | **`0.60 s`** (`46,620 tok/s`) | `81.5 tok/s` | **`4/4` (100%)** vs `0/4` cold |

#### Step-by-Step Usage Manual

```bash
# 1. Build a repository LSA memory pack offline
#    Automatically reads kvMode from ~/.qwg/profiles/<model>.json (fallback: q4)
.build/release/qwg index /path/to/repo \
  --model ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --max-tokens 65536

# 2. Interactive REPL inside the repository (auto-discovers .qwg/manifest.json)
cd /path/to/repo && .build/release/qwg repl --model Qwen3.8-Flash-Next-hq8g32-hq4g64
# Inside REPL:
#   /memory                 -> inspect mounted repoId, commit, token count, enforced kvMode, and git delta
#   /memory unmount         -> detach active repository memory pack
#   /memory mount /path     -> mount another repository's .qwg/manifest.json
#   /cache                  -> inspect warm RAM and cold SSD cache residency

# 3. Single-shot CLI query with mounted repository memory (--no-memory-pack to bypass)
.build/release/qwg run ~/.qwg/models/Qwen3.6-35B-A3B-hq4g32 \
  --repo /path/to/repo \
  --prompt "Summarize AGENTS.md Law 20 and QWDashboardBuilder.build."

# 4. Coding-agent HTTP server (auto-discovers <repo>/.qwg/manifest.json from agent system prompts)
.build/release/qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 --port 8080 --no-lazy

# Explicit REST mount / unmount for external harnesses:
curl -X POST http://127.0.0.1:8080/v1/memory/mount \
  -H 'Content-Type: application/json' \
  -d '{"repoPath": "/path/to/repo"}'
curl -X POST http://127.0.0.1:8080/v1/memory/unmount
```

#### Core Capabilities & Guarantees

- **3-Tier Stability Ordering & 512-Token Superblock Alignment:** `qwg index` orders tracked files from most stable to most volatile (**Tier 0** Architecture/Manifests/Docs → **Tier 1** `Sources/**` sorted oldest-git-churn-first → **Tier 2** `Tests/**` & `Scripts/**` via `--include-tests`), aligns file boundaries to 512-token LSA Ward superblocks, respects `.gitignore` and `.qwgignore`, and registers `.qwg/` in `.git/info/exclude`.
- **Two-Tier Copy-On-Write (COW) Prefix Cache:** The repository `.qwcache` is mounted at token positions `0 ..< N_mem` with `isSharedBase = true` (protecting it from LRU eviction across up to 8 concurrent subagents), while session turns (`N_mem ..< N_total`) are cached in `~/.qwg/cache/*.qwcache`.
- **Zero-Rebuild Live Working-Tree Delta (`<|memory_delta|>`):** When tracked files are edited or deleted on disk after `qwg index`, Qwegn keeps `0 ..< N_mem` frozen (`100%` base cache hit) and appends only the modified/deleted files inside `<|memory_delta|>` at `N_mem`.
- **Automatic `kvMode` Enforcement & Settings Lock:** Mounting `.qwg/manifest.json` enforces the pack's `kvMode` (`q4` / `q8` / `f16`) across `qwg repl`, `qwg run`, `qwg serve`, and `qwg-menu` (locking the **KV Cache Mode** picker until unmounted).
- **Two-Tier Hierarchical LSA Cortex + Sub-Millisecond Off-Context Vault (`QWMemoryVault`):** By default, `qwg index` packs a **`32,768`-token Resident Cortex `.qwcache`** (`~3.1%` of the 1M context window: `264.61 MiB` on `Qwen3.6-35B-A3B` `q4` and `464.54 MiB` on `Qwen3.8-Flash-Next` `q4` — **`23.7×` smaller** than a monolithic 1M pack) using **domain-agnostic structural depth ($\text{depth}(f) \ge 0$) and temporal stability ordering** (with zero hardcoded folder names) plus a universal **`[QWG REPOSITORY CORTEX MAP]`** indexing 100% of files and their multi-language declarations, data schema keys (SQL/JSON/YAML/TOML/CSV), and document headings (Markdown/LaTeX), while storing all remaining files in the **Off-Context LSA Vault** (`.qwg/manifest.json`). On each turn, `QWMemoryVault.recallVaultContext` pages matching Vault files in **`< 0.5 ms`** into `<|memory_vault_recall|>` attached to the current turn's user message—preserving **100% `[warm_ram]` prefix cache identity** across all prior turns (`0.12–0.24 s` Cortex TTFT, **`150.5–163.9 tok/s` decode (`+53%` faster than 1M monolithic prefix)**, and **`1,015,834` tokens (`96.9%`) free** for coding agents under the **Elastic Context Budget Guard**).
