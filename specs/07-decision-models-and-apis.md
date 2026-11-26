# 07 — Generalized Decision Models, Dual-Slot Sidecar & Decision APIs

> Normative specification for Qwegn's capability-generalized System-1 Decision Engine (`L0.0 QWCore`, `L1.0 QWToken`, `L1.5 QWDecision`, `L6.0 QWMM`, `L4–L7` adapters, `QWServer`, `QWCli`, and `Qwegn.app`), codifying `ADR-001`–`ADR-012`.

## 1. Five-Axis Model Generalization (`QWDecisionModelConfig`)

Decision execution never branches on `descriptor.family` or `model_type` strings outside `QWCore` config decoding and `QWLoad` tensor resolution (`Scripts/lint_capability_seams.py`). Instead, every checkpoint reduces to **5 orthogonal capability axes** inside `QWDecisionModelConfig`:

1. **Sequence Framing (`SequenceFraming`)**:
   - `.turnVerbalizer`: Prompt/verbalizer template (`colon` or chat-template slot formatting, `codesPerOption` `1|2`, `numbers: Bool`, `packSet: Bool`, `stateSeparator`, `\u200b` special-token neutralization). Used by `d1-2B`, `d1-3B`, `Microsoft-Decision-1`/`1.1`, `ShieldGemma`, and **any** existing generative Qwegn LLM via synthesized `QWChatTemplate` verbalizers.
   - `.optionMarkers`: Bidirectional `[MASK]` / `<|mask|>` marker placement in either `.questionOptionsState` (`[CLS] ... [MASK] ...`) or `.stateQuestionOptions` (`<bos> <state> ... <mask> ...`), with dynamic option/state token budgeting and `<¦name¦>` marker escaping. Used by `d1-omni-600M`, `openjev`, and TRL `AutoModelForDecision`.
   - `.sequencePair`: Single-pair or $K$-pair batch construction. Supports cross-encoder $K$-way fan-out for `1`-logit / `2`-logit relevance or NLI rerankers (`Qwen3-Reranker`, `BGE-Reranker`, `MiniLM`) and strict normalized `id2label` subset matching for fixed $C \ge 3$ classifiers (`ModernBERT`, `DeBERTa`).

2. **Attention Topology (`AttentionTopology`)**:
   - `.causal(supportsPrefixCache: Bool)`
   - `.prefixBidirectional` (bidirectional over state/prefix tokens, causal over query suffix)
   - `.promptBidirectional`
   - `.packedSet(maxSetSize: Int)` (`state` sees `[state + all questions]` on full attention; each `question_i` sees `[state + question_i]` bidirectionally; sibling questions masked across sets; `sliding_attention` and `ShortConv` remain causal)
   - `.bidirectionalEncoder(isolateMediaPrefix: Bool)` (full self-attention, optionally isolating right-side image/audio media prefixes for `ShortConv(centered)`)

3. **Readout Head (`HeadKind`)**:
   - `.vocabVerbalizer(vocabSize: Int, tieEmbeddings: Bool, sparseGatherThreshold: Int)` — evaluates only $U \le 256$ unique candidate vocabulary rows via softmax translation invariance (`0`-ULP equivalent to full `log_softmax` on $128\text{K–}248\text{K}$ vocabs), falling back to batched `QWHead` when $U > 256$.
   - `.markerScorer(hiddenSize: Int, headLayers: Int, chains: [String])` — `type_emb` broadcast + optional pre-norm encoder stack + marker gather + single/multi-chain MLP (`LayerNorm -> Linear -> GELU/SiLU -> Linear(D, 1)`).
   - `.pooledClassifier(pooling: PoolingKind, numLabels: Int, id2label: [Int: String])` — `.firstToken`, `.lastToken`, or `.meanNonPad` pooling + dense/classifier projection.
   - `.jointOptionProjector(hiddenSize: Int)` — projected option/query scorer without full vocab matrix multiplication.
   - Orthogonal sub-heads: `hasStatePointer: Bool` (`statePointer`: `querySlots[2*R, H] @ stateHidden[W, H]^T / sqrt(H)` + `max_pool1d` over `1..stateEnd`), `gridReadout` (`Estimate` + `Box`/`Boxes` 4D Halton IoU), and `boxEdgesMode` (`median` vs `interp`).

4. **Calibration Profile (`QWCalibrationEngine`)**:
   - Unified multi-source loader normalizing `calibration.json` (bucket `temperatures` + PAV isotonic knot `tables`), `config.json` inline `temperatures`, `rl_agent_config.json` per-chain temperatures, and Platt scaling parameters.
   - Bucket resolution order matches upstream conventions: exact match (`"choice:\(k)"`, `"score:\(k)"`, `"noul"`, `"noul:2"`), closed-range match (`"choice:small"`/`mid`/`large`/`inventory`, `"choice:2"`/`3-5`/`6-10`/`11+`, `"score:3-5"`/`6-10`), `"default"`, else `1.0`.
   - Enforces `roundPy` 10-decimal half-to-even normalization (`massSlack = 1e-12`, residual renorm onto insertion-order argmax, `residualMax = 1e-9`) so $\sum p_i = 1.0 \pm 10^{-12}$ identically across HTTP/MCP dialects.

5. **Backbone & Multimodal Capabilities**:
   - `ShortConvConfig(kernelSize: Int, hasBias: Bool, isCentered: Bool, stateKey: String?)` (`k = 3..7`, causal ring vs centered bidirectional with media prefix boundary).
   - `NormKind` (`.rmsNorm(eps:hasUnitOffset:)` vs `.layerNorm(eps:hasBias:)`) and `useSandwichNorm: Bool`.
   - `PerKindRoPEConfig` supporting per-layer-kind `ropeTheta` + `partialRotaryFactor` + `zeroTailPartialRotation: Bool`.
   - `AttnGateMode` (`.none`, `.packedElementWiseQProj`, `.separateHeadWiseSigmoid`).
   - Vision (`VisionTowerConfig`: `LFM3-VL` hybrid `VisionConv2d` + 2D Axial-RoPE and `SigLIP2` NaFlex/fixed-patch, honoring `--load-vision` / `--no-load-vision`) in Phase 1 (`M1–M5`); 16 kHz Audio (`QWAudio`) deferred to Phase 2 (`M6`).

---

## 2. Architecture Seam & Layer Placement (`L0.0`, `L1.0`, `L1.5`, `L7.0`, `L7.8`)

```
L0.0 QWCore     ── QWDecisionModelConfig, ShortConvConfig, NormKind, PerKindRoPEConfig,
                   AttnGateMode, QWGPUTurnArbitrator, QWWeightSourceKind.coreAIModel,
                   QWResidencyAdmission.admitDecisionSidecar
L1.0 QWToken    ── QWTokenizer+Offsets (encodeWithOffsets, singleTokenId, allSpecialTokenStrings)
L1.5 QWDecision ── Zero-Metal deep subsystem:
                   QWDecisionTypes, QWDecisionExecutionBackend & QWEmbeddingExecutionBackend,
                   QWSequenceFramingEngine, QWCalibrationEngine, QWCalibrationFitter,
                   QWTypedAnswerDecoder, QWDecisionRunner, QWCognitiveRails (7 policies)
L6.0 QWMM       ── Unified EXIF-oriented vImage RGB888 slab preprocessing
                   (QWMM+OrientedImage: qwInspectImageHeader, qwRenderOrientedRGBSlab,
                   qwDecodeOrientedRGBSlab) shared by System-1 & QWImageDecoder
L7.0 QWEngine   ── QWDecisionPackEngine (+Weights/+HeadWeights/+Forward/+Heads:
                   vocabVerbalizer, markerScorer, pooledClassifier, jointOptionProjector,
                   LayerNorm, GELU/GELU-tanh, centered ShortConv, bidirectional masks,
                   embedBatch Matryoshka L2), QWEngineDecisionAdapter, QWCoreAIDecisionAdapter
L7.8 QWServer   ── QWResidencyPool multi-slot UMA orchestrator, dual-slot & standalone lifecycle,
                   coalesced batching, /v1/decisions[/batch], /v1/systemone[/batch], /v1/qwegn,
                   /v1/embeddings, /v1/rerank, /v1/cognitive/*, QWEpisodicTranscriptStore (0o600),
                   Stateless HTTP & stdio MCP Cognitive+Vault suite (qwg mcp), /ui?tab=decisions
```

**Backend Parity Guarantee:** `L1.5 QWDecision` depends strictly on `QWCore` and `QWToken` (never `Metal` or `QWHAL`). Every unit test runs natively on Apple Silicon and virtualized `GITHUB_ACTIONS=true` runners (`QWSyntheticDecisionBackend` lives strictly in `Tests/QWDecisionTests/`, never in `Sources/`).

---

## 3. Binding Architectural Decision Records (`ADR-001`–`ADR-010`)

- **ADR-001 (Dual-Slot & Standalone Co-Residency):** `QWEngineLifecycleActor` hosts an optional `decisionSidecar` slot (`decisionModelPackPath`) alongside the primary LLM, falling back to the primary LLM when unset or when `model` names the primary model, and also boots standalone via `qwg serve <decision-pack> --no-lazy`. `QWDecisionWeightLoader.estimateResidentBytes(root:descriptor:loadVision:)` includes resident vision buffers (`<= 10%` UMA error across `hq8`/`bf16` × `vis0`/`vis1`), and `QWResidencyAdmission.admitDecisionSidecar` enforces `primaryWired + sidecarEstimate + warmCache <= availableBytes` (clamping warm cache first; non-zero refusal before allocation).
- **ADR-002 (Step-Boundary GPU Preemption):** `QWGPUTurnArbitrator` (`OSAllocatedUnfairLock` + Swift 6 `withCheckedContinuation`) coordinates primary Metal decode/prefill steps with queued Metal sidecar calls at step boundaries (`~10–25 ms` p99 wait). CoreAI ANE `.aimodel` sidecars bypass the arbitrator (`0.0 ms` GPU contention).
- **ADR-003 (Phase 1 Text/JSON/Vision Scope):** Phase 1 (`M1–M5`) supports Text, JSON, and Vision (JSON base64/URLs and RFC 7578 `multipart/form-data`) across all 4 decision paradigms. Raw 16 kHz audio (`QWAudio`) is deferred to Phase 2 (`M6`).
- **ADR-004 (Heterogeneous Metal 4 + CoreAI ANE Backends):** `QWDecisionExecutionBackend` supports Metal 4 (`QWEngineDecisionAdapter`), macOS 27 CoreAI ANE (`#if canImport(CoreAI) @available(macOS 27.0, *) QWCoreAIDecisionAdapter` for `.aimodel` bundles discovered via `combined_provenance.json` + `*.aimodel`), and CI test fake.
- **ADR-005 (Multi-Dialect Wire APIs, Coalesced Batching & `'mean'` Score Default):** `QWServer` serves `POST /v1/decisions` (+ coalesced `/batch`), `POST /v1/systemone` (+ coalesced `/batch`), `POST /v1/qwegn`, and stateless `POST /mcp` (`qwg_decide`, `qwg_ask`, `qwg_guard`, `qwg_status`), publishing decision capability metadata in `GET /v1/models`. Unified default `score_readout` is `"mean"` ($\sum i \cdot p_i$, `spread_reported: true`), honoring explicit `"median" | "argmax" | "mean"`.
- **ADR-006 (Sparse Candidate-Row Verbalizer Gather):** `.vocabVerbalizer` batches with $U \le 256$ unique candidate tokens project only the $U$ rows of `lm_head` (`softmax(z_cand / T) == softmax(log_softmax(z_full)[cand] / T)` with `0`-ULP drift).
- **ADR-007 (Cross-Encoder Reranker Fan-Out & Fixed-Head Honesty):** `1/2`-logit `.pooledClassifier` checkpoints evaluate arbitrary $K$-way `choice`/`score` options via single-batch $K$-pair cross-encoder fan-out; fixed $C \ge 3$ `id2label` heads validate option subsets case-insensitively (`allowed_labels` HTTP `400` on mismatch).
- **ADR-008 (Native Context Cap & Honest Truncation):** `/v1/qwegn` and `qwg_ask` enforce the `60,000`-char wire rail; `/v1/decisions` and `/v1/systemone` accept up to full `contextBudget`. Fixed-window `.optionMarkers` / ANE encoders auto-truncate `state` (`usage.truncated_tokens`) while preserving options; `.turnVerbalizer` and `Read` pointer heads default to `truncation: "error"` (`400`) unless `"head" | "tail" | "middle"` is requested.
- **ADR-009 (Calibration-Preserving Safetensors & `qwg calibrate`):** Unquantized Safetensors decision packs ($\le 8\text{B}$) load at `BF16`/`HQ8_G32` rather than online `HQ4` to preserve shipped `calibration.json` tables; `qwg calibrate` fits NLL bucket temperatures + PAV monotone knot tables + 15-bin `ECE` / Brier diagnostics; `4-bit` `.qwp` decision packs require $\Delta\text{ECE} \le 0.005$, $\Delta\text{Brier} \le 0.003$, Top-1 $\ge 99.0\%$.
- **ADR-010 (Opt-In `reasoning.effort="auto"`, Configuration Parity & UI):** When `reasoning.effort == "auto"` (across `off | low | medium | high | xhigh | auto`) and a decision sidecar is loaded on a thinking template, `QWCognitiveRails.routeTurn` evaluates a `< 10 ms` 3-tier complexity + costly-risk rubric (`["trivial", "moderate", "deep"]`) with a sticky `>= 32K` context guard (`confidence >= 0.7`), routing `off | low | xhigh`, failing open otherwise, and emitting `X-Qwegn-Decision-Route` (CORS-exposed). All 6 decision knobs (`--decision-model` / `QWG_DECISION_MODEL`, `--decision-compute-unit` / `QWG_DECISION_COMPUTE_UNIT`, `--decision-load-vision` / `QWG_DECISION_LOAD_VISION`, `--decision-score-readout` / `QWG_DECISION_SCORE_READOUT`, `--decision-truncation` / `QWG_DECISION_TRUNCATION`, `--decision-always-emit-box` / `QWG_DECISION_ALWAYS_EMIT_BOX`) stay synchronized across CLI, `/v1/settings`, `Qwegn.app` `SettingsView`, and `/ui` Settings; `Qwegn.app` and `/ui?tab=decisions` render from single-source `QWDashboardViewModel` and never report `0`-token decisions in `decodeRateSeries`.
- **ADR-011 (`QWCognitiveRails`, Episodic Recovery & Cognitive/Vault MCP Suite):** Seven pure-Swift deterministic rails in `QWDecision` (`guardCommand` red lines + `0.70` acting rail + fail-closed `confirm`; `screenAndRerankPassages` local injection screen + coalesced dual-gate rerank + local-only fallback; `selectCompaction` pinning `system`/pinned/tail-6 turns + `0.70` drop rail; `selectSkills` filtering `existsOnDisk` + 120-skill Stage-1 shortlist + `needs_skill` Stage-2 verify; `evaluateSearchSufficiency` query dedup + conflict detection; `superviseDelegatedRun` exit-code & `180 s` silence fact overrides + `alert_after = 2` streak; `routeTurn` `on|shadow|off` + sticky `32K` guard) back `/v1/cognitive/*` and the expanded `POST /mcp` / `qwg mcp` tool suite (`qwg_rerank_screen`, `qwg_compact_select`, `qwg_episodic_recover`, `qwg_skill_select`, `qwg_search_enough`, `qwg_supervise`, `qwg_route_turn`, `qwg_vault_query`, `qwg_vault_status`) alongside `QWEpisodicTranscriptStore` (`0o700`/`0o600`, BM25 recovery) and workspace-contained Vault queries.
- **ADR-012 (Generalized Decision Pack Engine, Dense Embeddings & Shared `QWMM` Image Preprocessing):** `QWDecisionPackEngine` generalizes natively across all 4 `HeadKind`s (`.vocabVerbalizer`, `.markerScorer`, `.pooledClassifier`, `.jointOptionProjector`), `NormKind.layerNorm`, `GELU`/`GELU-tanh` (`qwg_vis_gelu_erf_f16` / `qwg_vis_gelu_tanh_f16`), centered/causal `ShortConv`, bidirectional/isolated-media attention topologies, and `QWEmbeddingExecutionBackend.embedBatch` (L2-normalized Matryoshka truncation powering `POST /v1/embeddings` and `POST /v1/rerank`); `QWMM+OrientedImage` serves as the single EXIF-oriented `vImage` RGB888 preprocessor shared by System-1 vision inputs and `QWImageDecoder`.

---

## 4. Question Types, Decoding Math & Invariants (`L1.5`)

- **`noul` / `predicate`:** Binary `["no", "yes"]` (`index 0 = no`, `index 1 = yes`; `p_yes = probs[1]`, `answer: Bool = p_yes >= 0.5`).
- **`choice` / `multiple` (`abstain_below` rail):** Calibrated categorical distribution `probs` (`probs.count == options.count`). Single choice picks first maximum in insertion order; optional `abstainBelow` overrides winner with `nil` + `abstained: true` when `max_i p_i < threshold`. Multi-choice computes sigmoid calibrated marginals with optional `selectThreshold`. Both `confidence` (`max_i p_i`) and `entropyConfidence` (`1 - H(p) / log(K)`) are returned cleanly.
- **`score`:** $L$-point ordinal scale (`L >= 2`), indices `0..<L`. Readouts: `"mean"` ($\sum_{i=0}^{L-1} i \cdot p_i$), `"median"` (exact `0.5` crossing on discrete CDF with linear interpolation), `"argmax"` (`Double(argmax_i p_i)`). Spread `sigma = sqrt(max(0, E[i^2] - E[i]^2))`, always reported as `spread_reported: true` for `"mean"` and `"median"`.
- **`read` (`extractText`, `extractNumber`, `extractDate`, `path`):** `found = -expm1(startLogProb0)`. `max_pool1d` over `1..stateEnd` across span lengths up to caps (`text: 512`, `number: 32`, `date: 32`) maximizing `startLogProb[s] + endLogProb[e]`, mapped through bijective `QWTokenizedSpanList` offsets (`trim()` for text; `Double` / ISO-8601 regex normalization for `number` / `date`, or RFC 6901 JSON pointer token extraction on `json_spans.string_paths`), emitting UTF-8 `char_span` (`charSpan`) alongside byte/scalar `span` offsets.
- **`estimate` (`[lo, hi]`):** Softmax quantile bins on uniform grid yielding `value` (`0.5` median) and `[p10, p90]` confidence interval.
- **`box` / `boxes`:** 4-edge (`x1, y1, x2, y2`) probability distributions over normalized `[0, 1]` (`Grid(0, 1, 128)`) or thousandths (`Grid(0, 0.999, 1000)` when `numbers: true`), supporting `"median"` and `"interp"` edge readouts and `always_emit_box` (`alwaysEmitBox`) to decode best-effort coordinates even below presence threshold. `confidence` equals fraction of 512 quasi-random 4D Halton `(2, 3, 5, 7)` draws satisfying $\text{IoU}(\hat{B}, B^{(d)}) \ge 0.5$. Multi-box (`boxes`) clamps monotone slot presence `P_k = min_{j <= k} p_{exist,j}`, keeps slots with `P_k >= threshold` (or `[1, 2]` count), and sorts deterministically by `(yCenter, xCenter)`.
