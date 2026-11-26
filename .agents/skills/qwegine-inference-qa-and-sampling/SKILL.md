---
name: qwegine-inference-qa-and-sampling
description: >-
  Use when modifying or debugging Qwegn runtime inference, sampling (QWSampler,
  repetition_penalty), multi-turn prefix caching (PrefixCache, QWFlashNext),
  speculative decoding (MTPVerification, DFlash 2, DSpine, .ngram, PLD),
  thinking budgets, constrained/tool decoding, or running live QWServer /
  llmprobe / 1M-context QA.
---

# Qwegn Inference Coherence, Sampling, `llmprobe` & Multi-Knob QA

Follow these five steps whenever modifying runtime inference, sampling, prefix caching, speculation, or `QWServer` endpoints.

## Procedure

1. **Enforce Generated-Only Sampling & Emitted-Token History (`QWEngine` + `QWFlashNext`):**
   - Scope `repetition_penalty` strictly to the trailing `256` generated tokens (`QWEngineSession.repetitionPenaltyWindow = 256`, `activeRepetitionHistory`, `FNGPUSessionData.recentGeneratedTokens`) using consecutive-cycle detection (`QWSampler.loopRepetitionTokens`: `4×` p1, `3×` p2, `2×` p3..96, `>= 6×` trailing-16 burst, 3rd trailing `8`-gram), keeping prompt tokens and `</think>` unpenalized.
   - Write only the actually sampled/emitted token (never parallel greedy `argmax`) into `tokenHistory`, `tokenRecycler`, `seenTokens`, `FNGPUSession.drawFirstToken`, and speculative seam/batch loops (`FNGPUEngine+Seam`, `FNGPUBatchScheduler+Drive`, `FNGPUDFlash`). Keep zero-copy greedy fast accept enabled via `!session.seenTokens.contains(draftTok)`.
2. **Preserve Multi-Turn Cache, Prefill/Decode Sync & Speculation Parity:**
   - Write live prefill head logits (`runsHead == true`), gate `tokenRecycler`/`.ngram`/`PLD`/`DFlash 2`/`DSpine` on `(constraint?.isUnconstrainedNow ?? true)`, pass `streamsCommittedRounds: true, onStopToken:` in `generateExperimentalStochastic` (note seeded stochastic runs execute serial intentionally), suppress `activeConstraint` on intermediate `.naive` prefill tokens (`DriverPrefill`), drain `session.lastCommitCB` before serial fallback in `DriverDecode.forwardStep`, and skip `mtpPromptEmbeddings` `memcpy` when `dflashDrafter != nil`.
   - Map `reasoning.effort` (`low` → `256`, `medium` → `1024`, `high` → `4096`, `xhigh` → `8192`) in `QWServerSchema.resolvedThinkingPolicy` (apply `reserved` cap when `enable_thinking: true` omits effort at `max_tokens > 32`) and clear stale `session.params.maxTokens` in `chat-presets.js`.
3. **Run Mandatory 9-Scenario Multi-Model Live `QWServer` QA Gate (`Law 28`):**
   - Confirm quiet GPU (`./Scripts/preflight.sh --require-quiet-gpu`), then run sequential QA across installed flagship families (`Dense/Looped`, `Dense`, `SHVQ MoE`, `Flash-Next`, `Hybrid MoE`) requiring `0` `ERROR`/`FATAL`/`panic` lines and clean `finish_reason == "stop"`:
     1. **`S1` Multi-turn WebUI:** 3 turns, warm prefix hits (`cached > 0`), `temperature: 0.4`, `top_p: 0.95`, `repetition_penalty: 1.5`, `tools`, `enable_thinking: false`.
     2. **`S2` Thinking budgets:** `low`/`medium`/`high`/`xhigh` + omitted effort at large `max_tokens` (`repetition_penalty: 1.5`).
     3. **`S3` Greedy + penalty sweep:** `temperature: 0.0` at `repetition_penalty` `1.0`, `1.05`, `1.5`.
     4. **`S4` Seeded determinism:** Fixed `seed` paired runs match exact length and text.
     5. **`S5` Uncached streaming:** `stream: true, cache_prompt: false` (`cached=0`, active `spec:` logs).
     6. **`S6` Structured JSON, logprobs & stops:** Schema conformance, finite `logprobs`, stripped `stop` sequences.
     7. **`S7` Forced tool call:** `tool_choice` valid JSON arguments.
     8. **`S8` Multi-protocol:** `/v1/responses` (`previous_response_id`), `/v1/completions` (`echo` + finite `logprobs`), `/v1/messages` (Anthropic `thinking` budget).
     9. **`S9` Concurrent `B=4` streaming:** 4 parallel `cache_prompt: false` requests with distinct answers (`0` bleed).
4. **Run Full `llmprobe` Server Qualification (`qwg serve`):**
   - On an idle GPU, launch `qwg serve` on an isolated port (e.g. `18299`) with `--context-tokens >= 40960` (since `llmprobe`'s `~32.8k` rung sends `32,865` tokens) and preserve outputs in `out/llmprobe/`:
     ```bash
     llmprobe http://127.0.0.1:18299 --full --eval --eval-suite code --reasoning off
     ```
   - Verify `0` HTTP/transport errors across `conformance`, `capability`, `agentic`, `fidelity`, and `bench`, steady `20m+` `loadDrift` (`0` UMA paging dips), deduplicated `"special": false` tool IDs, and clean parallel `<tool_call>` splitting.
5. **Run `1M`-Context & Long-NIAH Budget Qualification:**
   - Calibrate looped/reasoning-capable models (`Inkling-2-Mini`) with an `8K` dry run first (`--max-tokens >= 1024`, `hit_max_tokens == false`, `steady_allocs == 0`) before launching `16K / 128K / 300K` 3-needle passkeys; verify `1M` (`1,048,576`) UMA admission (`planned <= available`, `0` steady allocs) with a deterministic short prompt.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| `POST /v1/messages` returns `404` on `"qwegn"` model alias | Pass the loaded pack's exact `model` name (`strictNotFound: true` in `QWHttpServer+Anthropic`). |
| `S9` concurrent `B=4` hits `finish_reason == "length"` | Set `max_tokens >= 160` so verbose/LaTeX models (`SenseNova-Loop-9B`) finish naturally with `"stop"`. |
| Bare `assert` in Python QA harness logs empty error string | Format assertion errors explicitly as `f"{type(e).__name__}: {e}"`. |
| `llmprobe` fails at `32.8k` scaling rung | Restart `qwg serve` with `--context-tokens >= 40960` (`32,865` tokens exceed `32768`). |
| Looped model hits `max_tokens` during `--no-thinking` NIAH | Calibrate at `8K` with `--max-tokens >= 1024` and grade extracted final-answer blocks. |
