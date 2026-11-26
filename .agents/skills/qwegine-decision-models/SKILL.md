---
name: qwegine-decision-models
description: >-
  Use when implementing, optimizing, quantizing, or qualifying Qwegn System-1
  decision models (QWDecision, QWDecisionPackEngine, d1-2B, QWCognitiveRails),
  decision HTTP/MCP APIs (/v1/decisions, /v1/systemone, /v1/qwegn, /v1/embeddings,
  /v1/rerank, /v1/cognitive/*, /mcp), dual-slot UMA sizing, or decision evals.
---

# Qwegn System-1 Decision Models, Cognitive Rails & Compliance Runbook

Follow these four gates whenever modifying `QWDecision`, `QWDecisionPackEngine`, `QWCognitiveRails`, decision Metal kernels, or decision/embedding/rerank HTTP & MCP endpoints.

## Procedure

1. **Zero-Shim External Wire & MCP Compliance Gate:**
   - Never certify APIs solely from `QWDecisionServerTests`. Run `swift test --no-parallel --filter QWDecisionTests` first, then test unmodified external clients against live `qwg serve`:
     - `jeval check --model Kev` (`POST /v1/systemone`), `jeval check --model OpenAI-Decisions` (`POST /v1/decisions`), `typesafe-sdk` (`models.list()`, `system_one` text, RFC 7578 `multipart/form-data` `Image(Path(...))`), `s1kit test --only qwegn` (`0` errors), Qwegn concise golden replay (`POST /v1/qwegn` `10/10` + `60,000`-char rail `HTTP 400`), and `/mcp` (`qwg_decide`, `qwg_ask`, `qwg_guard`, `qwg_status`, `qwg_rerank_screen`).
   - Verify UTF-8 multibyte text (CJK + emoji) emits Unicode character indices in `char_span`, `always_emit_box` meets `>= 90%` valid boxes on `ImageJevBench-Box`, `reasoning.effort = "auto"` emits CORS-exposed `X-Qwegn-Decision-Route` (failing open below `0.70`), and `.sequencePair` cross-encoder batches dispatch natively on `/v1/rerank`.
2. **ADR-009 Calibration-Gated Quantization & Accuracy Benchmarks:**
   - Gate every quantized decision pack against BF16 + CPU PyTorch oracle on `ΔECE <= 0.005`, `ΔBrier <= 0.003`, and `Top-1 >= 99.0%` (`ADR-009`) — reject formats that keep Top-1 while breaking calibration.
   - Qualify `/v1/systemone` on `JevBench` (`231` questions, `0` failed, NLL/ECE parity), `ImageJevBench-Box` (`50` items, `>= 90%` valid boxes), and `3×` CLI `qwg decide` bit-exact determinism.
3. **Standalone Boot, Honest UMA Sizing (`<= 10%` Error) & 6-Knob Parity:**
   - Confirm `qwg serve <decision-pack> --no-lazy` auto-routes to the decision slot (`/health` `loaded: true`), `--no-load-vision` drops vision RSS and refuses `--image` nonzero, and `/v1/decisions/status` `estimated_bytes` matches live RSS deltas within `10%` across `hq8`/`bf16` (`load_vision` `true`/`false`).
   - Keep the 6 `--decision-*` knobs synchronized across `QWKnobRegistry.swift`, `Scripts/gen_knob_cli.py`, `/v1/settings`, `SettingsView`, and `/ui`, verifying `swift test --no-parallel --filter QWCliGuiWebParityTests`.
4. **Metal 4 NAX Prefill, Coalesced Batching & Spike-Gated ANE (`ADR-019`):**
   - Keep decision prefill on Apple10 NAX (`qwg_gemm_bf16_nax_coop_fast`, `qwg_attn_topology_prefill_nax_d64`/`d128`, `qwg_cast_f16_to_f32`), reuse `QWDecisionScratchArena`, coalesce `/v1/decisions/batch` + `/v1/systemone/batch`, and verify quiet-GPU paired A/B (`Scripts/paired_perf.py`) across `41`, `1035`, `4059`, `8091` tokens + dual-slot co-residency.
   - Keep any ANE `CoreML` `IOSurface` offload gated inside `Scripts/spike/` (`>= +10%` net `tok/s` under `16 req/s` MCP + `100%` SHA-256 parity) before touching `Sources/`.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| Quantized decision pack hits `100%` Top-1 but `ΔECE > 0.005` | Reject and delete the candidate; switch to mixed `hq8_g32` + `bf16` (`ADR-009` calibration gate). |
| `char_span` offsets drift on CJK or emoji prompts | Emit Unicode scalar/character indices rather than raw UTF-8 byte offsets. |
| `ImageJevBench-Box` valid boxes fall below `90%` | Pass `--decision-always-emit-box` (`always_emit_box: true`) so best-effort coordinates emit below threshold. |
| `--no-load-vision` fails silently or crashes on `--image` | Ensure vision weight guard exits nonzero before encoder dispatch and verify RSS delta matches `estimated_bytes`. |
| `estimateResidentBytes` deviates `> 10%` from live RSS | Audit both backbone planes and resident vision buffer overhead in `QWDecisionWeightLoader.estimateResidentBytes`. |
