---
name: qwegine-model-onboarding
description: >-
  Use when onboarding a new HuggingFace model architecture or variant to Qwegn
  (config decoding, capability structs, QWP1 packer adapters, 2-layer PyTorch
  oracles, synthetic E2E tests, and pack qualification).
---

# Qwegn New Model Onboarding & Oracle Runbook

Follow these five steps when adding support for a new HuggingFace checkpoint architecture without model-coupled hacks.

## Procedure

1. **Audit HF Config, Python Defaults & Synthetic Test Constraints:**
   - Read `config.json`, `configuration_*.py` (`PretrainedConfig.__init__` defaults omitted from `config.json`), and `modeling_*.py` side by side before editing `Sources/`; strip `*template*` keys and never ingest `*.jinja` chat templates into context (`Law 24`).
   - Verify boolean/dimension flags verbatim against the real checkpoint (e.g., `use_mla_nope`, `num_kv_heads_for_linear_attn == 0` fallback, `vision_config.num_position_embeddings`, 3D MRoPE cursor delta).
   - Keep synthetic MoE test fixtures at `E % 32 == 0` (`32 <= E <= 1024`, `K <= 32`) for `QWGemmMoERouter`, and pass `mtpBackend: nil` whenever `mtpDrafts == 0` (`"standard"` requires `1...15`).
2. **Decouple via Capability Seams (`Scripts/lint_capability_seams.py`):**
   - Extend capability structs in `QWCore` (`KDAConfig`, `MLAConfig`, `MoERoutingConfig`, `SwiGLUClampConfig`, `VisionTowerConfig`, `MTPConfig`, `HCAConfig`, `BlockSparseGQAConfig`, `MRoPELayout`) and decode in `QWModelDescriptor+Decode.swift`.
   - Express custom XML tool/thinking envelopes via `QWChatTemplate` properties (`toolCallEnvelope`, `thinkOpenTag`, `thinkCloseTag`, `usesStandardToolCallHint`) and iterative trie traversal in `QWToolCallConstraint.allowedTokens()`, then gate zero model-name leaks:
     ```bash
     python3 Scripts/lint_capability_seams.py
     ```
3. **Preserve QWP1 Packer Adapter Invariants (`Scripts/quantize/`):**
   - Check per-tensor `reader.has(name + "_scale_inv")` rather than global `modules_to_not_convert`, de-interleave Direct-Q MLA projections from per-head `[nope, rope]` to planar `[nope; rope]`, guard cold-expert routing permutations with `(geometry.moeRoutingConfig?.nGroup ?? 1) <= 1`, keep `< 32`-row projections (`b_proj`, `g_proj`) as raw `bf16` for load-time `fwht32OrthoInPlace`, and decode OCP `mxfp4` / FP8 through shared helpers in `Scripts/quantize/hq_blocks.py`.
   - Run packer unit tests via `.venv-oracle` `unittest`:
     ```bash
     .venv-oracle/bin/python3 -m unittest discover -s Scripts/quantize -p 'test_*.py'
     ```
4. **Verify 2-Layer PyTorch-to-Metal Oracle (`max_abs_diff < 5e-3` + Argmax Identity):**
   - Register pure-PyTorch shims into `sys.modules` for unavailable CUDA packages (`fla`, `einops`) with `_attn_implementation = "eager"`, pack a 2-layer synthetic checkpoint (`hq8_g32`), reload exact dequantized QWP1 tensors (re-interleaving MLA `q_proj`; keeping FP32 `gate.weight`/`expert_bias`/`A_log`/`dt_bias`), and assert text + `QWVisualSpan` parity via env-gated `XCTest` (`QW_*_ORACLE_DIR`).
5. **Pack, Audit & Qualify End-to-End:**
   - Name quantized packs per `AGENTS.md` Law 35 (`<Model>-QWP_{4,8,3,2,15,1}` or `<Model>-QWP-Mixed[-Fast]`) and document the exact quant recipe in `README.md`.
   - Run `./Scripts/preflight.sh --require-quiet-gpu`, pack + audit (`Scripts/quantize/pack.py` and `Scripts/quantize/audit_release.py`), verify greedy SHA-256 parity (`d=0` vs `d=1 --mtp-backend standard --mtp-relaxed-margin 0 --mtp-draft-strategy neural`) across `f16`/`q8`/`q4` KV, and complete `QWServer` `S1–S9` + vision + `16K+` needle + `llmprobe` + paired A/B vs `master`.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| Synthetic MoE test traps inside `QWGemmMoERouter` (`nExp = 8`) | Set fixture experts `E % 32 == 0` (`32 <= E <= 1024`, `K <= 32`) and `mtpBackend: nil` when `mtpDrafts == 0`. |
| `modeling_*.py` fails importing `fla` or `einops` | Inject pure-PyTorch stub modules into `sys.modules` before dynamic import and force `_attn_implementation = "eager"`. |
| `pytest` command not found inside `.venv-oracle` | Use `.venv-oracle/bin/python3 -m unittest ...` (venv uses stdlib `unittest` only). |
| Stray `.venv-oracle` or `Scripts/spike/out` symlink pollutes git worktree | Run `rm -f .venv-oracle Scripts/spike/out out/vision_decision_oracle` before `git commit` (`Law 2`). |
| Loaded checkpoint diverges at runtime on omitted config field | Inspect `configuration_*.py` `__init__` defaults and add missing default decoding in `QWModelDescriptor+Decode.swift`. |
