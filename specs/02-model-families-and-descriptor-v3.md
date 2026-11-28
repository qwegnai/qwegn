# 02 — Zero-Overload Model Families & `geometryDigestV3`

> Normative specification for `QWModelDescriptor`, `QWModelRole`, `QWModelFamilyRegistry`, config-sourced geometry, and digest versioning (`Sources/Foundation/QWCore/`).

## 1. Zero-Overload 19-Family Contract
Every HuggingFace `model_type` maps 1:1 through `QWModelFamilyRegistry` to an explicit `QWModelDescriptor.Family` case; no family case may be overloaded onto another architecture's traits or kernels. Unknown `model_type` strings throw `QWModelDescriptorError.unsupportedModelType`.

| `QWModelDescriptor.Family` | `model_type` keys in `builtinSpecs` | Role / Key Architectural Traits |
| :--- | :--- | :--- |
| `.qwen38Dense` | `qwen3_5`, `qwen3_5_text` | `defaultAttnOutputGate: true`, `+1` RMSNorm, Qwen chat template IDs |
| `.qwen38MoE` | `qwen3_5_moe`, `qwen3_5_moe_text` | Sparse MoE + attention output gate, `+1` RMSNorm |
| `.qwen38FlashNext` | `qwen4_exp`, `qwen4_exp_text` | Hybrid QSA + GDN + HyperPLE (`maxSpeculativeDrafts: 3`, sigmoid expert gate) |
| `.senseNovaLoop` | `sensenova_loop`, `sensenova_loop_text` | Looped middle blocks (`LoopSettings`) + `MHCConfig` Sinkhorn mixing |
| `.glm5Next` | `glm5_next`, `glm5_next_text` | Hybrid NoPE MLA (`HCAConfig`/`HybridMLAConfig`) + KDA, standard RMSNorm, sigmoid gate |
| `.ernie55MoE` | `ernie5_5_moe`, `ernie5_5_moe_text` | Latent-MoE + HCA/MLA-DSA hybrid, standard RMSNorm, sigmoid gate |
| `.maxwellMoE` | `maxwell_moe` | Block-sparse GQA (MBSA) + mHC + sigmoid MoE w/ F32 bias + dedicated mHC-MTP |
| `.qwen3Dense` | `qwen3` | Standard HF/MLX dense Qwen3 (`defaultAttnOutputGate: false`) |
| `.qwen3MoE` | `qwen3_moe` | Standard HF/MLX sparse MoE (`defaultAttnOutputGate: false`) |
| `.qwen2Dense` | `qwen2` | Standard dense Qwen2 (`defaultAttnOutputGate: false`) |
| `.llamaDense` | `llama` | Standard dense Llama (`defaultAttnOutputGate: false`) |
| `.mistralDense` | `mistral` | Registered descriptor (`isOfficialSupported: false`; trait-decoded `slidingWindow` routed through `QWAttnDecodeApply` general decode) |
| `.inkling2Loop` | `inkling2_loop`, `inkling2_loop_text` | Looped hybrid GDN + full attention (`mhc` mixing) |
| `.bailingMoeV3` | `bailing_moe_v3`, `bailing_moe_v3_text` | Sparse MoE + hybrid gate (`bailing_moe_v3`) |
| `.lfm2` | `lfm2`, `lfm2_text`, `lfm2_vl` | Causal GQA + ShortConv-3 (`conv` + `full_attention`), optional `SigLIP2` vision & verbalizer decision config (`d1-3B`) |
| `.lfm3` | `lfm3`, `lfm3_text`, `lfm3_vl` | Sandwich RMSNorm + ShortConv-7 + SWA-256 + `full_attention` + nested `PerKindRoPEConfig` + optional `LFM3-VL` vision & packed-set decision config (`d1-2B`) |
| `.lfm2Encoder` | `d1_omni`, `lfm2_encoder` | Bidirectional ShortConv (`isCentered: true`) + full encoder attention + `.optionMarkers` / multi-chain `.markerScorer` (`d1-omni-600M`, `.aimodel` CoreAI) |
| `.modernBert` | `modernbert`, `bert`, `roberta`, `deberta`, `deberta-v2`, `xlm-roberta` | Bidirectional `LayerNorm` encoder (`global`/`local` RoPE) + `.pooledClassifier` / `.markerScorer` (`SequenceClassification` & cross-encoder rerankers) |
| `.gigachat4` | `gigachat4` | Hybrid GDN (`GDNGatingConfig` sigmoid `*2.0`) + decoupled-RoPE MLA-DSA (`alphaQ`/`alphaKV` + YaRN `softmaxScale`), `GatedNormConfig` rank-16 `pre_post` sandwich norms, sigmoid bias-corrected MoE, `finalLogitSoftcapping = 30.0`, heterogeneous dense NextN MTP |

## 2. Explicit Execution Roles (`QWModelRole`)
`QWModelDescriptor.role` (`typealias Role = QWModelRole`) separates execution semantics:
- `.causalLM`: text-only autoregressive decoder.
- `.multimodalConditionalLM`: vision-language model declaring `.vision` capability and verified `QWVisionConfig`.
- `.speculativeDrafter`: dedicated draft model (e.g., `DSpine` / `DFlash 2`); verified against target compatibility before attachment and rejected if invoked standalone as `--model` without target verifier context.
- `.decisionEncoder`: encoder or option-marker checkpoint emitting logits/probabilities (`output_tokens == 0`, e.g., `d1-omni`, `ModernBERT`, `DeBERTa`), exempt from autoregressive `eos_token_id` requirements when omitted by checkpoint config.

## 3. Config-Sourced Geometry & Linter Gate
- Geometry, layer cadence, RoPE/YaRN parameters, MoE/Latent/HCA/MLA fields, and `QWVisionConfig` come strictly from the model's `assets/config.json` (and companion pack files under `~/.qwg/models`), never hardcoded or synthesized when absent (`Scripts/lint_geometry_defaults.py` enforces zero hidden geometry fallbacks across `Sources/`).
- `QWModelDescriptor.verifyInternalConsistency()` re-derives `qProjOut`, `rotaryDims`, MRoPE sums, `gdnQKVChannels`, `gdnZChannels`, full-attention cadence, and PLE linear-layer IDs prior to runtime plan construction.
- `QWModelDescriptor.decode` uses unified trait-based decoding via `QWModelFamilyRegistry.shared` (`QWStandardConfigDecoder`), propagating `slidingWindow` (served via `QWAttnDecodeApply` general decode without fast-path shared-prefix assumptions), `tieWordEmbeddings`, and `normGammaShift`.
- All families resolve through `QWModuleRegistry` onto the unified `DenseModule` / `QWEngineDriver` (routing `hcCount > 1` / `.qwen38FlashNext` execution to `QWHyperStreamRuntime`, with `DriverWeightKernels.resolveHeadSpec` and `QWWeightAllocationPlan` supporting tied embeddings and `.nativeAffine` heads/MTP).

## 4. `geometryDigestV2` vs `geometryDigestV3`
- **`geometryDigestV2` (`qwg-geo-v2:...`)**: Preserved **byte-identical** for on-disk `.qwp` pack headers and `.qlsa` sidecar trailers so existing packs validate without re-packing.
- **`geometryDigestV3` (`qwg-geo-v3:...`)**: Extends `V2` with `family`, `role`, `env_type`, `kda`, `mla`, `loop`, `mhc`, native YaRN (`yarn=<factor>,<originalMaxPositionEmbeddings>` when `yarnRopeConfig != nil`), block-sparse GQA (`bsa=<blockSize>,<prefixBlocks>,<localBlocks>,<remoteBlocks>` when `blockSparseGQAConfig != nil`), vision geometry (`vis=...` when `.vision` capability is present), ShortConv (`sconv=<kernel>,<bias>,<centered>`), sandwich/layer normalization (`swnorm=<bool>,<kind>`), decision topology (`topo=...`), and decision head (`dhead=...`). Used for `QWEnginePlan`, compiled layer caches, and newly generated artifacts to eliminate cross-family/cross-role digest collisions while preserving existing model digests bit-identically when new fields are absent.
