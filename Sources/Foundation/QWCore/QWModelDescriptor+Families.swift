// QWModelDescriptor+Families.swift — the model-family types behind the descriptor.
import Foundation

/// Special token IDs associated with a model family's tokenizer and chat template.
public struct QWModelSpecialTokens: Sendable, Hashable, Equatable {
    public let eosTokenId: Int
    public let imStartTokenId: Int
    public let imEndTokenId: Int
    public let thinkTokenId: Int
    public let endThinkTokenId: Int
    /// Stop ids the checkpoint itself declares — a config may list several.
    public let extraStopTokenIds: [Int]
    /// `false` for a tokenizer that has no Qwen3.8 chat-template ids.
    public let usesChatTemplateIds: Bool

    public init(
        eosTokenId: Int = 248044,
        imStartTokenId: Int = 248045,
        imEndTokenId: Int = 248046,
        thinkTokenId: Int = 248068,
        endThinkTokenId: Int = 248069,
        extraStopTokenIds: [Int] = [],
        usesChatTemplateIds: Bool = true
    ) {
        self.eosTokenId = eosTokenId
        self.imStartTokenId = imStartTokenId
        self.imEndTokenId = imEndTokenId
        self.thinkTokenId = thinkTokenId
        self.endThinkTokenId = endThinkTokenId
        self.extraStopTokenIds = extraStopTokenIds
        self.usesChatTemplateIds = usesChatTemplateIds
    }

    /// Standard generation stop token IDs (`<|endoftext|>` and `<|im_end|>`).
    public var defaultStopTokenIds: Set<Int> {
        var stops = Set(extraStopTokenIds)
        stops.insert(eosTokenId)
        if usesChatTemplateIds { stops.insert(imEndTokenId) }
        return stops
    }
}

/// Specification binding a HuggingFace `model_type` identifier to a `QWModelDescriptor.Family`
/// and architecture-specific default traits.
public struct QWModelFamilySpec: Sendable, Hashable, Equatable {
    public let modelType: String
    public let family: QWModelDescriptor.Family
    public let defaultAttnOutputGate: Bool
    public let defaultExpertGateType: String?
    public let usesPlusOneRMSNorm: Bool
    public let usesQwenChatTemplateIds: Bool
    public let defaultHasQKNorm: Bool
    public let defaultAllFullAttention: Bool
    public let defaultNormTopKProb: Bool
    public let requiresSparseMoEConfig: Bool
    public let disallowsMoEConfig: Bool
    public let maxSpeculativeDrafts: Int
    public let supportsExpertStreaming: Bool
    public let isOfficialSupported: Bool
    public let defaultMHCEpsFromRMSNorm: Bool
    public let defaultBlockSparseGQA: Bool
    public let defaultMTPSplitEProjHProj: Bool
    public let defaultSharedExpertHasGate: Bool?
    public let defaultRopeInterleave: Bool
    public let defaultBailingVideoRope: Bool
    public let defaultSpecialTokens: QWModelSpecialTokens?
    public let defaultVisionMean: [Float]
    public let defaultVisionStd: [Float]
    public let defaultRole: QWModelRole
    public let defaultHeadDimFromHidden: Bool
    public let defaultPartialRotaryOne: Bool
    public let allowsAutoMap: Bool
    public let allowsOptionalEosTokenId: Bool
    public let usesLayerNormByDefault: Bool
    public let isHyperStreamTrunk: Bool
    public let isRecurrentLoopTrunk: Bool
    public let moeRouterEnableExpertBias: Bool
    public let defaultMTPIsDense: Bool
    public let defaultMTPAttentionIsSparse: Bool?
    public let defaultIndexerKNormIsRMS: Bool

    public init(
        modelType: String,
        family: QWModelDescriptor.Family,
        defaultAttnOutputGate: Bool,
        defaultExpertGateType: String? = nil,
        usesPlusOneRMSNorm: Bool? = nil,
        usesQwenChatTemplateIds: Bool? = nil,
        defaultHasQKNorm: Bool? = nil,
        defaultAllFullAttention: Bool? = nil,
        defaultNormTopKProb: Bool? = nil,
        requiresSparseMoEConfig: Bool? = nil,
        disallowsMoEConfig: Bool? = nil,
        maxSpeculativeDrafts: Int? = nil,
        supportsExpertStreaming: Bool? = nil,
        isOfficialSupported: Bool = true,
        defaultMHCEpsFromRMSNorm: Bool = false,
        defaultBlockSparseGQA: Bool = false,
        defaultMTPSplitEProjHProj: Bool = false,
        defaultSharedExpertHasGate: Bool? = nil,
        defaultRopeInterleave: Bool = false,
        defaultBailingVideoRope: Bool = false,
        defaultSpecialTokens: QWModelSpecialTokens? = nil,
        defaultVisionMean: [Float] = [0.5, 0.5, 0.5],
        defaultVisionStd: [Float] = [0.5, 0.5, 0.5],
        defaultRole: QWModelRole = .causalLM,
        defaultHeadDimFromHidden: Bool = false,
        defaultPartialRotaryOne: Bool = false,
        allowsAutoMap: Bool = false,
        allowsOptionalEosTokenId: Bool = false,
        usesLayerNormByDefault: Bool = false,
        isHyperStreamTrunk: Bool? = nil,
        isRecurrentLoopTrunk: Bool? = nil,
        moeRouterEnableExpertBias: Bool = false,
        defaultMTPIsDense: Bool = false,
        defaultMTPAttentionIsSparse: Bool? = nil,
        defaultIndexerKNormIsRMS: Bool = false
    ) {
        self.modelType = modelType
        self.family = family
        self.defaultAttnOutputGate = defaultAttnOutputGate
        self.defaultExpertGateType = defaultExpertGateType
        self.usesPlusOneRMSNorm = usesPlusOneRMSNorm ?? [.qwen38Dense, .qwen38MoE, .senseNovaLoop].contains(family)
        self.usesQwenChatTemplateIds = usesQwenChatTemplateIds
            ?? [.qwen38Dense, .qwen38MoE, .qwen38FlashNext, .senseNovaLoop].contains(family)
        self.defaultHasQKNorm = defaultHasQKNorm ?? ![.qwen2Dense, .llamaDense, .mistralDense, .gigachat4].contains(family)
        self.defaultAllFullAttention = defaultAllFullAttention
            ?? [.qwen3Dense, .qwen3MoE, .qwen2Dense, .llamaDense, .mistralDense].contains(family)
        self.defaultNormTopKProb = defaultNormTopKProb
            ?? [.qwen38MoE, .qwen38FlashNext, .glm5Next, .ernie55MoE, .maxwellMoE, .bailingMoeV3, .gigachat4].contains(family)
        self.requiresSparseMoEConfig = requiresSparseMoEConfig ?? (family == .qwen3MoE)
        self.disallowsMoEConfig = disallowsMoEConfig
            ?? [.qwen3Dense, .qwen2Dense, .llamaDense, .mistralDense, .qwen38Dense, .senseNovaLoop].contains(family)
        self.maxSpeculativeDrafts = maxSpeculativeDrafts ?? 15
        self.supportsExpertStreaming = supportsExpertStreaming ?? true
        self.isOfficialSupported = isOfficialSupported
        self.defaultMHCEpsFromRMSNorm = defaultMHCEpsFromRMSNorm
        self.defaultBlockSparseGQA = defaultBlockSparseGQA
        self.defaultMTPSplitEProjHProj = defaultMTPSplitEProjHProj
        self.defaultSharedExpertHasGate = defaultSharedExpertHasGate
        self.defaultRopeInterleave = defaultRopeInterleave
        self.defaultBailingVideoRope = defaultBailingVideoRope
        self.defaultSpecialTokens = defaultSpecialTokens
        self.defaultVisionMean = defaultVisionMean
        self.defaultVisionStd = defaultVisionStd
        self.defaultRole = defaultRole
        self.defaultHeadDimFromHidden = defaultHeadDimFromHidden
        self.defaultPartialRotaryOne = defaultPartialRotaryOne
        self.allowsAutoMap = allowsAutoMap
        self.allowsOptionalEosTokenId = allowsOptionalEosTokenId
        self.usesLayerNormByDefault = usesLayerNormByDefault
        self.isHyperStreamTrunk = isHyperStreamTrunk ?? (family == .qwen38FlashNext)
        self.isRecurrentLoopTrunk = isRecurrentLoopTrunk ?? (family == .inkling2Loop)
        self.moeRouterEnableExpertBias = moeRouterEnableExpertBias
        self.defaultMTPIsDense = defaultMTPIsDense
        self.defaultMTPAttentionIsSparse = defaultMTPAttentionIsSparse
        self.defaultIndexerKNormIsRMS = defaultIndexerKNormIsRMS
    }
}

/// Thread-safe registry mapping HuggingFace `model_type` strings to `QWModelFamilySpec`.
/// Follows the Registry and Open/Closed patterns so additional model types can be registered
/// without modifying `QWModelDescriptor.decode` or `QWModelRuntimeAssets`.
public final class QWModelFamilyRegistry: @unchecked Sendable {
    public static let shared = QWModelFamilyRegistry()

    private let lock = NSLock()
    private var specs: [String: QWModelFamilySpec]

    public init(specs: [QWModelFamilySpec] = QWModelFamilyRegistry.builtinSpecs) {
        var map: [String: QWModelFamilySpec] = [:]
        for s in specs { map[s.modelType] = s }
        self.specs = map
    }

    public static let builtinSpecs: [QWModelFamilySpec] = [
        QWModelFamilySpec(modelType: "qwen3_5", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "qwen3_5_text", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "qwen3_5_moe", family: .qwen38MoE, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "qwen3_5_moe_text", family: .qwen38MoE, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "sensenova_loop", family: .senseNovaLoop, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "sensenova_loop_text", family: .senseNovaLoop, defaultAttnOutputGate: true),
        QWModelFamilySpec(
            modelType: "glm5_next",
            family: .glm5Next,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "glm5_next_text",
            family: .glm5Next,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "ernie5_5_moe",
            family: .ernie55MoE,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "ernie5_5_moe_text",
            family: .ernie55MoE,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "maxwell_moe",
            family: .maxwellMoE,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultMHCEpsFromRMSNorm: true,
            defaultBlockSparseGQA: true,
            defaultMTPSplitEProjHProj: true,
            defaultSharedExpertHasGate: false,
            defaultSpecialTokens: QWModelSpecialTokens(
                eosTokenId: 200001,
                imStartTokenId: 200000,
                imEndTokenId: 200001,
                thinkTokenId: 200006,
                endThinkTokenId: 200007,
                usesChatTemplateIds: true
            )
        ),
        QWModelFamilySpec(
            modelType: "bailing_moe_v3",
            family: .bailingMoeV3,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true,
            defaultNormTopKProb: true,
            defaultSharedExpertHasGate: false,
            defaultRopeInterleave: true,
            defaultBailingVideoRope: true,
            defaultSpecialTokens: QWModelSpecialTokens(
                eosTokenId: 156895,
                imStartTokenId: 157151,
                imEndTokenId: 156895,
                thinkTokenId: 156903,
                endThinkTokenId: 156904,
                usesChatTemplateIds: false
            ),
            defaultVisionMean: [0.48145466, 0.4578275, 0.40821073],
            defaultVisionStd: [0.26862954, 0.26130258, 0.27577711]
        ),
        QWModelFamilySpec(
            modelType: "bailing_moe_v3_vl",
            family: .bailingMoeV3,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true,
            defaultNormTopKProb: true,
            defaultSharedExpertHasGate: false,
            defaultRopeInterleave: true,
            defaultBailingVideoRope: true,
            defaultSpecialTokens: QWModelSpecialTokens(
                eosTokenId: 156895,
                imStartTokenId: 157151,
                imEndTokenId: 156895,
                thinkTokenId: 156903,
                endThinkTokenId: 156904,
                usesChatTemplateIds: false
            ),
            defaultVisionMean: [0.48145466, 0.4578275, 0.40821073],
            defaultVisionStd: [0.26862954, 0.26130258, 0.27577711]
        ),
        QWModelFamilySpec(
            modelType: "bailing_moe_v3_text",
            family: .bailingMoeV3,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true,
            defaultNormTopKProb: true,
            defaultSharedExpertHasGate: false,
            defaultRopeInterleave: true,
            defaultBailingVideoRope: true,
            defaultSpecialTokens: QWModelSpecialTokens(
                eosTokenId: 156895,
                imStartTokenId: 157151,
                imEndTokenId: 156895,
                thinkTokenId: 156903,
                endThinkTokenId: 156904,
                usesChatTemplateIds: false
            ),
            defaultVisionMean: [0.48145466, 0.4578275, 0.40821073],
            defaultVisionStd: [0.26862954, 0.26130258, 0.27577711]
        ),
        QWModelFamilySpec(
            modelType: "qwen4_exp",
            family: .qwen38FlashNext,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid",
            maxSpeculativeDrafts: 3,
            supportsExpertStreaming: false,
            isHyperStreamTrunk: true
        ),
        QWModelFamilySpec(
            modelType: "qwen4_exp_text",
            family: .qwen38FlashNext,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid",
            maxSpeculativeDrafts: 3,
            supportsExpertStreaming: false,
            isHyperStreamTrunk: true
        ),
        QWModelFamilySpec(
            modelType: "inkling2_loop",
            family: .inkling2Loop,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false,
            defaultAllFullAttention: false,
            defaultNormTopKProb: false,
            maxSpeculativeDrafts: 15,
            supportsExpertStreaming: false,
            isOfficialSupported: true,
            isRecurrentLoopTrunk: true
        ),
        QWModelFamilySpec(
            modelType: "inkling2_loop_text",
            family: .inkling2Loop,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false,
            defaultAllFullAttention: false,
            defaultNormTopKProb: false,
            maxSpeculativeDrafts: 15,
            supportsExpertStreaming: false,
            isOfficialSupported: true,
            isRecurrentLoopTrunk: true
        ),
        // The standard HF/MLX families. None packs an attention output gate.
        QWModelFamilySpec(
            modelType: "qwen3", family: .qwen3Dense, defaultAttnOutputGate: false,
            defaultHasQKNorm: true, defaultAllFullAttention: true,
            defaultNormTopKProb: false, isOfficialSupported: true
        ),
        QWModelFamilySpec(
            modelType: "qwen3_moe", family: .qwen3MoE, defaultAttnOutputGate: false,
            defaultHasQKNorm: true, defaultAllFullAttention: true,
            defaultNormTopKProb: false, requiresSparseMoEConfig: true, isOfficialSupported: true
        ),
        QWModelFamilySpec(
            modelType: "qwen2", family: .qwen2Dense, defaultAttnOutputGate: false,
            defaultHasQKNorm: false, defaultAllFullAttention: true, defaultNormTopKProb: false
        ),
        QWModelFamilySpec(
            modelType: "llama", family: .llamaDense, defaultAttnOutputGate: false,
            defaultHasQKNorm: false, defaultAllFullAttention: true, defaultNormTopKProb: false
        ),
        QWModelFamilySpec(
            modelType: "mistral", family: .mistralDense, defaultAttnOutputGate: false,
            defaultHasQKNorm: false, defaultAllFullAttention: true, defaultNormTopKProb: false,
            isOfficialSupported: false
        ),
        // LFM2 / LFM3 / LFM2-Encoder / ModernBERT decision & hybrid encoder families.
        QWModelFamilySpec(
            modelType: "lfm2", family: .lfm2, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "lfm2_text", family: .lfm2, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "lfm2_vl", family: .lfm2, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "lfm3", family: .lfm3, defaultAttnOutputGate: true,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "lfm3_text", family: .lfm3, defaultAttnOutputGate: true,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "lfm3_vl", family: .lfm3, defaultAttnOutputGate: true,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: true, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultHeadDimFromHidden: true,
            defaultPartialRotaryOne: true, allowsAutoMap: true
        ),
        QWModelFamilySpec(
            modelType: "d1_omni", family: .lfm2Encoder, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true
        ),
        QWModelFamilySpec(
            modelType: "lfm2_encoder", family: .lfm2Encoder, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true
        ),
        QWModelFamilySpec(
            modelType: "modernbert", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "bert", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "roberta", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "deberta", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "deberta-v2", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "xlm-roberta", family: .modernBert, defaultAttnOutputGate: false,
            usesPlusOneRMSNorm: false, usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false, defaultAllFullAttention: false, defaultNormTopKProb: false,
            disallowsMoEConfig: true, maxSpeculativeDrafts: 0, supportsExpertStreaming: false,
            isOfficialSupported: true, defaultRole: .decisionEncoder,
            defaultHeadDimFromHidden: true, defaultPartialRotaryOne: true,
            allowsAutoMap: true, allowsOptionalEosTokenId: true, usesLayerNormByDefault: true
        ),
        QWModelFamilySpec(
            modelType: "gigachat4",
            family: .gigachat4,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false,
            defaultHasQKNorm: false,
            defaultAllFullAttention: false,
            defaultNormTopKProb: true,
            isOfficialSupported: true,
            defaultSharedExpertHasGate: false,
            defaultRopeInterleave: true,
            defaultSpecialTokens: QWModelSpecialTokens(
                eosTokenId: 2,
                imStartTokenId: 128000,
                imEndTokenId: 128001,
                thinkTokenId: 128014,
                endThinkTokenId: 128015,
                extraStopTokenIds: [2, 128001],
                usesChatTemplateIds: true
            ),
            allowsAutoMap: true,
            moeRouterEnableExpertBias: true,
            defaultMTPIsDense: true,
            defaultMTPAttentionIsSparse: false,
            defaultIndexerKNormIsRMS: true
        ),
    ]

    public func register(_ spec: QWModelFamilySpec) {
        lock.lock()
        defer { lock.unlock() }
        specs[spec.modelType] = spec
    }

    public func unregister(modelType: String) {
        lock.lock()
        defer { lock.unlock() }
        specs.removeValue(forKey: modelType)
    }

    public func spec(forModelType modelType: String) -> QWModelFamilySpec? {
        lock.lock()
        defer { lock.unlock() }
        return specs[modelType]
    }

    public func spec(for family: QWModelDescriptor.Family) -> QWModelFamilySpec? {
        lock.lock()
        defer { lock.unlock() }
        return specs.values.first { $0.family == family }
    }

    public func isSupported(modelType: String) -> Bool {
        spec(forModelType: modelType)?.isOfficialSupported == true
    }
}
