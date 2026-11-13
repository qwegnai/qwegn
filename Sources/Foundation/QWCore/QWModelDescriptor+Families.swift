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
    public let isOfficialSupported: Bool

    public init(
        modelType: String,
        family: QWModelDescriptor.Family,
        defaultAttnOutputGate: Bool,
        defaultExpertGateType: String? = nil,
        usesPlusOneRMSNorm: Bool? = nil,
        usesQwenChatTemplateIds: Bool? = nil,
        isOfficialSupported: Bool = true
    ) {
        self.modelType = modelType
        self.family = family
        self.defaultAttnOutputGate = defaultAttnOutputGate
        self.defaultExpertGateType = defaultExpertGateType
        self.usesPlusOneRMSNorm = usesPlusOneRMSNorm ?? (family == .qwen38Dense)
        self.usesQwenChatTemplateIds = usesQwenChatTemplateIds
            ?? (family == .qwen38Dense || family == .qwen38FlashNext)
        self.isOfficialSupported = isOfficialSupported
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
        QWModelFamilySpec(modelType: "qwen3_5_moe", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "qwen3_5_moe_text", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "sensenova_loop", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(modelType: "sensenova_loop_text", family: .qwen38Dense, defaultAttnOutputGate: true),
        QWModelFamilySpec(
            modelType: "glm5_next",
            family: .qwen38Dense,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "glm5_next_text",
            family: .qwen38Dense,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "ernie5_5_moe",
            family: .qwen38Dense,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "ernie5_5_moe_text",
            family: .qwen38Dense,
            defaultAttnOutputGate: false,
            defaultExpertGateType: "sigmoid",
            usesPlusOneRMSNorm: false,
            usesQwenChatTemplateIds: false
        ),
        QWModelFamilySpec(
            modelType: "qwen4_exp",
            family: .qwen38FlashNext,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid"
        ),
        QWModelFamilySpec(
            modelType: "qwen4_exp_text",
            family: .qwen38FlashNext,
            defaultAttnOutputGate: true,
            defaultExpertGateType: "sigmoid"
        ),
        // The standard HF/MLX families. None packs an attention output gate.
        // `qwen2`, `llama` and the `qwen3` families are officially supported;
        // `mistral` waits for the sliding window (C116).
        QWModelFamilySpec(modelType: "qwen3", family: .qwen3Dense, defaultAttnOutputGate: false, isOfficialSupported: true),
        QWModelFamilySpec(modelType: "qwen3_moe", family: .qwen3MoE, defaultAttnOutputGate: false, isOfficialSupported: true),
        QWModelFamilySpec(modelType: "qwen2", family: .qwen2Dense, defaultAttnOutputGate: false),
        QWModelFamilySpec(modelType: "llama", family: .llamaDense, defaultAttnOutputGate: false),
        QWModelFamilySpec(modelType: "mistral", family: .mistralDense, defaultAttnOutputGate: false, isOfficialSupported: false),
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

    public func isSupported(modelType: String) -> Bool {
        spec(forModelType: modelType)?.isOfficialSupported == true
    }

    public var registeredModelTypes: [String] {
        lock.lock()
        defer { lock.unlock() }
        return specs.keys.sorted()
    }
}

