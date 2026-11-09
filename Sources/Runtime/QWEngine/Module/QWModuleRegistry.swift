// QWModuleRegistry.swift — Which module serves which model family.
// The one admission point: a pack is served by the module its family names,
// and nothing else loads a model.
//
// This is the only place in the tree that switches on the family. Every
// other consumer asks the registry, so a new family is one case here.

import Foundation
import QWCore
import QWHAL
import QWKV

/// Maps a model family to the module that serves it.
public enum QWModuleRegistry {
    /// The module type for a family, or nil when no module serves it.
    public static func moduleType(for family: QWModelDescriptor.Family) -> (any QWInferenceModule.Type)? {
        switch family {
        case .qwen38Dense: return DenseModule.self
        case .qwen38FlashNext: return FlashNextModule.self
        case .qwen2Dense: return Qwen2DenseModule.self
        case .llamaDense: return LlamaModule.self
        case .mistralDense: return MistralModule.self
        case .qwen3Dense: return Qwen3DenseModule.self
        case .qwen3MoE: return Qwen3MoEModule.self
        }
    }

    /// Every family a module serves. Derived from the family list and the
    /// module map, so a new family cannot slip past either one.
    public static var supportedFamilies: [QWModelDescriptor.Family] {
        QWModelDescriptor.Family.allCases.filter { moduleType(for: $0) != nil }
    }

    /// The family a pack declares, when it declares one we know.
    public static func family(forPackModelId modelId: String) -> QWModelDescriptor.Family? {
        if let direct = supportedFamilies.first(where: { $0.rawValue == modelId }) {
            return direct
        }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelId),
           supportedFamilies.contains(spec.family) {
            return spec.family
        }
        return nil
    }

    /// The module that serves a pack's declared model id.
    public static func moduleType(forPackModelId modelId: String) -> (any QWInferenceModule.Type)? {
        family(forPackModelId: modelId).flatMap(moduleType(for:))
    }

    /// Builds the module that serves a loaded model.
    /// - Throws: `unsupportedFamily` when no module serves the plan's family,
    ///   `unsupportedCapability` when the module cannot honour what it advertises.
    public static func makeModule(plan: QWEnginePlan,
                                  engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        guard let type = moduleType(for: plan.family) else {
            throw QWModuleError.unsupportedFamily(plan.family)
        }
        for capability in requiredCapabilities(advertised: plan.modelCapabilities) {
            guard type.capabilities.contains(capability) else {
                throw QWModuleError.unsupportedCapability(model: plan.modelId, capability: capability)
            }
        }
        return try type.make(plan: plan, engine: engine)
    }

    /// The capabilities the module that serves a family offers. A model
    /// that is not served by any module offers none.
    public static func capabilities(for family: QWModelDescriptor.Family) -> Set<QWKnobCapability> {
        moduleType(for: family)?.capabilities ?? []
    }

    /// Builds the engine that serves a family. The caller names no
    /// engine type: the registry asks the module that serves the family.
    public static func makeEngine(for family: QWModelDescriptor.Family,
                                  options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        guard let type = moduleType(for: family) else {
            throw QWModuleError.unsupportedFamily(family)
        }
        return try type.makeEngine(options: options)
    }

    /// The capabilities a model must find in the module that serves it.
    /// The model's own config advertises them; the module implements them.
    public static func requiredCapabilities(
        advertised: Set<QWModelDescriptor.Capability>) -> Set<QWKnobCapability> {
        var required: Set<QWKnobCapability> = []
        if advertised.contains(.vision) { required.insert(.vision) }
        if advertised.contains(.mtp) { required.insert(.speculative) }
        return required
    }

    /// Predicts total GPU memory required (weights + KV + scratch) for a model descriptor and context length.
    public static func estimateContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String? = nil,
        contextTokens: Int,
        kvMode: QWKVMode = .f16,
        mtpDrafts: Int = 0,
        mtpBackend: String = "variational",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        tuning: QWEngineTuning = .default,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate {
        QWEngineDriver.estimateContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes
        )
    }
}
