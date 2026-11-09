// QWUnifiedEngine.swift — Unified Top-Level Inference Engine Handle (QWEngine).
// Loads and serves any supported model architecture from its config/pack
// through a single concrete engine handle, compiled layer stack, and batch scheduler.

import Foundation
import QWCore
import QWHead
import QWKV
import QWQuant

/// Single unified inference engine handle capable of loading and serving any supported model architecture.
public final class QWEngine: QWInferenceEngine, @unchecked Sendable {
    public let descriptor: QWModelDescriptor
    public let plan: QWEnginePlan
    public var compiledStack: QWCompiledLayerStack {
        QWCompiledLayerStack(plan: plan, descriptor: descriptor)
    }
    public let underlying: any QWInferenceEngine
    public let batchScheduler: QWBatchScheduler

    public init(
        engine: any QWInferenceEngine,
        descriptor: QWModelDescriptor,
        plan: QWEnginePlan? = nil
    ) {
        self.underlying = engine
        self.descriptor = descriptor
        let resolvedPlan: QWEnginePlan
        if let plan {
            resolvedPlan = plan
        } else if let sessionEngine = engine as? any QWSessionInferenceEngine {
            resolvedPlan = sessionEngine.plan
        } else if let moduleEngine = engine as? any QWInferenceModule {
            resolvedPlan = moduleEngine.plan
        } else {
            let geom = QWModelGeometry(descriptor: descriptor)
            resolvedPlan = try! QWEngineBootstrap.makePlan(
                geometry: geom,
                packPath: descriptor.family.rawValue,
                prefillChunkRows: QWPrefillLimits.chunkRows,
                kvStorageMode: "f16"
            )
        }
        self.plan = resolvedPlan
        self.batchScheduler = QWBatchScheduler(engine: engine)
    }

    /// Loads the unified engine for any model pack and build options, resolving the architecture from the pack's own descriptor.
    public convenience init(
        options: QWEngineBuildOptions,
        descriptor: QWModelDescriptor? = nil,
        moduleType: (any QWInferenceModule.Type)? = nil
    ) throws {
        let resolvedDescriptor: QWModelDescriptor
        if let descriptor {
            resolvedDescriptor = descriptor
        } else {
            let location = try QWModelLocation(url: URL(fileURLWithPath: options.packPath), assetsOverride: nil)
            let configURL = location.assets.appendingPathComponent("config.json")
            resolvedDescriptor = try QWModelDescriptor.decode(configJSON: Data(contentsOf: configURL))
        }
        let resolvedModule: any QWInferenceModule.Type
        if let moduleType {
            resolvedModule = moduleType
        } else if let found = QWModuleRegistry.moduleType(for: resolvedDescriptor.family) {
            resolvedModule = found
        } else {
            throw QWModuleError.unsupportedFamily(resolvedDescriptor.family)
        }
        let engine = try resolvedModule.makeEngine(options: options)
        self.init(engine: engine, descriptor: resolvedDescriptor)
    }

    public var capabilities: Set<QWKnobCapability> { underlying.capabilities }
    public var hasPrefixCache: Bool { underlying.hasPrefixCache }
    public var prefixCache: QWPrefixCacheState { underlying.prefixCache }

    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { underlying.onCacheEvent }
        set { underlying.onCacheEvent = newValue }
    }

    public var report: QWEngineReport { underlying.report }
    public var lastTurn: QWGenerationResult? { underlying.lastTurn }
    public var supportsTokenConstraints: Bool { underlying.supportsTokenConstraints }

    public func makeConversation() -> QWConversation? {
        underlying.makeConversation()
    }

    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> QWGenerationResult {
        try batchScheduler.generate(request: request, onToken: onToken)
    }

    public func buildStandalonePrefixCache(tokens: [Int], outputURL: URL, pinAsSharedBase: Bool = true) throws -> Int {
        try underlying.buildStandalonePrefixCache(tokens: tokens, outputURL: outputURL, pinAsSharedBase: pinAsSharedBase)
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        try underlying.mountStandalonePrefixCache(from: fileURL)
    }

    public func unmountSharedBasePrefixes() {
        underlying.unmountSharedBasePrefixes()
    }

    /// Engine-neutral context-memory prediction for sizing UIs and admission
    /// preflight. Delegates to the residency planner; consumers must never name
    /// a concrete engine (the protocol guard enforces this).
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
            physicalMemoryBytes: physicalMemoryBytes)
    }
}
