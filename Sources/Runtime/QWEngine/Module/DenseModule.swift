// DenseModule.swift — Unified dense, MoE, and hybrid family module.
// Drives `QWEngineDriver` directly behind `QWInferenceModule` / `QWInferenceEngine`.

import Foundation
import QWCore
import QWHead
import QWKV

/// Unified engine module serving dense, MoE, and hybrid families via `QWEngineDriver`.
public final class DenseModule: QWInferenceModule, QWInferenceEngine, @unchecked Sendable {
    public static let family: QWModelDescriptor.Family = .qwen38Dense

    /// Full capability set offered by the unified dense/hybrid driver.
    public static let capabilities: Set<QWKnobCapability> = [
        .prefixCache, .kvQuantQ8, .kvQuantQ4, .gdnPrefill, .dflash, .ropeYaRN,
        .speculative, .mtpAdaptive, .vision, .lsa,
    ]

    /// `QWKVMode` supports f16, q8, and q4 storage modes.
    public static let supportedKVStorageModes: Set<String> = ["f16", "q8", "q4"]

    /// Every kernel door the dense engine dispatches.
    public static let requiredDoors: [String] = QWKernelInventory.all

    /// Kernel doors required for a specific model geometry.
    public static func requiredDoors(for descriptor: QWModelDescriptor) -> [String] {
        (descriptor.hcCount ?? 0) > 1 ? QWKernelInventory.flash : QWKernelInventory.dense
    }

    /// KV cache storage modes supported for a specific model geometry.
    public static func supportedKVStorageModes(for descriptor: QWModelDescriptor) -> Set<String> {
        QWModelCapabilities.supportedKVStorageModes(for: descriptor)
    }

    public static func resolveSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        try defaultResolveSpeculation(
            requested, descriptor: descriptor, environment: environment, modelSwitch: modelSwitch)
    }

    /// The dense engine. Nil before the engine is loaded.
    public let engine: QWEngineDriver?
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    private var storedOnCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    /// Where the module reports what its prefix cache did.
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { engine?.onCacheEvent ?? storedOnCacheEvent }
        set {
            storedOnCacheEvent = newValue
            engine?.onCacheEvent = newValue
        }
    }
    /// The kernel each layer of the loaded model runs.
    private let kernels: [any QWLayerKernel]

    public init(engine: QWEngineDriver) {
        self.engine = engine
        self.plan = engine.plan
        self.kernels = engine.layerKernels
    }

    /// A module with no engine yet. Calls requiring a loaded engine fail loud.
    public init(plan: QWEnginePlan) {
        self.engine = nil
        self.plan = plan
        self.kernels = []
    }

    public static func make(plan: QWEnginePlan,
                            engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        let usesLatentOrCompressed = plan.usesAnyMLA
            || plan.layers.contains { $0.attention == .hcaCompressed }
        let isHyperStreamPlan = plan.modelCapabilities.contains(.ple)
            || (plan.modelCapabilities.contains(.hyperConnections) && plan.modelCapabilities.contains(.qsaIndexer))
        if !usesLatentOrCompressed {
            try QWEngineDriver.validateServing(
                headDim: plan.headDim,
                kvHeads: plan.kvHeads,
                qHeads: plan.qHeads,
                kvMode: plan.kvStorageMode,
                allowF32: isHyperStreamPlan
            )
        }
        guard let engine else { return DenseModule(plan: plan) }
        guard let dense = engine as? QWEngineDriver else {
            throw QWModuleError.unsupportedFamily(family)
        }
        return DenseModule(engine: dense)
    }

    public var capabilities: Set<QWKnobCapability> {
        engine?.capabilities ?? Self.capabilities
    }

    public var prefixCache: QWPrefixCacheState {
        engine?.prefixCache ?? QWPrefixCacheState(isConfigured: false)
    }

    public var report: QWEngineReport {
        engine?.report ?? QWEngineReport(
            capacityLog: "", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0
        )
    }

    public var lastTurn: QWGenerationResult? { engine?.lastTurn }
    public var supportsTokenConstraints: Bool { engine?.supportsTokenConstraints ?? true }

    public func makeConversation() -> QWConversation? {
        engine?.makeConversation()
    }

    public func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.generate(request: request, onToken: onToken)
    }

    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.buildStandalonePrefixCache(
            tokens: tokens,
            outputURL: outputURL,
            pinAsSharedBase: pinAsSharedBase,
            onPrefillProgress: onPrefillProgress
        )
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        guard let engine else {
            throw QWModuleError.notImplemented("DenseModule has no engine yet")
        }
        return try engine.mountStandalonePrefixCache(from: fileURL)
    }

    public func unmountSharedBasePrefixes() {
        engine?.unmountSharedBasePrefixes()
    }

    /// The kernel each layer of the loaded model runs. The plan resolved the
    /// kinds when the model was loaded; the hot loop looks a kernel up.
    public var layerKernels: [any QWLayerKernel] { kernels }

    /// The KV storage mode the plan was frozen with.
    public var kvStorageMode: String { plan.kvStorageMode }

    /// Whether a prefix cache is configured on the engine right now.
    public var hasPrefixCache: Bool { engine?.hasPrefixCache ?? false }

    /// Configures the prefix cache on the underlying engine when present.
    public func configurePrefixCache(_ config: QWCacheConfig) throws {
        guard let engine else {
            throw QWModuleError.notImplemented(
                "DenseModule.configurePrefixCache — engine not loaded")
        }
        try engine.configurePrefixCache(config: config)
    }
}

/// The dense engine is a loaded engine the registry can hand over.
extension QWEngineDriver: QWLoadedEngine {}

extension DenseModule {
    /// Builds the dense engine for a configuration. The module owns how
    /// its engine is made and configured, so the consumer names no
    /// concrete engine: it asks the module for an engine.
    public static func makeEngine(
        packPath: String,
        maxTokens: Int,
        kvMode: QWKVMode,
        mtpDrafts: Int,
        mtpBackend: QWMTPBackend?,
        prefillScratchEviction: QWPrefillScratchEvictionPolicy,
        gdnPrefill: String,
        cacheConfig: QWCacheConfig,
        mtpAdaptiveDepth: Bool,
        mtpAdaptiveEntropyThreshold: Double,
        mtpRelaxedMargin: Double = 1.5,
        mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid",
        mtpQuadraticSeeding: Bool = true,
        ropeYarnFactor: Double?,
        dflashDraftPackPath: String?,
        dflashSelectorTopK: Int,
        dsparkConfidenceSchedule: Bool? = nil,
        prefillChunkSize: Int,
        lsaConfig: QWLSAConfig = QWLSAConfig(),
        residentExpertsPerLayer: Int? = nil,
        osReserveGiB: Double? = nil,
        tuning: QWEngineTuning = .default,
        rawKVMode: String? = nil,
        prefixCacheDirectory: URL? = nil,
        maxConcurrentSessions: Int? = nil
    ) throws -> QWEngineDriver {
        let engine = try QWEngineDriver(
            packPath: packPath,
            maxTokens: maxTokens,
            kvMode: kvMode,
            rawKVMode: rawKVMode,
            mtpDrafts: mtpDrafts,
            mtpBackend: mtpBackend,
            prefillScratchEviction: prefillScratchEviction,
            mtpAdaptiveDepth: mtpAdaptiveDepth,
            mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
            mtpRelaxedMargin: mtpRelaxedMargin,
            mtpDraftStrategy: mtpDraftStrategy,
            mtpQuadraticSeeding: mtpQuadraticSeeding,
            ropeYarnFactor: ropeYarnFactor,
            dflashDraftPackPath: dflashDraftPackPath,
            dflashSelectorTopK: dflashSelectorTopK,
            dsparkConfidenceSchedule: dsparkConfidenceSchedule,
            osReserveGiB: osReserveGiB,
            prefillChunkSize: prefillChunkSize,
            lsaConfig: lsaConfig,
            residentExpertsPerLayer: residentExpertsPerLayer,
            tuning: tuning,
            prefixCacheConfig: cacheConfig,
            prefixCacheDirectory: prefixCacheDirectory,
            maxConcurrentSessions: maxConcurrentSessions
        )
        engine.mtpRelaxedMinProb = min(1.0, max(0.0, mtpRelaxedMinProb))
        switch gdnPrefill.lowercased() {
        case "naive": engine.gdnPrefillMode = GDNPrefillMode.naive
        case "chunked": engine.gdnPrefillMode = GDNPrefillMode.chunked
        case "regres": engine.gdnPrefillMode = GDNPrefillMode.regres
        default: break
        }
        if engine.standardState != nil && cacheConfig.warmByteLimit > 0 {
            try engine.configurePrefixCache(config: cacheConfig)
        }
        return engine
    }

    /// Builds the unified engine from what a consumer asked for.
    public static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        let rawMode = options.kvMode.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let parsedKVMode: QWKVMode = switch rawMode {
        case "f16", "f32": .f16
        case "q8": .q8
        case "q4": .q4
        default: throw QWModuleError.invalidState("unknown KV mode '\(options.kvMode)'")
        }
        return try makeEngine(
            packPath: options.packPath,
            maxTokens: options.maxTokens,
            kvMode: parsedKVMode,
            mtpDrafts: options.mtpDrafts,
            mtpBackend: options.mtpBackend,
            prefillScratchEviction: options.prefillScratchEviction,
            gdnPrefill: options.gdnPrefill,
            cacheConfig: options.cacheConfig
                ?? QWCacheConfig(warmByteLimit: 0, ssdByteLimit: 0,
                                 cacheDirectory: options.prefixCacheDirectory
                                    ?? QWCacheConfig.defaultCacheDirectory,
                                 wipeOnStart: false),
            mtpAdaptiveDepth: options.mtpAdaptiveDepth ?? false,
            mtpAdaptiveEntropyThreshold: options.mtpAdaptiveEntropyThreshold ?? 0,
            mtpRelaxedMargin: options.mtpRelaxedMargin,
            mtpRelaxedMinProb: options.mtpRelaxedMinProb,
            mtpDraftStrategy: options.mtpDraftStrategy,
            mtpQuadraticSeeding: options.mtpQuadraticSeeding,
            ropeYarnFactor: options.ropeYarnFactor,
            dflashDraftPackPath: options.dflashDraftPackPath,
            dflashSelectorTopK: options.dflashSelectorTopK ?? 16,
            dsparkConfidenceSchedule: options.dsparkConfidenceSchedule,
            prefillChunkSize: options.prefillChunkSize,
            lsaConfig: options.lsaConfig,
            residentExpertsPerLayer: options.residentExpertsPerLayer,
            osReserveGiB: options.osReserveGiB,
            tuning: options.tuning,
            rawKVMode: rawMode,
            prefixCacheDirectory: options.prefixCacheDirectory,
            maxConcurrentSessions: options.maxConcurrentSessions
        )
    }

    /// The KV layout the consumer asked for. A layout this family does
    /// not implement is refused, never silently replaced.
    static func kvMode(from raw: String) throws -> QWKVMode {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "f16": return .f16
        case "q8": return .q8
        case "q4": return .q4
        default: throw QWModuleError.invalidState("unknown KV mode '\(raw)'")
        }
    }
}
