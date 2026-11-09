// QWModelBackend.swift — Multi-model backend Strategy, Adapter, and Factory Registry.
// Decouples QWEngineLifecycleActor from the concrete model family drivers
// while keeping GPU hot-path execution 100% concrete inside each family's engine.

import Foundation
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWQuant
import QWVision

/// Input parameters for a single backend token generation pass.
public struct QWModelGenerationRequest: Sendable {
    public let promptTokens: [Int]
    public let maxNewTokens: Int
    public let sampling: QWSamplingParameters
    public let stopTokenIds: Set<Int>
    public let retainPrefixLength: Int?
    public let hasImages: Bool
    public let collectLogprobs: Bool
    public let topLogprobsCount: Int
    public let onTokenLogprob: (@Sendable (QWTokenLogprob) -> Void)?
    /// Request-local structured-output constraint; nil when unconstrained.
    public let constraint: (any QWTokenConstraint)?
    /// Forces exact target-model speculative verification (`mtpRelaxedMargin = 0`) for tool-calling turns.
    public let exactSpeculation: Bool
    /// Merged visual feature spans for multimodal prompts.
    public let visualSpans: [QWVisualSpan]

    public init(
        promptTokens: [Int],
        maxNewTokens: Int,
        sampling: QWSamplingParameters,
        stopTokenIds: Set<Int>,
        retainPrefixLength: Int? = nil,
        hasImages: Bool = false,
        collectLogprobs: Bool = false,
        topLogprobsCount: Int = 0,
        onTokenLogprob: (@Sendable (QWTokenLogprob) -> Void)? = nil,
        constraint: (any QWTokenConstraint)? = nil,
        exactSpeculation: Bool = false,
        visualSpans: [QWVisualSpan] = []
    ) {
        self.constraint = constraint
        self.exactSpeculation = exactSpeculation
        self.promptTokens = promptTokens
        self.maxNewTokens = maxNewTokens
        self.sampling = sampling
        self.stopTokenIds = stopTokenIds
        self.retainPrefixLength = retainPrefixLength
        self.hasImages = hasImages || !visualSpans.isEmpty
        self.collectLogprobs = collectLogprobs
        self.topLogprobsCount = topLogprobsCount
        self.onTokenLogprob = onTokenLogprob
        self.visualSpans = visualSpans
    }
}

/// Normalized completion telemetry returned by any `QWModelBackend` after generation.
public struct QWModelCompletionMetrics: Sendable {
    public let tokens: [Int]
    public let prefillSec: Double?
    public let decodeSec: Double?
    public let firstTokenSec: Double?
    public let activeContextCapacity: Int
    public let cacheHitTokens: Int
    public let cacheTier: String?
    public let cacheActive: Bool
    public let warmCacheBytes: Int?
    public let ssdCacheBytes: Int?
    public let speculativeSummary: String?
    /// The GPU milliseconds each stage spent, when the engine measured
    /// them. Absent for a stage nobody timed.
    public let stageGPU: [String: Double]
    public let tokenLogprobs: [QWTokenLogprob]

    public init(
        tokens: [Int],
        prefillSec: Double?,
        decodeSec: Double?,
        firstTokenSec: Double?,
        activeContextCapacity: Int,
        cacheHitTokens: Int = 0,
        cacheTier: String? = nil,
        cacheActive: Bool = false,
        warmCacheBytes: Int? = nil,
        ssdCacheBytes: Int? = nil,
        speculativeSummary: String? = nil,
        stageGPU: [String: Double] = [:],
        tokenLogprobs: [QWTokenLogprob] = []
    ) {
        self.tokens = tokens
        self.prefillSec = prefillSec
        self.decodeSec = decodeSec
        self.firstTokenSec = firstTokenSec
        self.activeContextCapacity = activeContextCapacity
        self.cacheHitTokens = cacheHitTokens
        self.cacheTier = cacheTier
        self.cacheActive = cacheActive
        self.warmCacheBytes = warmCacheBytes
        self.ssdCacheBytes = ssdCacheBytes
        self.speculativeSummary = speculativeSummary
        self.stageGPU = stageGPU
        self.tokenLogprobs = tokenLogprobs
    }
}

/// Strategy protocol abstracting model family loading, capabilities, and generation execution.
public protocol QWModelBackend: Sendable {
    var family: QWModelDescriptor.Family { get }
    var descriptor: QWModelDescriptor { get }
    var parametersInfo: QWEngineParametersInfo { get }
    var visionBackend: (any QWVisionBackend)? { get }
    var hardwareCapabilities: QWDeviceCaps? { get }
    /// The cache the engine holds. A byte count nobody measured is absent.
    var prefixCache: QWPrefixCacheState { get }
    /// What the engine behind this backend can do. Absence means "not
    /// available", never a zero.
    var capabilities: Set<QWKnobCapability> { get }
    /// Whether `generate` enforces `QWModelGenerationRequest.constraint`.
    var supportsTokenConstraints: Bool { get }

    func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int)
    func unmountSharedBasePrefixes()
}

/// The builder that makes a `QWModelBackend` for a module.
extension QWModelBackend {
    public var supportsTokenConstraints: Bool { false }
    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        ([], 0)
    }
    public func unmountSharedBasePrefixes() {}
}

public typealias QWModelBackendBuilder = @Sendable (
    QWEngineLifecycleActor.Config,
    QWModelDescriptor,
    QWServerLogger
) throws -> any QWModelBackend

/// A module and the builder that makes a backend for it.
public struct QWModelBackendRegistration: @unchecked Sendable {
    public let module: Any.Type
    public let builder: QWModelBackendBuilder

    public init(module: Any.Type, builder: @escaping QWModelBackendBuilder) {
        self.module = module
        self.builder = builder
    }
}

/// Thread-safe registry of the backend builders, one per module. The
/// lookup asks the module registry which module serves a family: the
/// server never names a family to load one.
public final class QWModelBackendRegistry: @unchecked Sendable {
    public static let shared = QWModelBackendRegistry()

    private let lock = NSLock()
    private var builders: [ObjectIdentifier: QWModelBackendRegistration]

    public init(registrations: [QWModelBackendRegistration] = []) {
        var map: [ObjectIdentifier: QWModelBackendRegistration] = [:]
        for registration in registrations {
            map[ObjectIdentifier(registration.module)] = registration
        }
        self.builders = map
    }

    public func register(module: Any.Type, builder: @escaping QWModelBackendBuilder) {
        lock.lock()
        defer { lock.unlock() }
        builders[ObjectIdentifier(module)] = QWModelBackendRegistration(module: module, builder: builder)
    }

    public func builder(for module: Any.Type) -> QWModelBackendBuilder? {
        lock.lock()
        let override = builders[ObjectIdentifier(module)]?.builder
        lock.unlock()
        if let override { return override }
        guard let moduleType = module as? any QWInferenceModule.Type else { return nil }
        return { config, descriptor, logger in
            try QWUnifiedModelBackend.load(moduleType: moduleType, config: config,
                                           descriptor: descriptor, logger: logger)
        }
    }

    /// The modules with explicitly registered backend overrides.
    public var registeredModules: [Any.Type] {
        lock.lock()
        defer { lock.unlock() }
        return builders.values.map(\.module)
    }

    public func loadBackend(
        config: QWEngineLifecycleActor.Config,
        descriptor: QWModelDescriptor,
        logger: QWServerLogger
    ) throws -> any QWModelBackend {
        // Which module serves this family is the module registry's
        // answer, and it is the only place in the tree that knows.
        guard let moduleType = QWModuleRegistry.moduleType(for: descriptor.family) else {
            throw QWEngineError.unsupportedModel(descriptor.modelType)
        }
        guard let builder = builder(for: moduleType) else {
            throw QWEngineError.unsupportedModel(descriptor.modelType)
        }
        return try builder(config, descriptor, logger)
    }
}

// MARK: - Pack facts the banner reports

/// Facts about a loaded pack that the banner reports. Read from the
/// pack's own metadata at load time — the banner states no model fact
/// it did not observe.
enum QWBackendFacts {
    /// How the pack actually stores the LM head, as its entry says.
    /// The raw planes are BF16, so that is the label they get; a head
    /// that is not in the pack is reported as not found, not assumed.
    static func headStorageLabel(packPath: String) -> String {
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        if root.kind == .safetensorsSingle || root.kind == .safetensorsSharded {
            return "HQ4_G32_M5 (online GPU)"
        }
        guard let pack = try? QWQuant.Pack(url: URL(fileURLWithPath: packPath)) else {
            return "pack metadata unreadable"
        }
        guard let head = try? pack.entry("lm_head.weight") else {
            return "not found in the pack"
        }
        if head.rawOff != nil { return "BF16 (verbatim)" }
        if head.hq4Off != nil { return "HQ4_G32_M5" }
        if head.hq8Off != nil { return "HQ8_G32_M5" }
        return "unknown layout"
    }

    /// The head's container, as the LOADED ENGINE reports it. The engine
    /// decided the container and the loader filled it, so the engine's own
    /// word is the only one that can be true.
    static func headStorageLabel(engine: any QWInferenceEngine, packPath: String) -> String {
        guard let format = engine.report.weightFormat else {
            return headStorageLabel(packPath: packPath)
        }
        return format == "HQ4_G32" ? "HQ4_G32_M5 (online GPU)" : "\(format) (native, GPU)"
    }

    /// Whether this pack carries vision weights. This says what the
    /// pack holds; it is not a claim that a vision tower was loaded.
    static func visionStatus(packPath: String) -> String {
        if QWVisionOnDemandBackend.hasVisionWeights(modelPath: packPath) {
            return "weights present in the pack"
        }
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        if root.kind == .qwp1, (try? QWQuant.Pack(url: URL(fileURLWithPath: packPath))) == nil {
            return "pack metadata unreadable"
        }
        return "not present in this pack"
    }
}

// MARK: - Unified Model Backend Adapter

/// Unified backend adapter over `any QWInferenceEngine` for all model families.
public struct QWUnifiedModelBackend: QWModelBackend {
    public let engine: any QWInferenceEngine
    public let descriptor: QWModelDescriptor
    public let parametersInfo: QWEngineParametersInfo
    public let visionBackend: (any QWVisionBackend)?
    public let hardwareCapabilities: QWDeviceCaps?
    private let resolvedSpeculation: QWSpeculativeParameters?

    public var family: QWModelDescriptor.Family { descriptor.family }
    /// The cache the engine holds, as the engine reports it.
    public var prefixCache: QWPrefixCacheState { engine.prefixCache }
    /// What the engine can do, as the engine reports it: the capabilities
    /// come from the module behind it, not from the family it serves.
    public var capabilities: Set<QWKnobCapability> { engine.capabilities }
    public var supportsTokenConstraints: Bool { engine.supportsTokenConstraints }

    public init(
        engine: any QWInferenceEngine,
        descriptor: QWModelDescriptor,
        parametersInfo: QWEngineParametersInfo,
        visionBackend: (any QWVisionBackend)? = nil,
        hardwareCapabilities: QWDeviceCaps? = nil,
        resolvedSpeculation: QWSpeculativeParameters? = nil
    ) {
        self.engine = engine
        self.descriptor = descriptor
        self.parametersInfo = parametersInfo
        self.visionBackend = visionBackend
        self.hardwareCapabilities = hardwareCapabilities
        self.resolvedSpeculation = resolvedSpeculation
    }

    public static func load(
        moduleType: any QWInferenceModule.Type,
        config: QWEngineLifecycleActor.Config,
        descriptor: QWModelDescriptor,
        logger: QWServerLogger
    ) throws -> QWUnifiedModelBackend {
        let speculation = try moduleType.resolveSpeculation(
            config.speculativeParameters, descriptor: descriptor,
            environment: ProcessInfo.processInfo.environment, modelSwitch: false)

        // The unified engine builds and configures the model for its descriptor.
        let engine = try QWEngine(
            options: try config.makeEngineBuildOptions(speculation: speculation),
            descriptor: descriptor,
            moduleType: moduleType
        )

        QWCacheReporting.reportCacheConfiguration(engine: engine, config: config, logger: logger)
        let visionBackend = (engine.capabilities.contains(.vision) && descriptor.visDepth > 0)
            ? try Self.loadVisionIfPresent(packPath: config.packPath, loadVision: config.loadVision, logger: logger)
            : nil
        let halCaps = QWDeviceCaps.current
        let reportedDrafts = engine.capabilities.contains(.speculative) ? speculation.reportedDrafts : 0
        let backendName = speculation.reportedBackend
        let headLabel = QWBackendFacts.headStorageLabel(engine: engine, packPath: config.packPath)

        let params = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: descriptor,
            headStorage: headLabel,
            visionStatus: (visionBackend?.isLoaded == true)
                ? "Active (vision tower loaded)"
                : QWBackendFacts.visionStatus(packPath: config.packPath),
            contextBudget: engine.report.contextCapacity ?? config.maxContext,
            kvMode: config.kvMode,
            mtpDrafts: reportedDrafts,
            mtpBackend: backendName,
            gdnPrefillMode: config.gdnPrefill,
            prefillScratchEviction: config.evictPrefillScratch,
            prefillChunkRows: config.prefillChunkSize,
            weightContainerNote: engine.report.weightFormatDeclineReason
        )

        return QWUnifiedModelBackend(
            engine: engine,
            descriptor: descriptor,
            parametersInfo: params,
            visionBackend: visionBackend,
            hardwareCapabilities: halCaps,
            resolvedSpeculation: speculation.parameters
        )
    }

    private static func loadVisionIfPresent(
        packPath: String,
        loadVision: Bool,
        logger: QWServerLogger
    ) throws -> (any QWVisionBackend)? {
        guard QWVisionOnDemandBackend.hasVisionWeights(modelPath: packPath) else {
            return nil
        }
        let clock = ContinuousClock()
        let vt0 = clock.now
        let eager = loadVision || ProcessInfo.processInfo.environment["QW_VISION_EAGER"] == "1"
        do {
            let vb = try QWVision.makeOnDemand(
                mode: .ref,
                modelPath: packPath,
                keepResident: eager,
                allowOnDemand: false
            )
            let d = vt0.duration(to: clock.now)
            let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
            let modeNote = eager ? "resident fp32 planes" : "not loaded (0 B RAM; enable Load Vision Tower in Model settings)"
            logger.log(.info, String(format: "vision_backend_make_ms=%.1f (%@, mode=ref)", ms, modeNote))
            return vb
        } catch {
            logger.log(.error, "vision_backend_make_failed: \(error) (probe key present — refusing silent no-vision boot)")
            throw error
        }
    }

    public func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        let cacheActive = engine.hasPrefixCache
        let samplerConfig = QWSamplerConfig(request.sampling, vocabulary: descriptor.vocab)
        // The server says what it wants; the engine decides whether it
        // can serve it and which path it takes.
        var speculation = resolvedSpeculation ?? config.speculativeParameters
        if request.exactSpeculation {
            speculation.mtpRelaxedMargin = 0
        }
        let engineRequest = QWGenerationRequest(
            prompt: request.promptTokens,
            maxNew: request.maxNewTokens,
            stopIDs: request.stopTokenIds,
            sampling: samplerConfig,
            usePrefixCache: cacheActive,
            speculative: speculation,
            retainPrefixLength: request.retainPrefixLength,
            collectTimings: true,
            collectLogprobs: request.collectLogprobs,
            topLogprobsCount: request.topLogprobsCount,
            onTokenLogprob: request.onTokenLogprob,
            constraint: request.constraint,
            visualSpans: request.visualSpans
        )
        let result = try engine.generate(request: engineRequest, onToken: onToken)
        let cache = engine.prefixCache
        let stageGPU: [String: Double] = result.prefillGPUSeconds.map { ["prefill": $0 * 1000.0] } ?? [:]
        return QWModelCompletionMetrics(
            tokens: result.tokens,
            prefillSec: result.prefillSeconds,
            decodeSec: result.decodeSeconds,
            firstTokenSec: result.firstTokenSeconds,
            activeContextCapacity: engine.report.contextCapacity ?? config.maxContext,
            cacheHitTokens: result.cache?.matchedTokens ?? 0,
            cacheTier: result.cache?.tier,
            cacheActive: cacheActive || (result.cache?.hit == true),
            warmCacheBytes: cache.warmBytes,
            ssdCacheBytes: cache.ssdBytes,
            speculativeSummary: result.speculativeSummary,
            stageGPU: stageGPU,
            tokenLogprobs: result.tokenLogprobs
        )
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        try engine.mountStandalonePrefixCache(from: fileURL)
    }

    public func unmountSharedBasePrefixes() {
        engine.unmountSharedBasePrefixes()
    }
}

/// The cache configuration the server reports, whichever family serves it.
private enum QWCacheReporting {
    static func reportCacheConfiguration(
        engine: any QWInferenceEngine,
        config: QWEngineLifecycleActor.Config,
        logger: QWServerLogger
    ) {
        if config.cacheConfig.warmByteLimit > 0 {
            // The module configured the cache when it built the engine;
            // the server reports what that configuration was.
            logger.log(.info, String(
                format: "Prefix cache initialized (kvMode=%@): warmLimit=%.2f GB, ssdLimit=%.2f GB, entryLimit=%d, maxConcurrentSessions=%d",
                String(describing: config.kvMode),
                Double(config.cacheConfig.warmByteLimit) / 1_073_741_824.0,
                Double(config.cacheConfig.ssdByteLimit) / 1_073_741_824.0,
                config.cacheConfig.entryLimit,
                config.maxConcurrentSessions))
        } else {
            logger.log(.info, "Prefix cache is disabled (warmByteLimit is 0).")
        }

        engine.onCacheEvent = { [logger] event in
            switch event {
            case .hit(let tier, let matched, let total):
                let pct = total > 0 ? (Double(matched) / Double(total)) * 100.0 : 0.0
                logger.log(.info, String(
                    format: "[PrefixCache] HIT (%@): %d/%d tokens matched (%.1f%%) | Suffix to prefill: %d tokens",
                    tier.isEmpty ? "unknown tier" : tier, matched, total, pct, total - matched))
            case .miss(let total):
                logger.log(.info, String(
                    format: "[PrefixCache] MISS: 0/%d tokens matched | Prefilling full prompt (%d tokens)",
                    total, total))
            case .retained(let tokens, let chargedBytes, let totalWarmBytes, let warmLimitBytes):
                logger.log(.info, String(
                    format: "[PrefixCache] RETAINED: %d tokens in warm RAM (entry: %.1f MB | store: %.1f / %.0f MB)",
                    tokens, Double(chargedBytes) / 1_048_576.0,
                    Double(totalWarmBytes) / 1_048_576.0, Double(warmLimitBytes) / 1_048_576.0))
            case .skippedOversize(let tokens, let charge, let warmLimitBytes):
                logger.log(.warn, String(
                    format: "[PrefixCache] RETENTION SKIPPED: %d tokens (%.1f MB) exceeds warm RAM limit (%.0f MB)",
                    tokens, Double(charge) / 1_048_576.0, Double(warmLimitBytes) / 1_048_576.0))
            case .evicted(let tokens, let freedBytes):
                logger.log(.info, String(
                    format: "[PrefixCache] EVICTED: %d tokens to make room for newer prefixes (freed %.1f MB)",
                    tokens, Double(freedBytes) / 1_048_576.0))
            case .ioFailure(let operation, let message):
                logger.log(.warn, "[PrefixCache] SSD \(operation.rawValue) failed: \(message)")
            }
        }
    }
}
