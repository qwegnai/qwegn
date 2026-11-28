// QWEngineLifecycleActor+Load.swift — Lifecycle Config and model load/release helpers.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

extension QWEngineLifecycleActor {
    public struct Config: Sendable {
        public let packPath: String, modelName: String
        public let maxContext: Int, kvMode: QWKVMode
        public let speculativeDecodingEnabled: Bool, mtpDrafts: Int, mtpBackend: String?
        public let mtpRelaxedMargin: Double, mtpDraftStrategy: String, mtpQuadraticSeeding: Bool
        public let mtpRelaxedMinProb: Double
        public let mtpAdaptiveDepth: Bool, mtpAdaptiveEntropyThreshold: Double
        public let dflashDraftPackPath: String?, dflashSelectorTopK: Int
        public let dsparkConfidenceSchedule: Bool?
        public let ropeScalingType: String, ropeScalingFactor: Double
        public let gdnPrefill: String, evictPrefillScratch: String, prefillChunkSize: Int
        public let maxConcurrentSessions: Int
        public let lsaConfig: QWLSAConfig
        public let cacheConfig: QWCacheConfig
        public let host: String, port: Int
        public let temperature: Double, topP: Double, topK: Int
        public let maxTokens: Int, modelMaxLength: Int?
        public let repetitionPenalty: Double, seed: Int
        public let enableThinking: Bool, reasoningEffort: String, systemPrompt: String
        public var loadVision: Bool
        public var allowLocalImagePaths: Bool = false
        public var allowRemoteImageURLs: Bool = false
        public var pinModel: Bool = false
        public var decisionModelPackPath: String = ""
        public var decisionComputeUnit: String = "auto"
        public var decisionLoadVision: Bool = true
        public var decisionDefaultScoreReadout: String = "mean"
        public var decisionDefaultTruncation: String = "error"
        public var decisionAlwaysEmitBox: Bool = false
        /// Where each setting came from, by setting key. The provenance
        /// travels with the values, so a reader is told without resolving
        /// the configuration a second time.
        public let sourceSummary: [String: String]
        /// Memory policy, residency and expert-streaming toggles for the engine.
        public var engineTuning: QWEngineTuning = .default

        public init(
            packPath: String, modelName: String? = nil,
            maxContext: Int = 4096, kvMode: QWKVMode = .f16,
            speculativeDecodingEnabled: Bool = true, mtpDrafts: Int = 3, mtpBackend: String? = "variational",
            mtpRelaxedMargin: Double = 1.5, mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
            mtpDraftStrategy: String = "hybrid", mtpQuadraticSeeding: Bool = true,
            mtpAdaptiveDepth: Bool = false, mtpAdaptiveEntropyThreshold: Double = 0.5,
            dflashDraftPackPath: String? = nil, dflashSelectorTopK: Int = 16,
            dsparkConfidenceSchedule: Bool? = nil,
            ropeScalingType: String = "none", ropeScalingFactor: Double = 1.0,
            gdnPrefill: String = "regres", evictPrefillScratch: String = "auto",
            prefillChunkSize: Int = QWPrefillLimits.chunkRows, maxConcurrentSessions: Int? = nil,
            lsaConfig: QWLSAConfig = QWLSAConfig(), cacheConfig: QWCacheConfig = QWCacheConfig(),
            host: String = "127.0.0.1", port: Int = 8080,
            temperature: Double = 0.7, topP: Double = 0.9, topK: Int = 40,
            maxTokens: Int = 512, modelMaxLength: Int? = nil,
            repetitionPenalty: Double = 1.05, seed: Int = -1,
            enableThinking: Bool = true, reasoningEffort: String = "xhigh", systemPrompt: String = "",
            loadVision: Bool = false,
            allowLocalImagePaths: Bool = false,
            allowRemoteImageURLs: Bool = false,
            pinModel: Bool = false,
            decisionModelPackPath: String = "",
            decisionComputeUnit: String = "auto",
            decisionLoadVision: Bool = true,
            decisionDefaultScoreReadout: String = "mean",
            decisionDefaultTruncation: String = "error",
            decisionAlwaysEmitBox: Bool = false,
            sourceSummary: [String: String] = [:]
        ) {
            self.packPath = packPath
            self.modelName = modelName ?? Self.derivePackName(from: packPath)
            self.maxContext = maxContext
            self.kvMode = kvMode
            self.speculativeDecodingEnabled = speculativeDecodingEnabled
            self.mtpDrafts = mtpDrafts
            self.mtpBackend = mtpBackend
            self.mtpRelaxedMargin = mtpRelaxedMargin
            self.mtpRelaxedMinProb = mtpRelaxedMinProb
            self.mtpDraftStrategy = mtpDraftStrategy
            self.mtpQuadraticSeeding = mtpQuadraticSeeding
            self.mtpAdaptiveDepth = mtpAdaptiveDepth
            self.mtpAdaptiveEntropyThreshold = mtpAdaptiveEntropyThreshold
            self.dflashDraftPackPath = dflashDraftPackPath
            self.dflashSelectorTopK = dflashSelectorTopK
            self.dsparkConfidenceSchedule = dsparkConfidenceSchedule
            self.ropeScalingType = ropeScalingType
            self.ropeScalingFactor = ropeScalingFactor
            self.gdnPrefill = gdnPrefill
            self.evictPrefillScratch = evictPrefillScratch
            self.prefillChunkSize = prefillChunkSize
            let resolvedSessions = max(1, min(16, maxConcurrentSessions ?? cacheConfig.maxConcurrentSessions))
            self.maxConcurrentSessions = resolvedSessions
            var syncedCacheConfig = cacheConfig
            syncedCacheConfig.maxConcurrentSessions = resolvedSessions
            self.lsaConfig = lsaConfig
            self.cacheConfig = syncedCacheConfig
            self.host = host
            self.port = port
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.maxTokens = maxTokens
            self.modelMaxLength = modelMaxLength
            self.repetitionPenalty = repetitionPenalty
            self.seed = seed
            self.enableThinking = enableThinking
            self.reasoningEffort = reasoningEffort
            self.systemPrompt = systemPrompt
            self.loadVision = loadVision
            self.allowLocalImagePaths = allowLocalImagePaths
            self.allowRemoteImageURLs = allowRemoteImageURLs
            self.pinModel = pinModel
            self.decisionModelPackPath = decisionModelPackPath
            self.decisionComputeUnit = decisionComputeUnit
            self.decisionLoadVision = decisionLoadVision
            self.decisionDefaultScoreReadout = decisionDefaultScoreReadout
            self.decisionDefaultTruncation = decisionDefaultTruncation
            self.decisionAlwaysEmitBox = decisionAlwaysEmitBox
            self.sourceSummary = sourceSummary
        }

        /// The pack's own name — what is really there, whatever the user
        /// chose to call it.
        public var packName: String {
            Self.derivePackName(from: packPath)
        }

        public static func derivePackName(from packPath: String) -> String {
            let url = URL(fileURLWithPath: packPath)
            if url.pathExtension.lowercased() == "qwp" {
                return url.deletingPathExtension().lastPathComponent
            }
            return url.lastPathComponent
        }

        public var samplingParameters: QWSamplingParameters {
            QWSamplingParameters(
                temperature: temperature,
                topP: topP,
                topK: topK,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                seed: seed
            )
        }

        public var speculativeParameters: QWSpeculativeParameters {
            QWSpeculativeParameters(
                enabled: speculativeDecodingEnabled,
                mtpDrafts: mtpDrafts,
                mtpBackend: mtpBackend,
                mtpRelaxedMargin: mtpRelaxedMargin,
                mtpRelaxedMinProb: mtpRelaxedMinProb,
                mtpDraftStrategy: mtpDraftStrategy,
                mtpQuadraticSeeding: mtpQuadraticSeeding,
                adaptiveDepth: mtpAdaptiveDepth,
                entropyThreshold: mtpAdaptiveEntropyThreshold,
                dsparkConfidenceSchedule: dsparkConfidenceSchedule
            )
        }
    }


    /// Releases the weights while the phase says the engine is unloading,
    /// then reports the memory that came back.
    func releaseResidentModel() async -> Double? {
        let before = Self.deviceAllocatedBytes()
        // The model leaves the state here, and the phase says so.
        transition(to: .unloading)
        let after = Self.deviceAllocatedBytes()
        // The weights are already gone: the release happens with the
        // phase change above. The phase stays `.unloading` for a moment
        // on purpose, so a reader can watch the unload happen instead of
        // seeing a jump from `loaded` to `unloaded`.
        try? await Task.sleep(nanoseconds: Self.unloadPhaseWindow)
        _ = await residencyPool.releaseLease(for: .primaryLLM)
        transition(to: .unloaded)

        let telemetry = QWEngineTelemetryRegistry.shared
        telemetry.resetToIdle(resetCacheCounters: true)
        telemetry.updateArena(reserved: 0)
        statusBox.resetLoggedSamplingMessage(nil)
        // The engine is gone, so no turn is running: absent, not empty.
        telemetry.updateContext(active: nil, capacity: config.maxContext)
        // The engine is gone, so nothing is measuring the cache: the
        // reading is absent, not an empty cache.
        telemetry.updateCacheUsage(warmBytes: nil, warmLimit: config.cacheConfig.warmByteLimit,
                                   ssdBytes: nil, ssdLimit: config.cacheConfig.ssdByteLimit)
        logger.log(.info, "Model '\(config.modelName)' successfully unloaded from memory.")
        return Self.reclaimedMB(before: before, after: after)
    }

    /// The device memory the engine holds, as Metal accounts it. Zero
    /// means the reading failed, not that the engine holds nothing.
    static func deviceAllocatedBytes() -> UInt64 {
        QWDeviceCaps.systemDeviceAllocatedBytes()
    }

    /// What the release gave back, in megabytes: the Metal device's own
    /// allocation delta across the release, not a weight estimate.
    static func reclaimedMB(before: UInt64, after: UInt64) -> Double? {
        guard before > 0, after > 0 else { return nil }
        let delta = before > after ? before - after : 0
        return Double(delta) / (1024.0 * 1024.0)
    }

    public func restartExplicit() async throws -> LoadedState {
        _ = try await unloadExplicit()
        return try await loadExplicit()
    }

    nonisolated public func getLoadedParamsInfo() -> QWEngineParametersInfo? {
        statusBox.getLoadedParamsInfo()
    }

    /// Resolves `config.dflashDraftPackPath` or selects a geometry-matched drafter pack for `targetDescriptor`.
    static func resolveEffectiveDrafterPackPath(
        for config: Config,
        targetDescriptor: QWModelDescriptor,
        drafters: [DiscoveredModel] = QWServerSettings.scanDrafterModelsDirectory()
    ) -> String? {
        if let resolved = QWServerSettings.effectiveDrafterPackPath(
            configuredPath: config.dflashDraftPackPath,
            for: targetDescriptor,
            preferredBackend: config.mtpBackend,
            drafters: drafters,
            targetModelName: config.modelName
        ) {
            return resolved
        }
        return QWServerSettings.normalizeExternalDrafterBackend(config.mtpBackend) == "dspine"
            ? QWDFlashConfig.defaultDSpinePackPath
            : nil
    }

    static func performLoad(
        config: Config,
        logger: QWServerLogger,
        backendRegistry: QWModelBackendRegistry
    ) throws -> LoadedState {
        logger.log(.info, "Loading model pack from '\(config.packPath)'...")
        let t0 = Date()

        _ = try QWModelLocation(url: URL(fileURLWithPath: config.packPath))
        let runtimeAssets = try QWModelRuntimeAssets.load(packPath: config.packPath)
        if config.speculativeDecodingEnabled,
           config.mtpDrafts > 0,
           ["dflash2", "dflash", "dspine"].contains(config.mtpBackend?.lowercased() ?? "") {
            let effDraftPack = resolveEffectiveDrafterPackPath(
                for: config,
                targetDescriptor: runtimeAssets.descriptor
            )
            try QWDFlashConfig.validateDrafter(
                targetDescriptor: runtimeAssets.descriptor,
                packPath: effDraftPack,
                drafts: config.mtpDrafts,
                selectorTopK: config.dflashSelectorTopK
            )
        }
        let tokenizer = try runtimeAssets.makeTokenizer()
        let telemetry = QWEngineTelemetryRegistry.shared
        let hardware = QWMachineHardwareInfo.current()

        let modelBackend = try backendRegistry.loadBackend(
            config: config,
            descriptor: runtimeAssets.descriptor,
            logger: logger
        )

        // A backend that cannot report its cache says nothing: a zero
        // here would be a cache that was never measured.
        let cache = modelBackend.prefixCache
        let warmBytes = cache.warmBytes
        let ssdBytes = cache.ssdBytes
        // No turn has run yet: the context is absent, not empty.
        telemetry.updateContext(active: nil, capacity: modelBackend.parametersInfo.contextBudget)
        telemetry.updateArena(reserved: Int(Self.deviceAllocatedBytes()))
        telemetry.updateCacheUsage(
            warmBytes: warmBytes,
            warmLimit: config.cacheConfig.warmByteLimit,
            ssdBytes: ssdBytes,
            ssdLimit: config.cacheConfig.ssdByteLimit
        )

        let loadSec = Date().timeIntervalSince(t0)
        let banner = QWHardwareBanner.renderBanner(
            hardware: hardware,
            caps: modelBackend.hardwareCapabilities,
            params: modelBackend.parametersInfo,
            host: config.host,
            port: config.port
        )
        logger.logStartupBanner(banner)
        logger.log(.info, Self.denseNaxSelectionLogMessage(
            temperature: config.temperature, topP: config.topP, topK: config.topK,
            paramsSource: Self.samplingParamsSourceLabel(config.sourceSummary)))
        logger.log(.info, String(format: "Model (%@) loaded in %.2f seconds.",
                                 modelBackend.descriptor.architectureDisplayName, loadSec))

        return LoadedState(
            modelBackend: modelBackend,
            tokenizer: tokenizer,
            chatTemplate: runtimeAssets.chatTemplate
        )
    }

    /// What the engine says when a load fails. The reason is the user's
    /// best clue, so it is the reason that is reported.
    static func detail(for error: any Error) -> String {
        if let localized = error as? any LocalizedError {
            if let message = localized.errorDescription { return message }
            if let reason = localized.failureReason { return reason }
        }
        return String(describing: error)
    }

    // handleChatCompletion and incremental wire encoding live in QWEngineLifecycleActor+Chat.swift.
}
