import Foundation
import QWCore
import QWEngine

public typealias DiscoveredModel = QWServerSettings.DiscoveredModel

/// Persistent server settings model for QWEGN.
public struct QWServerSettings: Codable, Sendable, Equatable {
    // MARK: - System & App
    public var runAtStartup: Bool
    public var autoStartServer: Bool
    public var keepServerRunningAfterQuit: Bool

    // MARK: - Network & Server
    public var host: String
    public var port: Int
    public var telemetryPollIntervalMs: Int

    // MARK: - Model & Architecture
    public var modelPackPath: String
    public var modelName: String
    public var modelAlias: String // User assigned alias for the active model (e.g. "qwen-27b", "chat")
    public var modelAliases: [String: String] // Per-model aliases persisted by modelName
    public var maxContext: Int
    public var kvMode: String // "f16", "q8", "q4"
    public var ropeScalingType: String // "none", "yarn"
    public var ropeScalingFactor: Double

    // MARK: - Sampling & Generation Defaults
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var maxTokens: Int
    public var repetitionPenalty: Double
    public var seed: Int // -1 = random

    // MARK: - Reasoning & Thinking
    public var enableThinking: Bool
    public var reasoningEffort: String // "low", "medium", "xhigh"
    public var systemPrompt: String
    public var lazyLoad: Bool
    public var loadVision: Bool

    // MARK: - Compute & Speculation
    public var speculativeDecodingEnabled: Bool
    public var mtpDrafts: Int // 0...7
    public var mtpBackend: String
    public var mtpRelaxedMargin: Double
    /// Variational acceptance probability floor (`0` disables it).
    public var mtpRelaxedMinProb: Double
    public var mtpDraftStrategy: String
    public var mtpQuadraticSeeding: Bool
    public var mtpAdaptiveDepth: Bool
    public var mtpEntropyThreshold: Double
    public var dflashDraftPackPath: String
    public var dflashSelectorTopK: Int
    public var gdnPrefill: String
    public var evictPrefillScratch: String
    public var prefillChunkSize: Int
    public var maxConcurrentSessions: Int
    public var lsaMode: String
    public var lsaBudget: Int
    public var lsaClusterBlock: Int
    public var lsaSidecarPath: String
    public var engineTuning: [String: QWKnobValue] = [:]  // QWEngineTuning knobs; absent = default

    // MARK: - Two-Tier KV Cache
    public var warmRAMLimitGB: Double
    public var ssdCacheEnabled: Bool
    public var ssdCacheLimitGB: Double
    public var cacheDir: String

    // MARK: - Logging & Diagnostics
    public var logDir: String
    public var logFileName: String
    public var logLevel: String // "debug", "info", "warn", "error"

    public struct DiscoveredModel: Identifiable, Hashable, Codable, Sendable {
        public var id: String { name }
        public let name: String
        public let packPath: String
        public let family: String?
        public let nativeWindow: Int?
        public let supported: Bool
        public let reason: String?
        public let capabilities: [String]
        public let maxSpeculativeDrafts: Int
        public let hasLSASidecar: Bool
        public let lsaSidecarPath: String?

        public init(name: String, packPath: String, family: String? = nil, nativeWindow: Int? = nil,
                    supported: Bool = true, reason: String? = nil,
                    capabilities: [String] = [], maxSpeculativeDrafts: Int = 5,
                    hasLSASidecar: Bool = false, lsaSidecarPath: String? = nil) {
            self.name = name
            self.packPath = packPath
            self.family = family
            self.nativeWindow = nativeWindow
            self.supported = supported
            self.reason = reason
            self.capabilities = capabilities
            self.maxSpeculativeDrafts = maxSpeculativeDrafts
            self.hasLSASidecar = hasLSASidecar
            self.lsaSidecarPath = lsaSidecarPath
        }
    }

    public init(
        runAtStartup: Bool = false,
        autoStartServer: Bool = true,
        keepServerRunningAfterQuit: Bool = false,
        host: String = "127.0.0.1",
        port: Int = 8080,
        telemetryPollIntervalMs: Int = 500,
        modelPackPath: String = "",
        modelName: String = "Qwen3.8-27B",
        modelAlias: String = "",
        modelAliases: [String: String] = [:],
        maxContext: Int = 4096,
        kvMode: String = "q4",
        ropeScalingType: String = "none",
        ropeScalingFactor: Double = 1.0,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        topK: Int = 40,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.05,
        seed: Int = -1,
        enableThinking: Bool = true,
        reasoningEffort: String = "xhigh",
        systemPrompt: String = "",
        lazyLoad: Bool = true,
        loadVision: Bool = false,
        speculativeDecodingEnabled: Bool = true,
        mtpDrafts: Int = 3,
        mtpBackend: String = "variational",
        mtpRelaxedMargin: Double = 1.5,
        mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid",
        mtpQuadraticSeeding: Bool = true,
        mtpAdaptiveDepth: Bool = false,
        mtpEntropyThreshold: Double = 0.7,
        dflashDraftPackPath: String = "",
        dflashSelectorTopK: Int = 16,
        gdnPrefill: String = "chunked",
        evictPrefillScratch: String = "auto",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        maxConcurrentSessions: Int = QWCacheConfig.defaultMaxConcurrentSessions,
        lsaMode: String = "auto",
        lsaBudget: Int = QWLSAConfig.defaultBudget,
        lsaClusterBlock: Int = QWLSAConfig.defaultClusterBlock,
        lsaSidecarPath: String = "",
        warmRAMLimitGB: Double = 4.0,
        ssdCacheEnabled: Bool = false,
        ssdCacheLimitGB: Double = 10.0,
        cacheDir: String = "~/.qwg/cache",
        logDir: String = "~/.qwg/logs",
        logFileName: String = "qwg.log",
        logLevel: String = "info"
    ) {
        self.runAtStartup = runAtStartup
        self.autoStartServer = autoStartServer
        self.keepServerRunningAfterQuit = keepServerRunningAfterQuit
        self.host = host
        self.port = port
        self.telemetryPollIntervalMs = telemetryPollIntervalMs
        self.modelPackPath = modelPackPath
        self.modelName = modelName
        self.modelAlias = modelAlias
        var mergedAliases = modelAliases
        let trimmedAlias = modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty && !trimmedAlias.isEmpty {
            mergedAliases[modelName] = trimmedAlias
        }
        self.modelAliases = mergedAliases
        self.maxContext = maxContext
        self.kvMode = kvMode
        self.ropeScalingType = ropeScalingType
        self.ropeScalingFactor = ropeScalingFactor
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.systemPrompt = systemPrompt
        self.lazyLoad = lazyLoad
        self.loadVision = loadVision
        self.speculativeDecodingEnabled = speculativeDecodingEnabled
        self.mtpDrafts = mtpDrafts
        self.mtpBackend = mtpBackend
        self.mtpRelaxedMargin = mtpRelaxedMargin
        self.mtpRelaxedMinProb = mtpRelaxedMinProb
        self.mtpDraftStrategy = mtpDraftStrategy
        self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.mtpAdaptiveDepth = mtpAdaptiveDepth
        self.mtpEntropyThreshold = mtpEntropyThreshold
        self.dflashDraftPackPath = dflashDraftPackPath
        self.dflashSelectorTopK = dflashSelectorTopK
        self.gdnPrefill = gdnPrefill
        self.evictPrefillScratch = evictPrefillScratch
        self.prefillChunkSize = prefillChunkSize
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions))
        self.lsaMode = lsaMode
        self.lsaBudget = lsaBudget
        self.lsaClusterBlock = lsaClusterBlock
        self.lsaSidecarPath = lsaSidecarPath
        self.warmRAMLimitGB = warmRAMLimitGB
        self.ssdCacheEnabled = ssdCacheEnabled
        self.ssdCacheLimitGB = ssdCacheLimitGB
        self.cacheDir = cacheDir
        self.logDir = logDir
        self.logFileName = logFileName
        self.logLevel = logLevel
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.runAtStartup = try c.decodeIfPresent(Bool.self, forKey: .runAtStartup) ?? false
        self.autoStartServer = try c.decodeIfPresent(Bool.self, forKey: .autoStartServer) ?? true
        self.keepServerRunningAfterQuit = try c.decodeIfPresent(Bool.self, forKey: .keepServerRunningAfterQuit) ?? false
        self.host = try c.decodeIfPresent(String.self, forKey: .host) ?? "127.0.0.1"
        self.port = try c.decodeIfPresent(Int.self, forKey: .port) ?? 8080
        self.telemetryPollIntervalMs = try c.decodeIfPresent(Int.self, forKey: .telemetryPollIntervalMs) ?? 500
        self.modelPackPath = try c.decodeIfPresent(String.self, forKey: .modelPackPath) ?? ""
        self.modelName = try c.decodeIfPresent(String.self, forKey: .modelName) ?? "Qwen3.8-27B"
        let decodedAlias = try c.decodeIfPresent(String.self, forKey: .modelAlias) ?? ""
        var decodedAliases = try c.decodeIfPresent([String: String].self, forKey: .modelAliases) ?? [:]
        let trimmedDecoded = decodedAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !self.modelName.isEmpty && !trimmedDecoded.isEmpty && decodedAliases[self.modelName] == nil {
            decodedAliases[self.modelName] = trimmedDecoded
        }
        self.modelAliases = decodedAliases
        self.modelAlias = decodedAliases[self.modelName] ?? trimmedDecoded
        self.maxContext = try c.decodeIfPresent(Int.self, forKey: .maxContext) ?? 4096
        self.kvMode = try c.decodeIfPresent(String.self, forKey: .kvMode) ?? "q4"
        func decodeDouble(forKey key: CodingKeys, defaultVal: Double) throws -> Double {
            try c.decodeIfPresent(Double.self, forKey: key) ?? defaultVal
        }
        self.ropeScalingType = try c.decodeIfPresent(String.self, forKey: .ropeScalingType) ?? "none"
        self.ropeScalingFactor = try decodeDouble(forKey: .ropeScalingFactor, defaultVal: 1.0)
        self.temperature = try decodeDouble(forKey: .temperature, defaultVal: 0.7)
        self.topP = try decodeDouble(forKey: .topP, defaultVal: 0.9)
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK) ?? 40
        self.maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 512
        self.repetitionPenalty = try decodeDouble(forKey: .repetitionPenalty, defaultVal: 1.05)
        self.seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? -1
        self.enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? true
        self.reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort) ?? "xhigh"
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        self.lazyLoad = try c.decodeIfPresent(Bool.self, forKey: .lazyLoad) ?? true
        self.loadVision = try c.decodeIfPresent(Bool.self, forKey: .loadVision) ?? false

        self.speculativeDecodingEnabled = try c.decodeIfPresent(Bool.self, forKey: .speculativeDecodingEnabled) ?? true
        self.mtpDrafts = try c.decodeIfPresent(Int.self, forKey: .mtpDrafts) ?? 3
        let rawBackend = try c.decodeIfPresent(String.self, forKey: .mtpBackend) ?? "variational"
        self.mtpBackend = (rawBackend == "speculative" || rawBackend == "lightening") ? "standard" : rawBackend
        self.mtpRelaxedMargin = try decodeDouble(forKey: .mtpRelaxedMargin, defaultVal: 1.5)
        self.mtpRelaxedMinProb = try decodeDouble(forKey: .mtpRelaxedMinProb, defaultVal: QWSpeculativeParameters.defaultRelaxedMinProb)
        self.mtpDraftStrategy = try c.decodeIfPresent(String.self, forKey: .mtpDraftStrategy) ?? "hybrid"
        self.mtpQuadraticSeeding = try c.decodeIfPresent(Bool.self, forKey: .mtpQuadraticSeeding) ?? true
        self.mtpAdaptiveDepth = try c.decodeIfPresent(Bool.self, forKey: .mtpAdaptiveDepth) ?? false
        self.mtpEntropyThreshold = try decodeDouble(forKey: .mtpEntropyThreshold, defaultVal: 0.7)
        self.dflashDraftPackPath = try c.decodeIfPresent(String.self, forKey: .dflashDraftPackPath) ?? ""
        self.dflashSelectorTopK = try c.decodeIfPresent(Int.self, forKey: .dflashSelectorTopK) ?? 16
        self.gdnPrefill = try c.decodeIfPresent(String.self, forKey: .gdnPrefill) ?? "chunked"
        self.evictPrefillScratch = try c.decodeIfPresent(String.self, forKey: .evictPrefillScratch) ?? "auto"
        self.prefillChunkSize = try c.decodeIfPresent(Int.self, forKey: .prefillChunkSize) ?? QWPrefillLimits.chunkRows
        let decodedSessions = try c.decodeIfPresent(Int.self, forKey: .maxConcurrentSessions) ?? QWCacheConfig.defaultMaxConcurrentSessions
        self.maxConcurrentSessions = max(1, min(16, decodedSessions))
        self.lsaMode = try c.decodeIfPresent(String.self, forKey: .lsaMode) ?? "auto"
        self.lsaBudget = try c.decodeIfPresent(Int.self, forKey: .lsaBudget) ?? QWLSAConfig.defaultBudget
        self.lsaClusterBlock = try c.decodeIfPresent(Int.self, forKey: .lsaClusterBlock) ?? QWLSAConfig.defaultClusterBlock
        self.lsaSidecarPath = try c.decodeIfPresent(String.self, forKey: .lsaSidecarPath) ?? ""
        self.engineTuning = try c.decodeIfPresent([String: QWKnobValue].self, forKey: .engineTuning) ?? [:]
        self.warmRAMLimitGB = try decodeDouble(forKey: .warmRAMLimitGB, defaultVal: 4.0)
        self.ssdCacheEnabled = try c.decodeIfPresent(Bool.self, forKey: .ssdCacheEnabled) ?? false
        self.ssdCacheLimitGB = try decodeDouble(forKey: .ssdCacheLimitGB, defaultVal: 10.0)
        self.cacheDir = try c.decodeIfPresent(String.self, forKey: .cacheDir) ?? "~/.qwg/cache"
        self.logDir = try c.decodeIfPresent(String.self, forKey: .logDir) ?? "~/.qwg/logs"
        self.logFileName = try c.decodeIfPresent(String.self, forKey: .logFileName) ?? "qwg.log"
        self.logLevel = try c.decodeIfPresent(String.self, forKey: .logLevel) ?? "info"
    }

    /// Full expanded path to the log file
    public var expandedLogFilePath: String {
        let expandedDir = logDir.expandingTildePath
        return (expandedDir as NSString).appendingPathComponent(logFileName)
    }

    /// Full expanded path to the cache directory
    public var expandedCacheDirPath: String {
        cacheDir.expandingTildePath
    }

    /// Cohesive sampling parameters view of these settings.
    public var samplingParameters: QWSamplingParameters {
        get {
            QWSamplingParameters(
                temperature: temperature,
                topP: topP,
                topK: topK,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                seed: seed
            )
        }
        set {
            temperature = newValue.temperature
            topP = newValue.topP
            topK = newValue.topK
            maxTokens = newValue.maxTokens
            repetitionPenalty = newValue.repetitionPenalty
            seed = newValue.seed
        }
    }

    /// Cohesive speculative decoding parameters view of these settings.
    public var speculativeParameters: QWSpeculativeParameters {
        get {
            QWSpeculativeParameters(
                enabled: speculativeDecodingEnabled,
                mtpDrafts: mtpDrafts,
                mtpBackend: mtpBackend,
                mtpRelaxedMargin: mtpRelaxedMargin,
                mtpRelaxedMinProb: mtpRelaxedMinProb,
                mtpDraftStrategy: mtpDraftStrategy,
                mtpQuadraticSeeding: mtpQuadraticSeeding,
                adaptiveDepth: mtpAdaptiveDepth,
                entropyThreshold: mtpEntropyThreshold,
                dflashDraftPackPath: dflashDraftPackPath.isEmpty ? nil : dflashDraftPackPath,
                dflashSelectorTopK: dflashSelectorTopK
            )
        }
        set {
            speculativeDecodingEnabled = newValue.enabled
            mtpDrafts = newValue.mtpDrafts
            mtpBackend = newValue.mtpBackend ?? "variational"
            mtpRelaxedMargin = newValue.mtpRelaxedMargin
            mtpRelaxedMinProb = newValue.mtpRelaxedMinProb
            mtpDraftStrategy = newValue.mtpDraftStrategy
            mtpQuadraticSeeding = newValue.mtpQuadraticSeeding
            mtpAdaptiveDepth = newValue.adaptiveDepth
            mtpEntropyThreshold = newValue.entropyThreshold
            dflashDraftPackPath = newValue.dflashDraftPackPath ?? ""
            dflashSelectorTopK = newValue.dflashSelectorTopK
        }
    }

    /// Cohesive Latent Sparse Attention configuration view of these settings.
    public var lsaConfig: QWLSAConfig {
        get {
            QWLSAConfig(
                mode: QWLSAMode.resolve(lsaMode) ?? .off,
                budget: lsaBudget,
                clusterBlock: lsaClusterBlock,
                sidecarPath: lsaSidecarPath
            )
        }
        set {
            lsaMode = newValue.mode.rawValue
            lsaBudget = newValue.budget
            lsaClusterBlock = newValue.clusterBlock
            lsaSidecarPath = newValue.sidecarPath
        }
    }

    /// Reinstates the engine's top-performance default settings while preserving
    /// the currently selected model, model aliases, and host/port configuration.
    public mutating func resetToTopPerformanceDefaults(for descriptor: QWModelDescriptor? = nil) {
        let resolvedDescriptor = descriptor ?? (try? validatePrimaryModel())
        let specDefaults = QWSpeculativeParameters.topPerformanceDefaults(for: resolvedDescriptor)
        speculativeParameters = specDefaults
        maxContext = resolvedDescriptor?.maxPositionEmbeddings ?? 4096
        kvMode = "q4"
        ropeScalingType = "none"
        ropeScalingFactor = 1.0
        temperature = 0.7
        topP = 0.9
        topK = 40
        maxTokens = 512
        repetitionPenalty = 1.05
        seed = -1
        enableThinking = true
        reasoningEffort = "xhigh"
        gdnPrefill = "chunked"
        evictPrefillScratch = "auto"
        lsaMode = "auto"
        lsaBudget = QWLSAConfig.defaultBudget
        lsaClusterBlock = QWLSAConfig.defaultClusterBlock
        lsaSidecarPath = ""
        engineTuning = [:]
        let isMoE = (resolvedDescriptor?.numExperts ?? 0) > 0 || resolvedDescriptor?.family == .qwen38FlashNext
        let hasLargeUnifiedMemory = Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000.0 > 36.5
        prefillChunkSize = (isMoE && hasLargeUnifiedMemory) ? 4096 : QWPrefillLimits.chunkRows
        maxConcurrentSessions = QWCacheConfig.defaultMaxConcurrentSessions
        warmRAMLimitGB = 4.0
    }

    /// Returns the persisted alias for `modelName` (or empty string if none).
    public func alias(for modelName: String) -> String {
        if let mapped = modelAliases[modelName], !mapped.isEmpty {
            return mapped
        }
        if modelName == self.modelName {
            return modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    /// Updates or removes the persisted alias for `modelName`.
    public mutating func setAlias(_ alias: String, for modelName: String) {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if trimmed.isEmpty {
                modelAliases.removeValue(forKey: modelName)
            } else {
                modelAliases[modelName] = trimmed
            }
        }
        if modelName == self.modelName {
            self.modelAlias = trimmed
        }
    }

    /// Synchronizes `modelAlias` and `modelAliases` for the currently active `modelName`.
    public mutating func syncActiveModelAlias() {
        let trimmed = modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if !trimmed.isEmpty {
                modelAliases[modelName] = trimmed
            } else if let mapped = modelAliases[modelName], !mapped.isEmpty {
                self.modelAlias = mapped
                return
            } else {
                modelAliases.removeValue(forKey: modelName)
            }
        }
        self.modelAlias = trimmed
    }

    /// Removes aliases of models that are no longer present in `~/.qwg/models`.
    /// Returns true when at least one orphaned alias was removed.
    @discardableResult
    public mutating func pruneRemovedModels(discovered: [DiscoveredModel] = Self.scanModelsDirectory()) -> Bool {
        let installed = Set(discovered.filter { !$0.packPath.isEmpty }.map(\.name))
        guard !installed.isEmpty else { return false }
        // The active model may be served under a name of the user's
        // choosing; the pack that backs it says whether it is still
        // there. An alias whose model is installed is not an orphan.
        let installedPacks = Set(discovered.filter { !$0.packPath.isEmpty }
            .map { $0.packPath.expandingTildePath })
        let activePack = modelPackPath.expandingTildePath
        let activeInstalled = installedPacks.contains(activePack)
        var changed = false
        for key in Array(modelAliases.keys)
        where !installed.contains(key) && !(key == modelName && activeInstalled) {
            modelAliases.removeValue(forKey: key)
            changed = true
        }
        if !installed.contains(modelName) {
            let modelsRoot = QWPaths.modelsDirectoryPath
            if !activeInstalled && (activePack.isEmpty || activePack.hasPrefix(modelsRoot)) {
                if !modelAlias.isEmpty {
                    modelAlias = ""
                    changed = true
                }
            }
        } else if let mapped = modelAliases[modelName], modelAlias != mapped {
            modelAlias = mapped
            changed = true
        }
        return changed
    }

    /// Resolves a model reference (CLI `--model` argument, model name in `~/.qwg/models`, alias, or pack path)
    /// to `(packPath, modelName, alias)`.
    public func resolveModelReference(
        _ ref: String,
        discovered: [DiscoveredModel] = Self.scanModelsDirectory()
    ) -> (packPath: String, modelName: String, alias: String) {
        let trimmed = ref.trimmingCharacters(in: .whitespacesAndNewlines)
        let activeAlias = alias(for: modelName)
        guard !trimmed.isEmpty else { return (modelPackPath.expandingTildePath, modelName, activeAlias) }
        // 0. Prefer the active model when `ref` matches its name or alias case-insensitively.
        if !modelPackPath.isEmpty && (trimmed.caseInsensitiveCompare(modelName) == .orderedSame
            || (!activeAlias.isEmpty && trimmed.caseInsensitiveCompare(activeAlias) == .orderedSame)) {
            return (modelPackPath.expandingTildePath, modelName, activeAlias)
        }
        // 1. Match by persisted alias in modelAliases
        if let (targetModel, targetAlias) = modelAliases.first(where: { $0.value == trimmed })
            ?? modelAliases.first(where: { $0.value.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            if let disc = discovered.first(where: { $0.name == targetModel }), !disc.packPath.isEmpty {
                return (disc.packPath, disc.name, targetAlias)
            }
            let candidate = (QWPaths.modelsDirectoryPath as NSString).appendingPathComponent(targetModel)
            if FileManager.default.fileExists(atPath: candidate) { return (candidate, targetModel, targetAlias) }
            if targetModel == modelName && !modelPackPath.isEmpty {
                return (modelPackPath.expandingTildePath, modelName, targetAlias)
            }
        }
        if !modelAlias.isEmpty && modelAlias.caseInsensitiveCompare(trimmed) == .orderedSame {
            return (modelPackPath.expandingTildePath, modelName, modelAlias)
        }
        // 2. Match by discovered model name in ~/.qwg/models
        if let disc = discovered.first(where: { $0.name == trimmed })
            ?? discovered.first(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }),
           !disc.packPath.isEmpty {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        // 3. Direct path or relative to ~/.qwg/models or HuggingFace / MLX cache
        let fm = FileManager.default
        func resolvedTuple(_ path: String) -> (String, String, String) {
            let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: path), fileManager: fm)
            let name = QWModelRoot.canonicalModelName(for: resolvedURL)
            return (resolvedURL.path, name, alias(for: name))
        }
        let inModelsDir = (QWPaths.modelsDirectoryPath as NSString).appendingPathComponent(trimmed)
        if !trimmed.contains("/") && !trimmed.hasPrefix("~") && fm.fileExists(atPath: inModelsDir) {
            return resolvedTuple(inModelsDir)
        }
        let expanded = trimmed.expandingTildePath
        if !fm.fileExists(atPath: expanded) {
            let repoLeaf = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
            for rootDir in ["./models", QWPaths.modelsDirectoryPath] {
                for sub in [trimmed, repoLeaf] {
                    let cand = (rootDir as NSString).appendingPathComponent(sub)
                    if fm.fileExists(atPath: cand) { return resolvedTuple(cand) }
                }
            }
            // Check HuggingFace hub cache (~/.cache/huggingface/hub/models--<org>--<repo>) or ~/.mlx-serve/models/<org>/<repo>
            let hfHub = QWPaths.hfHubCacheDirectoryURL.path
            let sanitized = trimmed.replacingOccurrences(of: "/", with: "--")
            let hfRepoCandidate = (hfHub as NSString).appendingPathComponent(
                sanitized.hasPrefix("models--") ? sanitized : "models--\(sanitized)"
            )
            if fm.fileExists(atPath: hfRepoCandidate) { return resolvedTuple(hfRepoCandidate) }
            let mlxRoot = ("~/.mlx-serve/models" as NSString).expandingTildeInPath
            let mlxCandidate = (mlxRoot as NSString).appendingPathComponent(trimmed)
            if fm.fileExists(atPath: mlxCandidate) { return resolvedTuple(mlxCandidate) }
        }
        let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: expanded), fileManager: fm)
        let resolvedPath = resolvedURL.path
        let canonical = QWModelRoot.canonicalModelName(for: resolvedURL)
        let derivedName = canonical.isEmpty ? modelName : canonical
        if let disc = discovered.first(where: { $0.packPath == resolvedPath || $0.packPath == expanded }) {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        return (resolvedPath, derivedName, alias(for: derivedName))
    }

    /// Scans ~/.qwg/models and the Hugging Face hub cache for installed primary model packs.
    public static func scanModelsDirectory() -> [DiscoveredModel] {
        let dir = QWPaths.modelsDirectoryPath
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir) { try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true) }
        let local = QWModelCatalog.scan(directory: dir)
        let hub = QWModelCatalog.scanHuggingFaceHub(directory: QWPaths.hfHubCacheDirectoryURL.path)
        let localNames = Set(local.map(\.name))
        return (local + hub.filter { !localNames.contains($0.name) }).map { e in
            DiscoveredModel(name: e.name, packPath: e.path, family: e.family?.rawValue,
                            nativeWindow: e.nativeWindow, supported: e.supported, reason: e.reason,
                            capabilities: e.capabilities.map(\.rawValue).sorted(),
                            maxSpeculativeDrafts: e.maxSpeculativeDrafts,
                            hasLSASidecar: e.hasLSASidecar, lsaSidecarPath: e.lsaSidecarPath)
        }
    }

    /// Scans ~/.qwg/models for installed DFlash 2 and DSpine drafter model packs via `config.json`.
    public static func scanDrafterModelsDirectory() -> [DiscoveredModel] {
        let dir = QWPaths.modelsDirectoryPath
        var models: [DiscoveredModel] = []
        if let items = try? FileManager.default.contentsOfDirectory(atPath: dir) {
            for item in items.sorted() where !item.hasPrefix(".") {
                let itemPath = (dir as NSString).appendingPathComponent(item)
                let url = URL(fileURLWithPath: itemPath)
                guard QWDFlashConfig.isDrafterPack(at: url) else { continue }
                models.append(DiscoveredModel(name: QWModelRoot.canonicalModelName(for: url), packPath: itemPath))
            }
        }
        return models
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferDrafterBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        QWDFlashConfig.inferBackendAndMaxDrafts(packPath: packPath)
    }

    /// Validates the currently selected primary model pack (fail-fast check before server launch or upon UI selection).
    public func validatePrimaryModel() throws -> QWModelDescriptor {
        let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
        let candidatePath = ref.packPath.isEmpty ? modelName : ref.packPath
        let assets = try QWModelRuntimeAssets.load(packPath: candidatePath)
        return assets.descriptor
    }

    /// Validates the DFlash 2 / DSpine drafter model configuration against the selected primary model.
    public func validateDFlashConfiguration() throws {
        let descriptor = try validatePrimaryModel()
        let maxD = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: dflashDraftPackPath).maxDrafts
        let drafts = max(1, min(maxD, mtpDrafts))
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: descriptor,
            packPath: dflashDraftPackPath,
            drafts: drafts,
            selectorTopK: dflashSelectorTopK
        )
    }

    /// Validates both the primary model and, when DFlash 2 / DSpine speculative decoding is enabled, the drafter pack.
    public func validateBeforeLaunch() throws {
        guard !modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
              !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        let descriptor = try validatePrimaryModel()
        let isExternalDrafter = ["dflash2", "dflash", "dspine"].contains(mtpBackend.lowercased())
        if speculativeDecodingEnabled && isExternalDrafter && mtpDrafts > 0 {
            try QWDFlashConfig.validateDrafter(targetDescriptor: descriptor, packPath: dflashDraftPackPath,
                                               drafts: mtpDrafts, selectorTopK: dflashSelectorTopK)
        }
        guard QWLSAMode.resolve(lsaMode) != nil else {
            throw QWEngineError.state("Unknown LSA mode '\(lsaMode)' (expected off, auto, sidecar, svd, or qsa-cluster)")
        }
        if lsaMode.lowercased() == "sidecar" {
            let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
            guard QWLSAConfig.resolveSidecarURL(packPath: ref.packPath,
                                                explicitSidecarPath: lsaSidecarPath.isEmpty ? nil : lsaSidecarPath) != nil else {
                throw QWEngineError.state("LSA mode 'sidecar' requires a .qlsa sidecar file in '\(ref.packPath)' or via lsaSidecarPath")
            }
        }
    }
}

/// Manager for loading and saving settings to ~/.qwg/settings.json
public final class QWSettingsManager: @unchecked Sendable {
    public static let shared = QWSettingsManager()
    private let lock = NSLock()
    private var cachedSettings: QWServerSettings?
    private var cachedBytes: Data?
    private var cachedURL: URL?

    public static var defaultSettingsDirectory: URL { QWPaths.rootDirectoryURL }
    public static var defaultSettingsFileURL: URL { QWPaths.settingsFileURL }
    public init() {}

    /// Returns the per-model `QWProfileStore` associated with a settings file URL.
    public func profileStore(forSettingsURL url: URL = defaultSettingsFileURL) -> QWProfileStore {
        QWProfileStore(root: url.deletingLastPathComponent().appendingPathComponent("profiles", isDirectory: true))
    }

    /// Loads the dedicated per-model knob profile for `modelName`, if one exists on disk.
    public func loadProfile(forModel modelName: String, from url: URL = defaultSettingsFileURL) -> QWKnobProfile? {
        let trimmed = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : (try? profileStore(forSettingsURL: url).load(modelId: trimmed))
    }

    /// Removes the dedicated per-model knob profile for `modelName`, if present.
    public func removeProfile(forModel modelName: String, from url: URL = defaultSettingsFileURL) {
        let trimmed = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        try? FileManager.default.removeItem(at: profileStore(forSettingsURL: url).profileURL(for: trimmed))
    }

    /// Resolves effective settings for `modelName` / `packPath`: starts from the model's config files + engine defaults,
    /// and overlays the dedicated per-model profile from `profiles/` if one exists.
    public func effectiveSettings(
        forModel modelName: String, packPath: String, base: QWServerSettings,
        from url: URL = defaultSettingsFileURL, ignoreProfile: Bool = false
    ) -> (settings: QWServerSettings, profile: QWKnobProfile?) {
        var resolved = base
        resolved.modelName = modelName
        if !packPath.isEmpty { resolved.modelPackPath = packPath }
        resolved.modelAlias = resolved.alias(for: modelName)
        resolved.applyModelConfigDefaults(packPath: packPath)
        if !ignoreProfile, let profile = loadProfile(forModel: modelName, from: url) {
            resolved.apply(knobProfile: profile)
            return (resolved, profile)
        }
        return (resolved, nil)
    }

    /// Persists customized per-model knob values for `settings.modelName` into `profiles/`.
    private func saveModelProfile(for settings: QWServerSettings, to url: URL) throws {
        let modelId = settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelId.isEmpty else { return }
        let rawPath = settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = settings.resolveModelReference(rawPath.isEmpty ? modelId : rawPath)
        var baseline = settings
        baseline.applyModelConfigDefaults(packPath: ref.packPath)
        let baseKnobs = baseline.validatedKnobValues
        let currentKnobs = settings.validatedKnobValues
        let store = profileStore(forSettingsURL: url)
        let existing = try? store.load(modelId: modelId)
        var diff: [String: QWKnobValue] = [:]
        for (id, value) in currentKnobs {
            if baseKnobs[id] != value || existing?.values[id] != nil {
                diff[id] = value
            }
        }
        let profile = QWKnobProfile(modelId: modelId, values: diff)
        try store.save(profile)
    }

    /// Loads settings from ~/.qwg/settings.json, falling back to defaults.
    /// The file is the truth: the decoded copy is reused only while the
    /// bytes on disk are the ones it came from.
    public func loadSettings(from url: URL = defaultSettingsFileURL) -> QWServerSettings {
        do {
            return try loadSettingsSnapshot(from: url).settings
        } catch {
            QWServerLogger.shared.log(.warn, "Failed to load settings from \(url.path): \(error)")
            return QWServerSettings()
        }
    }

    func loadSettingsSnapshot(from url: URL) throws -> (settings: QWServerSettings, keys: Set<String>) {
        lock.lock()
        defer { lock.unlock() }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            cachedSettings = nil
            cachedBytes = nil
            cachedURL = nil
            return (QWServerSettings(), [])
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QWServerConfigurationResolver.Error.invalidSettingsFile(url.path)
        }
        let keys = Set(object.keys)
        if let cached = cachedSettings, cachedBytes == data, cachedURL == url.standardizedFileURL {
            return (cached, keys)
        }

        var settings = try JSONDecoder().decode(QWServerSettings.self, from: data)
        var persistedBytes = data
        if url.standardizedFileURL == Self.defaultSettingsFileURL.standardizedFileURL && settings.pruneRemovedModels() {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let prunedData = try encoder.encode(settings)
                try prunedData.write(to: url, options: .atomic)
                persistedBytes = prunedData
            } catch {
                QWServerLogger.shared.log(.warn, "Failed to prune settings at \(url.path): \(error)")
            }
        }
        cachedSettings = settings
        cachedBytes = persistedBytes
        cachedURL = url.standardizedFileURL
        return (settings, keys)
    }

    /// Saves settings atomically to ~/.qwg/settings.json, creating directories as needed.
    public func saveSettings(_ settings: QWServerSettings, to url: URL = defaultSettingsFileURL) throws {
        lock.lock()
        defer { lock.unlock() }

        var normalized = settings
        normalized.syncActiveModelAlias()
        if url == Self.defaultSettingsFileURL {
            normalized.pruneRemovedModels()
        }

        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: dir.path) {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
        let logDirURL = normalized.logDir.canonicalFileURL
        if !fm.fileExists(atPath: logDirURL.path) {
            try? fm.createDirectory(at: logDirURL, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(normalized)
        try data.write(to: url, options: .atomic)
        try? saveModelProfile(for: normalized, to: url)
        cachedSettings = normalized
        cachedBytes = data
        cachedURL = url.standardizedFileURL
    }

    /// Forces a reload of settings from disk on next read.
    public func invalidateCache() {
        lock.lock(); defer { lock.unlock() }
        cachedSettings = nil; cachedBytes = nil; cachedURL = nil
    }
}
