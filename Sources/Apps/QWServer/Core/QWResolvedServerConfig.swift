// QWResolvedServerConfig.swift — Resolved immutable runtime server configuration.

import Foundation
import QWCore
import QWEngine
import QWQuant

/// Resolved runtime configuration adhering to the standard configuration hierarchy:
/// CLI Options > Environment Variables > Config File (~/.qwg/settings.json) > Built-in Defaults
public struct QWResolvedServerConfig: Sendable, Equatable {
    public var host: String, port: Int
    public var apiKey: String, corsAllowedOrigins: [String]
    public var allowLocalImagePaths: Bool, allowRemoteImageURLs: Bool
    public var pinModel: Bool = false
    public var modelPackPath: String, modelName: String, modelAlias: String
    public var maxContext: Int, kvMode: String
    public var ropeScalingType: String, ropeScalingFactor: Double
    public var temperature: Double, topP: Double, topK: Int
    public var maxTokens: Int, modelMaxLength: Int?
    public var repetitionPenalty: Double, seed: Int
    public var enableThinking: Bool, reasoningEffort: String, systemPrompt: String
    public var speculativeDecodingEnabled: Bool, mtpDrafts: Int, mtpBackend: String?
    public var mtpRelaxedMargin: Double, mtpRelaxedMinProb: Double, mtpDraftStrategy: String, mtpQuadraticSeeding: Bool
    public var mtpAdaptiveDepth: Bool, mtpEntropyThreshold: Double
    public var dflashDraftPackPath: String, dflashSelectorTopK: Int
    public var dsparkConfidenceSchedule: Bool? = nil
    public var gdnPrefill: String, evictPrefillScratch: String, prefillChunkSize: Int
    public var maxConcurrentSessions: Int
    public var lsaMode: String, lsaBudget: Int, lsaClusterBlock: Int, lsaSidecarPath: String
    public var lsaCrossLoopReuse: Bool = false
    public var lsaLoopShareStride: Int = 0
    public var warmRAMLimitGB: Double, ssdCacheEnabled: Bool, ssdCacheLimitGB: Double, cacheDir: String
    public var logDir: String, logFileName: String, logLevel: String
    public var lazy: Bool, sourceSummary: [String: String]
    public var loadVision: Bool = false
    public var decisionModelPackPath: String = ""
    public var decisionComputeUnit: String = "auto"
    public var decisionLoadVision: Bool = true
    public var decisionDefaultScoreReadout: String = "mean"
    public var decisionDefaultTruncation: String = "error"
    public var decisionAlwaysEmitBox: Bool = false
    /// Engine tuning from the settings/profile; `QW_*` env vars overlay it in the engine.
    public var engineTuning: QWEngineTuning = .default

    public init(
        host: String = "127.0.0.1", port: Int = 8080,
        apiKey: String = "", corsAllowedOrigins: [String] = [],
        allowLocalImagePaths: Bool = false, allowRemoteImageURLs: Bool = false,
        pinModel: Bool = false,
        modelPackPath: String = "", modelName: String = "Qwen3.8-27B", modelAlias: String = "",
        maxContext: Int = 4096, kvMode: String = "q4",
        ropeScalingType: String = "none", ropeScalingFactor: Double = 1.0,
        temperature: Double = 0.7, topP: Double = 0.9, topK: Int = 40,
        maxTokens: Int = 512, modelMaxLength: Int? = nil,
        repetitionPenalty: Double = 1.05, seed: Int = -1,
        enableThinking: Bool = true, reasoningEffort: String = "xhigh", systemPrompt: String = "",
        speculativeDecodingEnabled: Bool = true, mtpDrafts: Int = 3, mtpBackend: String? = "variational",
        mtpRelaxedMargin: Double = 1.5, mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid", mtpQuadraticSeeding: Bool = true,
        mtpAdaptiveDepth: Bool = false, mtpEntropyThreshold: Double = 0.7,
        dflashDraftPackPath: String = "", dflashSelectorTopK: Int = 16,
        dsparkConfidenceSchedule: Bool? = nil,
        gdnPrefill: String = "chunked", evictPrefillScratch: String = "auto",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        maxConcurrentSessions: Int = 8,
        lsaMode: String = "auto", lsaBudget: Int = QWLSAConfig.defaultBudget,
        lsaClusterBlock: Int = QWLSAConfig.defaultClusterBlock, lsaSidecarPath: String = "",
        lsaCrossLoopReuse: Bool = false, lsaLoopShareStride: Int = 0,
        warmRAMLimitGB: Double = 4.0, ssdCacheEnabled: Bool = false, ssdCacheLimitGB: Double = 10.0, cacheDir: String = "~/.qwg/cache",
        logDir: String = "~/.qwg/logs", logFileName: String = "qwg.log", logLevel: String = "info",
        lazy: Bool = true, loadVision: Bool = false,
        decisionModelPackPath: String = "", decisionComputeUnit: String = "auto",
        decisionLoadVision: Bool = true, decisionDefaultScoreReadout: String = "mean",
        decisionDefaultTruncation: String = "error", decisionAlwaysEmitBox: Bool = false,
        sourceSummary: [String: String] = [:]
    ) {
        self.host = host; self.port = port
        self.apiKey = apiKey; self.corsAllowedOrigins = corsAllowedOrigins
        self.allowLocalImagePaths = allowLocalImagePaths; self.allowRemoteImageURLs = allowRemoteImageURLs
        self.pinModel = pinModel
        self.modelPackPath = modelPackPath; self.modelName = modelName; self.modelAlias = modelAlias
        self.maxContext = maxContext; self.kvMode = kvMode
        self.ropeScalingType = ropeScalingType; self.ropeScalingFactor = ropeScalingFactor
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.maxTokens = maxTokens; self.modelMaxLength = modelMaxLength
        self.repetitionPenalty = repetitionPenalty; self.seed = seed
        self.enableThinking = enableThinking; self.reasoningEffort = reasoningEffort; self.systemPrompt = systemPrompt
        self.speculativeDecodingEnabled = speculativeDecodingEnabled; self.mtpDrafts = mtpDrafts; self.mtpBackend = mtpBackend
        self.mtpRelaxedMargin = mtpRelaxedMargin; self.mtpRelaxedMinProb = mtpRelaxedMinProb; self.mtpDraftStrategy = mtpDraftStrategy; self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.mtpAdaptiveDepth = mtpAdaptiveDepth; self.mtpEntropyThreshold = mtpEntropyThreshold
        self.dflashDraftPackPath = dflashDraftPackPath; self.dflashSelectorTopK = dflashSelectorTopK
        self.dsparkConfidenceSchedule = dsparkConfidenceSchedule
        self.gdnPrefill = gdnPrefill; self.evictPrefillScratch = evictPrefillScratch
        self.prefillChunkSize = prefillChunkSize
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions))
        self.lsaMode = lsaMode; self.lsaBudget = lsaBudget
        self.lsaClusterBlock = lsaClusterBlock; self.lsaSidecarPath = lsaSidecarPath
        self.lsaCrossLoopReuse = lsaCrossLoopReuse; self.lsaLoopShareStride = lsaLoopShareStride
        self.warmRAMLimitGB = warmRAMLimitGB; self.ssdCacheEnabled = ssdCacheEnabled; self.ssdCacheLimitGB = ssdCacheLimitGB; self.cacheDir = cacheDir
        self.logDir = logDir; self.logFileName = logFileName; self.logLevel = logLevel
        self.lazy = lazy; self.loadVision = loadVision
        self.decisionModelPackPath = decisionModelPackPath; self.decisionComputeUnit = decisionComputeUnit
        self.decisionLoadVision = decisionLoadVision; self.decisionDefaultScoreReadout = decisionDefaultScoreReadout
        self.decisionDefaultTruncation = decisionDefaultTruncation; self.decisionAlwaysEmitBox = decisionAlwaysEmitBox
        self.sourceSummary = sourceSummary
    }

    /// Cohesive sampling parameters view resolved for the server.
    public var samplingParameters: QWSamplingParameters {
        QWSamplingParameters(temperature: temperature, topP: topP, topK: topK,
                             maxTokens: maxTokens, repetitionPenalty: repetitionPenalty, seed: seed)
    }

    /// Cohesive speculative decoding parameters resolved for the server.
    public var speculativeParameters: QWSpeculativeParameters {
        QWSpeculativeParameters(enabled: speculativeDecodingEnabled, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                                mtpRelaxedMargin: mtpRelaxedMargin, mtpRelaxedMinProb: mtpRelaxedMinProb,
                                mtpDraftStrategy: mtpDraftStrategy, mtpQuadraticSeeding: mtpQuadraticSeeding,
                                adaptiveDepth: mtpAdaptiveDepth, entropyThreshold: mtpEntropyThreshold,
                                dsparkConfidenceSchedule: dsparkConfidenceSchedule)
    }

    /// Cohesive Latent Sparse Attention configuration resolved for the server.
    public var lsaConfig: QWLSAConfig {
        let explicitBudget = sourceSummary["lsaBudget"] == "cli"
            || sourceSummary["lsaBudget"] == "env"
            || lsaBudget != QWLSAConfig.defaultBudget
        return QWLSAConfig(mode: QWLSAMode.resolve(lsaMode) ?? .off, budget: lsaBudget,
                           clusterBlock: lsaClusterBlock, sidecarPath: lsaSidecarPath,
                           crossLoopReuse: lsaCrossLoopReuse, loopShareStride: lsaLoopShareStride,
                           explicitBudget: explicitBudget)
    }
}
