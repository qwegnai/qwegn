// QWEngineBuildOptions.swift — What a consumer asks an engine to be.
//
// The consumer fills in what it knows. The module reads what it
// understands and ignores the rest: a Flash-Next engine has no MTP
// drafts to configure, and a dense engine has no hybrid cache.

import Foundation
import QWCore
import QWKV

/// Everything a consumer can ask an engine to be built with.
public struct QWEngineBuildOptions: Sendable {
    /// The pack the engine serves.
    public var packPath: String
    /// The context the engine holds.
    public var maxTokens: Int
    /// The KV layout the consumer asked for. The module resolves it:
    /// a layout a family does not implement is not silently accepted.
    public var kvMode: String
    /// The RAM the engine leaves for the rest of the machine.
    public var osReserveGiB: Double?
    /// How many prompt tokens the engine prefills at once.
    public var prefillChunkSize: Int
    /// Where the prefix cache lives, when a cache is wanted.
    public var prefixCacheDirectory: URL?
    /// The cache's limits, when a cache is wanted.
    public var cacheConfig: QWCacheConfig?
    /// How many drafts the drafter may propose.
    public var mtpDrafts: Int
    /// Which drafter the consumer asked for.
    public var mtpBackend: QWMTPBackend?
    /// When the prefill scratch is released.
    public var prefillScratchEviction: QWPrefillScratchEvictionPolicy
    /// Whether the drafter adapts its depth.
    public var mtpAdaptiveDepth: Bool?
    /// The entropy above which the drafter adapts.
    public var mtpAdaptiveEntropyThreshold: Double?
    /// Log-probability / logit margin for Variational MTP relaxed acceptance.
    public var mtpRelaxedMargin: Double
    /// Probability floor for Variational MTP relaxed acceptance (`0` disables it).
    public var mtpRelaxedMinProb: Double
    /// Multi-draft proposal strategy ("hybrid", "mtp", "ngram").
    public var mtpDraftStrategy: String
    /// Whether Quadratic Decoding prefix seeding is enabled.
    public var mtpQuadraticSeeding: Bool
    /// The rope YaRN factor, when the consumer asks for one.
    public var ropeYarnFactor: Double?
    /// The drafter pack, when the consumer asks for a DFlash drafter.
    public var dflashDraftPackPath: String?
    /// How many candidates the DFlash selector keeps.
    public var dflashSelectorTopK: Int?
    /// The GDN prefill mode the consumer asked for.
    public var gdnPrefill: String
    /// Latent Sparse Attention (LSA) configuration.
    public var lsaConfig: QWLSAConfig
    /// Number of resident hot experts per MoE layer in unified memory; cold experts stream from SSD.
    public var residentExpertsPerLayer: Int?
    /// Maximum concurrent sessions for KV prefix cache slots and batch scheduling (1...16).
    public var maxConcurrentSessions: Int
    /// Per-model performance and memory tunables (environment overrides apply in the engine).
    public var tuning: QWEngineTuning

    public init(packPath: String, maxTokens: Int = 4096, kvMode: String = "f16",
                osReserveGiB: Double? = nil, prefillChunkSize: Int = QWPrefillLimits.chunkRows,
                prefixCacheDirectory: URL? = nil, cacheConfig: QWCacheConfig? = nil,
                mtpDrafts: Int = 0, mtpBackend: QWMTPBackend? = nil,
                prefillScratchEviction: QWPrefillScratchEvictionPolicy = .auto,
                mtpAdaptiveDepth: Bool? = nil, mtpAdaptiveEntropyThreshold: Double? = nil,
                mtpRelaxedMargin: Double = 1.5,
                mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
                mtpDraftStrategy: String = "hybrid",
                mtpQuadraticSeeding: Bool = true,
                ropeYarnFactor: Double? = nil, dflashDraftPackPath: String? = nil,
                dflashSelectorTopK: Int? = nil, gdnPrefill: String = "chunked",
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                residentExpertsPerLayer: Int? = nil,
                maxConcurrentSessions: Int? = nil,
                tuning: QWEngineTuning = .default) {
        self.packPath = packPath
        self.maxTokens = maxTokens
        self.kvMode = kvMode
        self.osReserveGiB = osReserveGiB ?? tuning.osReserveGiB
        self.prefillChunkSize = prefillChunkSize
        self.prefixCacheDirectory = prefixCacheDirectory
        self.cacheConfig = cacheConfig
        self.mtpDrafts = mtpDrafts
        self.mtpBackend = mtpBackend
        self.prefillScratchEviction = prefillScratchEviction
        self.mtpAdaptiveDepth = mtpAdaptiveDepth
        self.mtpAdaptiveEntropyThreshold = mtpAdaptiveEntropyThreshold
        self.mtpRelaxedMargin = mtpRelaxedMargin
        self.mtpRelaxedMinProb = mtpRelaxedMinProb
        self.mtpDraftStrategy = mtpDraftStrategy
        self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.ropeYarnFactor = ropeYarnFactor
        self.dflashDraftPackPath = dflashDraftPackPath
        self.dflashSelectorTopK = dflashSelectorTopK
        self.gdnPrefill = gdnPrefill
        self.lsaConfig = lsaConfig
        self.residentExpertsPerLayer = residentExpertsPerLayer ?? tuning.residentExpertsPerLayer
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions ?? cacheConfig?.maxConcurrentSessions ?? QWCacheConfig.defaultMaxConcurrentSessions))
        self.tuning = tuning
    }
}
