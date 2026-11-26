// QWEngineDriver.swift — C23: the kernel-encoding integration leg (spec-21
// "leg 2/3": the plans QWEngine.swift binds become REAL launches here).
//
// Zero new math law: every compute step rides a gated door exactly as its
// parity test drove it. The ONLY new kernels are the integration epilogues
// (qwengine.metal): the q̂/k̂ l2-norm and the β/γ gate folds the PYTHON
// oracle used to compute between doors — no component ever needed a device
import Foundation
import Darwin
import Metal
import os
import Synchronization
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWVision

public final class QWEngineDriver: @unchecked Sendable {
    static let P = "model.language_model."
    /// Geometry of the loaded model, read from its own config.json.
    public let geometry: QWModelGeometry
    /// The container this model's weights are held in, and so the basis its
    /// activations are in.
    public let weightFormat: QWWeightFormat
    /// Why the size-preserving family was declined, when it was.
    public let weightFormatDeclineReason: String?

    let hal: QWHal
    let hasVisionWeights: Bool
    public let ropeRows: Int
    let hyperRuntime: QWHyperStreamRuntime?
    let inkling2Runtime: QWInkling2Runtime?
    let standardState: QWDriverStandardState?
    let prefillModeLock = NSLock()
    var storedPrefillMode: GDNPrefillMode = .chunked
    public var gdnPrefillMode: GDNPrefillMode {
        get { prefillModeLock.withLock { storedPrefillMode } }
        set { prefillModeLock.withLock { storedPrefillMode = newValue } }
    }
    public var gdnSolveBackend: QWGDNPrefill.SolveBackend {
        get { prefillModeLock.withLock { gdnPrefill.solveBackend } }
        set { prefillModeLock.withLock { gdnPrefill.solveBackend = newValue } }
    }

    /// The kernel each layer runs. Resolved once at load time.
    let layerKernels: [any QWLayerKernel]

    /// 15 MTP Layer weights (Phase 4, section 5A.4 / 5A.5).
    typealias MTPWeights = QWMTPDraftWeights
    public let mtpAdaptiveDepth: Bool
    public let mtpAdaptiveEntropyThreshold: Double
    public var mtpRelaxedMargin: Double
    public var mtpDraftStrategy: String
    public var mtpQuadraticSeeding: Bool
    var activeRelaxedMarginOverride: Float?
    var activeCollectLogprobs: Bool = false
    var activeConstraint: (constraint: any QWTokenConstraint, sampler: QWSampler)? = nil
    var activeVisualSpans: [QWVisualSpan] = []
    var activeCacheSalt: String? = nil
    public var mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb {
        didSet { inkling2Runtime?.configuredRelaxedMinProb = mtpRelaxedMinProb }
    }

    /// Effective log-probability / logit margin for Variational MTP verification.
    public var effectiveRelaxedMargin: Float {
        if let override = activeRelaxedMarginOverride {
            return override
        }
        guard mtpBackend == .variational else { return 0 }
        return Float(max(0.0, mtpRelaxedMargin))
    }
    /// EXP-12: batched q4g verify head (reuses mtpDraftHead's planes).
    let mtpVerifyHeadQ4: Bool
    let mtpStageProfiling: Bool
    let mtpVerificationProfiling: Bool
    let admission = QWSessionAdmissionManager()
    let commandFrame = QWCommandFrame()
    let speculativeCoordinator = QWSpeculativeSessionCoordinator()

    var verificationGpuMsByStage: [String: Double]? {
        get { speculativeCoordinator.verificationGpuMsByStage }
        set { speculativeCoordinator.verificationGpuMsByStage = newValue }
    }
    var pendingMTPProfile: (draft: QWMTPStageTiming, verification: QWMTPStageTiming)? {
        get { speculativeCoordinator.pendingMTPProfile }
        set { speculativeCoordinator.pendingMTPProfile = newValue }
    }
    var pendingMTPAdaptiveEntropy: [Double]? {
        get { speculativeCoordinator.pendingMTPAdaptiveEntropy }
        set { speculativeCoordinator.pendingMTPAdaptiveEntropy = newValue }
    }
    var lastMTPRoundProfile: QWMTPRoundProfile? {
        get { speculativeCoordinator.lastMTPRoundProfile }
        set { speculativeCoordinator.lastMTPRoundProfile = newValue }
    }

    /// Dedicated MTP draft scratch buffers (Phase 4, section 5A.6).
    typealias MTPScratch = QWMTPDraftScratch

    public internal(set) var mtpDraftProposals: Int {
        get { speculativeCoordinator.mtpDraftProposals }
        set { speculativeCoordinator.mtpDraftProposals = newValue }
    }
    public internal(set) var mtpDraftAccepted: Int {
        get { speculativeCoordinator.mtpDraftAccepted }
        set { speculativeCoordinator.mtpDraftAccepted = newValue }
    }
    var lastRoundUsedPLD: Bool {
        get { speculativeCoordinator.lastRoundUsedPLD }
        set { speculativeCoordinator.lastRoundUsedPLD = newValue }
    }
    var lastDraftedPairs: Int {
        get { speculativeCoordinator.lastDraftedPairs }
        set { speculativeCoordinator.lastDraftedPairs = newValue }
    }
    public var dflashDraftPackPath: String? {
        hyperRuntime?.dflash?.packPath ?? standardState?.dflashDrafter?.packPath
    }
    public var dflashSelectorTopK: Int? {
        hyperRuntime?.dflash?.config.selectorTopK ?? standardState?.dflashDrafter?.config.selectorTopK
    }
    var verifiedSession: QWEngineSession? {
        get { speculativeCoordinator.verifiedSession }
        set { speculativeCoordinator.verifiedSession = newValue }
    }
    var verifiedInputs: [Int] {
        get { speculativeCoordinator.verifiedInputs }
        set { speculativeCoordinator.verifiedInputs = newValue }
    }
    var preparedGreedyLease: QWGreedyRoundLease? {
        get { speculativeCoordinator.preparedGreedyLease }
        set { speculativeCoordinator.preparedGreedyLease = newValue }
    }
    var preparedStochasticWitness: QWStochasticRoundWitness? {
        get { speculativeCoordinator.preparedStochasticWitness }
        set { speculativeCoordinator.preparedStochasticWitness = newValue }
    }
    var requestGate: QWGenerationRequestGate { admission.requestGate }
    private let statsLock = NSLock()
    private var _lastPrefixCacheStats: QWPrefixCacheStats?
    public internal(set) var lastPrefixCacheStats: QWPrefixCacheStats? {
        get { statsLock.withLock { _lastPrefixCacheStats } }
        set { statsLock.withLock { _lastPrefixCacheStats = newValue } }
    }
    public let mtpDrafts: Int
    /// Performance and memory tunables, environment overrides applied.
    public let tuning: QWEngineTuning
    public let kvMode: QWKVMode
    public let rawKVMode: String
    public let capacityPlan: QWCapacityPlan
    public let capacityLogMessage: String
    /// The MTP execution backend (LighteningMTP L1). The selection is
    /// immutable for the driver lifetime and is part of the session and
    /// prefix-cache identity; an explicitly selected backend cannot be
    public let mtpBackend: QWMTPBackend

    /// The prefix-cache policy key (LighteningMTP L1). It carries the
    /// backend's numerical identity: a Standard and a Lightening run are
    /// never admitted to each other's cached prefixes, even while the L1
    static func prefixPolicyKey(backend: QWMTPBackend, drafts: Int,
                                mode: GDNPrefillMode, cacheSalt: String? = nil) -> String {
        let base = drafts > 0 ? "\(backend.rawValue)+\(mode.rawValue)" : mode.rawValue
        if let salt = cacheSalt?.trimmingCharacters(in: .whitespacesAndNewlines), !salt.isEmpty {
            return "\(base)|salt=\(salt)"
        }
        return base
    }

    public let prefillScratchEviction: QWPrefillScratchEvictionPolicy
    public let prefillChunkSize: Int
    /// Latent Sparse Attention configuration for this driver instance.
    public let lsaConfig: QWLSAConfig
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    var flight: NSLock { admission.flight }
    var generateLock: NSRecursiveLock { admission.generateLock }
    var activeCB: (any MTLCommandBuffer)? {
        get { commandFrame.activeCB }
        set { commandFrame.activeCB = newValue }
    }
    var activeEnc: (any MTLComputeCommandEncoder)? {
        get { commandFrame.activeEnc }
        set { commandFrame.activeEnc = newValue }
    }
    var pendingColdUnloads: [(streamer: ColdExpertStreamer, stagedCount: Int)] {
        get { commandFrame.pendingColdUnloads }
        set { commandFrame.pendingColdUnloads = newValue }
    }
    var stepCBs: [any MTLCommandBuffer]? {
        get { commandFrame.stepCBs }
        set { commandFrame.stepCBs = newValue }
    }
    var decodeTel: QWDecodeTelemetry {
        get { commandFrame.decodeTel }
        set { commandFrame.decodeTel = newValue }
    }
    var coldEvents: ColdEventPipeline? {
        get { commandFrame.coldEvents }
        set { commandFrame.coldEvents = newValue }
    }
    var spanMapIds: (any MTLBuffer)?  // qwg_moe_span_ids output of the current prefill span (host-read)

    var slotLock: NSLock { admission.slotLock }
    var freeSlots: [GdnSlot] {
        get { admission.freeSlots }
        set { admission.freeSlots = newValue }
    }

    public init(packPath: String, maxTokens: Int = 4096, ropeRows: Int = 8192,
                kvMode: QWKVMode = .f16,
                rawKVMode: String? = nil,
                gdnPrefillMode: GDNPrefillMode = .chunked,
                weightArenaSizing: QWWeightArenaSizing = .planned,
                mtpDrafts: Int = 0,
                mtpBackend: QWMTPBackend? = nil,
                prefillScratchEviction: QWPrefillScratchEvictionPolicy = .auto,
                mtpAdaptiveDepth: Bool? = nil,
                mtpAdaptiveEntropyThreshold: Double? = nil,
                mtpRelaxedMargin: Double = 1.5,
                mtpDraftStrategy: String = "hybrid",
                mtpQuadraticSeeding: Bool = true,
                ropeYarnFactor: Double? = nil,
                dflashDraftPackPath: String? = nil,
                dflashSelectorTopK: Int? = nil,
                dsparkConfidenceSchedule: Bool? = nil,
                osReserveGiB: Double? = nil,
                prefillChunkSize: Int = QWPrefillLimits.chunkRows,
                prewarmDoors: [String] = DenseModule.requiredDoors,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                residentExpertsPerLayer: Int? = nil,
                tuning: QWEngineTuning = .default,
                prefixCacheConfig: QWCacheConfig? = nil,
                prefixCacheDirectory: URL? = nil,
                maxConcurrentSessions: Int? = nil) throws {
        let resolvedRawKV = rawKVMode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            ?? String(describing: kvMode)
        self.kvMode = kvMode
        self.rawKVMode = resolvedRawKV
        let cfg = try Self.resolveStartup(packPath: packPath, kvMode: kvMode,
                                          rawKVMode: rawKVMode,
                                          mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                                          prefillScratchEviction: prefillScratchEviction,
                                          mtpAdaptiveDepth: mtpAdaptiveDepth,
                                          mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
                                          mtpRelaxedMargin: mtpRelaxedMargin,
                                          mtpDraftStrategy: mtpDraftStrategy,
                                          mtpQuadraticSeeding: mtpQuadraticSeeding,
                                          dflashDraftPackPath: dflashDraftPackPath,
                                          dflashSelectorTopK: dflashSelectorTopK,
                                          dsparkConfidenceSchedule: dsparkConfidenceSchedule)
        if cfg.isHyperStream {
            let resolvedTuning = tuning.overlaid(environment: cfg.environment)
            let evictPrefill: Bool = switch cfg.prefillScratchEviction {
            case .always: true
            case .never: false
            case .auto: Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000.0 <= 36.5
            }
            let cacheDir = prefixCacheDirectory ?? prefixCacheConfig?.cacheDirectory
            let warmLimit = prefixCacheConfig?.warmByteLimit ?? 0
            let ssdLimit = prefixCacheConfig?.ssdByteLimit ?? 0
            let sessions = maxConcurrentSessions
                ?? prefixCacheConfig?.maxConcurrentSessions
                ?? FNPrefixWarmStore.defaultMaxSlots
            let runtime = try autoreleasepool {
                try QWHyperStreamRuntime(
                    packPath: packPath,
                    maxTokens: maxTokens,
                    kvMode: resolvedRawKV,
                    osReserveGiB: osReserveGiB ?? resolvedTuning.osReserveGiB,
                    prefillChunkSize: prefillChunkSize,
                    prefixCacheDirectory: cacheDir,
                    prefixCacheWarmByteLimit: warmLimit,
                    prefixCacheSSDByteLimit: ssdLimit,
                    maxConcurrentSessions: sessions,
                    evictPrefillScratch: evictPrefill,
                    prewarmDoors: QWKernelInventory.flash,
                    lsaConfig: lsaConfig,
                    streamNgrams: resolvedTuning.streamNgrams,
                    ropeYarnFactor: ropeYarnFactor,
                    mtpBackend: mtpBackend?.rawValue,
                    mtpDrafts: mtpDrafts,
                    dflashDraftPackPath: dflashDraftPackPath ?? cfg.dflashDraftPackPath,
                    dflashSelectorTopK: dflashSelectorTopK ?? cfg.dflashSelectorTopK,
                    mtpDraftStrategy: cfg.mtpDraftStrategy
                )
            }
            let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
            let formatChoice = try Self.weightFormatChoice(
                for: root,
                quantization: runtime.descriptor.quantization,
                geometry: runtime.geometry,
                mtpDrafts: 0,
                pack: runtime.pack
            )
            self.geometry = runtime.geometry
            self.plan = runtime.plan
            self.weightFormat = formatChoice.format
            self.weightFormatDeclineReason = formatChoice.declineReason
            self.hal = runtime.hal
            self.hasVisionWeights = runtime.hasVisionWeights
            self.ropeRows = runtime.geometry.descriptor.maxPositionEmbeddings
            self.layerKernels = runtime.layerKernels
            self.mtpAdaptiveDepth = cfg.mtpAdaptiveDepth
            self.mtpAdaptiveEntropyThreshold = cfg.mtpAdaptiveEntropyThreshold
            self.mtpRelaxedMargin = cfg.mtpRelaxedMargin
            self.mtpDraftStrategy = cfg.mtpDraftStrategy
            self.mtpQuadraticSeeding = cfg.mtpQuadraticSeeding
            self.mtpStageProfiling = cfg.mtpStageProfiling
            self.mtpVerificationProfiling = cfg.mtpVerificationProfiling
            self.mtpVerifyHeadQ4 = cfg.verifyHeadQ4Enabled
            self.storedPrefillMode = gdnPrefillMode
            self.mtpDrafts = mtpDrafts
            self.tuning = resolvedTuning
            self.capacityPlan = runtime.capacityPlan
            self.capacityLogMessage = runtime.capacityLogMessage
            self.mtpBackend = cfg.mtpBackend
            self.prefillScratchEviction = cfg.prefillScratchEviction
            self.prefillChunkSize = runtime.maxPrefillRows
            self.lsaConfig = lsaConfig
            self.hyperRuntime = runtime
            self.inkling2Runtime = nil
            self.standardState = nil
            return
        }
        if cfg.isInkling2 {
            let resolvedTuning = tuning.overlaid(environment: cfg.environment)
            let warmLimit = prefixCacheConfig?.warmByteLimit ?? 0
            let sessions = maxConcurrentSessions ?? prefixCacheConfig?.maxConcurrentSessions ?? 8
            let inkRelaxedMargin = cfg.mtpBackend == .variational ? cfg.mtpRelaxedMargin : 0.0
            let initialMinProb = mtpRelaxedMinProb
            let runtime = try autoreleasepool {
                try QWInkling2Runtime(
                    packPath: packPath,
                    maxTokens: maxTokens,
                    kvMode: kvMode,
                    rawKVMode: resolvedRawKV,
                    osReserveGiB: osReserveGiB ?? resolvedTuning.osReserveGiB,
                    prewarmDoors: QWKernelInventory.inkling2,
                    prefixCacheWarmByteLimit: warmLimit,
                    maxConcurrentSessions: sessions,
                    mtpDrafts: mtpDrafts,
                    prefillChunkSize: prefillChunkSize,
                    lsaConfig: lsaConfig,
                    mtpRelaxedMargin: inkRelaxedMargin,
                    mtpRelaxedMinProb: initialMinProb,
                    moeSpecBudget: resolvedTuning.moeSpecBudget
                )
            }
            self.geometry = runtime.geometry
            self.plan = runtime.plan
            self.weightFormat = runtime.weightFormat
            self.weightFormatDeclineReason = runtime.weightFormatDeclineReason
            self.hal = runtime.hal
            self.hasVisionWeights = runtime.hasVisionWeights
            self.ropeRows = runtime.geometry.descriptor.maxPositionEmbeddings
            self.layerKernels = runtime.layerKernels
            self.mtpAdaptiveDepth = cfg.mtpAdaptiveDepth
            self.mtpAdaptiveEntropyThreshold = cfg.mtpAdaptiveEntropyThreshold
            self.mtpRelaxedMargin = cfg.mtpRelaxedMargin
            self.mtpDraftStrategy = cfg.mtpDraftStrategy
            self.mtpQuadraticSeeding = cfg.mtpQuadraticSeeding
            self.mtpStageProfiling = cfg.mtpStageProfiling
            self.mtpVerificationProfiling = cfg.mtpVerificationProfiling
            self.mtpVerifyHeadQ4 = cfg.verifyHeadQ4Enabled
            self.storedPrefillMode = gdnPrefillMode
            self.mtpDrafts = runtime.dspark != nil ? runtime.requestedMtpDrafts : max(0, min(15, mtpDrafts))
            self.tuning = resolvedTuning
            self.capacityPlan = runtime.capacityPlan
            self.capacityLogMessage = runtime.capacityLogMessage
            self.mtpBackend = cfg.mtpBackend
            self.prefillScratchEviction = cfg.prefillScratchEviction
            self.prefillChunkSize = runtime.session.maxChunk
            self.lsaConfig = runtime.lsaOperator?.config ?? lsaConfig.resolvingAdaptiveBudget(forMaxContext: runtime.capacity)
            self.hyperRuntime = nil
            self.inkling2Runtime = runtime
            self.standardState = nil
            return
        }
        let resolvedSessions = maxConcurrentSessions ?? prefixCacheConfig?.maxConcurrentSessions
        let resolvedSlotCount = Self.gdnSlotCount(requestedSessions: resolvedSessions)
        let startup = try Self.prepareContainersAndGeometry(
            packPath: packPath, maxTokens: maxTokens, ropeRows: ropeRows, kvMode: kvMode,
            prefillChunkSize: prefillChunkSize, mtpDrafts: mtpDrafts,
            residentExpertsPerLayer: residentExpertsPerLayer, osReserveGiB: osReserveGiB,
            cfg: cfg, baseTuning: tuning, requestedSessions: resolvedSessions)
        let resolvedCfg = startup.cfg
        let chunkRows = startup.prefillChunkSize
        let geometry = startup.geometry
        let prelimHal = startup.hal
        let stContainer = startup.stContainer
        self.mtpDrafts = startup.mtpDrafts
        self.weightFormat = startup.weightFormat
        self.weightFormatDeclineReason = startup.weightFormatDeclineReason
        self.tuning = startup.tuning
        self.geometry = geometry
        self.hal = prelimHal
        self.mtpBackend = resolvedCfg.mtpBackend
        self.prefillChunkSize = chunkRows
        let frozenPlan = try QWEngineBootstrap.makePlan(
            geometry: geometry, packPath: packPath,
            prefillChunkRows: chunkRows, kvStorageMode: String(describing: kvMode))
        self.plan = frozenPlan
        self.prefillScratchEviction = resolvedCfg.prefillScratchEviction
        self.mtpAdaptiveDepth = resolvedCfg.mtpAdaptiveDepth
        self.mtpAdaptiveEntropyThreshold = resolvedCfg.mtpAdaptiveEntropyThreshold
        self.mtpRelaxedMargin = resolvedCfg.mtpRelaxedMargin
        self.mtpDraftStrategy = resolvedCfg.mtpDraftStrategy
        self.mtpQuadraticSeeding = resolvedCfg.mtpQuadraticSeeding
        self.mtpStageProfiling = resolvedCfg.mtpStageProfiling
        self.mtpVerificationProfiling = resolvedCfg.mtpVerificationProfiling
        self.mtpVerifyHeadQ4 = resolvedCfg.verifyHeadQ4Enabled
        self.storedPrefillMode = gdnPrefillMode
        let environment = resolvedCfg.environment
        let numV = geometry.numVirtualLayers
        let attnL = (0..<numV).filter { geometry.isVirtualLayerFullAttention($0) }
        let hcaL = (0..<numV).filter { geometry.isVirtualLayerCompressedAttention($0) }
        let gdnL = (0..<numV).filter { geometry.isVirtualLayerGDN($0) }

        let pipes = try Self.resolveCorePipelines(
            hal: prelimHal, weightFormat: startup.weightFormat, mhcConfig: geometry.mhcConfig)
        let mhcSplitVal = try QWMHCSplitRouter(hal: prelimHal, geometry: geometry)
        let attnPrefillVal = try QWAttnPrefill(hal: prelimHal, backend: .naxX64, geometry: geometry)
        let pagedAttnPrefillVal = try QWAttnPagedPrefill(hal: prelimHal, geometry: geometry)
        let pagedAttnPrefillGenVal = try QWAttnPagedPrefillGeneral(hal: prelimHal, geometry: geometry)

        var gMap = [Int](repeating: -1, count: numV); for (i, gl) in gdnL.enumerated() { gMap[gl] = i }
        var aMap = [Int](repeating: -1, count: numV); for (i, al) in attnL.enumerated() { aMap[al] = i }
        var hMap = [Int](repeating: -1, count: numV); for (i, hl) in hcaL.enumerated() { hMap[hl] = i }
        let headVal = try QWHead(hal: prelimHal, geometry: geometry)
        let gdnPrefillVal = try QWGDNPrefill(hal: prelimHal, scanBackend: .hybrid, solveBackend: .scalar)

        let budget = try Self.planArenaAndBudget(
            geometry: geometry, hal: prelimHal, weightFormat: startup.weightFormat,
            weightFormatDeclineReason: startup.weightFormatDeclineReason, tuning: startup.tuning,
            osReserveGiB: startup.osReserveGiB, prefillChunkSize: chunkRows, maxTokens: maxTokens,
            ropeRows: ropeRows, kvMode: kvMode, attnLayerCount: attnL.count, gdnLayerCount: gdnL.count,
            cfg: resolvedCfg, mtpDrafts: startup.mtpDrafts, mtpWeightDrafts: startup.mtpWeightDrafts,
            requestedSessions: resolvedSessions)
        self.capacityPlan = budget.capacityPlan
        self.capacityLogMessage = budget.capacityLogMessage
        let arenaVal = budget.arena

        let wsub = budget.arena.weightsRegion()
        var L: WeightLoader
        if let pk = startup.pack {
            L = WeightLoader(pk: pk, hl: prelimHal, psoCvt: pipes.psoBf16F16,
                             wsub: wsub, plan: budget.weightPlan, geometry: geometry, format: startup.weightFormat)
        } else if let st = stContainer {
            let psoQuant = try prelimHal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
            L = WeightLoader(st: st, hl: prelimHal, psoCvt: pipes.psoBf16F16, psoQuantBF16: psoQuant,
                             wsub: wsub, plan: budget.weightPlan, geometry: geometry, format: startup.weightFormat)
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }

        let hasMTPDrafter = (startup.mtpWeightDrafts > 0 || resolvedCfg.mtpBackend == .dflash2 || resolvedCfg.mtpBackend == .dspark) && startup.mtpDrafts > 0
        let embHead = try Self.loadEmbeddingsAndHead(
            loader: &L, pack: startup.pack, stContainer: stContainer, packPath: packPath,
            hal: prelimHal, geometry: geometry, cfg: resolvedCfg,
            mtpDrafts: startup.mtpDrafts, hasMTPDrafter: hasMTPDrafter, budget: budget)
        let visionWeightsPresent: Bool
        if geometry.descriptor.capabilities.contains(.vision), geometry.descriptor.visDepth > 0 {
            if let pk = startup.pack {
                visionWeightsPresent = QWVisionOnDemandBackend.hasVisionWeights(modelPath: pk.url.path)
            } else if let er = embHead.embedReader {
                visionWeightsPresent = QWVisionOnDemandBackend.hasVisionWeights(
                    modelPath: er.shardURL.deletingLastPathComponent().path)
            } else {
                visionWeightsPresent = false
            }
        } else {
            visionWeightsPresent = false
        }
        self.hasVisionWeights = visionWeightsPresent

        let hcaSidecar = Self.loadHCALSASidecar(
            hal: prelimHal, geometry: geometry, lsaConfig: lsaConfig,
            packPath: packPath, pack: startup.pack, safetensors: stContainer, hcaLayers: hcaL)
        let hcaLsaBufferVal = hcaSidecar?.buffer
        L.hcaLsaByLayer = hcaSidecar?.byLayer ?? [:]
        let decoder = try L.loadDecoderWeights(geometry: geometry)
        if startup.tuning.expertCache {
            for lw in decoder.layers { lw.moe?.coldStreamer?.enableDynamicResidency() }
        }
        self.layerKernels = try QWLayerKernelRegistry.resolve(layers: decoder.layers, plan: frozenPlan)
        let finalNormVal = decoder.finalNorm
        let mhcExitHeadVal = try L.loadMHCExitHeadIfNeeded()
        let multimaxParamsVal = try L.loadMultimaxIfNeeded()

        let origMaxPos = geometry.originalMaxPositionEmbeddings
        let nativeYarn = geometry.yarnRopeConfig
        let betaFast = Float(nativeYarn?.betaFast ?? 32.0)
        let betaSlow = Float(nativeYarn?.betaSlow ?? 1.0)
        let yarnFactor = try Self.resolveYaRNFactor(environment["QW_ROPE_YARN_FACTOR"] ?? ropeYarnFactor.map { String($0) })
        let yarnConfig: QWYaRNConfig? = yarnFactor.map {
            QWYaRNConfig(factor: $0, originalMaxPosition: origMaxPos, betaFast: betaFast, betaSlow: betaSlow)
        } ?? (budget.effectiveMaxTokens > geometry.maxPositionEmbeddings
              ? QWYaRNConfig.config1M(targetContext: budget.effectiveMaxTokens, originalMaxPosition: origMaxPos,
                                      betaFast: betaFast, betaSlow: betaSlow)
              : nativeYarn.map {
                  QWYaRNConfig(factor: Float($0.factor), originalMaxPosition: $0.originalMaxPositionEmbeddings,
                               betaFast: betaFast, betaSlow: betaSlow)
              })
        let tableRopeRows = geometry.usesLatentOrCompressedAttention ? 1 : budget.effectiveRopeRows
        let tables = QWRopeTables(rows: tableRopeRows, yarn: yarnConfig, geometry: geometry)
        self.ropeRows = budget.effectiveRopeRows
        try budget.weightPlan.check(QWTensorShape([tableRopeRows, geometry.rotaryDims], .f32), at: L.weightRequest)
        let ropeApp = try QWRopeApply(tables: tables, hal: prelimHal, weights: wsub, arena: budget.arena, geometry: geometry)
        L.weightRequest += 1
        let eWeights = try L.loadEchoDraftWeightsIfNeeded(drafts: startup.mtpWeightDrafts)
        let mWeights = try Self.loadMTPWeights(&L, drafts: startup.mtpWeightDrafts, geometry: geometry)
        try budget.weightPlan.finish(requests: L.weightRequest, usedBytes: wsub.usedBytes)
        wsub.markSteadyState()
        startup.pack?.adviseDontNeed()
        stContainer?.adviseDontNeed()

        let aDec = geometry.usesLatentOrCompressedAttention ? nil
            : try QWAttnDecode(hal: prelimHal, sharedPrefixRows: startup.mtpDrafts > 0 ? min(8, startup.mtpDrafts + 1) : nil,
                               kvMode: kvMode, geometry: geometry)
        let gdnConvVal = try QWGDNConv(hal: prelimHal, geometry: geometry)
        let gdnDecodeVal = try QWGDNDecode(hal: prelimHal)
        let needsMTPCache = startup.mtpWeightDrafts > 0 && startup.mtpDrafts > 0
        let caches = try Self.makeKVCaches(
            hal: prelimHal, arena: budget.arena, attnLayers: attnL, kvMode: kvMode,
            geometry: geometry, mtpDrafts: needsMTPCache ? startup.mtpDrafts : 0,
            trunkKVBytes: budget.trunkKVBytes, mtpKVBytes: budget.mtpKVBytes)

        let asub = budget.arena.activationsRegion(); let gsub = budget.arena.gdnRegion()
        let pOff = try gsub.allocate(bytes: budget.pCarveBytes)
        let mtpPartialsOff: Int = startup.mtpWeightDrafts > 0 ? try gsub.allocate(bytes: budget.pCarveBytes) : 0
        let builtScratch = try Self.makeDecodeAndPrefillScratch(
            asub: asub, gsub: gsub, geometry: geometry, maxPrefillChunk: chunkRows,
            pOff: pOff, pCarveBytes: budget.pCarveBytes, useGroupedNAX: budget.useGroupedNAX)
        let dflash: QWDFlashDrafter? = (resolvedCfg.mtpBackend == .dflash2 && startup.mtpDrafts > 0)
            ? try QWDFlashDrafter(geometry: geometry, hal: prelimHal, packPath: resolvedCfg.dflashDraftPackPath,
                                  drafts: startup.mtpDrafts, selectorTopK: resolvedCfg.dflashSelectorTopK,
                                  sessionSlotCount: min(4, resolvedSlotCount)) : nil
        let dspark: QWDSparkDrafter?
        if resolvedCfg.mtpBackend == .dspark && startup.mtpDrafts > 0, let dsCfg = geometry.dsparkConfig {
            let dsURL = resolvedCfg.dflashDraftPackPath.isEmpty
                ? QWModelDescriptor.DSparkConfig.resolvePackURL(near: packPath)
                : URL(fileURLWithPath: resolvedCfg.dflashDraftPackPath)
            guard let dsURL, FileManager.default.fileExists(atPath: dsURL.path) else {
                throw QWEngineDriverError.packMissing(resolvedCfg.dflashDraftPackPath.isEmpty ? "dspark.qwp" : resolvedCfg.dflashDraftPackPath)
            }
            let headURL = QWModelDescriptor.DSparkConfig.resolveDraftHeadURL(near: packPath)
                ?? QWModelDescriptor.DSparkConfig.resolveDraftHeadURL(near: dsURL.path)
            dspark = try QWDSparkDrafter(
                geometry: geometry, hal: prelimHal, config: dsCfg,
                packURL: dsURL, draftHeadURL: headURL,
                maxPrefillChunk: chunkRows, sessionSlotCount: min(4, resolvedSlotCount),
                confidenceScheduleEnabled: resolvedCfg.dsparkConfidenceSchedule
            )
        } else {
            dspark = nil
        }
        admission.totalSlots = resolvedSlotCount
        admission.freeSlots = try Self.buildGDNSlots(
            gsub: gsub, geometry: geometry, gdnLayerCount: gdnL.count,
            mtpDrafts: startup.mtpDrafts, dflashDrafter: dflash, dsparkDrafter: dspark,
            slotCount: resolvedSlotCount)

        let effectiveLSAConfig = (!geometry.usesAnyMLA && geometry.blockSparseGQAConfig == nil)
            ? lsaConfig.resolvingAdaptiveBudget(forMaxContext: budget.effectiveMaxTokens)
            : lsaConfig
        self.lsaConfig = effectiveLSAConfig
        let lsaOp = try QWLSAOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, lsaConfig: effectiveLSAConfig, packPath: packPath,
            pack: startup.pack, safetensors: stContainer, attnLayers: attnL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows)
        let mlaOp = try QWMLAAttnOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, attnLayers: attnL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows,
            lsaConfig: lsaConfig, extraAttnLayers: needsMTPCache ? 1 : 0)
        let mbsaOp = try QWMBSAAttnOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, attnLayers: attnL,
            kvMode: kvMode, maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows)
        let hcaOp = try QWHCAAttnOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, hcaLayers: hcaL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows,
            lsaConfig: lsaConfig, indexDim: hcaSidecar?.indexDim ?? QWLSAConfig.defaultIndexDim)

        let mtpRes = try Self.buildMTPVerificationResources(
            geometry: geometry, hal: prelimHal, asub: asub, gsub: gsub,
            attnDecode: aDec, attnLayers: attnL, gdnLayerCount: gdnL.count,
            mlaOperator: mlaOp, layersW: decoder.layers,
            mtpWeights: mWeights, echoWeights: eWeights, dflashDrafter: dflash,
            mtpDrafts: startup.mtpDrafts, effectiveMaxTokens: budget.effectiveMaxTokens,
            maxT: chunkRows, mtpPartialsOff: mtpPartialsOff, pCarveBytes: budget.pCarveBytes,
            headIsQ4g: resolvedCfg.headIsQ4g, verifyHeadQ4Enabled: resolvedCfg.verifyHeadQ4Enabled,
            useGroupedNAX: budget.useGroupedNAX, scratch: builtScratch.sc,
            ropeApply: ropeApp, effectiveRopeRows: budget.effectiveRopeRows, kvMode: kvMode)

        self.hyperRuntime = nil
        self.inkling2Runtime = nil
        self.standardState = QWDriverStandardState(
            pack: startup.pack,
            embedReader: embHead.embedReader,
            embedScalesReader: embHead.embedScalesReader,
            embedBiasesReader: embHead.embedBiasesReader,
            arena: arenaVal,
            kvCache: caches.kvCache,
            ropeApply: ropeApp,
            attnLayers: attnL,
            hcaLayers: hcaL,
            gdnLayers: gdnL,
            attnDecode: aDec,
            pagedAttnPrefill: pagedAttnPrefillVal,
            pagedAttnPrefillGeneral: pagedAttnPrefillGenVal,
            gdnConv: gdnConvVal,
            gdnDecode: gdnDecodeVal,
            gdnPrefill: gdnPrefillVal,
            head: headVal,
            pipelines: pipes,
            mhcSplit: mhcSplitVal,
            attnPrefill: attnPrefillVal,
            gdnIndexMap: gMap,
            attnIndexMap: aMap,
            hcaIndexMap: hMap,
            layersW: decoder.layers,
            mtpWeights: mWeights,
            echoWeights: eWeights,
            mhcExitHead: mhcExitHeadVal,
            multimaxParams: multimaxParamsVal,
            embedPlane: embHead.embedPlane,
            embedRawOff: embHead.embedRawOff,
            mtpDeviceEmbedding: embHead.mtpDeviceEmbedding,
            mtpBatchedMLP: mtpRes.mtpBatchedMLP,
            mtpBatchedTrunk: mtpRes.mtpBatchedTrunk,
            mtpSharedAttentionPartials: mtpRes.mtpSharedAttentionPartials,
            embedStaging: embHead.embedStaging,
            headStorage: embHead.headStorage,
            finalNorm: finalNormVal,
            mtpDraftHead: embHead.mtpDraftHead,
            mtpScratch: mtpRes.mtpScratch,
            mtpDraftEncoder: mtpRes.mtpDraftEncoder,
            echoDrafter: mtpRes.echoDrafter,
            dflashDrafter: dflash,
            dsparkDrafter: dspark,
            gdnReplayCheckpoint: mtpRes.gdnReplayCheckpoint,
            mtpPromptEmbeddings: mtpRes.mtpPromptEmbeddings,
            verifyWindow: mtpRes.verifyWindow,
            mtpCache: caches.mtpCache,
            sc: builtScratch.sc,
            psc: builtScratch.psc,
            lsaOperator: lsaOp,
            mlaOperator: mlaOp,
            mbsaOperator: mbsaOp,
            hcaOperator: hcaOp,
            hcaLsaBuffer: hcaLsaBufferVal,
            pscBuffer: asub.region.buffer,
            pscByteOffset: asub.region.base + builtScratch.gsOff,
            pscByteLength: asub.usedBytes - builtScratch.gsOff
        )
        self.mtpDraftEncoder?.driver = self

        asub.markSteadyState(); gsub.markSteadyState()
        let hasDynamicMoECache = decoder.layers.contains { $0.moe?.coldStreamer?.isDynamic == true }
        if tuning.expertCache {
            for lw in decoder.layers { _ = lw.moe?.coldStreamer?.attachGPUSlotMap(device: prelimHal.device) }
            if hasDynamicMoECache, spanMapIds == nil {
                spanMapIds = prelimHal.device.makeBuffer(
                    length: max(4 * chunkRows * geometry.expertsPerTok, 4096),
                    options: .storageModeShared)
            }
        }
        if tuning.coldEvents, !Self.decodeProfileEnabled, !QWMoEChain.coldTimingProxy,
           hasDynamicMoECache, coldEvents == nil {
            coldEvents = ColdEventPipeline(device: prelimHal.device)
        }
        prelimHal.adoptPendingResidency()
        prelimHal.makeResident(allResidentBuffers())
        prelimHal.primeResidency()
        // These selectors are only dispatched by QWLSAOperator.
        let activePrewarmDoors = Self.prewarmDoors(prewarmDoors, hasLSAOperator: lsaOp != nil)
        try Self.prewarmPipelines(hal: prelimHal, doors: activePrewarmDoors)
        if let (cb, enc) = prelimHal.begin() {
            enc.useResource(wsub.region.buffer, usage: .read)
            enc.useResource(asub.region.buffer, usage: [.read, .write])
            enc.useResource(gsub.region.buffer, usage: [.read, .write])
            bindResidentResources(enc)
            if let pso = try? prelimHal.pipeline("qwg_cast_f32_to_f16") {
                enc.setComputePipelineState(pso); builtScratch.sc.gs.bind(enc, index: 0); builtScratch.sc.x.bind(enc, index: 1)
                var one: UInt32 = 1; enc.setBytes(&one, length: 4, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            prelimHal.end(cb, enc); cb.waitUntilCompleted()
        }
    }

    private var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get {
            if let hyper = hyperRuntime { return hyper.onCacheEvent }
            if let ink = inkling2Runtime { return ink.onCacheEvent }
            return statsLock.withLock { _onCacheEvent }
        }
        set {
            if let hyper = hyperRuntime {
                hyper.onCacheEvent = newValue
                return
            }
            if let ink = inkling2Runtime {
                ink.onCacheEvent = newValue
                return
            }
            statsLock.withLock { _onCacheEvent = newValue }
        }
    }
    var activePrefillProgress: (@Sendable (Int, Int) -> Void)?
    var activePrefillTotalTokens: Int?
    var activePrefillBasePosition: Int = 0
    private var _lastExperimentalGeneration: QWExperimentalGenerationStats?
    private var _lastGeneration: QWGenerationStats?
    private var _lastRequestedBackend: String?
    private var _lastPromptTokens: Int = 0
    private var _lastStopToken: Int?
    private var _gpuTurnArbitrator: QWGPUTurnArbitrator?
    /// Step-boundary GPU arbitrator assigned on the primary engine so decode/prefill steps
    /// yield cleanly to queued Metal decision sidecars (`ADR-002`).
    public var gpuTurnArbitrator: QWGPUTurnArbitrator? {
        get { statsLock.withLock { _gpuTurnArbitrator } }
        set { statsLock.withLock { _gpuTurnArbitrator = newValue } }
    }
    public internal(set) var lastExperimentalGeneration: QWExperimentalGenerationStats? {
        get { statsLock.withLock { _lastExperimentalGeneration } }
        set { statsLock.withLock { _lastExperimentalGeneration = newValue } }
    }
    public internal(set) var lastGeneration: QWGenerationStats? {
        get { statsLock.withLock { _lastGeneration } }
        set { statsLock.withLock { _lastGeneration = newValue } }
    }
    public internal(set) var lastRequestedBackend: String? {
        get { statsLock.withLock { _lastRequestedBackend } }
        set { statsLock.withLock { _lastRequestedBackend = newValue } }
    }
    public internal(set) var lastPromptTokens: Int {
        get { statsLock.withLock { _lastPromptTokens } }
        set { statsLock.withLock { _lastPromptTokens = newValue } }
    }
    public internal(set) var lastStopToken: Int? {
        get { statsLock.withLock { _lastStopToken } }
        set { statsLock.withLock { _lastStopToken = newValue } }
    }
    public let cpuEncodeLedger = QWEncodeLedger()
    public let prefillTimingLedger = QWPrefillTimingLedger()
    public internal(set) var commandBufferErrors: Int {
        get { admission.commandBufferErrors }
        set { admission.commandBufferErrors = newValue }
    }
    public let decodeGpuLedger = QWDecodeGPULedger()

    func recordCommandBufferError(for cb: (any MTLCommandBuffer)?) {
        admission.recordCommandBufferError(for: cb)
    }

    func recordAsyncCommandBufferFailure(_ cb: any MTLCommandBuffer, session: QWEngineSession?) {
        admission.recordAsyncCommandBufferFailure(cb, session: session)
    }

    func syncCommitFailure(for session: QWEngineSession, cb: (any MTLCommandBuffer)? = nil) -> Bool {
        admission.syncCommitFailure(for: session, cb: cb)
    }

    func addMTPStats(proposals: Int = 0, accepted: Int = 0) {
        speculativeCoordinator.addMTPStats(proposals: proposals, accepted: accepted)
    }

    deinit {
        guard let std = standardState else { return }
        std.coldWriteController.closeAndDrain()
        if std.cacheDiskManager?.config.wipeOnStart == true {
            std.cacheDiskManager?.purgeAll()
        }
    }
}
