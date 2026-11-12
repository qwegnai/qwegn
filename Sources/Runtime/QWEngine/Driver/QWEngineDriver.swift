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
import QWState

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
    let pack: QWQuant.Pack?
    let embedReader: QWSafeTensorsReader?
    let embedScalesReader: QWSafeTensorsReader?
    let embedBiasesReader: QWSafeTensorsReader?
    let arena: QWArena
    let kvCache: QWKVCache
    let ropeApply: QWRopeApply
    public let ropeRows: Int
    let attnLayers: [Int]
    let gdnLayers: [Int]

    let attnDecode: QWAttnDecode
    let pagedAttnPrefill: QWAttnPagedPrefill
    /// The standard path's paged prefill: one scalar kernel over the pool.
    let pagedAttnPrefillGeneral: QWAttnPagedPrefillGeneral
    let gdnConv: QWGDNConv
    let gdnDecode: QWGDNDecode
    let gdnPrefill: QWGDNPrefill
    let head: QWHead
    let psoBf16F16: any MTLComputePipelineState
    let psoGEMVDxH16T4: any MTLComputePipelineState
    let psoGEMVDxH16DualT4: any MTLComputePipelineState
    let psoGEMVDxH16GdnInProjT4: any MTLComputePipelineState
    let psoGEMVSiluPairH16: any MTLComputePipelineState
    let psoFwht32: any MTLComputePipelineState
    let psoRMSNorm: any MTLComputePipelineState
    let psoRMSNormResidual: any MTLComputePipelineState
    let psoRMSNormResidualWide: any MTLComputePipelineState
    let psoHeadF16: any MTLComputePipelineState
    let psoGemmBatchedH16: any MTLComputePipelineState
    let psoGemmBatchedH16GdnInProj: any MTLComputePipelineState
    let psoSiluMulF16: any MTLComputePipelineState
    let psoKVPagedWrite: any MTLComputePipelineState
    /// The general writer: a head narrower than 256 halves (C103).
    let psoKVPagedWriteGeneral: any MTLComputePipelineState
    let psoZeroPadAttn: any MTLComputePipelineState
    let psoMHCEnter: (any MTLComputePipelineState)?
    let psoMHCRouterNorm: (any MTLComputePipelineState)?
    let psoMHCRouterNormRows: (any MTLComputePipelineState)?
    let psoMHCMerge: (any MTLComputePipelineState)?
    let mhcSplit: QWMHCSplitRouter?
    let attnPrefill: QWAttnPrefill
    let gdnIndexMap: [Int]
    let attnIndexMap: [Int]
    let prefillModeLock = NSLock()
    var storedPrefillMode: GDNPrefillMode = .chunked
    public var gdnPrefillMode: GDNPrefillMode {
        get { prefillModeLock.withLock { storedPrefillMode } }
        set { prefillModeLock.withLock { storedPrefillMode = newValue } }
    }
    public var gdnSolveBackend: QWGDNPrefill.SolveBackend {
        get { gdnPrefill.solveBackend }
        set { gdnPrefill.solveBackend = newValue }
    }

    // Weight views are module-internal: an internal property may not expose a private type.
    let layersW: [LayerW]
    /// The kernel each layer runs. The plan resolved the kinds when the
    /// model was loaded, so the hot loop looks a kernel up instead of
    /// switching on the kind.
    let layerKernels: [any QWLayerKernel]

    /// 15 MTP Layer weights (Phase 4, section 5A.4 / 5A.5).
    typealias MTPWeights = QWMTPDraftWeights
    let mtpWeights: MTPWeights?
    // P6 FIX: the 2.5 GB f16 embed table is GONE from GPU memory — it
    // existed only so copyEmbedRow could memcpy a row. A decode step now
    // pulls its ONE 10 KiB row straight from the pack's mmap'd bf16 plane
    let embedPlane: String                       // pack name of the bf16 embed plane
    let embedRawOff: Int                         // cached byte offset in pack.data
    let mtpDeviceEmbedding: QWMTPEmbeddingTable?
    public let mtpAdaptiveDepth: Bool
    public let mtpAdaptiveEntropyThreshold: Double
    public var mtpRelaxedMargin: Double
    public var mtpDraftStrategy: String
    public var mtpQuadraticSeeding: Bool
    var activeRelaxedMarginOverride: Float?
    var activeCollectLogprobs: Bool = false
    var activeConstraint: (constraint: any QWTokenConstraint, sampler: QWSampler)? = nil
    var activeVisualSpans: [QWVisualSpan] = []
    public var mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb

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
    var verificationGpuMsByStage: [String: Double]?
    let mtpBatchedMLP: QWMTPBatchedMLP?
    let mtpBatchedTrunk: QWMTPBatchedTrunk?
    let mtpSharedAttentionPartials: QWMRegion?
    var pendingMTPProfile: (draft: QWMTPStageTiming, verification: QWMTPStageTiming)?
    var pendingMTPAdaptiveEntropy: [Double]?
    var lastMTPRoundProfile: QWMTPRoundProfile?
    let embedStaging: any MTLBuffer              // one [H] f16 host-shared row buffer (init-time)
    let headStorage: QWHeadStorage
    let finalNorm: QWTensorValue
    /// L1 (P4-AD-60): the draft-only q4g head (nil = the bf16 head serves
    /// both draft and verify — the pre-L1 behavior). File-backed via mmap;
    /// the weight arena never carries it.
    let mtpDraftHead: QWMTPTDraftHead?

    public internal(set) var mtpDraftProposals: Int = 0
    public internal(set) var mtpDraftAccepted: Int = 0


    /// Dedicated MTP draft scratch buffers (Phase 4, section 5A.6).
    typealias MTPScratch = QWMTPDraftScratch
    let mtpScratch: MTPScratch?
    let mtpDraftEncoder: QWMTPDraftEncoder?
    let dflashDrafter: QWDFlashDrafter?
    public var dflashDraftPackPath: String? { dflashDrafter?.packPath }
    public var dflashSelectorTopK: Int? { dflashDrafter?.config.selectorTopK }
    let gdnReplayCheckpoint: QWGDNReplayCheckpoint?
    let mtpPromptEmbeddings: QWTensorValue?
    let verifyWindow: QWVerifyWindow?
    var verifiedSession: QWEngineSession?
    var verifiedInputs: [Int] = []
    var preparedGreedyLease: QWGreedyRoundLease?
    var preparedStochasticWitness: QWStochasticRoundWitness?
    let requestGate = QWGenerationRequestGate()
    var prefixStore: QWEnginePrefixStore<QWEngineCachedPrefix>?
    var cacheDiskManager: QWCacheDiskManager?
    public internal(set) var lastPrefixCacheStats: QWPrefixCacheStats?
    let mtpCache: QWKVCache?
    public let mtpDrafts: Int
    /// Performance and memory tunables, environment overrides applied.
    public let tuning: QWEngineTuning
    public let kvMode: QWKVMode
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
                                mode: GDNPrefillMode) -> String {
        drafts > 0 ? "\(backend.rawValue)+\(mode.rawValue)" : mode.rawValue
    }

    func prefixPolicyKey(mode: GDNPrefillMode) -> String {
        Self.prefixPolicyKey(backend: mtpBackend, drafts: mtpDrafts, mode: mode)
    }


    let sc: Scratch
    let psc: PrefillScratch
    public let prefillScratchEviction: QWPrefillScratchEvictionPolicy
    public let prefillChunkSize: Int
    /// Latent Sparse Attention configuration for this driver instance.
    public let lsaConfig: QWLSAConfig
    /// Latent Sparse Attention GPU operator (nil when `lsaConfig.mode == .off` or no full-attention layers).
    public let lsaOperator: QWLSAOperator?
    /// Native NoPE MLA + Key-Pool Indexer operator (nil when `geometry.usesNoPEMLA == false`).
    public let mlaOperator: QWMLAAttnOperator?
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    let pscBuffer: any MTLBuffer
    let pscByteOffset: Int
    let pscByteLength: Int
    let flight = NSLock()          // single-flight admission
    var activeCB: (any MTLCommandBuffer)?
    var activeEnc: (any MTLComputeCommandEncoder)?
    var pendingColdUnloads: [(streamer: ColdExpertStreamer, stagedCount: Int)] = []
    /// Every command buffer a decode step splits into (cold staging ends and
    /// reopens the CB per MoE layer). Non-nil only inside `forwardStep`, so the
    /// step's GPU time covers all of them, not just the last.
    var stepCBs: [any MTLCommandBuffer]?
    var decodeTel = QWDecodeTelemetry()
    var coldEvents: ColdEventPipeline?
    var spanMapIds: (any MTLBuffer)?  // qwg_moe_span_ids output of the current prefill span (host-read)

    func flushPendingColdUnloads() {
        guard !pendingColdUnloads.isEmpty else { return }
        var maxByPtr: [UnsafeMutableRawPointer: (streamer: ColdExpertStreamer, maxCount: Int)] = [:]
        for item in pendingColdUnloads {
            let key = item.streamer.guBasePtr
            if let cur = maxByPtr[key] {
                if item.stagedCount > cur.maxCount {
                    maxByPtr[key] = (item.streamer, item.stagedCount)
                }
            } else {
                maxByPtr[key] = (item.streamer, item.stagedCount)
            }
        }
        for (_, entry) in maxByPtr {
            entry.streamer.unload(stagedCount: entry.maxCount)
        }
        pendingColdUnloads.removeAll(keepingCapacity: true)
    }

    // Pre-carved GDN session slots (two: the engine holds at most two live
    // sequences — one computing, one being admitted; more would share the
    // step scratch anyway under the single-flight law). Each slot carries
    let slotLock = NSLock()
    var freeSlots: [GdnSlot] = []

    public init(packPath: String, maxTokens: Int = 4096, ropeRows: Int = 8192,
                kvMode: QWKVMode = .f16,
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
                osReserveGiB: Double? = nil,
                prefillChunkSize: Int = QWPrefillLimits.chunkRows,
                prewarmDoors: [String] = DenseModule.requiredDoors,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                residentExpertsPerLayer: Int? = nil,
                tuning: QWEngineTuning = .default) throws {
        self.kvMode = kvMode
        self.lsaConfig = lsaConfig
        let cfg = try Self.resolveStartup(packPath: packPath, kvMode: kvMode,
                                          mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                                          prefillScratchEviction: prefillScratchEviction,
                                          mtpAdaptiveDepth: mtpAdaptiveDepth,
                                          mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
                                          mtpRelaxedMargin: mtpRelaxedMargin,
                                          mtpDraftStrategy: mtpDraftStrategy,
                                          mtpQuadraticSeeding: mtpQuadraticSeeding,
                                          dflashDraftPackPath: dflashDraftPackPath,
                                          dflashSelectorTopK: dflashSelectorTopK)
        let tuning = tuning.overlaid(environment: cfg.environment)
        let osReserveGiB = osReserveGiB ?? tuning.osReserveGiB
        let explicitResE = residentExpertsPerLayer ?? tuning.residentExpertsPerLayer
        let rawGeometry = try QWModelGeometry.load(packPath: packPath)
            .withResidentExperts(explicitResE)
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
        let stContainer = isSafetensors ? try QWSafeTensorsContainer(url: root.weights) : nil
        pack = isSafetensors ? nil : try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        let nextNMTPName = "model.language_model.layers.\(rawGeometry.numLayers).enorm.weight"
        let nextNMTPAltName = "model.layers.\(rawGeometry.numLayers).enorm.weight"
        let hasMTPWeights: Bool
        if let st = stContainer {
            hasMTPWeights = st.tensor("mtp.pre_fc_norm_embedding.weight") != nil
                || st.tensor(nextNMTPName) != nil
                || st.tensor(nextNMTPAltName) != nil
        } else if let pk = pack {
            hasMTPWeights = (try? pk.entry("mtp.pre_fc_norm_embedding.weight")) != nil
                || (try? pk.entry(nextNMTPName)) != nil
                || (try? pk.entry(nextNMTPAltName)) != nil
        } else {
            hasMTPWeights = false
        }
        let mtpDrafts = (cfg.mtpBackend == .dflash2 || hasMTPWeights) ? mtpDrafts : 0
        let mtpWeightDrafts = hasMTPWeights ? mtpDrafts : 0
        self.mtpDrafts = mtpDrafts
        // The compressed KV formats are the 27B's layout, and the lane mapping
        // covers a head in groups of 8 halves up to 256 of them. Both are
        // refused here, before any GPU work (C103, C108).
        try Self.validateGeometry(rawGeometry, kvMode: kvMode)
        // The container, and the basis that goes with it (specs/02-qwhal.md).
        let choice = try Self.weightFormatChoice(for: root, quantization: rawGeometry.descriptor.quantization,
                                                 geometry: rawGeometry, mtpDrafts: mtpDrafts, pack: pack)
        self.weightFormat = choice.format
        self.weightFormatDeclineReason = choice.declineReason
        let prelimHal = try QWHal(geometry: rawGeometry, weightBasis: self.weightFormat.basis)
        let geometry = explicitResE != nil || rawGeometry.numExperts == 0 ? rawGeometry
            : Self.planResidentGeometry(
                raw: rawGeometry, cfg: cfg, hal: prelimHal, format: self.weightFormat, tuning: tuning,
                osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
                ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts, kvMode: kvMode)
        self.tuning = tuning
        self.geometry = geometry
        hal = prelimHal
        self.mtpBackend = cfg.mtpBackend
        self.prefillChunkSize = max(32, min(4096, (prefillChunkSize / 32) * 32))
        // The plan freezes what this engine actually runs: its geometry, its
        // KV mode and its prefill chunk. The digest binds all of it.
        self.plan = try QWEngineBootstrap.makePlan(
            geometry: geometry,
            packPath: packPath,
            prefillChunkRows: self.prefillChunkSize,
            kvStorageMode: String(describing: kvMode))
        self.prefillScratchEviction = cfg.prefillScratchEviction
        self.mtpAdaptiveDepth = cfg.mtpAdaptiveDepth
        self.mtpAdaptiveEntropyThreshold = cfg.mtpAdaptiveEntropyThreshold
        self.mtpRelaxedMargin = cfg.mtpRelaxedMargin
        self.mtpDraftStrategy = cfg.mtpDraftStrategy
        self.mtpQuadraticSeeding = cfg.mtpQuadraticSeeding
        mtpStageProfiling = cfg.mtpStageProfiling
        mtpVerificationProfiling = cfg.mtpVerificationProfiling
        mtpVerifyHeadQ4 = cfg.verifyHeadQ4Enabled
        storedPrefillMode = gdnPrefillMode
        let headIsQ4g = cfg.headIsQ4g
        let environment = cfg.environment
        let draftHeadEnabled = cfg.draftHeadEnabled
        let verifyHeadQ4Enabled = cfg.verifyHeadQ4Enabled
        let H = geometry.hidden
        let numV = geometry.numVirtualLayers
        attnLayers = (0..<numV).filter { geometry.isVirtualLayerFullAttention($0) }
        gdnLayers = (0..<numV).filter { !geometry.isVirtualLayerFullAttention($0) }

        psoBf16F16 = try hal.pipeline("qwg_bf16_to_f16")
        psoGEMVDxH16T4 = try hal.pipeline(Self.decodeKernelName(format: self.weightFormat))
        psoGEMVDxH16DualT4 = try hal.pipeline("qwg_gemv_q4g_dx_h16_dual_t4")
        psoGEMVDxH16GdnInProjT4 = try hal.pipeline(Self.gdnInProjDecodeKernelName(format: self.weightFormat))
        psoGEMVSiluPairH16 = try hal.pipeline(Self.siluPairKernelName(format: self.weightFormat))
        psoFwht32 = try hal.pipeline("qwg_fwht32_f16")
        psoRMSNorm = try hal.pipeline("qwg_rmsnorm")
        psoRMSNormResidual = try hal.pipeline("qwg_rmsnorm_residual")
        psoRMSNormResidualWide = try hal.pipeline("qwg_rmsnorm_residual_wide")
        psoHeadF16 = try hal.pipeline("qwg_gemv_f16_v4")
        psoGemmBatchedH16 = try hal.pipeline(Self.gemmKernelName(format: self.weightFormat, bm128: false))
        psoGemmBatchedH16GdnInProj = try hal.pipeline(Self.gdnInProjPrefillKernelName(format: self.weightFormat))
        psoSiluMulF16 = try hal.pipeline("qwg_silu_mul_f16")
        psoKVPagedWrite = try hal.pipeline("qwg_kv_paged_write")
        psoKVPagedWriteGeneral = try hal.pipeline("qwg_kv_paged_write_general")
        psoZeroPadAttn = try hal.pipeline("qwg_zero_pad_attn_f16")
        if geometry.mhcConfig != nil {
            psoMHCEnter = try hal.pipeline("qwg_mhc_enter_f16")
            psoMHCRouterNorm = try hal.pipeline("qwg_mhc_router_norm_f16")
            psoMHCRouterNormRows = ProcessInfo.processInfo.environment["QW_MHC_ROWS_KERNEL"] == "0"
                ? nil : try hal.pipeline("qwg_mhc_router_norm_rows_f16")
            psoMHCMerge = try hal.pipeline("qwg_mhc_merge_f16")
        } else {
            psoMHCEnter = nil
            psoMHCRouterNorm = nil
            psoMHCRouterNormRows = nil
            psoMHCMerge = nil
        }
        mhcSplit = try QWMHCSplitRouter(hal: hal, geometry: geometry)
        attnPrefill = try QWAttnPrefill(hal: hal, backend: .naxX64, geometry: geometry)
        pagedAttnPrefill = try QWAttnPagedPrefill(hal: hal, geometry: geometry)
        pagedAttnPrefillGeneral = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geometry)

        var gMap = [Int](repeating: -1, count: numV)
        for (i, gl) in gdnLayers.enumerated() { gMap[gl] = i }
        gdnIndexMap = gMap

        var aMap = [Int](repeating: -1, count: numV)
        for (i, al) in attnLayers.enumerated() { aMap[al] = i }
        attnIndexMap = aMap

        head = try QWHead(hal: hal, geometry: geometry)
        gdnPrefill = try QWGDNPrefill(hal: hal, scanBackend: .hybrid, solveBackend: .scalar)

        // Region sizes (C03): weights = planes + converted matrices; act =
        // one step's scratch; kv = ctx budget (both kinds, all 16 layers) +
        // per-(seq,layer,kind) table slots; gdn = 2 session slots × THE
        let useGroupedNAX = hal.caps.isMetal4TensorCapable
        let bytesPerToken = Self.prefillBytesPerToken(geometry: geometry, useGroupedNAX: useGroupedNAX)
        let pscBytes = self.prefillChunkSize * bytesPerToken
        let decodeActBytes = Self.decodeAndMTPScratchBytes(
            geometry: geometry, prefillChunkRows: self.prefillChunkSize,
            mtpDrafts: mtpDrafts, maxTokens: maxTokens, useGroupedNAX: useGroupedNAX)
        let actBytes = pscBytes + decodeActBytes
        let gdnLayerCount = gdnLayers.count
        let recB = QWSlotGeometry.recElems(geometry: geometry) * 4
        let ringB = QWSlotGeometry.ringElems(geometry: geometry) * 2
        let slotBytes = gdnLayerCount * (recB + ringB)
        let gdnBytes = 2 * slotBytes + Self.partialsScratchBytes(geometry: geometry, mtpDrafts: mtpDrafts, maxTokens: maxTokens)
            + (mtpDrafts > 0 ? QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: gdnLayerCount, rows: mtpDrafts + 1) : 0)
            + (mtpDrafts > 0 ? 2 * geometry.hidden * 2 : 0)
            + (1 << 20)
        let activeAttnLayers = attnLayers.count + (mtpDrafts > 0 ? 1 : 0)
        let mlaPerTokBytes = geometry.usesNoPEMLA ? activeAttnLayers * ((geometry.indexerHeadDim * 2) / 4) : 0
        let mlaFixedBytes = Self.mlaFixedScratchBytes(
            geometry: geometry, prefillChunkRows: self.prefillChunkSize, activeAttnLayers: activeAttnLayers)
        let perTokBytes = 2 * activeAttnLayers * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens) + mlaPerTokBytes
        let maxMTP = cfg.mtpBackend == .dflash2 ? 15 : 5
        let prelimPlan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: max(ropeRows, maxTokens),
                                                    mtpDrafts: mtpWeightDrafts, headQ4g: headIsQ4g, maxMTPDrafts: maxMTP,
                                                    format: self.weightFormat)
        let arenaCap = max(1 << 30, hal.caps.maxWorkingSetBytes - (64 << 20))
        let isMoE = geometry.numExperts > 0
        let isColdStreamingMoE = isMoE && geometry.effectiveResidentExperts < geometry.numExperts
        let effectiveReserveGiB = isMoE
            ? Self.effectiveOSReserveGiB(policy: tuning.memoryPolicy, explicitReserveGiB: osReserveGiB)
            : osReserveGiB
        let availBytes = isMoE
            ? Self.residencyBudgetBytes(policy: tuning.memoryPolicy, osReserveGiB: osReserveGiB, arenaCap: arenaCap)
            : min(QWCapacityBudget.availableBytes(osReserveGiB: osReserveGiB), arenaCap)
        let requestedMaxTokens = (isColdStreamingMoE && maxTokens == geometry.maxPositionEmbeddings)
            ? max(16_384, min(maxTokens, 32_768))
            : maxTokens
        let prelimCap = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings, requested: requestedMaxTokens,
            perTokenBytes: perTokBytes, fixedBytes: prelimPlan.bytes + actBytes + gdnBytes + mlaFixedBytes,
            availableBytes: availBytes, yarnAllowed: true)
        let effectiveMaxTokens = (!prelimCap.fits && requestedMaxTokens == geometry.maxPositionEmbeddings && prelimCap.maxFittingContext >= QWCapacityPlan.kvStepTokens)
            ? (prelimCap.maxFittingContext / QWCapacityPlan.kvStepTokens) * QWCapacityPlan.kvStepTokens
            : requestedMaxTokens
        let effectiveRopeRows = max(ropeRows, effectiveMaxTokens)
        let weightPlan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: effectiveRopeRows, mtpDrafts: mtpWeightDrafts,
                                                    headQ4g: headIsQ4g, maxMTPDrafts: maxMTP,
                                                    format: self.weightFormat)
        let weightsBytes = weightPlan.bytes
        let trunkKVBytes = Self.kvBudgetBytes(maxTokens: effectiveMaxTokens, mtpDrafts: 0, kvMode: kvMode, geometry: geometry)
        let mtpKVBytes = Self.kvBudgetBytes(maxTokens: effectiveMaxTokens, mtpDrafts: mtpDrafts, kvMode: kvMode, geometry: geometry) - trunkKVBytes
        let kvBytes = trunkKVBytes + mtpKVBytes
        let fixedBytes = weightsBytes + actBytes + gdnBytes + mlaFixedBytes
        let capPlan = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings,
            requested: effectiveMaxTokens,
            perTokenBytes: perTokBytes,
            fixedBytes: fixedBytes,
            availableBytes: availBytes,
            yarnAllowed: true
        )
        self.capacityPlan = capPlan
        let formatSuffix = " weights_format=\(self.weightFormat.name) weights_bpw="
            + String(format: "%.2f", Double(self.weightFormat.bits) + 32.0 / Double(self.weightFormat.groupSize))
            + (self.weightFormatDeclineReason.map { " weights_declined=\($0)" } ?? "")
        self.capacityLogMessage = try QWEngineBootstrap.admit(
            weightsBytes: weightsBytes,
            fixedScratchBytes: actBytes + gdnBytes + mlaFixedBytes,
            kvMode: String(describing: kvMode),
            perTokenBytes: perTokBytes,
            maxTokens: effectiveMaxTokens,
            totalPlannedBytes: fixedBytes + kvBytes,
            availableBytes: availBytes,
            osReserveGiB: effectiveReserveGiB,
            logSuffix: formatSuffix
        )
        arena = try QWArena(hal: hal, weights: weightsBytes, activations: actBytes,
                            kv: kvBytes, gdnStates: gdnBytes)

        let wsub = arena.weightsRegion()
        let hl = hal
        var L: WeightLoader
        if let pk = pack {
            L = WeightLoader(pk: pk, hl: hal, psoCvt: psoBf16F16,
                             wsub: wsub, plan: weightPlan, geometry: geometry,
                             format: self.weightFormat)
        } else if let st = stContainer {
            let psoQuant = try hal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
            L = WeightLoader(st: st, hl: hal, psoCvt: psoBf16F16, psoQuantBF16: psoQuant,
                             wsub: wsub, plan: weightPlan, geometry: geometry,
                             format: self.weightFormat)
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }

        let embedName = L.resolveName(Self.P + "embed_tokens.weight")
        embedPlane = embedName
        let embedBytes = geometry.vocab * geometry.hidden * 2
        let devEmbedEnv = environment["QW_MTP_DEVICE_EMBEDDING"] ?? environment["QW_MTP_DEVICE_EMBED"]
        let allowDeviceEmbed = devEmbedEnv != "0" && mtpDrafts > 0 && (
            devEmbedEnv == "1"
            || (mtpDrafts > 1
                && geometry.effectiveResidentExperts == geometry.numExperts
                && fixedBytes + kvBytes + embedBytes <= availBytes)
        )
        if let pk = pack {
            guard let rawOff = try pk.entry(embedName).rawOff else {
                throw QWEngineDriverError.weightMissing(embedName, "not a bf16 raw plane")
            }
            embedRawOff = rawOff; embedReader = nil; embedScalesReader = nil; embedBiasesReader = nil
            mtpDeviceEmbedding = allowDeviceEmbed
                ? try QWMTPEmbeddingTable(hal: hl, path: pk.url.path, offset: rawOff,
                                         vocabulary: geometry.vocab, hidden: geometry.hidden) : nil
        } else if let st = stContainer {
            embedRawOff = 0
            let eReader = try st.reader(of: embedName)
            embedReader = eReader
            if eReader.tensorInfo.dtype == .u32, embedName.hasSuffix(".weight") {
                let stem = String(embedName.dropLast(".weight".count))
                embedScalesReader = try st.reader(of: stem + ".scales")
                embedBiasesReader = try st.reader(of: stem + ".biases")
            } else {
                embedScalesReader = nil; embedBiasesReader = nil
            }
            if allowDeviceEmbed, eReader.tensorInfo.dtype != .u32,
               let sURL = st.shardURL(of: embedName), let rng = st.byteRange(of: embedName) {
                mtpDeviceEmbedding = try QWMTPEmbeddingTable(hal: hl, path: sURL.path, offset: rng.lowerBound,
                                                             vocabulary: geometry.vocab, hidden: geometry.hidden)
            } else {
                mtpDeviceEmbedding = nil
            }
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }
        embedStaging = hl.device.makeBuffer(length: geometry.hidden * 2, options: .storageModeShared)!
        if draftHeadEnabled, let sidecarPath = environment["QW_MTP_DRAFT_HEAD_SIDECAR"] {
            let sidecar = try QWQuant.Pack(url: URL(fileURLWithPath: sidecarPath))
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: sidecar)
        } else if draftHeadEnabled, headIsQ4g, let pk = pack {
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: pk)
        } else {
            if draftHeadEnabled && pack != nil {
                throw QWEngineDriverError.state("QW_MTP_DRAFT_HEAD=1 requires QW_MTP_DRAFT_HEAD_SIDECAR")
            }
            mtpDraftHead = nil
        }
        headStorage = try Self.loadHead(loader: &L, headIsQ4g: headIsQ4g, geometry: geometry)
        if case .q4g(let hw) = headStorage, !hw.view.isIncumbentContainer {
            for name in [QWGemmAQ.kernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ),
                         QWGemmAQ.decodeKernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ),
                         QWGemmAQ.headArgmaxKernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ)].compactMap({ $0 }) {
                _ = try hal.pipeline(name)
            }
        }

        // ── per-layer weights ───────────────────────────────────────────────
        let decoder = try L.loadDecoderWeights(geometry: geometry)
        layersW = decoder.layers
        // Trunk cold-expert streamers run as per-layer LRU caches over the resident bank (expertCache off
        // keeps the static hot set). The MTP draft layer keeps its static map: its MoE encodes without a driver.
        if tuning.expertCache {
            for lw in decoder.layers { lw.moe?.coldStreamer?.enableDynamicResidency() }
        }
        layerKernels = try QWLayerKernelRegistry.resolve(layers: decoder.layers, plan: plan)
        finalNorm = decoder.finalNorm

        // rope table BEFORE markSteadyState (QWRopeApply allocates its
        // device table row-block from the weights sub-allocator).
        // QW_ROPE_YARN_FACTOR (4b): frequency-window extension, HF yarn
        let yarnFactor = try Self.resolveYaRNFactor(environment["QW_ROPE_YARN_FACTOR"] ?? ropeYarnFactor.map { String($0) })
        let yarnConfig: QWYaRNConfig? = yarnFactor.map { QWYaRNConfig(factor: $0) }
            ?? (effectiveMaxTokens > geometry.maxPositionEmbeddings
                ? QWYaRNConfig.config1M(targetContext: effectiveMaxTokens,
                                        originalMaxPosition: geometry.maxPositionEmbeddings)
                : nil)
        let tables = QWRopeTables(rows: effectiveRopeRows, yarn: yarnConfig, geometry: geometry)
        self.ropeRows = effectiveRopeRows
        try weightPlan.check(QWTensorShape([effectiveRopeRows, geometry.rotaryDims], .f32), at: L.weightRequest)
        ropeApply = try QWRopeApply(tables: tables, hal: hal, weights: wsub, arena: arena, geometry: geometry)
        L.weightRequest += 1
        mtpWeights = try Self.loadMTPWeights(&L, drafts: mtpWeightDrafts, geometry: geometry)
        try weightPlan.finish(requests: L.weightRequest, usedBytes: wsub.usedBytes)
        wsub.markSteadyState()
        // Release physical memory for any file-backed mapped pack pages
        pack?.adviseDontNeed()
        stContainer?.adviseDontNeed()

        attnDecode = try QWAttnDecode(hal: hal, sharedPrefixRows: mtpDrafts > 0 ? min(8, mtpDrafts + 1) : nil,
                                     kvMode: self.kvMode, geometry: geometry)
        gdnConv = try QWGDNConv(hal: hal, geometry: geometry)
        gdnDecode = try QWGDNDecode(hal: hal)
        let caches = try Self.makeKVCaches(
            hal: hal, arena: arena, attnLayers: attnLayers, kvMode: kvMode,
            geometry: geometry, mtpDrafts: mtpDrafts,
            trunkKVBytes: trunkKVBytes, mtpKVBytes: mtpKVBytes)
        kvCache = caches.kvCache
        mtpCache = caches.mtpCache

        let asub = arena.activationsRegion()
        let gsub = arena.gdnRegion()
        let pCarveBytes = Self.partialsCarveBytes(geometry: geometry, maxTokens: effectiveMaxTokens)
        let pOff = try gsub.allocate(bytes: pCarveBytes)
        let mtpPartialsOff: Int = mtpDrafts > 0
            ? try gsub.allocate(bytes: pCarveBytes)
            : 0
        let builtScratch = try Self.makeDecodeAndPrefillScratch(
            asub: asub, gsub: gsub, geometry: geometry,
            maxPrefillChunk: self.prefillChunkSize,
            pOff: pOff, pCarveBytes: pCarveBytes,
            useGroupedNAX: useGroupedNAX)
        sc = builtScratch.sc
        psc = builtScratch.psc
        let gsOff = builtScratch.gsOff
        let maxT = self.prefillChunkSize
        self.pscBuffer = asub.region.buffer; self.pscByteOffset = asub.region.base + gsOff; self.pscByteLength = asub.usedBytes - gsOff
        if cfg.mtpBackend == .dflash2 && mtpDrafts > 0 {
            dflashDrafter = try QWDFlashDrafter(geometry: geometry, hal: hal,
                packPath: cfg.dflashDraftPackPath, drafts: mtpDrafts, selectorTopK: cfg.dflashSelectorTopK)
        } else {
            dflashDrafter = nil
        }

        // Pre-carve 2 GDN session slots: ONE contiguous rec slab + ONE ring
        // slab per slot, sliced into per-GDN-layer views (exact strides;
        // both are page-multiples, so every layer's state is a page-aligned
        for slotIdx in 0..<2 {
            let recSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.recElems(geometry: geometry))], .f32))
            let ringSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.ringElems(geometry: geometry))], .f16))
            let recViews = Self.slotViews(recSlab, elem: QWSlotGeometry.recElems(geometry: geometry), dt: .f32, layers: gdnLayerCount, geometry: geometry)
            let ringViews = Self.slotViews(ringSlab, elem: QWSlotGeometry.ringElems(geometry: geometry), dt: .f16, layers: gdnLayerCount, geometry: geometry)
            // ANTI-ALIASING LAW (the C23 clobber class): door guards bind
            // shapes, never identity — two legal views of ONE buffer pass
            // every door check. The integrator asserts byte-disjointness
            func disjoint(_ views: [QWTensorValue]) -> Bool {
                guard views.count > 1 else { return true }
                var spans = views.map { ($0.region.base + $0.offset, $0.bytes) }
                spans.sort { $0.0 < $1.0 }
                for i in 1..<spans.count where spans[i].0 < spans[i - 1].0 + spans[i - 1].1 { return false }
                return true
            }
            precondition(recViews.count == gdnLayerCount && disjoint(recViews),
                         "QWEngineDriver: GDN rec state views ALIAS (per-layer state law)")
            precondition(ringViews.count == gdnLayerCount && disjoint(ringViews),
                         "QWEngineDriver: GDN ring views ALIAS (per-layer state law)")
            freeSlots.append(GdnSlot(recSlab: recSlab, ringSlab: ringSlab,
                                     rec: recViews, ring: ringViews,
                                     trunkHidden: mtpDrafts > 0 ? try gsub.allocate(QWTensorShape([H], .f16)) : nil,
                                     dflashSlot: dflashDrafter?.sessionSlots[slotIdx]))
        }

        self.lsaOperator = try QWLSAOperator.makeIfEnabled(
            hal: hal, geometry: geometry, lsaConfig: lsaConfig,
            packPath: packPath, pack: pack, safetensors: stContainer, attnLayers: attnLayers,
            maxContext: effectiveMaxTokens, maxPrefillChunk: self.prefillChunkSize)
        self.mlaOperator = try QWMLAAttnOperator.makeIfEnabled(
            hal: hal, geometry: geometry, attnLayers: attnLayers,
            maxContext: effectiveMaxTokens, maxPrefillChunk: self.prefillChunkSize,
            lsaConfig: lsaConfig, extraAttnLayers: mtpDrafts > 0 ? 1 : 0)

        if mtpDrafts > 0 && (mtpWeights != nil || dflashDrafter != nil) {
            gdnReplayCheckpoint = try QWGDNReplayCheckpoint(geometry: geometry, allocator: gsub, layers: gdnLayerCount,
                                                            rows: mtpDrafts + 1, fusedCapture: true)
            _ = try hal.pipeline("qwg_gdn_capture_replay"); _ = try hal.pipeline("qwg_gdn_replay_state"); _ = try hal.pipeline("qwg_verify_copy")
            verifyWindow = try QWVerifyWindow(geometry: geometry, allocator: asub, rows: mtpDrafts + 1,
                                             layerMajor: true, gpuTargets: true,
                                             q4Head: verifyHeadQ4Enabled || headIsQ4g,
                                             useGroupedNAX: useGroupedNAX)
            mtpBatchedMLP = try QWMTPBatchedMLP(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
            mtpBatchedTrunk = try QWMTPBatchedTrunk(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
            let maxDecodeSplits = effectiveMaxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
            let bytes = QWAttnDecode.decodePartialsBytesBatched(T: min(8, mtpDrafts + 1), geometry: geometry, maxSplits: maxDecodeSplits)
            let offset = try asub.allocate(bytes: bytes)
            mtpSharedAttentionPartials = QWMRegion(buffer: asub.region.buffer, base: asub.region.base + offset, length: bytes)
            _ = try hal.pipeline("qwg_mtp_argmax_target_checked")
            let partials = QWMRegion(buffer: gsub.region.buffer, base: gsub.region.base + mtpPartialsOff, length: pCarveBytes)
            let scratch = try MTPScratch(geometry: geometry, allocator: asub, partials: partials)
            mtpScratch = scratch
            mtpPromptEmbeddings = try asub.allocate(QWTensorShape([maxT * H], .f16))
            if let mtpWeights {
                mtpDraftEncoder = try QWMTPDraftEncoder(geometry: geometry, hal: hal, weights: mtpWeights, scratch: scratch,
                                                       rope: ropeApply, ropeRows: effectiveRopeRows, kvMode: kvMode,
                                                       mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count)
            } else {
                mtpDraftEncoder = nil
            }
            for m in ["m2", "m3", "m5", "m6", "m7", "m8"] { _ = try hal.pipeline("qwg_head_f16_\(m)") }
            for m in 2...8 {
                _ = try hal.pipeline("qwg_head_q4g_m\(m)")
                if m >= 4 { _ = try hal.pipeline("qwg_head_q4g_rp_m\(m)") }
            }
        } else {
            gdnReplayCheckpoint = nil; verifyWindow = nil; mtpBatchedMLP = nil; mtpBatchedTrunk = nil
            mtpSharedAttentionPartials = nil; mtpScratch = nil; mtpPromptEmbeddings = nil; mtpDraftEncoder = nil
        }

        asub.markSteadyState()
        gsub.markSteadyState()

        hal.adoptPendingResidency()
        var resident: [any MTLBuffer] = arena.allBuffers
        if let lsa = lsaOperator { resident.append(lsa.region.buffer) }
        if let mla = mlaOperator { resident.append(mla.region.buffer) }
        switch headStorage {
        case .f16(let t): resident.append(t.region.buffer)
        case .q4g(let w): resident.append(w.view.blocks.region.buffer)
        }
        if let emb = mtpDeviceEmbedding { resident += [emb.buffer, emb.tokens] }
        if let dh = mtpDraftHead { resident += dh.mappedBuffers }
        for lw in layersW {
            resident += [lw.gu.view.blocks.region.buffer, lw.down.blocks.region.buffer]
            if let g = lw.gdn { resident += [g.inProj.blocks.region.buffer, g.out.blocks.region.buffer] }
            if let a = lw.attn { resident += [a.qkvProj.blocks.region.buffer, a.o.blocks.region.buffer] }
        }
        if let mw = mtpWeights {
            resident += [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                         mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer]
        }
        hal.makeResident(resident)

        try Self.prewarmPipelines(hal: hal, doors: prewarmDoors)
        if let (cb, enc) = hal.begin() {
            enc.useResource(wsub.region.buffer, usage: .read)
            enc.useResource(asub.region.buffer, usage: [.read, .write])
            enc.useResource(gsub.region.buffer, usage: [.read, .write])
            if let lsa = lsaOperator { enc.useResource(lsa.region.buffer, usage: [.read, .write]) }
            if let mla = mlaOperator { enc.useResource(mla.region.buffer, usage: [.read, .write]) }
            for buf in arena.allBuffers { enc.useResource(buf, usage: [.read, .write]) }
            switch headStorage {
            case .f16(let t): enc.useResource(t.region.buffer, usage: .read)
            case .q4g(let w): enc.useResource(w.view.blocks.region.buffer, usage: .read)
            }
            if let emb = mtpDeviceEmbedding {
                enc.useResource(emb.buffer, usage: .read); enc.useResource(emb.tokens, usage: [.read, .write])
            }
            if let dh = mtpDraftHead { for buf in dh.mappedBuffers { enc.useResource(buf, usage: .read) } }
        for lw in layersW {
            enc.useResource(lw.gu.view.blocks.region.buffer, usage: .read); enc.useResource(lw.down.blocks.region.buffer, usage: .read)
            if let g = lw.gdn { enc.useResource(g.inProj.blocks.region.buffer, usage: .read); enc.useResource(g.out.blocks.region.buffer, usage: .read) }
            if let a = lw.attn { enc.useResource(a.qkvProj.blocks.region.buffer, usage: .read); enc.useResource(a.o.blocks.region.buffer, usage: .read) }
        }
            if let mw = mtpWeights {
                for b in [mw.fc.blocks.region.buffer, mw.qkvProj.blocks.region.buffer, mw.o.blocks.region.buffer,
                          mw.gu.view.blocks.region.buffer, mw.down.blocks.region.buffer] { enc.useResource(b, usage: .read) }
            }
            if let pso = try? hal.pipeline("qwg_cast_f32_to_f16") {
                enc.setComputePipelineState(pso); sc.gs.bind(enc, index: 0); sc.x.bind(enc, index: 1)
                var one: UInt32 = 1; enc.setBytes(&one, length: 4, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            hal.end(cb, enc); cb.waitUntilCompleted()
        }
    }

    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    public internal(set) var lastExperimentalGeneration: QWExperimentalGenerationStats?
    public internal(set) var lastGeneration: QWGenerationStats?
    public internal(set) var lastRequestedBackend: String?
    public internal(set) var lastPromptTokens: Int = 0
    public internal(set) var lastStopToken: Int?
    public let cpuEncodeLedger = QWEncodeLedger()
    public let prefillTimingLedger = QWPrefillTimingLedger()
    public internal(set) var commandBufferErrors: Int = 0
    public let decodeGpuLedger = QWDecodeGPULedger()

    deinit {
        if cacheDiskManager?.config.wipeOnStart == true {
            cacheDiskManager?.purgeAll()
        }
    }
}
