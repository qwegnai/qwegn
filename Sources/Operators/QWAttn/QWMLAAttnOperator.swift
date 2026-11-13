// QWMLAAttnOperator.swift — Matrix-Absorbed Multi-head Latent Attention GPU operator
// supporting both NoPE MLA + Key-Pool(4) (GLM-5.x) and Decoupled-RoPE MLA-DSA (ERNIE-5.5).
// Manages per-MLA-layer indexer rings/pools, optional decoupled k_pe cache, e-means CSR,
// and shared latent/query/attention scratch in a single preallocated Metal region.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import QWGEMM
import QWKV

public final class QWMLAAttnOperator: @unchecked Sendable {
    public struct LayerWeights: Sendable {
        public let qANormW: QWTensorValue
        public let kvANormW: QWTensorValue
        public let kNormW: QWTensorValue
        public let kNormB: QWTensorValue
        public let wGateWeights: QWTensorValue
        public let ape: QWTensorValue?
        public let qBProj: QWWeightView
        public let kvBProj: QWWeightView
        public let vhaMix: QWTensorValue?
        public let attnSink: QWTensorValue?
        public let hasOGate: Bool

        public init(qANormW: QWTensorValue,
                    kvANormW: QWTensorValue,
                    kNormW: QWTensorValue,
                    kNormB: QWTensorValue,
                    wGateWeights: QWTensorValue,
                    ape: QWTensorValue?,
                    qBProj: QWWeightView,
                    kvBProj: QWWeightView,
                    vhaMix: QWTensorValue? = nil,
                    attnSink: QWTensorValue? = nil,
                    hasOGate: Bool = false) {
            self.qANormW = qANormW
            self.kvANormW = kvANormW
            self.kNormW = kNormW
            self.kNormB = kNormB
            self.wGateWeights = wGateWeights
            self.ape = ape
            self.qBProj = qBProj
            self.kvBProj = kvBProj
            self.vhaMix = vhaMix
            self.attnSink = attnSink
            self.hasOGate = hasOGate
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let kPending: QWTensorValue
        public let gatePending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let kPeCache: QWTensorValue?
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
    }

    public static let minSelectBatchRows = 32
    public static let emeansProbeClusters = 320
    public static let emeansCandCap = 10240
    public static let dsaEmeansProbeClusters = 480
    public static let dsaEmeansCandCap = 16384
    public static let selectScratchBudgetBytes = 16 << 20
    public let selectBatchRows: Int

    public let geometry: QWModelGeometry
    public let isDecoupledDSA: Bool
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let qHeads: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let nopeDim: Int
    public let ropeDim: Int
    public let vHeadDim: Int
    public let idxHeads: Int
    public let idxDim: Int
    public let poolShift: Int
    public let windowTokens: Int
    public let useFastHadamard: Bool
    public let ropeMscale: Float
    public let topKPools: Int
    public let listStride: Int
    public let ringCap: Int
    public let maxBlocks: Int
    public let latentStride: Int
    public let qChannels: Int
    public let qRopeOffset: Int
    public let oGateOffset: Int
    public let qFullStride: Int

    public let region: QWMRegion
    public let layers: [LayerState]
    public weak var stateOwner: AnyObject? {
        didSet {
            if stateOwner !== oldValue { resetCoarseWatermarks() }
        }
    }

    public let mlaLatentScratch: QWTensorValue
    public let qResidScratch: QWTensorValue
    public let mlaQFullScratch: QWTensorValue
    public let idxWeightsScratch: QWTensorValue
    public let qAbsScratch: QWTensorValue
    public let uLatScratch: QWTensorValue
    public let yVhaScratch: QWTensorValue
    public let invFreqBuf: QWTensorValue
    public let zeroSinkBuf: QWTensorValue
    public let scoreScratch: QWTensorValue
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue

    internal let hal: QWHal
    internal let psoLatentPrep: any MTLComputePipelineState
    internal let psoDSALatentPrep: any MTLComputePipelineState
    internal let psoPoolKeys: any MTLComputePipelineState
    internal let psoGateWeightsProj: any MTLComputePipelineState
    internal let psoQRopeIdxPrep: any MTLComputePipelineState
    internal let psoIndexerScoreSelect: any MTLComputePipelineState
    internal let psoAbsorbQ: any MTLComputePipelineState
    internal let psoAbsorbQSub2: any MTLComputePipelineState
    internal let psoAbsorbQMMA: any MTLComputePipelineState
    internal let psoAbsorbQMMASub2: any MTLComputePipelineState
    internal let psoSparseAttn: any MTLComputePipelineState
    internal let psoSparseAttnMMA: any MTLComputePipelineState
    internal let psoDSASparseAttn: any MTLComputePipelineState
    internal let psoDSASparseAttnMMA: any MTLComputePipelineState
    internal let psoIndexerSplit: (score: any MTLComputePipelineState, select: any MTLComputePipelineState)?
    internal let psoSparseAttnNAX: (any MTLComputePipelineState)?
    internal let psoDSASparseAttnNAX: (any MTLComputePipelineState)?
    internal let psoSparseAttnSplit: (attend: any MTLComputePipelineState, combine: any MTLComputePipelineState)?
    internal let psoDSASparseAttnSplit: (attend: any MTLComputePipelineState, combine: any MTLComputePipelineState)?
    public static let splitMaxRows = 15
    public let attnSplits: Int
    internal let partOScratch: QWTensorValue
    internal let partMLScratch: QWTensorValue
    public var sparseAttnMMAEnabled = ProcessInfo.processInfo.environment["QW_MLA_ATTN_MMA"] != "0"
    public var emeansCoarseEnabled: Bool
    internal let psoProjUV: any MTLComputePipelineState
    internal let psoProjUVSub2: any MTLComputePipelineState
    internal let psoProjUVMMA: any MTLComputePipelineState
    internal let psoProjUVMMASub2: any MTLComputePipelineState
    internal let psoVHAGateRot: any MTLComputePipelineState
    internal let coarsePipelines: QWCoarsePoolIndex.Pipelines?
    internal let psoEmeansCoarse: (scoreCoarse: any MTLComputePipelineState, scoreCands: any MTLComputePipelineState, flatFallback: any MTLComputePipelineState)?
    internal let coarseScratch: QWCoarsePoolIndex.Scratch?
    internal var clusteredMacros: [Int]

    public func resetCoarseWatermarks() {
        for i in clusteredMacros.indices { clusteredMacros[i] = 0 }
    }

    public func truncatePosition(_ validPosition: Int) {
        let maxMacro = max(0, validPosition) / QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        for i in clusteredMacros.indices where clusteredMacros[i] > maxMacro {
            clusteredMacros[i] = maxMacro
        }
    }

    internal struct LatentPrepParams {
        var T: UInt32
        var pos0: UInt32
        var hidden: UInt32
        var qLoraRank: UInt32
        var kvLoraRank: UInt32
        var idxDim: UInt32
        var idxHeads: UInt32
        var ringMask: UInt32
        var rmsEps: Float
        var kNormEps: Float
        var kvMode: UInt32
    }

    internal struct IndexerParams {
        var T: UInt32
        var pos0: UInt32
        var qChannels: UInt32
        var idxHeads: UInt32
        var idxDim: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var scoreStride: UInt32
    }

    internal struct AbsorbParams {
        var T: UInt32
        var qHeads: UInt32
        var nopeDim: UInt32
        var vHeadDim: UInt32
        var kvLoraRank: UInt32
        var qFullStride: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    internal struct SparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var kvMode: UInt32
    }

    internal static func kvModeCode(_ mode: QWKVMode) -> UInt32 {
        switch mode {
        case .f16: return 0
        case .q8: return 1
        case .q4: return 2
        }
    }

    public static func makeIfEnabled(hal: QWHal,
                                     geometry: QWModelGeometry,
                                     attnLayers: [Int],
                                     maxContext: Int,
                                     maxPrefillChunk: Int = 512,
                                     lsaConfig: QWLSAConfig = QWLSAConfig(),
                                     extraAttnLayers: Int = 0) throws -> QWMLAAttnOperator? {
        let hasMLA = geometry.usesAnyMLA && !attnLayers.isEmpty
        guard hasMLA else { return nil }
        return try QWMLAAttnOperator(
            hal: hal,
            geometry: geometry,
            numAttnLayers: attnLayers.count + max(0, extraAttnLayers),
            maxContext: maxContext,
            maxPrefillChunk: maxPrefillChunk,
            lsaConfig: lsaConfig
        )
    }

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                numAttnLayers: Int,
                maxContext: Int,
                maxPrefillChunk: Int = 512,
                lsaConfig: QWLSAConfig = QWLSAConfig()) throws {
        self.hal = hal
        self.geometry = geometry
        let dsa = geometry.usesDecoupledRopeMLA
        self.isDecoupledDSA = dsa
        self.maxContext = max(512, maxContext)
        self.maxPrefillChunk = max(32, maxPrefillChunk)

        if let hmla = geometry.hybridMLAConfig, dsa {
            self.qHeads = hmla.qHeads
            self.qLoraRank = hmla.qLoraRank
            self.kvLoraRank = hmla.kvLoraRank
            self.nopeDim = hmla.qkNopeHeadDim
            self.ropeDim = hmla.qkRopeHeadDim
            self.vHeadDim = hmla.vHeadDim
            self.poolShift = geometry.indexerPoolShift
            self.windowTokens = max(0, hmla.slidingWindow)
            self.useFastHadamard = hmla.indexerUseFastHadamard
        } else {
            self.qHeads = geometry.qHeads
            self.qLoraRank = geometry.qLoraRank
            self.kvLoraRank = geometry.kvLoraRank
            self.nopeDim = geometry.qkNopeHeadDim > 0 ? geometry.qkNopeHeadDim : geometry.headDim
            self.ropeDim = 0
            self.vHeadDim = geometry.vHeadDim > 0 ? geometry.vHeadDim : geometry.headDim
            self.poolShift = 2
            self.windowTokens = 0
            self.useFastHadamard = false
        }

        self.idxHeads = max(1, geometry.indexerHeads)
        self.idxDim = max(32, geometry.indexerHeadDim)
        let modelBudget = geometry.indexerBudget > 0 ? geometry.indexerBudget : 2048
        let requestedBudget = (lsaConfig.budget != QWLSAConfig.defaultBudget)
            ? lsaConfig.alignedBudget
            : modelBudget
        if dsa {
            let budgetTok = min(2048, max(1, requestedBudget))
            self.topKPools = budgetTok
            self.listStride = max(512, ((budgetTok + self.windowTokens + 127) & ~127))
            self.latentStride = self.qLoraRank + self.kvLoraRank + self.ropeDim + self.idxDim
            self.qRopeOffset = self.qHeads * self.nopeDim
            self.oGateOffset = self.qHeads * (self.nopeDim + self.ropeDim)
            let hasGate = geometry.attnOutputGate
            self.qChannels = self.qHeads * (self.nopeDim + self.ropeDim + (hasGate ? self.vHeadDim : 0))
            self.qFullStride = self.qChannels + self.idxHeads * self.idxDim
            self.maxBlocks = max(128, self.maxContext >> self.poolShift)
        } else {
            let budgetTok = min(4096, max(128, requestedBudget))
            self.topKPools = max(1, min(1024, budgetTok / 4))
            self.listStride = max(512, self.topKPools)
            self.latentStride = self.qLoraRank + self.kvLoraRank + self.idxDim
            self.qRopeOffset = self.qHeads * self.nopeDim
            self.oGateOffset = self.qHeads * self.nopeDim
            self.qChannels = self.qHeads * self.nopeDim
            self.qFullStride = self.qChannels + self.idxHeads * self.idxDim
            self.maxBlocks = max(128, (self.maxContext + 3) / 4)
        }

        let (invFreqArr, mscale) = dsa && self.ropeDim > 0
            ? QWModelDescriptor.computeRopeInvFreq(
                rotaryDim: self.ropeDim,
                base: geometry.ropeTheta,
                yarn: geometry.descriptor.hybridMLAConfig?.yarn
            )
            : ([Float](repeating: 0, count: 32), Float(1.0))
        self.ropeMscale = mscale

        var rc = 4096
        while rc < self.maxPrefillChunk * 2 { rc <<= 1 }
        self.ringCap = rc

        self.psoLatentPrep = try hal.pipeline("qwg_mla_latent_prep")
        self.psoDSALatentPrep = try hal.pipeline("qwg_mla_dsa_latent_prep")
        self.psoPoolKeys = try hal.pipeline("qwg_mla_pool_keys")
        self.psoGateWeightsProj = try hal.pipeline("qwg_mla_gate_weights_proj")
        self.psoQRopeIdxPrep = try hal.pipeline("qwg_mla_q_rope_idx_prep")
        self.psoIndexerScoreSelect = try hal.pipeline("qwg_mla_indexer_score_select")
        self.psoAbsorbQ = try hal.pipeline("qwg_mla_absorb_q")
        self.psoAbsorbQSub2 = try hal.pipeline("qwg_mla_absorb_q_sub2")
        self.psoAbsorbQMMA = try hal.pipeline("qwg_mla_absorb_q_mma")
        self.psoAbsorbQMMASub2 = try hal.pipeline("qwg_mla_absorb_q_mma_sub2")
        self.psoSparseAttn = try hal.pipeline("qwg_mla_sparse_attn")
        self.psoSparseAttnMMA = try hal.pipeline("qwg_mla_sparse_attn_mma")
        self.psoDSASparseAttn = try hal.pipeline("qwg_mla_dsa_sparse_attn")
        self.psoDSASparseAttnMMA = try hal.pipeline("qwg_mla_dsa_sparse_attn_mma")
        self.psoIndexerSplit = (self.idxHeads == 32 || self.idxHeads == 64) && self.idxDim == 128
            && ProcessInfo.processInfo.environment["QW_MLA_IDX_SPLIT"] != "0"
            ? (try hal.pipeline("qwg_mla_indexer_score_mma"), try hal.pipeline("qwg_mla_indexer_select")) : nil
        let naxOK = hal.caps.isMetal4TensorCapable
            && ProcessInfo.processInfo.environment["QW_MLA_ATTN_NAX"] != "0"
        self.psoSparseAttnNAX = naxOK ? try hal.pipeline("qwg_mla_sparse_attn_nax") : nil
        self.psoDSASparseAttnNAX = naxOK ? try hal.pipeline("qwg_mla_dsa_sparse_attn_nax") : nil
        let splits = Int(ProcessInfo.processInfo.environment["QW_MLA_ATTN_SPLIT"] ?? "") ?? 16
        self.attnSplits = (naxOK && self.kvLoraRank == 512) ? max(0, min(64, splits)) : 0
        self.psoSparseAttnSplit = self.attnSplits > 1
            ? (try hal.pipeline("qwg_mla_sparse_attn_nax_split"), try hal.pipeline("qwg_mla_sparse_attn_combine")) : nil
        self.psoDSASparseAttnSplit = self.attnSplits > 1
            ? (try hal.pipeline("qwg_mla_dsa_sparse_attn_nax_split"), try hal.pipeline("qwg_mla_dsa_sparse_attn_combine")) : nil
        self.psoProjUV = try hal.pipeline("qwg_mla_proj_uv")
        self.psoProjUVSub2 = try hal.pipeline("qwg_mla_proj_uv_sub2")
        self.psoProjUVMMA = try hal.pipeline("qwg_mla_proj_uv_mma")
        self.psoProjUVMMASub2 = try hal.pipeline("qwg_mla_proj_uv_mma_sub2")
        self.psoVHAGateRot = try hal.pipeline("qwg_mla_vha_gate_rot")

        let coarseMinCtx = dsa ? 16_384 : 32_768
        let useCoarse = (self.psoIndexerSplit != nil) && (self.maxContext >= coarseMinCtx)
        if useCoarse {
            self.coarsePipelines = try QWCoarsePoolIndex.Pipelines(hal: hal)
            self.psoEmeansCoarse = (
                try hal.pipeline("qwg_emeans_mla_score_coarse"),
                try hal.pipeline("qwg_emeans_mla_score_cands"),
                try hal.pipeline("qwg_emeans_mla_flat_fallback")
            )
        } else {
            self.coarsePipelines = nil
            self.psoEmeansCoarse = nil
        }
        if let rawEnv = ProcessInfo.processInfo.environment["QW_MLA_EMEANS"]?.lowercased() {
            self.emeansCoarseEnabled = useCoarse && rawEnv != "0" && rawEnv != "false" && rawEnv != "off"
        } else {
            self.emeansCoarseEnabled = useCoarse && lsaConfig.mode != .off
        }
        self.clusteredMacros = Array(repeating: 0, count: numAttnLayers)

        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }

        let candCap = dsa ? Self.dsaEmeansCandCap : Self.emeansCandCap
        let maxMacros = useCoarse ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext, poolShift: self.poolShift) : 0
        let csrSizes = QWCoarsePoolIndex.csrBytesPerMacro(indexDim: self.idxDim)
        var layerOffs: [(kPend: Int, gatePend: Int, pooled: Int, kPe: Int, cCent: Int, cMem: Int, cSpan: Int)] = []
        layerOffs.reserveCapacity(numAttnLayers)
        for _ in 0..<numAttnLayers {
            let kOff = dsa ? 0 : reserve(self.ringCap * self.idxDim * 2)
            let gOff = dsa ? 0 : reserve(self.ringCap * self.idxDim * 2)
            let pOff = reserve(((self.maxBlocks + 7) & ~7) * self.idxDim * 2)
            let peOff = (dsa && self.ropeDim > 0) ? reserve(((self.maxBlocks + 7) & ~7) * self.ropeDim * 2) : 0
            let cCent = useCoarse ? reserve(maxMacros * csrSizes.centroids) : 0
            let cMem = useCoarse ? reserve(maxMacros * csrSizes.members) : 0
            let cSpan = useCoarse ? reserve(maxMacros * csrSizes.spans) : 0
            layerOffs.append((kOff, gOff, pOff, peOff, cCent, cMem, cSpan))
        }

        let maxT = self.maxPrefillChunk
        let selectBatch = min(maxT, max(Self.minSelectBatchRows,
                                        Self.selectScratchBudgetBytes / max(1, self.maxBlocks * 4)))
        self.selectBatchRows = selectBatch
        let latentOff = reserve(maxT * self.latentStride * 2)
        let qResidOff = reserve(maxT * self.qLoraRank * 2)
        let qFullOff = reserve(maxT * self.qFullStride * 2)
        let idxWOff = reserve(maxT * self.idxHeads * 4)
        let qAbsOff = reserve(maxT * self.qHeads * self.kvLoraRank * 2)
        let uLatOff = reserve(maxT * self.qHeads * self.kvLoraRank * 2)
        let yVhaOff = reserve(dsa ? maxT * self.qHeads * self.vHeadDim * 2 : 0)
        let invFreqCount = max(1, invFreqArr.count)
        let invFreqOff = reserve(invFreqCount * 4)
        let zeroSinkCount = max(16, self.qHeads)
        let zeroSinkOff = reserve(zeroSinkCount * 4)
        let scoreOff = reserve(selectBatch * self.maxBlocks * 4)
        let listOff = reserve(maxT * self.listStride * 4)
        let countOff = reserve(maxT * 4)
        let splitRows = min(maxT, Self.splitMaxRows) * max(1, self.attnSplits) * self.qHeads
        let partOOff = reserve(self.attnSplits > 1 ? splitRows * self.kvLoraRank * 2 : 0)
        let partMLOff = reserve(self.attnSplits > 1 ? splitRows * 8 : 0)
        let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
        let cScSizes = QWCoarsePoolIndex.scratchByteSizes(
            maxBatchRows: selectBatch, maxClusters: maxClusters, candCap: candCap)
        let cScoreOff = useCoarse ? reserve(cScSizes.cScore) : 0
        let cHistOff = useCoarse ? reserve(cScSizes.uHist) : 0
        let cStateOff = useCoarse ? reserve(cScSizes.candState) : 0
        let cPoolsOff = useCoarse ? reserve(cScSizes.candPools) : 0
        let cUOff = useCoarse ? reserve(cScSizes.candU) : 0

        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "mla-cache", asked: totalBytes, free: 0)
        }
        buf.label = "mla-cache"
        memset(buf.contents(), 0, totalBytes)
        invFreqArr.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty {
                buf.contents().advanced(by: invFreqOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let negInf = Array(repeating: -Float.infinity, count: zeroSinkCount)
        negInf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress {
                buf.contents().advanced(by: zeroSinkOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(numAttnLayers)
        for i in 0..<numAttnLayers {
            let o = layerOffs[i]
            let csr: QWCoarsePoolIndex.LayerCSR? = useCoarse ? QWCoarsePoolIndex.LayerCSR(
                centroids: QWTensorValue(region: reg, offset: o.cCent, shape: QWTensorShape([maxClusters, self.idxDim], .f16)),
                memberPools: QWTensorValue(region: reg, offset: o.cMem, shape: QWTensorShape([maxMacros * 512], .i32)),
                clusterSpans: QWTensorValue(region: reg, offset: o.cSpan, shape: QWTensorShape([maxClusters], .i32)),
                maxMacros: maxMacros
            ) : nil
            let kPe: QWTensorValue? = (dsa && self.ropeDim > 0)
                ? QWTensorValue(region: reg, offset: o.kPe, shape: QWTensorShape([self.maxBlocks, self.ropeDim], .f16))
                : nil
            builtLayers.append(LayerState(
                fullAttentionIndex: i,
                kPending: QWTensorValue(region: reg, offset: o.kPend, shape: QWTensorShape([dsa ? 1 : self.ringCap, self.idxDim], .f16)),
                gatePending: QWTensorValue(region: reg, offset: o.gatePend, shape: QWTensorShape([dsa ? 1 : self.ringCap, self.idxDim], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([self.maxBlocks, self.idxDim], .f16)),
                kPeCache: kPe,
                coarseCSR: csr
            ))
        }
        self.layers = builtLayers
        self.coarseScratch = useCoarse ? QWCoarsePoolIndex.Scratch(
            cScore: QWTensorValue(region: reg, offset: cScoreOff, shape: QWTensorShape([selectBatch, maxClusters], .i32)),
            uHist: QWTensorValue(region: reg, offset: cHistOff, shape: QWTensorShape([selectBatch, 512], .i32)),
            candState: QWTensorValue(region: reg, offset: cStateOff, shape: QWTensorShape([selectBatch * 2], .i32)),
            candPools: QWTensorValue(region: reg, offset: cPoolsOff, shape: QWTensorShape([selectBatch, candCap], .i32)),
            candU: QWTensorValue(region: reg, offset: cUOff, shape: QWTensorShape([selectBatch, candCap], .i32)),
            maxBatchRows: selectBatch,
            clusterStride: maxClusters,
            candCap: candCap
        ) : nil

        self.mlaLatentScratch = QWTensorValue(region: reg, offset: latentOff, shape: QWTensorShape([maxT, self.latentStride], .f16))
        self.qResidScratch = QWTensorValue(region: reg, offset: qResidOff, shape: QWTensorShape([maxT, self.qLoraRank], .f16))
        self.mlaQFullScratch = QWTensorValue(region: reg, offset: qFullOff, shape: QWTensorShape([maxT, self.qFullStride], .f16))
        self.idxWeightsScratch = QWTensorValue(region: reg, offset: idxWOff, shape: QWTensorShape([maxT, self.idxHeads], .f32))
        self.qAbsScratch = QWTensorValue(region: reg, offset: qAbsOff, shape: QWTensorShape([maxT, self.qHeads * self.kvLoraRank], .f16))
        self.uLatScratch = QWTensorValue(region: reg, offset: uLatOff, shape: QWTensorShape([maxT, self.qHeads * self.kvLoraRank], .f16))
        self.yVhaScratch = QWTensorValue(region: reg, offset: yVhaOff, shape: QWTensorShape([max(1, dsa ? maxT : 1), self.qHeads * self.vHeadDim], .f16))
        self.invFreqBuf = QWTensorValue(region: reg, offset: invFreqOff, shape: QWTensorShape([invFreqCount], .f32))
        self.zeroSinkBuf = QWTensorValue(region: reg, offset: zeroSinkOff, shape: QWTensorShape([zeroSinkCount], .f32))
        self.scoreScratch = QWTensorValue(region: reg, offset: scoreOff, shape: QWTensorShape([selectBatch, self.maxBlocks], .f32))
        self.blockListScratch = QWTensorValue(region: reg, offset: listOff, shape: QWTensorShape([maxT, self.listStride], .i32))
        self.blockCountScratch = QWTensorValue(region: reg, offset: countOff, shape: QWTensorShape([maxT], .i32))
        self.partOScratch = QWTensorValue(region: reg, offset: partOOff, shape: QWTensorShape([max(1, splitRows), self.kvLoraRank], .f16))
        self.partMLScratch = QWTensorValue(region: reg, offset: partMLOff, shape: QWTensorShape([max(1, splitRows), 2], .f32))
    }

    public func indexerStateBytes(position: Int) -> Int {
        if isDecoupledDSA {
            return layers.count * position * (idxDim + ropeDim) * 2
        }
        let rowBytes = idxDim * 2
        return layers.count * ((position / 4) + 2 * (position % 4)) * rowBytes
    }

    public func captureIndexerState(position: Int) -> Data {
        if isDecoupledDSA { return captureDSAIndexerState(position: position) }
        precondition(position >= 0 && position / 4 <= maxBlocks)
        let rowBytes = idxDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let openStart = (position / 4) * 4
        var out = Data(capacity: indexerStateBytes(position: position))
        let base = region.pointer
        for ls in layers {
            out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self),
                       count: pooledBytes)
            for ring in [ls.kPending, ls.gatePending] {
                for p in openStart..<position {
                    let row = p & (ringCap - 1)
                    out.append(base.advanced(by: ring.offset + row * rowBytes).assumingMemoryBound(to: UInt8.self),
                               count: rowBytes)
                }
            }
        }
        return out
    }

    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        if isDecoupledDSA { return restoreDSAIndexerState(data, position: position) }
        guard position >= 0, position / 4 <= maxBlocks,
              data.count == indexerStateBytes(position: position) else { return false }
        let rowBytes = idxDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let openStart = (position / 4) * 4
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var src = 0
            for ls in layers {
                base.advanced(by: ls.pooledKeys.offset)
                    .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: pooledBytes)
                src += pooledBytes
                for ring in [ls.kPending, ls.gatePending] {
                    for p in openStart..<position {
                        let row = p & (ringCap - 1)
                        base.advanced(by: ring.offset + row * rowBytes)
                            .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: rowBytes)
                        src += rowBytes
                    }
                }
            }
        }
        resetCoarseWatermarks()
        return true
    }

    public func encodeLatentPrepAndPool(_ enc: any MTLComputeCommandEncoder,
                                        ln1: QWTensorValue,
                                        mlaW: LayerWeights,
                                        kv: (k: QWKVLayerView, v: QWKVLayerView),
                                        fullAttnLayer: Int,
                                        pos0: Int,
                                        rows: Int) {
        if isDecoupledDSA {
            encodeDSALatentPrepAndCluster(enc, ln1: ln1, mlaW: mlaW, kv: kv,
                                          fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
            return
        }
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        let ls = layers[fullAttnLayer]
        var prepP = LatentPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            hidden: UInt32(geometry.hidden),
            qLoraRank: UInt32(qLoraRank),
            kvLoraRank: UInt32(kvLoraRank),
            idxDim: UInt32(idxDim),
            idxHeads: UInt32(idxHeads),
            ringMask: UInt32(ringCap - 1),
            rmsEps: geometry.rmsEps,
            kNormEps: 1e-6,
            kvMode: Self.kvModeCode(kv.k.mode)
        )

        enc.setComputePipelineState(psoLatentPrep)
        mlaLatentScratch.bind(enc, index: 0)
        ln1.bind(enc, index: 1)
        mlaW.qANormW.bind(enc, index: 2)
        mlaW.kvANormW.bind(enc, index: 3)
        mlaW.kNormW.bind(enc, index: 4)
        mlaW.kNormB.bind(enc, index: 5)
        mlaW.wGateWeights.bind(enc, index: 6)
        qResidScratch.bind(enc, index: 7)
        kv.k.pool.bindTo(enc, index: 8)
        kv.v.pool.bindTo(enc, index: 9)
        kv.k.table.bind(enc, index: 10)
        kv.v.table.bind(enc, index: 11)
        ls.kPending.bind(enc, index: 12)
        ls.gatePending.bind(enc, index: 13)
        idxWeightsScratch.bind(enc, index: 14)
        enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoGateWeightsProj)
        ln1.bind(enc, index: 0)
        mlaW.wGateWeights.bind(enc, index: 1)
        ls.gatePending.bind(enc, index: 2)
        idxWeightsScratch.bind(enc, index: 3)
        enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (idxDim + idxHeads + 7) / 8, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let startBlock = pos0 / 4
        let endBlock = (pos0 + rows) / 4
        let numBlocks = endBlock - startBlock
        if numBlocks > 0, let ape = mlaW.ape {
            var sB = UInt32(startBlock)
            var nB = UInt32(numBlocks)
            var dU = UInt32(idxDim)
            var rM = UInt32(ringCap - 1)
            enc.setComputePipelineState(psoPoolKeys)
            ls.kPending.bind(enc, index: 0)
            ls.gatePending.bind(enc, index: 1)
            ape.bind(enc, index: 2)
            ls.pooledKeys.bind(enc, index: 3)
            enc.setBytes(&sB, length: 4, index: 4)
            enc.setBytes(&nB, length: 4, index: 5)
            enc.setBytes(&dU, length: 4, index: 6)
            enc.setBytes(&rM, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(128, idxDim), height: 1, depth: 1))
        }
        updateCoarseClusters(enc, ls: ls, fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
    }

    internal func updateCoarseClusters(_ enc: any MTLComputeCommandEncoder,
                                       ls: LayerState,
                                       fullAttnLayer: Int,
                                       pos0: Int,
                                       rows: Int) {
        guard emeansCoarseEnabled, let pipes = coarsePipelines, let csr = ls.coarseCSR else { return }
        let chunkTok = QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        let curMacro = pos0 / chunkTok
        if curMacro < clusteredMacros[fullAttnLayer] {
            clusteredMacros[fullAttnLayer] = curMacro
        }
        let targetMacros = min(csr.maxMacros, (pos0 + rows) / chunkTok)
        let prevMacros = clusteredMacros[fullAttnLayer]
        if targetMacros > prevMacros {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            QWCoarsePoolIndex.encodeClusterMacros(
                enc, pso: pipes.cluster1024, pooledKeys: ls.pooledKeys,
                csr: csr, startMacro: prevMacros, numMacros: targetMacros - prevMacros, indexDim: idxDim)
            clusteredMacros[fullAttnLayer] = targetMacros
        }
    }

    public func encodeScoreSelectAbsorbAttendAndProjUV(_ enc: any MTLComputeCommandEncoder,
                                                       mlaW: LayerWeights,
                                                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                       attOut: QWTensorValue,
                                                       fullAttnLayer: Int,
                                                       pos0: Int,
                                                       rows: Int,
                                                       stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        if isDecoupledDSA {
            encodeDSAScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: attOut,
                fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows, stage: stage)
            return
        }
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[fullAttnLayer]
        mark("mla.score_select")

        encodeIndexerSelection(enc, ls: ls, pos0: pos0, rows: rows,
                               probeClusters: Self.emeansProbeClusters, windowBlocks: 0)

        var absP = AbsorbParams(
            T: UInt32(rows),
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank),
            qFullStride: UInt32(qFullStride),
            bits: UInt32(mlaW.kvBProj.bits),
            group: UInt32(mlaW.kvBProj.groupSize),
            isIncumbent: mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0)
        )
        let mTiles = (rows + 7) / 8
        mark("mla.absorb_q")
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: rows, mTiles: mTiles)

        var attnP = SparseAttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(nopeDim),
            kvLoraRank: UInt32(kvLoraRank),
            topKPools: UInt32(topKPools),
            listStride: UInt32(listStride),
            kvMode: Self.kvModeCode(kv.k.mode)
        )
        mark("mla.sparse_attn")
        let useMMA = sparseAttnMMAEnabled && attnP.kvMode <= 2 && rows >= 16
            && kvLoraRank == 512 && qHeads % 16 == 0
        let split = (sparseAttnMMAEnabled && !useMMA && attnP.kvMode <= 2 && rows <= Self.splitMaxRows
            && qHeads % 16 == 0) ? psoSparseAttnSplit : nil
        enc.setComputePipelineState(split?.attend ?? (useMMA ? (psoSparseAttnNAX ?? psoSparseAttnMMA) : psoSparseAttn))
        qAbsScratch.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        blockListScratch.bind(enc, index: 5)
        (split != nil ? partOScratch : uLatScratch).bind(enc, index: 6)
        enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 7)
        if let split {
            partMLScratch.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: attnSplits),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            var nSplitU = UInt32(attnSplits)
            enc.setComputePipelineState(split.combine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            uLatScratch.bind(enc, index: 2)
            enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 3)
            enc.setBytes(&nSplitU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if useMMA {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        mark("mla.proj_uv")
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: attOut, absP: &absP,
                     absorbMMA: absorbMMA, rows: rows, mTiles: mTiles)
    }
}
