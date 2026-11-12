// QWMLAAttnOperator.swift — Matrix-Absorbed Multi-head Latent Attention (NoPE MLA)
// and Key-Pool(4) Sparse Indexer GPU operator for hybrid MLA + MoE models.
//
// Manages per-MLA-layer indexer ring buffers (`kPending`, `gatePending`) and
// compressed 4-token key pools (`pooledKeys`), plus shared latent/absorbed
// query/attention scratch in a single preallocated Metal region.

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
        public let ape: QWTensorValue
        public let qBProj: QWWeightView
        public let kvBProj: QWWeightView

        public init(qANormW: QWTensorValue,
                    kvANormW: QWTensorValue,
                    kNormW: QWTensorValue,
                    kNormB: QWTensorValue,
                    wGateWeights: QWTensorValue,
                    ape: QWTensorValue,
                    qBProj: QWWeightView,
                    kvBProj: QWWeightView) {
            self.qANormW = qANormW
            self.kvANormW = kvANormW
            self.kNormW = kNormW
            self.kNormB = kNormB
            self.wGateWeights = wGateWeights
            self.ape = ape
            self.qBProj = qBProj
            self.kvBProj = kvBProj
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let kPending: QWTensorValue
        public let gatePending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
    }

    public static let minSelectBatchRows = 32
    public static let emeansProbeClusters = 320
    public static let emeansCandCap = 10240
    /// Indexer score scratch budget; bounds rows per select dispatch so long
    /// contexts stay small while short ones fill the GPU in few dispatches.
    public static let selectScratchBudgetBytes = 16 << 20
    public let selectBatchRows: Int

    public let geometry: QWModelGeometry
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let nopeDim: Int
    public let vHeadDim: Int
    public let idxHeads: Int
    public let idxDim: Int
    public let topKPools: Int
    public let listStride: Int
    public let ringCap: Int
    public let maxBlocks: Int
    public let latentStride: Int
    public let qFullStride: Int

    public let region: QWMRegion
    public let layers: [LayerState]
    /// The session whose indexer pools the layers currently hold (driver-managed).
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
    public let scoreScratch: QWTensorValue
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue

    private let hal: QWHal
    private let psoLatentPrep: any MTLComputePipelineState
    private let psoPoolKeys: any MTLComputePipelineState
    private let psoGateWeightsProj: any MTLComputePipelineState
    private let psoIndexerScoreSelect: any MTLComputePipelineState
    private let psoAbsorbQ: any MTLComputePipelineState
    private let psoAbsorbQSub2: any MTLComputePipelineState
    private let psoAbsorbQMMA: any MTLComputePipelineState
    private let psoAbsorbQMMASub2: any MTLComputePipelineState
    private let psoSparseAttn: any MTLComputePipelineState
    private let psoSparseAttnMMA: any MTLComputePipelineState
    /// Pool-parallel MMA scoring + per-row select (32 heads x 128 dims); QW_MLA_IDX_SPLIT=0 keeps the fused kernel.
    private let psoIndexerSplit: (score: any MTLComputePipelineState, select: any MTLComputePipelineState)?
    /// NAX tensor-MMA variant (Apple10+); QW_MLA_ATTN_NAX=0 falls back to simdgroup MMA.
    private let psoSparseAttnNAX: (any MTLComputePipelineState)?
    /// Split-K NAX decode attention + log-sum-exp combine; QW_MLA_ATTN_SPLIT=<n> sets splits (0 disables).
    private let psoSparseAttnSplit: (attend: any MTLComputePipelineState, combine: any MTLComputePipelineState)?
    public static let splitMaxRows = 15
    public let attnSplits: Int
    private let partOScratch: QWTensorValue
    private let partMLScratch: QWTensorValue
    /// Prefill-sized (>= 16 rows) absorb and sparse-attention run on simdgroup MMA; QW_MLA_ATTN_MMA=0 forces the scalar kernels.
    public var sparseAttnMMAEnabled = ProcessInfo.processInfo.environment["QW_MLA_ATTN_MMA"] != "0"
    /// Coarse `e-means` MLA pruning (`<41K` flat bypass + `>=41K` centroid Top-8); controlled by `lsaConfig.mode` or `QW_MLA_EMEANS`.
    public var emeansCoarseEnabled: Bool
    private let psoProjUV: any MTLComputePipelineState
    private let psoProjUVSub2: any MTLComputePipelineState
    private let psoProjUVMMA: any MTLComputePipelineState
    private let psoProjUVMMASub2: any MTLComputePipelineState
    private let coarsePipelines: QWCoarsePoolIndex.Pipelines?
    private let psoEmeansCoarse: (scoreCoarse: any MTLComputePipelineState, scoreCands: any MTLComputePipelineState, flatFallback: any MTLComputePipelineState)?
    private let coarseScratch: QWCoarsePoolIndex.Scratch?
    private var clusteredMacros: [Int]

    public func resetCoarseWatermarks() {
        for i in clusteredMacros.indices { clusteredMacros[i] = 0 }
    }

    private struct LatentPrepParams {
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

    private struct IndexerParams {
        var T: UInt32
        var pos0: UInt32
        var qChannels: UInt32
        var idxHeads: UInt32
        var idxDim: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var scoreStride: UInt32
    }

    private struct AbsorbParams {
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

    private struct SparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topKPools: UInt32
        var listStride: UInt32
        var kvMode: UInt32
    }

    private static func kvModeCode(_ mode: QWKVMode) -> UInt32 {
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
        guard geometry.kvLoraRank > 0 && geometry.qLoraRank > 0, !attnLayers.isEmpty else {
            return nil
        }
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
        self.maxContext = max(512, maxContext)
        self.maxPrefillChunk = max(32, maxPrefillChunk)
        self.qLoraRank = geometry.qLoraRank
        self.kvLoraRank = geometry.kvLoraRank
        self.nopeDim = geometry.qkNopeHeadDim > 0 ? geometry.qkNopeHeadDim : geometry.headDim
        self.vHeadDim = geometry.vHeadDim > 0 ? geometry.vHeadDim : geometry.headDim
        self.idxHeads = max(1, geometry.indexerHeads)
        self.idxDim = max(32, geometry.indexerHeadDim)
        let modelBudget = geometry.indexerBudget > 0 ? geometry.indexerBudget : 2048
        let requestedBudget = (lsaConfig.isEnabled || lsaConfig.budget != QWLSAConfig.defaultBudget)
            ? lsaConfig.alignedBudget
            : modelBudget
        let budgetTok = min(4096, max(128, requestedBudget))
        self.topKPools = max(1, min(1024, budgetTok / 4))
        self.listStride = max(512, self.topKPools)
        self.latentStride = self.qLoraRank + self.kvLoraRank + self.idxDim
        self.qFullStride = geometry.qHeads * self.nopeDim + self.idxHeads * self.idxDim

        var rc = 4096
        while rc < self.maxPrefillChunk * 2 { rc <<= 1 }
        self.ringCap = rc
        self.maxBlocks = max(128, (self.maxContext + 3) / 4)

        self.psoLatentPrep = try hal.pipeline("qwg_mla_latent_prep")
        self.psoPoolKeys = try hal.pipeline("qwg_mla_pool_keys")
        self.psoGateWeightsProj = try hal.pipeline("qwg_mla_gate_weights_proj")
        self.psoIndexerScoreSelect = try hal.pipeline("qwg_mla_indexer_score_select")
        self.psoAbsorbQ = try hal.pipeline("qwg_mla_absorb_q")
        self.psoAbsorbQSub2 = try hal.pipeline("qwg_mla_absorb_q_sub2")
        self.psoAbsorbQMMA = try hal.pipeline("qwg_mla_absorb_q_mma")
        self.psoAbsorbQMMASub2 = try hal.pipeline("qwg_mla_absorb_q_mma_sub2")
        self.psoSparseAttn = try hal.pipeline("qwg_mla_sparse_attn")
        self.psoSparseAttnMMA = try hal.pipeline("qwg_mla_sparse_attn_mma")
        self.psoIndexerSplit = idxHeads == 32 && idxDim == 128
            && ProcessInfo.processInfo.environment["QW_MLA_IDX_SPLIT"] != "0"
            ? (try hal.pipeline("qwg_mla_indexer_score_mma"), try hal.pipeline("qwg_mla_indexer_select")) : nil
        self.psoSparseAttnNAX = hal.caps.isMetal4TensorCapable
            && ProcessInfo.processInfo.environment["QW_MLA_ATTN_NAX"] != "0"
            ? try hal.pipeline("qwg_mla_sparse_attn_nax") : nil
        let splits = Int(ProcessInfo.processInfo.environment["QW_MLA_ATTN_SPLIT"] ?? "") ?? 16
        self.attnSplits = (self.psoSparseAttnNAX != nil && self.kvLoraRank == 512) ? max(0, min(64, splits)) : 0
        self.psoSparseAttnSplit = attnSplits > 1
            ? (try hal.pipeline("qwg_mla_sparse_attn_nax_split"), try hal.pipeline("qwg_mla_sparse_attn_combine")) : nil
        self.psoProjUV = try hal.pipeline("qwg_mla_proj_uv")
        self.psoProjUVSub2 = try hal.pipeline("qwg_mla_proj_uv_sub2")
        self.psoProjUVMMA = try hal.pipeline("qwg_mla_proj_uv_mma")
        self.psoProjUVMMASub2 = try hal.pipeline("qwg_mla_proj_uv_mma_sub2")
        let useCoarse = (self.psoIndexerSplit != nil) && (self.maxContext >= 32_768)
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

        let maxMacros = useCoarse ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext) : 0
        let csrSizes = QWCoarsePoolIndex.csrBytesPerMacro(indexDim: idxDim)
        var layerOffs: [(kPend: Int, gatePend: Int, pooled: Int, cCent: Int, cMem: Int, cSpan: Int)] = []
        layerOffs.reserveCapacity(numAttnLayers)
        for _ in 0..<numAttnLayers {
            let kOff = reserve(ringCap * idxDim * 2)
            let gOff = reserve(ringCap * idxDim * 2)
            let pOff = reserve(((maxBlocks + 7) & ~7) * idxDim * 2)
            let cCent = useCoarse ? reserve(maxMacros * csrSizes.centroids) : 0
            let cMem = useCoarse ? reserve(maxMacros * csrSizes.members) : 0
            let cSpan = useCoarse ? reserve(maxMacros * csrSizes.spans) : 0
            layerOffs.append((kOff, gOff, pOff, cCent, cMem, cSpan))
        }

        let maxT = self.maxPrefillChunk
        let selectBatch = min(maxT, max(Self.minSelectBatchRows,
                                        Self.selectScratchBudgetBytes / max(1, maxBlocks * 4)))
        self.selectBatchRows = selectBatch
        let latentOff = reserve(maxT * latentStride * 2)
        let qResidOff = reserve(maxT * qLoraRank * 2)
        let qFullOff = reserve(maxT * qFullStride * 2)
        let idxWOff = reserve(maxT * idxHeads * 4)
        let qAbsOff = reserve(maxT * geometry.qHeads * kvLoraRank * 2)
        let uLatOff = reserve(maxT * geometry.qHeads * kvLoraRank * 2)
        let scoreOff = reserve(selectBatch * maxBlocks * 4)
        let listOff = reserve(maxT * listStride * 4)
        let countOff = reserve(maxT * 4)
        let splitRows = min(maxT, Self.splitMaxRows) * max(1, attnSplits) * geometry.qHeads
        let partOOff = reserve(attnSplits > 1 ? splitRows * kvLoraRank * 2 : 0)
        let partMLOff = reserve(attnSplits > 1 ? splitRows * 8 : 0)
        let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
        let cScSizes = QWCoarsePoolIndex.scratchByteSizes(
            maxBatchRows: selectBatch, maxClusters: maxClusters, candCap: Self.emeansCandCap)
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
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(numAttnLayers)
        for i in 0..<numAttnLayers {
            let o = layerOffs[i]
            let csr: QWCoarsePoolIndex.LayerCSR? = useCoarse ? QWCoarsePoolIndex.LayerCSR(
                centroids: QWTensorValue(region: reg, offset: o.cCent, shape: QWTensorShape([maxClusters, idxDim], .f16)),
                memberPools: QWTensorValue(region: reg, offset: o.cMem, shape: QWTensorShape([maxMacros * 512], .i32)),
                clusterSpans: QWTensorValue(region: reg, offset: o.cSpan, shape: QWTensorShape([maxClusters], .i32)),
                maxMacros: maxMacros
            ) : nil
            builtLayers.append(LayerState(
                fullAttentionIndex: i,
                kPending: QWTensorValue(region: reg, offset: o.kPend, shape: QWTensorShape([ringCap, idxDim], .f16)),
                gatePending: QWTensorValue(region: reg, offset: o.gatePend, shape: QWTensorShape([ringCap, idxDim], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([maxBlocks, idxDim], .f16)),
                coarseCSR: csr
            ))
        }
        self.layers = builtLayers
        self.coarseScratch = useCoarse ? QWCoarsePoolIndex.Scratch(
            cScore: QWTensorValue(region: reg, offset: cScoreOff, shape: QWTensorShape([selectBatch, maxClusters], .i32)),
            uHist: QWTensorValue(region: reg, offset: cHistOff, shape: QWTensorShape([selectBatch, 512], .i32)),
            candState: QWTensorValue(region: reg, offset: cStateOff, shape: QWTensorShape([selectBatch * 2], .i32)),
            candPools: QWTensorValue(region: reg, offset: cPoolsOff, shape: QWTensorShape([selectBatch, Self.emeansCandCap], .i32)),
            candU: QWTensorValue(region: reg, offset: cUOff, shape: QWTensorShape([selectBatch, Self.emeansCandCap], .i32)),
            maxBatchRows: selectBatch,
            clusterStride: maxClusters,
            candCap: Self.emeansCandCap
        ) : nil

        self.mlaLatentScratch = QWTensorValue(region: reg, offset: latentOff, shape: QWTensorShape([maxT, latentStride], .f16))
        self.qResidScratch = QWTensorValue(region: reg, offset: qResidOff, shape: QWTensorShape([maxT, qLoraRank], .f16))
        self.mlaQFullScratch = QWTensorValue(region: reg, offset: qFullOff, shape: QWTensorShape([maxT, qFullStride], .f16))
        self.idxWeightsScratch = QWTensorValue(region: reg, offset: idxWOff, shape: QWTensorShape([maxT, idxHeads], .f32))
        self.qAbsScratch = QWTensorValue(region: reg, offset: qAbsOff, shape: QWTensorShape([maxT, geometry.qHeads * kvLoraRank], .f16))
        self.uLatScratch = QWTensorValue(region: reg, offset: uLatOff, shape: QWTensorShape([maxT, geometry.qHeads * kvLoraRank], .f16))
        self.scoreScratch = QWTensorValue(region: reg, offset: scoreOff, shape: QWTensorShape([selectBatch, maxBlocks], .f32))
        self.blockListScratch = QWTensorValue(region: reg, offset: listOff, shape: QWTensorShape([maxT, listStride], .i32))
        self.blockCountScratch = QWTensorValue(region: reg, offset: countOff, shape: QWTensorShape([maxT], .i32))
        self.partOScratch = QWTensorValue(region: reg, offset: partOOff, shape: QWTensorShape([splitRows, kvLoraRank], .f16))
        self.partMLScratch = QWTensorValue(region: reg, offset: partMLOff, shape: QWTensorShape([splitRows, 2], .f32))
    }

    /// Indexer state lives in this operator, not in the session slabs, so a
    /// prefix snapshot must carry it: every completed pooled block plus the
    /// pending ring rows of the open block, for every layer.
    public func indexerStateBytes(position: Int) -> Int {
        let rowBytes = idxDim * 2
        return layers.count * ((position / 4) + 2 * (position % 4)) * rowBytes
    }

    /// The caller must have waited for every command buffer that writes this state.
    public func captureIndexerState(position: Int) -> Data {
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

    /// Returns false when `data` does not match this operator's geometry at `position`.
    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        guard position >= 0, position / 4 <= maxBlocks,
              data.count == indexerStateBytes(position: position) else { return false }
        let rowBytes = idxDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let openStart = (position / 4) * 4
        let base = region.pointer
        data.withUnsafeBytes { raw in
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
        if numBlocks > 0 {
            var sB = UInt32(startBlock)
            var nB = UInt32(numBlocks)
            var dU = UInt32(idxDim)
            var rM = UInt32(ringCap - 1)
            enc.setComputePipelineState(psoPoolKeys)
            ls.kPending.bind(enc, index: 0)
            ls.gatePending.bind(enc, index: 1)
            mlaW.ape.bind(enc, index: 2)
            ls.pooledKeys.bind(enc, index: 3)
            enc.setBytes(&sB, length: 4, index: 4)
            enc.setBytes(&nB, length: 4, index: 5)
            enc.setBytes(&dU, length: 4, index: 6)
            enc.setBytes(&rM, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(128, idxDim), height: 1, depth: 1))
        }
        if emeansCoarseEnabled, let pipes = coarsePipelines, let csr = ls.coarseCSR {
            let curMacro = pos0 / QWCoarsePoolIndex.macroChunkTokens
            if curMacro < clusteredMacros[fullAttnLayer] {
                clusteredMacros[fullAttnLayer] = curMacro
            }
            let targetMacros = min(csr.maxMacros, (pos0 + rows) / QWCoarsePoolIndex.macroChunkTokens)
            let prevMacros = clusteredMacros[fullAttnLayer]
            if targetMacros > prevMacros {
                QWCoarsePoolIndex.barrierIfConcurrent(enc)
                QWCoarsePoolIndex.encodeClusterMacros(
                    enc, pso: pipes.cluster1024, pooledKeys: ls.pooledKeys,
                    csr: csr, startMacro: prevMacros, numMacros: targetMacros - prevMacros, indexDim: idxDim)
                clusteredMacros[fullAttnLayer] = targetMacros
            }
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
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        var enc = enc
        // Optional diagnostic hook: the caller may rotate the encoder at sub-stage boundaries.
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[fullAttnLayer]
        let qChannels = geometry.qHeads * nopeDim
        mark("mla.score_select")

        // 1. Indexer score + top-512 pool selection (sub-batched to selectBatchRows for bounded scoreScratch)
        var rowStart = 0
        while rowStart < rows {
            let batchRows = min(selectBatchRows, rows - rowStart)
            let qFullSub = mlaQFullScratch.slice(byteOffset: rowStart * qFullStride * 2,
                                                 count: batchRows * qFullStride, dtype: .f16)
            let idxWSub = idxWeightsScratch.slice(byteOffset: rowStart * idxHeads * 4,
                                                  count: batchRows * idxHeads, dtype: .f32)
            let listSub = blockListScratch.slice(byteOffset: rowStart * listStride * 4,
                                                 count: batchRows * listStride, dtype: .i32)
            let countSub = blockCountScratch.slice(byteOffset: rowStart * 4,
                                                   count: batchRows, dtype: .i32)
            var idxP = IndexerParams(
                T: UInt32(batchRows),
                pos0: UInt32(pos0 + rowStart),
                qChannels: UInt32(qChannels),
                idxHeads: UInt32(idxHeads),
                idxDim: UInt32(idxDim),
                topKPools: UInt32(topKPools),
                listStride: UInt32(listStride),
                scoreStride: UInt32(maxBlocks)
            )
            let tg256 = MTLSize(width: 256, height: 1, depth: 1)
            if emeansCoarseEnabled,
               let pipes = coarsePipelines,
               let mlaCoarse = psoEmeansCoarse,
               let csr = ls.coarseCSR,
               let cScratch = coarseScratch {
                let bg = QWCoarsePoolIndex.batchGeometry(
                    pos0: pos0 + rowStart, units: batchRows, budgetBlocks: topKPools,
                    windowBlocks: 0, hasSink: false, writeCount: true,
                    probeClusters: Self.emeansProbeClusters)
                if bg.shouldPrune {
                    var ePrm = QWCoarsePoolIndex.SelectParams(
                        T: batchRows, pos0: pos0 + rowStart, indexDim: idxDim,
                        budgetBlocks: topKPools, windowBlocks: 0, hasSink: false,
                        writeCount: true, listStride: listStride, rowStep: 1, maxRow: batchRows,
                        clusterStride: cScratch.clusterStride, probeClusters: Self.emeansProbeClusters,
                        candCap: cScratch.candCap, qChannels: qChannels, idxHeads: idxHeads)
                    let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
                    enc.setComputePipelineState(mlaCoarse.scoreCoarse)
                    qFullSub.bind(enc, index: 0)
                    idxWSub.bind(enc, index: 1)
                    csr.centroids.bind(enc, index: 2)
                    csr.clusterSpans.bind(enc, index: 3)
                    cScratch.cScore.bind(enc, index: 4)
                    cScratch.uHist.bind(enc, index: 5)
                    enc.setBytes(&ePrm, length: plen, index: 6)
                    enc.dispatchThreadgroups(
                        MTLSize(width: (bg.numClusters + 63) / 64, height: batchRows, depth: 1),
                        threadsPerThreadgroup: tg256)
                    QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
                        enc, pipelines: pipes, csr: csr, scratch: cScratch,
                        blockList: listSub, blockCount: countSub, params: &ePrm,
                        numClusters: bg.numClusters, maxMiddlePools: bg.middleEnd,
                        encodeScoreCands: { enc, maxCand, prm in
                            enc.setComputePipelineState(mlaCoarse.scoreCands)
                            qFullSub.bind(enc, index: 0)
                            idxWSub.bind(enc, index: 1)
                            ls.pooledKeys.bind(enc, index: 2)
                            cScratch.candState.bind(enc, index: 3)
                            cScratch.candPools.bind(enc, index: 4)
                            cScratch.candU.bind(enc, index: 5)
                            enc.setBytes(&prm, length: plen, index: 6)
                            enc.dispatchThreadgroups(
                                MTLSize(width: (maxCand + 63) / 64, height: batchRows, depth: 1),
                                threadsPerThreadgroup: tg256)
                        },
                        encodeFlatFallback: { enc, prm in
                            enc.setComputePipelineState(mlaCoarse.flatFallback)
                            qFullSub.bind(enc, index: 0)
                            idxWSub.bind(enc, index: 1)
                            ls.pooledKeys.bind(enc, index: 2)
                            cScratch.candState.bind(enc, index: 3)
                            listSub.bind(enc, index: 4)
                            countSub.bind(enc, index: 5)
                            enc.setBytes(&prm, length: plen, index: 6)
                            enc.dispatchThreadgroups(
                                MTLSize(width: batchRows, height: 1, depth: 1),
                                threadsPerThreadgroup: tg256)
                        }
                    )
                    QWCoarsePoolIndex.barrierIfConcurrent(enc)
                    rowStart += batchRows
                    continue
                }
            }
            qFullSub.bind(enc, index: 0)
            idxWSub.bind(enc, index: 1)
            ls.pooledKeys.bind(enc, index: 2)
            scoreScratch.bind(enc, index: 3)
            listSub.bind(enc, index: 4)
            countSub.bind(enc, index: 5)
            enc.setBytes(&idxP, length: MemoryLayout<IndexerParams>.stride, index: 6)
            if let split = psoIndexerSplit {
                let maxPools = (pos0 + rowStart + batchRows) / 4
                if maxPools > topKPools {
                    enc.setComputePipelineState(split.score)
                    enc.dispatchThreadgroups(MTLSize(width: (maxPools + 63) / 64, height: batchRows, depth: 1),
                                             threadsPerThreadgroup: tg256)
                }
                enc.setComputePipelineState(split.select)
            } else {
                enc.setComputePipelineState(psoIndexerScoreSelect)
            }
            enc.dispatchThreadgroups(MTLSize(width: batchRows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
            rowStart += batchRows
        }

        // 2. Absorb W_{UK, h} into q_nope -> qAbs [rows, qHeads, kvLoraRank]
        var absP = AbsorbParams(
            T: UInt32(rows),
            qHeads: UInt32(geometry.qHeads),
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
        let isSub2SHVQ = absP.isIncumbent == 2 && (absP.bits == 1 || absP.bits == 15)
        let weightSupported = absP.group >= 32 && (absP.group % 32 == 0) && (kvLoraRank % Int(absP.group) == 0) && (
            absP.isIncumbent == 1
            || (absP.isIncumbent == 2 && (isSub2SHVQ || (absP.bits >= 2 && absP.bits <= 4)))
            || (absP.isIncumbent == 0 && (absP.bits == 2 || absP.bits == 3 || absP.bits == 4 || absP.bits == 6 || absP.bits == 8))
        )
        let absorbMMA = sparseAttnMMAEnabled && rows >= 16 && weightSupported
            && nopeDim == 256 && kvLoraRank == 512
        enc.setComputePipelineState(absorbMMA ? (isSub2SHVQ ? psoAbsorbQMMASub2 : psoAbsorbQMMA)
                                              : (isSub2SHVQ ? psoAbsorbQSub2 : psoAbsorbQ))
        mlaQFullScratch.bind(enc, index: 0)
        mlaW.kvBProj.blocks.bind(enc, index: 1)
        qAbsScratch.bind(enc, index: 2)
        enc.setBytes(&absP, length: MemoryLayout<AbsorbParams>.stride, index: 3)
        if absorbMMA {
            enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: (rows + 31) / 32, depth: kvLoraRank / 128),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: mTiles, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        }

        // 3. Sparse latent attention over selected 4-token pools + incomplete causal tail -> uLat
        var attnP = SparseAttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(geometry.qHeads),
            headDim: UInt32(nopeDim),
            kvLoraRank: UInt32(kvLoraRank),
            topKPools: UInt32(topKPools),
            listStride: UInt32(listStride),
            kvMode: Self.kvModeCode(kv.k.mode)
        )
        mark("mla.sparse_attn")
        // Decode/verify rows split keys across threadgroups (NAX) or keep the per-head scalar kernel:
        // unsplit MMA would launch only qHeads/16 threadgroups per row.
        let useMMA = sparseAttnMMAEnabled && attnP.kvMode <= 2 && rows >= 16
            && kvLoraRank == 512 && geometry.qHeads % 16 == 0
        let split = (sparseAttnMMAEnabled && !useMMA && attnP.kvMode <= 2 && rows <= Self.splitMaxRows
            && geometry.qHeads % 16 == 0) ? psoSparseAttnSplit : nil
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
            enc.dispatchThreadgroups(MTLSize(width: rows, height: geometry.qHeads / 16, depth: attnSplits),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            var nSplitU = UInt32(attnSplits)
            enc.setComputePipelineState(split.combine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            uLatScratch.bind(enc, index: 2)
            enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 3)
            enc.setBytes(&nSplitU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: rows * geometry.qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if useMMA {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: geometry.qHeads / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // 4. Value up-projection by W_{UV, h} + fused H_32 rotation -> attOut [rows, qHeads * vHeadDim]
        mark("mla.proj_uv")
        let projUVMMA = absorbMMA && vHeadDim == 256
        enc.setComputePipelineState(projUVMMA ? (isSub2SHVQ ? psoProjUVMMASub2 : psoProjUVMMA)
                                              : (isSub2SHVQ ? psoProjUVSub2 : psoProjUV))
        uLatScratch.bind(enc, index: 0)
        mlaW.kvBProj.blocks.bind(enc, index: 1)
        attOut.bind(enc, index: 2)
        enc.setBytes(&absP, length: MemoryLayout<AbsorbParams>.stride, index: 3)
        if projUVMMA {
            enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: (rows + 31) / 32, depth: vHeadDim / 128),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: mTiles, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
