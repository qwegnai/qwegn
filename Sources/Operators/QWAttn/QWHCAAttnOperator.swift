// QWHCAAttnOperator.swift — Hybrid Compressed Attention (HCA) GPU operator:
// per-HCA-layer SWA ring, 128:1 channel-wise softmax compressor cache, optional
// Weight-SVD LSA + e-means coarse pool index (>262K tokens), Split-K decode MQA,
// fused 8x8 MMA VHA + inverse YaRN RoPE, and 3D-batched grouped low-rank wo_a.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import QWGEMM

public final class QWHCAAttnOperator: @unchecked Sendable {
    public struct LayerWeights: Sendable {
        public let wFusedIn: QWWeightView?
        public let qNormW: QWTensorValue
        public let kvNormW: QWTensorValue
        public let compNormW: QWTensorValue
        public let ape: QWTensorValue
        public let wQUpGate: QWWeightView?
        public let woA: QWWeightView?
        public let woARawF16: QWTensorValue?
        public let woB: QWWeightView?
        public let vhaMix: QWTensorValue?
        public let attnSink: QWTensorValue?
        public let lsaWQIdx: QWTensorValue?
        public let lsaWKIdx: QWTensorValue?

        public init(wFusedIn: QWWeightView? = nil,
                    qNormW: QWTensorValue,
                    kvNormW: QWTensorValue,
                    compNormW: QWTensorValue,
                    ape: QWTensorValue,
                    wQUpGate: QWWeightView? = nil,
                    woA: QWWeightView? = nil,
                    woARawF16: QWTensorValue? = nil,
                    woB: QWWeightView? = nil,
                    vhaMix: QWTensorValue? = nil,
                    attnSink: QWTensorValue? = nil,
                    lsaWQIdx: QWTensorValue? = nil,
                    lsaWKIdx: QWTensorValue? = nil) {
            self.wFusedIn = wFusedIn
            self.qNormW = qNormW
            self.kvNormW = kvNormW
            self.compNormW = compNormW
            self.ape = ape
            self.wQUpGate = wQUpGate
            self.woA = woA
            self.woARawF16 = woARawF16
            self.woB = woB
            self.vhaMix = vhaMix
            self.attnSink = attnSink
            self.lsaWQIdx = lsaWQIdx
            self.lsaWKIdx = lsaWKIdx
        }
    }

    public struct LayerState: Sendable {
        public let hcaLayerIndex: Int
        public let swaKV: QWTensorValue
        public let compKvPend: QWTensorValue
        public let compScorePend: QWTensorValue
        public let compKV: QWTensorValue
        public let pooledKeys: QWTensorValue?
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
    }

    public static let splitMaxRows = 15
    public static let maxSelectBatchRows = 32
    public static let emeansProbeClusters = 128
    public static let emeansCandCap = 16384

    public let geometry: QWModelGeometry
    public let hcaConfig: QWModelDescriptor.HCAConfig
    public let lsaConfig: QWLSAConfig
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let qHeads: Int
    public let headDim: Int
    public let nopeDim: Int
    public let ropeDim: Int
    public let qLoraRank: Int
    public let oGroups: Int
    public let oLoraRank: Int
    public let groupIn: Int
    public let compressRatio: Int
    public let poolShift: Int
    public let slidingWindow: Int
    public let indexDim: Int
    public let lsaBudgetBlocks: Int
    public let compBudget: Int
    public let listStride: Int
    public let ringCap: Int
    public let maxCompBlocks: Int
    public let selectBatchRows: Int
    public let inStride: Int
    public let qUpStride: Int
    public let oGateOffset: Int
    public let zOutStride: Int
    public let ropeMscale: Float
    public let attnSplits: Int
    public let lsaAllocated: Bool
    public var emeansCoarseEnabled: Bool

    public let region: QWMRegion
    public let layers: [LayerState]
    public weak var stateOwner: AnyObject? {
        didSet {
            if stateOwner !== oldValue { resetCoarseWatermarks() }
        }
    }

    public let fusedInScratch: QWTensorValue
    public let qrScratch: QWTensorValue
    public let qUpGateScratch: QWTensorValue
    public let attScratch: QWTensorValue
    public let zVhaScratch: QWTensorValue
    public let zGatedScratch: QWTensorValue
    public let invFreqBuf: QWTensorValue
    public let zeroSinkBuf: QWTensorValue
    internal let partOScratch: QWTensorValue
    internal let partMLScratch: QWTensorValue
    public let qIdxScratch: QWTensorValue?
    public let blockListScratch: QWTensorValue?
    public let blockCountScratch: QWTensorValue?
    internal let coarseScratch: QWCoarsePoolIndex.Scratch?

    internal let hal: QWHal
    internal let psoInPrep: any MTLComputePipelineState
    internal let psoCompressBlocks: any MTLComputePipelineState
    internal let psoQPrep: any MTLComputePipelineState
    internal let psoLsaQPrep: any MTLComputePipelineState
    internal let psoAttnMMA: any MTLComputePipelineState
    internal let psoAttnNAX: (any MTLComputePipelineState)?
    internal let psoAttnCombine: any MTLComputePipelineState
    internal let psoVhaInvRope: any MTLComputePipelineState
    internal let psoGroupedWoAGemv: any MTLComputePipelineState
    internal let psoGroupedWoAMMA: any MTLComputePipelineState
    internal let coarsePipelines: QWCoarsePoolIndex.Pipelines?
    internal let psoEmeansLsa: (
        scoreCoarse: any MTLComputePipelineState,
        scoreCands: any MTLComputePipelineState,
        flatFallback: any MTLComputePipelineState
    )?
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

    internal struct InPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qLoraRank: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var compressRatio: UInt32
        var ringMask: UInt32
        var rmsEps: Float
        var mscale: Float
    }

    internal struct CompressParams {
        var startBlock: UInt32
        var numBlocks: UInt32
        var compressRatio: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var ringMask: UInt32
        var indexDim: UInt32
        var rmsEps: Float
        var mscale: Float
        var hasLSA: UInt32
    }

    internal struct QPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var qUpStride: UInt32
        var indexDim: UInt32
        var rmsEps: Float
        var mscale: Float
    }

    internal struct AttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var qUpStride: UInt32
        var compressRatio: UInt32
        var slidingWindow: UInt32
        var ringMask: UInt32
        var compBudget: UInt32
        var listStride: UInt32
        var useBlockList: UInt32
        var hasSink: UInt32
    }

    internal struct VhaInvRopeParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var invMscale: Float
        var hasVHA: UInt32
    }

    internal struct GroupedWoAParams {
        var T: UInt32
        var oGroups: UInt32
        var oLoraRank: UInt32
        var groupIn: UInt32
        var qUpStride: UInt32
        var oGateOffset: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    public static func makeIfEnabled(hal: QWHal,
                                     geometry: QWModelGeometry,
                                     hcaLayers: [Int],
                                     maxContext: Int,
                                     maxPrefillChunk: Int = 512,
                                     lsaConfig: QWLSAConfig = QWLSAConfig(),
                                     indexDim: Int = QWLSAConfig.defaultIndexDim,
                                     extraHcaLayers: Int = 0) throws -> QWHCAAttnOperator? {
        guard geometry.hcaConfig != nil, !hcaLayers.isEmpty else { return nil }
        return try QWHCAAttnOperator(
            hal: hal,
            geometry: geometry,
            numHcaLayers: hcaLayers.count + max(0, extraHcaLayers),
            maxContext: maxContext,
            maxPrefillChunk: maxPrefillChunk,
            lsaConfig: lsaConfig,
            indexDim: indexDim
        )
    }

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                numHcaLayers: Int,
                maxContext: Int,
                maxPrefillChunk: Int = 512,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                lsaBudgetBlocksOverride: Int? = nil) throws {
        guard let hca = geometry.hcaConfig else {
            throw QWAllocError.outOfSpace(region: "hca-cache", asked: 0, free: 0)
        }
        self.hal = hal
        self.geometry = geometry
        self.hcaConfig = hca
        self.lsaConfig = lsaConfig
        self.maxContext = max(64, maxContext)
        self.maxPrefillChunk = max(16, maxPrefillChunk)
        self.qHeads = hca.qHeads
        self.headDim = hca.headDim
        self.nopeDim = hca.qkNopeHeadDim
        self.ropeDim = hca.qkRopeHeadDim
        self.qLoraRank = hca.qLoraRank
        self.oGroups = max(1, hca.oGroups)
        self.oLoraRank = hca.oLoraRank
        self.groupIn = (self.qHeads / self.oGroups) * self.headDim
        self.compressRatio = max(1, hca.defaultCompressRatio)
        self.poolShift = self.compressRatio > 1 ? self.compressRatio.trailingZeroBitCount : 0
        self.slidingWindow = max(1, hca.slidingWindow)
        self.indexDim = max(16, min(64, indexDim))

        let defaultBlkBudget = min(QWCoarsePoolIndex.maxBudgetBlocks,
                                   max(1, lsaConfig.isEnabled ? lsaConfig.budgetBlocks : 2048))
        let blkBudget = min(QWCoarsePoolIndex.maxBudgetBlocks,
                            max(1, lsaBudgetBlocksOverride ?? defaultBlkBudget))
        self.lsaBudgetBlocks = blkBudget
        self.compBudget = 1 + blkBudget
        self.listStride = max(64, ((self.compBudget + 63) & ~63))

        let minRing = max(self.slidingWindow, self.compressRatio) + max(32, self.maxPrefillChunk)
        var rc = 256
        while rc < minRing { rc <<= 1 }
        self.ringCap = rc
        self.maxCompBlocks = max(16, (self.maxContext + self.compressRatio - 1) / self.compressRatio)
        self.selectBatchRows = min(self.maxPrefillChunk, Self.maxSelectBatchRows)
        self.inStride = self.qLoraRank + 3 * self.headDim
        self.oGateOffset = self.qHeads * self.headDim
        self.qUpStride = self.oGateOffset + self.oGroups * self.oLoraRank
        self.zOutStride = self.oGroups * self.oLoraRank

        let (invFreqArr, mscale) = QWModelDescriptor.computeRopeInvFreq(
            rotaryDim: self.ropeDim,
            base: hca.compressRopeTheta,
            yarn: hca.yarn
        )
        self.ropeMscale = mscale

        let splits = Int(ProcessInfo.processInfo.environment["QW_HCA_ATTN_SPLIT"] ?? "") ?? 8
        self.attnSplits = (self.headDim == 512 && self.qHeads % 16 == 0) ? max(1, min(32, splits)) : 1

        self.psoInPrep = try hal.pipeline("qwg_hca_in_prep")
        self.psoCompressBlocks = try hal.pipeline("qwg_hca_compress_blocks")
        self.psoQPrep = try hal.pipeline("qwg_hca_q_prep")
        self.psoLsaQPrep = try hal.pipeline("qwg_hca_lsa_q_prep")
        self.psoAttnMMA = try hal.pipeline("qwg_hca_attn_mma")
        self.psoAttnNAX = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_hca_attn_nax") : nil
        self.psoAttnCombine = try hal.pipeline("qwg_hca_attn_combine")
        self.psoVhaInvRope = try hal.pipeline("qwg_hca_vha_invrope")
        self.psoGroupedWoAGemv = try hal.pipeline("qwg_hca_grouped_wo_a_gemv")
        self.psoGroupedWoAMMA = try hal.pipeline("qwg_hca_grouped_wo_a_mma")

        let needLsa = (lsaConfig.mode != .off)
            && (self.compressRatio.nonzeroBitCount == 1)
            && (self.maxCompBlocks > self.compBudget)
        self.lsaAllocated = needLsa
        if needLsa {
            self.coarsePipelines = try QWCoarsePoolIndex.Pipelines(hal: hal)
            self.psoEmeansLsa = (
                try hal.pipeline("qwg_emeans_lsa_score_coarse"),
                try hal.pipeline("qwg_emeans_lsa_score_cands"),
                try hal.pipeline("qwg_emeans_lsa_flat_fallback")
            )
        } else {
            self.coarsePipelines = nil
            self.psoEmeansLsa = nil
        }
        if let rawEnv = ProcessInfo.processInfo.environment["QW_HCA_EMEANS"]?.lowercased() {
            self.emeansCoarseEnabled = needLsa && rawEnv != "0" && rawEnv != "false" && rawEnv != "off"
        } else {
            self.emeansCoarseEnabled = needLsa
        }
        self.clusteredMacros = Array(repeating: 0, count: numHcaLayers)

        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }

        let maxMacros = needLsa
            ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext, poolShift: self.poolShift)
            : 0
        let csrSizes = QWCoarsePoolIndex.csrBytesPerMacro(indexDim: self.indexDim)
        let paddedCompBlocks = (self.maxCompBlocks + 7) & ~7

        struct LayerOffs {
            let swa: Int
            let kvPend: Int
            let scPend: Int
            let comp: Int
            let pooled: Int
            let cCent: Int
            let cMem: Int
            let cSpan: Int
        }
        var layerOffs: [LayerOffs] = []
        layerOffs.reserveCapacity(numHcaLayers)
        for _ in 0..<numHcaLayers {
            let swaOff = reserve(self.ringCap * self.headDim * 2)
            let kvPendOff = reserve(self.ringCap * self.headDim * 4)
            let scPendOff = reserve(self.ringCap * self.headDim * 4)
            let compOff = reserve(paddedCompBlocks * self.headDim * 2)
            let pOff = needLsa ? reserve(paddedCompBlocks * self.indexDim * 2) : 0
            let cCent = needLsa ? reserve(maxMacros * csrSizes.centroids) : 0
            let cMem = needLsa ? reserve(maxMacros * csrSizes.members) : 0
            let cSpan = needLsa ? reserve(maxMacros * csrSizes.spans) : 0
            layerOffs.append(LayerOffs(
                swa: swaOff, kvPend: kvPendOff, scPend: scPendOff, comp: compOff,
                pooled: pOff, cCent: cCent, cMem: cMem, cSpan: cSpan
            ))
        }

        let maxT = self.maxPrefillChunk
        let fusedInOff = reserve(maxT * self.inStride * 2)
        let qrOff = reserve(maxT * self.qLoraRank * 2)
        let qUpOff = reserve(maxT * self.qUpStride * 2)
        let attOff = reserve(maxT * self.qHeads * self.headDim * 2)
        let zVhaOff = reserve(maxT * self.qHeads * self.headDim * 2)
        let zGatedOff = reserve(maxT * self.zOutStride * 2)
        let invFreqCount = max(1, invFreqArr.count)
        let invFreqOff = reserve(invFreqCount * 4)
        let zeroSinkCount = max(16, self.qHeads)
        let zeroSinkOff = reserve(zeroSinkCount * 4)
        let splitRows = min(maxT, Self.splitMaxRows) * max(1, self.attnSplits) * self.qHeads
        let partOOff = reserve(self.attnSplits > 1 ? splitRows * self.headDim * 2 : 0)
        let partMLOff = reserve(self.attnSplits > 1 ? splitRows * 8 : 0)

        let qIdxOff = needLsa ? reserve(maxT * self.indexDim * 2) : 0
        let listOff = needLsa ? reserve(maxT * self.listStride * 4) : 0
        let countOff = needLsa ? reserve(maxT * 4) : 0
        let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
        let cScSizes = QWCoarsePoolIndex.scratchByteSizes(
            maxBatchRows: self.selectBatchRows, maxClusters: maxClusters, candCap: Self.emeansCandCap)
        let cScoreOff = needLsa ? reserve(cScSizes.cScore) : 0
        let cHistOff = needLsa ? reserve(cScSizes.uHist) : 0
        let cStateOff = needLsa ? reserve(cScSizes.candState) : 0
        let cPoolsOff = needLsa ? reserve(cScSizes.candPools) : 0
        let cUOff = needLsa ? reserve(cScSizes.candU) : 0

        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "hca-cache", asked: totalBytes, free: 0)
        }
        buf.label = "hca-cache"
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
        builtLayers.reserveCapacity(numHcaLayers)
        for i in 0..<numHcaLayers {
            let o = layerOffs[i]
            let csr: QWCoarsePoolIndex.LayerCSR? = needLsa ? QWCoarsePoolIndex.LayerCSR(
                centroids: QWTensorValue(region: reg, offset: o.cCent, shape: QWTensorShape([maxClusters, self.indexDim], .f16)),
                memberPools: QWTensorValue(region: reg, offset: o.cMem, shape: QWTensorShape([maxMacros * 512], .i32)),
                clusterSpans: QWTensorValue(region: reg, offset: o.cSpan, shape: QWTensorShape([maxClusters], .i32)),
                maxMacros: maxMacros
            ) : nil
            let pKeys: QWTensorValue? = needLsa
                ? QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([paddedCompBlocks, self.indexDim], .f16))
                : nil
            builtLayers.append(LayerState(
                hcaLayerIndex: i,
                swaKV: QWTensorValue(region: reg, offset: o.swa, shape: QWTensorShape([self.ringCap, self.headDim], .f16)),
                compKvPend: QWTensorValue(region: reg, offset: o.kvPend, shape: QWTensorShape([self.ringCap, self.headDim], .f32)),
                compScorePend: QWTensorValue(region: reg, offset: o.scPend, shape: QWTensorShape([self.ringCap, self.headDim], .f32)),
                compKV: QWTensorValue(region: reg, offset: o.comp, shape: QWTensorShape([paddedCompBlocks, self.headDim], .f16)),
                pooledKeys: pKeys,
                coarseCSR: csr
            ))
        }
        self.layers = builtLayers

        self.fusedInScratch = QWTensorValue(region: reg, offset: fusedInOff, shape: QWTensorShape([maxT, self.inStride], .f16))
        self.qrScratch = QWTensorValue(region: reg, offset: qrOff, shape: QWTensorShape([maxT, self.qLoraRank], .f16))
        self.qUpGateScratch = QWTensorValue(region: reg, offset: qUpOff, shape: QWTensorShape([maxT, self.qUpStride], .f16))
        self.attScratch = QWTensorValue(region: reg, offset: attOff, shape: QWTensorShape([maxT, self.qHeads * self.headDim], .f16))
        self.zVhaScratch = QWTensorValue(region: reg, offset: zVhaOff, shape: QWTensorShape([maxT, self.qHeads * self.headDim], .f16))
        self.zGatedScratch = QWTensorValue(region: reg, offset: zGatedOff, shape: QWTensorShape([maxT, self.zOutStride], .f16))
        self.invFreqBuf = QWTensorValue(region: reg, offset: invFreqOff, shape: QWTensorShape([invFreqCount], .f32))
        self.zeroSinkBuf = QWTensorValue(region: reg, offset: zeroSinkOff, shape: QWTensorShape([zeroSinkCount], .f32))
        self.partOScratch = QWTensorValue(region: reg, offset: partOOff, shape: QWTensorShape([max(1, splitRows), self.headDim], .f16))
        self.partMLScratch = QWTensorValue(region: reg, offset: partMLOff, shape: QWTensorShape([max(1, splitRows), 2], .f32))

        if needLsa {
            self.qIdxScratch = QWTensorValue(region: reg, offset: qIdxOff, shape: QWTensorShape([maxT, self.indexDim], .f16))
            self.blockListScratch = QWTensorValue(region: reg, offset: listOff, shape: QWTensorShape([maxT, self.listStride], .i32))
            self.blockCountScratch = QWTensorValue(region: reg, offset: countOff, shape: QWTensorShape([maxT], .i32))
            self.coarseScratch = QWCoarsePoolIndex.Scratch(
                cScore: QWTensorValue(region: reg, offset: cScoreOff, shape: QWTensorShape([self.selectBatchRows, maxClusters], .i32)),
                uHist: QWTensorValue(region: reg, offset: cHistOff, shape: QWTensorShape([self.selectBatchRows, 512], .i32)),
                candState: QWTensorValue(region: reg, offset: cStateOff, shape: QWTensorShape([self.selectBatchRows * 2], .i32)),
                candPools: QWTensorValue(region: reg, offset: cPoolsOff, shape: QWTensorShape([self.selectBatchRows, Self.emeansCandCap], .i32)),
                candU: QWTensorValue(region: reg, offset: cUOff, shape: QWTensorShape([self.selectBatchRows, Self.emeansCandCap], .i32)),
                maxBatchRows: self.selectBatchRows,
                clusterStride: maxClusters,
                candCap: Self.emeansCandCap
            )
        } else {
            self.qIdxScratch = nil
            self.blockListScratch = nil
            self.blockCountScratch = nil
            self.coarseScratch = nil
        }
    }

    public func encodeInPrepAndCompress(_ enc: any MTLComputeCommandEncoder,
                                        hcaW: LayerWeights,
                                        hcaLayer: Int,
                                        pos0: Int,
                                        rows: Int) {
        guard hcaLayer >= 0, hcaLayer < layers.count, rows > 0 else { return }
        let ls = layers[hcaLayer]
        var inP = InPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qLoraRank: UInt32(qLoraRank),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            compressRatio: UInt32(compressRatio),
            ringMask: UInt32(ringCap - 1),
            rmsEps: geometry.rmsEps,
            mscale: ropeMscale
        )
        enc.setComputePipelineState(psoInPrep)
        fusedInScratch.bind(enc, index: 0)
        hcaW.qNormW.bind(enc, index: 1)
        hcaW.kvNormW.bind(enc, index: 2)
        hcaW.ape.bind(enc, index: 3)
        invFreqBuf.bind(enc, index: 4)
        qrScratch.bind(enc, index: 5)
        ls.swaKV.bind(enc, index: 6)
        ls.compKvPend.bind(enc, index: 7)
        ls.compScorePend.bind(enc, index: 8)
        enc.setBytes(&inP, length: MemoryLayout<InPrepParams>.stride, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let startBlock = pos0 / compressRatio
        let endBlock = (pos0 + rows) / compressRatio
        let numBlocks = endBlock - startBlock
        let hasLsa = (lsaAllocated && hcaW.lsaWKIdx != nil && ls.pooledKeys != nil)
        if numBlocks > 0 {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            var cP = CompressParams(
                startBlock: UInt32(startBlock),
                numBlocks: UInt32(numBlocks),
                compressRatio: UInt32(compressRatio),
                headDim: UInt32(headDim),
                nopeDim: UInt32(nopeDim),
                ropeDim: UInt32(ropeDim),
                ringMask: UInt32(ringCap - 1),
                indexDim: UInt32(indexDim),
                rmsEps: geometry.rmsEps,
                mscale: ropeMscale,
                hasLSA: hasLsa ? 1 : 0
            )
            enc.setComputePipelineState(psoCompressBlocks)
            ls.compKvPend.bind(enc, index: 0)
            ls.compScorePend.bind(enc, index: 1)
            hcaW.compNormW.bind(enc, index: 2)
            invFreqBuf.bind(enc, index: 3)
            (hcaW.lsaWKIdx ?? ls.compKV).bind(enc, index: 4)
            ls.compKV.bind(enc, index: 5)
            (ls.pooledKeys ?? ls.compKV).bind(enc, index: 6)
            enc.setBytes(&cP, length: MemoryLayout<CompressParams>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if hasLsa {
            updateCoarseClusters(enc, ls: ls, hcaLayer: hcaLayer, pos0: pos0, rows: rows)
        }
    }

    public func encodeQPrepAttendVHAAndWoA(_ enc: any MTLComputeCommandEncoder,
                                           hcaW: LayerWeights,
                                           zGatedOut: QWTensorValue,
                                           hcaLayer: Int,
                                           pos0: Int,
                                           rows: Int,
                                           skipExitRotation: Bool = false,
                                           stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        guard hcaLayer >= 0, hcaLayer < layers.count, rows > 0 else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[hcaLayer]

        mark("hca.q_prep")
        var qP = QPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            qUpStride: UInt32(qUpStride),
            indexDim: UInt32(indexDim),
            rmsEps: geometry.rmsEps,
            mscale: ropeMscale
        )
        enc.setComputePipelineState(psoQPrep)
        qUpGateScratch.bind(enc, index: 0)
        invFreqBuf.bind(enc, index: 1)
        enc.setBytes(&qP, length: MemoryLayout<QPrepParams>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let maxTotalComp = (pos0 + rows) / compressRatio
        let useSparseLsa = lsaAllocated
            && maxTotalComp > compBudget
            && hcaW.lsaWQIdx != nil
            && hcaW.lsaWKIdx != nil
            && ls.pooledKeys != nil
        if useSparseLsa {
            mark("hca.lsa_select")
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            encodeLSASelection(enc, ls: ls, wqIdx: hcaW.lsaWQIdx!, qPrepParams: &qP, pos0: pos0, rows: rows)
        }

        mark("hca.attn")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        let sinkTensor = hcaW.attnSink ?? zeroSinkBuf
        var attP = AttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            qUpStride: UInt32(qUpStride),
            compressRatio: UInt32(compressRatio),
            slidingWindow: UInt32(slidingWindow),
            ringMask: UInt32(ringCap - 1),
            compBudget: UInt32(compBudget),
            listStride: UInt32(listStride),
            useBlockList: useSparseLsa ? 1 : 0,
            hasSink: hcaW.attnSink != nil ? 1 : 0
        )
        let minTotalComp = (pos0 + 1) / compressRatio
        let approxSlots = min(useSparseLsa ? compBudget : minTotalComp, minTotalComp)
            + min(pos0 + 1, slidingWindow)
        let useSplit = attnSplits > 1 && rows <= Self.splitMaxRows && approxSlots >= 64
        let nSplit = useSplit ? attnSplits : 1
        let useNAX = psoAttnNAX != nil && headDim == 512 && qHeads % 16 == 0
        if let naxPSO = psoAttnNAX, useNAX {
            enc.setComputePipelineState(naxPSO)
        } else {
            enc.setComputePipelineState(psoAttnMMA)
        }
        qUpGateScratch.bind(enc, index: 0)
        ls.compKV.bind(enc, index: 1)
        ls.swaKV.bind(enc, index: 2)
        (blockListScratch ?? zeroSinkBuf).bind(enc, index: 3)
        (blockCountScratch ?? zeroSinkBuf).bind(enc, index: 4)
        sinkTensor.bind(enc, index: 5)
        (useSplit ? partOScratch : attScratch).bind(enc, index: 6)
        partMLScratch.bind(enc, index: 7)
        enc.setBytes(&attP, length: MemoryLayout<AttnParams>.stride, index: 8)
        let hgPerTG = useNAX ? 32 : 16
        let tgThreads = useNAX ? 256 : 128
        enc.dispatchThreadgroups(MTLSize(width: rows, height: (qHeads + hgPerTG - 1) / hgPerTG, depth: nSplit),
                                 threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        if useSplit {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            var nSplitU = UInt32(nSplit)
            enc.setComputePipelineState(psoAttnCombine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            sinkTensor.bind(enc, index: 2)
            attScratch.bind(enc, index: 3)
            enc.setBytes(&attP, length: MemoryLayout<AttnParams>.stride, index: 4)
            enc.setBytes(&nSplitU, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        mark("hca.vha_invrope")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        let isRawWoA = (hcaW.woA == nil && hcaW.woARawF16 != nil)
        let skipInterFwht = isRawWoA || (hcaW.woA != nil && !hcaW.woA!.isRotated)
        var vhaP = VhaInvRopeParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            invMscale: ropeMscale > 0 ? (1.0 / ropeMscale) : 1.0,
            hasVHA: (hcaW.vhaMix != nil ? 1 : 0) | (skipInterFwht ? 0x10000 : 0)
        )
        enc.setComputePipelineState(psoVhaInvRope)
        attScratch.bind(enc, index: 0)
        (hcaW.vhaMix ?? attScratch).bind(enc, index: 1)
        invFreqBuf.bind(enc, index: 2)
        zVhaScratch.bind(enc, index: 3)
        enc.setBytes(&vhaP, length: MemoryLayout<VhaInvRopeParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: headDim / 32, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        mark("hca.grouped_wo_a")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        encodeGroupedWoA(enc, hcaW: hcaW, zGatedOut: zGatedOut, rows: rows, skipExitRotation: skipExitRotation)
    }
}
