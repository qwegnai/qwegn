// QWCoarsePoolIndex.swift — Shared capacity-balanced e-means (arXiv:2611.09501)
// on-GPU CSR coarse pool index (4096 tokens = 1024 pools -> 64 clusters) and
// deterministic coarse-to-fine / exact-flat pool selector for LSA, QSA, and MLA.

import Foundation
import Metal
import QWAlloc
import QWHAL

public typealias QWAttentionSelectionMode = QWCoarsePoolIndex.SelectionMode

public enum QWCoarsePoolIndex {
    public static let macroChunkTokens = 4096
    public static let macroChunkPools = 1024
    public static let clustersPerMacro = 64
    public static let defaultCandidateCap = 16384
    public static let maxBudgetBlocks = 2048

    /// Explicit attention candidate-selection contract (`exactFlat` vs `approximateCoarse`).
    public enum SelectionMode: String, Sendable, Equatable {
        case exactFlat = "exact-flat"
        case approximateCoarse = "approximate-coarse"
    }

    /// Mirrors `EmeansSelParams` in `Sources/QWMetal/include/qw_emeans_coarse.metalh`.
    public struct SelectParams {
        public var T: UInt32
        public var pos0: UInt32
        public var indexDim: UInt32
        public var budgetBlocks: UInt32
        public var windowBlocks: UInt32
        public var hasSink: UInt32
        public var writeCount: UInt32
        public var listStride: UInt32
        public var rowStep: UInt32
        public var maxRow: UInt32
        public var clusterStride: UInt32
        public var probeClusters: UInt32
        public var candCap: UInt32
        public var qChannels: UInt32
        public var idxHeads: UInt32
        public var flags: UInt32

        public init(
            T: Int,
            pos0: Int,
            indexDim: Int,
            budgetBlocks: Int,
            windowBlocks: Int = 0,
            hasSink: Bool = false,
            writeCount: Bool = true,
            listStride: Int,
            rowStep: Int = 1,
            maxRow: Int? = nil,
            clusterStride: Int,
            probeClusters: Int,
            candCap: Int,
            qChannels: Int = 0,
            idxHeads: Int = 0,
            flags: UInt32 = 0
        ) {
            self.T = UInt32(T)
            self.pos0 = UInt32(pos0)
            self.indexDim = UInt32(indexDim)
            self.budgetBlocks = UInt32(budgetBlocks)
            self.windowBlocks = UInt32(windowBlocks)
            self.hasSink = hasSink ? 1 : 0
            self.writeCount = writeCount ? 1 : 0
            self.listStride = UInt32(listStride)
            self.rowStep = UInt32(rowStep)
            self.maxRow = UInt32(maxRow ?? T)
            self.clusterStride = UInt32(clusterStride)
            self.probeClusters = UInt32(probeClusters)
            self.candCap = UInt32(candCap)
            self.qChannels = UInt32(qChannels)
            self.idxHeads = UInt32(idxHeads)
            self.flags = flags
        }
    }

    /// Per-layer CSR coarse index (`centroids` f16, `memberPools` u16, `clusterSpans` u16x2).
    public struct LayerCSR: Sendable {
        public let centroids: QWTensorValue
        public let memberPools: QWTensorValue
        public let clusterSpans: QWTensorValue
        public let maxMacros: Int
        public var maxClusters: Int { maxMacros * QWCoarsePoolIndex.clustersPerMacro }

        public init(centroids: QWTensorValue,
                    memberPools: QWTensorValue,
                    clusterSpans: QWTensorValue,
                    maxMacros: Int) {
            self.centroids = centroids
            self.memberPools = memberPools
            self.clusterSpans = clusterSpans
            self.maxMacros = maxMacros
        }
    }

    /// Shared zero-at-rest selection scratch (`qwg_emeans_finalize` restores zeros).
    public struct Scratch: Sendable {
        public let cScore: QWTensorValue
        public let uHist: QWTensorValue
        public let candState: QWTensorValue
        public let candPools: QWTensorValue
        public let candU: QWTensorValue
        public let maxBatchRows: Int
        public let clusterStride: Int
        public let candCap: Int

        public init(cScore: QWTensorValue,
                    uHist: QWTensorValue,
                    candState: QWTensorValue,
                    candPools: QWTensorValue,
                    candU: QWTensorValue,
                    maxBatchRows: Int,
                    clusterStride: Int,
                    candCap: Int) {
            self.cScore = cScore
            self.uHist = uHist
            self.candState = candState
            self.candPools = candPools
            self.candU = candU
            self.maxBatchRows = maxBatchRows
            self.clusterStride = clusterStride
            self.candCap = candCap
        }
    }

    /// Prewarmed shared pipelines used across all three attention subsystems.
    public struct Pipelines: @unchecked Sendable {
        public let cluster1024: any MTLComputePipelineState
        public let collect: any MTLComputePipelineState
        public let finalize: any MTLComputePipelineState

        public init(hal: QWHal) throws {
            self.cluster1024 = try hal.pipeline("qwg_emeans_cluster_1024")
            self.collect = try hal.pipeline("qwg_emeans_collect")
            self.finalize = try hal.pipeline("qwg_emeans_finalize")
        }
    }

    public static func maxMacros(forMaxContext maxContext: Int) -> Int {
        max(1, (max(4096, maxContext) + macroChunkTokens - 1) / macroChunkTokens)
    }

    public static func csrBytesPerMacro(indexDim: Int) -> (centroids: Int, members: Int, spans: Int, total: Int) {
        let c = clustersPerMacro * indexDim * 2
        let m = macroChunkPools * 2
        let s = clustersPerMacro * 4
        return (c, m, s, c + m + s)
    }

    public static func scratchByteSizes(
        maxBatchRows: Int,
        maxClusters: Int,
        candCap: Int = defaultCandidateCap
    ) -> (cScore: Int, uHist: Int, candState: Int, candPools: Int, candU: Int) {
        let rows = max(1, maxBatchRows)
        let cStride = max(64, maxClusters)
        return (
            cScore: rows * cStride * 4,
            uHist: rows * 512 * 4,
            candState: rows * 2 * 4,
            candPools: rows * candCap * 4,
            candU: rows * candCap * 4
        )
    }

    /// Geometry summary for the largest-position query in `[pos0 ..< pos0 + units * rowStep)`.
    public struct BatchGeometry: Sendable {
        public let totalBlocks: Int
        public let middleEnd: Int
        public let numMacros: Int
        public let numClusters: Int
        public let minProbe: Int
        public let isDense: Bool
        public let shouldPrune: Bool
        public var selectionMode: SelectionMode { shouldPrune ? .approximateCoarse : .exactFlat }
    }

    public static func batchGeometry(
        pos0: Int,
        units: Int,
        rowStep: Int = 1,
        maxRow: Int? = nil,
        budgetBlocks: Int,
        windowBlocks: Int = 0,
        hasSink: Bool = false,
        writeCount: Bool = true,
        probeClusters: Int
    ) -> BatchGeometry {
        let effMaxRow = maxRow ?? units
        let lastRow = rowStep > 1
            ? min((units - 1) * rowStep + rowStep - 1, max(0, effMaxRow - 1))
            : max(0, units - 1)
        let totalBlocks = (pos0 + lastRow + 1) >> 2
        let denseLim = (hasSink ? 1 : 0) + budgetBlocks + windowBlocks
        if totalBlocks <= denseLim {
            return BatchGeometry(totalBlocks: totalBlocks, middleEnd: 0, numMacros: 0,
                                 numClusters: 0, minProbe: 0, isDense: true, shouldPrune: false)
        }
        let middleEnd = totalBlocks - windowBlocks
        let kEff = min(budgetBlocks, middleEnd - (hasSink ? 1 : 0))
        let numMacros = middleEnd >> 10
        let numClusters = numMacros * clustersPerMacro
        let minProbe = writeCount ? max(probeClusters, 64) : max(probeClusters, kEff + 32)
        let prune = (budgetBlocks <= maxBudgetBlocks) && (numClusters >= minProbe * 2)
        return BatchGeometry(totalBlocks: totalBlocks, middleEnd: middleEnd, numMacros: numMacros,
                             numClusters: numClusters, minProbe: minProbe, isDense: false, shouldPrune: prune)
    }

    @inline(__always)
    public static func barrierIfConcurrent(_ enc: any MTLComputeCommandEncoder) {
        if enc.dispatchType == .concurrent {
            enc.memoryBarrier(scope: .buffers)
        }
    }

    /// Clusters `[startMacro ..< startMacro + numMacros)` completed 4096-token chunks on GPU.
    public static func encodeClusterMacros(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        pooledKeys: QWTensorValue,
        csr: LayerCSR,
        startMacro: Int,
        numMacros: Int,
        indexDim: Int
    ) {
        guard numMacros > 0, startMacro >= 0 else { return }
        let clamped = min(numMacros, max(0, csr.maxMacros - startMacro))
        guard clamped > 0 else { return }
        var startU = UInt32(startMacro)
        var countU = UInt32(clamped)
        var dimU = UInt32(indexDim)
        enc.setComputePipelineState(pso)
        pooledKeys.bind(enc, index: 0)
        csr.centroids.bind(enc, index: 1)
        csr.memberPools.bind(enc, index: 2)
        csr.clusterSpans.bind(enc, index: 3)
        enc.setBytes(&startU, length: 4, index: 4)
        enc.setBytes(&countU, length: 4, index: 5)
        enc.setBytes(&dimU, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: clamped, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encodes deterministic single-TG `collect -> scoreCands -> finalize` (with zero-Stage-1 flat bypass
    /// when `params.probeClusters == 0` and on-device exact flat fallback when `totalValid > candCap`).
    public static func encodeHistCollectScoreFinalize(
        _ enc: any MTLComputeCommandEncoder,
        pipelines: Pipelines,
        csr: LayerCSR,
        scratch: Scratch,
        blockList: QWTensorValue,
        blockCount: QWTensorValue,
        params: inout SelectParams,
        numClusters: Int,
        maxMiddlePools: Int,
        encodeScoreCands: (any MTLComputeCommandEncoder, Int, inout SelectParams) -> Void,
        encodeFlatFallback: (any MTLComputeCommandEncoder, inout SelectParams) -> Void
    ) {
        let units = Int(params.T)
        let plen = MemoryLayout<SelectParams>.stride
        let tpg = MTLSize(width: 256, height: 1, depth: 1)
        let minPool = params.hasSink != 0 ? 1 : 0
        let maxPossibleMiddle = max(0, maxMiddlePools - minPool)

        if params.probeClusters == 0 {
            if maxPossibleMiddle > scratch.candCap {
                encodeFlatFallback(enc, &params)
                return
            }
            let maxCand = max(1, min(scratch.candCap, maxPossibleMiddle))
            encodeScoreCands(enc, maxCand, &params)
            barrierIfConcurrent(enc)
            enc.setComputePipelineState(pipelines.finalize)
            blockList.bind(enc, index: 0)
            blockCount.bind(enc, index: 1)
            scratch.uHist.bind(enc, index: 2)
            scratch.candState.bind(enc, index: 3)
            scratch.candPools.bind(enc, index: 4)
            scratch.candU.bind(enc, index: 5)
            enc.setBytes(&params, length: plen, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: units, height: 1, depth: 1), threadsPerThreadgroup: tpg)
            return
        }

        if numClusters > 0 {
            barrierIfConcurrent(enc)
        }
        enc.setComputePipelineState(pipelines.collect)
        csr.memberPools.bind(enc, index: 0)
        csr.clusterSpans.bind(enc, index: 1)
        scratch.cScore.bind(enc, index: 2)
        scratch.uHist.bind(enc, index: 3)
        scratch.candState.bind(enc, index: 4)
        scratch.candPools.bind(enc, index: 5)
        enc.setBytes(&params, length: plen, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: units, depth: 1),
            threadsPerThreadgroup: tpg)
        barrierIfConcurrent(enc)

        let maxCand = max(1, min(scratch.candCap, maxPossibleMiddle))
        encodeScoreCands(enc, maxCand, &params)
        barrierIfConcurrent(enc)

        enc.setComputePipelineState(pipelines.finalize)
        blockList.bind(enc, index: 0)
        blockCount.bind(enc, index: 1)
        scratch.uHist.bind(enc, index: 2)
        scratch.candState.bind(enc, index: 3)
        scratch.candPools.bind(enc, index: 4)
        scratch.candU.bind(enc, index: 5)
        enc.setBytes(&params, length: plen, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: units, height: 1, depth: 1),
            threadsPerThreadgroup: tpg)

        if maxPossibleMiddle > scratch.candCap {
            barrierIfConcurrent(enc)
            encodeFlatFallback(enc, &params)
        }
    }
}
