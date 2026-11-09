// QWLSAOperator.swift — Latent Sparse Attention (arXiv:2610.42147) GPU operator
// for QWEngineDriver (all-attention, hybrid GDN+Attention, and looped mHC models).
//
// Manages per-full-attention-layer LSA projection matrices (`wq_idx`, `wk_idx`),
// pooled 4-token key cache, dynamic 512-token Ward cluster centroids/members/sizes,
// and shared block-selection scratch in a single preallocated Metal region
// (preserving zero steady-state allocations).
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant

public final class QWLSAOperator: @unchecked Sendable {
    public struct LayerWeights: Sendable {
        public let physicalLayer: Int
        public let wqIdxF16: [UInt16]
        public let wkIdxF16: [UInt16]
        public init(physicalLayer: Int, wqIdxF16: [UInt16], wkIdxF16: [UInt16]) {
            self.physicalLayer = physicalLayer
            self.wqIdxF16 = wqIdxF16
            self.wkIdxF16 = wkIdxF16
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let wqIdx: QWTensorValue
        public let wkIdx: QWTensorValue
        public let kPending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let centroids: QWTensorValue
        public let bucketMembers: QWTensorValue
        public let bucketSizes: QWTensorValue
    }

    public static let maxSelectBatchRows = 32

    public let config: QWLSAConfig
    public let geometry: QWModelGeometry
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let indexDim: Int
    public let sliceStart: Int
    public let sliceDim: Int
    public let inputDim: Int
    public let ringCap: Int
    public let maxBlocks: Int
    public let maxSuperblocks: Int
    public let maxBuckets: Int
    public let listStride: Int
    public let region: QWMRegion
    public let layers: [LayerState]
    /// Session whose indexer pools and Ward clusters the layers currently hold.
    public weak var stateOwner: AnyObject?

    public let qIdxScratch: QWTensorValue
    public let scoreScratch: QWTensorValue
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue

    private let hal: QWHal
    private let psoIndexPrep: any MTLComputePipelineState
    private let psoPoolKeys: any MTLComputePipelineState
    private let psoWardCluster: any MTLComputePipelineState
    private let psoSelectBlocks: any MTLComputePipelineState
    private let psoSelectScore: any MTLComputePipelineState
    private let psoSelectHist2: any MTLComputePipelineState
    private let psoSelectExpand: any MTLComputePipelineState
    private let psoSelectFinalize: any MTLComputePipelineState

    /// Parallel selector scratch (zero at rest; restored by `qwg_lsa_select_finalize`).
    private let selCScore: QWTensorValue
    private let selHist: QWTensorValue
    private let selCount: QWTensorValue
    private let selMarks: QWTensorValue
    private let selCandM: QWTensorValue
    private let selCandU: QWTensorValue
    private let selMarkStride: Int
    static let selectCandidateCap = 16384
    static let deltaMagic: UInt64 = 0x31544C4441534C51 // "QLSADLT1"
    /// Route centroid-filter-regime selections through the multi-threadgroup selector
    /// (bit-identical to `qwg_lsa_select_blocks`). Disabled only by parity tests.
    var useParallelSelect = true
    var pinnedBaseData: Data? = nil
    var pinnedBaseFullPosition: Int = 0
    var pinnedBasePos512: Int = 0
    var residentPos512: Int = 0

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                config: QWLSAConfig,
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                layerWeights: [LayerWeights],
                maxContext: Int,
                maxPrefillChunk: Int = 512) throws {
        self.hal = hal
        self.geometry = geometry
        self.config = config
        self.maxContext = max(512, maxContext)
        self.maxPrefillChunk = max(32, maxPrefillChunk)
        self.indexDim = indexDim
        let inDim = geometry.lsaInputDim
        self.sliceStart = geometry.lsaHeadSliceStart
        self.sliceDim = geometry.lsaHeadSliceDim
        self.inputDim = inDim

        var rc = 4096
        while rc < self.maxPrefillChunk * 2 { rc <<= 1 }
        let ringCap = rc
        let maxBlocks = max(128, (self.maxContext + 3) / 4)
        let maxSuperblocks = max(1, (self.maxContext + 511) / 512)
        let maxBuckets = maxSuperblocks * 64
        let listStride = max(512, config.maxTotalSelectedBlocks)
        self.ringCap = ringCap
        self.maxBlocks = maxBlocks
        self.maxSuperblocks = maxSuperblocks
        self.maxBuckets = maxBuckets
        self.listStride = listStride

        self.psoIndexPrep = try hal.pipeline("qwg_lsa_index_prep")
        self.psoPoolKeys = try hal.pipeline("qwg_lsa_pool_keys")
        self.psoWardCluster = try hal.pipeline("qwg_lsa_ward_cluster_512")
        self.psoSelectBlocks = try hal.pipeline("qwg_lsa_select_blocks")
        self.psoSelectScore = try hal.pipeline("qwg_lsa_select_score")
        self.psoSelectHist2 = try hal.pipeline("qwg_lsa_select_hist2")
        self.psoSelectExpand = try hal.pipeline("qwg_lsa_select_expand")
        self.psoSelectFinalize = try hal.pipeline("qwg_lsa_select_finalize")

        let nAttnLayers = layerWeights.count
        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }

        struct LayerOffsets {
            let wqOff: Int
            let wkOff: Int
            let kPendOff: Int
            let pooledOff: Int
            let centOff: Int
            let memOff: Int
            let sizeOff: Int
        }

        var layerOffsets: [LayerOffsets] = []
        layerOffsets.reserveCapacity(nAttnLayers)
        for _ in 0..<nAttnLayers {
            let wqOff = reserve(indexDim * inDim * 2)
            let wkOff = reserve(indexDim * inDim * 2)
            let kPendOff = reserve(ringCap * indexDim * 2)
            let pooledOff = reserve(maxBlocks * indexDim * 2)
            let centOff = reserve(maxBuckets * indexDim * 2)
            let memOff = reserve(maxBuckets * 4)
            let sizeOff = reserve(maxBuckets)
            layerOffsets.append(LayerOffsets(
                wqOff: wqOff, wkOff: wkOff, kPendOff: kPendOff,
                pooledOff: pooledOff, centOff: centOff, memOff: memOff, sizeOff: sizeOff
            ))
        }

        let selectBatch = min(self.maxPrefillChunk, Self.maxSelectBatchRows)
        let qIdxOff = reserve(self.maxPrefillChunk * indexDim * 2)
        let scoreOff = reserve(selectBatch * maxBlocks * 4)
        let listOff = reserve(self.maxPrefillChunk * listStride * 4)
        let countOff = reserve(self.maxPrefillChunk * 4)
        let markStride = (maxBlocks + 31) / 32
        let candCap = Self.selectCandidateCap
        let selCScoreOff = reserve(selectBatch * maxBuckets * 4)
        let selHistOff = reserve(selectBatch * 512 * 4)
        let selCountOff = reserve(selectBatch * 4)
        let selMarksOff = reserve(selectBatch * markStride * 4)
        let selCandMOff = reserve(selectBatch * candCap * 4)
        let selCandUOff = reserve(selectBatch * candCap * 4)

        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "lsa-cache", asked: totalBytes, free: 0)
        }
        buf.label = "lsa-cache"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(nAttnLayers)
        let basePtr = reg.pointer
        let wBytes = indexDim * inDim * 2
        for i in 0..<nAttnLayers {
            let offs = layerOffsets[i]
            let lw = layerWeights[i]
            lw.wqIdxF16.withUnsafeBytes { src in
                if let ptr = src.baseAddress {
                    memcpy(basePtr.advanced(by: offs.wqOff), ptr, min(src.count, wBytes))
                }
            }
            lw.wkIdxF16.withUnsafeBytes { src in
                if let ptr = src.baseAddress {
                    memcpy(basePtr.advanced(by: offs.wkOff), ptr, min(src.count, wBytes))
                }
            }
            builtLayers.append(LayerState(
                fullAttentionIndex: i,
                wqIdx: QWTensorValue(region: reg, offset: offs.wqOff, shape: QWTensorShape([indexDim, inDim], .f16)),
                wkIdx: QWTensorValue(region: reg, offset: offs.wkOff, shape: QWTensorShape([indexDim, inDim], .f16)),
                kPending: QWTensorValue(region: reg, offset: offs.kPendOff, shape: QWTensorShape([ringCap, indexDim], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: offs.pooledOff, shape: QWTensorShape([maxBlocks, indexDim], .f16)),
                centroids: QWTensorValue(region: reg, offset: offs.centOff, shape: QWTensorShape([maxBuckets, indexDim], .f16)),
                bucketMembers: QWTensorValue(region: reg, offset: offs.memOff, shape: QWTensorShape([maxBuckets], .i32)),
                bucketSizes: QWTensorValue(region: reg, offset: offs.sizeOff, shape: QWTensorShape([max(1, maxBuckets / 4)], .i32))
            ))
        }
        self.layers = builtLayers

        self.qIdxScratch = QWTensorValue(
            region: reg, offset: qIdxOff,
            shape: QWTensorShape([self.maxPrefillChunk, indexDim], .f16))
        self.scoreScratch = QWTensorValue(
            region: reg, offset: scoreOff,
            shape: QWTensorShape([selectBatch, maxBlocks], .i32))
        self.blockListScratch = QWTensorValue(
            region: reg, offset: listOff,
            shape: QWTensorShape([self.maxPrefillChunk, listStride], .i32))
        self.blockCountScratch = QWTensorValue(
            region: reg, offset: countOff,
            shape: QWTensorShape([self.maxPrefillChunk], .i32))
        self.selMarkStride = markStride
        self.selCScore = QWTensorValue(region: reg, offset: selCScoreOff,
                                       shape: QWTensorShape([selectBatch, maxBuckets], .i32))
        self.selHist = QWTensorValue(region: reg, offset: selHistOff,
                                     shape: QWTensorShape([selectBatch, 512], .i32))
        self.selCount = QWTensorValue(region: reg, offset: selCountOff,
                                      shape: QWTensorShape([selectBatch], .i32))
        self.selMarks = QWTensorValue(region: reg, offset: selMarksOff,
                                      shape: QWTensorShape([selectBatch, markStride], .i32))
        self.selCandM = QWTensorValue(region: reg, offset: selCandMOff,
                                      shape: QWTensorShape([selectBatch, candCap], .i32))
        self.selCandU = QWTensorValue(region: reg, offset: selCandUOff,
                                      shape: QWTensorShape([selectBatch, candCap], .i32))
    }

    /// Mirrors `LSASelParams` in qwlsa.metal.
    private struct SelectParams {
        var T: UInt32
        var pos0: UInt32
        var indexDim: UInt32
        var budgetBlocks: UInt32
        var windowBlocks: UInt32
        var listStride: UInt32
        var scoreStride: UInt32
        var rowStep: UInt32
        var maxRow: UInt32
        var bucketStride: UInt32
        var candCap: UInt32
        var markStride: UInt32
    }

    /// Centroid-bucket count of the last (largest-position) unit when it is in the
    /// centroid-filter regime of the selector, else nil. Mirrors `lsa_sel_geom`.
    private func filterBuckets(pos0: Int, units: Int, rowStep: Int, maxRow: Int) -> Int? {
        let lastRow = rowStep > 1 ? min((units - 1) * rowStep + rowStep - 1, maxRow - 1) : units - 1
        let totalBlocks = (pos0 + lastRow + 1) >> 2
        let budget = config.budgetBlocks
        let window = QWLSAConfig.windowTokens / QWLSAConfig.poolSize
        guard totalBlocks > 1 + budget + window else { return nil }
        let middleEnd = totalBlocks - window
        let kEff = min(budget, middleEnd - 1)
        let numBuckets = (middleEnd >> 7) * 64
        return numBuckets > max(256, kEff * 2) ? numBuckets : nil
    }

    /// Encode index Q/K projection (`qwg_lsa_index_prep`), 4-token key pooling (`qwg_lsa_pool_keys`),
    /// and 512-token dynamic Ward clustering (`qwg_lsa_ward_cluster_512`) for `rows` tokens starting at `pos0`.
    public func encodeIndexAndCluster(_ enc: any MTLComputeCommandEncoder,
                                      qgate: QWTensorValue,
                                      kIn: QWTensorValue,
                                      fullAttnLayer: Int,
                                      pos0: Int,
                                      rows: Int) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        if pos0 < residentPos512 {
            residentPos512 = (pos0 / 512) * 512
        }
        let ls = layers[fullAttnLayer]
        var tU = UInt32(rows)
        var pos0U = UInt32(pos0)
        var qHeadsU = UInt32(geometry.qHeads)
        var kvHeadsU = UInt32(geometry.kvHeads)
        var headDimU = UInt32(geometry.headDim)
        var qProjOutU = UInt32(geometry.qProjOut)
        var qStrideU = UInt32(geometry.descriptor.attnOutputGate ? 2 : 1)
        var sliceStartU = UInt32(sliceStart)
        var sliceDimU = UInt32(sliceDim)
        var idxDimU = UInt32(indexDim)
        var ringMaskU = UInt32(ringCap - 1)

        // 1. Project Q_idx (into qIdxScratch) and K_idx (into kPending ring buffer)
        enc.setComputePipelineState(psoIndexPrep)
        qgate.bind(enc, index: 0)
        kIn.bind(enc, index: 1)
        ls.wqIdx.bind(enc, index: 2)
        ls.wkIdx.bind(enc, index: 3)
        qIdxScratch.bind(enc, index: 4)
        ls.kPending.bind(enc, index: 5)
        enc.setBytes(&tU, length: 4, index: 6)
        enc.setBytes(&pos0U, length: 4, index: 7)
        enc.setBytes(&qHeadsU, length: 4, index: 8)
        enc.setBytes(&kvHeadsU, length: 4, index: 9)
        enc.setBytes(&headDimU, length: 4, index: 10)
        enc.setBytes(&qProjOutU, length: 4, index: 11)
        enc.setBytes(&qStrideU, length: 4, index: 12)
        enc.setBytes(&sliceStartU, length: 4, index: 13)
        enc.setBytes(&sliceDimU, length: 4, index: 14)
        enc.setBytes(&idxDimU, length: 4, index: 15)
        enc.setBytes(&ringMaskU, length: 4, index: 16)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        // 2. Pool any newly completed 4-token blocks in [pos0 / 4 ..< (pos0 + rows) / 4]
        let startBlock = pos0 / 4
        let endBlock = min(maxBlocks, (pos0 + rows) / 4)
        if endBlock > startBlock {
            let numBlocks = endBlock - startBlock
            var startBlkU = UInt32(startBlock)
            var numBlkU = UInt32(numBlocks)
            enc.setComputePipelineState(psoPoolKeys)
            ls.kPending.bind(enc, index: 0)
            ls.pooledKeys.bind(enc, index: 1)
            enc.setBytes(&startBlkU, length: 4, index: 2)
            enc.setBytes(&numBlkU, length: 4, index: 3)
            enc.setBytes(&idxDimU, length: 4, index: 4)
            enc.setBytes(&ringMaskU, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        // 3. Ward-cluster any newly completed 512-token superblocks in [pos0 / 512 ..< (pos0 + rows) / 512]
        let startSB = pos0 / 512
        let endSB = min(maxSuperblocks, (pos0 + rows) / 512)
        if endSB > startSB {
            let numSB = endSB - startSB
            var startSBU = UInt32(startSB)
            var numSBU = UInt32(numSB)
            enc.setComputePipelineState(psoWardCluster)
            ls.pooledKeys.bind(enc, index: 0)
            ls.centroids.bind(enc, index: 1)
            ls.bucketMembers.bind(enc, index: 2)
            ls.bucketSizes.bind(enc, index: 3)
            enc.setBytes(&startSBU, length: 4, index: 4)
            enc.setBytes(&numSBU, length: 4, index: 5)
            enc.setBytes(&idxDimU, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: numSB, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    /// Encode LSA block selection into `blockListScratch`/`blockCountScratch` (parallel 4-stage selector in the
    /// centroid-filter regime, else `qwg_lsa_select_blocks`). With `tiled32` (NAX prefill), selects only for
    /// reference row `min(tile * 32 + 31, rows - 1)` of each 32-row tile.
    public func encodeSelectBlocks(_ enc: any MTLComputeCommandEncoder,
                                   fullAttnLayer: Int,
                                   pos0: Int,
                                   rows: Int,
                                   tiled32: Bool = false) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        let ls = layers[fullAttnLayer]
        var idxDimU = UInt32(indexDim)
        var budgetBlkU = UInt32(config.budgetBlocks)
        var windowBlkU = UInt32(QWLSAConfig.windowTokens / QWLSAConfig.poolSize)
        var strideU = UInt32(listStride)
        var scratchStrideU = UInt32(maxBlocks)
        let step = tiled32 ? 32 : 1
        var rowStepU = UInt32(step)
        let totalUnits = tiled32 ? ((rows + 31) / 32) : rows

        var u0 = 0
        while u0 < totalUnits {
            let batchUnits = min(Self.maxSelectBatchRows, totalUnits - u0)
            let r0 = u0 * step
            let remRows = rows - r0
            var subUnitsU = UInt32(batchUnits)
            var subPos0U = UInt32(pos0 + r0)
            var maxRowU = UInt32(remRows)
            let qIdxSub = QWTensorValue(
                region: qIdxScratch.region,
                offset: qIdxScratch.offset + r0 * indexDim * 2,
                shape: QWTensorShape([remRows, indexDim], .f16))
            let listSub = QWTensorValue(
                region: blockListScratch.region,
                offset: blockListScratch.offset + r0 * listStride * 4,
                shape: QWTensorShape([remRows, listStride], .i32))
            let countSub = QWTensorValue(
                region: blockCountScratch.region,
                offset: blockCountScratch.offset + r0 * 4,
                shape: QWTensorShape([remRows], .i32))

            if useParallelSelect, let numBuckets = filterBuckets(pos0: pos0 + r0, units: batchUnits,
                                              rowStep: step, maxRow: remRows) {
                var prm = SelectParams(
                    T: subUnitsU, pos0: subPos0U, indexDim: idxDimU, budgetBlocks: budgetBlkU,
                    windowBlocks: windowBlkU, listStride: strideU, scoreStride: scratchStrideU,
                    rowStep: rowStepU, maxRow: maxRowU, bucketStride: UInt32(maxBuckets),
                    candCap: UInt32(Self.selectCandidateCap), markStride: UInt32(selMarkStride))
                let plen = MemoryLayout<SelectParams>.stride
                let grid = MTLSize(width: (numBuckets + 255) / 256, height: batchUnits, depth: 1)
                let tpg = MTLSize(width: 256, height: 1, depth: 1)

                enc.setComputePipelineState(psoSelectScore)
                qIdxSub.bind(enc, index: 0)
                ls.centroids.bind(enc, index: 1)
                selCScore.bind(enc, index: 2)
                selHist.bind(enc, index: 3)
                enc.setBytes(&prm, length: plen, index: 4)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)

                enc.setComputePipelineState(psoSelectHist2)
                selCScore.bind(enc, index: 0)
                selHist.bind(enc, index: 1)
                enc.setBytes(&prm, length: plen, index: 2)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)

                enc.setComputePipelineState(psoSelectExpand)
                qIdxSub.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                ls.bucketMembers.bind(enc, index: 2)
                ls.bucketSizes.bind(enc, index: 3)
                selCScore.bind(enc, index: 4)
                selHist.bind(enc, index: 5)
                selMarks.bind(enc, index: 6)
                selCount.bind(enc, index: 7)
                selCandM.bind(enc, index: 8)
                selCandU.bind(enc, index: 9)
                enc.setBytes(&prm, length: plen, index: 10)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)

                enc.setComputePipelineState(psoSelectFinalize)
                qIdxSub.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                ls.centroids.bind(enc, index: 2)
                ls.bucketMembers.bind(enc, index: 3)
                ls.bucketSizes.bind(enc, index: 4)
                scoreScratch.bind(enc, index: 5)
                listSub.bind(enc, index: 6)
                countSub.bind(enc, index: 7)
                selHist.bind(enc, index: 8)
                selMarks.bind(enc, index: 9)
                selCount.bind(enc, index: 10)
                selCandM.bind(enc, index: 11)
                selCandU.bind(enc, index: 12)
                enc.setBytes(&prm, length: plen, index: 13)
                enc.dispatchThreadgroups(MTLSize(width: batchUnits, height: 1, depth: 1),
                                         threadsPerThreadgroup: tpg)
                u0 += batchUnits
                continue
            }

            enc.setComputePipelineState(psoSelectBlocks)
            qIdxSub.bind(enc, index: 0)
            ls.pooledKeys.bind(enc, index: 1)
            ls.centroids.bind(enc, index: 2)
            ls.bucketMembers.bind(enc, index: 3)
            ls.bucketSizes.bind(enc, index: 4)
            scoreScratch.bind(enc, index: 5)
            listSub.bind(enc, index: 6)
            countSub.bind(enc, index: 7)
            enc.setBytes(&subUnitsU, length: 4, index: 8)
            enc.setBytes(&subPos0U, length: 4, index: 9)
            enc.setBytes(&idxDimU, length: 4, index: 10)
            enc.setBytes(&budgetBlkU, length: 4, index: 11)
            enc.setBytes(&windowBlkU, length: 4, index: 12)
            enc.setBytes(&strideU, length: 4, index: 13)
            enc.setBytes(&scratchStrideU, length: 4, index: 14)
            enc.setBytes(&rowStepU, length: 4, index: 15)
            enc.setBytes(&maxRowU, length: 4, index: 16)
            enc.dispatchThreadgroups(
                MTLSize(width: batchUnits, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            u0 += batchUnits
        }
    }

    public static func makeIfEnabled(
        hal: QWHal,
        geometry: QWModelGeometry,
        lsaConfig: QWLSAConfig,
        packPath: String,
        pack: QWQuant.Pack?,
        safetensors: QWSafeTensorsContainer? = nil,
        attnLayers: [Int],
        maxContext: Int,
        maxPrefillChunk: Int
    ) throws -> QWLSAOperator? {
        guard lsaConfig.isEnabled && !geometry.usesNoPEMLA && !attnLayers.isEmpty && geometry.lsaInputDim > 0 else { return nil }
        let sidecarURL = QWLSAConfig.resolveSidecarURL(packPath: packPath, explicitSidecarPath: lsaConfig.sidecarPath)
        let inDim = geometry.lsaInputDim
        var indexDim = QWLSAConfig.defaultIndexDim
        var layerWeights: [LayerWeights] = []
        if lsaConfig.mode != .svd, let url = sidecarURL,
           let loaded = try? QWLSASidecar(url: url, descriptor: geometry.descriptor) {
            indexDim = loaded.header.provenance.indexDim
            let shape = [indexDim, inDim]
            for vLayer in attnLayers {
                let physLayer = geometry.executionPlan[vLayer]
                let wq = (try? loaded.readF16Tensor("layers.\(physLayer).wq_idx", expectedShape: shape))
                    ?? QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(physLayer &* 2 &+ 1))
                let wk = (try? loaded.readF16Tensor("layers.\(physLayer).wk_idx", expectedShape: shape)) ?? wq
                layerWeights.append(.init(physicalLayer: physLayer, wqIdxF16: wq, wkIdxF16: wk))
            }
        } else {
            let synth: [Int: (wq: [UInt16], wk: [UInt16])]?
            if let pk = pack {
                synth = try? QWLSASidecar.synthesizeFromPack(
                    pack: pk, descriptor: geometry.descriptor, indexDim: indexDim)
            } else if let st = safetensors {
                synth = try? QWLSASidecar.synthesizeFromSafeTensors(
                    container: st, descriptor: geometry.descriptor, indexDim: indexDim)
            } else {
                synth = nil
            }
            for vLayer in attnLayers {
                let physLayer = geometry.executionPlan[vLayer]
                if let pair = synth?[physLayer] {
                    layerWeights.append(.init(physicalLayer: physLayer, wqIdxF16: pair.wq, wkIdxF16: pair.wk))
                } else {
                    let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(physLayer &* 2 &+ 1))
                    layerWeights.append(.init(physicalLayer: physLayer, wqIdxF16: basis, wkIdxF16: basis))
                }
            }
        }
        return try QWLSAOperator(
            hal: hal, geometry: geometry, config: lsaConfig,
            indexDim: indexDim, layerWeights: layerWeights,
            maxContext: maxContext, maxPrefillChunk: maxPrefillChunk)
    }
}
