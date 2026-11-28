// QWMBSAAttnOperator.swift — Maxwell Block-Sparse GQA (MBSA) GPU operator.
// Manages per-layer indexer K caches, block-max remote top-k selection, and sparse paged GQA.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWKV
import QWRope

public final class QWMBSAAttnOperator: @unchecked Sendable {
    public struct LayerWeights: Sendable {
        public let qkvIdxProj: QWWeightView?
        public let wqIdx: QWTensorValue
        public let wkIdx: QWTensorValue

        public init(qkvIdxProj: QWWeightView? = nil, wqIdx: QWTensorValue, wkIdx: QWTensorValue) {
            self.qkvIdxProj = qkvIdxProj
            self.wqIdx = wqIdx
            self.wkIdx = wkIdx
        }
    }

    public let geometry: QWModelGeometry
    public let config: QWModelDescriptor.BlockSparseGQAConfig
    public let kvMode: QWKVMode
    public let maxContext: Int
    public let maxRows: Int
    public let layerCount: Int
    public let kvHeads: Int
    public let qHeads: Int
    public let headDim: Int
    public let idxHeads: Int
    public let idxDim: Int
    public let listStride: Int
    public let tileListStride: Int
    public let maxTiles: Int
    public let gpuCores: Int
    public let supportsNAXPrefill: Bool
    public let supportsSplitKDecode: Bool
    public let layerKCacheElems: Int
    public let layerKCacheStrideBytes: Int

    public let region: QWMRegion
    public let idxQKVScratch: QWTensorValue
    public let idxQScratch: QWTensorValue
    public let idxKCaches: [QWTensorValue]
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue
    public let tileBlockListScratch: QWTensorValue
    public let tileBlockMaskScratch: QWTensorValue
    public let tileBlockCountScratch: QWTensorValue
    public weak var stateOwner: AnyObject?

    internal let psoIndexPrep: any MTLComputePipelineState
    internal let psoSelectBlocks: any MTLComputePipelineState
    internal let psoSelectBlocksTiled32: (any MTLComputePipelineState)?
    internal let psoSparseAttn: any MTLComputePipelineState
    internal let psoPrefillNAX: (any MTLComputePipelineState)?
    internal let psoDecodeSparse: any MTLComputePipelineState
    internal let psoCombineBatched: any MTLComputePipelineState

    public static func makeIfEnabled(
        hal: QWHal,
        geometry: QWModelGeometry,
        attnLayers: [Int],
        kvMode: QWKVMode,
        maxContext: Int,
        maxPrefillChunk: Int
    ) throws -> QWMBSAAttnOperator? {
        guard let bsa = geometry.blockSparseGQAConfig, !attnLayers.isEmpty else { return nil }
        return try QWMBSAAttnOperator(
            hal: hal, geometry: geometry, config: bsa, kvMode: kvMode,
            layerCount: attnLayers.count, maxContext: maxContext, maxPrefillChunk: maxPrefillChunk)
    }

    public static func tileListStride(config: QWModelDescriptor.BlockSparseGQAConfig) -> Int {
        let block = max(1, config.blockSize)
        let raw = config.prefixBlocks + config.localBlocks + (32 + block - 1) / block + 32 * config.remoteBlocks
        return max(max(1, config.maxSelectedBlocks), ((raw + 31) / 32) * 32)
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a), y = abs(b)
        while y != 0 { (x, y) = (y, x % y) }
        return max(1, x)
    }

    public static func targetSplits(
        rows: Int,
        kvHeads: Int,
        maxKeys: Int,
        gpuCores: Int = 40
    ) -> Int {
        let r = max(1, rows)
        let kvh = max(1, kvHeads)
        let cap = r == 1 ? 64 : 32
        let raw = r == 1
            ? min(cap, max(8, (maxKeys + 191) / 192))
            : min(cap, max(4, 32 / r))
        guard gpuCores > 0 else { return raw }
        let activeGroups = r * kvh
        let step = gpuCores / gcd(activeGroups, gpuCores)
        guard step <= cap else { return raw }
        let aligned = (raw / step) * step
        if aligned >= 1 {
            return min(cap, aligned)
        }
        return step * 64 <= maxKeys ? min(cap, step) : raw
    }

    public init(
        hal: QWHal,
        geometry: QWModelGeometry,
        config: QWModelDescriptor.BlockSparseGQAConfig,
        kvMode: QWKVMode,
        layerCount: Int,
        maxContext: Int,
        maxPrefillChunk: Int
    ) throws {
        self.geometry = geometry
        self.config = config
        self.kvMode = kvMode
        self.layerCount = layerCount
        self.maxContext = max(1, maxContext)
        self.maxRows = max(8, maxPrefillChunk)
        self.kvHeads = geometry.kvHeads
        self.qHeads = geometry.qHeads
        self.headDim = geometry.headDim
        self.idxHeads = geometry.mbsaIndexerHeads
        self.idxDim = geometry.mbsaIndexerHeadDim
        self.listStride = max(1, config.maxSelectedBlocks)
        self.tileListStride = Self.tileListStride(config: config)
        self.maxTiles = (self.maxRows + 31) / 32
        self.gpuCores = hal.caps.gpuCores
        self.supportsNAXPrefill = hal.caps.isMetal4TensorCapable
            && geometry.supportsNAXPagedAttnPrefill
            && config.blockSize >= 32
            && self.headDim <= 128
            && self.idxDim <= 128
            && self.idxHeads <= self.kvHeads
            && config.remoteBlocks <= 32
        self.supportsSplitKDecode = geometry.gqaRatio <= 8 && self.headDim <= 256
        self.layerKCacheElems = self.maxContext * self.idxDim
        let rawKBytes = self.layerKCacheElems * 2
        self.layerKCacheStrideBytes = (rawKBytes + 255) & ~255

        self.psoIndexPrep = try hal.pipeline("qwg_mbsa_index_prep_f16")
        self.psoSelectBlocks = try hal.pipeline("qwg_mbsa_select_blocks")
        self.psoSelectBlocksTiled32 = hal.caps.isMetal4TensorCapable
            ? try hal.pipeline("qwg_mbsa_select_blocks_tiled32") : nil
        let attnKernel: String
        let naxKernel: String
        let decodeKernel: String
        switch kvMode {
        case .f16:
            attnKernel = "qwg_mbsa_attn_f16"
            naxKernel = "qwg_mbsa_prefill_paged_nax_f16"
            decodeKernel = "qwg_mbsa_decode_sparse_f16"
        case .q8:
            attnKernel = "qwg_mbsa_attn_q8"
            naxKernel = "qwg_mbsa_prefill_paged_nax_q8"
            decodeKernel = "qwg_mbsa_decode_sparse_q8"
        case .q4:
            attnKernel = "qwg_mbsa_attn_q4"
            naxKernel = "qwg_mbsa_prefill_paged_nax_q4"
            decodeKernel = "qwg_mbsa_decode_sparse_q4"
        }
        self.psoSparseAttn = try hal.pipeline(attnKernel)
        self.psoPrefillNAX = hal.caps.isMetal4TensorCapable ? try hal.pipeline(naxKernel) : nil
        self.psoDecodeSparse = try hal.pipeline(decodeKernel)
        self.psoCombineBatched = try hal.pipeline("qwg_attn_decode_combine_batched")

        func align256(_ n: Int) -> Int { (n + 255) & ~255 }
        let qkvScratchElems = self.maxRows * (self.idxHeads + 1) * self.idxDim
        let qScratchElems = self.maxRows * self.idxHeads * self.idxDim
        let blkListElems = self.maxRows * self.kvHeads * self.listStride
        let blkCntElems = self.maxRows * self.kvHeads
        let tileListElems = self.maxTiles * self.kvHeads * self.tileListStride
        let tileCntElems = self.maxTiles * self.kvHeads
        let totalBytes = align256(qkvScratchElems * 2)
            + align256(qScratchElems * 2)
            + align256(blkListElems * 4)
            + align256(blkCntElems * 4)
            + align256(tileListElems * 4) * 2
            + align256(tileCntElems * 4)
            + layerCount * self.layerKCacheStrideBytes
            + 4096

        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "mbsa-attn", asked: totalBytes, free: 0)
        }
        buf.label = "mbsa-attn"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg
        let sub = QWSubAllocator(region: reg)
        self.idxQKVScratch = try sub.allocate(QWTensorShape([qkvScratchElems], .f16))
        self.idxQScratch = try sub.allocate(QWTensorShape([qScratchElems], .f16))
        self.blockListScratch = try sub.allocate(QWTensorShape([blkListElems], .i32))
        self.blockCountScratch = try sub.allocate(QWTensorShape([blkCntElems], .i32))
        self.tileBlockListScratch = try sub.allocate(QWTensorShape([tileListElems], .i32))
        self.tileBlockMaskScratch = try sub.allocate(QWTensorShape([tileListElems], .i32))
        self.tileBlockCountScratch = try sub.allocate(QWTensorShape([tileCntElems], .i32))
        var kCaches: [QWTensorValue] = []
        kCaches.reserveCapacity(layerCount)
        for _ in 0..<layerCount {
            kCaches.append(try sub.allocate(QWTensorShape([self.layerKCacheElems], .f16)))
        }
        self.idxKCaches = kCaches
    }

    public func encodeIndexPrep(
        _ enc: any MTLComputeCommandEncoder,
        weights: LayerWeights,
        ropeTable: QWTensorValue,
        layerIndex: Int,
        pos0: Int,
        rows: Int,
        qkvSource: QWTensorValue? = nil,
        qkvRowStrideCols: Int = 0
    ) {
        let qkvSlice = qkvSource ?? idxQKVScratch.slice(byteOffset: 0, count: rows * (idxHeads + 1) * idxDim, dtype: .f16)
        let qSlice = idxQScratch.slice(byteOffset: 0, count: rows * idxHeads * idxDim, dtype: .f16)
        let kCache = idxKCaches[layerIndex]
        enc.setComputePipelineState(psoIndexPrep)
        qkvSlice.bind(enc, index: 0)
        qSlice.bind(enc, index: 1)
        kCache.bind(enc, index: 2)
        weights.wqIdx.bind(enc, index: 3)
        weights.wkIdx.bind(enc, index: 4)
        ropeTable.bind(enc, index: 5)
        var dims: (UInt32, UInt32, UInt32, UInt32) = (UInt32(pos0), UInt32(idxHeads), UInt32(idxDim / 8), UInt32(qkvRowStrideCols / 8))
        var eps = geometry.rmsEps
        enc.setBytes(&dims, length: 16, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: idxHeads + 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeSelectAndAttend(
        _ enc: any MTLComputeCommandEncoder,
        qgate: QWTensorValue,
        kv: (k: QWKVLayerView, v: QWKVLayerView),
        out: QWTensorValue,
        layerIndex: Int,
        pos0: Int,
        rows: Int,
        partials: QWMRegion? = nil
    ) {
        let qSlice = idxQScratch.slice(byteOffset: 0, count: rows * idxHeads * idxDim, dtype: .f16)
        let kCache = idxKCaches[layerIndex]
        let packedHeads = UInt32(kvHeads & 0xFFFF) | (UInt32(idxHeads & 0xFFFF) << 16)
        if rows > 8 && supportsNAXPrefill,
           let psoSelectBlocksTiled32, let psoPrefillNAX {
            let numTiles = (rows + 31) / 32
            enc.setComputePipelineState(psoSelectBlocksTiled32)
            qSlice.bind(enc, index: 0)
            kCache.bind(enc, index: 1)
            tileBlockListScratch.bind(enc, index: 2)
            tileBlockMaskScratch.bind(enc, index: 3)
            tileBlockCountScratch.bind(enc, index: 4)
            var p0: (UInt32, UInt32, UInt32, UInt32) = (
                UInt32(pos0), packedHeads, UInt32(idxDim), UInt32(config.blockSize))
            var p1: (UInt32, UInt32, UInt32, UInt32) = (
                UInt32(config.prefixBlocks), UInt32(config.localBlocks),
                UInt32(config.remoteBlocks), UInt32(tileListStride))
            var p2: (UInt32, UInt32, UInt32, UInt32) = (UInt32(rows), UInt32(listStride), 0, 0)
            enc.setBytes(&p0, length: 16, index: 5)
            enc.setBytes(&p1, length: 16, index: 6)
            enc.setBytes(&p2, length: 16, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: kvHeads, height: numTiles, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(psoPrefillNAX)
            qgate.bind(enc, index: 0)
            kv.k.pool.bindTo(enc, index: 1)
            kv.v.pool.bindTo(enc, index: 2)
            kv.k.table.bind(enc, index: 3)
            kv.v.table.bind(enc, index: 4)
            out.bind(enc, index: 5)
            var posU = UInt32(pos0)
            var tU = UInt32(rows)
            var strideU = UInt32(tileListStride)
            var blockU = UInt32(config.blockSize)
            enc.setBytes(&posU, length: 4, index: 6)
            enc.setBytes(&tU, length: 4, index: 7)
            tileBlockListScratch.bind(enc, index: 8)
            tileBlockMaskScratch.bind(enc, index: 9)
            tileBlockCountScratch.bind(enc, index: 10)
            enc.setBytes(&strideU, length: 4, index: 11)
            enc.setBytes(&blockU, length: 4, index: 12)
            let gqaRatio = max(1, qHeads / max(1, kvHeads))
            enc.dispatchThreadgroups(MTLSize(width: numTiles * kvHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64 * gqaRatio, height: 1, depth: 1))
            return
        }

        enc.setComputePipelineState(psoSelectBlocks)
        qSlice.bind(enc, index: 0)
        kCache.bind(enc, index: 1)
        blockListScratch.bind(enc, index: 2)
        blockCountScratch.bind(enc, index: 3)
        var p0: (UInt32, UInt32, UInt32, UInt32) = (
            UInt32(pos0), packedHeads, UInt32(idxDim), UInt32(config.blockSize))
        var p1: (UInt32, UInt32, UInt32, UInt32) = (
            UInt32(config.prefixBlocks), UInt32(config.localBlocks),
            UInt32(config.remoteBlocks), UInt32(listStride))
        enc.setBytes(&p0, length: 16, index: 4)
        enc.setBytes(&p1, length: 16, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: kvHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        if rows <= 8 && supportsSplitKDecode {
            let maxPartialRowBytes = qHeads * 64 * (headDim + 2) * 4
            let canSplit = partials.map { $0.length >= rows * maxPartialRowBytes } ?? false
            let maxKeys = config.maxSelectedBlocks * config.blockSize
            let targetSplits = Self.targetSplits(rows: rows, kvHeads: kvHeads, maxKeys: maxKeys, gpuCores: self.gpuCores)
            let splits = canSplit ? targetSplits : 1
            enc.setComputePipelineState(psoDecodeSparse)
            qgate.bind(enc, index: 0)
            kv.k.pool.bindTo(enc, index: 1)
            kv.v.pool.bindTo(enc, index: 2)
            kv.k.table.bind(enc, index: 3)
            kv.v.table.bind(enc, index: 4)
            blockListScratch.bind(enc, index: 5)
            blockCountScratch.bind(enc, index: 6)
            if let partials, splits > 1 {
                enc.setBuffer(partials.buffer, offset: partials.base, index: 7)
            } else {
                blockCountScratch.bind(enc, index: 7)
            }
            out.bind(enc, index: 8)
            var params: (UInt32, UInt32, UInt32, UInt32) = (
                UInt32(pos0), UInt32(config.blockSize), UInt32(listStride), UInt32(splits))
            enc.setBytes(&params, length: 16, index: 9)
            let stageDim = headDim <= 128 ? 128 : 256
            enc.setThreadgroupMemoryLength(2 * 16 * stageDim * 2, index: 0)
            let gqaRatio = max(1, qHeads / max(1, kvHeads))
            enc.dispatchThreadgroups(MTLSize(width: kvHeads, height: splits, depth: rows),
                                     threadsPerThreadgroup: MTLSize(width: gqaRatio * 32, height: 1, depth: 1))
            if let partials, splits > 1 {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(psoCombineBatched)
                qgate.bind(enc, index: 0)
                out.bind(enc, index: 1)
                enc.setBuffer(partials.buffer, offset: partials.base, index: 2)
                var splitsU = UInt32(splits)
                enc.setBytes(&splitsU, length: 4, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: qHeads, height: rows, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            return
        }

        enc.setComputePipelineState(psoSparseAttn)
        qgate.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        blockListScratch.bind(enc, index: 5)
        blockCountScratch.bind(enc, index: 6)
        out.bind(enc, index: 7)
        var params: (UInt32, UInt32, UInt32, UInt32) = (UInt32(pos0), UInt32(config.blockSize), UInt32(listStride), 0)
        enc.setBytes(&params, length: 16, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func indexerStateBytes(position: Int) -> Int {
        guard position > 0, position <= maxContext else { return 0 }
        return layerCount * position * idxDim * 2
    }

    public func captureIndexerState(position: Int) -> Data {
        let len = min(position, maxContext)
        guard len > 0 else { return Data() }
        let rowBytes = len * idxDim * 2
        var data = Data(count: layerCount * rowBytes)
        let srcs = (0..<layerCount).map { i -> UnsafeRawPointer in
            let c = idxKCaches[i]
            return UnsafeRawPointer(c.region.buffer.contents().advanced(by: c.region.base + c.offset))
        }
        withExtendedLifetime(idxKCaches) {
            data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let dst = raw.baseAddress else { return }
                for i in 0..<layerCount {
                    dst.advanced(by: i * rowBytes).copyMemory(from: srcs[i], byteCount: rowBytes)
                }
            }
        }
        return data
    }

    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        let len = min(position, maxContext)
        guard len >= 0 else { return false }
        let rowBytes = len * idxDim * 2
        guard data.count == layerCount * rowBytes else { return false }
        if rowBytes > 0 {
            let dsts = (0..<layerCount).map { i -> UnsafeMutableRawPointer in
                let c = idxKCaches[i]
                return c.region.buffer.contents().advanced(by: c.region.base + c.offset)
            }
            withExtendedLifetime(idxKCaches) {
                data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let src = raw.baseAddress else { return }
                    for i in 0..<layerCount {
                        dsts[i].copyMemory(from: src.advanced(by: i * rowBytes), byteCount: rowBytes)
                    }
                }
            }
        }
        return true
    }
}
