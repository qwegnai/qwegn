// QWQSAEncoders.swift — Layer-4 QSA attention dispatch encoders (`Sources/QWMetal/qwattn_sparse.metal`).
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

private enum QWQSAEnvironmentToggles {
    static let qsaMatEnabled = ProcessInfo.processInfo.environment["QW_FN_QSA_MAT"] != "0"
}

/// Canonical Layer-4 Metal encoders for Query-Sparse Attention (QSA) indexing, selection, and gated attention.
public enum QWQSAEncoders {
    /// Indexer q/k prep from the grouped projection output: q heads RMSNorm + partial RoPE; raw k copied.
    public static func encodeIndexPrep(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                       src: QWTensorValue, iqNorm: QWTensorValue, cos: QWTensorValue,
                                       sin: QWTensorValue, idxQ: QWTensorValue, idxK: QWTensorValue,
                                       rows: Int, rowStride: Int, iqOff: Int, ikOff: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_index_prep"))
        enc.bindTensors(src, iqNorm, cos, sin, idxQ, idxK, startIndex: 0)
        enc.setUInt32s(rows, rowStride, iqOff, ikOff, startIndex: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows * 5, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Pooled block keys: mean of `compress` raw keys, RMSNorm, RoPE at the block start. One TG per block.
    public static func encodeIndexPool(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                       idxK: QWTensorValue, ikNorm: QWTensorValue, bcos: QWTensorValue,
                                       bsin: QWTensorValue, pooled: QWTensorValue,
                                       blocks: Int, compress: Int) throws {
        enc.setComputePipelineState(try hal.pipeline("qwg_index_pool"))
        enc.bindTensors(idxK, ikNorm, bcos, bsin, pooled, startIndex: 0)
        enc.setUInt32s(compress, startIndex: 5)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Dynamic Ward clustering over newly completed 512-token (128-pooled-block) superblocks.
    public static func encodeLSAWardCluster(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                            pooled: QWTensorValue,
                                            clusters: QWQSACacheState.LSAClusterState,
                                            startSuperblock: Int, numSuperblocks: Int,
                                            indexDim: Int) throws {
        guard numSuperblocks > 0 else { return }
        enc.setComputePipelineState(try hal.pipeline("qwg_lsa_ward_cluster_512"))
        enc.bindTensors(pooled, clusters.centroids, clusters.bucketMembers, clusters.bucketSizes, startIndex: 0)
        enc.setUInt32s(startSuperblock, numSuperblocks, indexDim, startIndex: 4)
        enc.dispatchThreadgroups(MTLSize(width: numSuperblocks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Clusters any newly completed chunks (`4096`-token `e-means` CSR or `512`-token Ward superblocks)
    /// across `[pos0 ..< pos0 + count)`. Returns `true` when at least one chunk was clustered.
    @discardableResult
    public static func encodeLSAClusterRange(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                             pooled: QWTensorValue,
                                             clusters: QWQSACacheState.LSAClusterState,
                                             pos0: Int, count: Int,
                                             indexDim: Int) throws -> Bool {
        let unitTok = clusters.isEmeans ? QWCoarsePoolIndex.macroChunkTokens : 512
        let u0 = pos0 / unitTok
        let u1 = (pos0 + count) / unitTok
        guard u1 > u0 else { return false }
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        if clusters.isEmeans {
            let pso = try hal.pipeline("qwg_emeans_cluster_1024")
            let csr = QWCoarsePoolIndex.LayerCSR(
                centroids: clusters.centroids, memberPools: clusters.bucketMembers,
                clusterSpans: clusters.bucketSizes, maxMacros: max(u1, clusters.maxUnits))
            QWCoarsePoolIndex.encodeClusterMacros(
                enc, pso: pso, pooledKeys: pooled, csr: csr,
                startMacro: u0, numMacros: u1 - u0, indexDim: indexDim)
        } else {
            try encodeLSAWardCluster(enc, hal: hal, pooled: pooled, clusters: clusters,
                                     startSuperblock: u0, numSuperblocks: u1 - u0, indexDim: indexDim)
        }
        return true
    }

    /// Exact 2-stage QSA top-k per query over complete blocks (or multi-TG `e-means` / legacy Ward coarse selector
    /// when `lsaClusters != nil`); list i32 [T,512] receives ascending selected block ids.
    public static func encodeQSASelect(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                       idxQ: QWTensorValue, pooled: QWTensorValue, list: QWTensorValue,
                                       rows: Int, blockTopK: Int, pos0: Int = 0,
                                       scores: QWTensorValue,
                                       lsaClusters: QWQSACacheState.LSAClusterState? = nil) throws {
        if let clusters = lsaClusters, let cScratch = clusters.emeansScratch {
            let probe = max(544, blockTopK + 32)
            let bg = QWCoarsePoolIndex.batchGeometry(
                pos0: pos0, units: rows, budgetBlocks: blockTopK,
                windowBlocks: 0, hasSink: false, writeCount: false, probeClusters: probe)
            if bg.shouldPrune {
                let pipes = try QWCoarsePoolIndex.Pipelines(hal: hal)
                let psoCoarse = try hal.pipeline("qwg_emeans_qsa_score_coarse")
                let psoCoarseTiled = try hal.pipeline("qwg_emeans_qsa_score_coarse_tiled")
                let psoCands = try hal.pipeline("qwg_emeans_qsa_score_cands")
                let psoFallback = try hal.pipeline("qwg_emeans_qsa_flat_fallback")
                let csr = QWCoarsePoolIndex.LayerCSR(
                    centroids: clusters.centroids, memberPools: clusters.bucketMembers,
                    clusterSpans: clusters.bucketSizes, maxMacros: max(bg.numMacros, clusters.maxUnits))
                let tpg = MTLSize(width: 256, height: 1, depth: 1)
                var r0 = 0
                while r0 < rows {
                    let subRows = min(cScratch.maxBatchRows, rows - r0)
                    let subBg = QWCoarsePoolIndex.batchGeometry(
                        pos0: pos0 + r0, units: subRows, budgetBlocks: blockTopK,
                        windowBlocks: 0, hasSink: false, writeCount: false, probeClusters: probe)
                    let qSlice = idxQ.slice(byteOffset: r0 * 4 * 128 * 4, count: subRows * 4 * 128, dtype: .f32)
                    let listSlice = list.slice(byteOffset: r0 * blockTopK * 4, count: subRows * blockTopK, dtype: .i32)
                    var ePrm = QWCoarsePoolIndex.SelectParams(
                        T: subRows, pos0: pos0 + r0, indexDim: 128, budgetBlocks: blockTopK,
                        windowBlocks: 0, hasSink: false, writeCount: false, listStride: blockTopK,
                        rowStep: 1, maxRow: subRows, clusterStride: cScratch.clusterStride,
                        probeClusters: probe, candCap: cScratch.candCap)
                    let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
                    if subBg.numClusters > 0 {
                        let useCoarseTiled = QWQSAEnvironmentToggles.qsaMatEnabled && subRows >= 4
                        enc.setComputePipelineState(useCoarseTiled ? psoCoarseTiled : psoCoarse)
                        qSlice.bind(enc, index: 0)
                        csr.centroids.bind(enc, index: 1)
                        csr.clusterSpans.bind(enc, index: 2)
                        cScratch.cScore.bind(enc, index: 3)
                        cScratch.uHist.bind(enc, index: 4)
                        enc.setBytes(&ePrm, length: plen, index: 5)
                        let grid = useCoarseTiled
                            ? MTLSize(width: (subBg.numClusters + 31) / 32, height: (subRows + 7) / 8, depth: 1)
                            : MTLSize(width: (subBg.numClusters + 255) / 256, height: subRows, depth: 1)
                        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                    }
                    QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
                        enc, pipelines: pipes, csr: csr, scratch: cScratch,
                        blockList: listSlice, blockCount: listSlice, params: &ePrm,
                        numClusters: subBg.numClusters, maxMiddlePools: subBg.middleEnd,
                        encodeScoreCands: { enc, maxCand, prm in
                            enc.setComputePipelineState(psoCands)
                            qSlice.bind(enc, index: 0)
                            pooled.bind(enc, index: 1)
                            cScratch.candState.bind(enc, index: 2)
                            cScratch.candPools.bind(enc, index: 3)
                            cScratch.candU.bind(enc, index: 4)
                            enc.setBytes(&prm, length: plen, index: 5)
                            enc.dispatchThreadgroups(
                                MTLSize(width: (maxCand + 255) / 256, height: subRows, depth: 1),
                                threadsPerThreadgroup: tpg)
                        },
                        encodeFlatFallback: { enc, prm in
                            enc.setComputePipelineState(psoFallback)
                            qSlice.bind(enc, index: 0)
                            pooled.bind(enc, index: 1)
                            cScratch.candState.bind(enc, index: 2)
                            listSlice.bind(enc, index: 3)
                            listSlice.bind(enc, index: 4)
                            enc.setBytes(&prm, length: plen, index: 5)
                            enc.dispatchThreadgroups(
                                MTLSize(width: subRows, height: 1, depth: 1),
                                threadsPerThreadgroup: tpg)
                        }
                    )
                    if r0 + subRows < rows {
                        QWCoarsePoolIndex.barrierIfConcurrent(enc)
                    }
                    r0 += subRows
                }
                return
            }
        }
        let maxBlocks = max(1, (pos0 + rows) / 4)
        let totalScoreElems = scores.shape.elementCount
        let rawTileRows = max(1, min(rows, totalScoreElems / max(1, maxBlocks)))
        let tileAlignMask = QWQSAEnvironmentToggles.qsaMatEnabled ? ~15 : ~7
        let tileRows = (rawTileRows >= 16 && rows > rawTileRows) ? (rawTileRows & tileAlignMask) : rawTileRows
        guard tileRows >= 1 && maxBlocks <= totalScoreElems else {
            throw QWQSAError.selectionLimitExceeded(positions: pos0 + rows, limit: totalScoreElems * 4)
        }
        if let clusters = lsaClusters, !clusters.isEmeans, (pos0 + rows) / 512 > 0 {
            let lsaPso = try hal.pipeline("qwg_lsa_qsa_select")
            var r0 = 0
            while r0 < rows {
                let subRows = min(tileRows, rows - r0)
                let subPos0 = pos0 + r0
                let qSlice = idxQ.slice(byteOffset: r0 * 4 * 128 * 4, count: subRows * 4 * 128, dtype: .f32)
                let listSlice = list.slice(byteOffset: r0 * blockTopK * 4, count: subRows * blockTopK, dtype: .i32)
                enc.setComputePipelineState(lsaPso)
                enc.bindTensors(
                    qSlice, pooled, clusters.centroids, clusters.bucketMembers,
                    clusters.bucketSizes, scores, listSlice, startIndex: 0)
                enc.setUInt32s(subRows, blockTopK, subPos0, maxBlocks, startIndex: 7)
                enc.dispatchThreadgroups(MTLSize(width: subRows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                if r0 + subRows < rows {
                    enc.memoryBarrier(scope: .buffers)
                }
                r0 += subRows
            }
            return
        }
        let useMatScores = QWQSAEnvironmentToggles.qsaMatEnabled && tileRows >= 16
        let useTiledScores = tileRows >= 8
        let scoreMatPso = useMatScores ? try hal.pipeline("qwg_qsa_scores_mat") : nil
        let scorePso = try hal.pipeline(useTiledScores ? "qwg_qsa_scores_tiled" : "qwg_qsa_scores")
        let radixPso = try hal.pipeline("qwg_qsa_topk_radix")
        var r0 = 0
        while r0 < rows {
            let subRows = min(tileRows, rows - r0)
            let subPos0 = pos0 + r0
            let subMaxBlocks = max(1, (subPos0 + subRows) / 4)
            let subNumChunks = max(1, (subMaxBlocks + 255) / 256)
            let qSlice = idxQ.slice(byteOffset: r0 * 4 * 128 * 4, count: subRows * 4 * 128, dtype: .f32)
            let listSlice = list.slice(byteOffset: r0 * blockTopK * 4, count: subRows * blockTopK, dtype: .i32)
            if let scoreMatPso, subRows >= 16 {
                enc.setComputePipelineState(scoreMatPso)
                enc.bindTensors(qSlice, pooled, scores, startIndex: 0)
                enc.setUInt32s(subRows, blockTopK, subPos0, subMaxBlocks, startIndex: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: (subMaxBlocks + 63) / 64, height: (subRows + 15) / 16, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else if useTiledScores && subRows >= 8 {
                enc.setComputePipelineState(scorePso)
                enc.bindTensors(qSlice, pooled, scores, startIndex: 0)
                enc.setUInt32s(subRows, blockTopK, subPos0, subMaxBlocks, startIndex: 3)
                enc.dispatchThreadgroups(MTLSize(width: subNumChunks, height: (subRows + 7) / 8, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                enc.setComputePipelineState(try hal.pipeline("qwg_qsa_scores"))
                enc.bindTensors(qSlice, pooled, scores, startIndex: 0)
                enc.setUInt32s(subRows, blockTopK, subPos0, subMaxBlocks, startIndex: 3)
                enc.dispatchThreadgroups(MTLSize(width: subNumChunks, height: subRows, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(radixPso)
            enc.bindTensors(scores, listSlice, startIndex: 0)
            enc.setUInt32s(subRows, blockTopK, subPos0, subMaxBlocks, startIndex: 2)
            enc.dispatchThreadgroups(MTLSize(width: subRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if r0 + subRows < rows {
                enc.memoryBarrier(scope: .buffers)
            }
            r0 += subRows
        }
    }

    /// Attention prep (`q`, `k`, `v`, `gate`, optional `indexK`) supporting KV modes (`.f32`, `.f16`, `.q8`, `.q4`).
    public static func encodeAttnPrepF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, src: QWTensorValue,
                                         qNorm: QWTensorValue, kNorm: QWTensorValue, cos: QWTensorValue,
                                         sin: QWTensorValue, q: QWTensorValue, k: QWTensorValue,
                                         v: QWTensorValue, gate: QWTensorValue, rows: Int, heads: Int,
                                         kvHeads: Int, rowStride: Int,
                                         idxK: QWTensorValue? = nil, ikOff: Int = 0,
                                         kvMode: FNKVStorageMode = .f32,
                                         kMeta: QWTensorValue? = nil, vMeta: QWTensorValue? = nil) throws {
        enc.setComputePipelineState(try hal.pipeline(kvMode.prepKernelName))
        enc.bindTensors(src, qNorm, kNorm, cos, sin, q, k, v, gate, startIndex: 0)
        enc.setUInt32s(rows, heads, kvHeads, rowStride, heads * 512, heads * 512 + kvHeads * 256, startIndex: 9)
        (idxK ?? v).bind(enc, index: 15)
        enc.setUInt32s(idxK != nil ? ikOff : 0, startIndex: 16)
        (kMeta ?? k).bind(enc, index: 17)
        (vMeta ?? v).bind(enc, index: 18)
        let perTok = heads + 2 * kvHeads + heads
        enc.dispatchThreadgroups(MTLSize(width: rows * perTok, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Computes wave-aligned Split-K split count for QSA NAX decode (`rows <= 8`).
    public static func recommendedNAXSplits(
        rows: Int,
        kvHeads: Int,
        pos0: Int,
        maxSplits: Int = 32,
        gpuCores: Int = 40
    ) -> Int {
        let clampedMax = max(1, maxSplits)
        let activeKeys = min(2051, max(1, pos0 + rows))
        let rawSplits = min(clampedMax, max(1, (activeKeys + 95) / 96))
        let baseTGs = max(1, rows * max(1, kvHeads))
        let targetCapTGs = min(rawSplits * baseTGs, 2 * max(1, gpuCores))
        if targetCapTGs <= gpuCores {
            return max(1, min(clampedMax, rawSplits))
        }
        let waves = max(1, targetCapTGs / max(1, gpuCores))
        let alignedSplits = max(1, (waves * max(1, gpuCores)) / baseTGs)
        return min(clampedMax, min(rawSplits, alignedSplits))
    }

    /// Full-attention core (`q`, `gate`, `out` f32; `k`, `v` in `.f32`, `.f16`, `.q8`, or `.q4`).
    public static func encodeAttnSparseGatedF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                q: QWTensorValue, k: QWTensorValue, v: QWTensorValue,
                                                gate: QWTensorValue, list: QWTensorValue, out: QWTensorValue,
                                                rows: Int, heads: Int, kvHeads: Int, pos0: Int,
                                                kvMode: FNKVStorageMode = .f32,
                                                kMeta: QWTensorValue? = nil, vMeta: QWTensorValue? = nil,
                                                out16: QWTensorValue? = nil,
                                                splitPartO: QWTensorValue? = nil,
                                                splitPartML: QWTensorValue? = nil,
                                                maxSplits: Int = 32,
                                                splits: Int? = nil) throws {
        let gqaRatio = heads / max(1, kvHeads)
        let usePrefillNAX = hal.caps.isMetal4TensorCapable
            && rows > 8 && gqaRatio <= 16 && heads % max(1, kvHeads) == 0
        let useSplitNAX = hal.caps.isMetal4TensorCapable
            && rows >= 1 && rows <= 8 && gqaRatio <= 16 && heads % max(1, kvHeads) == 0
            && pos0 >= 256 && maxSplits >= 1
        if useSplitNAX, let partO = splitPartO, let partML = splitPartML {
            let resolvedSplits = max(1, min(maxSplits, splits ?? recommendedNAXSplits(
                rows: rows, kvHeads: kvHeads, pos0: pos0, maxSplits: maxSplits, gpuCores: hal.caps.gpuCores)))
            let minPartOElems = rows * heads * resolvedSplits * 256
            let minPartMLElems = rows * heads * resolvedSplits * 2
            if partO.shape.elementCount >= minPartOElems && partML.shape.elementCount >= minPartMLElems {
                enc.setComputePipelineState(try hal.pipeline(kvMode.sparseGatedNAXSplitKernelName))
                enc.bindTensors(q, k, v, list, partO, partML, startIndex: 0)
                enc.setUInt32s(rows, heads, kvHeads, pos0, resolvedSplits, startIndex: 6)
                (kMeta ?? k).bind(enc, index: 11)
                (vMeta ?? v).bind(enc, index: 12)
                enc.dispatchThreadgroups(
                    MTLSize(width: rows, height: kvHeads, depth: resolvedSplits),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(try hal.pipeline(FNKVStorageMode.sparseGatedNAXCombineKernelName))
                enc.bindTensors(partO, partML, gate, out, startIndex: 0)
                enc.setUInt32s(rows, heads, resolvedSplits, startIndex: 4)
                (out16 ?? out).bind(enc, index: 7)
                enc.setUInt32s(out16 != nil ? 1 : 0, startIndex: 8)
                enc.dispatchThreadgroups(
                    MTLSize(width: rows * heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                return
            }
        }
        let useGQA3 = !usePrefillNAX && (rows >= 2 || pos0 >= 64) && heads % 3 == 0 && gqaRatio % 3 == 0
        let kernelName = usePrefillNAX
            ? kvMode.sparseGatedNAXKernelName
            : (useGQA3 ? kvMode.sparseGatedGQA3KernelName : kvMode.sparseGatedKernelName)
        enc.setComputePipelineState(try hal.pipeline(kernelName))
        enc.bindTensors(q, k, v, gate, list, out, startIndex: 0)
        enc.setUInt32s(rows, heads, kvHeads, pos0, startIndex: 6)
        (kMeta ?? k).bind(enc, index: 10)
        (vMeta ?? v).bind(enc, index: 11)
        (out16 ?? out).bind(enc, index: 12)
        enc.setUInt32s(out16 != nil ? 1 : 0, startIndex: 13)
        if usePrefillNAX {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: kvHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        } else {
            let tgCount = useGQA3 ? rows * (heads / 3) : rows * heads
            enc.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}

/// Transitional `QWGemm` forwarding seams for Wave 3A (`QWFlashNext` / `Tests`).
extension QWGemm {
    public static func encodeFnIndexPrep(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                         src: QWTensorValue, iqNorm: QWTensorValue, cos: QWTensorValue,
                                         sin: QWTensorValue, idxQ: QWTensorValue, idxK: QWTensorValue,
                                         rows: Int, rowStride: Int, iqOff: Int, ikOff: Int) throws {
        try QWQSAEncoders.encodeIndexPrep(enc, hal: hal, src: src, iqNorm: iqNorm, cos: cos, sin: sin,
                                          idxQ: idxQ, idxK: idxK, rows: rows, rowStride: rowStride,
                                          iqOff: iqOff, ikOff: ikOff)
    }

    public static func encodeFnIndexPool(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                         idxK: QWTensorValue, ikNorm: QWTensorValue, bcos: QWTensorValue,
                                         bsin: QWTensorValue, pooled: QWTensorValue,
                                         blocks: Int, compress: Int) throws {
        try QWQSAEncoders.encodeIndexPool(enc, hal: hal, idxK: idxK, ikNorm: ikNorm, bcos: bcos,
                                          bsin: bsin, pooled: pooled, blocks: blocks, compress: compress)
    }

    @discardableResult
    public static func encodeFnLSAClusterRange(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                               pooled: QWTensorValue,
                                               clusters: QWQSACacheState.LSAClusterState,
                                               pos0: Int, count: Int,
                                               indexDim: Int) throws -> Bool {
        try QWQSAEncoders.encodeLSAClusterRange(enc, hal: hal, pooled: pooled, clusters: clusters,
                                                pos0: pos0, count: count, indexDim: indexDim)
    }

    public static func encodeFnQSASelect(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                         idxQ: QWTensorValue, pooled: QWTensorValue, list: QWTensorValue,
                                         rows: Int, blockTopK: Int, pos0: Int = 0,
                                         scores: QWTensorValue,
                                         lsaClusters: QWQSACacheState.LSAClusterState? = nil) throws {
        try QWQSAEncoders.encodeQSASelect(enc, hal: hal, idxQ: idxQ, pooled: pooled, list: list,
                                          rows: rows, blockTopK: blockTopK, pos0: pos0,
                                          scores: scores, lsaClusters: lsaClusters)
    }

    public static func encodeFnAttnPrepF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, src: QWTensorValue,
                                           qNorm: QWTensorValue, kNorm: QWTensorValue, cos: QWTensorValue,
                                           sin: QWTensorValue, q: QWTensorValue, k: QWTensorValue,
                                           v: QWTensorValue, gate: QWTensorValue, rows: Int, heads: Int,
                                           kvHeads: Int, rowStride: Int,
                                           idxK: QWTensorValue? = nil, ikOff: Int = 0,
                                           kvMode: FNKVStorageMode = .f32,
                                           kMeta: QWTensorValue? = nil, vMeta: QWTensorValue? = nil) throws {
        try QWQSAEncoders.encodeAttnPrepF32(enc, hal: hal, src: src, qNorm: qNorm, kNorm: kNorm, cos: cos, sin: sin,
                                            q: q, k: k, v: v, gate: gate, rows: rows, heads: heads, kvHeads: kvHeads,
                                            rowStride: rowStride, idxK: idxK, ikOff: ikOff, kvMode: kvMode,
                                            kMeta: kMeta, vMeta: vMeta)
    }

    public static func encodeFnAttnSparseGatedF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                  q: QWTensorValue, k: QWTensorValue, v: QWTensorValue,
                                                  gate: QWTensorValue, list: QWTensorValue, out: QWTensorValue,
                                                  rows: Int, heads: Int, kvHeads: Int, pos0: Int,
                                                  kvMode: FNKVStorageMode = .f32,
                                                  kMeta: QWTensorValue? = nil, vMeta: QWTensorValue? = nil,
                                                  out16: QWTensorValue? = nil,
                                                  splitPartO: QWTensorValue? = nil,
                                                  splitPartML: QWTensorValue? = nil,
                                                  maxSplits: Int = 32,
                                                  splits: Int? = nil) throws {
        try QWQSAEncoders.encodeAttnSparseGatedF32(enc, hal: hal, q: q, k: k, v: v, gate: gate, list: list, out: out,
                                                   rows: rows, heads: heads, kvHeads: kvHeads, pos0: pos0,
                                                   kvMode: kvMode, kMeta: kMeta, vMeta: vMeta, out16: out16,
                                                   splitPartO: splitPartO, splitPartML: splitPartML,
                                                   maxSplits: maxSplits, splits: splits)
    }
}
