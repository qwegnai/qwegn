import Metal
import QWAlloc
import QWGEMM
import QWHAL

// FlashNext full-attention core dispatch (Sources/QWMetal/qwattn_sparse.metal).
extension QWGemm {
    /// Indexer q/k prep from the grouped projection output: q heads RMSNorm + partial RoPE; raw k copied.
    public static func encodeFnIndexPrep(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
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
    public static func encodeFnIndexPool(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
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
    public static func encodeFnLSAWardCluster(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                              pooled: QWTensorValue,
                                              clusters: FNGPUAttnDecodeState.LSAClusterState,
                                              startSuperblock: Int, numSuperblocks: Int,
                                              indexDim: Int) throws {
        guard numSuperblocks > 0 else { return }
        enc.setComputePipelineState(try hal.pipeline("qwg_lsa_ward_cluster_512"))
        enc.bindTensors(pooled, clusters.centroids, clusters.bucketMembers, clusters.bucketSizes, startIndex: 0)
        enc.setUInt32s(startSuperblock, numSuperblocks, indexDim, startIndex: 4)
        enc.dispatchThreadgroups(MTLSize(width: numSuperblocks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Exact 2-stage QSA top-k per query over complete blocks (or hierarchical Ward-clustered LSA selector when `lsaClusters != nil`);
    /// list i32 [T,512] receives ascending selected block ids.
    public static func encodeFnQSASelect(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                         idxQ: QWTensorValue, pooled: QWTensorValue, list: QWTensorValue,
                                         rows: Int, blockTopK: Int, pos0: Int = 0,
                                         scores: QWTensorValue,
                                         lsaClusters: FNGPUAttnDecodeState.LSAClusterState? = nil) throws {
        let maxBlocks = max(1, (pos0 + rows) / 4)
        let totalScoreElems = scores.shape.elementCount
        let rawTileRows = max(1, min(rows, totalScoreElems / max(1, maxBlocks)))
        let tileRows = (rawTileRows >= 16 && rows > rawTileRows) ? (rawTileRows & ~7) : rawTileRows
        guard tileRows >= 1 && maxBlocks <= totalScoreElems else {
            throw FNGPUError.selectionLimitExceeded(positions: pos0 + rows, limit: totalScoreElems * 4)
        }
        if let clusters = lsaClusters, (pos0 + rows) / 512 > 0 {
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
        let useTiledScores = tileRows >= 8
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
            if useTiledScores && subRows >= 8 {
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

    /// Attention prep (q, k, v, gate, optional indexK) supporting KV cache modes (.f32, .f16, .q8, .q4).
    public static func encodeFnAttnPrepF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal, src: QWTensorValue,
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

    /// Full-attention core (q, gate, out f32; k, v in .f32, .f16, .q8, or .q4).
    public static func encodeFnAttnSparseGatedF32(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                                  q: QWTensorValue, k: QWTensorValue, v: QWTensorValue,
                                                  gate: QWTensorValue, list: QWTensorValue, out: QWTensorValue,
                                                  rows: Int, heads: Int, kvHeads: Int, pos0: Int,
                                                  kvMode: FNKVStorageMode = .f32,
                                                  kMeta: QWTensorValue? = nil, vMeta: QWTensorValue? = nil,
                                                  out16: QWTensorValue? = nil) throws {
        let gqaRatio = heads / max(1, kvHeads)
        let useNAX = rows > 8 && gqaRatio <= 16 && heads % max(1, kvHeads) == 0
        let useGQA3 = !useNAX && (rows >= 2 || pos0 >= 64) && heads % 3 == 0 && gqaRatio % 3 == 0
        let kernelName = useNAX
            ? kvMode.sparseGatedNAXKernelName
            : (useGQA3 ? kvMode.sparseGatedGQA3KernelName : kvMode.sparseGatedKernelName)
        enc.setComputePipelineState(try hal.pipeline(kernelName))
        enc.bindTensors(q, k, v, gate, list, out, startIndex: 0)
        enc.setUInt32s(rows, heads, kvHeads, pos0, startIndex: 6)
        (kMeta ?? k).bind(enc, index: 10)
        (vMeta ?? v).bind(enc, index: 11)
        (out16 ?? out).bind(enc, index: 12)
        enc.setUInt32s(out16 != nil ? 1 : 0, startIndex: 13)
        if useNAX {
            enc.dispatchThreadgroups(MTLSize(width: rows, height: kvHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        } else {
            let tgCount = useGQA3 ? rows * (heads / 3) : rows * heads
            enc.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
