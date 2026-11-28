// QWMLAAttnOperator+DecoupledRoPE.swift — Decoupled-RoPE + DSA pipeline, shared
// indexer selection / absorb / proj_uv helpers, and DSA `.qwcache` state snapshot.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import QWGEMM
import QWKV

extension QWMLAAttnOperator {

    internal struct DSALatentPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qLoraRank: UInt32
        var kvLoraRank: UInt32
        var ropeDim: UInt32
        var idxDim: UInt32
        var latentStride: UInt32
        var kvMode: UInt32
        var rmsEps: Float
        var kNormEps: Float
        var mscale: Float
        var useHadamard: UInt32
        var ropeFlags: UInt32
        var gatedFlags: UInt32
        var qAGateRank: UInt32
        var kvAGateRank: UInt32
        var qAGatingWeight: Float
        var kvAGatingWeight: Float
        var alphaQ: Float
        var alphaKV: Float
    }

    internal struct QRopeIdxPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var idxHeads: UInt32
        var ropeDim: UInt32
        var idxDim: UInt32
        var qRopeOffset: UInt32
        var qChannels: UInt32
        var qFullStride: UInt32
        var mscale: Float
        var useHadamard: UInt32
        var ropeFlags: UInt32
    }

    internal struct DSASparseAttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var kvLoraRank: UInt32
        var topK: UInt32
        var windowTokens: UInt32
        var listStride: UInt32
        var kvMode: UInt32
        var ropeDim: UInt32
        var qRopeOffset: UInt32
        var qFullStride: UInt32
        var hasSink: UInt32
        var softmaxScale: Float
    }

    internal struct VHAGateParams {
        var T: UInt32
        var qHeads: UInt32
        var vHeadDim: UInt32
        var gateStride: UInt32
        var oGateOffset: UInt32
        var hasVHA: UInt32
        var hasOGate: UInt32
    }

    internal struct HeadGateProjParams {
        var T: UInt32
        var hidden: UInt32
        var nGate: UInt32
    }

    internal func stageMRoPEPositions(_ mropePositions: [Int32]?, rows: Int) -> Bool {
        guard mropeLayout != .none1D, let pos = mropePositions, pos.count >= 3 * rows, rows > 0 else {
            return false
        }
        let dst = region.pointer.advanced(by: mropePosScratch.offset)
        pos.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress {
                dst.copyMemory(from: src, byteCount: 3 * rows * MemoryLayout<Int32>.stride)
            }
        }
        return true
    }

    internal func captureDSAIndexerState(position: Int) -> Data {
        let capLimit = max(maxBlocks, maxContext)
        precondition(position >= 0 && position <= capLimit)
        let idxBytes = position * idxDim * 2
        let peBytes = position * ropeDim * 2
        var out = Data(capacity: indexerStateBytes(position: position))
        let base = region.pointer
        for ls in layers {
            if ls.hasIndexer, idxBytes > 0 {
                out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self),
                           count: idxBytes)
            }
            if peBytes > 0, let kPe = ls.kPeCache {
                out.append(base.advanced(by: kPe.offset).assumingMemoryBound(to: UInt8.self),
                           count: peBytes)
            }
        }
        return out
    }

    internal func restoreDSAIndexerState(_ data: Data, position: Int) -> Bool {
        let capLimit = max(maxBlocks, maxContext)
        guard position >= 0, position <= capLimit,
              data.count == indexerStateBytes(position: position) else { return false }
        let idxBytes = position * idxDim * 2
        let peBytes = position * ropeDim * 2
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let srcBase = raw.baseAddress else { return }
            var src = 0
            for ls in layers {
                if ls.hasIndexer, idxBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset)
                        .copyMemory(from: srcBase.advanced(by: src), byteCount: idxBytes)
                    src += idxBytes
                }
                if peBytes > 0, let kPe = ls.kPeCache {
                    base.advanced(by: kPe.offset)
                        .copyMemory(from: srcBase.advanced(by: src), byteCount: peBytes)
                    src += peBytes
                }
            }
        }
        resetCoarseWatermarks()
        return true
    }

    internal func liveDSASessionStashBytes(trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Int {
        let idxStride = idxDim * 2
        let peStride = ropeDim * 2
        var total = 0
        for (i, ls) in layers.enumerated() {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            if ls.hasIndexer { total += pos * idxStride }
            if ls.kPeCache != nil { total += pos * peStride }
        }
        return total
    }

    internal func captureDSALiveSessionStash(trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Data {
        precondition(trunkPosition >= 0 && mtpPosition >= 0)
        let capLimit = max(maxBlocks, maxContext)
        var out = Data(capacity: liveDSASessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition))
        let base = region.pointer
        for (i, ls) in layers.enumerated() {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            precondition(pos <= capLimit)
            let idxBytes = ls.hasIndexer ? (pos * idxDim * 2) : 0
            let peBytes = pos * ropeDim * 2
            if idxBytes > 0 {
                out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self), count: idxBytes)
            }
            if peBytes > 0, let kPe = ls.kPeCache {
                out.append(base.advanced(by: kPe.offset).assumingMemoryBound(to: UInt8.self), count: peBytes)
            }
        }
        return out
    }

    internal func restoreDSALiveSessionStash(_ data: Data, trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Bool {
        let capLimit = max(maxBlocks, maxContext)
        guard trunkPosition >= 0, mtpPosition >= 0,
              data.count == liveDSASessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition) else {
            return false
        }
        for (i, ls) in layers.enumerated() {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            if pos > (ls.hasIndexer ? maxBlocks : capLimit) { return false }
        }
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let srcBase = raw.baseAddress else { return }
            var src = 0
            for (i, ls) in layers.enumerated() {
                let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                let idxBytes = ls.hasIndexer ? (pos * idxDim * 2) : 0
                let peBytes = pos * ropeDim * 2
                if idxBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: idxBytes)
                    src += idxBytes
                }
                if peBytes > 0, let kPe = ls.kPeCache {
                    base.advanced(by: kPe.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: peBytes)
                    src += peBytes
                }
            }
        }
        resetCoarseWatermarks()
        return true
    }

    internal func encodeDSALatentPrepAndCluster(_ enc: any MTLComputeCommandEncoder,
                                                ln1: QWTensorValue,
                                                mlaW: LayerWeights,
                                                kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                fullAttnLayer: Int,
                                                pos0: Int,
                                                rows: Int,
                                                mropePositions: [Int32]? = nil) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        let ls = layers[fullAttnLayer]
        let kPe = ls.kPeCache ?? mlaLatentScratch
        let hasMRoPE = stageMRoPEPositions(mropePositions, rows: rows)
        var dsaP = buildDSALatentPrepParams(
            mlaW: mlaW, state: ls, kvMode: kv.k.mode, pos0: pos0, rows: rows, hasMRoPE: hasMRoPE
        )
        let qAW = mlaW.qAGatedNorm?.weight ?? mlaW.qANormW
        let kvAW = mlaW.kvAGatedNorm?.weight ?? mlaW.kvANormW
        let tgHBytes = ((max(512, max(qLoraRank, kvLoraRank)) * 2 + 15) & ~15)

        enc.setComputePipelineState(psoDSALatentPrep)
        mlaLatentScratch.bind(enc, index: 0)
        qAW.bind(enc, index: 1)
        kvAW.bind(enc, index: 2)
        mlaW.kNormW.bind(enc, index: 3)
        mlaW.kNormB.bind(enc, index: 4)
        invFreqBuf.bind(enc, index: 5)
        qResidScratch.bind(enc, index: 6)
        kv.k.pool.bindTo(enc, index: 7)
        kv.v.pool.bindTo(enc, index: 8)
        kv.k.table.bind(enc, index: 9)
        kv.v.table.bind(enc, index: 10)
        kPe.bind(enc, index: 11)
        ls.pooledKeys.bind(enc, index: 12)
        enc.setBytes(&dsaP, length: MemoryLayout<DSALatentPrepParams>.stride, index: 13)
        mropePosScratch.bind(enc, index: 14)
        ropeAxisMapBuf.bind(enc, index: 15)
        (mlaW.qAGatedNorm?.gateUp ?? qAW).bind(enc, index: 16)
        (mlaW.qAGatedNorm?.gateDown ?? qAW).bind(enc, index: 17)
        (mlaW.kvAGatedNorm?.gateUp ?? kvAW).bind(enc, index: 18)
        (mlaW.kvAGatedNorm?.gateDown ?? kvAW).bind(enc, index: 19)
        enc.setThreadgroupMemoryLength(tgHBytes, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let effDense = isLayerDenseOnly(mlaW, state: ls)
        if !effDense, idxHeads > 0, idxDim > 0 {
            var gwP = LatentPrepParams(
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
                kvMode: Self.kvModeCode(kv.k.mode) | 0x100
            )
            enc.setComputePipelineState(psoGateWeightsProj)
            ln1.bind(enc, index: 0)
            mlaW.wGateWeights.bind(enc, index: 1)
            ls.gatePending.bind(enc, index: 2)
            idxWeightsScratch.bind(enc, index: 3)
            enc.setBytes(&gwP, length: MemoryLayout<LatentPrepParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (idxHeads + 7) / 8, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            updateCoarseClusters(enc, ls: ls, fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
        }
    }

    internal func encodeIndexerSelection(_ enc: any MTLComputeCommandEncoder,
                                         ls: LayerState,
                                         pos0: Int,
                                         rows: Int,
                                         probeClusters: Int,
                                         windowBlocks: Int,
                                         qFullOverride: QWTensorValue? = nil,
                                         idxWeightsOverride: QWTensorValue? = nil,
                                         activeIdxHeads: Int? = nil,
                                         activeQChannels: Int? = nil) {
        let tg256 = MTLSize(width: 256, height: 1, depth: 1)
        let selQFull = qFullOverride ?? mlaQFullScratch
        let selIdxW = idxWeightsOverride ?? idxWeightsScratch
        let effIdxHeads = activeIdxHeads ?? idxHeads
        let effQChannels = activeQChannels ?? qChannels
        let effQFullStride = effQChannels + effIdxHeads * idxDim
        let effSplit = (effIdxHeads == 32 || effIdxHeads == 64) ? psoIndexerSplit : nil
        let packedIdxHeads: UInt32 = (poolShift == 2 && windowBlocks == 0)
            ? UInt32(effIdxHeads)
            : (UInt32(effIdxHeads & 0xFF) | 0x1000 | (UInt32(poolShift & 0xF) << 8) | (UInt32(windowBlocks) << 16))
        var rowStart = isDecoupledDSA ? max(0, (topKPools + windowBlocks) - pos0) : 0
        while rowStart < rows {
            let batchRows = min(selectBatchRows, rows - rowStart)
            let qFullSub = selQFull.slice(byteOffset: rowStart * effQFullStride * 2,
                                          count: batchRows * effQFullStride, dtype: .f16)
            let idxWSub = selIdxW.slice(byteOffset: rowStart * effIdxHeads * 4,
                                        count: batchRows * effIdxHeads, dtype: .f32)
            let listSub = blockListScratch.slice(byteOffset: rowStart * listStride * 4,
                                                 count: batchRows * listStride, dtype: .i32)
            let countSub = blockCountScratch.slice(byteOffset: rowStart * 4,
                                                   count: batchRows, dtype: .i32)
            if emeansCoarseEnabled,
               let pipes = coarsePipelines,
               let mlaCoarse = psoEmeansCoarse,
               let csr = ls.coarseCSR,
               let cScratch = coarseScratch {
                let bg = QWCoarsePoolIndex.batchGeometry(
                    pos0: pos0 + rowStart, units: batchRows, budgetBlocks: topKPools,
                    windowBlocks: windowBlocks, hasSink: false, writeCount: true,
                    probeClusters: probeClusters, poolShift: poolShift)
                if bg.shouldPrune {
                    var ePrm = QWCoarsePoolIndex.SelectParams(
                        T: batchRows, pos0: pos0 + rowStart, indexDim: idxDim,
                        budgetBlocks: topKPools, windowBlocks: windowBlocks, hasSink: false,
                        writeCount: true, listStride: listStride, rowStep: 1, maxRow: batchRows,
                        clusterStride: cScratch.clusterStride, probeClusters: probeClusters,
                        candCap: cScratch.candCap, qChannels: effQChannels, idxHeads: effIdxHeads,
                        poolShift: poolShift)
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

            let maxPools = (pos0 + rowStart + batchRows) >> poolShift
            if maxPools > topKPools + windowBlocks {
                var idxP = IndexerParams(
                    T: UInt32(batchRows),
                    pos0: UInt32(pos0 + rowStart),
                    qChannels: UInt32(effQChannels),
                    idxHeads: packedIdxHeads,
                    idxDim: UInt32(idxDim),
                    topKPools: UInt32(topKPools),
                    listStride: UInt32(listStride),
                    scoreStride: UInt32(maxBlocks)
                )
                qFullSub.bind(enc, index: 0)
                idxWSub.bind(enc, index: 1)
                ls.pooledKeys.bind(enc, index: 2)
                scoreScratch.bind(enc, index: 3)
                listSub.bind(enc, index: 4)
                countSub.bind(enc, index: 5)
                enc.setBytes(&idxP, length: MemoryLayout<IndexerParams>.stride, index: 6)
                if let split = effSplit {
                    let midPools = maxPools - windowBlocks
                    enc.setComputePipelineState(split.score)
                    enc.dispatchThreadgroups(MTLSize(width: (midPools + 63) / 64, height: batchRows, depth: 1),
                                             threadsPerThreadgroup: tg256)
                    QWCoarsePoolIndex.barrierIfConcurrent(enc)
                    enc.setComputePipelineState(split.select)
                } else {
                    enc.setComputePipelineState(psoIndexerScoreSelect)
                }
                enc.dispatchThreadgroups(MTLSize(width: batchRows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
                QWCoarsePoolIndex.barrierIfConcurrent(enc)
            } else if !isDecoupledDSA {
                var idxP = IndexerParams(
                    T: UInt32(batchRows),
                    pos0: UInt32(pos0 + rowStart),
                    qChannels: UInt32(effQChannels),
                    idxHeads: packedIdxHeads,
                    idxDim: UInt32(idxDim),
                    topKPools: UInt32(topKPools),
                    listStride: UInt32(listStride),
                    scoreStride: UInt32(maxBlocks)
                )
                qFullSub.bind(enc, index: 0)
                idxWSub.bind(enc, index: 1)
                ls.pooledKeys.bind(enc, index: 2)
                scoreScratch.bind(enc, index: 3)
                listSub.bind(enc, index: 4)
                countSub.bind(enc, index: 5)
                enc.setBytes(&idxP, length: MemoryLayout<IndexerParams>.stride, index: 6)
                enc.setComputePipelineState(effSplit?.select ?? psoIndexerScoreSelect)
                enc.dispatchThreadgroups(MTLSize(width: batchRows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
                QWCoarsePoolIndex.barrierIfConcurrent(enc)
            }
            rowStart += batchRows
        }
    }

    internal func encodeDSAScoreSelectAbsorbAttendAndProjUV(_ enc: any MTLComputeCommandEncoder,
                                                            mlaW: LayerWeights,
                                                            kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                            attOut: QWTensorValue,
                                                            fullAttnLayer: Int,
                                                            pos0: Int,
                                                            rows: Int,
                                                            ln1: QWTensorValue? = nil,
                                                            mropePositions: [Int32]? = nil,
                                                            externalGatePrecomputed: Bool = false,
                                                            stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[fullAttnLayer]
        let kPe = ls.kPeCache ?? mlaLatentScratch
        let hasMRoPE = stageMRoPEPositions(mropePositions, rows: rows)
        let ropeFlags: UInt32 = (ropeInterleave ? 1 : 0) | (hasMRoPE ? 2 : 0)
        let effDense = isLayerDenseOnly(mlaW, state: ls)
        let effIdxHeads = effDense ? 0 : idxHeads
        let effIdxDim = effDense ? 0 : idxDim
        let effTopK = effDense ? 0 : topKPools
        let callQChannels = qChannels(for: mlaW)
        let callQFullStride = qFullStride(for: mlaW, state: ls)
        let effOGateInQKV = oGateInQKV || mlaW.oGateInQKV
        let callLatStride = latentStride(for: mlaW, state: ls)

        // 0. In-place RoPE (GPT-J or NeoX; 1D or 3D MRoPE) on q_rope + optional NeoX/H_128 on q_idx.
        mark("mla.q_rope_idx")
        var qrP = QRopeIdxPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            idxHeads: UInt32(effIdxHeads),
            ropeDim: UInt32(ropeDim),
            idxDim: UInt32(effIdxDim),
            qRopeOffset: UInt32(qRopeOffset),
            qChannels: UInt32(callQChannels),
            qFullStride: UInt32(callQFullStride),
            mscale: ropeMscale,
            useHadamard: useFastHadamard ? 1 : 0,
            ropeFlags: ropeFlags
        )
        enc.setComputePipelineState(psoQRopeIdxPrep)
        mlaQFullScratch.bind(enc, index: 0)
        invFreqBuf.bind(enc, index: 1)
        enc.setBytes(&qrP, length: MemoryLayout<QRopeIdxPrepParams>.stride, index: 2)
        mropePosScratch.bind(enc, index: 3)
        ropeAxisMapBuf.bind(enc, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: qHeads + effIdxHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        QWCoarsePoolIndex.barrierIfConcurrent(enc)

        let extGateMode = !effOGateInQKV && (
            outputGateMode == .headWiseFromHidden
            || outputGateMode == .elementWiseFromHidden
            || mlaW.oGateProj != nil
            || mlaW.oGateF16 != nil
        )
        let extGateCols = outputGateMode == .headWiseFromHidden ? qHeads : (qHeads * vHeadDim)
        if !externalGatePrecomputed, extGateMode, let ln1 {
            if let gateProj = mlaW.oGateProj {
                let gateOut = oGateScratch.slice(byteOffset: 0, count: rows * extGateCols, dtype: .f16)
                try? encodeMLAWeightViewH16(enc, x: ln1, weight: gateProj, y16: gateOut, rows: rows)
            } else if let gateF16 = mlaW.oGateF16 {
                var hgP = HeadGateProjParams(
                    T: UInt32(rows),
                    hidden: UInt32(geometry.hidden),
                    nGate: UInt32(extGateCols)
                )
                enc.setComputePipelineState(psoHeadGateProj)
                ln1.bind(enc, index: 0)
                gateF16.bind(enc, index: 1)
                oGateScratch.bind(enc, index: 2)
                enc.setBytes(&hgP, length: MemoryLayout<HeadGateProjParams>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: extGateCols, height: rows, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }

        // 1. Exact dense bypass at seqLen <= topK + windowTokens (or O(1)-SRAM dense stream when effDense).
        mark("mla.score_select")
        if !effDense, pos0 + rows > effTopK + windowTokens {
            let useFlashIndexer = flashIndexerEnabled
                && rows >= 16
                && effIdxHeads > Self.flashIndexerSelHeads
                && effIdxHeads <= 64
                && geometry.hidden > 0
            if useFlashIndexer,
               let gateW = mlaW.flashGateW,
               let bias = mlaW.flashBias,
               let ln1 {
                enc.useResource(gateW.region.buffer, usage: .read)
                enc.useResource(bias.region.buffer, usage: .read)
                var fiP = FlashIndexerParams(
                    T: UInt32(rows),
                    hidden: UInt32(geometry.hidden),
                    fullIdxHeads: UInt32(effIdxHeads),
                    selHeads: UInt32(Self.flashIndexerSelHeads),
                    idxDim: UInt32(effIdxDim),
                    qChannels: UInt32(callQChannels),
                    qFullStride: UInt32(callQFullStride)
                )
                enc.setComputePipelineState(psoFlashIndexerSelectHeads)
                ln1.bind(enc, index: 0)
                gateW.bind(enc, index: 1)
                bias.bind(enc, index: 2)
                idxWeightsScratch.bind(enc, index: 3)
                mlaQFullScratch.bind(enc, index: 4)
                flashQSelScratch.bind(enc, index: 5)
                flashWSelScratch.bind(enc, index: 6)
                enc.setBytes(&fiP, length: MemoryLayout<FlashIndexerParams>.stride, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: (rows + 31) / 32, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                QWCoarsePoolIndex.barrierIfConcurrent(enc)
                encodeIndexerSelection(enc, ls: ls, pos0: pos0, rows: rows,
                                       probeClusters: Self.dsaEmeansProbeClusters,
                                       windowBlocks: windowTokens,
                                       qFullOverride: flashQSelScratch,
                                       idxWeightsOverride: flashWSelScratch,
                                       activeIdxHeads: Self.flashIndexerSelHeads,
                                       activeQChannels: 0)
            } else {
                encodeIndexerSelection(enc, ls: ls, pos0: pos0, rows: rows,
                                       probeClusters: Self.dsaEmeansProbeClusters,
                                       windowBlocks: windowTokens,
                                       activeIdxHeads: effIdxHeads,
                                       activeQChannels: callQChannels)
            }
        }

        // 2. Absorb W_{UK, h} into q_nope -> qAbs [rows, qHeads, kvLoraRank]
        let baseInc: UInt32 = mlaW.kvBProj.bits == 16
            ? 3
            : (mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0))
        let hasInlineOrFlagGate = mlaW.hasOGate
            || outputGateMode == .elementWiseFromQLora
            || (geometry.mlaConfig == nil && geometry.attnOutputGate)
        let hasAnyOGate = effOGateInQKV || extGateMode || hasInlineOrFlagGate
        let needsPostEpilogue = (mlaW.vhaMix != nil) || hasAnyOGate
        var absP = AbsorbParams(
            T: UInt32(rows),
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank),
            qFullStride: UInt32(callQFullStride),
            bits: UInt32(mlaW.kvBProj.bits),
            group: UInt32(mlaW.kvBProj.groupSize),
            isIncumbent: baseInc | (needsPostEpilogue ? 0x10000 : 0)
        )
        let mTiles = (rows + 7) / 8
        mark("mla.absorb_q")
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: rows, mTiles: mTiles)

        // 3. Decoupled-RoPE DSA or dense-stream attention (c_KV + k_pe + attn_sink) -> uLat
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        let sinkTensor = mlaW.attnSink ?? zeroSinkBuf
        var attnP = DSASparseAttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(nopeDim + ropeDim),
            kvLoraRank: UInt32(kvLoraRank),
            topK: UInt32(effTopK),
            windowTokens: UInt32(windowTokens),
            listStride: UInt32(listStride),
            kvMode: Self.kvModeCode(kv.k.mode),
            ropeDim: UInt32(ropeDim),
            qRopeOffset: UInt32(qRopeOffset),
            qFullStride: UInt32(callQFullStride),
            hasSink: mlaW.attnSink != nil ? 1 : 0,
            softmaxScale: effectiveSoftmaxScale(for: mlaW)
        )
        mark("mla.sparse_attn")
        let supportsNAXRopeDim = (ropeDim == 0 || ropeDim == 32 || ropeDim == 64 || ropeDim == 128)
        let supportsMMARopeDim = (ropeDim == 32 || ropeDim == 64)
        let naxAvail = (psoDSASparseAttnNAX != nil) && supportsNAXRopeDim
        let useMMA = sparseAttnMMAEnabled && attnP.kvMode <= 2 && rows >= 16
            && kvLoraRank == 512 && qHeads % 16 == 0
            && (naxAvail || supportsMMARopeDim)
        let split = (sparseAttnMMAEnabled && !useMMA && attnP.kvMode <= 2 && rows <= Self.splitMaxRows
            && qHeads % 16 == 0 && kvLoraRank == 512 && supportsNAXRopeDim) ? psoDSASparseAttnSplit : nil
        let allowShared = dsaSharedMode != "0" && dsaSharedMode != "h16"
        let useH64 = (split == nil) && useMMA && naxAvail && allowShared && dsaSharedMode != "h32"
            && qHeads % 64 == 0 && psoDSASparseAttnNAXH64 != nil
        let useH32 = (split == nil) && useMMA && naxAvail && allowShared && !useH64
            && qHeads % 32 == 0 && psoDSASparseAttnNAXH32 != nil
        let dsaNaxPSO = useH64 ? psoDSASparseAttnNAXH64 : (useH32 ? psoDSASparseAttnNAXH32 : psoDSASparseAttnNAX)
        enc.setComputePipelineState(split?.attend ?? (useMMA ? (naxAvail ? dsaNaxPSO! : psoDSASparseAttnMMA) : psoDSASparseAttn))
        qAbsScratch.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        blockListScratch.bind(enc, index: 5)
        (split != nil ? partOScratch : uLatScratch).bind(enc, index: 6)
        enc.setBytes(&attnP, length: MemoryLayout<DSASparseAttnParams>.stride, index: 7)
        if let split {
            partMLScratch.bind(enc, index: 8)
            mlaQFullScratch.bind(enc, index: 9)
            kPe.bind(enc, index: 10)
            sinkTensor.bind(enc, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: attnSplits),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            var nSplitU = UInt32(attnSplits)
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            enc.setComputePipelineState(split.combine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            uLatScratch.bind(enc, index: 2)
            enc.setBytes(&attnP, length: MemoryLayout<DSASparseAttnParams>.stride, index: 3)
            enc.setBytes(&nSplitU, length: 4, index: 4)
            sinkTensor.bind(enc, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if useMMA {
            mlaQFullScratch.bind(enc, index: 8)
            kPe.bind(enc, index: 9)
            sinkTensor.bind(enc, index: 10)
            if useH64 {
                enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 64, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            } else if useH32 {
                enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 32, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        } else {
            mlaQFullScratch.bind(enc, index: 8)
            kPe.bind(enc, index: 9)
            sinkTensor.bind(enc, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // 4. Value up-projection -> yVhaScratch (if VHA/o_gate active) or directly attOut
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        mark("mla.proj_uv")
        let projDst = needsPostEpilogue ? yVhaScratch : attOut
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: projDst, absP: &absP,
                     absorbMMA: absorbMMA, rows: rows, mTiles: mTiles)

        if needsPostEpilogue {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            mark("mla.vha_gate")
            let gateCode: UInt32 = switch outputGateMode {
            case .headWiseFromHidden: 2
            case .elementWiseFromHidden, .elementWiseFromQLora: 1
            case .none: hasAnyOGate ? 1 : 0
            }
            let gateSrc: QWTensorValue
            let resolvedGateStride: Int
            let resolvedGateOffset: Int
            if effOGateInQKV {
                gateSrc = mlaLatentScratch
                resolvedGateStride = callLatStride
                resolvedGateOffset = qLoraRank + kvLoraRank + ropeDim + effIdxDim
            } else if extGateMode {
                gateSrc = oGateScratch
                resolvedGateStride = extGateCols
                resolvedGateOffset = 0
            } else {
                gateSrc = mlaQFullScratch
                resolvedGateStride = callQFullStride
                resolvedGateOffset = qHeads * (nopeDim + ropeDim)
            }
            var vgP = VHAGateParams(
                T: UInt32(rows),
                qHeads: UInt32(qHeads),
                vHeadDim: UInt32(vHeadDim),
                gateStride: UInt32(resolvedGateStride),
                oGateOffset: UInt32(resolvedGateOffset),
                hasVHA: mlaW.vhaMix != nil ? 1 : 0,
                hasOGate: gateCode
            )
            enc.setComputePipelineState(psoVHAGateRot)
            yVhaScratch.bind(enc, index: 0)
            gateSrc.bind(enc, index: 1)
            (mlaW.vhaMix ?? yVhaScratch).bind(enc, index: 2)
            attOut.bind(enc, index: 3)
            enc.setBytes(&vgP, length: MemoryLayout<VHAGateParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: vHeadDim / 32, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
