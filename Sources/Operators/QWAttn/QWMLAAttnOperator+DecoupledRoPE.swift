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

    private struct DSALatentPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qLoraRank: UInt32
        var kvLoraRank: UInt32
        var ropeDim: UInt32
        var idxDim: UInt32
        var rmsEps: Float
        var kNormEps: Float
        var kvMode: UInt32
        var mscale: Float
        var useHadamard: UInt32
    }

    private struct QRopeIdxPrepParams {
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
    }

    private struct DSASparseAttnParams {
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
    }

    private struct VHAGateParams {
        var T: UInt32
        var qHeads: UInt32
        var vHeadDim: UInt32
        var qFullStride: UInt32
        var oGateOffset: UInt32
        var hasVHA: UInt32
        var hasOGate: UInt32
    }

    internal func captureDSAIndexerState(position: Int) -> Data {
        precondition(position >= 0 && position <= maxBlocks)
        let idxBytes = position * idxDim * 2
        let peBytes = position * ropeDim * 2
        var out = Data(capacity: indexerStateBytes(position: position))
        let base = region.pointer
        for ls in layers {
            if idxBytes > 0 {
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
        guard position >= 0, position <= maxBlocks,
              data.count == indexerStateBytes(position: position) else { return false }
        let idxBytes = position * idxDim * 2
        let peBytes = position * ropeDim * 2
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let srcBase = raw.baseAddress else { return }
            var src = 0
            for ls in layers {
                if idxBytes > 0 {
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

    internal func encodeDSALatentPrepAndCluster(_ enc: any MTLComputeCommandEncoder,
                                                ln1: QWTensorValue,
                                                mlaW: LayerWeights,
                                                kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                fullAttnLayer: Int,
                                                pos0: Int,
                                                rows: Int) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0,
              let kPe = layers[fullAttnLayer].kPeCache else { return }
        let ls = layers[fullAttnLayer]
        var dsaP = DSALatentPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qLoraRank: UInt32(qLoraRank),
            kvLoraRank: UInt32(kvLoraRank),
            ropeDim: UInt32(ropeDim),
            idxDim: UInt32(idxDim),
            rmsEps: geometry.rmsEps,
            kNormEps: 1e-6,
            kvMode: Self.kvModeCode(kv.k.mode),
            mscale: ropeMscale,
            useHadamard: useFastHadamard ? 1 : 0
        )

        enc.setComputePipelineState(psoDSALatentPrep)
        mlaLatentScratch.bind(enc, index: 0)
        mlaW.qANormW.bind(enc, index: 1)
        mlaW.kvANormW.bind(enc, index: 2)
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
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

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

    internal func encodeIndexerSelection(_ enc: any MTLComputeCommandEncoder,
                                         ls: LayerState,
                                         pos0: Int,
                                         rows: Int,
                                         probeClusters: Int,
                                         windowBlocks: Int) {
        let tg256 = MTLSize(width: 256, height: 1, depth: 1)
        let packedIdxHeads: UInt32 = (poolShift == 2 && windowBlocks == 0)
            ? UInt32(idxHeads)
            : (UInt32(idxHeads & 0xFF) | 0x1000 | (UInt32(poolShift & 0xF) << 8) | (UInt32(windowBlocks) << 16))
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
                        candCap: cScratch.candCap, qChannels: qChannels, idxHeads: idxHeads,
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
                    qChannels: UInt32(qChannels),
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
                if let split = psoIndexerSplit {
                    let midPools = maxPools - windowBlocks
                    enc.setComputePipelineState(split.score)
                    enc.dispatchThreadgroups(MTLSize(width: (midPools + 63) / 64, height: batchRows, depth: 1),
                                             threadsPerThreadgroup: tg256)
                    enc.setComputePipelineState(split.select)
                } else {
                    enc.setComputePipelineState(psoIndexerScoreSelect)
                }
                enc.dispatchThreadgroups(MTLSize(width: batchRows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
            } else if !isDecoupledDSA {
                var idxP = IndexerParams(
                    T: UInt32(batchRows),
                    pos0: UInt32(pos0 + rowStart),
                    qChannels: UInt32(qChannels),
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
                enc.setComputePipelineState(psoIndexerSplit?.select ?? psoIndexerScoreSelect)
                enc.dispatchThreadgroups(MTLSize(width: batchRows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
            }
            rowStart += batchRows
        }
    }

    @discardableResult
    internal func encodeAbsorbQ(_ enc: any MTLComputeCommandEncoder,
                                kvBProj: QWWeightView,
                                absP: inout AbsorbParams,
                                rows: Int,
                                mTiles: Int) -> Bool {
        let inc = absP.isIncumbent & 0xFF
        let isSub2SHVQ = inc == 2 && (absP.bits == 1 || absP.bits == 15)
        let weightSupported = absP.group >= 32 && (absP.group % 32 == 0) && (kvLoraRank % Int(absP.group) == 0) && (
            inc == 1
            || (inc == 2 && (isSub2SHVQ || (absP.bits >= 2 && absP.bits <= 4)))
            || (inc == 0 && (absP.bits == 2 || absP.bits == 3 || absP.bits == 4 || absP.bits == 6 || absP.bits == 8))
        )
        let absorbMMA = sparseAttnMMAEnabled && rows >= 16 && weightSupported
            && nopeDim >= 32 && (nopeDim % 32 == 0) && kvLoraRank == 512
        enc.setComputePipelineState(absorbMMA ? (isSub2SHVQ ? psoAbsorbQMMASub2 : psoAbsorbQMMA)
                                              : (isSub2SHVQ ? psoAbsorbQSub2 : psoAbsorbQ))
        mlaQFullScratch.bind(enc, index: 0)
        kvBProj.blocks.bind(enc, index: 1)
        qAbsScratch.bind(enc, index: 2)
        enc.setBytes(&absP, length: MemoryLayout<AbsorbParams>.stride, index: 3)
        if absorbMMA {
            enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: kvLoraRank / 128),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: qHeads, height: mTiles, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        }
        return absorbMMA
    }

    internal func encodeProjUV(_ enc: any MTLComputeCommandEncoder,
                               kvBProj: QWWeightView,
                               dst: QWTensorValue,
                               absP: inout AbsorbParams,
                               absorbMMA: Bool,
                               rows: Int,
                               mTiles: Int) {
        let inc = absP.isIncumbent & 0xFF
        let isSub2SHVQ = inc == 2 && (absP.bits == 1 || absP.bits == 15)
        let projUVMMA = absorbMMA && vHeadDim == 256
        enc.setComputePipelineState(projUVMMA ? (isSub2SHVQ ? psoProjUVMMASub2 : psoProjUVMMA)
                                              : (isSub2SHVQ ? psoProjUVSub2 : psoProjUV))
        uLatScratch.bind(enc, index: 0)
        kvBProj.blocks.bind(enc, index: 1)
        dst.bind(enc, index: 2)
        enc.setBytes(&absP, length: MemoryLayout<AbsorbParams>.stride, index: 3)
        if projUVMMA {
            enc.dispatchThreadgroups(MTLSize(width: qHeads, height: (rows + 31) / 32, depth: vHeadDim / 128),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: qHeads, height: mTiles, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    internal func encodeDSAScoreSelectAbsorbAttendAndProjUV(_ enc: any MTLComputeCommandEncoder,
                                                            mlaW: LayerWeights,
                                                            kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                            attOut: QWTensorValue,
                                                            fullAttnLayer: Int,
                                                            pos0: Int,
                                                            rows: Int,
                                                            stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0,
              let kPe = layers[fullAttnLayer].kPeCache else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[fullAttnLayer]

        // 0. In-place GPT-J RoPE on q_rope + NeoX RoPE and H_128 on q_idx inside mlaQFull.
        mark("mla.q_rope_idx")
        var qrP = QRopeIdxPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            idxHeads: UInt32(idxHeads),
            ropeDim: UInt32(ropeDim),
            idxDim: UInt32(idxDim),
            qRopeOffset: UInt32(qRopeOffset),
            qChannels: UInt32(qChannels),
            qFullStride: UInt32(qFullStride),
            mscale: ropeMscale,
            useHadamard: useFastHadamard ? 1 : 0
        )
        enc.setComputePipelineState(psoQRopeIdxPrep)
        mlaQFullScratch.bind(enc, index: 0)
        invFreqBuf.bind(enc, index: 1)
        enc.setBytes(&qrP, length: MemoryLayout<QRopeIdxPrepParams>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: qHeads + idxHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // 1. Exact dense bypass at seqLen <= topK + windowTokens; split MMA or e-means above.
        mark("mla.score_select")
        if pos0 + rows > topKPools + windowTokens {
            encodeIndexerSelection(enc, ls: ls, pos0: pos0, rows: rows,
                                   probeClusters: Self.dsaEmeansProbeClusters,
                                   windowBlocks: windowTokens)
        }

        // 2. Absorb W_{UK, h} into q_nope -> qAbs [rows, qHeads, kvLoraRank]
        let baseInc: UInt32 = mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0)
        let needsPostEpilogue = (mlaW.vhaMix != nil) || mlaW.hasOGate || geometry.attnOutputGate
        var absP = AbsorbParams(
            T: UInt32(rows),
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank),
            qFullStride: UInt32(qFullStride),
            bits: UInt32(mlaW.kvBProj.bits),
            group: UInt32(mlaW.kvBProj.groupSize),
            isIncumbent: baseInc | (needsPostEpilogue ? 0x10000 : 0)
        )
        let mTiles = (rows + 7) / 8
        mark("mla.absorb_q")
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: rows, mTiles: mTiles)

        // 3. Decoupled-RoPE DSA attention (c_KV + k_pe + attn_sink) -> uLat
        let sinkTensor = mlaW.attnSink ?? zeroSinkBuf
        var attnP = DSASparseAttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(nopeDim + ropeDim),
            kvLoraRank: UInt32(kvLoraRank),
            topK: UInt32(topKPools),
            windowTokens: UInt32(windowTokens),
            listStride: UInt32(listStride),
            kvMode: Self.kvModeCode(kv.k.mode),
            ropeDim: UInt32(ropeDim),
            qRopeOffset: UInt32(qRopeOffset),
            qFullStride: UInt32(qFullStride),
            hasSink: mlaW.attnSink != nil ? 1 : 0
        )
        mark("mla.sparse_attn")
        let useMMA = sparseAttnMMAEnabled && attnP.kvMode <= 2 && rows >= 16
            && kvLoraRank == 512 && qHeads % 16 == 0
        let split = (sparseAttnMMAEnabled && !useMMA && attnP.kvMode <= 2 && rows <= Self.splitMaxRows
            && qHeads % 16 == 0) ? psoDSASparseAttnSplit : nil
        enc.setComputePipelineState(split?.attend ?? (useMMA ? (psoDSASparseAttnNAX ?? psoDSASparseAttnMMA) : psoDSASparseAttn))
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
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            mlaQFullScratch.bind(enc, index: 8)
            kPe.bind(enc, index: 9)
            sinkTensor.bind(enc, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // 4. Value up-projection -> yVhaScratch (if VHA/o_gate active) or directly attOut
        mark("mla.proj_uv")
        let projDst = needsPostEpilogue ? yVhaScratch : attOut
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: projDst, absP: &absP,
                     absorbMMA: absorbMMA, rows: rows, mTiles: mTiles)

        if needsPostEpilogue {
            mark("mla.vha_gate")
            var vgP = VHAGateParams(
                T: UInt32(rows),
                qHeads: UInt32(qHeads),
                vHeadDim: UInt32(vHeadDim),
                qFullStride: UInt32(qFullStride),
                oGateOffset: UInt32(oGateOffset),
                hasVHA: mlaW.vhaMix != nil ? 1 : 0,
                hasOGate: (mlaW.hasOGate || geometry.attnOutputGate) ? 1 : 0
            )
            enc.setComputePipelineState(psoVHAGateRot)
            yVhaScratch.bind(enc, index: 0)
            mlaQFullScratch.bind(enc, index: 1)
            (mlaW.vhaMix ?? yVhaScratch).bind(enc, index: 2)
            attOut.bind(enc, index: 3)
            enc.setBytes(&vgP, length: MemoryLayout<VHAGateParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: vHeadDim / 32, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
