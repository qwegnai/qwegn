// QWHCAAttnOperator+Snapshot.swift — HCA Weight-SVD LSA + e-means selection,
// grouped low-rank wo_a dispatch, and `.qwcache` state snapshot / restore.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

extension QWHCAAttnOperator {

    internal func updateCoarseClusters(_ enc: any MTLComputeCommandEncoder,
                                       ls: LayerState,
                                       hcaLayer: Int,
                                       pos0: Int,
                                       rows: Int) {
        guard emeansCoarseEnabled,
              let pipes = coarsePipelines,
              let csr = ls.coarseCSR,
              let pKeys = ls.pooledKeys else { return }
        let chunkTok = QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        let curMacro = pos0 / chunkTok
        if curMacro < clusteredMacros[hcaLayer] {
            clusteredMacros[hcaLayer] = curMacro
        }
        let targetMacros = min(csr.maxMacros, (pos0 + rows) / chunkTok)
        let prevMacros = clusteredMacros[hcaLayer]
        if targetMacros > prevMacros {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            QWCoarsePoolIndex.encodeClusterMacros(
                enc, pso: pipes.cluster1024, pooledKeys: pKeys,
                csr: csr, startMacro: prevMacros, numMacros: targetMacros - prevMacros, indexDim: indexDim)
            clusteredMacros[hcaLayer] = targetMacros
        }
    }

    internal func encodeLSASelection(_ enc: any MTLComputeCommandEncoder,
                                     ls: LayerState,
                                     wqIdx: QWTensorValue,
                                     qPrepParams: inout QPrepParams,
                                     pos0: Int,
                                     rows: Int) {
        guard let qIdx = qIdxScratch,
              let bList = blockListScratch,
              let bCount = blockCountScratch,
              let pKeys = ls.pooledKeys,
              let pipes = coarsePipelines,
              let lsaPso = psoEmeansLsa,
              let csr = ls.coarseCSR,
              let cScratch = coarseScratch else { return }

        let tg256 = MTLSize(width: 256, height: 1, depth: 1)
        enc.setComputePipelineState(psoLsaQPrep)
        qUpGateScratch.bind(enc, index: 0)
        wqIdx.bind(enc, index: 1)
        qIdx.bind(enc, index: 2)
        enc.setBytes(&qPrepParams, length: MemoryLayout<QPrepParams>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg256)
        QWCoarsePoolIndex.barrierIfConcurrent(enc)

        var rowStart = 0
        while rowStart < rows {
            let batchRows = min(selectBatchRows, rows - rowStart)
            let bg = QWCoarsePoolIndex.batchGeometry(
                pos0: pos0 + rowStart, units: batchRows, budgetBlocks: lsaBudgetBlocks,
                windowBlocks: 0, hasSink: true, writeCount: true,
                probeClusters: Self.emeansProbeClusters, poolShift: poolShift)
            if bg.isDense {
                rowStart += batchRows
                continue
            }
            let qSub = qIdx.slice(byteOffset: rowStart * indexDim * 2, count: batchRows * indexDim, dtype: .f16)
            let listSub = bList.slice(byteOffset: rowStart * listStride * 4, count: batchRows * listStride, dtype: .i32)
            let countSub = bCount.slice(byteOffset: rowStart * 4, count: batchRows, dtype: .i32)
            let usePrune = emeansCoarseEnabled && bg.shouldPrune
            var ePrm = QWCoarsePoolIndex.SelectParams(
                T: batchRows, pos0: pos0 + rowStart, indexDim: indexDim,
                budgetBlocks: lsaBudgetBlocks, windowBlocks: 0, hasSink: true,
                writeCount: true, listStride: listStride, rowStep: 1, maxRow: batchRows,
                clusterStride: cScratch.clusterStride,
                probeClusters: usePrune ? Self.emeansProbeClusters : 0,
                candCap: cScratch.candCap, poolShift: poolShift)
            let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride

            if usePrune {
                enc.setComputePipelineState(lsaPso.scoreCoarse)
                qSub.bind(enc, index: 0)
                csr.centroids.bind(enc, index: 1)
                csr.clusterSpans.bind(enc, index: 2)
                cScratch.cScore.bind(enc, index: 3)
                cScratch.uHist.bind(enc, index: 4)
                enc.setBytes(&ePrm, length: plen, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: (bg.numClusters + 255) / 256, height: batchRows, depth: 1),
                    threadsPerThreadgroup: tg256)
            }

            QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
                enc, pipelines: pipes, csr: csr, scratch: cScratch,
                blockList: listSub, blockCount: countSub, params: &ePrm,
                numClusters: usePrune ? bg.numClusters : 0,
                maxMiddlePools: bg.middleEnd,
                encodeScoreCands: { enc, maxCand, prm in
                    enc.setComputePipelineState(lsaPso.scoreCands)
                    qSub.bind(enc, index: 0)
                    pKeys.bind(enc, index: 1)
                    cScratch.candState.bind(enc, index: 2)
                    cScratch.candPools.bind(enc, index: 3)
                    cScratch.candU.bind(enc, index: 4)
                    enc.setBytes(&prm, length: plen, index: 5)
                    enc.dispatchThreadgroups(
                        MTLSize(width: (maxCand + 255) / 256, height: batchRows, depth: 1),
                        threadsPerThreadgroup: tg256)
                },
                encodeFlatFallback: { enc, prm in
                    enc.setComputePipelineState(lsaPso.flatFallback)
                    qSub.bind(enc, index: 0)
                    pKeys.bind(enc, index: 1)
                    cScratch.candState.bind(enc, index: 2)
                    listSub.bind(enc, index: 3)
                    countSub.bind(enc, index: 4)
                    enc.setBytes(&prm, length: plen, index: 5)
                    enc.dispatchThreadgroups(
                        MTLSize(width: batchRows, height: 1, depth: 1),
                        threadsPerThreadgroup: tg256)
                }
            )
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            rowStart += batchRows
        }
    }

    internal func encodeGroupedWoA(_ enc: any MTLComputeCommandEncoder,
                                   hcaW: LayerWeights,
                                   zGatedOut: QWTensorValue,
                                   rows: Int,
                                   skipExitRotation: Bool) {
        let baseInc: UInt32
        let bits: UInt32
        let group: UInt32
        if let w = hcaW.woA {
            baseInc = w.isIncumbentContainer ? 1 : (w.isSHVQ ? 2 : 0)
            bits = UInt32(w.bits)
            group = UInt32(w.groupSize)
        } else if hcaW.woARawF16 != nil {
            baseInc = 3
            bits = 16
            group = 32
        } else {
            return
        }
        var woP = GroupedWoAParams(
            T: UInt32(rows),
            oGroups: UInt32(oGroups),
            oLoraRank: UInt32(oLoraRank),
            groupIn: UInt32(groupIn),
            qUpStride: UInt32(qUpStride),
            oGateOffset: UInt32(oGateOffset),
            bits: bits,
            group: group,
            isIncumbent: baseInc | (skipExitRotation ? 0x10000 : 0)
        )
        let useMMA = rows >= 16 && (oLoraRank % 128 == 0) && (groupIn % 32 == 0)
        enc.setComputePipelineState(useMMA ? psoGroupedWoAMMA : psoGroupedWoAGemv)
        zVhaScratch.bind(enc, index: 0)
        if let w = hcaW.woA {
            w.blocks.bind(enc, index: 1)
        } else if let raw = hcaW.woARawF16 {
            raw.bind(enc, index: 1)
        }
        qUpGateScratch.bind(enc, index: 2)
        zGatedOut.bind(enc, index: 3)
        enc.setBytes(&woP, length: MemoryLayout<GroupedWoAParams>.stride, index: 4)
        if useMMA {
            enc.dispatchThreadgroups(
                MTLSize(width: oGroups, height: (rows + 31) / 32, depth: oLoraRank / 128),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(
                MTLSize(width: oLoraRank / 32, height: oGroups, depth: rows),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    public func stateBytes(position: Int) -> Int {
        guard position >= 0 else { return 0 }
        let numComp = position / compressRatio
        let swaCount = min(position, slidingWindow)
        let openCount = position - numComp * compressRatio
        let compBytes = numComp * headDim * 2
        let pooledBytes = lsaAllocated ? (numComp * indexDim * 2) : 0
        let swaBytes = swaCount * headDim * 2
        let openBytes = openCount * headDim * 8
        return layers.count * (compBytes + pooledBytes + swaBytes + openBytes)
    }

    public func captureState(position: Int) -> Data {
        precondition(position >= 0 && position <= maxContext)
        let numComp = position / compressRatio
        let swaCount = min(position, slidingWindow)
        let swaStart = position - swaCount
        let openStart = numComp * compressRatio
        let f16RowBytes = headDim * 2
        let f32RowBytes = headDim * 4
        let compBytes = numComp * f16RowBytes
        let pooledBytes = lsaAllocated ? (numComp * indexDim * 2) : 0
        var out = Data(capacity: stateBytes(position: position))
        let base = region.pointer
        for ls in layers {
            if compBytes > 0 {
                out.append(base.advanced(by: ls.compKV.offset).assumingMemoryBound(to: UInt8.self),
                           count: compBytes)
            }
            if pooledBytes > 0, let pk = ls.pooledKeys {
                out.append(base.advanced(by: pk.offset).assumingMemoryBound(to: UInt8.self),
                           count: pooledBytes)
            }
            for p in swaStart..<position {
                let slot = p & (ringCap - 1)
                out.append(base.advanced(by: ls.swaKV.offset + slot * f16RowBytes).assumingMemoryBound(to: UInt8.self),
                           count: f16RowBytes)
            }
            for ring in [ls.compKvPend, ls.compScorePend] {
                for p in openStart..<position {
                    let slot = p & (ringCap - 1)
                    out.append(base.advanced(by: ring.offset + slot * f32RowBytes).assumingMemoryBound(to: UInt8.self),
                               count: f32RowBytes)
                }
            }
        }
        return out
    }

    public func restoreState(_ data: Data, position: Int) -> Bool {
        guard position >= 0, position <= maxContext,
              data.count == stateBytes(position: position) else { return false }
        let numComp = position / compressRatio
        let swaCount = min(position, slidingWindow)
        let swaStart = position - swaCount
        let openStart = numComp * compressRatio
        let f16RowBytes = headDim * 2
        let f32RowBytes = headDim * 4
        let compBytes = numComp * f16RowBytes
        let pooledBytes = lsaAllocated ? (numComp * indexDim * 2) : 0
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let srcBase = raw.baseAddress else { return }
            var src = 0
            for ls in layers {
                if compBytes > 0 {
                    base.advanced(by: ls.compKV.offset)
                        .copyMemory(from: srcBase.advanced(by: src), byteCount: compBytes)
                    src += compBytes
                }
                if pooledBytes > 0, let pk = ls.pooledKeys {
                    base.advanced(by: pk.offset)
                        .copyMemory(from: srcBase.advanced(by: src), byteCount: pooledBytes)
                    src += pooledBytes
                }
                for p in swaStart..<position {
                    let slot = p & (ringCap - 1)
                    base.advanced(by: ls.swaKV.offset + slot * f16RowBytes)
                        .copyMemory(from: srcBase.advanced(by: src), byteCount: f16RowBytes)
                    src += f16RowBytes
                }
                for ring in [ls.compKvPend, ls.compScorePend] {
                    for p in openStart..<position {
                        let slot = p & (ringCap - 1)
                        base.advanced(by: ring.offset + slot * f32RowBytes)
                            .copyMemory(from: srcBase.advanced(by: src), byteCount: f32RowBytes)
                        src += f32RowBytes
                    }
                }
            }
        }
        resetCoarseWatermarks()
        return true
    }
}
