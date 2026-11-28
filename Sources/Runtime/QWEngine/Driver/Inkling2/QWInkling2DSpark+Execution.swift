// QWInkling2DSpark+Execution.swift — Pipelined 1-CB commit+ingest+draft execution across BF16/QWP1 containers.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWHead

extension QWInkling2DSpark {
    func beginStage(
        _ stage: Ink2Stage?,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws -> any MTLComputeCommandEncoder {
        if let stage, let profiler {
            profiler.beginStage(stage, on: cb, encoder: &encOpt, hal: hal)
        }
        if let e = encOpt { return e }
        guard let plain = cb.makeComputeCommandEncoder() else {
            throw QWInkling2Error.badState("failed to create DSpark compute encoder")
        }
        hal.touchKeepAlive(plain)
        encOpt = plain
        return plain
    }

    func encodeFPGemm(
        _ enc: any MTLComputeCommandEncoder,
        bank: QWGemmFP.BF16Bank,
        xBuf: any MTLBuffer, xOff: Int = 0,
        yBF16: any MTLBuffer, yOff: Int = 0,
        m: Int,
        minNaxRows: Int = 17
    ) throws {
        let xSlice = QWFPBufferSlice(buffer: xBuf, offset: xOff, bytes: max(0, xBuf.length - xOff))
        let ySlice = QWFPBufferSlice(buffer: yBF16, offset: yOff, bytes: max(0, yBF16.length - yOff))
        try QWGemmFP.encodeBF16(
            enc, hal: hal, weight: bank, x: xSlice, y: ySlice,
            m: m, inDtype: .bf16, outDtype: .bf16, minNaxRows: minNaxRows
        )
    }

    func encodeRotateToF16(
        _ enc: any MTLComputeCommandEncoder,
        srcBF16: any MTLBuffer, srcOff: Int = 0,
        dst: any MTLBuffer, dstOff: Int = 0,
        count: Int, rotate: Bool
    ) {
        guard count > 0 else { return }
        var cnt = UInt32(count)
        var flags: UInt32 = rotate ? 1 : 0
        enc.setComputePipelineState(pBF16ToF16FWHT32)
        enc.setBuffer(srcBF16, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cnt, length: 4, index: 2)
        enc.setBytes(&flags, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (count + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeDSparkLinear(
        _ enc: any MTLComputeCommandEncoder,
        weight: Ink2LinearW,
        xBuf: any MTLBuffer, xOff: Int = 0,
        yBF16: any MTLBuffer, yOff: Int = 0,
        m: Int
    ) throws {
        guard m > 0 else { return }
        switch weight {
        case .bf16(let bank):
            try encodeFPGemm(enc, bank: bank, xBuf: xBuf, xOff: xOff, yBF16: yBF16, yOff: yOff, m: m)
        case .quant(let view):
            encodeRotateToF16(enc, srcBF16: xBuf, srcOff: xOff, dst: rotF16Buf,
                              count: m * view.k, rotate: view.isRotated)
            let x16TV = rotF16TV.slice(byteOffset: 0, count: m * view.k, dtype: .f16)
            let yBF16TV = QWTensorValue(region: QWMRegion(buffer: yBF16, base: 0, length: yBF16.length),
                                        offset: yOff, shape: QWTensorShape([m * view.n], .bf16))
            if m >= 5 && hal.caps.supportsSimdgroupMMA
                && ((view.n % 64) == 0 || !view.isIncumbentContainer) && (view.k % 64) == 0 {
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16TV, y16: x16TV, y32: yBF16TV, rows: m)
                return
            }
            if view.isSHVQ && m <= 16 {
                try QWGemmAQ.encodeDecodeBF16(
                    enc, hal: hal, blocks: view.blocks, x: x16TV, yBF16: yBF16TV,
                    n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                    rows: m, isSHVQ: true
                )
                return
            }
            if !view.isSHVQ && view.bits == 4 && view.groupSize == 32 && m <= 16 {
                try QWGemm.encodeHQ4DecodeBF16(
                    enc, hal: hal, weight: view, x16: x16TV, yBF16: yBF16TV, rows: m
                )
                return
            }
            func dispatchF32ToBF16(count: Int, dstOff: Int) {
                var outCnt = UInt32(count)
                var zeroCap: Float = 0.0
                enc.setComputePipelineState(pF32ToBF16)
                enc.setBuffer(f32OutBuf, offset: 0, index: 0)
                enc.setBuffer(yBF16, offset: dstOff, index: 1)
                enc.setBytes(&outCnt, length: 4, index: 2)
                enc.setBytes(&zeroCap, length: 4, index: 3)
                let threads = (count + 3) / 4
                enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, threads), height: 1, depth: 1))
            }
            if !view.isIncumbentContainer {
                let maxRowsPerChunk = max(1, f32OutElems / max(1, view.n))
                var r0 = 0
                while r0 < m {
                    let chunk = min(maxRowsPerChunk, m - r0)
                    let xSub = x16TV.slice(byteOffset: r0 * view.k * 2, count: chunk * view.k, dtype: .f16)
                    let ySub = f32OutTV.slice(byteOffset: 0, count: chunk * view.n, dtype: .f32)
                    try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xSub, y: ySub,
                                        n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                        rows: chunk, isSHVQ: view.isSHVQ)
                    dispatchF32ToBF16(count: chunk * view.n, dstOff: yOff + r0 * view.n * 2)
                    r0 += chunk
                }
                return
            }
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: view.n, k: view.k), dtype: .q4g, epi: .none, outDtype: .f32)
            for rr in 0..<m {
                let xSub = x16TV.slice(byteOffset: rr * view.k * 2, count: view.k, dtype: .f16)
                let ySub = f32OutTV.slice(byteOffset: 0, count: view.n, dtype: .f32)
                try QWGemm.encode(enc, hal: hal, plan: plan, x: xSub, wQ4: view, y: ySub)
                dispatchF32ToBF16(count: view.n, dstOff: yOff + rr * view.n * 2)
            }
        }
    }

    func encodeCopyBF16(
        _ enc: any MTLComputeCommandEncoder,
        src: any MTLBuffer, srcOff: Int,
        dst: any MTLBuffer, dstOff: Int,
        count: Int
    ) {
        guard count > 0 else { return }
        var cntU = UInt32(count)
        enc.setComputePipelineState(pCopyBF16)
        enc.setBuffer(src, offset: srcOff, index: 0)
        enc.setBuffer(dst, offset: dstOff, index: 1)
        enc.setBytes(&cntU, length: 4, index: 2)
        let total = (count + 7) / 8
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, total), height: 1, depth: 1))
    }

    func encodeSConvVerifyCommit(
        _ enc: any MTLComputeCommandEncoder,
        session: QWInkling2Session,
        kept: Int
    ) {
        guard kept > 0, let vInBuf = session.convVerifyInputsBuf else { return }
        let d = session.descriptor
        let H = d.hidden
        let kvDim = d.kvHeads * d.headDim
        let numVisits = session.executionPlan.count
        var p = SConvVerifyCommitP(
            numVisits: UInt32(numVisits), kvDim: UInt32(kvDim), H: UInt32(H), kept: UInt32(kept),
            stateKvStride: UInt32(session.convStateKvStrideElems),
            stateHStride: UInt32(session.convStateHStrideElems),
            verifyKvStride: UInt32(session.convVerifyKvStrideElems),
            verifyHStride: UInt32(session.convVerifyHStrideElems)
        )
        let totalCh = 2 * kvDim + 2 * H
        enc.setComputePipelineState(pSConvVerifyCommit)
        enc.setBuffer(session.convStatesSlab, offset: 0, index: 0)
        enc.setBuffer(vInBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SConvVerifyCommitP>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: totalCh, height: numVisits, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, totalCh), height: 1, depth: 1))
    }

    func encodeIngestContextTaps(
        count: Int,
        posOffset: Int,
        rowStride: Int,
        rowMap: [Int]? = nil,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws {
        guard count > 0 else { return }
        let H = config.hiddenSize
        let kvDim = config.kvHeads * config.headDim
        let tapPlaneStride = rowStride * H
        let enc = try beginStage(.dsparkIngest, on: cb, encoder: &encOpt, profiler: profiler)
        let mapped = rowMap ?? []
        let useMap = !mapped.isEmpty && mapped.count == count && count <= 16
        func rm(_ i: Int) -> UInt32 { i < mapped.count ? UInt32(mapped[i]) : UInt32(i) }

        for (lIdx, l) in layers.enumerated() {
            let a = alphas[lIdx]
            if useMap {
                var mp = MixMapP(
                    T: UInt32(count), H: UInt32(H), tapPlaneStride: UInt32(tapPlaneStride), useRowMap: 1,
                    eps: config.rmsEps, a0: a.0, a1: a.1, a2: a.2, a3: a.3,
                    rowMap: (rm(0), rm(1), rm(2), rm(3), rm(4), rm(5), rm(6), rm(7),
                             rm(8), rm(9), rm(10), rm(11), rm(12), rm(13), rm(14), rm(15))
                )
                enc.setComputePipelineState(pMixNormMap)
                enc.setBuffer(tapBuf, offset: 0, index: 0)
                enc.setBuffer(l.dflareNorm.buffer, offset: l.dflareNorm.offset, index: 1)
                enc.setBuffer(ctxMixBuf, offset: 0, index: 2)
                enc.setBytes(&mp, length: MemoryLayout<MixMapP>.stride, index: 3)
            } else {
                var mp = MixP(T: UInt32(count), H: UInt32(H), tapPlaneStride: UInt32(tapPlaneStride),
                              eps: config.rmsEps, a0: a.0, a1: a.1, a2: a.2, a3: a.3)
                enc.setComputePipelineState(pMixNorm)
                enc.setBuffer(tapBuf, offset: 0, index: 0)
                enc.setBuffer(l.dflareNorm.buffer, offset: l.dflareNorm.offset, index: 1)
                enc.setBuffer(ctxMixBuf, offset: 0, index: 2)
                enc.setBytes(&mp, length: MemoryLayout<MixP>.stride, index: 3)
            }
            enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var cp = CtxKVP(T: UInt32(count), nKV: UInt32(config.kvHeads), headDim: UInt32(config.headDim),
                            isSliding: l.usesRingBuffer ? 1 : 0, posOff: UInt32(posOffset), ringCap: UInt32(l.ringCap),
                            eps: config.rmsEps, ropeTheta: Float(config.ropeTheta))
            switch l.kvCtxW {
            case .fused(let w):
                try encodeDSparkLinear(enc, weight: w, xBuf: ctxMixBuf, yBF16: ctxKVProjBuf, m: count)
                enc.setComputePipelineState(pCtxKVFused)
                enc.setBuffer(ctxKVProjBuf, offset: 0, index: 0)
                enc.setBuffer(l.kNormCtx.buffer, offset: l.kNormCtx.offset, index: 1)
                enc.setBuffer(l.kCache, offset: 0, index: 2)
                enc.setBuffer(l.vCache, offset: 0, index: 3)
                enc.setBytes(&cp, length: MemoryLayout<CtxKVP>.stride, index: 4)
            case .split(let kw, let vw):
                let vByteOff = count * kvDim * 2
                try encodeDSparkLinear(enc, weight: kw, xBuf: ctxMixBuf, yBF16: ctxKVProjBuf, yOff: 0, m: count)
                try encodeDSparkLinear(enc, weight: vw, xBuf: ctxMixBuf, yBF16: ctxKVProjBuf, yOff: vByteOff, m: count)
                enc.setComputePipelineState(pCtxKV)
                enc.setBuffer(ctxKVProjBuf, offset: 0, index: 0)
                enc.setBuffer(ctxKVProjBuf, offset: vByteOff, index: 1)
                enc.setBuffer(l.kNormCtx.buffer, offset: l.kNormCtx.offset, index: 2)
                enc.setBuffer(l.kCache, offset: 0, index: 3)
                enc.setBuffer(l.vCache, offset: 0, index: 4)
                enc.setBytes(&cp, length: MemoryLayout<CtxKVP>.stride, index: 5)
            }
            enc.dispatchThreadgroups(MTLSize(width: config.kvHeads, height: count, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        let lastRow = useMap ? mapped[count - 1] : (count - 1)
        for p in 0..<4 {
            let srcOff = (p * tapPlaneStride + lastRow * H) * 2
            encodeCopyBF16(enc, src: tapBuf, srcOff: srcOff,
                           dst: lastTapBuf, dstOff: p * H * 2, count: H)
        }
        ctxLen = posOffset + count
    }

    func encodeDraftTransformerAndBaseLogits(
        anchor: Int,
        draftCount: Int,
        tokOutBuf: any MTLBuffer,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws {
        let B = config.blockSize
        let H = config.hiddenSize
        let nH = config.qHeads
        let nKV = config.kvHeads
        let hd = config.headDim
        let qDim = nH * hd
        let kvDim = nKV * hd
        let mlpI = config.intermediateSize
        let R = config.markovRank
        let V = ink2.unpaddedVocabSize

        tokOutBuf.contents().assumingMemoryBound(to: Int32.self)[0] = Int32(anchor)
        let enc = try beginStage(.dsparkDraft, on: cb, encoder: &encOpt, profiler: profiler)
        for i in 1..<B {
            encodeCopyBF16(enc, src: maskEmbed.buffer, srcOff: maskEmbed.offset,
                           dst: propXBuf, dstOff: i * H * 2, count: H)
        }

        var ep = QWInkling2Forward.EmbedP(T: 1, H: UInt32(H), V: UInt32(fullVocab), eps: embedRmsEps)
        enc.setComputePipelineState(pEmbed)
        enc.setBuffer(tokOutBuf, offset: 0, index: 0)
        enc.setBuffer(embedTokens.buffer, offset: embedTokens.offset, index: 1)
        enc.setBuffer(embedNorm.buffer, offset: embedNorm.offset, index: 2)
        enc.setBuffer(propXBuf, offset: 0, index: 3)
        enc.setBytes(&ep, length: MemoryLayout<QWInkling2Forward.EmbedP>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let gqaDepth = ((max(1, nH / nKV)) + 1) / 2
        let qkvStride = qDim + 2 * kvDim
        for (lIdx, l) in layers.enumerated() {
            let isFinalLayer = (lIdx == layers.count - 1)
            let qRowOff = isFinalLayer ? 1 : 0
            let activeRows = isFinalLayer ? draftCount : B
            let rowXOff = isFinalLayer ? (H * 2) : 0
            let rowAttnOff = isFinalLayer ? (qDim * 2) : 0

            var anp1 = AddNormP(rows: UInt32(B), dim: UInt32(H), hasRes: lIdx == 0 ? 0 : 1, writeRes: 1, eps: config.rmsEps)
            enc.setComputePipelineState(pAddNorm)
            enc.setBuffer(propXBuf, offset: 0, index: 0)
            enc.setBuffer(propResBuf, offset: 0, index: 1)
            enc.setBuffer(l.inputNorm.buffer, offset: l.inputNorm.offset, index: 2)
            enc.setBuffer(propNormBuf, offset: 0, index: 3)
            enc.setBytes(&anp1, length: MemoryLayout<AddNormP>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            let ctxStart = (l.usesRingBuffer && ctxLen > l.effectiveWindow) ? (ctxLen - l.effectiveWindow) : 0
            let ctxKeys = max(0, ctxLen - ctxStart)
            let kOff: Int
            let vOff: Int
            let qStrideVal: UInt32
            let kvStrideVal: UInt32
            switch l.qkvW {
            case .fused(let w):
                try encodeDSparkLinear(enc, weight: w, xBuf: propNormBuf, yBF16: propQKVBuf, m: B)
                var qkr = PropQKRStridedP(
                    B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd),
                    posOff: UInt32(ctxLen), rowStride: UInt32(qkvStride),
                    eps: config.rmsEps, ropeTheta: Float(config.ropeTheta)
                )
                enc.setComputePipelineState(pPropQKRStrided)
                enc.setBuffer(propQKVBuf, offset: 0, index: 0)
                enc.setBuffer(l.qNorm.buffer, offset: l.qNorm.offset, index: 1)
                enc.setBuffer(l.kNorm.buffer, offset: l.kNorm.offset, index: 2)
                enc.setBytes(&qkr, length: MemoryLayout<PropQKRStridedP>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: nH + nKV, height: B, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                kOff = qDim * 2
                vOff = (qDim + kvDim) * 2
                qStrideVal = UInt32(qkvStride)
                kvStrideVal = UInt32(qkvStride)
            case .split(let qw, let kw, let vw):
                kOff = B * qDim * 2
                vOff = B * (qDim + kvDim) * 2
                try encodeDSparkLinear(enc, weight: qw, xBuf: propNormBuf, yBF16: propQKVBuf, yOff: 0, m: B)
                try encodeDSparkLinear(enc, weight: kw, xBuf: propNormBuf, yBF16: propQKVBuf, yOff: kOff, m: B)
                try encodeDSparkLinear(enc, weight: vw, xBuf: propNormBuf, yBF16: propQKVBuf, yOff: vOff, m: B)
                var qkr = PropQKRP(B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd),
                                   posOff: UInt32(ctxLen), eps: config.rmsEps, ropeTheta: Float(config.ropeTheta))
                enc.setComputePipelineState(pPropQKR)
                enc.setBuffer(propQKVBuf, offset: 0, index: 0)
                enc.setBuffer(l.qNorm.buffer, offset: l.qNorm.offset, index: 1)
                enc.setBuffer(propQKVBuf, offset: kOff, index: 2)
                enc.setBuffer(l.kNorm.buffer, offset: l.kNorm.offset, index: 3)
                enc.setBytes(&qkr, length: MemoryLayout<PropQKRP>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: nH + nKV, height: B, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                qStrideVal = 0
                kvStrideVal = 0
            }

            if ctxKeys > 256 || isFinalLayer {
                let splits = (ctxKeys > 256) ? min(Self.maxSplitKV, (ctxKeys + 255) / 256) : 1
                var sp = AttnSplitP(
                    B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd),
                    ctxStart: UInt32(ctxStart), ctxEnd: UInt32(ctxLen),
                    sliding: l.usesRingBuffer ? UInt32(l.effectiveWindow) : 0,
                    ringCap: UInt32(l.ringCap), qStride: qStrideVal, kvPropStride: kvStrideVal,
                    qRowOffset: UInt32(qRowOff), activeQRows: UInt32(activeRows),
                    splits: UInt32(splits), scale: 1.0 / Float(hd).squareRoot()
                )
                enc.setComputePipelineState(pAttnGQASplit)
                enc.setBuffer(propQKVBuf, offset: 0, index: 0)
                enc.setBuffer(propQKVBuf, offset: kOff, index: 1)
                enc.setBuffer(propQKVBuf, offset: vOff, index: 2)
                enc.setBuffer(l.kCache, offset: 0, index: 3)
                enc.setBuffer(l.vCache, offset: 0, index: 4)
                enc.setBuffer(propAttnPartOBuf, offset: 0, index: 5)
                enc.setBuffer(propAttnPartMLBuf, offset: 0, index: 6)
                enc.setBuffer(propAttnBuf, offset: 0, index: 7)
                enc.setBytes(&sp, length: MemoryLayout<AttnSplitP>.stride, index: 8)
                enc.dispatchThreadgroups(
                    MTLSize(width: nKV, height: ((activeRows + 3) / 4) * splits, depth: gqaDepth),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                if splits > 1 {
                    var cp = AttnCombineP(
                        nH: UInt32(nH), headDim: UInt32(hd), qRowOffset: UInt32(qRowOff),
                        activeQRows: UInt32(activeRows), splits: UInt32(splits)
                    )
                    enc.setComputePipelineState(pAttnGQACombine)
                    enc.setBuffer(propAttnPartOBuf, offset: 0, index: 0)
                    enc.setBuffer(propAttnPartMLBuf, offset: 0, index: 1)
                    enc.setBuffer(propAttnBuf, offset: 0, index: 2)
                    enc.setBytes(&cp, length: MemoryLayout<AttnCombineP>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: nH, height: activeRows, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                }
            } else {
                var ap = AttnBatchP(
                    B: UInt32(B), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(hd),
                    ctxStart: UInt32(ctxStart), ctxEnd: UInt32(ctxLen),
                    sliding: l.usesRingBuffer ? UInt32(l.effectiveWindow) : 0,
                    ringCap: UInt32(l.ringCap), qStride: qStrideVal, kvPropStride: kvStrideVal,
                    scale: 1.0 / Float(hd).squareRoot()
                )
                enc.setComputePipelineState(pAttnGQABatch)
                enc.setBuffer(propQKVBuf, offset: 0, index: 0)
                enc.setBuffer(propQKVBuf, offset: kOff, index: 1)
                enc.setBuffer(propQKVBuf, offset: vOff, index: 2)
                enc.setBuffer(l.kCache, offset: 0, index: 3)
                enc.setBuffer(l.vCache, offset: 0, index: 4)
                enc.setBuffer(propAttnBuf, offset: 0, index: 5)
                enc.setBytes(&ap, length: MemoryLayout<AttnBatchP>.stride, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: nKV, height: (B + 3) / 4, depth: gqaDepth),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            try encodeDSparkLinear(
                enc, weight: l.oW, xBuf: propAttnBuf, xOff: rowAttnOff,
                yBF16: propXBuf, yOff: rowXOff, m: activeRows
            )

            var anp2 = AddNormP(rows: UInt32(activeRows), dim: UInt32(H), hasRes: 1, writeRes: 1, eps: config.rmsEps)
            enc.setComputePipelineState(pAddNorm)
            enc.setBuffer(propXBuf, offset: rowXOff, index: 0)
            enc.setBuffer(propResBuf, offset: rowXOff, index: 1)
            enc.setBuffer(l.postAttnNorm.buffer, offset: l.postAttnNorm.offset, index: 2)
            enc.setBuffer(propNormBuf, offset: rowXOff, index: 3)
            enc.setBytes(&anp2, length: MemoryLayout<AddNormP>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: activeRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            try encodeDSparkLinear(
                enc, weight: l.gateUpW, xBuf: propNormBuf, xOff: rowXOff,
                yBF16: propGUBuf, yOff: 0, m: activeRows
            )
            try QWGemmFP.encodeSwiGLUSoftcap(
                enc, hal: hal, gu: propGUSlice, act: propActSlice,
                rows: activeRows, intermediate: mlpI, gateCap: 0, upCap: 0, inDtype: .bf16, outDtype: .bf16
            )
            try encodeDSparkLinear(
                enc, weight: l.downW, xBuf: propActBuf, xOff: 0,
                yBF16: propXBuf, yOff: rowXOff, m: activeRows
            )
        }

        let finalXOff = H * 2
        var anpFinal = AddNormP(rows: UInt32(draftCount), dim: UInt32(H), hasRes: 1, writeRes: 0, eps: config.rmsEps)
        enc.setComputePipelineState(pAddNorm)
        enc.setBuffer(propXBuf, offset: finalXOff, index: 0)
        enc.setBuffer(propResBuf, offset: finalXOff, index: 1)
        enc.setBuffer(finalNorm.buffer, offset: finalNorm.offset, index: 2)
        enc.setBuffer(propNormBuf, offset: finalXOff, index: 3)
        enc.setBytes(&anpFinal, length: MemoryLayout<AddNormP>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: draftCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        switch lmHeadStorage {
        case .bf16:
            guard let lmHeadBank else {
                throw QWInkling2Error.badState("DSpark missing BF16 lmHeadBank")
            }
            try encodeFPGemm(enc, bank: lmHeadBank, xBuf: propNormBuf, xOff: H * 2,
                             yBF16: baseLogitsBF16Buf, yOff: 0, m: draftCount, minNaxRows: 17)
        case .q4g(let hw):
            encodeRotateToF16(enc, srcBF16: propNormBuf, srcOff: H * 2,
                              dst: baseLogitsF16InBuf, count: draftCount * H, rotate: hw.view.isRotated)
            let x16 = baseLogitsF16InTV.slice(byteOffset: 0, count: draftCount * H, dtype: .f16)
            let useNAX = hal.caps.isMetal4TensorCapable
            let canUseNax = draftCount >= 3 && hal.caps.supportsSimdgroupMMA
                && (H % 64) == 0 && hw.view.n == V
                && ((hw.view.n % 64) == 0 || !hw.view.isIncumbentContainer)
                && (hw.view.isIncumbentContainer
                    || QWGemmAQ.naxF32KernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ, useNAX: useNAX) != nil)
            if canUseNax {
                let yBF16 = QWTensorValue(region: QWMRegion(buffer: baseLogitsBF16Buf, base: 0, length: baseLogitsBF16Buf.length),
                                          offset: 0, shape: QWTensorShape([draftCount * V], .bf16))
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: hw.view, x16: x16, y16: x16, y32: yBF16, rows: draftCount)
            } else if hw.view.isSHVQ {
                let yBF16 = QWTensorValue(region: QWMRegion(buffer: baseLogitsBF16Buf, base: 0, length: baseLogitsBF16Buf.length),
                                          offset: 0, shape: QWTensorShape([draftCount * V], .bf16))
                try QWGemmAQ.encodeDecodeBF16(
                    enc, hal: hal, blocks: hw.view.blocks, x: x16, yBF16: yBF16,
                    n: V, k: H, bits: hw.view.bits, group: hw.view.groupSize, rows: draftCount, isSHVQ: true
                )
            } else if !hw.view.isSHVQ && hw.view.bits == 4 && hw.view.groupSize == 32 && draftCount <= 16 {
                let yBF16 = QWTensorValue(region: QWMRegion(buffer: baseLogitsBF16Buf, base: 0, length: baseLogitsBF16Buf.length),
                                          offset: 0, shape: QWTensorShape([draftCount * V], .bf16))
                try QWGemm.encodeHQ4DecodeBF16(
                    enc, hal: hal, weight: hw.view, x16: x16, yBF16: yBF16, rows: draftCount
                )
            } else {
                let y32 = baseLogitsF32TV.slice(byteOffset: 0, count: draftCount * V, dtype: .f32)
                if draftCount == 1 {
                    try QWHead.encodeQ4g(enc, hal: hal, weight: hw, x: x16, y: y32)
                } else {
                    try QWHead.encodeBatched(enc, hal: hal, view: hw.view, x: x16, y: y32, m: draftCount)
                }
                var outCnt = UInt32(draftCount * V)
                var zeroCap: Float = 0.0
                enc.setComputePipelineState(pF32ToBF16)
                enc.setBuffer(baseLogitsF32Buf, offset: 0, index: 0)
                enc.setBuffer(baseLogitsBF16Buf, offset: 0, index: 1)
                enc.setBytes(&outCnt, length: 4, index: 2)
                enc.setBytes(&zeroCap, length: 4, index: 3)
                let f32Threads = (draftCount * V + 3) / 4
                enc.dispatchThreads(MTLSize(width: f32Threads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, f32Threads), height: 1, depth: 1))
            }
        case .f16:
            throw QWInkling2Error.badState("unsupported .f16 lmHeadStorage for DSpark")
        }

        let mEnc = try beginStage(.dsparkMarkov, on: cb, encoder: &encOpt, profiler: profiler)
        let aLast = alphas[3]
        var mpLast = MixP(T: 1, H: UInt32(H), tapPlaneStride: UInt32(H),
                          eps: config.rmsEps, a0: aLast.0, a1: aLast.1, a2: aLast.2, a3: aLast.3)
        mEnc.setComputePipelineState(pMixNorm)
        mEnc.setBuffer(lastTapBuf, offset: 0, index: 0)
        mEnc.setBuffer(markovDFlareNorm.buffer, offset: markovDFlareNorm.offset, index: 1)
        mEnc.setBuffer(fRnnBuf, offset: 0, index: 2)
        mEnc.setBytes(&mpLast, length: MemoryLayout<MixP>.stride, index: 3)
        mEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        var mcp = MarkovCtxP(count: UInt32(draftCount), H: UInt32(H), projRows: UInt32(3 * R), rank: UInt32(R))
        mEnc.setComputePipelineState(pMarkovCtx)
        mEnc.setBuffer(propNormBuf, offset: H * 2, index: 0)
        mEnc.setBuffer(fRnnBuf, offset: 0, index: 1)
        mEnc.setBuffer(rnnProjW.buffer, offset: rnnProjW.offset, index: 2)
        mEnc.setBuffer(rnnProjB.buffer, offset: rnnProjB.offset, index: 3)
        mEnc.setBuffer(markovCtxF32Buf, offset: 0, index: 4)
        mEnc.setBytes(&mcp, length: MemoryLayout<MarkovCtxP>.stride, index: 5)
        mEnc.dispatchThreadgroups(MTLSize(width: (3 * R + 7) / 8, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeDraftBlock(
        anchor: Int,
        count: Int,
        on cb: any MTLCommandBuffer,
        encoder encOpt: inout (any MTLComputeCommandEncoder)?,
        profiler: QWInkling2Profiler?
    ) throws {
        guard count > 0, count < config.blockSize else { return }
        try encodeDraftTransformerAndBaseLogits(
            anchor: anchor, draftCount: count, tokOutBuf: draftTokBuf,
            on: cb, encoder: &encOpt, profiler: profiler
        )
        let H = config.hiddenSize
        let R = config.markovRank
        let V = ink2.unpaddedVocabSize
        let mEnc = try beginStage(.dsparkMarkov, on: cb, encoder: &encOpt, profiler: profiler)
        let partCount = min(1024, (V + 7) / 8)
        let logitsPSO = foldWIsAQ8 ? pLogitsAQ8Stage1 : pLogitsStage1
        for step in 0..<count {
            var msp = MarkovStepP(stepIdx: UInt32(step), H: UInt32(H), projRows: UInt32(3 * R),
                                  rank: UInt32(R), resetState: step == 0 ? 1 : 0)
            mEnc.setComputePipelineState(pMarkovStep)
            mEnc.setBuffer(draftTokBuf, offset: 0, index: 0)
            mEnc.setBuffer(markovW1.buffer, offset: markovW1.offset, index: 1)
            mEnc.setBuffer(rnnProjW.buffer, offset: rnnProjW.offset, index: 2)
            mEnc.setBuffer(markovCtxF32Buf, offset: 0, index: 3)
            mEnc.setBuffer(markovStateBF16Buf, offset: 0, index: 4)
            mEnc.setBuffer(markovLatentBF16Buf, offset: 0, index: 5)
            mEnc.setBytes(&msp, length: MemoryLayout<MarkovStepP>.stride, index: 6)
            mEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var lp = LogitsP(stepIdx: UInt32(step), V: UInt32(V), rank: UInt32(R),
                             partCount: UInt32(partCount), softcap: ink2.finalLogitSoftcapping)
            mEnc.setComputePipelineState(logitsPSO)
            mEnc.setBuffer(baseLogitsBF16Buf, offset: 0, index: 0)
            mEnc.setBuffer(markovLatentBF16Buf, offset: 0, index: 1)
            mEnc.setBuffer(foldWBuf, offset: 0, index: 2)
            mEnc.setBuffer(foldBBuf, offset: 0, index: 3)
            mEnc.setBuffer(argmaxPartValBuf, offset: 0, index: 4)
            mEnc.setBuffer(argmaxPartIdxBuf, offset: 0, index: 5)
            mEnc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 6)
            mEnc.dispatchThreadgroups(MTLSize(width: partCount, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            mEnc.setComputePipelineState(pArgmaxFinal)
            mEnc.setBuffer(argmaxPartValBuf, offset: 0, index: 0)
            mEnc.setBuffer(argmaxPartIdxBuf, offset: 0, index: 1)
            mEnc.setBuffer(draftTokBuf, offset: 0, index: 2)
            mEnc.setBytes(&lp, length: MemoryLayout<LogitsP>.stride, index: 3)
            mEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func makeDSparkCB(profiler: QWInkling2Profiler?) -> (any MTLCommandBuffer)? {
        (profiler == nil)
            ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
            : hal.queue.makeCommandBuffer()
    }

    @discardableResult
    func ingestContextTaps(
        count: Int, posOffset: Int, rowStride: Int,
        profiler: QWInkling2Profiler? = nil, wait: Bool = true
    ) throws -> (any MTLCommandBuffer)? {
        guard count > 0 else { return nil }
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        guard let cb = makeDSparkCB(profiler: profiler) else {
            throw QWInkling2Error.badState("failed to create DSpark ingest CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        try encodeIngestContextTaps(
            count: count, posOffset: posOffset, rowStride: rowStride,
            on: cb, encoder: &encOpt, profiler: profiler
        )
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        if !wait && profiler == nil {
            return cb
        }
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark ingest failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: 0)
        return nil
    }

    func draftBlock(anchor: Int, count: Int, profiler: QWInkling2Profiler? = nil) throws -> [Int] {
        guard count > 0, count < config.blockSize else { return [] }
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        guard let cb = makeDSparkCB(profiler: profiler) else {
            throw QWInkling2Error.badState("failed to create DSpark draft CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        try encodeDraftBlock(anchor: anchor, count: count, on: cb, encoder: &encOpt, profiler: profiler)
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark draft failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: 0)
        let ptr = draftTokBuf.contents().assumingMemoryBound(to: Int32.self)
        return (1...count).map { Int(ptr[$0]) }
    }

    func finishSpeculativeRound(
        session: QWInkling2Session,
        kept: Int,
        oldPos: Int,
        rowStride: Int,
        ingestTaps: Bool,
        nextAnchor: Int,
        nextDraftCount: Int,
        profiler: QWInkling2Profiler? = nil
    ) throws -> [Int] {
        let hostStartNs = profiler?.beginHostTimer() ?? 0
        guard let cb = makeDSparkCB(profiler: profiler) else {
            throw QWInkling2Error.badState("failed to create speculative round CB")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        let commitEnc = try beginStage(.sconv, on: cb, encoder: &encOpt, profiler: profiler)
        encodeSConvVerifyCommit(commitEnc, session: session, kept: kept)
        if ingestTaps {
            try encodeIngestContextTaps(
                count: kept, posOffset: oldPos, rowStride: rowStride,
                on: cb, encoder: &encOpt, profiler: profiler
            )
            if nextDraftCount > 0, nextDraftCount < config.blockSize {
                try encodeDraftBlock(
                    anchor: nextAnchor, count: nextDraftCount,
                    on: cb, encoder: &encOpt, profiler: profiler
                )
            }
        }
        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DSpark round CB failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: ingestTaps ? kept : 0)
        guard ingestTaps, nextDraftCount > 0, nextDraftCount < config.blockSize else { return [] }
        let ptr = draftTokBuf.contents().assumingMemoryBound(to: Int32.self)
        return (1...nextDraftCount).map { Int(ptr[$0]) }
    }

    func capturePrefixSlices(pos: Int) -> (slices: [KVSlice], lastTap: Data) {
        let rowBytes = config.kvHeads * config.headDim * 2
        var slices: [KVSlice] = []
        slices.reserveCapacity(layers.count)
        for l in layers {
            let startPos = l.usesRingBuffer ? max(0, pos - l.effectiveWindow) : 0
            let count = pos - startPos
            var kData = Data(); kData.reserveCapacity(count * rowBytes)
            var vData = Data(); vData.reserveCapacity(count * rowBytes)
            let kp = l.kCache.contents().assumingMemoryBound(to: UInt8.self)
            let vp = l.vCache.contents().assumingMemoryBound(to: UInt8.self)
            for absPos in startPos..<pos {
                let slot = l.usesRingBuffer ? (absPos % l.ringCap) : absPos
                kData.append(kp.advanced(by: slot * rowBytes), count: rowBytes)
                vData.append(vp.advanced(by: slot * rowBytes), count: rowBytes)
            }
            slices.append(KVSlice(isSliding: l.isSliding, startPos: startPos, count: count, kData: kData, vData: vData))
        }
        let tapBytes = Data(bytes: lastTapBuf.contents(), count: 4 * config.hiddenSize * 2)
        return (slices, tapBytes)
    }

    func restorePrefixSlices(_ slices: [KVSlice], lastTap: Data, pos: Int) {
        let rowBytes = config.kvHeads * config.headDim * 2
        for (idx, slice) in slices.enumerated() where idx < layers.count {
            let l = layers[idx]
            let kp = l.kCache.contents()
            let vp = l.vCache.contents()
            withExtendedLifetime(l.kCache) {
                slice.kData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.count {
                        let absPos = slice.startPos + r
                        let slot = l.usesRingBuffer ? (absPos % l.ringCap) : absPos
                        kp.advanced(by: slot * rowBytes).copyMemory(from: base.advanced(by: r * rowBytes), byteCount: rowBytes)
                    }
                }
            }
            withExtendedLifetime(l.vCache) {
                slice.vData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    guard let base = raw.baseAddress else { return }
                    for r in 0..<slice.count {
                        let absPos = slice.startPos + r
                        let slot = l.usesRingBuffer ? (absPos % l.ringCap) : absPos
                        vp.advanced(by: slot * rowBytes).copyMemory(from: base.advanced(by: r * rowBytes), byteCount: rowBytes)
                    }
                }
            }
        }
        let tapDst = lastTapBuf.contents()
        let tapBytes = min(lastTap.count, lastTapBuf.length)
        withExtendedLifetime(lastTapBuf) {
            lastTap.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                tapDst.copyMemory(from: base, byteCount: tapBytes)
            }
        }
        ctxLen = pos
    }
}
