// QWDSparkDrafter+Encode.swift — Metal encoding helpers for DSpark draft block, Markov chain, and attention.
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWMTP

extension QWDSparkDrafter {
    func makeProposeEncoder(_ cb: any MTLCommandBuffer) throws -> any MTLComputeCommandEncoder {
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("DSpark propose compute encoder failed")
        }
        hal.touchKeepAlive(enc)
        return enc
    }

    func encodeBlockAndBaseLogits(
        _ encRef: inout any MTLComputeCommandEncoder,
        cb: any MTLCommandBuffer,
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int,
        anchorToken: Int,
        headRows: Int? = nil
    ) throws {
        try encodeBlockTrunk(&encRef, cb: cb, driver: driver, slot: slot, pos0: pos0)
        let rows = (config.sampleOffset == 0) ? min(config.blockSize, headRows ?? config.blockSize) : config.blockSize
        try encodeHeadNormAndProjection(encRef, driver: driver, rows: rows)
        encRef.memoryBarrier(scope: .buffers)
        _ = anchorToken
    }

    func encodeBlockTrunk(
        _ encRef: inout any MTLComputeCommandEncoder,
        cb: any MTLCommandBuffer,
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int
    ) throws {
        var enc = encRef
        defer { encRef = enc }

        let B = config.blockSize
        let H = weights.hidden
        let aGeo = weights.attnGeometry

        useResources(on: enc)
        if let embedTable = driver.mtpDeviceEmbedding {
            enc.useResource(embedTable.buffer, usage: .read)
            try embedTable.encodeRows(enc, tokensBuffer: promptToksBuf, tokensByteOffset: 0,
                                      rows: B, destination: blkX)
        }

        let hasMHC = targetGeometry.mhcConfig != nil && weights.layers.first?.mhc != nil
        if hasMHC, let mhcStream = blkMhcStream {
            try driver.encodeMHCEnter(enc, x: blkX, stream: mhcStream, rows: B)
            enc.memoryBarrier(scope: .buffers)
        }

        for i in 0..<config.numHiddenLayers {
            let lw = weights.layers[i]
            if let mhc = lw.mhc, let mhcStream = blkMhcStream, let mhcControls = blkMhcControls {
                try driver.encodeMHCRouterNorm(
                    enc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                    gamma: lw.ln1, controls: mhcControls, out: blkLn1,
                    rows: B, mergeBranchOut: i > 0 ? blkDn16 : nil
                )
            } else if i == 0 {
                try driver.encodeRMSNormFast(enc, x: blkX, gamma: lw.ln1, out: blkLn1, rows: B)
            } else {
                try driver.encodeRMSNormResidualFast(
                    enc, x: blkDn16, gamma: lw.ln1, out: blkLn1, residual: blkX,
                    rows: B, rowParallel: false
                )
            }
            enc.memoryBarrier(scope: .buffers)

            try encodeMatrixProj(enc, driver: driver, m: B, n: aGeo.qLoraRank, k: H,
                                 x: blkLn1, w: lw.qDownProj, y16: blkQDownRaw)
            try encodeMatrixProj(enc, driver: driver, m: B, n: aGeo.headDim, k: H,
                                 x: blkLn1, w: lw.kvProj, y16: blkKVRaw)
            enc.memoryBarrier(scope: .buffers)

            let qNormFlags: UInt32 = 1 | (lw.qUpGateProj.isRotated ? 2 : 0)
            encodeNormRoPE(enc, rawIn: blkQDownRaw, normW: lw.qNormW, kvOut: blkQNormRot,
                           rows: B, dim: aGeo.qLoraRank, nopeDim: aGeo.qLoraRank, ropeDim: 0,
                           pos0: 0, ringSize: 1, flags: qNormFlags)
            encodeNormRoPE(enc, rawIn: blkKVRaw, normW: lw.kvNormW, kvOut: blkKV,
                           rows: B, dim: aGeo.headDim, nopeDim: aGeo.nopeDim, ropeDim: aGeo.ropeDim,
                           pos0: pos0, ringSize: config.slidingWindow, flags: 1)
            enc.memoryBarrier(scope: .buffers)

            try encodeMatrixProj(enc, driver: driver, m: B, n: aGeo.qUpStride, k: aGeo.qLoraRank,
                                 x: blkQNormRot, w: lw.qUpGateProj, y16: blkQUpGate)
            enc.memoryBarrier(scope: .buffers)

            encodeQPrepAndBlockAttention(enc, lw: lw, swaKV: slot.swaKV[i], pos0: pos0, B: B)
            enc.memoryBarrier(scope: .buffers)

            encodeVhaInvRopeAndWoA(enc, lw: lw, pos0: pos0, B: B)
            enc.memoryBarrier(scope: .buffers)

            try encodeMatrixProj(enc, driver: driver, m: B, n: H, k: aGeo.zOutStride,
                                 x: blkZGated, w: lw.oProj, y16: blkO16)
            enc.memoryBarrier(scope: .buffers)

            if let mhc = lw.mhc, let mhcStream = blkMhcStream, let mhcControls = blkMhcControls {
                try driver.encodeMHCRouterNorm(
                    enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                    gamma: lw.ln2, controls: mhcControls, out: blkLn2,
                    rows: B, mergeBranchOut: blkO16, moeBridge: nil
                )
            } else {
                try driver.encodeRMSNormResidualFast(
                    enc, x: blkO16, gamma: lw.ln2, out: blkLn2, residual: blkX,
                    rows: B, rowParallel: false
                )
            }
            enc.memoryBarrier(scope: .buffers)

            if let moe = lw.moe, let s = blkMoe {
                let prevEnc = enc
                try QWMoEChain.encode(
                    enc: &enc, hal: hal, moe: moe, s: s,
                    x16: blkLn2, dn32: blkDn32, dn16: blkDn16,
                    rows: B, geometry: draftGeometry, cb: cb, skipOut32: true
                )
                if enc !== prevEnc { useResources(on: enc) }
            } else if let gu = lw.denseGateUp, let dn = lw.denseDown, let act16 = blkDenseAct16 {
                try driver.gemmBatchedH16SwiGLU(
                    enc, m: B, i: dn.k, k: H, x: blkLn2, wQ4: gu, out16: act16,
                    swigluClamp: draftGeometry.routedSwiGLUClamp(layer: i)
                )
                enc.memoryBarrier(scope: .buffers)
                if dn.isRotated {
                    driver.encodeFwht32Fast(enc, x: act16, count: B * dn.k)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeMatrixProj(enc, driver: driver, m: B, n: H, k: dn.k, x: act16, w: dn, y16: blkDn16)
            }
            enc.memoryBarrier(scope: .buffers)

            if i == config.numHiddenLayers - 1 {
                if let _ = lw.mhc, let mhcStream = blkMhcStream, let mhcControls = blkMhcControls,
                   let cw = weights.mhcContractW, let cbW = weights.mhcContractB {
                    try driver.encodeMHCMerge(
                        enc, stream: mhcStream, branchOut: blkDn16, controls: mhcControls,
                        xOut: blkX, nextGamma: lw.ln2, nextNormOut: blkLn2,
                        rows: B, isLeave: false,
                        sinkhornIters: targetGeometry.mhcConfig?.iterations ?? 20, hasNextNorm: false
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try driver.encodeMHCExitHead(
                        enc, stream: mhcStream, headW: cw, headB: cbW,
                        normGamma: weights.finalNorm, outNorm: blkHeadHidden,
                        rows: B, applyNorm: false
                    )
                } else {
                    try driver.encodeRMSNormResidualFast(
                        enc, x: blkDn16, gamma: lw.ln2, out: blkLn2, residual: blkX,
                        rows: B, rowParallel: false
                    )
                    enc.memoryBarrier(scope: .buffers)
                    try QWVerifyWindow.encodeCopy(enc, hal: hal, source: blkX, destination: blkHeadHidden)
                }
                enc.memoryBarrier(scope: .buffers)
            }
        }
    }

    func encodeHeadNormAndProjection(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        rows: Int
    ) throws {
        let H = weights.hidden
        let V = weights.vocab
        let headRot = headRequiresRotatedInput(driver: driver)
        let hiddenSlice = rows < config.blockSize
            ? blkHeadHidden.slice(byteOffset: 0, count: rows * H, dtype: .f16)
            : blkHeadHidden
        let normSlice = rows < config.blockSize
            ? blkHeadNorm.slice(byteOffset: 0, count: rows * H, dtype: .f16)
            : blkHeadNorm
        encodeNormRoPE(enc, rawIn: hiddenSlice, normW: weights.finalNorm, kvOut: normSlice,
                       rows: rows, dim: H, nopeDim: H, ropeDim: 0, pos0: 0,
                       ringSize: 1, flags: 1 | (headRot ? 2 : 0))
        enc.memoryBarrier(scope: .buffers)
        try encodeHeadProjection(enc, driver: driver, normIn: normSlice, rows: rows, vocab: V)
    }

    func headRequiresRotatedInput(driver: QWEngineDriver) -> Bool {
        if let dh = weights.draftHead {
            switch dh {
            case .quantized(let view): return view.isRotated
            case .bf16: return false
            }
        }
        switch driver.headStorage {
        case .q4g(let hw): return hw.view.isRotated
        case .f16, .bf16: return false
        }
    }

    func encodeHeadProjection(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        normIn: QWTensorValue? = nil,
        rows: Int,
        vocab: Int
    ) throws {
        let xIn = normIn ?? blkHeadNorm
        let gsIn = rows < config.blockSize
            ? headGS.slice(byteOffset: 0, count: rows * max(1, weights.hidden / 128), dtype: .f32)
            : headGS
        let yOut = rows < config.blockSize
            ? baseLogits.slice(byteOffset: 0, count: rows * vocab, dtype: .f32)
            : baseLogits
        if let dh = weights.draftHead {
            switch dh {
            case .quantized(let view):
                if rows == 1 {
                    if view.isIncumbentContainer {
                        guard let hw = QWHeadWeight(
                            blocks: view.blocks, n: view.n, k: view.k,
                            bits: view.bits, groupSize: view.groupSize,
                            isRotated: view.isRotated, isIncumbentContainer: true, isSHVQ: view.isSHVQ
                        ) else {
                            throw QWHeadError.badShape("invalid DSpark draft_head weight")
                        }
                        try QWHead.encodeQ4g(enc, hal: hal, weight: hw, x: xIn, gs: gsIn, y: yOut)
                    } else {
                        try QWGemmAQ.encode(enc, hal: hal, blocks: view.blocks, x: xIn, y: yOut,
                                            n: view.n, k: view.k, bits: view.bits, group: view.groupSize,
                                            isSHVQ: view.isSHVQ)
                    }
                } else {
                    try QWHead.encodeBatched(enc, hal: hal, view: view, x: xIn, gs: gsIn, y: yOut, m: rows)
                }
            case .bf16(let w, let n, _):
                try QWHead.encodeBF16(enc, hal: hal, w: w, x: xIn, y: yOut,
                                      geometry: draftGeometry, M: rows, N: n)
            }
            return
        }
        switch driver.headStorage {
        case .q4g(let hw):
            if rows == 1 {
                try driver.head.encode(enc, storage: driver.headStorage, x: xIn, gs: gsIn, y: yOut)
            } else {
                try QWHead.encodeBatched(enc, hal: hal, view: hw.view, x: xIn, gs: gsIn, y: yOut, m: rows)
            }
        case .bf16(let w):
            try QWHead.encodeBF16(enc, hal: hal, w: w, x: xIn, y: yOut,
                                  geometry: targetGeometry, M: rows, N: vocab)
        case .f16(let w):
            try QWHead.encodeF16(enc, hal: hal, w: w, x: xIn, y: yOut,
                                 geometry: targetGeometry, M: rows, N: vocab)
        }
    }

    func encodeGreedyMarkovSteps(
        _ enc: any MTLComputeCommandEncoder,
        count: Int
    ) throws {
        let sampleOff = config.sampleOffset
        let baseByteOff = sampleOff * weights.vocab * 4
        let baseSlice = baseLogits.slice(byteOffset: baseByteOff, count: count * weights.vocab, dtype: .f32)
        let seqPtrRegion = QWMRegion(buffer: tokSeqBuf, base: 0, length: (config.blockSize + 1) * 4)
        let seqTensor = QWTensorValue(region: seqPtrRegion, offset: 0, shape: QWTensorShape([config.blockSize + 1], .i32))
        let rotFlag: UInt32 = weights.markovW2.isRotated ? 1 : 0
        var nTG32 = UInt32(argmaxPartCount)

        for j in 0..<count {
            var mp = DSparkMarkovParams(
                stepIdx: UInt32(j), V: UInt32(weights.vocab), rank: UInt32(config.markovRank),
                flags: rotFlag, partCount: UInt32(argmaxPartCount)
            )
            enc.setComputePipelineState(psoMarkovArgmax)
            baseSlice.bind(enc, index: 0)
            weights.markovW1.bind(enc, index: 1)
            weights.markovW2.blocks.bind(enc, index: 2)
            seqTensor.bind(enc, index: 3)
            keysHi.bind(enc, index: 4)
            keysLo.bind(enc, index: 5)
            enc.setBytes(&mp, length: MemoryLayout<DSparkMarkovParams>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: argmaxPartCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if schedulerEnabled, weights.confidenceHead != nil {
                encodeConfidenceDot(enc, stepIdx: j, hiddenRowIdx: sampleOff + j)
            }
            enc.memoryBarrier(scope: .buffers)

            let outTok = seqTensor.slice(byteOffset: (j + 1) * 4, count: 1, dtype: .i32)
            enc.setComputePipelineState(psoArgmaxFin)
            keysHi.bind(enc, index: 0)
            keysLo.bind(enc, index: 1)
            outTok.bind(enc, index: 2)
            enc.setBytes(&nTG32, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            if j + 1 < count {
                enc.memoryBarrier(scope: .buffers)
            }
        }
    }

    func encodeMarkovLogitsStep(
        _ enc: any MTLComputeCommandEncoder,
        baseLogitsSlice: QWTensorValue,
        stepIdx: Int
    ) {
        let seqTensor = QWTensorValue(
            region: QWMRegion(buffer: tokSeqBuf, base: 0, length: (config.blockSize + 1) * 4),
            offset: 0, shape: QWTensorShape([config.blockSize + 1], .i32)
        )
        var mp = DSparkMarkovParams(
            stepIdx: UInt32(stepIdx), V: UInt32(weights.vocab), rank: UInt32(config.markovRank),
            flags: weights.markovW2.isRotated ? 1 : 0, partCount: UInt32((weights.vocab + 7) / 8)
        )
        enc.setComputePipelineState(psoMarkovLogits)
        baseLogitsSlice.bind(enc, index: 0)
        weights.markovW1.bind(enc, index: 1)
        weights.markovW2.blocks.bind(enc, index: 2)
        seqTensor.bind(enc, index: 3)
        stepLogits.bind(enc, index: 4)
        enc.setBytes(&mp, length: MemoryLayout<DSparkMarkovParams>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (weights.vocab + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeConfidenceDot(
        _ enc: any MTLComputeCommandEncoder,
        stepIdx: Int,
        hiddenRowIdx: Int
    ) {
        guard let confW = weights.confidenceHead else { return }
        let seqTensor = QWTensorValue(
            region: QWMRegion(buffer: tokSeqBuf, base: 0, length: (config.blockSize + 1) * 4),
            offset: 0, shape: QWTensorShape([config.blockSize + 1], .i32)
        )
        var cp = DSparkConfParams(
            stepIdx: UInt32(stepIdx), H: UInt32(weights.hidden), R: UInt32(config.markovRank),
            hiddenRowIdx: UInt32(hiddenRowIdx), V: UInt32(weights.vocab), flags: 0
        )
        enc.setComputePipelineState(psoConfDot)
        blkHeadHidden.bind(enc, index: 0)
        weights.markovW1.bind(enc, index: 1)
        confW.bind(enc, index: 2)
        seqTensor.bind(enc, index: 3)
        confOut.bind(enc, index: 4)
        enc.setBytes(&cp, length: MemoryLayout<DSparkConfParams>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeMatrixProj(
        _ enc: any MTLComputeCommandEncoder,
        driver: QWEngineDriver,
        m: Int, n: Int, k: Int,
        x: QWTensorValue, w: QWWeightView, y16: QWTensorValue
    ) throws {
        if m == 1 {
            try driver.gemvH16T4(enc, n: n, k: k, x: x, gs: headGS, wQ4: w, y16: y16)
        } else {
            try driver.gemmBatchedH16(enc, m: m, n: n, k: k, x: x, wQ4: w, y16: y16)
        }
    }

    func encodeNormRoPE(
        _ enc: any MTLComputeCommandEncoder,
        rawIn: QWTensorValue,
        normW: QWTensorValue,
        kvOut: QWTensorValue,
        rows: Int, dim: Int, nopeDim: Int, ropeDim: Int, pos0: Int, ringSize: Int, flags: UInt32
    ) {
        var kp = DSparkCtxKVParams(
            T: UInt32(rows), dim: UInt32(dim), nopeDim: UInt32(nopeDim), ropeDim: UInt32(ropeDim),
            pos0: UInt32(pos0), ringSize: UInt32(max(1, ringSize)), flags: flags,
            rmsEps: targetGeometry.rmsEps
        )
        enc.setComputePipelineState(psoCtxKVWrite)
        enc.bindTensors(rawIn, normW, invFreqBuf, kvOut, startIndex: 0)
        enc.setBytes(&kp, length: MemoryLayout<DSparkCtxKVParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeQPrepAndBlockAttention(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWDSparkWeights.DraftLayer,
        swaKV: QWTensorValue,
        pos0: Int,
        B: Int
    ) {
        let aGeo = weights.attnGeometry
        var qP = DSparkQPrepParams(
            T: UInt32(B), pos0: UInt32(pos0), qHeads: UInt32(aGeo.qHeads),
            headDim: UInt32(aGeo.headDim), nopeDim: UInt32(aGeo.nopeDim), ropeDim: UInt32(aGeo.ropeDim),
            qUpStride: UInt32(aGeo.qUpStride), indexDim: 0,
            rmsEps: targetGeometry.rmsEps, mscale: 1.0
        )
        enc.setComputePipelineState(psoQPrep)
        blkQUpGate.bind(enc, index: 0)
        invFreqBuf.bind(enc, index: 1)
        enc.setBytes(&qP, length: MemoryLayout<DSparkQPrepParams>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: (aGeo.qHeads + 7) / 8, height: B, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        var attP = DSparkBlockAttnParams(
            B: UInt32(B), qHeads: UInt32(aGeo.qHeads), headDim: UInt32(aGeo.headDim),
            qUpStride: UInt32(aGeo.qUpStride), pos0: UInt32(pos0),
            ringSize: UInt32(config.slidingWindow), slidingWindow: UInt32(config.slidingWindow),
            hasSink: lw.attnSink != nil ? 1 : 0, swaMaskMode: 1
        )
        enc.setComputePipelineState(psoBlockAttn)
        blkQUpGate.bind(enc, index: 0)
        swaKV.bind(enc, index: 1)
        blkKV.bind(enc, index: 2)
        (lw.attnSink ?? zeroSinkBuf).bind(enc, index: 3)
        blkAttOut.bind(enc, index: 4)
        enc.setBytes(&attP, length: MemoryLayout<DSparkBlockAttnParams>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (aGeo.qHeads + 7) / 8, height: B, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeVhaInvRopeAndWoA(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWDSparkWeights.DraftLayer,
        pos0: Int,
        B: Int
    ) {
        let aGeo = weights.attnGeometry
        let skipInter = !lw.woA.isRotated
        var vhaP = DSparkVhaInvRopeParams(
            T: UInt32(B), pos0: UInt32(pos0), qHeads: UInt32(aGeo.qHeads),
            headDim: UInt32(aGeo.headDim), nopeDim: UInt32(aGeo.nopeDim), ropeDim: UInt32(aGeo.ropeDim),
            invMscale: 1.0, hasVHA: (lw.vhaMix != nil ? 1 : 0) | (skipInter ? 0x10000 : 0)
        )
        enc.setComputePipelineState(psoVhaInvRope)
        enc.bindTensors(blkAttOut, lw.vhaMix ?? zeroVhaBuf, invFreqBuf, blkZVha, startIndex: 0)
        enc.setBytes(&vhaP, length: MemoryLayout<DSparkVhaInvRopeParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: aGeo.headDim / 32, height: B, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        let baseInc: UInt32 = lw.woA.isIncumbentContainer ? 1 : (lw.woA.isSHVQ ? 2 : 0)
        let skipExit = !lw.oProj.isRotated
        var woP = DSparkGroupedWoAParams(
            T: UInt32(B), oGroups: UInt32(aGeo.oGroups), oLoraRank: UInt32(aGeo.oLoraRank),
            groupIn: UInt32(aGeo.groupIn), qUpStride: UInt32(aGeo.qUpStride),
            oGateOffset: UInt32(aGeo.qHeads * aGeo.headDim),
            bits: UInt32(lw.woA.bits), group: UInt32(lw.woA.groupSize),
            isIncumbent: baseInc | (skipExit ? 0x10000 : 0)
        )
        let isSub2 = baseInc == 2 && (lw.woA.bits == 1 || lw.woA.bits == 15)
        let useMR = B > 1 && B <= 8 && (aGeo.oLoraRank % 32 == 0) && (aGeo.groupIn % 32 == 0)
        let pso: any MTLComputePipelineState = useMR
            ? (isSub2 ? psoGroupedWoAGemvMRSub2 : psoGroupedWoAGemvMR)
            : (isSub2 ? psoGroupedWoAGemvSub2 : psoGroupedWoAGemv)
        enc.setComputePipelineState(pso)
        enc.bindTensors(blkZVha, lw.woA.blocks, blkQUpGate, blkZGated, startIndex: 0)
        enc.setBytes(&woP, length: MemoryLayout<DSparkGroupedWoAParams>.stride, index: 4)
        let grid = useMR
            ? MTLSize(width: aGeo.oLoraRank / 32, height: aGeo.oGroups, depth: 1)
            : MTLSize(width: aGeo.oLoraRank / 32, height: aGeo.oGroups, depth: B)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeGreedyProfiledCBs(
        driver: QWEngineDriver,
        slot: SessionSlot,
        pos0: Int,
        draftLimit: Int,
        headRows: Int,
        journalBuffer: (any MTLBuffer)?
    ) throws -> (blockCB: any MTLCommandBuffer, headCB: any MTLCommandBuffer, markovCB: any MTLCommandBuffer) {
        guard let blockCB = hal.makeUnretainedCommandBuffer(),
              let headCB = hal.makeUnretainedCommandBuffer(),
              let markovCB = hal.makeUnretainedCommandBuffer() else {
            throw QWEngineDriverError.state("DSpark profile command buffer failed")
        }
        var encBlock = try makeProposeEncoder(blockCB)
        do {
            try encodeBlockTrunk(&encBlock, cb: blockCB, driver: driver, slot: slot, pos0: pos0)
        } catch {
            encBlock.endEncoding()
            throw error
        }
        encBlock.endEncoding()
        blockCB.commit()

        let encHead = try makeProposeEncoder(headCB)
        do {
            useResources(on: encHead)
            try encodeHeadNormAndProjection(encHead, driver: driver, rows: headRows)
        } catch {
            encHead.endEncoding()
            throw error
        }
        encHead.endEncoding()
        headCB.commit()

        let encMarkov = try makeProposeEncoder(markovCB)
        do {
            useResources(on: encMarkov)
            try encodeGreedyMarkovSteps(encMarkov, count: draftLimit)
        } catch {
            encMarkov.endEncoding()
            throw error
        }
        encMarkov.endEncoding()
        if let journalBuffer {
            guard let blit = markovCB.makeBlitCommandEncoder() else {
                throw QWEngineDriverError.state("DSpark journal blit encoder failed")
            }
            blit.copy(from: tokSeqBuf, sourceOffset: 0, to: journalBuffer, destinationOffset: 0,
                      size: (1 + draftLimit) * 4)
            blit.endEncoding()
        }
        markovCB.commit()
        return (blockCB, headCB, markovCB)
    }

    func logProfileStagesIfNeeded(
        stages: (blockCB: any MTLCommandBuffer, headCB: any MTLCommandBuffer, pos0: Int)?,
        markovCB: any MTLCommandBuffer,
        drafts: Int
    ) {
        guard let stages else { return }
        stages.blockCB.waitUntilCompleted()
        stages.headCB.waitUntilCompleted()
        let blockMs = (try? QWMTPStageTiming.gpuDurationMs(stages.blockCB)) ?? 0
        let headMs = (try? QWMTPStageTiming.gpuDurationMs(stages.headCB)) ?? 0
        let markovMs = (try? QWMTPStageTiming.gpuDurationMs(markovCB)) ?? 0
        let line = String(
            format: "[dspark-profile] pos=%d drafts=%d block_moe=%.2fms head=%.2fms markov=%.2fms total=%.2fms\n",
            stages.pos0, drafts, blockMs, headMs, markovMs, blockMs + headMs + markovMs
        )
        FileHandle.standardError.write(Data(line.utf8))
    }
}
