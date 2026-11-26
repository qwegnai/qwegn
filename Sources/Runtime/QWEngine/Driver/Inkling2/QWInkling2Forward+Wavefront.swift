// QWInkling2Forward+Wavefront.swift — Slot-targeted prefill/decode and B=2..4 wavefront forward step.
import Foundation
import Metal
import QWAlloc
import QWCore

extension QWInkling2Forward {
    /// Executes chunked prefill or single-slot decode into an arbitrary `Ink2SlotState`.
    @discardableResult
    func forwardOnSlot(
        slot: Ink2SlotState,
        tokens: [Int],
        posOffset: Int,
        runHeadRows: Int,
        greedyHead: Bool = false,
        useLSA: Bool = false,
        profiler: QWInkling2Profiler? = nil,
        onPrefillProgress: ((Int, Int) -> Void)? = nil
    ) throws -> Int {
        guard !tokens.isEmpty, posOffset >= 0, posOffset + tokens.count <= slot.capacity else {
            throw QWInkling2Error.badState(
                "invalid forwardOnSlot range pos=\(posOffset) count=\(tokens.count) cap=\(slot.capacity)"
            )
        }
        guard runHeadRows >= 0, runHeadRows <= min(16, tokens.count) else {
            throw QWInkling2Error.badState("invalid forwardOnSlot runHeadRows=\(runHeadRows)")
        }
        let maxChunk = session.maxChunk
        if tokens.count <= maxChunk {
            try forward(
                tokens: tokens,
                posOffset: posOffset,
                runHeadRows: runHeadRows,
                greedyHead: greedyHead,
                updateState: true,
                slot: slot,
                useLSA: useLSA,
                profiler: profiler
            )
            onPrefillProgress?(tokens.count, tokens.count)
            return tokens.count
        }
        var consumed = 0
        var lastChunkRows = 0
        while consumed < tokens.count {
            let rem = tokens.count - consumed
            let chunkSize: Int
            if rem > maxChunk {
                let nChunks = (rem + maxChunk - 1) / maxChunk
                chunkSize = min(maxChunk, max(64, ((rem + nChunks - 1) / nChunks + 63) & ~63))
            } else {
                chunkSize = rem
            }
            let nextEnd = consumed + chunkSize
            let isLast = (nextEnd == tokens.count)
            let chunk = Array(tokens[consumed..<nextEnd])
            try forward(
                tokens: chunk,
                posOffset: posOffset + consumed,
                runHeadRows: isLast ? runHeadRows : 0,
                greedyHead: isLast && greedyHead,
                updateState: true,
                slot: slot,
                useLSA: useLSA,
                profiler: profiler
            )
            lastChunkRows = chunk.count
            consumed = nextEnd
            onPrefillProgress?(consumed, tokens.count)
        }
        return lastChunkRows
    }

    private func sliceAttnProjRow(
        _ proj: Ink2AttnProjViews,
        row b: Int,
        lw: QWInkling2Weights.LayerWeights,
        qDim: Int,
        rDim: Int,
        nH: Int,
        kvDim: Int
    ) -> (views: Ink2AttnProjViews, kStride: Int) {
        let totalRows = lw.projLayout.totalRows
        let qStride = (proj.qBuf === session.fusedAttnProjBuf) ? totalRows : qDim
        let gateStride = (proj.gateBuf === session.fusedAttnProjBuf) ? totalRows : qDim
        let rStride = (proj.rBuf === session.fusedAttnProjBuf) ? totalRows : rDim
        let sinkStride = (proj.sinkBuf === session.fusedAttnProjBuf) ? totalRows : nH
        let kStride = (proj.kBuf === session.fusedAttnProjBuf) ? totalRows : kvDim
        let vStride = (proj.vBuf === session.fusedAttnProjBuf) ? totalRows : kvDim
        let rowViews = Ink2AttnProjViews(
            qBuf: proj.qBuf, qOff: proj.qOff + b * qStride * 2,
            gateBuf: proj.gateBuf, gateOff: proj.gateOff + b * gateStride * 2,
            rBuf: proj.rBuf, rOff: proj.rOff + b * rStride * 2,
            sinkBuf: proj.sinkBuf, sinkOff: proj.sinkOff + b * sinkStride * 2,
            kBuf: proj.kBuf, kOff: proj.kOff + b * kStride * 2,
            vBuf: proj.vBuf, vOff: proj.vOff + b * vStride * 2
        )
        return (rowViews, kStride)
    }

    /// Executes one B=2..4 continuous-wavefront decode step across distinct active slots in 1 CB.
    func forwardWavefrontStep(
        slots: [Ink2SlotState],
        positions: [Int],
        tokens: [Int],
        greedyHead: Bool = false,
        profiler: QWInkling2Profiler? = nil
    ) throws {
        let B = slots.count
        guard (2...4).contains(B), positions.count == B, tokens.count == B, B <= session.maxChunk else {
            throw QWInkling2Error.badState("invalid wavefront batch size B=\(B)")
        }
        let d = session.descriptor
        let ink = session.ink2
        var seenSlotMask = 0
        for b in 0..<B {
            let slotBit = 1 << slots[b].slotIndex
            guard (seenSlotMask & slotBit) == 0 else {
                throw QWInkling2Error.badState("duplicate wavefront slot index \(slots[b].slotIndex)")
            }
            seenSlotMask |= slotBit
            guard tokens[b] >= 0, tokens[b] < d.vocab else {
                throw QWInkling2Error.badState("invalid wavefront token \(tokens[b])")
            }
            guard positions[b] >= 0, positions[b] < slots[b].capacity else {
                throw QWInkling2Error.badState(
                    "wavefront position \(positions[b]) exceeds slot \(slots[b].slotIndex) capacity \(slots[b].capacity)"
                )
            }
        }

        let H = d.hidden
        let nH = d.qHeads
        let nKV = d.kvHeads
        let headDim = d.headDim
        let qDim = nH * headDim
        let kvDim = nKV * headDim
        let rDim = nH * ink.dRel
        let moeI = d.moeIntermediate ?? 1152
        let nExp = d.numExperts ?? 128
        let topK = d.expertsPerTok ?? 8
        let eps = d.rmsEps
        let xBytes = B * H * 2
        let thStride = B * H

        let tokPtr = session.tokIdsBuf.contents().assumingMemoryBound(to: Int32.self)
        for b in 0..<B { tokPtr[b] = Int32(tokens[b]) }

        let hostStartNs = profiler?.beginHostTimer() ?? 0
        let cbOpt = (profiler == nil)
            ? (hal.queue.makeCommandBufferWithUnretainedReferences() ?? hal.queue.makeCommandBuffer())
            : hal.queue.makeCommandBuffer()
        guard let cb = cbOpt else {
            throw QWInkling2Error.badState("failed to create wavefront command buffer")
        }
        var encOpt: (any MTLComputeCommandEncoder)? = nil
        func encoder(_ stage: Ink2Stage? = nil) throws -> any MTLComputeCommandEncoder {
            if let stage, let profiler {
                profiler.beginStage(stage, on: cb, encoder: &encOpt, hal: hal)
            }
            if let e = encOpt { return e }
            guard let e = cb.makeComputeCommandEncoder() else {
                throw QWInkling2Error.badState("failed to create compute encoder")
            }
            hal.touchKeepAlive(e)
            encOpt = e
            return e
        }

        var (curXBuf, curXOff) = initialXTarget(ink: ink, xBytes: xBytes)
        do {
            let e0 = try encoder(.embed)
            var ep = EmbedP(T: UInt32(B), H: UInt32(H), V: UInt32(d.vocab), eps: eps)
            e0.setComputePipelineState(pEmbed)
            e0.setBuffer(session.tokIdsBuf, offset: 0, index: 0)
            e0.setBuffer(weights.embedTokens.buffer, offset: weights.embedTokens.offset, index: 1)
            e0.setBuffer(weights.embedNorm.buffer, offset: weights.embedNorm.offset, index: 2)
            e0.setBuffer(curXBuf, offset: curXOff, index: 3)
            e0.setBytes(&ep, length: MemoryLayout<EmbedP>.stride, index: 4)
            e0.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        var blocksSlots: [Int] = []
        var preludeBlocksCount = 0
        var lastBankStateCount = 0
        var bankValid = false

        for visit in 0..<session.executionPlan.count {
            let lIdx = session.executionPlan[visit]
            let lw = weights.layers[lIdx]
            let cBase = visit * 4

            try encodeLoopCoreTransition(
                visit: visit, lIdx: lIdx, t: B, H: H, thStride: thStride, eps: eps, ink: ink,
                curXBuf: &curXBuf, curXOff: &curXOff, blocksSlots: &blocksSlots,
                preludeBlocksCount: &preludeBlocksCount, lastBankStateCount: &lastBankStateCount,
                bankValid: &bankValid, beginEncoder: encoder
            )

            let useBank = bankValid && lIdx >= ink.loopCoreStart && lIdx < ink.loopCoreEnd
            if useBank {
                let bEnc = try encoder(.loopResCtrlNorm)
                encodeLoopBankAttend(
                    bEnc, lIdx: lIdx, t: B, H: H, thStride: thStride,
                    lastBankStateCount: lastBankStateCount, eps: eps, ink: ink
                )
            }

            let enc = try encoder(.loopResCtrlNorm)
            let (kNormOff, vNormOff) = try encodePreAttnCtrlNorm(
                enc, lIdx: lIdx, lw: lw, useBank: useBank, t: B, H: H, thStride: thStride,
                eps: eps, ink: ink, blocksSlots: blocksSlots, curXBuf: curXBuf, curXOff: curXOff
            )

            let attnProjEnc = try encoder(.denseAttnProj)
            let proj = try encodeAttnProjections(
                attnProjEnc, lw: lw, t: B, kNormOff: kNormOff, vNormOff: vNormOff
            )

            for b in 0..<B {
                let (rowProj, kStride) = sliceAttnProjRow(
                    proj, row: b, lw: lw, qDim: qDim, rDim: rDim, nH: nH, kvDim: kvDim
                )
                try encodeKVSConvAndQKVR(
                    lw: lw, kv: slots[b].visitKVs[visit], proj: rowProj, convSlotBase: cBase,
                    convSlab: slots[b].convStatesSlab,
                    kNormOff: kNormOff + b * H * 2, vNormOff: vNormOff + b * H * 2,
                    qPrepOff: b * qDim * 2, rPrepOff: b * rDim * 2,
                    kConvOff: b * kvDim * 2, vConvOff: b * kvDim * 2,
                    dynWOff: b * nKV * 4 * 4, kvInRowStride: kStride,
                    t: 1, posOffset: positions[b], updateState: true,
                    recordVerifyConv: false, writeKConv: false, treePlan: nil,
                    beginEncoder: encoder
                )
            }

            let faEnc = try encoder(.flashAttn)
            for b in 0..<B {
                let (rowProj, _) = sliceAttnProjRow(
                    proj, row: b, lw: lw, qDim: qDim, rDim: rDim, nH: nH, kvDim: kvDim
                )
                try encodeAttentionBlock(
                    faEnc, visit: visit, lw: lw, kv: slots[b].visitKVs[visit], proj: rowProj,
                    t: 1, posOffset: positions[b],
                    qPrepOff: b * qDim * 2, rPrepOff: b * rDim * 2, attnOutOff: b * qDim * 2,
                    useLSA: false, treePlan: nil
                )
            }

            let oEnc = try encoder(.denseAttnProj)
            try encodeLinear(oEnc, weight: lw.oProjW, xBuf: session.attnOutBuf, yBuf: session.oOutBuf, m: B)

            let isBoundary = ink.isAttnResBlockBoundary(lIdx)
            if isBoundary {
                let nextSlot = blocksSlots.count
                if !(curXBuf === session.blocksRingBuf && curXOff == nextSlot * xBytes) {
                    let bEnc = try encoder(.loopResCtrlNorm)
                    encodeCopyBF16(bEnc, src: curXBuf, srcOff: curXOff,
                                   dst: session.blocksRingBuf, dstOff: nextSlot * xBytes, count: thStride)
                }
                blocksSlots.append(nextSlot)
            }

            let enc2 = try encoder(.sconv)
            for b in 0..<B {
                encodeSConvStaticTap(
                    enc2, xBF16: session.oOutBuf, xOff: b * H * 2,
                    weight: lw.attnSConv, convSlot: cBase + 2, convSlab: slots[b].convStatesSlab,
                    resBF16: curXBuf, resBF16Off: curXOff + b * H * 2,
                    resF32: session.dummyF, resF32Off: 0,
                    outBF16: session.dummyB, outBF16Off: 0,
                    outF32: session.resF32Buf, outF32Off: b * H * 4,
                    t: 1, c: H, resMode: isBoundary ? 1 : 2, denom: 5.0,
                    updateState: true, recordVerifyConv: false, treePlan: nil
                )
            }

            let resEnc = try encoder(.loopResCtrlNorm)
            try encodePostAttnCtrlNorm(
                resEnc, lw: lw, useBank: useBank, t: B, H: H, thStride: thStride,
                eps: eps, ink: ink, blocksSlots: blocksSlots
            )

            encOpt = try encodeMoEBlock(
                lw: lw, t: B, H: H, nExp: nExp, topK: topK, moeI: moeI, ink: ink,
                profiler: profiler, cb: cb, beginEncoder: encoder, moeSpecBudget: 0
            )

            let (nextBuf, nextOff) = nextXTarget(
                nextVisit: visit + 1, blocksSlots: blocksSlots, ink: ink, xBytes: xBytes
            )

            let enc3 = try encoder(.sconv)
            for b in 0..<B {
                encodeSConvStaticTap(
                    enc3, xBF16: session.moeOutBuf, xOff: b * H * 2,
                    weight: lw.mlpSConv, convSlot: cBase + 3, convSlab: slots[b].convStatesSlab,
                    resBF16: session.dummyB, resBF16Off: 0,
                    resF32: session.resF32Buf, resF32Off: b * H * 4,
                    outBF16: nextBuf, outBF16Off: nextOff + b * H * 2,
                    outF32: session.dummyF, outF32Off: 0,
                    t: 1, c: H, resMode: 3, denom: 5.0,
                    updateState: true, recordVerifyConv: false, treePlan: nil
                )
            }
            curXBuf = nextBuf
            curXOff = nextOff
        }

        try encodeFinalNormAndHead(
            t: B, H: H, thStride: thStride, xBytes: xBytes, eps: eps,
            blocksSlots: blocksSlots, partialBF16: curXBuf, partialBF16Off: curXOff,
            captureTapBuf: nil, finalTapSlot: nil,
            runHeadRows: B, greedyHead: greedyHead, beginEncoder: encoder
        )

        profiler?.endActiveStage(encoder: &encOpt)
        encOpt?.endEncoding()
        profiler?.endHostTimer(hostStartNs)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("Metal wavefront execution failed: \(err.localizedDescription)")
        }
        profiler?.finishCommandBuffer(cb, tokens: B)
    }
}
