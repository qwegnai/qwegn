// QWInkling2Runtime+DFlash.swift — DFlash 2 startup sizing, prefill tap ingest,
// and greedy/stochastic block-diffusion speculative decode on Inkling-2.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWHead
import QWMTP
import QWQuant

extension QWInkling2Runtime {
    struct DFlashStartupSelection {
        let path: String
        let config: QWDFlashConfig
        let weightBytes: Int
        let fixedBytes: Int
    }

    static func resolveDFlashStartup(
        descriptor: QWModelDescriptor,
        packPath: String,
        mtpBackend: QWMTPBackend,
        requestedDrafts: Int,
        dflashDraftPackPath: String?,
        dflashSelectorTopK: Int?,
        prefillChunkSize: Int
    ) throws -> DFlashStartupSelection? {
        guard mtpBackend == .dflash2, requestedDrafts > 0 else { return nil }
        let explicit = dflashDraftPackPath?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let rawPath = explicit.isEmpty
            ? QWDFlashConfig.defaultPackPath(for: descriptor, targetPackPath: packPath)
            : explicit.expandingTildePath
        let url = rawPath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            if !explicit.isEmpty {
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: descriptor, packPath: url.path,
                    drafts: requestedDrafts, selectorTopK: dflashSelectorTopK
                )
            }
            return nil
        }
        guard QWDFlashConfig.isDrafterPack(at: url) else {
            if !explicit.isEmpty {
                try QWDFlashConfig.validateDrafter(
                    targetDescriptor: descriptor, packPath: url.path,
                    drafts: requestedDrafts, selectorTopK: dflashSelectorTopK
                )
            }
            return nil
        }
        let cfg = QWDFlashConfig.load(from: url, topKOverride: dflashSelectorTopK, target: descriptor)
        guard !cfg.isDSpine else { return nil }
        let validDrafts = min(requestedDrafts, max(1, cfg.blockSize - 1))
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: descriptor, packPath: url.path,
            drafts: validDrafts, selectorTopK: dflashSelectorTopK
        )
        let weightEst = QWDFlashFootprint.drafterWeightBytes(packPath: url.path, isDSpine: false)
        let maxChunkPlan = max(32, min(8192, prefillChunkSize))
        let baseScratch = QWDFlashFootprint.estimateScratchBytes(config: cfg, sessionSlotCount: 1)
        let extraTapBytes = max(0, maxChunkPlan - cfg.maxCacheSize) * cfg.targetLayerIds.count * descriptor.hidden * 2
        return DFlashStartupSelection(
            path: url.path, config: cfg, weightBytes: weightEst, fixedBytes: baseScratch + extraTapBytes
        )
    }

    func ingestDFlashPrefillSlice(
        df: QWDFlashDrafter,
        tapBuf: any MTLBuffer,
        chunkRows: Int,
        slice: QWDFlashConfig.PrefillCaptureSlice
    ) throws {
        guard slice.captureRows > 0 else { return }
        guard let (cb, enc) = hal.beginUnretained(.serial) else {
            throw QWInkling2Error.badState("DFlash prefill tap ingest command buffer failed")
        }
        df.encodePackBF16Taps(
            enc, source: tapBuf, srcPlaneStrideElems: chunkRows * descriptor.hidden,
            srcRowOffset: slice.localStart, dstRowOffset: 0, rows: slice.captureRows
        )
        enc.memoryBarrier(scope: .buffers)
        try df.encodeContextUpdate(
            enc, slot: df.sessionSlots[0], startPosition: slice.ctxStartPos, totalCount: slice.captureRows
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw QWInkling2Error.badState("DFlash prefill tap ingest failed: \(err.localizedDescription)")
        }
    }

    func commitVerifyAndIngestDFlash(
        df: QWDFlashDrafter,
        tapBuf: any MTLBuffer,
        rowStride: Int,
        oldPos: Int,
        kept: Int
    ) throws {
        guard kept > 0 else { return }
        try withGPUTurn {
            guard let (cb, enc) = hal.beginUnretained(.serial) else {
                throw QWInkling2Error.badState("DFlash post-verify ingest command buffer failed")
            }
            try session.encodeCommitVerifyConv(enc, hal: hal, kept: kept)
            df.encodePackBF16Taps(
                enc, source: tapBuf, srcPlaneStrideElems: rowStride * descriptor.hidden,
                srcRowOffset: 0, dstRowOffset: 0, rows: kept
            )
            enc.memoryBarrier(scope: .buffers)
            try df.encodeContextUpdate(enc, slot: df.sessionSlots[0], startPosition: oldPos, totalCount: kept)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error {
                throw QWInkling2Error.badState("DFlash post-verify ingest failed: \(err.localizedDescription)")
            }
        }
    }

    func runDFlashSpeculativeDecode(
        dflash: QWDFlashDrafter,
        tapBuf: any MTLBuffer,
        requestedDrafts: Int,
        allowPLD: Bool,
        sampler: QWSampler,
        promptTokens: [Int],
        pos: inout Int,
        effectiveMaxNew: Int,
        stopIDs: Set<Int>,
        generated: inout [Int],
        recentGeneratedForPenalty: inout [Int],
        penaltyHistory: inout [Int],
        stepMs: inout [Double],
        stepCounts: inout [Int],
        stopTokenID: inout Int?,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?,
        profiler: QWInkling2Profiler? = nil
    ) throws -> (stats: SpeculativeStats, finished: Bool) {
        let clock = ContinuousClock()
        let V = ink2.unpaddedVocabSize
        let H = descriptor.hidden
        let isGreedy = sampler.config.isGreedy
        let usePenalty = sampler.config.repetitionPenalty != 1.0
        let logitsPtr = session.logitsF32Buf.contents().assumingMemoryBound(to: Float.self)
        let headTokPtr = session.headTokensBuf.contents().assumingMemoryBound(to: Int32.self)
        let dSlot = dflash.sessionSlots[0]
        let dflashSeed: UInt64 = sampler.config.hasExplicitSeed
            ? (sampler.config.seed ^ 0x4446_4C53_9E37_79B9)
            : UInt64.random(in: 1...UInt64.max)
        let stream: QWMTPRandomStream? = isGreedy
            ? nil
            : QWMTPRandomStream(seed: dflashSeed, drawIndex: sampler.drawsConsumed)

        var stats = SpeculativeStats(label: "dflash")
        session.tokenRecycler.reset()
        session.tokenRecycler.ingestSequence(promptTokens)
        if let lastPrompt = promptTokens.last {
            let prev = promptTokens.count >= 2 ? promptTokens[promptTokens.count - 2] : nil
            session.tokenRecycler.recordTransition(previous: prev, current: lastPrompt, next: generated[0])
        }

        var history = [Int]()
        history.reserveCapacity(promptTokens.count + effectiveMaxNew)
        history.append(contentsOf: promptTokens)
        history.append(generated[0])

        let pldCap = min(15, max(7, requestedDrafts))
        let linCap = min(requestedDrafts, max(1, dflash.config.blockSize - 1))

        func maxVerifyDrafts(cap: Int) -> Int {
            let remB = effectiveMaxNew - generated.count
            let remC = (capacity - pos) - 2
            return max(0, min(cap, remB - 1, remC))
        }

        func proposeNextDrafts(
            anchor: Int,
            stream: QWMTPRandomStream?,
            pendingIngest: (rowStride: Int, oldPos: Int, kept: Int)? = nil
        ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal], usedPLD: Bool) {
            if pendingIngest == nil && allowPLD && isGreedy {
                let pld = makePLDDrafts(history: history, maxCount: maxVerifyDrafts(cap: pldCap), allowRecyclerFallback: false)
                if !pld.isEmpty { return (pld, [], true) }
            }
            let cnt = maxVerifyDrafts(cap: linCap)
            guard cnt > 0 else {
                if let pi = pendingIngest {
                    try commitVerifyAndIngestDFlash(df: dflash, tapBuf: tapBuf, rowStride: pi.rowStride, oldPos: pi.oldPos, kept: pi.kept)
                }
                return ([], [], false)
            }
            let draftStream: QWMTPRandomStream? = try stream.map { s in
                let draftBase = try s.reserveDraws(UInt64(cnt + 1) * UInt64(V))
                let ds = QWMTPRandomStream(seed: s.seed, drawIndex: draftBase)
                try ds.beginGumbelRound(drafts: cnt, vocabulary: V, maxDrafts: max(7, dflash.config.blockSize - 1))
                return ds
            }
            do {
                let preEncodeHook: ((_ enc: any MTLComputeCommandEncoder) throws -> Void)? = pendingIngest.map { pi in
                    { enc in
                        try self.session.encodeCommitVerifyConv(enc, hal: self.hal, kept: pi.kept)
                        dflash.encodePackBF16Taps(
                            enc, source: tapBuf, srcPlaneStrideElems: pi.rowStride * H,
                            srcRowOffset: 0, dstRowOffset: 0, rows: pi.kept
                        )
                    }
                }
                let preCtx = pendingIngest.map { (slot: dSlot, startPosition: $0.oldPos, totalCount: $0.kept) }
                let res = try withGPUTurn {
                    try dflash.propose(
                        hal: hal, slot: dSlot, cacheOffset: pos, pendingToken: anchor,
                        draftCount: cnt,
                        draftVocab: QWDFlashDrafter.resolveDraftVocab(fullVocab: V, defaultCap: nil) ?? V,
                        headStorage: self.weights.lmHeadStorage,
                        encodeHeadLogits: { enc, xRot16, xNorm16, _, yLogits32, rows, vocabRows in
                            try self.forwardPass.encodeDFlashHeadLogits(
                                enc, xNorm16: xNorm16, xRot16: xRot16,
                                logitsOut: yLogits32, rows: rows, vocabRows: vocabRows
                            )
                        },
                        encodeAnchorMaskEmbed: { enc, propX32, anchorTok, maskTok, rows in
                            let tokPtr = self.session.tokIdsBuf.contents().assumingMemoryBound(to: Int32.self)
                            withExtendedLifetime(self.session.tokIdsBuf) {
                                tokPtr[0] = Int32(anchorTok)
                                tokPtr[1] = Int32(maskTok)
                            }
                            var ep = QWInkling2Forward.EmbedP(
                                T: 2, H: UInt32(H), V: UInt32(self.descriptor.vocab), eps: self.descriptor.rmsEps
                            )
                            enc.setComputePipelineState(self.forwardPass.pEmbed)
                            enc.setBuffer(self.session.tokIdsBuf, offset: 0, index: 0)
                            enc.setBuffer(self.weights.embedTokens.buffer, offset: self.weights.embedTokens.offset, index: 1)
                            enc.setBuffer(self.weights.embedNorm.buffer, offset: self.weights.embedNorm.offset, index: 2)
                            enc.setBuffer(self.session.oOutBuf, offset: 0, index: 3)
                            enc.setBytes(&ep, length: MemoryLayout<QWInkling2Forward.EmbedP>.stride, index: 4)
                            enc.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1),
                                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                            enc.memoryBarrier(scope: .buffers)
                            dflash.encodeStagePropBF16(
                                enc, tableBuffer: self.session.oOutBuf, byteOffset: 0,
                                propX: propX32, anchorToken: 0, maskToken: 1, rows: rows
                            )
                        },
                        stochastic: draftStream.map { (Float(max(sampler.config.temperature, 1e-6)), $0) },
                        preEncode: preEncodeHook,
                        preContextUpdate: preCtx
                    )
                }
                return (res.drafts, res.proposals, false)
            } catch {
                draftStream?.cancelActiveRound()
                throw error
            }
        }

        var nextProposal = try proposeNextDrafts(anchor: generated[generated.count - 1], stream: stream)
        if nextProposal.drafts.isEmpty { return (stats, false) }

        var verifyTokens = [Int]()
        verifyTokens.reserveCapacity(16)
        var lowYieldStreak = 0
        var cooldownSteps = 0

        while !nextProposal.drafts.isEmpty && generated.count < effectiveMaxNew {
            let roundStart = clock.now
            let drafts = nextProposal.drafts
            let proposals = nextProposal.proposals
            let usedPLD = nextProposal.usedPLD
            let anchor = generated[generated.count - 1]
            verifyTokens.removeAll(keepingCapacity: true)
            verifyTokens.append(anchor)
            verifyTokens.append(contentsOf: drafts)
            let B = verifyTokens.count

            let runGreedyHead = isGreedy && (!usePenalty || penaltyHistory.isEmpty)
            try withGPUTurn {
                try forwardPass.forward(
                    tokens: verifyTokens, posOffset: pos, runHeadRows: B, greedyHead: runGreedyHead,
                    updateState: true, captureTapBuf: tapBuf, tapVisits: dflash.config.targetLayerIds,
                    recordVerifyConv: true, profiler: profiler, moeSpecBudget: activeMoESpecBudget(verifyRows: B)
                )
            }

            let draftProposals: [QWSpeculativeRoundResolver.DraftProposal] = !proposals.isEmpty
                ? proposals.map { .distribution(token: $0.token, distribution: $0.distribution) }
                : drafts.map { .greedyToken($0) }
            let roundRes = try QWSpeculativeRoundResolver.resolve(
                draftProposals: draftProposals,
                stream: stream,
                isGreedy: isGreedy,
                maxNew: effectiveMaxNew - generated.count,
                stopIDs: stopIDs
            ) { row, draftPrefix, _ in
                if runGreedyHead {
                    return try QWSamplingDistribution.sparse(vocabulary: V, ids: [Int(headTokPtr[row])], weights: [1.0])
                }
                let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * V), count: V)
                let rep = usePenalty
                    ? Self.loopPenaltyHistory(
                        baseRecent: recentGeneratedForPenalty,
                        baseActive: penaltyHistory,
                        draftPrefix: draftPrefix,
                        vocab: V
                    )
                    : []
                if isGreedy {
                    let tok = try sampler.penalisedGreedyToken(logits: rowPtr, repetitionHistory: rep)
                    return try QWSamplingDistribution.sparse(vocabulary: V, ids: [tok], weights: [1.0])
                }
                return try sampler.distribution(logits: rowPtr, repetitionHistory: rep)
            }
            let (accepted, emitted, stopped, cancelled) = try consumeResolvedRound(
                roundRes, vocab: V, effectiveMaxNew: effectiveMaxNew, stopIDs: stopIDs,
                usePenalty: usePenalty,
                recentGeneratedForPenalty: &recentGeneratedForPenalty,
                penaltyHistory: &penaltyHistory,
                history: &history, generated: &generated, stopTokenID: &stopTokenID,
                onStopToken: onStopToken, onToken: onToken
            )

            let oldPos = pos
            let kept = 1 + accepted
            pos += kept
            var pendingIngest: (rowStride: Int, oldPos: Int, kept: Int)? = (B, oldPos, kept)
            func flushPendingIngest() throws {
                if let pi = pendingIngest {
                    try commitVerifyAndIngestDFlash(df: dflash, tapBuf: tapBuf, rowStride: pi.rowStride, oldPos: pi.oldPos, kept: pi.kept)
                    pendingIngest = nil
                }
            }

            let elapsedMs = elapsedSeconds(roundStart.duration(to: clock.now)) * 1000.0
            stepMs.append(elapsedMs)
            stepCounts.append(emitted)
            stats.recordRound(emitted: emitted, proposals: drafts.count, accepted: accepted)

            if stopped || cancelled || generated.count >= effectiveMaxNew {
                try flushPendingIngest()
                return (stats, stopped || cancelled || generated.count >= effectiveMaxNew)
            }
            if !usedPLD {
                if accepted == 0 { lowYieldStreak += 1 } else { lowYieldStreak = 0 }
                if lowYieldStreak >= 2 { cooldownSteps = 3; lowYieldStreak = 0 }
            }
            var hasCooldownPLD = false
            while cooldownSteps > 0 && generated.count < effectiveMaxNew {
                try flushPendingIngest()
                if allowPLD && isGreedy {
                    let pld = makePLDDrafts(history: history, maxCount: maxVerifyDrafts(cap: pldCap), allowRecyclerFallback: false)
                    if !pld.isEmpty { cooldownSteps = 0; nextProposal = (pld, [], true); hasCooldownPLD = true; break }
                }
                let cStart = clock.now
                let stepPos = pos
                let aTok = generated[generated.count - 1]
                let stepGreedy = isGreedy && (!usePenalty || penaltyHistory.isEmpty)
                try withGPUTurn {
                    try forwardPass.forward(
                        tokens: [aTok], posOffset: stepPos, runHeadRows: 1, greedyHead: stepGreedy,
                        updateState: true, captureTapBuf: tapBuf, tapVisits: dflash.config.targetLayerIds,
                        profiler: profiler
                    )
                    pos = stepPos + 1
                    try ingestDFlashPrefillSlice(
                        df: dflash, tapBuf: tapBuf, chunkRows: 1,
                        slice: QWDFlashConfig.PrefillCaptureSlice(localStart: 0, captureRows: 1, ctxStartPos: stepPos)
                    )
                }
                let cTok: Int
                if stepGreedy {
                    cTok = Int(headTokPtr[0])
                } else {
                    let singleRes = try QWSpeculativeRoundResolver.resolve(
                        draftProposals: [], stream: stream, isGreedy: isGreedy, maxNew: 1, stopIDs: []
                    ) { _, _, _ in
                        let rowPtr = UnsafeBufferPointer(start: logitsPtr, count: V)
                        let rep = usePenalty ? penaltyHistory : []
                        if isGreedy {
                            let gTok = try sampler.penalisedGreedyToken(logits: rowPtr, repetitionHistory: rep)
                            return try QWSamplingDistribution.sparse(vocabulary: V, ids: [gTok], weights: [1.0])
                        }
                        return try sampler.distribution(logits: rowPtr, repetitionHistory: rep)
                    }
                    cTok = singleRes.firstToken ?? 0
                }
                stepMs.append(elapsedSeconds(cStart.duration(to: clock.now)) * 1000.0)
                let outcome = try consumeSpeculativeToken(
                    cTok, vocab: V, stopIDs: stopIDs, usePenalty: usePenalty,
                    recentGeneratedForPenalty: &recentGeneratedForPenalty, penaltyHistory: &penaltyHistory,
                    history: &history, generated: &generated, stopTokenID: &stopTokenID,
                    onStopToken: onStopToken, onToken: onToken)
                if outcome == .stopped { stepCounts.append(0); return (stats, true) }
                stepCounts.append(1)
                if outcome == .cancelled || generated.count >= effectiveMaxNew { return (stats, true) }
                cooldownSteps -= 1
            }
            if !hasCooldownPLD {
                let pld = (allowPLD && isGreedy)
                    ? makePLDDrafts(history: history, maxCount: maxVerifyDrafts(cap: pldCap), allowRecyclerFallback: false)
                    : []
                if !pld.isEmpty {
                    try flushPendingIngest(); nextProposal = (pld, [], true)
                } else {
                    let pi = pendingIngest; pendingIngest = nil
                    nextProposal = try proposeNextDrafts(
                        anchor: generated[generated.count - 1], stream: stream, pendingIngest: pi)
                }
            }
            if nextProposal.drafts.isEmpty { try flushPendingIngest(); break }
        }
        return (stats, false)
    }
}
