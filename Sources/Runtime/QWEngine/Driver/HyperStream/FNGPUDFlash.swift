// FNGPUDFlash.swift — DFlash 2 / DSpine speculative decoding, hybrid PLD bypass,
// sliding-window capture, prefix-cache persistence, and residency helpers for Flash-Next.
import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWMTP
import QWQuant
import QWAttn
import QWHyperPLE

public enum FNGPUDFlash {
    public static let maxDrafts = QWFourSlotRollbackRing.maxVerifyRows - 1
    public static let defaultMTPPackSavingsBytes = QWDFlashFootprint.defaultMTPPackSavingsBytes
    public static let defaultDFlashWeightBytes = QWDFlashFootprint.defaultDFlashWeightBytes
    public static let defaultDFlashScratchBytes = QWDFlashFootprint.defaultDFlashScratchBytes

    public static func isDFlashName(_ raw: String?) -> Bool { QWDFlashFootprint.isDFlashName(raw) }

    public static func adjustedPerTokenKVBytes(_ perTokenBytes: Int, usesDFlash: Bool) -> Int {
        QWDFlashFootprint.adjustedPerTokenKVBytes(perTokenBytes, usesDFlash: usesDFlash)
    }

    static func resolveDrafterPackPath(targetDescriptor: QWModelDescriptor?, explicitPath: String?) -> String {
        QWDFlashFootprint.resolveDrafterPackPath(targetDescriptor: targetDescriptor, explicitPath: explicitPath)
    }

    /// Draft vocab rows for DFlash proposals (`QWG_DRAFT_VOCAB`, default 98304, 0 = full vocab).
    static func resolveDraftVocab(fullVocab: Int) -> Int? {
        QWDFlashDrafter.resolveDraftVocab(fullVocab: fullVocab, defaultCap: 98_304)
    }

    public static func drafterWeightBytes(packPath: String?, isDSpine: Bool) -> Int {
        QWDFlashFootprint.drafterWeightBytes(packPath: packPath, isDSpine: isDSpine)
    }

    public static func estimateFootprint(
        targetDescriptor: QWModelDescriptor, targetPackPath: String?, dflashPackPath: String?, isDFlash: Bool
    ) -> (nativeMTPWeightBytes: Int, dflashWeightBytes: Int, dflashScratchBytes: Int) {
        QWDFlashFootprint.estimateFootprint(
            targetDescriptor: targetDescriptor, targetPackPath: targetPackPath,
            dflashPackPath: dflashPackPath, isDFlash: isDFlash)
    }

    static func prefillWindowEnd(historyCount: Int, targetEnd: Int?) -> Int { min(historyCount, targetEnd ?? historyCount) }

    static func prefillCaptureSlice(startPos: Int, rows: Int, historyCount: Int, maxCacheSize: Int) -> (localStart: Int, captureRows: Int, ctxStartPos: Int)? {
        QWDFlashConfig.prefillCaptureSlice(
            startPos: startPos, rows: rows, windowEnd: historyCount, maxCacheSize: maxCacheSize
        ).map { ($0.localStart, $0.captureRows, $0.ctxStartPos) }
    }

    static func encodeLayerCapture(
        _ enc: any MTLComputeCommandEncoder, dflash: QWDFlashDrafter, layerIndex: Int, deferOut: Bool,
        base: QWTensorValue, src: QWTensorValue, inj: QWTensorValue, cur: QWTensorValue,
        srcRowOffset: Int, dstRowOffset: Int, rows: Int, hc: Int
    ) {
        guard rows > 0, (0..<dflash.targetLayerSlotMap.count).contains(layerIndex),
              case let slot = dflash.targetLayerSlotMap[layerIndex], slot >= 0 else { return }
        enc.memoryBarrier(scope: .buffers)
        dflash.encodeCaptureHCTargetHidden(
            enc, base: deferOut ? base : cur, srcRowOffset: srcRowOffset, dstRowOffset: dstRowOffset,
            rows: rows, slot: slot, hc: hc, deferredInject: deferOut ? (src: src, inj: inj, isMHC: false) : nil)
        enc.memoryBarrier(scope: .buffers)
    }

    static func dflashGeometryMarker(config: QWDFlashConfig) -> String { QWDFlashDrafter.prefixGeometryMarker(config: config) }

    static func prefixPayloadBytes(config: QWDFlashConfig, tokenCount: Int) -> Int {
        QWDFlashDrafter.prefixPayloadBytes(config: config, tokenCount: tokenCount)
    }

    static func appendPrefixPayload(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        dflash: QWDFlashDrafter, marker: String, tokenCount: Int
    ) {
        appendPrefixKV(to: dstBase, at: &writeOffset, slots: dflash.sessionSlots.first, config: dflash.config,
                       marker: marker, tokenCount: tokenCount)
    }

    static func appendPrefixKV(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, marker: String, tokenCount: Int
    ) {
        QWDFlashDrafter.appendPrefixKV(
            to: dstBase, at: &writeOffset, slot: slots, config: cfg, marker: marker, tokenCount: tokenCount)
    }

    static func restorePrefixPayload(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        dflash: QWDFlashDrafter, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        restorePrefixKV(from: base, dataCount: dataCount, offset: &offset, slots: dflash.sessionSlots.first,
                        config: dflash.config, expectedMarker: expectedMarker, expectedTokens: expectedTokens)
    }

    static func restorePrefixKV(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        QWDFlashDrafter.restorePrefixKV(
            from: base, dataCount: dataCount, offset: &offset, slot: slots, config: cfg,
            expectedMarker: expectedMarker, expectedTokens: expectedTokens)
    }

    static func validatePrefixBlob(data: Data, expectedMarker: String, expectedTokens: Int, config: QWDFlashConfig) -> Bool {
        QWDFlashDrafter.validatePrefixBlob(data: data, expectedMarker: expectedMarker, expectedTokens: expectedTokens, config: config)
    }

    static func encodeHeadLogits(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal, headWeight: QWWeightView,
        xRot16: QWTensorValue, xNorm16: QWTensorValue, xRot32: QWTensorValue?,
        yLogits32: QWTensorValue, rows: Int, vocabRows: Int
    ) throws {
        guard let hw = QWHeadWeight(
            blocks: headWeight.blocks, n: headWeight.n, k: headWeight.k,
            bits: headWeight.bits, groupSize: headWeight.groupSize,
            isRotated: headWeight.isRotated, isIncumbentContainer: headWeight.isIncumbentContainer, isSHVQ: headWeight.isSHVQ
        ) else { throw QWDFlashError.state("FNGPUDFlash: invalid head weight for vocabRows=\(vocabRows)") }
        try QWDFlashDrafter.encodeSlicedHeadLogits(
            enc, hal: hal, headStorage: .q4g(hw), geometry: hal.geometry,
            xRot16: xRot16, xNorm16: xNorm16, xRot32: xRot32, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
    }

    static func selectHybridBypassDrafts(
        history: [Int64], pendingToken: Int, mode: QWDraftStrategy, maxDrafts: Int, recycler: QWTokenRecycler
    ) -> [Int] {
        QWContextCopyDrafter.selectHybridPLDDrafts(
            history: history, extraToken: Int64(pendingToken), previousToken: history.last.map(Int.init),
            pendingToken: pendingToken, mode: mode, maxDrafts: maxDrafts, recycler: recycler)
    }

    static func maxFittingContext(activeCapacity: Int) -> Int { QWDFlashFootprint.maxFittingContext(activeCapacity: activeCapacity) }

    static func consumePendingContextUpdate(
        pos: Int, count: inout Int, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        let pending = count; count = 0
        guard pending > 0, let slot else { return nil }
        return (slot, pos, pending)
    }

    static func consumePendingContextUpdate(
        _ session: inout FNGPUSessionData, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        consumePendingContextUpdate(pos: session.dflashPendingCtxPos, count: &session.dflashPendingCtxCount, slot: slot)
    }

    static func step(
        _ engine: QWHyperStreamRuntime, session: inout FNGPUSessionData, token: Int,
        vStack: FNGPUDecodeStack, dflash: QWDFlashDrafter
    ) throws -> Int {
        guard let dSlot = dflash.sessionSlots.first else {
            throw QWDFlashError.state("FNGPUDFlash: missing DFlash session slot")
        }
        let prevCount = session.history.count
        let dflashCap = min(maxDrafts, max(1, dflash.config.blockSize - 1))
        let maxCap = max(1, min(min(session.maxDrafts, dflashCap), engine.activeCapacity - prevCount - 2))
        let stepT0 = Date(), trackRep = session.sampler.config.repetitionPenalty != 1.0
        let isUnconstrained = session.constraint?.isUnconstrainedNow ?? true, isGreedy = session.sampler.config.isGreedy
        let allowPLD = isUnconstrained && isGreedy && session.seenTokens.isEmpty && !session.sampler.config.hasExplicitSeed
        let preCtx = consumePendingContextUpdate(&session, slot: dSlot)
        let canonical = (!isUnconstrained && QWConstrainedSpeculationGate.isEnabled)
            ? (session.constraint?.proposeCanonicalDrafts(maxCount: maxCap) ?? []) : []
        var drafts = !canonical.isEmpty ? canonical : (allowPLD
            ? selectHybridBypassDrafts(history: session.history, pendingToken: token,
                                       mode: session.draftMode, maxDrafts: maxCap, recycler: session.tokenRecycler) : [])
        let usedPLD = !drafts.isEmpty
        var proposals: [QWMTPGumbelProposal] = [], draftGPUMs = 0.0
        let stream: QWMTPRandomStream? = (!usedPLD && !isGreedy) ? QWMTPRandomStream(cloning: try session.ensureDFlashStreamState()) : nil
        if !usedPLD {
            let draftCount = session.dflashPolicy.selectDraftCount(maxDrafts: maxCap)
            let draftStream: QWMTPRandomStream? = try stream.map { s in
                let ds = QWMTPRandomStream(seed: s.seed, drawIndex: try s.reserveDraws(UInt64(draftCount + 1) * UInt64(engine.vocab)))
                try ds.beginGumbelRound(drafts: draftCount, vocabulary: engine.vocab, maxDrafts: dflashCap)
                return ds
            }
            do {
                let proposed = try dflash.propose(
                    hal: engine.ctx.hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                    pendingToken: token, draftCount: draftCount, draftVocab: resolveDraftVocab(fullVocab: engine.vocab),
                    headStorage: engine.dflashHeadStorage,
                    encodeHeadLogits: { enc, xRot16, xNorm16, xRot32, yLogits32, rows, vocabRows in
                        try encodeHeadLogits(enc, hal: engine.ctx.hal, headWeight: engine.head.weight, xRot16: xRot16,
                                             xNorm16: xNorm16, xRot32: xRot32, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
                    },
                    encodeAnchorMaskEmbed: { enc, propX32, anchorToken, maskToken, rows in
                        try dflash.encodeStageProp(enc, embedTable: engine.embedTable, propX: propX32,
                                                   anchorToken: anchorToken, maskToken: maskToken, rows: rows)
                    },
                    stochastic: draftStream.map { (Float(max(session.sampler.config.temperature, 1e-6)), $0) },
                    preContextUpdate: preCtx,
                    completed: { cb in draftGPUMs = max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0) })
                drafts = proposed.drafts; proposals = proposed.proposals
            } catch { draftStream?.cancelActiveRound(); throw error }
        }
        let dCount = drafts.count, activeRows = 1 + dCount
        let prevBefore = prevCount > 0 ? Int(session.history[prevCount - 1]) : nil
        session.history.append(Int64(token)); drafts.forEach { session.history.append(Int64($0)) }
        if engine.residentKVHistory.count == prevCount {
            engine.residentKVHistory.append(Int64(token)); drafts.forEach { engine.residentKVHistory.append(Int64($0)) }
        }
        engine.highWaterPos = max(engine.highWaterPos, session.history.count)

        let margin: Float = (usedPLD || !isGreedy || !isUnconstrained) ? 0 : session.relaxedMargin
        _ = try vStack.forwardRows(
            history: session.history, embedTable: engine.embedTable, head: engine.head,
            readStream: false, verifyAllRows: true, mtp: nil, activeRows: activeRows,
            relaxedMargin: margin, relaxedRows: margin > 0 ? activeRows : 0,
            relaxedMinProb: session.relaxedMinProb, dflash: dflash, preContextUpdate: usedPLD ? preCtx : nil)
        session.specProposals += dCount
        let (tok0, accepted) = try resolveVerifiedTokens(
            engine, session: &session, drafts: drafts, proposals: proposals, stream: stream,
            prevCount: prevCount, prevBefore: prevBefore, trackRep: trackRep,
            verifyMargin: margin, usedPLD: usedPLD)
        if accepted < dCount {
            let rejected = dCount - accepted
            session.history.removeLast(rejected)
            if engine.residentKVHistory.count == prevCount + activeRows { engine.residentKVHistory.removeLast(rejected) }
            vStack.rollbackVerifyRows(rejectedCount: rejected)
        }
        let roundMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.specAccepted += accepted
        if !usedPLD {
            session.dflashPolicy.observeRound(
                draftCount: dCount, accepted: accepted, roundMs: roundMs,
                isGreedy: isGreedy, hasExplicitSeed: session.sampler.config.hasExplicitSeed)
        }
        session.dflashPendingCtxPos = prevCount; session.dflashPendingCtxCount = 1 + accepted
        session.dflashCacheOffset = prevCount + 1 + accepted; session.headToken = -1
        let divisor = Double(1 + session.pendingTokens.count)
        session.stepMs = roundMs / divisor; session.lastStepGPUMillis = (draftGPUMs + vStack.lastGPUMillis) / divisor
        session.roundTokenCounts.append(1)
        return tok0
    }

    private static func resolveVerifiedTokens(
        _ engine: QWHyperStreamRuntime, session: inout FNGPUSessionData,
        drafts: [Int], proposals: [QWMTPGumbelProposal], stream: QWMTPRandomStream?,
        prevCount: Int, prevBefore: Int?, trackRep: Bool, verifyMargin: Float, usedPLD: Bool
    ) throws -> (tok0: Int, accepted: Int) {
        let dCount = drafts.count, vocab = engine.vocab
        let startUnconstrained = session.constraint?.isUnconstrainedNow ?? true
        let isGreedy = session.sampler.config.isGreedy
        let v = engine.head.argmaxTokens(count: 1 + dCount)
        if verifyMargin == 0 && isGreedy && startUnconstrained {
            for row in (0..<(1 + dCount)).reversed() {
                let prev = (row == 0) ? prevBefore : Int(session.history[prevCount + row - 1])
                session.tokenRecycler.recordTransition(previous: prev, current: Int(session.history[prevCount + row]), next: v[row])
            }
        }
        let logitsPtr = engine.arena.cpuPointer(of: engine.head.logits).assumingMemoryBound(to: Float.self)
        let repByRow = trackRep
            ? FNGPUSessionData.precomputeLoopRepetitionByRow(
                recentGenerated: session.recentGeneratedTokens,
                seenTokenList: session.seenTokenList, drafts: drafts, vocab: vocab)
            : []
        let rowMargin: Float = (!usedPLD && !startUnconstrained && isGreedy) ? session.relaxedMargin : verifyMargin
        let relaxedSampler = (isGreedy && rowMargin > 0)
            ? QWSampler(config: QWSamplerConfig(
                temperature: 1.0, topK: vocab, topP: 1.0,
                uniformVersion: session.sampler.config.uniformVersion,
                repetitionPenalty: session.sampler.config.repetitionPenalty))
            : nil
        let roundSampler = session.sampler
        let draftProps: [QWSpeculativeRoundResolver.DraftProposal] = !proposals.isEmpty
            ? proposals.map { .distribution(token: $0.token, distribution: $0.distribution) }
            : drafts.map { .greedyToken($0) }
        let res = try QWSpeculativeRoundResolver.resolve(
            draftProposals: draftProps, stream: stream, isGreedy: isGreedy, maxNew: 1 + dCount,
            stopIDs: [], constraint: session.constraint, margin: rowMargin, minProbability: session.relaxedMinProb
        ) { row, _, probe in
            let rep = trackRep ? repByRow[row] : []
            let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * vocab), count: vocab)
            let isConstrainedRow = try probe?.writeAllowedTokens(into: &roundSampler.allowedScratch) ?? false
            if isGreedy {
                if rowMargin > 0, row < dCount, let relaxedSampler {
                    return isConstrainedRow
                        ? try relaxedSampler.constrainedDistribution(
                            logits: rowPtr, allowed: roundSampler.allowedScratch.withUnsafeBufferPointer { $0 },
                            repetitionHistory: rep)
                        : try relaxedSampler.distribution(logits: rowPtr, repetitionHistory: rep)
                }
                let tok = isConstrainedRow
                    ? try roundSampler.constrainedTokenFromScratch(logits: rowPtr, repetitionHistory: rep)
                    : (trackRep ? try roundSampler.penalisedGreedyToken(logits: rowPtr, repetitionHistory: rep) : v[row])
                return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: [tok], weights: [1.0])
            }
            return isConstrainedRow
                ? try roundSampler.constrainedDistributionFromScratch(logits: rowPtr, repetitionHistory: rep)
                : try roundSampler.distribution(logits: rowPtr, repetitionHistory: rep)
        }
        if let stream { session.dflashStreamState = try stream.frozenState() }
        let tok0 = res.firstToken ?? 0
        if session.constraint != nil {
            try session.constraint?.accept(tok0)
            session.pendingConstraintAcceptCount += res.trailingTokens.count
        }
        if !res.trailingTokens.isEmpty { session.pendingTokens.append(contentsOf: res.trailingTokens) }
        if trackRep { for t in res.tokens { session.recordSeenToken(t, vocab: vocab) } }
        return (tok0, res.acceptedDrafts)
    }
}
